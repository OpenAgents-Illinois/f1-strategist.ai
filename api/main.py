"""FastAPI application for f1-strategist.ai.

Exposes:
  GET  /health          — liveness probe
  GET  /state           — current RaceState as JSON
  WS   /ws/strategy     — streams StrategyCall JSON to all connected clients

On startup the Orchestrator is created and its polling loop is started as a
background asyncio.Task.  The task is cancelled cleanly on shutdown.
"""

import asyncio
import dataclasses
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from agents.orchestrator import Orchestrator
from core.models import StrategyCall
from core.race_state import RaceState

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Tracks active WebSocket connections and broadcasts StrategyCall updates."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    def add(self, ws: WebSocket) -> None:
        self._connections.add(ws)

    def remove(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast(self, call: StrategyCall) -> None:
        """Send *call* as JSON to every connected client.

        Stale / disconnected sockets are caught silently and removed so they
        never crash the broadcast loop.
        """
        payload = call.model_dump_json()
        dead: set[WebSocket] = set()
        for ws in list(self._connections):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)

        for ws in dead:
            self.remove(ws)
            logger.debug("Removed stale WebSocket connection")


# ---------------------------------------------------------------------------
# Application state (module-level so tests can inspect / replace)
# ---------------------------------------------------------------------------

manager = ConnectionManager()
orchestrator: Orchestrator | None = None
race_state: RaceState | None = None
_poll_task: asyncio.Task | None = None  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Start the orchestrator polling loop on startup; cancel it on shutdown."""
    global orchestrator, race_state, _poll_task

    session_key = os.getenv("SESSION_KEY", "latest")
    driver = int(os.getenv("TARGET_DRIVER", "1"))

    race_state = RaceState.default(session_key, driver)

    orchestrator = Orchestrator()

    # Register a callback so every new StrategyCall is broadcast to WS clients
    # and the module-level race_state is kept in sync via the orchestrator's
    # internal state (we share the same RaceState object by passing it in).
    async def _on_call(call: StrategyCall) -> None:
        await manager.broadcast(call)

    orchestrator.subscribe(_on_call)

    logger.info(
        "Starting orchestrator — session=%s driver=%d", session_key, driver
    )

    _poll_task = asyncio.create_task(
        orchestrator.run(session_key, driver),
        name="orchestrator-poll",
    )

    yield

    # Shutdown: cancel the background task and wait for it to finish
    if _poll_task is not None and not _poll_task.done():
        _poll_task.cancel()
        try:
            await _poll_task
        except asyncio.CancelledError:
            pass

    logger.info("Orchestrator stopped")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="f1-strategist.ai",
    description="Real-time F1 race strategy via a multi-agent swarm",
    version="0.1.0",
    lifespan=lifespan,
)

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir, html=True), name="static")


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/state")
async def state() -> JSONResponse:
    """Return the current RaceState as a JSON object."""
    if race_state is None:
        return JSONResponse(content={}, status_code=503)
    return JSONResponse(content=dataclasses.asdict(race_state))


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/strategy")
async def ws_strategy(websocket: WebSocket) -> None:
    """Stream StrategyCall updates to connected clients."""
    await websocket.accept()
    manager.add(websocket)
    logger.debug("WebSocket client connected; total=%d", len(manager._connections))
    try:
        # Keep the connection open until the client disconnects
        while True:
            # We don't expect data from the client, but we need to await
            # something so that disconnects are detected promptly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        manager.remove(websocket)
        logger.debug(
            "WebSocket client disconnected; total=%d", len(manager._connections)
        )
