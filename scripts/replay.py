"""Offline race replay — feeds fixture data through the full agent pipeline."""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

from agents.gap_monitor import GapMonitor
from agents.safety_car_detector import SafetyCarDetector
from agents.synthesizer import Synthesizer
from agents.tire_strategist import TireStrategist
from core.race_state import RaceState

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def load_fixture(name: str) -> list[dict]:
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        print(f"[replay] WARNING: fixture {path} not found, using empty list")
        return []
    with open(path) as f:
        return json.load(f)


async def replay(session: str, speed: float, driver: int = 1) -> None:
    positions = load_fixture("positions")
    intervals = load_fixture("intervals")
    stints = load_fixture("stints")
    race_control = load_fixture("race_control")

    # Determine number of "ticks" — use the longest fixture
    num_ticks = max(len(positions), len(intervals), len(stints), len(race_control), 10)

    tire_strategist = TireStrategist()
    gap_monitor = GapMonitor()
    sc_detector = SafetyCarDetector()
    synthesizer = Synthesizer()

    state = RaceState.default(session, driver)
    base_interval = 2.0 / speed  # 2s poll interval scaled by speed

    print(f"[replay] session={session} driver={driver} speed={speed}x ticks={num_ticks}")
    print("-" * 60)

    calls_made = 0

    for tick in range(num_ticks):
        # Slice fixture data up to current tick to simulate incremental telemetry
        pos_slice = positions[: tick + 1] if positions else []
        int_slice = intervals[: tick + 1] if intervals else []
        stint_slice = stints[: tick + 1] if stints else []
        rc_slice = race_control[: tick + 1] if race_control else []

        state.update_from_poll(
            positions=pos_slice,
            intervals=int_slice,
            stints=stint_slice,
            race_control=rc_slice,
        )

        tire_signal = tire_strategist.analyze(state, stint_slice)
        gap_signal = gap_monitor.analyze(state, int_slice)
        sc_signal = sc_detector.analyze(state, rc_slice)
        call = await synthesizer.synthesize(tire_signal, gap_signal, sc_signal, state)

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[{ts}] lap={state.lap:>2} compound={state.compound:<6} stint_lap={state.stint_lap:>2} "
            f"| {call.action.value:<9} conf={call.confidence:.0%} | {call.reasoning}"
        )
        calls_made += 1

        await asyncio.sleep(base_interval)

    print("-" * 60)
    print(f"[replay] done — {calls_made} strategy calls emitted")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an F1 race session offline")
    parser.add_argument("--session", default="9158", help="Session key (default: 9158)")
    parser.add_argument("--speed", default="10x", help="Replay speed, e.g. 10x (default: 10x)")
    parser.add_argument("--driver", type=int, default=1, help="Driver number (default: 1)")
    args = parser.parse_args()

    speed_str = args.speed.rstrip("xX")
    try:
        speed = float(speed_str)
    except ValueError:
        print(f"[replay] Invalid speed '{args.speed}', defaulting to 10x")
        speed = 10.0

    # Disable Claude for replay — use fallback logic
    os.environ.pop("ANTHROPIC_API_KEY", None)

    asyncio.run(replay(args.session, speed, args.driver))


if __name__ == "__main__":
    main()
