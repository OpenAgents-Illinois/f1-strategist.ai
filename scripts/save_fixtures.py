"""Fetch live OpenF1 data and save as JSON fixtures for offline testing."""

import argparse
import asyncio
import json
from pathlib import Path

from core.openf1_client import OpenF1Client

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


async def save_fixtures(session: str, driver: int) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    async with OpenF1Client(request_delay=1.0) as client:
        endpoints = {
            "positions.json": client.get_positions(session, driver),
            "intervals.json": client.get_intervals(session),
            "stints.json": client.get_stints(session, driver),
            "race_control.json": client.get_race_control(session),
        }

        for filename, coro in endpoints.items():
            print(f"Fetching {filename}...", end=" ", flush=True)
            try:
                data = await coro
            except Exception as e:
                print(f"WARNING: {e} — saving empty list")
                data = []
            path = FIXTURES_DIR / filename
            path.write_text(json.dumps(data, indent=2))
            print(f"{len(data)} records saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save OpenF1 API responses as test fixtures")
    parser.add_argument("--session", default="9655", help="OpenF1 session key (default: 9655 — 2024 Qatar GP)")
    parser.add_argument("--driver", type=int, default=1, help="Driver number (default: 1 — Verstappen)")
    args = parser.parse_args()

    print(f"Saving fixtures for session={args.session}, driver={args.driver}")
    asyncio.run(save_fixtures(args.session, args.driver))
    print("Done.")


if __name__ == "__main__":
    main()
