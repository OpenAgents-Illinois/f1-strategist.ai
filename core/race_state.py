import time
from dataclasses import dataclass, field


@dataclass
class RaceState:
    session_key: str
    driver: int
    lap: int = 0
    compound: str = "UNKNOWN"
    stint_lap: int = 0
    gap_ahead: float = 0.0
    gap_behind: float = 0.0
    sc_active: bool = False
    vsc_active: bool = False
    last_updated: float = field(default_factory=time.time)

    @classmethod
    def default(cls, session_key: str, driver: int) -> "RaceState":
        return cls(session_key=session_key, driver=driver)

    def update_from_poll(
        self,
        positions: list[dict] | None = None,
        intervals: list[dict] | None = None,
        stints: list[dict] | None = None,
        race_control: list[dict] | None = None,
    ) -> None:
        """Mutate state from raw OpenF1 response dicts. Missing keys are ignored."""
        if positions:
            latest = positions[-1]
            self.lap = latest.get("lap_number", self.lap)

        if stints:
            latest = stints[-1]
            self.compound = latest.get("compound", self.compound)
            self.stint_lap = latest.get("lap_number", self.stint_lap)

        if intervals:
            # Find target driver's interval entry
            for entry in reversed(intervals):
                if entry.get("driver_number") == self.driver:
                    gap_to_leader = entry.get("gap_to_leader")
                    interval = entry.get("interval")
                    if gap_to_leader is not None:
                        try:
                            self.gap_ahead = float(gap_to_leader)
                        except (ValueError, TypeError):
                            pass
                    if interval is not None:
                        try:
                            self.gap_behind = float(interval)
                        except (ValueError, TypeError):
                            pass
                    break

        if race_control:
            sc = False
            vsc = False
            for msg in race_control:
                flag = msg.get("flag", "")
                if flag == "SAFETY CAR":
                    sc = True
                elif flag == "VIRTUAL SAFETY CAR":
                    vsc = True
                elif flag in ("GREEN", "CLEAR"):
                    sc = False
                    vsc = False
            self.sc_active = sc
            self.vsc_active = vsc

        self.last_updated = time.time()
