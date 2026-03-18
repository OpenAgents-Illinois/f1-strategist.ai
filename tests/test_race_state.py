import time

from core.race_state import RaceState


class TestRaceStateDefault:
    def test_fields(self):
        state = RaceState.default("9158", 1)
        assert state.session_key == "9158"
        assert state.driver == 1
        assert state.lap == 0
        assert state.compound == "UNKNOWN"
        assert state.stint_lap == 0
        assert state.gap_ahead == 0.0
        assert state.gap_behind == 0.0
        assert state.sc_active is False
        assert state.vsc_active is False

    def test_last_updated_is_recent(self):
        before = time.time()
        state = RaceState.default("9158", 1)
        assert state.last_updated >= before


class TestUpdateFromPoll:
    def setup_method(self):
        self.state = RaceState.default("9158", 1)

    def test_updates_lap_from_positions(self):
        self.state.update_from_poll(positions=[{"lap_number": 24}])
        assert self.state.lap == 24

    def test_updates_compound_and_stint_lap_from_stints(self):
        self.state.update_from_poll(stints=[{"compound": "SOFT", "lap_number": 3}])
        assert self.state.compound == "SOFT"
        assert self.state.stint_lap == 3

    def test_updates_gaps_from_intervals(self):
        self.state.update_from_poll(
            intervals=[
                {"driver_number": 1, "gap_to_leader": "12.4", "interval": "3.1"}
            ]
        )
        assert self.state.gap_ahead == 12.4
        assert self.state.gap_behind == 3.1

    def test_ignores_intervals_for_other_drivers(self):
        self.state.update_from_poll(
            intervals=[{"driver_number": 44, "gap_to_leader": "5.0", "interval": "1.0"}]
        )
        assert self.state.gap_ahead == 0.0
        assert self.state.gap_behind == 0.0

    def test_sets_sc_active(self):
        self.state.update_from_poll(race_control=[{"flag": "SAFETY CAR"}])
        assert self.state.sc_active is True
        assert self.state.vsc_active is False

    def test_sets_vsc_active(self):
        self.state.update_from_poll(race_control=[{"flag": "VIRTUAL SAFETY CAR"}])
        assert self.state.vsc_active is True
        assert self.state.sc_active is False

    def test_clears_sc_on_green(self):
        self.state.sc_active = True
        self.state.update_from_poll(race_control=[{"flag": "GREEN"}])
        assert self.state.sc_active is False

    def test_partial_update_preserves_existing_values(self):
        self.state.update_from_poll(stints=[{"compound": "HARD", "lap_number": 10}])
        self.state.update_from_poll(positions=[{"lap_number": 25}])
        assert self.state.compound == "HARD"  # preserved
        assert self.state.lap == 25  # updated

    def test_empty_dicts_do_not_raise(self):
        self.state.update_from_poll(positions=[{}], stints=[{}], intervals=[{}])

    def test_none_args_do_not_raise(self):
        self.state.update_from_poll()

    def test_last_updated_changes(self):
        old_ts = self.state.last_updated
        time.sleep(0.01)
        self.state.update_from_poll(positions=[{"lap_number": 1}])
        assert self.state.last_updated > old_ts
