import pytest
from pydantic import ValidationError

from core.models import Action, GapSignal, SafetyCarSignal, StrategyCall, TireSignal


class TestTireSignal:
    def test_valid(self):
        t = TireSignal(
            driver=1,
            recommend_pit=True,
            suggested_compound="MEDIUM",
            pit_window_laps=(10, 15),
            deg_rate=0.08,
        )
        assert t.driver == 1
        assert t.suggested_compound == "MEDIUM"
        assert t.pit_window_laps == (10, 15)

    def test_rejects_missing_fields(self):
        with pytest.raises(ValidationError):
            TireSignal(driver=1, recommend_pit=True)


class TestGapSignal:
    def test_valid(self):
        g = GapSignal(
            driver=1,
            undercut_viable=True,
            overcut_viable=False,
            gap_ahead=18.5,
            gap_behind=30.0,
        )
        assert g.undercut_viable is True
        assert g.gap_ahead == 18.5

    def test_rejects_missing_fields(self):
        with pytest.raises(ValidationError):
            GapSignal(driver=1)


class TestSafetyCarSignal:
    def test_valid(self):
        s = SafetyCarSignal(
            sc_active=True,
            vsc_active=False,
            pit_opportunity=True,
            reasoning="SC deployed, free stop available",
        )
        assert s.sc_active is True
        assert s.pit_opportunity is True

    def test_rejects_missing_fields(self):
        with pytest.raises(ValidationError):
            SafetyCarSignal(sc_active=True)


class TestStrategyCall:
    def test_valid_box_now(self):
        c = StrategyCall(
            driver=1,
            action=Action.BOX_NOW,
            confidence=0.92,
            reasoning="Undercut window open",
            lap=24,
        )
        assert c.action == Action.BOX_NOW
        assert c.action == "BOX NOW"

    def test_valid_stay_out(self):
        c = StrategyCall(
            driver=1,
            action=Action.STAY_OUT,
            confidence=0.75,
            reasoning="Gap too large to undercut",
            lap=30,
        )
        assert c.action == Action.STAY_OUT

    def test_valid_monitor(self):
        c = StrategyCall(
            driver=1,
            action=Action.MONITOR,
            confidence=0.5,
            reasoning="No clear signal",
            lap=12,
        )
        assert c.action == Action.MONITOR

    def test_rejects_invalid_action(self):
        with pytest.raises(ValidationError):
            StrategyCall(
                driver=1, action="INVALID", confidence=0.5, reasoning="x", lap=1
            )

    def test_rejects_confidence_above_1(self):
        with pytest.raises(ValidationError):
            StrategyCall(
                driver=1, action=Action.BOX_NOW, confidence=1.1, reasoning="x", lap=1
            )

    def test_rejects_confidence_below_0(self):
        with pytest.raises(ValidationError):
            StrategyCall(
                driver=1, action=Action.BOX_NOW, confidence=-0.1, reasoning="x", lap=1
            )

    def test_confidence_boundary_values(self):
        for val in (0.0, 1.0):
            c = StrategyCall(
                driver=1, action=Action.MONITOR, confidence=val, reasoning="x", lap=1
            )
            assert c.confidence == val
