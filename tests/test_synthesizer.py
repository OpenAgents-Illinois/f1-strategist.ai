"""Tests for agents/synthesizer.py — Synthesizer agent."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.synthesizer import Synthesizer, _fallback_call, _parse_claude_response
from core.models import Action, GapSignal, SafetyCarSignal, StrategyCall, TireSignal
from core.race_state import RaceState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_state(**kwargs) -> RaceState:
    defaults = dict(session_key="9158", driver=1, lap=30, compound="MEDIUM", stint_lap=15)
    defaults.update(kwargs)
    return RaceState(**defaults)


def make_tire(**kwargs) -> TireSignal:
    defaults = dict(
        driver=1,
        recommend_pit=False,
        suggested_compound="HARD",
        pit_window_laps=(0, 0),
        deg_rate=0.05,
    )
    defaults.update(kwargs)
    return TireSignal(**defaults)


def make_gap(**kwargs) -> GapSignal:
    defaults = dict(driver=1, undercut_viable=False, overcut_viable=False, gap_ahead=30.0, gap_behind=10.0)
    defaults.update(kwargs)
    return GapSignal(**defaults)


def make_sc(**kwargs) -> SafetyCarSignal:
    defaults = dict(sc_active=False, vsc_active=False, pit_opportunity=False, reasoning="No SC")
    defaults.update(kwargs)
    return SafetyCarSignal(**defaults)


# ---------------------------------------------------------------------------
# Fallback logic tests
# ---------------------------------------------------------------------------


def test_fallback_sc_priority():
    """SC pit opportunity takes priority over all other signals."""
    state = make_state()
    tire = make_tire(recommend_pit=False)
    gap = make_gap(undercut_viable=False)
    sc = make_sc(sc_active=True, pit_opportunity=True, reasoning="SC deployed")

    call = _fallback_call(tire, gap, sc, state)

    assert call.action == Action.BOX_NOW
    assert call.confidence >= 0.90
    assert call.driver == 1
    assert call.lap == 30


def test_fallback_undercut_second_priority():
    """Undercut takes priority over tire deg when SC is not active."""
    state = make_state()
    tire = make_tire(recommend_pit=True)
    gap = make_gap(undercut_viable=True, gap_ahead=15.0)
    sc = make_sc()

    call = _fallback_call(tire, gap, sc, state)

    assert call.action == Action.BOX_NOW
    assert "undercut" in call.reasoning.lower() or "gap" in call.reasoning.lower()


def test_fallback_tire_deg_third_priority():
    """Tire degradation triggers BOX NOW when SC and undercut are not active."""
    state = make_state()
    tire = make_tire(recommend_pit=True, deg_rate=0.08, suggested_compound="HARD")
    gap = make_gap(undercut_viable=False)
    sc = make_sc()

    call = _fallback_call(tire, gap, sc, state)

    assert call.action == Action.BOX_NOW
    assert "HARD" in call.reasoning or "deg" in call.reasoning.lower()


def test_fallback_overcut_viable():
    """Overcut viable returns STAY OUT when no higher priority signal is active."""
    state = make_state()
    tire = make_tire(recommend_pit=False)
    gap = make_gap(undercut_viable=False, overcut_viable=True, gap_behind=30.0, gap_ahead=35.0)
    sc = make_sc()

    call = _fallback_call(tire, gap, sc, state)

    assert call.action == Action.STAY_OUT


def test_fallback_monitor_when_no_signals():
    """MONITOR is returned when no decisive signal is present."""
    state = make_state()
    tire = make_tire()
    gap = make_gap()
    sc = make_sc()

    call = _fallback_call(tire, gap, sc, state)

    assert call.action == Action.MONITOR
    assert 0.0 <= call.confidence <= 1.0


def test_fallback_returns_valid_strategy_call():
    """Fallback always returns a properly typed StrategyCall."""
    state = make_state()
    call = _fallback_call(make_tire(), make_gap(), make_sc(), state)
    assert isinstance(call, StrategyCall)
    assert call.action in (Action.BOX_NOW, Action.STAY_OUT, Action.MONITOR)
    assert 0.0 <= call.confidence <= 1.0


# ---------------------------------------------------------------------------
# _parse_claude_response tests
# ---------------------------------------------------------------------------


def test_parse_box_now():
    text = "BOX NOW — undercut window open, 2.1s gap to Verstappen"
    call = _parse_claude_response(text, driver=1, lap=30)
    assert call.action == Action.BOX_NOW
    assert "undercut" in call.reasoning.lower()


def test_parse_stay_out():
    text = "STAY OUT — SC window missed, tires have 8 laps left"
    call = _parse_claude_response(text, driver=1, lap=30)
    assert call.action == Action.STAY_OUT


def test_parse_monitor():
    text = "MONITOR — no decisive signal at this stage"
    call = _parse_claude_response(text, driver=1, lap=30)
    assert call.action == Action.MONITOR


def test_parse_confidence_clamped():
    """Confidence is always between 0.0 and 1.0."""
    text = "BOX NOW — pit now, confidence: 150%"
    call = _parse_claude_response(text, driver=1, lap=30)
    assert 0.0 <= call.confidence <= 1.0


def test_parse_default_confidence_when_absent():
    """When no confidence marker is present, a default is used."""
    text = "BOX NOW — switch to hards"
    call = _parse_claude_response(text, driver=1, lap=5)
    assert 0.0 <= call.confidence <= 1.0


def test_parse_driver_and_lap_attached():
    text = "STAY OUT — overcut viable"
    call = _parse_claude_response(text, driver=44, lap=55)
    assert call.driver == 44
    assert call.lap == 55


# ---------------------------------------------------------------------------
# Synthesizer class — no API key (fallback path)
# ---------------------------------------------------------------------------


def test_synthesizer_no_api_key_uses_fallback(monkeypatch):
    """When ANTHROPIC_API_KEY is unset, Synthesizer falls back gracefully."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    synth = Synthesizer()
    state = make_state()
    sc = make_sc(sc_active=True, pit_opportunity=True, reasoning="SC out")

    call = asyncio.get_event_loop().run_until_complete(
        synth.synthesize(make_tire(), make_gap(), sc, state)
    )

    assert isinstance(call, StrategyCall)
    assert call.action == Action.BOX_NOW


def test_synthesizer_no_api_key_never_raises(monkeypatch):
    """Synthesizer with no key never raises an exception."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    synth = Synthesizer()
    call = asyncio.get_event_loop().run_until_complete(
        synth.synthesize(make_tire(), make_gap(), make_sc(), make_state())
    )
    assert isinstance(call, StrategyCall)


# ---------------------------------------------------------------------------
# Synthesizer class — mocked Claude client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesizer_calls_claude_and_parses_response(monkeypatch):
    """When API key is present, Synthesizer calls Claude and parses its response."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

    mock_content = MagicMock()
    mock_content.text = "BOX NOW — undercut window open, 3.2s gap ahead"

    mock_response = MagicMock()
    mock_response.content = [mock_content]

    mock_create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = mock_create

        synth = Synthesizer()
        synth._client = instance

        call = await synth.synthesize(make_tire(), make_gap(), make_sc(), make_state())

    assert call.action == Action.BOX_NOW
    assert "undercut" in call.reasoning.lower()
    mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_synthesizer_falls_back_on_claude_exception(monkeypatch):
    """If Claude raises an exception, Synthesizer uses fallback logic."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

    mock_create = AsyncMock(side_effect=Exception("API timeout"))

    with patch("anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = mock_create

        synth = Synthesizer()
        synth._client = instance

        sc = make_sc(pit_opportunity=True, sc_active=True, reasoning="SC active")
        call = await synth.synthesize(make_tire(), make_gap(), sc, make_state())

    assert isinstance(call, StrategyCall)
    assert call.action == Action.BOX_NOW  # fallback SC priority


@pytest.mark.asyncio
async def test_synthesizer_prompt_includes_all_signals(monkeypatch):
    """The prompt sent to Claude includes tire, gap, and SC signal data."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

    mock_content = MagicMock()
    mock_content.text = "MONITOR — no signal"

    mock_response = MagicMock()
    mock_response.content = [mock_content]

    mock_create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = mock_create

        synth = Synthesizer()
        synth._client = instance

        tire = make_tire(recommend_pit=True, suggested_compound="HARD", deg_rate=0.09)
        gap = make_gap(undercut_viable=True, gap_ahead=18.5)
        sc = make_sc(sc_active=False, reasoning="Clear")
        state = make_state(lap=42, compound="SOFT", stint_lap=20)

        await synth.synthesize(tire, gap, sc, state)

    call_kwargs = mock_create.call_args
    user_content = call_kwargs.kwargs["messages"][0]["content"]

    assert "TIRE ANALYST" in user_content
    assert "GAP MONITOR" in user_content
    assert "SAFETY CAR DETECTOR" in user_content
    assert "42" in user_content  # lap number
    assert "18.50" in user_content or "18.5" in user_content  # gap_ahead
