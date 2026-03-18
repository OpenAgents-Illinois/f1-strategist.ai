import logging
import os
import re

import anthropic

from core.models import Action, GapSignal, SafetyCarSignal, StrategyCall, TireSignal
from core.race_state import RaceState

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM = """
You are a Formula 1 strategy engineer on the pit wall during a live race.
You receive signals from three specialist analysts every few seconds.
Your job: make one clear, confident strategy call per update.

Rules:
- Safety car opportunity always takes priority
- Be decisive. "Monitor" is only valid if no signal exceeds 70% confidence
- Keep calls under 20 words. Engineers are busy.
- Format: ACTION — one-line reasoning

Examples:
  BOX NOW — undercut window open, 2.1s gap to Verstappen
  STAY OUT — SC window missed, tires have 8 laps left
  BOX NOW — VSC deployed, free stop, switch to hards
"""

_USER_TEMPLATE = """\
Lap: {lap}
Driver: {driver}
Current compound: {compound} (stint lap {stint_lap})

TIRE ANALYST:
  recommend_pit={recommend_pit}, suggested_compound={suggested_compound}
  pit_window_laps={pit_window_laps}, deg_rate={deg_rate:.3f} s/lap

GAP MONITOR:
  undercut_viable={undercut_viable}, overcut_viable={overcut_viable}
  gap_ahead={gap_ahead:.2f}s, gap_behind={gap_behind:.2f}s

SAFETY CAR DETECTOR:
  sc_active={sc_active}, vsc_active={vsc_active}
  pit_opportunity={pit_opportunity}
  reasoning={sc_reasoning}

What is your strategy call?
"""

_ACTION_KEYWORDS = {
    "BOX NOW": Action.BOX_NOW,
    "STAY OUT": Action.STAY_OUT,
    "MONITOR": Action.MONITOR,
}


def _fallback_call(
    tire: TireSignal,
    gap: GapSignal,
    sc: SafetyCarSignal,
    state: RaceState,
) -> StrategyCall:
    """Derive a StrategyCall from signal priority without calling Claude."""
    if sc.pit_opportunity:
        return StrategyCall(
            driver=state.driver,
            action=Action.BOX_NOW,
            confidence=0.95,
            reasoning=sc.reasoning or "Safety car / VSC — free pit stop opportunity",
            lap=state.lap,
        )
    if gap.undercut_viable:
        return StrategyCall(
            driver=state.driver,
            action=Action.BOX_NOW,
            confidence=0.80,
            reasoning=f"Undercut window open, {gap.gap_ahead:.1f}s gap ahead",
            lap=state.lap,
        )
    if tire.recommend_pit:
        return StrategyCall(
            driver=state.driver,
            action=Action.BOX_NOW,
            confidence=0.75,
            reasoning=f"Tire degradation: {tire.deg_rate:.3f}s/lap, switch to {tire.suggested_compound}",
            lap=state.lap,
        )
    if gap.overcut_viable:
        return StrategyCall(
            driver=state.driver,
            action=Action.STAY_OUT,
            confidence=0.70,
            reasoning=f"Overcut viable, {gap.gap_behind:.1f}s gap behind",
            lap=state.lap,
        )
    return StrategyCall(
        driver=state.driver,
        action=Action.MONITOR,
        confidence=0.50,
        reasoning="No decisive signal — monitoring",
        lap=state.lap,
    )


def _parse_claude_response(
    text: str,
    driver: int,
    lap: int,
) -> StrategyCall:
    """Extract action, confidence, and reasoning from Claude's response text."""
    text = text.strip()

    action = Action.MONITOR
    for keyword, act in _ACTION_KEYWORDS.items():
        if keyword in text.upper():
            action = act
            break

    # Try to extract a confidence value like "confidence: 0.85" or "85%"
    confidence = 0.80
    pct_match = re.search(r"(\d{1,3})\s*%", text)
    decimal_match = re.search(r"confidence[:\s]+([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if decimal_match:
        try:
            confidence = float(decimal_match.group(1))
            if confidence > 1.0:
                confidence = confidence / 100.0
        except ValueError:
            pass
    elif pct_match:
        try:
            confidence = float(pct_match.group(1)) / 100.0
        except ValueError:
            pass

    confidence = max(0.0, min(1.0, confidence))

    # Reasoning is everything after the first "—" dash, or the full text
    reasoning = text
    dash_idx = text.find("—")
    if dash_idx != -1:
        reasoning = text[dash_idx + 1 :].strip()
    elif " - " in text:
        reasoning = text.split(" - ", 1)[1].strip()

    # Strip leading action keyword from reasoning if present
    for keyword in _ACTION_KEYWORDS:
        if reasoning.upper().startswith(keyword):
            reasoning = reasoning[len(keyword) :].strip().lstrip("—- ")
            break

    return StrategyCall(
        driver=driver,
        action=action,
        confidence=confidence,
        reasoning=reasoning or text,
        lap=lap,
    )


class Synthesizer:
    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._client: anthropic.AsyncAnthropic | None = None
        if api_key:
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        else:
            logger.warning(
                "ANTHROPIC_API_KEY not set — Synthesizer will use fallback logic only"
            )

    async def synthesize(
        self,
        tire: TireSignal,
        gap: GapSignal,
        sc: SafetyCarSignal,
        state: RaceState,
    ) -> StrategyCall:
        """Return a StrategyCall by asking Claude or falling back to priority logic."""
        if self._client is None:
            logger.debug("No API client — using fallback strategy call")
            return _fallback_call(tire, gap, sc, state)

        user_msg = _USER_TEMPLATE.format(
            lap=state.lap,
            driver=state.driver,
            compound=state.compound,
            stint_lap=state.stint_lap,
            recommend_pit=tire.recommend_pit,
            suggested_compound=tire.suggested_compound,
            pit_window_laps=tire.pit_window_laps,
            deg_rate=tire.deg_rate,
            undercut_viable=gap.undercut_viable,
            overcut_viable=gap.overcut_viable,
            gap_ahead=gap.gap_ahead,
            gap_behind=gap.gap_behind,
            sc_active=sc.sc_active,
            vsc_active=sc.vsc_active,
            pit_opportunity=sc.pit_opportunity,
            sc_reasoning=sc.reasoning,
        )

        try:
            response = await self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                system=SYNTHESIZER_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            reply = response.content[0].text
            return _parse_claude_response(reply, driver=state.driver, lap=state.lap)
        except Exception as exc:
            logger.error("Claude call failed (%s) — using fallback", exc)
            return _fallback_call(tire, gap, sc, state)
