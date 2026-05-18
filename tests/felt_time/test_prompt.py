"""Tests for brain.felt_time.prompt — render_prompt_context()."""

from brain.felt_time.prompt import render_prompt_context
from brain.felt_time.state import Anchor, FeltTimeState, PressureCounters


def test_render_prompt_context_cold_start():
    s = FeltTimeState.cold_start()
    blob = render_prompt_context(s)
    assert "too new" in blob.lower()
    assert blob.count("\n") < 5


def test_render_prompt_context_full_state():
    s = FeltTimeState(
        lived_age_hours=412.7,
        anchors={
            "dream": Anchor(
                "dream", "2026-05-17T20:00:00+00:00", "the boat one", "dreams.log.jsonl:1"
            ),
            "growth": Anchor(
                "growth", "2026-05-13T18:00:00+00:00", "the way you ask", "growth.log.jsonl:2"
            ),
            "soul": Anchor(
                "soul", "2026-05-06T22:00:00+00:00", "the kitchen one", "soul.log.jsonl:1"
            ),
            "weather_shift": Anchor(
                "weather_shift",
                "2026-05-11T08:00:00+00:00",
                "sorrow lifted",
                "weather_shifts.log.jsonl:1",
            ),
        },
        pressure=PressureCounters(
            heartbeats=152, chat_turns=0, reflex_firings=3, wall_clock_s=137580.0
        ),
        last_tick_ts="2026-05-19T10:00:00+00:00",
    )
    blob = render_prompt_context(s)
    assert "felt time" in blob.lower()
    assert "412" in blob  # lived age
    assert "the boat one" in blob
    assert "sorrow lifted" in blob


def test_render_prompt_context_stays_under_token_budget():
    # Use a generous fake state — even the worst case should stay <= 150 tokens.
    s = FeltTimeState(
        lived_age_hours=9999.9,
        anchors={
            "dream": Anchor("dream", "2026-05-17T20:00:00+00:00", "a" * 80, "dreams.log.jsonl:1"),
            "growth": Anchor("growth", "2026-05-13T18:00:00+00:00", "b" * 80, "growth.log.jsonl:2"),
            "soul": Anchor("soul", "2026-05-06T22:00:00+00:00", "c" * 80, "soul.log.jsonl:1"),
            "weather_shift": Anchor(
                "weather_shift", "2026-05-11T08:00:00+00:00", "d" * 80, "weather_shifts.log.jsonl:1"
            ),
        },
        pressure=PressureCounters(
            heartbeats=99999, chat_turns=9999, reflex_firings=9999, wall_clock_s=9_999_999.0
        ),
        last_tick_ts="2026-05-19T10:00:00+00:00",
    )
    blob = render_prompt_context(s)
    # Rough token estimate: 1 token ≈ 4 chars in English. 150 tokens ≈ 600 chars budget.
    assert len(blob) <= 600
