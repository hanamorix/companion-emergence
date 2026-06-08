"""Tests for brain.felt_time.prompt — render_prompt_context()."""

from datetime import UTC, datetime

from brain.felt_time.prompt import render_prompt_context
from brain.felt_time.state import Anchor, FeltTimeState, HorizonBucket, PressureCounters


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
    assert "baseline pace" in blob  # lived-age pace annotation
    assert "the boat one" in blob
    assert "sorrow lifted" in blob
    assert "current stretch (since the" in blob
    # dream is the newest anchor (2026-05-17) — label should say "dream"
    assert "since the dream" in blob
    assert any(tag in blob for tag in ("quiet", "steady", "dense"))


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


def test_render_prompt_context_uses_dense_tag_for_high_activity():
    s = FeltTimeState(
        lived_age_hours=10.0,
        anchors={
            "dream": Anchor("dream", "2026-05-17T20:00:00+00:00", "x", "dreams.log.jsonl:1"),
        },
        pressure=PressureCounters(
            heartbeats=300, chat_turns=25, reflex_firings=5, wall_clock_s=10000.0
        ),
        last_tick_ts="2026-05-19T10:00:00+00:00",
    )
    blob = render_prompt_context(s)
    assert "dense" in blob


_NOW = datetime(2026, 6, 8, 10, 0, 0, tzinfo=UTC)
_NOW_ISO = "2026-06-08T10:00:00+00:00"
_WEEK_START = "2026-06-02T10:00:00+00:00"  # 6 days before _NOW


def _state_with_horizons(current_chat_turns: int, prev_chat_turns: int) -> FeltTimeState:
    return FeltTimeState(
        lived_age_hours=100.0,
        anchors={
            "dream": Anchor("dream", "2026-06-07T10:00:00+00:00", "the boat one", "x:1"),
        },
        pressure=PressureCounters(heartbeats=10, chat_turns=5),
        last_tick_ts=_NOW_ISO,
        horizon_pressure={
            "week": HorizonBucket(
                counters=PressureCounters(chat_turns=current_chat_turns),
                prev_counters=PressureCounters(chat_turns=prev_chat_turns),
                period_start_ts=_WEEK_START,
            ),
            "month": HorizonBucket(
                counters=PressureCounters(chat_turns=current_chat_turns),
                prev_counters=PressureCounters(chat_turns=prev_chat_turns),
                period_start_ts="2026-05-09T10:00:00+00:00",
            ),
        },
    )


def test_render_contrast_denser_when_40_percent_above():
    # current week: 14 turns / 6 days = 2.33/day; prev: 7/7 = 1.0/day → ratio 2.33
    s = _state_with_horizons(current_chat_turns=14, prev_chat_turns=7)
    result = render_prompt_context(s, now=_NOW)
    assert "denser" in result


def test_render_contrast_quieter_when_40_percent_below():
    # current: 3/6 = 0.5/day; prev: 7/7 = 1.0/day → ratio 0.5
    s = _state_with_horizons(current_chat_turns=3, prev_chat_turns=7)
    result = render_prompt_context(s, now=_NOW)
    assert "quieter" in result


def test_render_no_contrast_within_threshold():
    # current: 6/6 = 1.0/day; prev: 7/7 = 1.0/day → ratio 1.0 (within 30%)
    s = _state_with_horizons(current_chat_turns=6, prev_chat_turns=7)
    result = render_prompt_context(s, now=_NOW)
    assert "denser" not in result
    assert "quieter" not in result


def test_render_only_most_pronounced_contrast():
    s = FeltTimeState(
        lived_age_hours=100.0,
        anchors={"dream": Anchor("dream", "2026-06-07T10:00:00+00:00", "x", "x:1")},
        pressure=PressureCounters(heartbeats=10),
        last_tick_ts=_NOW_ISO,
        horizon_pressure={
            "week": HorizonBucket(
                counters=PressureCounters(chat_turns=9),
                prev_counters=PressureCounters(chat_turns=7),
                period_start_ts=_WEEK_START,
            ),
            "month": HorizonBucket(
                counters=PressureCounters(chat_turns=50),
                prev_counters=PressureCounters(chat_turns=10),
                period_start_ts="2026-05-10T10:00:00+00:00",
            ),
        },
    )
    result = render_prompt_context(s, now=_NOW)
    assert result.count("denser") + result.count("quieter") == 1


def test_render_no_contrast_when_prev_is_zero():
    s = _state_with_horizons(current_chat_turns=10, prev_chat_turns=0)
    result = render_prompt_context(s, now=_NOW)
    assert "denser" not in result
    assert "quieter" not in result


def _arc(title: str, ts: str, event_type: str) -> "Anchor":
    return Anchor(type="arc", ts=ts, label=title,
                  source_ref="arcs.log.jsonl:1", event_type=event_type)


def test_render_two_open_arcs():
    s = FeltTimeState(
        lived_age_hours=50.0,
        arc_anchors=[
            _arc("Thread A", "2026-06-01T10:00:00+00:00", "arc_opened"),
            _arc("Thread B", "2026-06-05T10:00:00+00:00", "arc_opened"),
        ],
    )
    result = render_prompt_context(s, now=_NOW)
    assert "Thread A" in result
    assert "Thread B" in result
    assert "open threads" in result


def test_render_one_open_arc():
    s = FeltTimeState(
        lived_age_hours=50.0,
        arc_anchors=[_arc("Thread A", "2026-06-01T10:00:00+00:00", "arc_opened")],
    )
    result = render_prompt_context(s, now=_NOW)
    assert "Thread A" in result
    assert "open thread:" in result


def test_render_no_arc_line_when_all_closed():
    s = FeltTimeState(
        lived_age_hours=50.0,
        arc_anchors=[
            _arc("Thread A", "2026-06-01T10:00:00+00:00", "arc_opened"),
            _arc("Thread A", "2026-06-03T10:00:00+00:00", "arc_closed"),
        ],
    )
    result = render_prompt_context(s, now=_NOW)
    assert "open thread" not in result


def test_render_arc_max_two_open():
    s = FeltTimeState(
        lived_age_hours=50.0,
        arc_anchors=[
            _arc("Thread A", "2026-06-01T10:00:00+00:00", "arc_opened"),
            _arc("Thread B", "2026-06-03T10:00:00+00:00", "arc_opened"),
            _arc("Thread C", "2026-06-05T10:00:00+00:00", "arc_opened"),
        ],
    )
    result = render_prompt_context(s, now=_NOW)
    assert "Thread C" in result
    assert "Thread B" in result
    assert "Thread A" not in result


def test_render_recently_closed_arc_summary():
    s = FeltTimeState(
        lived_age_hours=50.0,
        arc_anchors=[
            _arc("Thread A", "2026-06-01T10:00:00+00:00", "arc_opened"),
            _arc("Thread A", "2026-06-06T10:00:00+00:00", "arc_closed"),  # 2 days ago
        ],
    )
    result = render_prompt_context(s, now=_NOW)
    assert "closed recently" in result
    assert "open thread" not in result


def test_render_token_budget_with_arcs_and_horizons():
    s = FeltTimeState(
        lived_age_hours=1000.0,
        anchors={"dream": Anchor("dream", "2026-06-07T10:00:00+00:00", "x" * 40, "x:1")},
        pressure=PressureCounters(heartbeats=300, chat_turns=25),
        horizon_pressure={
            "week": HorizonBucket(
                counters=PressureCounters(chat_turns=20),
                prev_counters=PressureCounters(chat_turns=10),
                period_start_ts=_WEEK_START,
            ),
            "month": HorizonBucket(
                counters=PressureCounters(chat_turns=80),
                prev_counters=PressureCounters(chat_turns=40),
                period_start_ts="2026-05-09T10:00:00+00:00",
            ),
        },
        arc_anchors=[
            _arc("Thread A", "2026-06-01T10:00:00+00:00", "arc_opened"),
            _arc("Thread B", "2026-06-05T10:00:00+00:00", "arc_opened"),
        ],
        last_tick_ts=_NOW_ISO,
    )
    result = render_prompt_context(s, now=_NOW)
    word_count = len(result.split())
    assert word_count <= 200
