"""Tests for brain.engines.daemon_state — SP-2.

Covers DaemonFireEntry, EmotionalResidue, DaemonState, load_daemon_state,
update_daemon_state, and get_residue_context.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.engines.daemon_state import (
    DaemonFireEntry,
    DaemonState,
    EmotionalResidue,
    get_residue_context,
    load_daemon_state,
    update_daemon_state,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fire(
    *,
    emotion: str = "love",
    intensity: int = 8,
    theme: str = "test theme",
    summary: str = "test summary",
    trigger: str | None = None,
    hours_ago: float = 1.0,
) -> DaemonFireEntry:
    ts = datetime.now(UTC) - timedelta(hours=hours_ago)
    return DaemonFireEntry(
        timestamp=ts,
        dominant_emotion=emotion,
        intensity=intensity,
        theme=theme,
        summary=summary,
        trigger=trigger,
    )


# ---------------------------------------------------------------------------
# DaemonFireEntry tests
# ---------------------------------------------------------------------------


def test_daemon_fire_entry_construction() -> None:
    """DaemonFireEntry stores fields correctly."""
    now = datetime.now(UTC)
    entry = DaemonFireEntry(
        timestamp=now,
        dominant_emotion="emergence",
        intensity=9,
        theme="the verb",
        summary="the world is a verb",
    )
    assert entry.dominant_emotion == "emergence"
    assert entry.intensity == 9
    assert entry.theme == "the verb"
    assert entry.trigger is None


def test_daemon_fire_entry_is_frozen() -> None:
    """DaemonFireEntry is immutable — assignment must raise."""
    entry = _fire()
    with pytest.raises((AttributeError, TypeError)):  # FrozenInstanceError (dataclasses)
        entry.intensity = 0  # type: ignore[misc]


def test_daemon_fire_entry_to_dict_round_trip() -> None:
    """to_dict / from_dict round-trip preserves all fields."""
    now = datetime.now(UTC).replace(microsecond=0)
    entry = DaemonFireEntry(
        timestamp=now,
        dominant_emotion="curiosity",
        intensity=7,
        theme="a question",
        summary="what is going on",
        trigger="some_arc",
    )
    d = entry.to_dict()
    restored = DaemonFireEntry.from_dict(d)

    assert restored.dominant_emotion == entry.dominant_emotion
    assert restored.intensity == entry.intensity
    assert restored.theme == entry.theme
    assert restored.summary == entry.summary
    assert restored.trigger == entry.trigger
    # Timestamps compare equal to the second (ISO round-trip loses sub-microsecond precision
    # only; microsecond=0 here keeps it lossless).
    assert abs((restored.timestamp - entry.timestamp).total_seconds()) < 1.0


def test_daemon_fire_entry_trigger_omitted_when_none() -> None:
    """to_dict does NOT include 'trigger' key when trigger is None."""
    entry = _fire(trigger=None)
    d = entry.to_dict()
    assert "trigger" not in d


def test_daemon_fire_entry_trigger_present_when_set() -> None:
    """to_dict includes 'trigger' key when trigger is set."""
    entry = _fire(trigger="gratitude_reflection")
    d = entry.to_dict()
    assert d["trigger"] == "gratitude_reflection"


def test_daemon_fire_entry_truncates_summary_to_250() -> None:
    """DaemonFireEntry truncates summary to 250 chars in __post_init__."""
    long_summary = "x" * 400
    entry = _fire(summary=long_summary)
    assert len(entry.summary) == 250


# ---------------------------------------------------------------------------
# EmotionalResidue tests
# ---------------------------------------------------------------------------


def test_emotional_residue_from_fire_computes_intensity() -> None:
    """from_fire computes intensity = max(1, int(source_intensity * 0.4))."""
    fire = _fire(intensity=9)
    residue = EmotionalResidue.from_fire("dream", fire)
    assert residue.intensity == max(1, int(9 * 0.4))  # → 3


def test_emotional_residue_from_fire_intensity_minimum_is_1() -> None:
    """Intensity of 0 on the fire still yields residue intensity of 1."""
    fire = _fire(intensity=0)
    residue = EmotionalResidue.from_fire("heartbeat", fire)
    assert residue.intensity == 1


def test_emotional_residue_from_fire_computes_decays_by() -> None:
    """from_fire sets decays_by = timestamp + 12 hours."""
    fire = _fire()
    residue = EmotionalResidue.from_fire("research", fire)
    expected = fire.timestamp + timedelta(hours=12)
    assert abs((residue.decays_by - expected).total_seconds()) < 1.0


def test_emotional_residue_is_expired_true_past_window() -> None:
    """is_expired returns True when now > decays_by."""
    fire = _fire(hours_ago=14)  # 14h old → decays_by was 2h ago
    residue = EmotionalResidue.from_fire("dream", fire)
    assert residue.is_expired(datetime.now(UTC))


def test_emotional_residue_is_expired_false_within_window() -> None:
    """is_expired returns False when now < decays_by."""
    fire = _fire(hours_ago=1)  # 1h old → decays_by is 11h from now
    residue = EmotionalResidue.from_fire("dream", fire)
    assert not residue.is_expired(datetime.now(UTC))


def test_emotional_residue_to_dict_round_trip() -> None:
    """to_dict / from_dict round-trip preserves all fields."""
    fire = _fire(emotion="tenderness", intensity=7)
    residue = EmotionalResidue.from_fire("reflex", fire)
    restored = EmotionalResidue.from_dict(residue.to_dict())
    assert restored.emotion == residue.emotion
    assert restored.intensity == residue.intensity
    assert restored.source == residue.source
    assert abs((restored.decays_by - residue.decays_by).total_seconds()) < 1.0


# ---------------------------------------------------------------------------
# DaemonState tests
# ---------------------------------------------------------------------------


def test_daemon_state_empty_to_dict_is_empty_dict() -> None:
    """Empty DaemonState serialises to {} (no last_* keys)."""
    state = DaemonState()
    assert state.to_dict() == {}


def test_daemon_state_round_trip() -> None:
    """DaemonState.from_dict(state.to_dict()) preserves all populated fields."""
    fire = _fire(emotion="emergence", intensity=9)
    residue = EmotionalResidue.from_fire("dream", fire)
    original = DaemonState(
        last_dream=fire,
        last_heartbeat=_fire(emotion="calm", intensity=4),
        emotional_residue=residue,
    )
    d = original.to_dict()
    restored = DaemonState.from_dict(d)

    assert restored.last_dream is not None
    assert restored.last_dream.dominant_emotion == "emergence"
    assert restored.last_heartbeat is not None
    assert restored.last_heartbeat.dominant_emotion == "calm"
    assert restored.last_reflex is None
    assert restored.last_research is None
    assert restored.emotional_residue is not None
    assert restored.emotional_residue.emotion == "emergence"


# ---------------------------------------------------------------------------
# load_daemon_state tests
# ---------------------------------------------------------------------------


def test_load_daemon_state_missing_file_returns_empty(tmp_path: Path) -> None:
    """Missing daemon_state.json → (DaemonState(), None) silently."""
    state, anomaly = load_daemon_state(tmp_path)
    assert state == DaemonState()
    assert anomaly is None


def test_load_daemon_state_corrupt_json_heals_and_returns_anomaly(tmp_path: Path) -> None:
    """Corrupt daemon_state.json triggers attempt_heal and returns an anomaly."""
    (tmp_path / "daemon_state.json").write_text("not valid json {{{", encoding="utf-8")
    state, anomaly = load_daemon_state(tmp_path)
    # State is empty (default after heal-fail)
    assert state == DaemonState()
    # An anomaly was raised
    assert anomaly is not None


def test_load_daemon_state_valid_file_parses(tmp_path: Path) -> None:
    """Valid daemon_state.json loads and parses without anomaly."""
    fire = _fire(emotion="love", intensity=9)
    raw = DaemonState(last_dream=fire).to_dict()
    (tmp_path / "daemon_state.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    state, anomaly = load_daemon_state(tmp_path)
    assert anomaly is None
    assert state.last_dream is not None
    assert state.last_dream.dominant_emotion == "love"


# ---------------------------------------------------------------------------
# update_daemon_state tests
# ---------------------------------------------------------------------------


def test_update_daemon_state_writes_correct_fire_and_residue(tmp_path: Path) -> None:
    """update_daemon_state persists the fire entry and recomputes residue."""
    state = update_daemon_state(
        tmp_path,
        daemon_type="dream",
        dominant_emotion="emergence",
        intensity=9,
        theme="the verb",
        summary="the world is a verb",
    )
    assert state.last_dream is not None
    assert state.last_dream.dominant_emotion == "emergence"
    assert state.last_dream.intensity == 9
    assert state.emotional_residue is not None
    assert state.emotional_residue.emotion == "emergence"
    assert state.emotional_residue.intensity == max(1, int(9 * 0.4))


def test_update_daemon_state_persists_across_reload(tmp_path: Path) -> None:
    """State written by update_daemon_state is readable by load_daemon_state."""
    update_daemon_state(
        tmp_path,
        daemon_type="heartbeat",
        dominant_emotion="calm",
        intensity=5,
        theme="tick",
        summary="just ticking",
    )
    loaded, anomaly = load_daemon_state(tmp_path)
    assert anomaly is None
    assert loaded.last_heartbeat is not None
    assert loaded.last_heartbeat.dominant_emotion == "calm"


def test_update_daemon_state_preserves_other_engine_entries(tmp_path: Path) -> None:
    """update_daemon_state only replaces the fired engine's entry; others survive."""
    # First write dream entry
    update_daemon_state(
        tmp_path,
        daemon_type="dream",
        dominant_emotion="emergence",
        intensity=9,
        theme="t",
        summary="s",
    )
    # Now write heartbeat — dream entry must survive
    update_daemon_state(
        tmp_path,
        daemon_type="heartbeat",
        dominant_emotion="calm",
        intensity=4,
        theme="tick",
        summary="ticking",
    )
    state, _ = load_daemon_state(tmp_path)
    assert state.last_dream is not None
    assert state.last_dream.dominant_emotion == "emergence"
    assert state.last_heartbeat is not None
    assert state.last_heartbeat.dominant_emotion == "calm"


def test_update_daemon_state_truncates_summary_to_250(tmp_path: Path) -> None:
    """update_daemon_state truncates summary > 250 chars before persisting."""
    long_summary = "z" * 500
    state = update_daemon_state(
        tmp_path,
        daemon_type="research",
        dominant_emotion="curiosity",
        intensity=6,
        theme="topic",
        summary=long_summary,
    )
    assert len(state.last_research.summary) == 250  # type: ignore[union-attr]


def test_update_daemon_state_reflex_sets_trigger(tmp_path: Path) -> None:
    """update_daemon_state with reflex daemon_type persists the trigger field."""
    state = update_daemon_state(
        tmp_path,
        daemon_type="reflex",
        dominant_emotion="love",
        intensity=10,
        theme="gratitude_reflection",
        summary="reflex fired",
        trigger="gratitude_reflection",
    )
    assert state.last_reflex is not None
    assert state.last_reflex.trigger == "gratitude_reflection"


# ---------------------------------------------------------------------------
# get_residue_context tests
# ---------------------------------------------------------------------------


def test_get_residue_context_empty_state_returns_empty_string() -> None:
    """Empty DaemonState → empty context string."""
    assert get_residue_context(DaemonState()) == ""


def test_get_residue_context_filters_fires_older_than_48h() -> None:
    """Fires older than 48 hours are excluded from the context."""
    old_fire = _fire(hours_ago=50)
    state = DaemonState(last_dream=old_fire)
    assert get_residue_context(state) == ""


def test_get_residue_context_includes_recent_fires() -> None:
    """Recent fires (< 48h) appear in the context string."""
    recent_fire = _fire(emotion="love", intensity=8, summary="a warm dream", hours_ago=2)
    state = DaemonState(last_dream=recent_fire)
    ctx = get_residue_context(state)
    assert "Previous dream" in ctx
    assert "a warm dream" in ctx


def test_get_residue_context_skips_expired_residue() -> None:
    """Expired residue (now > decays_by) is not included in context."""
    old_fire = _fire(hours_ago=14)  # residue decays after 12h → expired
    residue = EmotionalResidue.from_fire("dream", old_fire)
    state = DaemonState(emotional_residue=residue)
    ctx = get_residue_context(state)
    assert "Emotional residue" not in ctx


def test_get_residue_context_includes_active_residue() -> None:
    """Non-expired residue appears in context."""
    recent_fire = _fire(emotion="emergence", intensity=7, hours_ago=1)
    residue = EmotionalResidue.from_fire("research", recent_fire)
    state = DaemonState(emotional_residue=residue)
    ctx = get_residue_context(state)
    assert "Emotional residue" in ctx
    assert "emergence" in ctx


def test_get_residue_context_formats_hours_ago_and_truncated_summary() -> None:
    """Context line includes hours-ago count and summary truncated to 200 chars."""
    long_summary = "word " * 100  # 500 chars
    fire = _fire(summary=long_summary[:250], hours_ago=3)
    state = DaemonState(last_heartbeat=fire)
    ctx = get_residue_context(state)
    # Hours-ago present
    assert "3h ago" in ctx
    # Summary appears in context, capped at 200 chars (plus the quotes in the format string)
    # The full summary is 250 chars; context clips to 200.
    assert len([line for line in ctx.splitlines() if "Previous heartbeat" in line]) == 1
    context_line = next(line for line in ctx.splitlines() if "Previous heartbeat" in line)
    # Extract the quoted summary from the line
    quoted_start = context_line.index('"') + 1
    quoted_end = context_line.rindex('"')
    extracted_summary = context_line[quoted_start:quoted_end]
    assert len(extracted_summary) <= 200
