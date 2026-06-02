"""Tests for supplementary backfill — schema-upgrade path for new categories."""
from __future__ import annotations

import json
from pathlib import Path

from brain.attunement.backfill import should_run_supplementary_backfill
from brain.attunement.schemas import SCHEMA_VERSION


def _state(tmp_path: Path, version: str, status: str = "complete"):
    d = tmp_path / "attunement"
    d.mkdir(parents=True, exist_ok=True)
    (d / "backfill_state.json").write_text(json.dumps({
        "started_at": "x", "total_windows": 1, "sampled_windows": 1,
        "processed_windows": 1, "patterns_emitted": 0, "status": status,
        "last_cursor": "", "schema_version": version,
    }))


def test_supplementary_runs_when_prior_complete_at_older_version(tmp_path: Path):
    _state(tmp_path, version="0.0.28-alpha.1", status="complete")
    assert should_run_supplementary_backfill(tmp_path) is True


def test_no_supplementary_when_already_current(tmp_path: Path):
    _state(tmp_path, version=SCHEMA_VERSION, status="complete")
    assert should_run_supplementary_backfill(tmp_path) is False


def test_no_supplementary_when_no_prior_backfill(tmp_path: Path):
    assert should_run_supplementary_backfill(tmp_path) is False


# ---------------------------------------------------------------------------
# Part 3 — run_supplementary_backfill
# ---------------------------------------------------------------------------

def _make_buffer_file(persona_dir: Path, n_turns: int = 15) -> list[dict]:
    """Write n_turns user messages to active_conversations/main.jsonl."""
    convs = persona_dir / "active_conversations"
    convs.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_turns):
        rows.append({
            "id": f"m-{i}",
            "role": "user",
            "ts": f"2026-05-{(i % 28) + 1:02d}T12:00:00Z",
            "content": f"this is turn number {i} about various topics",
        })
    import json as _json
    (convs / "main.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in rows) + "\n"
    )
    return rows


def test_run_supplementary_backfill_completes_at_current_schema_version(tmp_path):
    """After supplementary backfill, state is complete at the new SCHEMA_VERSION."""
    from datetime import UTC, datetime

    from brain.attunement.backfill import run_supplementary_backfill
    from brain.attunement.schemas import CurrentRead, DetectorOutput, Evidence, PatternCandidate

    rows = _make_buffer_file(tmp_path, n_turns=15)
    # Seed an alpha.1-complete state (older schema version)
    _state(tmp_path, version="0.0.28-alpha.1", status="complete")

    # Stub detector: returns one topic_affinity candidate grounded in the first turn
    first_content = rows[0]["content"]
    first_id = rows[0]["id"]

    def stub_detector(*, buffer_slice, reply_text):
        return DetectorOutput(
            current_read=CurrentRead(
                ts="2026-06-01T12:00:00Z",
                source_turn_id=first_id,
                tone_label="unknown",
                tone_justification="",
                cadence_label="unknown",
                cadence_justification="",
                mood_valence=0.0,
                mood_intensity=0.0,
                predicted_arc_shape="",
                schema_version=SCHEMA_VERSION,
            ),
            pattern_candidates=[
                PatternCandidate(
                    category="topic_affinity",
                    canonical_key="topic_affinity:various-topics",
                    description="drawn to various topics",
                    evidence=[Evidence(quote=first_content, turn_id=first_id)],
                )
            ],
        )

    state = run_supplementary_backfill(
        tmp_path,
        detector_fn=stub_detector,
        now_dt=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
    )

    assert state.schema_version == SCHEMA_VERSION
    assert state.status == "complete"

    # Verify a topic_affinity pattern was written to learned_patterns
    from brain.attunement.store import read_learned_patterns
    patterns = read_learned_patterns(tmp_path)
    topic_patterns = [p for p in patterns if p.category == "topic_affinity"]
    assert len(topic_patterns) >= 1


def test_supplementary_resets_cursor_not_inheriting_stale_completed_cursor(tmp_path):
    """A supplementary pass must start from an EMPTY cursor, not inherit the
    prior completed full-backfill's end-of-list cursor. Otherwise, if the daily
    cap is exhausted before the first window, it defers with a stale end cursor
    and on resume skips every window — bootstrapping zero new-category patterns."""
    from datetime import UTC, datetime

    from brain.attunement.backfill import run_supplementary_backfill

    _make_buffer_file(tmp_path, n_turns=15)
    # Completed alpha.1 state whose cursor points at the LAST turn.
    d = tmp_path / "attunement"
    d.mkdir(parents=True, exist_ok=True)
    (d / "backfill_state.json").write_text(json.dumps({
        "started_at": "x", "total_windows": 5, "sampled_windows": 5,
        "processed_windows": 5, "patterns_emitted": 3, "status": "complete",
        "last_cursor": "m-14", "schema_version": "0.0.28-alpha.1",
    }))

    # cap=0 → budget exhausted at the first window → defers immediately, before
    # any window updates the cursor. The persisted cursor is whatever the fresh
    # state was initialised with — which must be "" for a supplementary pass.
    state = run_supplementary_backfill(
        tmp_path,
        detector_fn=lambda *, buffer_slice, reply_text: None,  # never called (cap=0)
        now_dt=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        cap=0,
    )

    assert state.status == "deferred_to_next_day"
    assert state.last_cursor == "", (
        f"supplementary pass inherited stale cursor {state.last_cursor!r}"
    )


def test_supplementary_preserves_prior_completion_record(tmp_path):
    """A supplementary pass augments the prior backfill — it must NOT clobber
    the original started_at or reset patterns_emitted. The completed record
    (what the attunement endpoint reports) survives the upgrade; the new pass
    accumulates onto it."""
    from datetime import UTC, datetime

    from brain.attunement.backfill import run_supplementary_backfill
    from brain.attunement.schemas import (
        CurrentRead,
        DetectorOutput,
        Evidence,
        PatternCandidate,
    )

    rows = _make_buffer_file(tmp_path, n_turns=15)
    original_started_at = "2026-05-31T08:00:00+00:00"
    d = tmp_path / "attunement"
    d.mkdir(parents=True, exist_ok=True)
    (d / "backfill_state.json").write_text(json.dumps({
        "started_at": original_started_at, "total_windows": 20, "sampled_windows": 10,
        "processed_windows": 10, "patterns_emitted": 3, "status": "complete",
        "last_cursor": "turn_xyz", "schema_version": "0.0.28-alpha.1",
    }))

    first_content, first_id = rows[0]["content"], rows[0]["id"]

    def stub_detector(*, buffer_slice, reply_text):
        return DetectorOutput(
            current_read=CurrentRead(
                ts="2026-06-01T12:00:00Z", source_turn_id=first_id,
                tone_label="unknown", tone_justification="",
                cadence_label="unknown", cadence_justification="",
                mood_valence=0.0, mood_intensity=0.0, predicted_arc_shape="",
                schema_version=SCHEMA_VERSION,
            ),
            pattern_candidates=[PatternCandidate(
                category="topic_affinity", canonical_key="topic_affinity:various-topics",
                description="drawn to various topics",
                evidence=[Evidence(quote=first_content, turn_id=first_id)],
            )],
        )

    state = run_supplementary_backfill(
        tmp_path, detector_fn=stub_detector,
        now_dt=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
    )

    # Original journey-start preserved (not reset to now); new patterns accumulate.
    assert state.started_at == original_started_at
    assert state.patterns_emitted >= 3
    assert state.schema_version == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Part 4 — supervisor wiring test
# ---------------------------------------------------------------------------

def test_supervisor_calls_supplementary_backfill_when_schema_upgraded(tmp_path):
    """Supervisor startup must call supplementary backfill when prior completed
    state is at an older schema version (mutually exclusive with full backfill)."""
    import threading
    from unittest.mock import patch

    from brain.bridge.events import EventBus
    from brain.bridge.provider import FakeProvider
    from brain.bridge.supervisor import run_folded

    persona_dir = tmp_path / "test-persona"
    persona_dir.mkdir()
    (persona_dir / "active_conversations").mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}')

    stop = threading.Event()
    stop.set()  # exit after startup

    with patch("brain.bridge.supervisor._attunement_should_run_backfill") as mock_full_should, \
         patch("brain.bridge.supervisor._attunement_run_backfill") as mock_full_run, \
         patch("brain.bridge.supervisor._attunement_should_run_supplementary_backfill") as mock_supp_should, \
         patch("brain.bridge.supervisor._attunement_run_supplementary_backfill") as mock_supp_run:

        # Full backfill does not fire (prior completed state exists)
        mock_full_should.return_value = False
        mock_supp_should.return_value = True

        run_folded(
            stop,
            persona_dir=persona_dir,
            provider=FakeProvider(),
            event_bus=EventBus(),
            tick_interval_s=0.0,
            heartbeat_interval_s=None,
            soul_review_interval_s=None,
            finalize_interval_s=None,
            log_rotation_interval_s=None,
            initiate_review_interval_s=None,
            voice_reflection_interval_s=None,
        )

        mock_full_should.assert_called_once_with(persona_dir)
        mock_full_run.assert_not_called()
        mock_supp_should.assert_called_once_with(persona_dir)
        mock_supp_run.assert_called_once_with(persona_dir)
