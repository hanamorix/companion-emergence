"""Tests for run_backfill — state file + cursor + cap-aware splitting."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from brain.attunement.backfill import run_backfill
from brain.attunement.schemas import (
    SCHEMA_VERSION,
    CurrentRead,
    DetectorOutput,
)


def _make_buffer_file(persona_dir: Path, n_turns: int) -> None:
    """Seed active_conversations/main.jsonl with n_turns user messages."""
    convs = persona_dir / "active_conversations"
    convs.mkdir(parents=True, exist_ok=True)
    out = convs / "main.jsonl"
    lines = []
    for i in range(n_turns):
        lines.append(json.dumps({
            "id": f"m-{i}",
            "role": "user",
            "ts": f"2026-05-{(i % 28) + 1:02d}T12:00:00Z",
            "content": f"this is message number {i} in the conversation",
        }))
    out.write_text("\n".join(lines) + "\n")


def _fake_detector_output() -> DetectorOutput:
    return DetectorOutput(
        current_read=CurrentRead(
            ts="2026-05-31T12:00:00Z",
            source_turn_id="m-0",
            tone_label="warm",
            tone_justification="x",
            cadence_label="measured",
            cadence_justification="y",
            mood_valence=0.0,
            mood_intensity=0.0,
            predicted_arc_shape="z",
            schema_version=SCHEMA_VERSION,
        ),
        pattern_candidates=[],
    )


def _fake_detector(buffer_slice, reply_text) -> DetectorOutput:  # noqa: ANN001
    return _fake_detector_output()


def test_run_backfill_completes_status_when_all_windows_processed(tmp_path: Path):
    _make_buffer_file(tmp_path, n_turns=25)
    fake_detector = _fake_detector
    state = run_backfill(
        tmp_path,
        detector_fn=fake_detector,
        now_dt=datetime(2026, 5, 31, 12, 0, tzinfo=UTC),
    )
    assert state.status == "complete"
    assert state.processed_windows == state.sampled_windows


def test_run_backfill_respects_daily_cap_splits_across_days(tmp_path: Path):
    # Use n_turns=500 to guarantee enough windows that cap=2 is exceeded
    # (near-identical content clusters into few windows; 500 turns → ~49 windows → ~10 sampled)
    _make_buffer_file(tmp_path, n_turns=500)
    fake_detector = _fake_detector
    state = run_backfill(
        tmp_path,
        detector_fn=fake_detector,
        now_dt=datetime(2026, 5, 31, 12, 0, tzinfo=UTC),
        cap=2,  # tiny cap forces split
    )
    assert state.status == "deferred_to_next_day"
    assert state.last_cursor != ""


def test_run_backfill_resumes_from_last_cursor(tmp_path: Path):
    _make_buffer_file(tmp_path, n_turns=500)
    fake_detector = _fake_detector

    # First run: deferred at cap=2
    first = run_backfill(
        tmp_path,
        detector_fn=fake_detector,
        now_dt=datetime(2026, 5, 31, 12, 0, tzinfo=UTC),
        cap=2,
    )
    assert first.status == "deferred_to_next_day"
    first_processed = first.processed_windows

    # Second run: fresh cap (different day), resumes past cursor
    second = run_backfill(
        tmp_path,
        detector_fn=fake_detector,
        now_dt=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        cap=100,
    )
    assert second.status == "complete"
    assert second.processed_windows > first_processed


def test_run_backfill_returns_existing_when_complete(tmp_path: Path):
    _make_buffer_file(tmp_path, n_turns=25)
    fake_detector = _fake_detector
    first = run_backfill(
        tmp_path,
        detector_fn=fake_detector,
        now_dt=datetime(2026, 5, 31, 12, 0, tzinfo=UTC),
    )
    assert first.status == "complete"
    # Second call: short-circuit before window_buffer is reached
    with patch("brain.attunement.backfill.window_buffer") as mock_window:
        second = run_backfill(
            tmp_path,
            detector_fn=fake_detector,
            now_dt=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        )
    assert second.status == "complete"
    mock_window.assert_not_called()


def test_run_backfill_skips_when_no_active_conversations(tmp_path: Path):
    fake_detector = _fake_detector
    state = run_backfill(
        tmp_path,
        detector_fn=fake_detector,
        now_dt=datetime(2026, 5, 31, 12, 0, tzinfo=UTC),
    )
    assert state.status == "complete"
    assert state.processed_windows == 0
    assert state.sampled_windows == 0


def test_run_backfill_persists_state_after_each_window(tmp_path: Path):
    _make_buffer_file(tmp_path, n_turns=30)
    fake_detector = _fake_detector
    run_backfill(
        tmp_path,
        detector_fn=fake_detector,
        now_dt=datetime(2026, 5, 31, 12, 0, tzinfo=UTC),
    )
    state_path = tmp_path / "attunement" / "backfill_state.json"
    assert state_path.exists()
    payload = json.loads(state_path.read_text())
    assert payload["status"] == "complete"
    assert payload["schema_version"] == SCHEMA_VERSION


def test_run_backfill_default_detector_threads_identity(tmp_path: Path):
    """v0.0.33 identity-grounding: the default detector closure passes
    companion_name (persona dir) AND user_name (persona_config.json) so
    backfilled pattern descriptions name the companion correctly."""
    _make_buffer_file(tmp_path, n_turns=25)
    (tmp_path / "persona_config.json").write_text(json.dumps({"user_name": "Alex"}))
    captured: list[dict] = []

    def _capture(**kwargs):  # noqa: ANN003
        captured.append(kwargs)
        return _fake_detector_output()

    with patch("brain.attunement.detector.run_detector", side_effect=_capture):
        run_backfill(tmp_path, now_dt=datetime(2026, 5, 31, 12, 0, tzinfo=UTC))

    assert captured, "expected the default detector closure to be invoked"
    assert captured[0]["companion_name"] == tmp_path.name
    assert captured[0]["user_name"] == "Alex"
