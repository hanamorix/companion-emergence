"""Tests for brain.initiate.adaptive — Bundle C adaptive-D layer."""
from __future__ import annotations

from pathlib import Path

from brain.initiate.adaptive import load_d_mode


def test_load_d_mode_missing_file_returns_stateless(tmp_path: Path):
    persona = tmp_path / "fresh"
    persona.mkdir()
    assert load_d_mode(persona) == "stateless"


def test_load_d_mode_stateless_explicit(tmp_path: Path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "d_mode.json").write_text('{"mode": "stateless"}')
    assert load_d_mode(persona) == "stateless"


def test_load_d_mode_adaptive(tmp_path: Path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "d_mode.json").write_text('{"mode": "adaptive"}')
    assert load_d_mode(persona) == "adaptive"


def test_load_d_mode_invalid_json_falls_back_to_stateless(tmp_path: Path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "d_mode.json").write_text("not json{")
    assert load_d_mode(persona) == "stateless"


def test_load_d_mode_unknown_value_falls_back_to_stateless(tmp_path: Path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "d_mode.json").write_text('{"mode": "experimental"}')
    assert load_d_mode(persona) == "stateless"


def test_load_d_mode_non_dict_json_falls_back_to_stateless(tmp_path: Path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "d_mode.json").write_text('"adaptive"')
    assert load_d_mode(persona) == "stateless"


def test_calibration_row_roundtrip():
    from brain.initiate.adaptive import CalibrationRow

    row = CalibrationRow(
        ts_decision="2026-05-13T10:00:00+00:00",
        ts_closed="2026-05-13T11:30:00+00:00",
        candidate_id="ic_abc",
        source="dream",
        decision="promote",
        confidence="high",
        model_tier="haiku",
        promoted_to_state="replied_explicit",
        filtered_recurred=None,
        reason_short="resonant memory return",
    )
    line = row.to_jsonl()
    parsed = CalibrationRow.from_jsonl(line)
    assert parsed == row


def test_calibration_row_filtered():
    from brain.initiate.adaptive import CalibrationRow

    row = CalibrationRow(
        ts_decision="2026-05-13T10:00:00+00:00",
        ts_closed="2026-05-15T10:00:00+00:00",
        candidate_id="ic_def",
        source="reflex_firing",
        decision="filter",
        confidence="high",
        model_tier="haiku",
        promoted_to_state=None,
        filtered_recurred=True,
        reason_short="reflex echo",
    )
    line = row.to_jsonl()
    parsed = CalibrationRow.from_jsonl(line)
    assert parsed.filtered_recurred is True
    assert parsed.promoted_to_state is None


def test_append_and_read_calibration_rows(tmp_path: Path):
    from brain.initiate.adaptive import (
        CalibrationRow,
        append_calibration_row,
        read_recent_calibration_rows,
    )

    persona = tmp_path / "p"
    for i in range(5):
        append_calibration_row(persona, CalibrationRow(
            ts_decision=f"2026-05-13T10:0{i}:00+00:00",
            ts_closed=f"2026-05-13T11:0{i}:00+00:00",
            candidate_id=f"ic_{i}",
            source="dream",
            decision="promote",
            confidence="high",
            model_tier="haiku",
            promoted_to_state="replied_explicit",
            filtered_recurred=None,
            reason_short=f"reason {i}",
        ))
    rows = list(read_recent_calibration_rows(persona, limit=3))
    assert len(rows) == 3
    # Newest first.
    assert rows[0].candidate_id == "ic_4"
    assert rows[2].candidate_id == "ic_2"
