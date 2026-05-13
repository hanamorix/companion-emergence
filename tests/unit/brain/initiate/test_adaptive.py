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


def test_build_calibration_block_with_mixed_outcomes(tmp_path):
    from brain.initiate.adaptive import (
        CalibrationRow,
        append_calibration_row,
        build_calibration_block,
    )

    persona = tmp_path / "p"
    # 3 promoted: 2 replied, 1 dismissed.
    for i, state in enumerate(["replied_explicit", "replied_explicit", "dismissed"]):
        append_calibration_row(persona, CalibrationRow(
            ts_decision=f"2026-05-13T10:0{i}:00+00:00",
            ts_closed=f"2026-05-13T11:0{i}:00+00:00",
            candidate_id=f"ic_p{i}", source="dream",
            decision="promote", confidence="high", model_tier="haiku",
            promoted_to_state=state, filtered_recurred=None,
            reason_short="x",
        ))
    # 2 filtered: 1 stayed silent, 1 recurred.
    for i, rec in enumerate([False, True]):
        append_calibration_row(persona, CalibrationRow(
            ts_decision=f"2026-05-13T10:1{i}:00+00:00",
            ts_closed=f"2026-05-15T10:1{i}:00+00:00",
            candidate_id=f"ic_f{i}", source="reflex_firing",
            decision="filter", confidence="high", model_tier="haiku",
            promoted_to_state=None, filtered_recurred=rec,
            reason_short="y",
        ))

    block = build_calibration_block(persona, user_name="Hana")
    assert "=== Your recent editorial track record ===" in block
    assert "2 reached replied_explicit  ← Hana engaged" in block
    assert "1 reached dismissed       ← Hana ↩'d" in block
    assert "1 stayed silent in draft (not re-emitted)" in block
    assert "1 re-emitted within 48h (you may have been too cautious)" in block


def test_build_calibration_block_empty_history(tmp_path):
    from brain.initiate.adaptive import build_calibration_block

    persona = tmp_path / "p"
    persona.mkdir()
    block = build_calibration_block(persona, user_name="Hana")
    assert "0 reached replied_explicit" in block
    assert "0 stayed silent in draft" in block


def test_detect_drift_below_bootstrap_returns_none(tmp_path):
    from datetime import UTC, datetime, timedelta

    from brain.initiate.adaptive import detect_drift
    from brain.initiate.audit import append_d_call_row
    from brain.initiate.d_call_schema import DCallRow

    persona = tmp_path / "p"
    now = datetime.now(UTC)
    # 50 d_call rows — below the 100 bootstrap floor.
    for i in range(50):
        append_d_call_row(persona, DCallRow(
            d_call_id=f"dc_{i}", ts=(now - timedelta(hours=i)).isoformat(),
            tick_id=f"t_{i}", model_tier_used="haiku",
            candidates_in=2, promoted_out=1, filtered_out=1,
            latency_ms=300, tokens_input=400, tokens_output=150,
        ))
    assert detect_drift(persona) is None


def test_detect_drift_stable_history_returns_none(tmp_path):
    from datetime import UTC, datetime, timedelta

    from brain.initiate.adaptive import detect_drift
    from brain.initiate.audit import append_d_call_row
    from brain.initiate.d_call_schema import DCallRow

    persona = tmp_path / "p"
    now = datetime.now(UTC)
    # 150 rows, all the same promote rate (flat → stdev ~ 0).
    for i in range(150):
        ts = (now - timedelta(hours=i)).isoformat()
        append_d_call_row(persona, DCallRow(
            d_call_id=f"dc_{i}", ts=ts,
            tick_id=f"t_{i}", model_tier_used="haiku",
            candidates_in=2, promoted_out=1, filtered_out=1,
            latency_ms=300, tokens_input=400, tokens_output=150,
        ))
    assert detect_drift(persona) is None


def test_detect_drift_clear_promote_rate_increase(tmp_path):
    """120 historical rows ~0.3 promote rate; 30 recent rows ~1.0 promote rate.
    Should emit a DriftAlert with delta_sigma > 2."""
    import random
    from datetime import UTC, datetime, timedelta

    from brain.initiate.adaptive import detect_drift
    from brain.initiate.audit import append_d_call_row
    from brain.initiate.d_call_schema import DCallRow

    persona = tmp_path / "p"
    now = datetime.now(UTC)
    rng = random.Random(42)

    # 120 older rows with ~0.3 promote rate, mild noise.
    for i in range(120):
        ts = (now - timedelta(days=60 + i // 4)).isoformat()
        promoted = 1 if rng.random() < 0.3 else 0
        append_d_call_row(persona, DCallRow(
            d_call_id=f"dc_h_{i}", ts=ts,
            tick_id=f"t_h_{i}", model_tier_used="haiku",
            candidates_in=2, promoted_out=promoted, filtered_out=2 - promoted,
            latency_ms=300, tokens_input=400, tokens_output=150,
        ))

    # 30 recent rows with 1.0 promote rate (clear drift).
    for i in range(30):
        ts = (now - timedelta(hours=i)).isoformat()
        append_d_call_row(persona, DCallRow(
            d_call_id=f"dc_r_{i}", ts=ts,
            tick_id=f"t_r_{i}", model_tier_used="haiku",
            candidates_in=2, promoted_out=2, filtered_out=0,
            latency_ms=300, tokens_input=400, tokens_output=150,
        ))

    alert = detect_drift(persona)
    assert alert is not None
    assert alert.delta_sigma > 2.0
    assert alert.current_rate > alert.historical_median
