"""Tests for `nell initiate d-stats` CLI subcommand."""
from __future__ import annotations

import argparse

import pytest

pytest.importorskip("brain.initiate")

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.cli import _initiate_d_stats_handler
from brain.initiate.audit import append_d_call_row
from brain.initiate.d_call_schema import DCallRow


def _args(persona_dir: Path, **kw) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "persona": persona_dir.name,
        "window": "24h",
    }
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_d_stats_empty_persona_returns_zero_counts(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _initiate_d_stats_handler(_args(persona_dir))
    assert rc == 0
    out = capsys.readouterr().out
    assert "candidates_in=0" in out or "no d calls" in out.lower()


def test_d_stats_aggregates_recent_calls(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    persona_dir = tmp_path / "p"
    now = datetime.now(UTC)
    for i in range(3):
        append_d_call_row(
            persona_dir,
            DCallRow(
                d_call_id=f"dc_{i}",
                ts=(now - timedelta(hours=i * 2)).isoformat(),
                tick_id=f"t_{i}",
                model_tier_used="haiku",
                candidates_in=2,
                promoted_out=1,
                filtered_out=1,
                latency_ms=300,
                tokens_input=400,
                tokens_output=150,
            ),
        )
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _initiate_d_stats_handler(_args(persona_dir, window="24h"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "candidates_in=6" in out
    assert "promoted_out=3" in out
    assert "filtered_out=3" in out
