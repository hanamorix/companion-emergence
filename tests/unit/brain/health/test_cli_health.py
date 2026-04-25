# tests/unit/brain/health/test_cli_health.py
"""Tests for `nell health show/check/acknowledge` CLI subcommands — Task 13."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_persona(personas_root: Path, name: str = "testpersona") -> Path:
    persona_dir = personas_root / name
    persona_dir.mkdir(parents=True)
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    MemoryStore(db_path=persona_dir / "memories.db").close()
    HebbianMatrix(db_path=persona_dir / "hebbian.db").close()
    return persona_dir


def _write_audit_log(persona_dir: Path, entries: list[dict]) -> None:
    p = persona_dir / "heartbeats.log.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def _anomaly_entry(
    file: str,
    when: datetime,
    action: str = "restored_from_bak1",
    kind: str = "json_parse_error",
    quarantine_path: str | None = None,
    likely_cause: str = "unknown",
) -> dict:
    return {
        "timestamp": when.isoformat().replace("+00:00", "Z"),
        "anomalies": [
            {
                "timestamp": when.isoformat().replace("+00:00", "Z"),
                "file": file,
                "kind": kind,
                "action": action,
                "quarantine_path": quarantine_path,
                "likely_cause": likely_cause,
                "detail": "",
            }
        ],
    }


# ---------------------------------------------------------------------------
# nell health show
# ---------------------------------------------------------------------------


def test_cli_health_show_clean(monkeypatch, tmp_path: Path, capsys):
    """Healthy persona → 'Pending alarms: 0' + 'Recent self-treatments: 0'."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    _setup_persona(tmp_path / "personas")

    rc = main(["health", "show", "--persona", "testpersona"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Pending alarms: 0" in out
    assert "Recent self-treatments" in out
    assert "0" in out


def test_cli_health_show_with_treatment_history(monkeypatch, tmp_path: Path, capsys):
    """Recent self-treatments listed; alarms section present (count 0)."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")

    # Seed a self-treatment (restored_from_bak1 — healed, not an alarm)
    when = datetime.now(UTC) - timedelta(hours=2)
    _write_audit_log(persona_dir, [_anomaly_entry("user_preferences.json", when)])

    rc = main(["health", "show", "--persona", "testpersona"])

    assert rc == 0
    out = capsys.readouterr().out
    # The treatment was healed, so should appear in recent treatments
    assert "Recent self-treatments" in out
    assert "user_preferences.json" in out
    # Not an alarm (restored_from_bak1 on a non-identity file)
    assert "Pending alarms: 0" in out


# ---------------------------------------------------------------------------
# nell health check
# ---------------------------------------------------------------------------


def test_cli_health_check_clean_persona(monkeypatch, tmp_path: Path, capsys):
    """All ✅ for a clean persona; exit 0."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    _setup_persona(tmp_path / "personas")

    rc = main(["health", "check", "--persona", "testpersona"])

    assert rc == 0
    out = capsys.readouterr().out
    # No anomalies → summary says healthy
    assert "healthy" in out.lower() or "0 file(s)" in out or "healed" in out


def test_cli_health_check_corrupt_file_self_heals(monkeypatch, tmp_path: Path, capsys):
    """Corrupt file shows ⚠️; exit 0 (self-treated, not alarming)."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")

    # Corrupt a healable file (non-identity, so healed not alarmed)
    (persona_dir / "user_preferences.json").write_text("{not json", encoding="utf-8")

    rc = main(["health", "check", "--persona", "testpersona"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "⚠️" in out or "WARNING" in out or "user_preferences.json" in out


def test_cli_health_check_unhealable_returns_exit_2(monkeypatch, tmp_path: Path):
    """SQLite corruption → ❌ + exit 2."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")

    # Corrupt the memories.db to force sqlite_integrity_fail
    (persona_dir / "memories.db").write_bytes(b"not sqlite data at all")

    rc = main(["health", "check", "--persona", "testpersona"])

    assert rc == 2


# ---------------------------------------------------------------------------
# nell health acknowledge
# ---------------------------------------------------------------------------


def test_cli_health_acknowledge_all_clears_alarms(monkeypatch, tmp_path: Path):
    """After acknowledge --all, compute_pending_alarms returns []."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")

    # Seed an alarm-level anomaly (reset_to_default on an identity file)
    when = datetime.now(UTC) - timedelta(hours=1)
    _write_audit_log(
        persona_dir,
        [_anomaly_entry("emotion_vocabulary.json", when, action="reset_to_default")],
    )

    # Verify alarm exists before acknowledge
    from brain.health.alarm import compute_pending_alarms

    assert len(compute_pending_alarms(persona_dir)) == 1

    rc = main(["health", "acknowledge", "--persona", "testpersona", "--all"])
    assert rc == 0

    # Alarm is now cleared
    assert compute_pending_alarms(persona_dir) == []


def test_cli_health_no_destructive_actions(monkeypatch, tmp_path: Path):
    """`nell health restore`, `nell health add`, `nell health delete` raise SystemExit."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))

    for forbidden in ("restore", "add", "delete"):
        with pytest.raises(SystemExit):
            main(["health", forbidden, "--persona", "testpersona"])
