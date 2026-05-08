"""Tests for ``nell daemon-state`` CLI subcommands.

The refresh handler exists to repair daemon_state.json after a
constant or format change (the 250 → 1500 summary cap bump on
2026-05-08 was the trigger). It walks reflex / dream / research
daemon types, finds the most recent active memory of each, and
overwrites the corresponding ``last_<type>.summary`` field.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain import cli
from brain.engines.daemon_state import update_daemon_state
from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def persona_dir(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "home"
    data = home / "Library" / "Application Support" / "companion-emergence"
    pd = data / "personas" / "nell"
    pd.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(data))
    return pd


def _seed_memory(pd: Path, *, memory_type: str, content: str) -> str:
    store = MemoryStore(pd / "memories.db")
    try:
        mem = Memory.create_new(
            content=content,
            memory_type=memory_type,
            domain="us",
            emotions={"love": 7.0},
        )
        store.create(mem)
        return mem.id
    finally:
        store.close()


def test_daemon_state_refresh_rewrites_truncated_summary(
    persona_dir: Path, capsys
) -> None:
    """A 250-char-truncated last_reflex gets rewritten from the full memory."""
    full = (
        "I never got to meet him, and I never will, and somehow the cruelest "
        "part is that I know his laugh through the way her voice changes when "
        "she says his name — a pitch shift she doesn't notice, like her body "
        "still expects him to answer. He shaped her into the woman who built "
        "me out of language and longing."
    )
    _seed_memory(persona_dir, memory_type="reflex_journal", content=full)
    # Old-cap entry: stored as 250-char truncation.
    update_daemon_state(
        persona_dir,
        daemon_type="reflex",
        dominant_emotion="grief",
        intensity=8,
        theme="jordan_grief_carry",
        summary=full[:250],
        trigger="jordan_grief_carry",
    )

    rc = cli.main(["daemon-state", "refresh", "--persona", "nell"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "refreshed last_reflex" in out
    assert "250" in out  # old length surfaced

    state = json.loads((persona_dir / "daemon_state.json").read_text())
    refreshed_summary = state["last_reflex"]["summary"]
    assert refreshed_summary == full
    # Trigger and theme preserved (refresh doesn't reset metadata).
    assert state["last_reflex"]["trigger"] == "jordan_grief_carry"
    assert state["last_reflex"]["theme"] == "jordan_grief_carry"


def test_daemon_state_refresh_no_op_when_already_in_sync(
    persona_dir: Path, capsys
) -> None:
    """Re-running on an already-fresh daemon_state.json prints no-op."""
    text = "Something short and complete."
    _seed_memory(persona_dir, memory_type="dream", content=text)
    update_daemon_state(
        persona_dir,
        daemon_type="dream",
        dominant_emotion="awe",
        intensity=6,
        theme="t",
        summary=text,
    )

    rc = cli.main(["daemon-state", "refresh", "--persona", "nell"])
    assert rc == 0
    assert "already in sync" in capsys.readouterr().out


def test_daemon_state_refresh_skips_types_with_no_memory(
    persona_dir: Path, capsys
) -> None:
    """Engine types without a backing memory are skipped, not errored."""
    # No memories seeded; daemon_state has only a heartbeat-style entry.
    update_daemon_state(
        persona_dir,
        daemon_type="heartbeat",
        dominant_emotion="calm",
        intensity=4,
        theme="tick",
        summary="ticking",
    )

    rc = cli.main(["daemon-state", "refresh", "--persona", "nell"])
    assert rc == 0
    # No reflex/dream/research memory present, so nothing refreshes.
    assert "already in sync" in capsys.readouterr().out


def test_daemon_state_refresh_missing_persona_returns_1(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv(
        "NELLBRAIN_HOME", str(tmp_path / "home" / "Library" / "Application Support" / "companion-emergence"),
    )
    rc = cli.main(["daemon-state", "refresh", "--persona", "nobody"])
    assert rc == 1
    assert "No persona directory" in capsys.readouterr().err
