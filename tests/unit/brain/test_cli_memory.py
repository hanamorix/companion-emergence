"""Tests for the `nell memory` CLI surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain import cli
from brain.memory.store import Memory, MemoryStore


def _make_persona_with_memories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    persona_dir = home / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))
    return persona_dir


def _store_memory(persona_dir: Path, memory: Memory) -> str:
    store = MemoryStore(persona_dir / "memories.db")
    try:
        return store.create(memory)
    finally:
        store.close()


def test_memory_list_prints_recent_active_memories(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`nell memory list` shows a safe compact view of active memories."""
    persona_dir = _make_persona_with_memories(tmp_path, monkeypatch)
    first_id = _store_memory(
        persona_dir,
        Memory.create_new(
            "Hana likes peach tea during late debugging sessions.",
            memory_type="conversation",
            domain="us",
            emotions={"warmth": 7.0},
            tags=["hana", "tea"],
            importance=6.5,
        ),
    )
    second_id = _store_memory(
        persona_dir,
        Memory.create_new(
            "Nell should keep release notes honest and boring.",
            memory_type="meta",
            domain="craft",
            emotions={},
            tags=[],
            importance=4.0,
        ),
    )

    result = cli.main(["memory", "list", "--persona", "nell", "--limit", "5"])

    assert result == 0
    captured = capsys.readouterr()
    assert "active memories for nell" in captured.out
    assert second_id[:8] in captured.out
    assert first_id[:8] in captured.out
    assert "meta" in captured.out
    assert "conversation" in captured.out
    assert "Nell should keep release notes honest" in captured.out
    assert "Hana likes peach tea" in captured.out


def test_memory_list_rejects_negative_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`nell memory list` rejects negative limits that could dump too much."""
    _make_persona_with_memories(tmp_path, monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["memory", "list", "--persona", "nell", "--limit", "-1"])

    assert excinfo.value.code == 2


def test_memory_search_requires_non_empty_query(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`nell memory search` rejects blank queries instead of scanning everything."""
    _make_persona_with_memories(tmp_path, monkeypatch)

    result = cli.main(["memory", "search", "", "--persona", "nell"])

    assert result == 2
    captured = capsys.readouterr()
    assert "query must not be empty" in captured.err.lower()


def test_memory_search_prints_matching_memories(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`nell memory search` uses the memory store text search."""
    persona_dir = _make_persona_with_memories(tmp_path, monkeypatch)
    matching_id = _store_memory(
        persona_dir,
        Memory.create_new(
            "The bridge status command must never print bearer tokens.",
            memory_type="security",
            domain="craft",
            importance=8.0,
        ),
    )
    _store_memory(
        persona_dir,
        Memory.create_new(
            "Peaches slept on the laptop again.",
            memory_type="conversation",
            domain="us",
        ),
    )

    result = cli.main(["memory", "search", "bearer", "--persona", "nell"])

    assert result == 0
    captured = capsys.readouterr()
    assert "memory search for nell" in captured.out
    assert matching_id[:8] in captured.out
    assert "bearer tokens" in captured.out
    assert "Peaches" not in captured.out


def test_memory_show_prints_full_memory_details(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`nell memory show` prints one complete memory record by id."""
    persona_dir = _make_persona_with_memories(tmp_path, monkeypatch)
    memory_id = _store_memory(
        persona_dir,
        Memory.create_new(
            "Hana trusts small surgical commits more than sprawling rewrites.",
            memory_type="preference",
            domain="craft",
            emotions={"trust": 8.0, "focus": 6.0},
            tags=["coding", "style"],
            importance=9.0,
            metadata={"source": "test"},
        ),
    )

    result = cli.main(["memory", "show", memory_id, "--persona", "nell"])

    assert result == 0
    captured = capsys.readouterr()
    assert f"id: {memory_id}" in captured.out
    assert "type: preference" in captured.out
    assert "domain: craft" in captured.out
    assert "importance: 9.00" in captured.out
    assert "tags: coding, style" in captured.out
    assert "emotions: focus=6.00, trust=8.00" in captured.out
    assert 'metadata: {"source": "test"}' in captured.out
    assert "Hana trusts small surgical commits" in captured.out


def test_memory_show_unknown_id_returns_nonzero(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`nell memory show` reports unknown ids cleanly."""
    persona_dir = _make_persona_with_memories(tmp_path, monkeypatch)
    store = MemoryStore(persona_dir / "memories.db")
    store.close()

    result = cli.main(["memory", "show", "missing-id", "--persona", "nell"])

    assert result == 1
    captured = capsys.readouterr()
    assert "unknown memory id" in captured.err.lower()


def test_memory_missing_persona_returns_nonzero(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`nell memory` does not create persona folders as a side effect."""
    home = tmp_path / "home"
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))

    result = cli.main(["memory", "list", "--persona", "nell"])

    assert result == 1
    assert not (home / "personas" / "nell").exists()
    captured = capsys.readouterr()
    assert "no persona directory" in captured.err.lower()
