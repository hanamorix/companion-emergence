"""Tests for the `nell embed backfill` CLI surface (F1 #259 increment 6 —
the manual one-go embedding-backfill CLI, spec §5b / S12).

Mirrors tests/unit/brain/test_cli_memory.py's fixture shape
(NELLBRAIN_HOME env var + a directly-built persona dir).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain import cli
from brain.memory import embeddings as embeddings_mod
from brain.memory.store import Memory, MemoryStore


def _make_persona_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    persona_dir = home / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))
    return persona_dir


def _seed_memories(persona_dir: Path, n: int) -> list[str]:
    store = MemoryStore(persona_dir / "memories.db")
    try:
        ids = []
        for i in range(n):
            ids.append(
                store.create(
                    Memory.create_new(
                        content=f"a synthetic backfill test memory, long enough, number {i}",
                        memory_type="conversation",
                        domain="us",
                    )
                )
            )
        return ids
    finally:
        store.close()


def _use_fake_provider(monkeypatch: pytest.MonkeyPatch, *, dim: int = 384):
    provider = embeddings_mod.FakeEmbeddingProvider(dim=dim)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    return provider


def test_embed_backfill_embeds_every_row_and_prints_summary(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`nell embed backfill` drains the whole backlog in one invocation and
    reports a clean summary with no failures."""
    persona_dir = _make_persona_dir(tmp_path, monkeypatch)
    ids = _seed_memories(persona_dir, 6)
    _use_fake_provider(monkeypatch)

    result = cli.main(["embed", "backfill", "--persona", "nell"])

    assert result == 0
    captured = capsys.readouterr()
    assert "embedding backfill for 'nell'" in captured.out
    assert "6 row(s) to embed" in captured.out
    assert "embedded" in captured.out  # the live ticker line
    assert "done: embedded=6 failed=0" in captured.out
    assert captured.err == ""

    store = MemoryStore(persona_dir / "memories.db")
    try:
        for memory_id in ids:
            row = store._conn.execute(
                "SELECT embedding FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            assert row["embedding"] is not None
    finally:
        store.close()


def test_embed_backfill_quiet_suppresses_ticker_keeps_summary(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`--quiet` drops the live ticker line but still prints the final
    summary — the summary is not optional, only the in-progress noise is."""
    persona_dir = _make_persona_dir(tmp_path, monkeypatch)
    _seed_memories(persona_dir, 4)
    _use_fake_provider(monkeypatch)

    result = cli.main(["embed", "backfill", "--persona", "nell", "--quiet"])

    assert result == 0
    captured = capsys.readouterr()
    assert "\rembedded" not in captured.out
    assert "done: embedded=4 failed=0" in captured.out


def test_embed_backfill_reports_failed_rows_with_nonzero_exit(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A permanently-failing row is reflected in the summary's `failed`
    count, a stderr note, and a non-zero exit — but the drain still
    terminates and finishes embedding every OTHER row."""
    persona_dir = _make_persona_dir(tmp_path, monkeypatch)
    _seed_memories(persona_dir, 5)

    class _FailsOnOne(embeddings_mod.EmbeddingProvider):
        def embed(self, text: str):  # noqa: ANN201
            if text.endswith("number 2"):
                raise RuntimeError("simulated PERMANENT provider failure")
            import numpy as np

            return np.ones(8, dtype="float32")

        def embedding_dim(self) -> int:
            return 8

        def model_id(self) -> str:
            return "fails-on-two-cli-test"

    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: _FailsOnOne())

    result = cli.main(["embed", "backfill", "--persona", "nell", "--quiet"])

    assert result == 1
    captured = capsys.readouterr()
    assert "done: embedded=4 failed=1" in captured.out
    assert "1 row(s) could not be embedded" in captured.err


def test_embed_backfill_empty_backlog_is_a_clean_no_op(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No un-embedded rows — the command reports there is nothing to do and
    exits cleanly without ever starting the drain loop."""
    persona_dir = _make_persona_dir(tmp_path, monkeypatch)
    (persona_dir / "memories.db").touch()
    store = MemoryStore(persona_dir / "memories.db")
    store.close()
    _use_fake_provider(monkeypatch)

    result = cli.main(["embed", "backfill", "--persona", "nell"])

    assert result == 0
    captured = capsys.readouterr()
    assert "already empty" in captured.out


def test_embed_backfill_missing_persona_returns_nonzero(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`nell embed backfill` against an uninstalled persona fails cleanly,
    mirroring `nell memory`'s missing-persona behavior."""
    home = tmp_path / "home"
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))

    result = cli.main(["embed", "backfill", "--persona", "nell"])

    assert result == 1
    captured = capsys.readouterr()
    assert "no persona directory" in captured.err.lower()
