"""Cost-series increment (name-recall fix S146, S164): nothing writes a timing
sample on the reply path (criterion CS17a), a `--no-bridge` turn is sized from
samples another connection saved (CS6d), and every test double standing in for
`rerank_for_recall` accepts the store keyword (CS16).

Synthetic data only.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from brain.chat.prompt import _build_recall_block
from brain.memory import cost_series, keyword_budget, relevance
from brain.memory.semantic_recall import run_semantic_recall
from brain.memory.store import MemoryStore
from tests.unit.brain.chat.test_keyword_budget_recall import Clock, make_world, render, wire
from tests.unit.brain.memory.test_paragraph_embed_cost import (
    _clocked,
    _CostEmbedder,
    _set_budget,
    _store,
)

_REPO = Path(__file__).resolve().parents[3]


def _cost_rows(path: Path) -> tuple[int, int]:
    con = sqlite3.connect(path)
    try:
        return (
            con.execute("SELECT COUNT(*) FROM cost_series_samples").fetchone()[0],
            con.execute("SELECT COUNT(*) FROM cost_series_carried").fetchone()[0],
        )
    finally:
        con.close()


def _forbid_cost_writes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def spy(self, *a, **k):
        calls.append("record_cost_samples")
        raise AssertionError("a cost sample was written on the reply path")

    monkeypatch.setattr(MemoryStore, "record_cost_samples", spy)

    def verdict_spy(self, *a, **k):
        calls.append("verdict")
        raise AssertionError("a rerank verdict was written or deleted on the reply path (S179)")

    monkeypatch.setattr(MemoryStore, "save_cost_verdict", verdict_spy)
    monkeypatch.setattr(MemoryStore, "clear_cost_verdict", verdict_spy)
    return calls


def _buffered_series() -> set[str]:
    with cost_series._pending_lock:  # noqa: SLF001
        return {s.series for s in cost_series._pending}  # noqa: SLF001


# ---------------------------------------------------------------------------
# CS17a: the reply path buffers and never writes.
# ---------------------------------------------------------------------------


def test_cs17a_the_semantic_step_buffers_its_samples_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 4.0)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    calls = _forbid_cost_writes(monkeypatch)
    before = _cost_rows(store.db_path)
    result = run_semantic_recall(store, tmp_path, "alpha one words\nbeta two words\ngamma three words")
    assert result is not None
    assert calls == [] and _cost_rows(store.db_path) == before == (0, 0)
    assert {"recall setup", "paragraph scan", "paragraph embed"} <= _buffered_series()
    store._conn.commit()  # noqa: SLF001


def test_cs17a_the_keyword_and_name_steps_of_passive_recall_write_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    w = make_world(tmp_path, n=40)
    calls = _forbid_cost_writes(monkeypatch)
    clock = Clock()
    monkeypatch.setattr(keyword_budget, "_clock", clock)
    wire(monkeypatch, clock, t_sem=0.0)
    render(w)
    render(w)  # the second search is the first one recorded (S134)
    assert calls == []
    assert "keyword extra words" in _buffered_series()


def test_cs17a_the_search_tool_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from brain.memory import known_names as kn
    from brain.tools.dispatch import dispatch
    from tests.unit.brain.tools.test_search_memories_keyword_merge import _ctx
    from tests.unit.brain.tools.test_search_memories_keyword_merge import _mem as tool_mem

    ctx = _ctx(tmp_path)
    tool_mem(ctx["store"], "Quill came by the harbour at dusk")
    assert kn.admit_names(ctx["persona_dir"], ["Quill"], "tool")
    calls = _forbid_cost_writes(monkeypatch)
    res = dispatch("search_memories", {"query": "Quill harbour", "mode": "lexical", "limit": 5}, **ctx)
    assert res["memories"]
    assert calls == [] and _cost_rows(ctx["store"].db_path) == (0, 0)


# ---------------------------------------------------------------------------
# CS6d: `--no-bridge`: samples another connection saved size this process's extras.
# ---------------------------------------------------------------------------


def test_cs6d_a_fresh_process_sizes_the_extras_from_samples_flushed_by_another_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    b = keyword_budget.recall_budget_seconds()
    w = make_world(tmp_path / "world", n=40)
    path = tmp_path / "persona" / "memories.db"
    path.parent.mkdir()
    other = MemoryStore(path, integrity_check=False)
    try:
        rate = b / 16
        for x in (4, 8):  # the bridge's saved keyword samples: a word costs B/16
            other.record_cost_samples(
                "keyword extra words", "test-machine", "", [(x, x * rate, 1_000.0)], now=1_000.0, window_seconds=1.0e12
            )
    finally:
        other.close()
    mine = MemoryStore(path, integrity_check=False)  # a new process: empty buffer
    try:
        searched: list[list[str]] = []
        real = relevance.rank_memories

        def spy(store, heb, query, **k):
            if isinstance(query, list):
                searched.append(list(query))
            return real(store, heb, query, **k)

        monkeypatch.setattr("brain.chat.prompt.run_semantic_recall", lambda *a, **k: None)
        monkeypatch.setattr(relevance, "rank_memories", spy)
        monkeypatch.setattr(keyword_budget, "_first_sample_spent", True)  # this process already discarded its first
        _build_recall_block(mine, w.paste, persona_dir=w.persona, limit=8)
        assert [len(q) for q in searched] == [16], "the saved fit (a word costs B/16: 16 fit B) sizes it, not the free seed"
    finally:
        mine.close()


# ---------------------------------------------------------------------------
# CS16: test doubles cannot pass for the wrong reason.
# ---------------------------------------------------------------------------


def _doubles_of_rerank_for_recall(path: Path) -> list[tuple[int, str]]:
    """Every replacement of `rerank_for_recall` set in a test file: (line, why not safe)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    bad: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 3:
            continue
        name = node.args[1]
        if not (isinstance(name, ast.Constant) and name.value == "rerank_for_recall"):
            continue
        repl = node.args[2]
        if isinstance(repl, ast.Name) and repl.id in defs:
            repl = defs[repl.id]
        if isinstance(repl, (ast.Lambda, ast.FunctionDef)):
            args = repl.args
            accepts = args.kwarg is not None or any(a.arg == "store" for a in args.kwonlyargs)
            if not accepts:
                bad.append((node.lineno, "does not accept store/**kwargs"))
        else:
            bad.append((node.lineno, "replacement is not a lambda or a local function"))
    return bad


def test_cs16_every_double_of_rerank_for_recall_accepts_the_store_keyword() -> None:
    found = 0
    for path in sorted((_REPO / "tests").rglob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        if "rerank_for_recall" not in source:
            continue  # a double names it as a string constant: skip the parse of every other test file
        bad = _doubles_of_rerank_for_recall(path)
        assert not bad, (path, bad)
        if '"rerank_for_recall"' in source:
            found += 1
    assert found >= 3, "the scan found the doubles it is meant to check"


def test_cs16_the_scan_rejects_a_double_without_the_keyword(tmp_path: Path) -> None:
    bad = tmp_path / "test_bad.py"
    bad.write_text(
        "def t(monkeypatch, m):\n"
        "    monkeypatch.setattr(m, 'rerank_for_recall', lambda provider, query, docs: None)\n"
        "    monkeypatch.setattr(m, 'rerank_for_recall', lambda *a, **kw: None)\n",
        encoding="utf-8",
    )
    assert [why for _, why in _doubles_of_rerank_for_recall(bad)] == ["does not accept store/**kwargs"]
