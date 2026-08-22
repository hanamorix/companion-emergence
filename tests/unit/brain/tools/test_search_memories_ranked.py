"""search_memories on the ranked/snippet path, via real dispatch (P2 — C5/C23 + shape).

Exercises the ASSEMBLED tool path (through `dispatch`), not the impl in isolation:
  - C23: a disjoint multi-term query returns the UNION on the real tool.
  - C5:  exclude_ids drops the excluded id on the real tool.
  - shape: results are snippets (`snippet: true` + id), never full bodies.
"""

from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.relevance import SNIPPET_MAX_CHARS
from brain.memory.store import Memory, MemoryStore
from brain.tools.dispatch import dispatch


def _seed(store: MemoryStore, content: str) -> Memory:
    m = Memory.create_new(content=content, memory_type="event", domain="d")
    store.create(m)
    return m


def _ctx(tmp_path: Path) -> dict:
    return {
        "store": MemoryStore(":memory:"),
        "hebbian": HebbianMatrix(":memory:"),
        "persona_dir": tmp_path,
    }


def test_c23_dispatched_search_memories_ors_disjoint_terms(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    m1 = _seed(ctx["store"], "henryk drinks coffee")
    m2 = _seed(ctx["store"], "her preferences about tea")

    res = dispatch("search_memories", {"query": "henryk preferences"}, **ctx)
    ids = {m["id"] for m in res["memories"]}
    assert m1.id in ids and m2.id in ids, "disjoint multi-term query must union, not AND"


def test_dispatched_search_memories_returns_snippet_shape(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    long_body = "henryk " + ("and a great deal of extra detail follows here. " * 20)
    _seed(ctx["store"], long_body)

    res = dispatch("search_memories", {"query": "henryk"}, **ctx)
    assert res["memories"], "expected at least one hit"
    for m in res["memories"]:
        assert m.get("snippet") is True
        assert "id" in m
        assert len(m["content"]) <= SNIPPET_MAX_CHARS  # truncated, not the full body


def test_c5_dispatched_search_memories_honors_exclude_ids(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    m1 = _seed(ctx["store"], "henryk one detail")
    m2 = _seed(ctx["store"], "henryk two detail")

    res = dispatch(
        "search_memories",
        {"query": "henryk", "exclude_ids": [m1.id]},
        **ctx,
    )
    ids = {m["id"] for m in res["memories"]}
    assert m1.id not in ids
    assert m2.id in ids
