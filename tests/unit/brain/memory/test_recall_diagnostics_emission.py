"""Name-recall fix R7 (spec section 6, criterion 10; S15, S30, S48, S53, S56, S58, S59;
plan P-22, P-34): the diagnostics record is written at every call site.

Criteria: R7-E1 (exactly one record per call: passive `run_semantic_recall`, the
search tool's semantic and lexical requests, through the real entries), R7-E2 (the
row content: width from the paragraph outcome, two rows for two scales), R7-E3
(fail-soft, no store access). The tick prune is in
`tests/unit/brain/bridge/test_calibration_tick_diagnostics_prune.py`.

All offline: the R3/R6 scripted embedders and fake rerankers, an in-tmp store.
Synthetic data only.
"""

from __future__ import annotations

import ast
import inspect
import json
import sqlite3
from pathlib import Path

import pytest

from brain import dev_constants, tunables
from brain.chat.prompt import _build_recall_block
from brain.memory import recall_diagnostics as rd
from brain.memory import reranker as reranker_mod
from brain.memory import semantic_recall as sr
from brain.memory.hebbian import HebbianMatrix
from brain.memory.semantic_recall import (
    COSINE_PATH,
    RERANKED_PATH,
    GatedRanking,
    ParagraphOutcome,
    ParagraphSearch,
    emit_recall_diagnostics,
    paragraph_diagnostics,
    run_semantic_recall,
    search_paragraphs,
)
from brain.memory.store import (
    CALIBRATION_SCORE_SCALE,
    COSINE_SCORE_SCALE,
    MemoryStore,
)
from brain.tools.dispatch import dispatch
from brain.tools.impls.search_memories import search_memories
from tests.unit.brain.memory.test_hybrid_family_tail import _first_rerank_case
from tests.unit.brain.memory.test_monologue_last import (
    _QUERY,
    _cosine_floor,
    _install_reranker,
    _no_reranker,
    _Recording,
    _rerank_floor,
    _seed,
    _warm,
)
from tests.unit.brain.memory.test_per_paragraph_recall import (
    _P1,
    _P2,
    _P3,
    _REST,
    _THREE,
    _AxisEmbedder,
    _budget,
    _Clock,
    _install,
    _mem,
    _Reranker,
    _two_query_store,
)

_LOG = dev_constants.RECALL_DIAGNOSTICS_LOG_FILENAME


def _records(persona: Path) -> list[dict]:
    path = persona / _LOG
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _cal_count(store: MemoryStore) -> int:
    return store._conn.execute("SELECT COUNT(*) FROM calibration_log").fetchone()[0]  # noqa: SLF001


def _ctx(persona: Path, store: MemoryStore) -> dict:
    return {"store": store, "hebbian": HebbianMatrix(":memory:"), "persona_dir": persona}


def _row(path, width, pass_mark, scale):
    return {"path": path, "width": width, "pass_mark": pass_mark, "scale": scale}


# ---------------------------------------------------------------------------
# R7-E2: the row builder (hand-built outcomes)
# ---------------------------------------------------------------------------


def _outcome(index, gated, reranked_width, *, best=0.9):
    return ParagraphOutcome(
        index=index,
        query=f"q{index}",
        gated=gated,
        candidate_count=10,
        best_cosine=best,
        reranked_width=reranked_width,
    )


def _ranking(path, scale, pass_mark, *, real_width=0, tail=None):
    return GatedRanking(
        path=path, scale=scale, ranked=[], pass_mark=pass_mark, tail=tail, real_width=real_width
    )


def _as_dicts(rows):
    return [
        {"path": r.path, "width": r.width, "pass_mark": r.pass_mark, "scale": r.scale}
        for r in rows
    ]


def test_a_reranked_row_carries_the_outcome_width_its_floor_and_scale() -> None:
    gated = _ranking(RERANKED_PATH, CALIBRATION_SCORE_SCALE, 1.5, real_width=7)
    rows = paragraph_diagnostics([_outcome(0, gated, 7)])
    assert _as_dicts(rows) == [_row("reranked", 7, 1.5, CALIBRATION_SCORE_SCALE)]


def test_a_cosine_path_row_has_width_zero_the_cosine_floor_and_scale() -> None:
    gated = _ranking(COSINE_PATH, COSINE_SCORE_SCALE, 0.61)
    rows = paragraph_diagnostics([_outcome(0, gated, 0)])
    assert _as_dicts(rows) == [_row("cosine", 0, 0.61, COSINE_SCORE_SCALE)]


def test_the_width_is_the_outcomes_reranked_width_never_the_rankings_real_width() -> None:
    """R7-E2 (f): a ranking whose `real_width` reads 0 while the outcome says the
    reranker scored 7 shows 7."""
    gated = _ranking(RERANKED_PATH, CALIBRATION_SCORE_SCALE, 1.0, real_width=0)
    assert paragraph_diagnostics([_outcome(0, gated, 7)])[0].width == 7
    cosine = _ranking(COSINE_PATH, COSINE_SCORE_SCALE, 0.5, real_width=0)
    assert paragraph_diagnostics([_outcome(0, cosine, 6)])[0].width == 6


def test_a_paragraph_with_no_gated_ranking_writes_one_row_with_no_pass_mark_and_scale() -> None:
    """R7-E2 (c): the rerank scored (width 7) but neither floor existed: the
    width is not lost (S58); with no rerank at all the row is the cosine path."""
    rows = paragraph_diagnostics([_outcome(0, None, 7), _outcome(1, None, 0)])
    assert _as_dicts(rows) == [_row("reranked", 7, None, None), _row("cosine", 0, None, None)]


def test_a_reranked_paragraph_with_a_cosine_tail_writes_two_rows_and_the_tail_counts_no_width() -> (
    None
):
    tail = _ranking(COSINE_PATH, COSINE_SCORE_SCALE, 0.7)
    gated = _ranking(RERANKED_PATH, CALIBRATION_SCORE_SCALE, 1.5, real_width=5, tail=tail)
    rows = paragraph_diagnostics([_outcome(0, gated, 5), _outcome(1, None, 0)])
    assert _as_dicts(rows) == [
        _row("reranked", 5, 1.5, CALIBRATION_SCORE_SCALE),
        _row("cosine", 0, 0.7, COSINE_SCORE_SCALE),
        _row("cosine", 0, None, None),
    ]
    assert sum(r.width for r in rows) == 5, "no real candidate counted twice"


# ---------------------------------------------------------------------------
# R7-E1 / R7-E2 end to end: passive recall
# ---------------------------------------------------------------------------


def test_passive_reranked_paragraph_with_a_tail_writes_one_record_with_two_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, _genuine, _family, _rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)
    _budget(monkeypatch, 4.0)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    (record,) = _records(tmp_path)
    assert record["source"] == "passive" and "mode" not in record
    assert record["paragraph_count"] == 1 and record["whole_message_fallback"] is False
    assert record["total_width"] == 5, "the first rerank of a process is 5 real candidates (S24)"
    assert record["budget"] == pytest.approx(4.0), "seconds, the per-message budget in force"
    assert record["paragraphs"] == [
        _row("reranked", 5, pytest.approx(1.0), CALIBRATION_SCORE_SCALE),
        _row("cosine", 0, pytest.approx(0.5), COSINE_SCORE_SCALE),
    ]
    assert result.total_width == record["total_width"] and result.budget == record["budget"]
    assert set(record) == {
        "ts",
        "source",
        "paragraph_count",
        "whole_message_fallback",
        "total_width",
        "budget",
        "paragraphs",
    }


def test_passive_cosine_path_writes_one_record_with_a_cosine_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    _seed(store, monkeypatch, [0.9, 0.8, 0.7], [])
    _cosine_floor(store, 0.5)
    _no_reranker(monkeypatch)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == COSINE_PATH
    (record,) = _records(tmp_path)
    assert record["total_width"] == 0 and record["paragraph_count"] == 1
    assert record["paragraphs"] == [_row("cosine", 0, pytest.approx(0.5), COSINE_SCORE_SCALE)]


def test_a_rerank_that_scored_but_had_no_floor_keeps_its_width_in_the_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R7-E2 (c), end to end: the rerank floor is missing, so the paragraph takes the
    cosine path; the rerank still used 5 of the 50, and the cosine row says so."""
    store = MemoryStore(tmp_path / "memories.db")
    genuine, _family = _seed(store, monkeypatch, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [])
    _cosine_floor(store, 0.1)
    _install_reranker(monkeypatch, _Recording({g.content: 5.0 for g in genuine}))

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == COSINE_PATH
    (record,) = _records(tmp_path)
    assert record["total_width"] == 5
    assert record["paragraphs"] == [_row("cosine", 5, pytest.approx(0.1), COSINE_SCORE_SCALE)]


def test_a_rerank_that_scored_with_no_floor_of_either_kind_still_records_its_width(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, _family = _seed(store, monkeypatch, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [])
    monkeypatch.setattr(MemoryStore, "get_cosine_floor", lambda self, model_id: None)
    _install_reranker(monkeypatch, _Recording({g.content: 5.0 for g in genuine}))

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is None, "no gate possible: inconclusive"
    (record,) = _records(tmp_path)
    assert record["total_width"] == 5 and record["paragraph_count"] == 1
    assert record["paragraphs"] == [_row("reranked", 5, None, None)]


def test_a_call_with_no_pool_still_writes_one_record_with_zero_paragraphs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, _AxisEmbedder({_QUERY: 0}))

    assert run_semantic_recall(store, tmp_path, _QUERY) is None

    (record,) = _records(tmp_path)
    assert record["paragraph_count"] == 0 and record["total_width"] == 0
    assert record["budget"] is None and record["paragraphs"] == []
    assert record["whole_message_fallback"] is False and record["source"] == "passive"


def test_a_search_that_raises_still_writes_the_zero_paragraph_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated search failure")

    monkeypatch.setattr(sr, "search_paragraphs", _boom)

    assert run_semantic_recall(store, tmp_path, _QUERY) is None

    (record,) = _records(tmp_path)
    assert record["paragraph_count"] == 0 and record["paragraphs"] == []


def test_a_failure_after_the_search_still_writes_the_searchs_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The record is written in a `finally`: a raise in the assembly after the
    per-paragraph search keeps what the search measured."""
    store = MemoryStore(tmp_path / "memories.db")
    _seed(store, monkeypatch, [0.9, 0.8], [])
    _cosine_floor(store, 0.5)
    _no_reranker(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated assembly failure")

    monkeypatch.setattr(sr, "assemble_with_guarantees", _boom)

    assert run_semantic_recall(store, tmp_path, _QUERY) is None

    (record,) = _records(tmp_path)
    assert record["paragraph_count"] == 1
    assert record["paragraphs"] == [_row("cosine", 0, pytest.approx(0.5), COSINE_SCORE_SCALE)]


def test_the_whole_message_fallback_flag_is_true_only_for_the_one_query_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Three paragraphs that do not fit the budget (warm series): one whole-message
    query, flag true."""
    _budget(monkeypatch, 4.0)
    embedder = _AxisEmbedder({_P1: 0, _P2: 1, _P3: 2})
    store = MemoryStore(tmp_path / "memories.db")
    sr._embed_cost.record(store, "r3-test-embedder", 100, 0.0)
    sr._scan_time.record(store, "", 1, 3.0)
    _install(monkeypatch, embedder)
    _mem(store, "a memory of the whole message", {383: 0.9})
    _cosine_floor(store, 0.5)
    _no_reranker(monkeypatch)

    run_semantic_recall(store, tmp_path, _THREE)

    (record,) = _records(tmp_path)
    assert record["whole_message_fallback"] is True and record["paragraph_count"] == 1


def test_a_first_message_scored_as_two_paragraphs_records_two_and_no_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R7-E2 (e), S63, S199: no measurement yet; the first paragraph is embedded and
    timed alone, neither every paragraph nor paragraphs 1 and 3 fit, so the rest is
    ONE query cut to what fits and the message is scored as those two paragraphs:
    `paragraph_count == 2`, not the whole-message fallback."""
    clock = _Clock()
    monkeypatch.setattr(sr, "_clock", clock)
    _budget(monkeypatch, 2.9)
    store, embedder, _rec, _first, _rest = _two_query_store(monkeypatch, tmp_path, clock, 1.5)

    result = run_semantic_recall(store, tmp_path, _THREE)

    assert result is not None and embedder.batches == [[_P1], ["beta heron"]]
    (record,) = _records(tmp_path)
    assert record["paragraph_count"] == 2 and record["whole_message_fallback"] is False
    assert len(record["paragraphs"]) >= 2
