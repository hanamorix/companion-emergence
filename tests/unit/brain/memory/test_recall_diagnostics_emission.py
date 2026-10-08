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
    """R7-E2 (e), S63: no measurement yet; the first paragraph is embedded and timed
    alone, 3 x its time does not fit, so the rest is ONE query and the message is
    scored as those two paragraphs: `paragraph_count == 2`, not the whole-message
    fallback."""
    clock = _Clock()
    monkeypatch.setattr(sr, "_clock", clock)
    _budget(monkeypatch, 4.0)
    store, embedder, _rec, _first, _rest = _two_query_store(monkeypatch, tmp_path, clock, 1.5)

    result = run_semantic_recall(store, tmp_path, _THREE)

    assert result is not None and embedder.batches == [[_P1], [_REST]]
    (record,) = _records(tmp_path)
    assert record["paragraph_count"] == 2 and record["whole_message_fallback"] is False
    assert [row["path"] for row in record["paragraphs"]] == ["reranked", "reranked"]
    assert sum(row["width"] for row in record["paragraphs"]) == record["total_width"] > 0


def test_total_width_never_exceeds_fifty_and_equals_the_sum_of_the_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """8 paragraphs of 8 candidates, all of which fit: 64 > 50, so six paragraphs
    rerank 8 each (48) and two take the cosine path (S58)."""
    bests = [0.61, 0.95, 0.70, 0.88, 0.52, 0.79, 0.66, 0.90]
    texts = [f"paragraph {chr(97 + i)} words" for i in range(8)]
    embedder = _AxisEmbedder({t: i for i, t in enumerate(texts)})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    contents = []
    for i, best in enumerate(bests):
        for j in range(8):
            contents.append(_mem(store, f"r7 para {i} memory {j}", {i: best - 0.001 * j}).content)
    _install(monkeypatch, embedder, _Reranker(dict.fromkeys(contents, 5.0)))
    _rerank_floor(store, 1.0)
    _cosine_floor(store, 0.1)
    _warm(store)

    run_semantic_recall(store, tmp_path, "\n".join(texts))

    (record,) = _records(tmp_path)
    assert record["paragraph_count"] == 8
    assert record["total_width"] == 48 <= 50
    assert sum(row["width"] for row in record["paragraphs"]) == record["total_width"]
    assert sorted(row["width"] for row in record["paragraphs"]) == [0, 0, 8, 8, 8, 8, 8, 8]


def test_a_passive_reranked_paragraph_still_writes_its_calibration_row_with_the_paragraph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R7-E1 (g), second clause: the diagnostics record is separate from, and does
    not replace, the passive calibration row."""
    store, _g, _f, _rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)
    before = _cal_count(store)

    run_semantic_recall(store, tmp_path, _QUERY)

    queries = [
        row[0]
        for row in store._conn.execute("SELECT query FROM calibration_log").fetchall()  # noqa: SLF001
    ]
    assert _cal_count(store) == before + 1 and queries == [_QUERY]


# ---------------------------------------------------------------------------
# R7-E1: the search tool
# ---------------------------------------------------------------------------


def test_tool_semantic_request_writes_one_tool_record_and_no_calibration_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, _g, _f, _rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)
    before = _cal_count(store)

    out = dispatch("search_memories", {"query": _QUERY}, **_ctx(tmp_path, store))

    assert out["mode"] == "semantic"
    (record,) = _records(tmp_path)
    assert record["source"] == "tool" and record["mode"] == "semantic"
    assert record["paragraph_count"] == 1 and record["total_width"] == 5
    assert [r["path"] for r in record["paragraphs"]] == ["reranked", "cosine"]
    assert _cal_count(store) == before, "S56: the tool writes no calibration row"


def test_tool_semantic_request_that_falls_back_keeps_its_one_semantic_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The record names the stage that ran; the call reports the mode it ended in.
    One record, not two."""
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, _AxisEmbedder({_QUERY: 0}))  # no vectors: the semantic stage has no pool

    out = dispatch("search_memories", {"query": _QUERY}, **_ctx(tmp_path, store))

    assert out["mode"] == "lexical"
    (record,) = _records(tmp_path)
    assert record["source"] == "tool" and record["mode"] == "semantic"
    assert record["paragraph_count"] == 0 and record["paragraphs"] == []


@pytest.mark.parametrize("requested", ["lexical", "garbage-mode-value"])
def test_tool_lexical_request_writes_one_record_with_no_paragraphs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, requested: str
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    before = _cal_count(store)

    out = dispatch("search_memories", {"query": _QUERY, "mode": requested}, **_ctx(tmp_path, store))

    assert out["mode"] == "lexical"
    (record,) = _records(tmp_path)
    assert record["source"] == "tool" and record["mode"] == "lexical"
    assert record["paragraph_count"] == 0 and record["total_width"] == 0
    assert record["paragraphs"] == [] and record["budget"] is None
    assert _cal_count(store) == before


# ---------------------------------------------------------------------------
# R7-E1 (h): the real entries; (f), (i): files and privacy
# ---------------------------------------------------------------------------


def test_the_recall_block_writes_exactly_one_passive_record_for_a_message_with_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, _g, _f, _rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)

    block = _build_recall_block(store, _QUERY, persona_dir=tmp_path)

    assert block
    (record,) = _records(tmp_path)
    assert record["source"] == "passive"


def test_a_message_with_no_recall_token_and_no_listed_name_writes_no_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Q-R7-3: no recall is performed (`_build_recall_block` returns before the
    semantic stage), so no record is written; the block is the same empty one."""
    store, _g, _f, _rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)

    assert _build_recall_block(store, "ok", persona_dir=tmp_path) == ""

    assert _records(tmp_path) == []


def _db_fingerprint(store: MemoryStore) -> dict[str, object]:
    """Row counts of the tables a diagnostics write could touch, and a checksum of
    the memories (id, content, importance, recall_count, embedding)."""
    import hashlib

    conn = store._conn  # noqa: SLF001
    counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
        for table in ("memories", "calibration_log", "reranker_floor_calibration")
    }
    digest = hashlib.sha256()
    for row in conn.execute(
        "SELECT id, content, importance, recall_count, embedding FROM memories ORDER BY id"
    ):
        digest.update(repr(tuple(row)).encode())
    return {**counts, "memories_sha256": digest.hexdigest()}


def test_only_the_log_and_its_lock_sidecar_appear_and_nothing_diagnostic_is_in_the_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R7-E1 (f). The recall touches the database only through its own (passive)
    calibration row, so the fingerprint is taken with the calibration log excluded
    from the equality and compared for everything else."""
    store, _g, _f, _rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)
    before = _db_fingerprint(store)
    existing = {p.name for p in tmp_path.iterdir()}

    run_semantic_recall(store, tmp_path, _QUERY)

    after = _db_fingerprint(store)
    new = {p.name for p in tmp_path.iterdir()} - existing
    assert new == {_LOG, _LOG + ".lock"}
    tables = {
        row[0]
        for row in store._conn.execute(  # noqa: SLF001
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert not {t for t in tables if "diagnos" in t.lower()}
    assert after["calibration_log"] == before["calibration_log"] + 1, "only the passive row"
    for key in ("memories", "reranker_floor_calibration", "memories_sha256"):
        assert after[key] == before[key], key


def test_a_record_holds_only_numbers_booleans_and_enum_strings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, genuine, family, _rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)

    run_semantic_recall(store, tmp_path, _QUERY)
    dispatch("search_memories", {"query": _QUERY}, **_ctx(tmp_path, store))

    text = (tmp_path / _LOG).read_text()
    assert _QUERY not in text and "memory" not in text.replace("paragraph", "")
    for memory in [*genuine, *family]:
        assert memory.id not in text and memory.content not in text


# ---------------------------------------------------------------------------
# R7-E3: fail-soft, no store access
# ---------------------------------------------------------------------------


def _run_all_three(store: MemoryStore, persona: Path | None) -> dict[str, list[str]]:
    """The same three calls the criteria name, on one store: a passive recall, the
    tool's semantic request and its lexical request. Each returns the contents it
    surfaced, in order."""
    passive = run_semantic_recall(store, persona, _QUERY)  # type: ignore[arg-type]
    semantic = search_memories(
        _QUERY,
        store=store,
        hebbian=HebbianMatrix(":memory:"),
        persona_dir=persona,  # type: ignore[arg-type]
    )
    lexical = search_memories(
        _QUERY,
        mode="lexical",
        store=store,
        hebbian=HebbianMatrix(":memory:"),
        persona_dir=persona,  # type: ignore[arg-type]
    )
    return {
        "passive": [] if passive is None else [m.content for m in [*passive.full, *passive.snippet]],
        "semantic": [m["content"] for m in semantic["memories"]],
        "semantic_mode": [semantic["mode"]],
        "lexical": [m["content"] for m in lexical["memories"]],
        "lexical_mode": [lexical["mode"]],
    }


def _healthy_outcome(monkeypatch, tmp_path_factory) -> dict[str, list[str]]:
    root = tmp_path_factory.mktemp("healthy")
    store, _g, _f, _rec = _first_rerank_case(monkeypatch, root, cosine_floor=0.5)
    outcome = _run_all_three(store, root)
    assert outcome["passive"] and outcome["semantic"] and outcome["semantic_mode"] == ["semantic"]
    # The comparison run starts from a fresh process state (the cost model and the
    # embed warm-up are process-level; the cost series live in each store).
    reranker_mod._reset_rerank_cost_model()
    sr._reset_paragraph_time_model()
    return outcome


@pytest.mark.parametrize(
    "breakage", ["file_for_dir", "persona_none", "log_recall_raises", "missing_directory"]
)
def test_an_emission_failure_never_changes_the_result_or_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    breakage: str,
) -> None:
    """R7-E3 (a): passive recall and both tool modes return exactly what a healthy
    run returns. A missing persona directory is not a failure: the lock sidecar
    creates it and the record lands there (the existing behaviour, asserted)."""
    healthy = _healthy_outcome(monkeypatch, tmp_path_factory)
    store, _g, _f, _rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)
    persona: Path | None = tmp_path
    if breakage == "file_for_dir":
        persona = tmp_path / "not-a-directory"
        persona.write_text("a file where the persona directory should be")
    elif breakage == "persona_none":
        persona = None
    elif breakage == "missing_directory":
        persona = tmp_path / "gone"
    else:

        def _raise(*args, **kwargs):
            raise RuntimeError("simulated diagnostics failure")

        monkeypatch.setattr(rd, "log_recall", _raise)

    assert _run_all_three(store, persona) == healthy

    if breakage == "missing_directory":
        assert persona is not None and len(_records(persona)) == 3, "the sidecar made the directory"
    elif breakage in ("file_for_dir", "persona_none", "log_recall_raises"):
        assert not (tmp_path / _LOG).exists()


def test_the_emission_opens_no_database_and_makes_one_append(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R7-E3 (b), (c): the helper takes no store, a sqlite connection attempted
    during it fails the test, and exactly one locked append happens."""
    assert "store" not in inspect.signature(emit_recall_diagnostics).parameters
    gated = _ranking(RERANKED_PATH, CALIBRATION_SCORE_SCALE, 1.0, real_width=5)
    search = ParagraphSearch(
        outcomes=(_outcome(0, gated, 5),),
        pool={},
        whole_message_fallback=False,
        total_width=5,
        budget=4.0,
        rerank_budget=3.0,
    )

    def _no_db(*args, **kwargs):
        raise AssertionError("the diagnostics emission opened a database")

    appends: list[Path] = []
    real_append = rd.append_record

    def _spy(path, record):
        appends.append(path)
        return real_append(path, record)

    monkeypatch.setattr(sqlite3, "connect", _no_db)
    monkeypatch.setattr(rd, "append_record", _spy)

    emit_recall_diagnostics(tmp_path, search, source="passive")

    assert appends == [tmp_path / _LOG]
    (record,) = _records(tmp_path)
    assert record["total_width"] == 5 and record["budget"] == pytest.approx(4.0)


def test_the_budget_in_the_record_is_the_tunable_in_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, _g, _f, _rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)
    real = tunables.get_tunable
    monkeypatch.setattr(
        tunables,
        "get_tunable",
        lambda key, default: 2.5 if key == "reranker.latency_budget_seconds" else real(key, default),
    )

    run_semantic_recall(store, tmp_path, _QUERY)

    assert _records(tmp_path)[0]["budget"] == pytest.approx(2.5)


def test_search_paragraphs_alone_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The record belongs to the call sites (passive recall, the tool), not to the
    shared per-paragraph search: a direct caller leaves no file behind."""
    store, _g, _f, _rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)

    search_paragraphs(store, _QUERY, log_calibration=False)

    assert not (tmp_path / _LOG).exists()


def test_the_emission_helper_is_called_only_from_the_three_call_sites() -> None:
    """The three sites (plan 1.2-1.3) and nowhere else in `brain/`."""
    callers: set[str] = set()
    root = Path(sr.__file__).resolve().parents[1]
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if name == "emit_recall_diagnostics":
                    callers.add(f"{path.relative_to(root)}")
    assert callers == {"memory/semantic_recall.py", "tools/impls/search_memories.py"}
