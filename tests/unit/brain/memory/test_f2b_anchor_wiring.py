"""F2b (#276) increment 2 — call-site WIRING tests.

Covers the acceptance criteria that only make sense at the call-site level
(the isolated `normalize_against_anchors` mechanism itself is already fully
covered, offline, in `test_reranker.py`'s "normalize_against_anchors — F2b"
section — this file does NOT re-test that arithmetic):

  - AC1 (integration): the combined document list a call site hands the
    reranker provider is the fitted real candidates with `k` anchors ON TOP
    (name-recall fix R1, S23: real + k documents, k = min(8, real // 2)), at
    BOTH `run_semantic_recall` (site 1) and `_semantic_top_k` (site 2,
    driven through `dispatch`).
  - AC2: an anchor — even one seeded to score astronomically high — never
    appears in the surfaced result, at either site.
  - AC4 (compose-under-the-floor) + AC5 (both sites identical): a raw score
    below the floor whose NORMALIZED score is above it (and the converse)
    is gated on the NORMALIZED value, proven at BOTH call sites with the
    SAME parametrized scenario — this IS the "shared/parametrized test
    proving both, not just one" AC5 calls for.
  - AC8: `run_semantic_recall`'s calibration-log write (site 1's only log
    write — site 2 has none) records the NORMALIZED score and the
    fitted real candidates' ids, never an anchor.
  - Fail-soft: a `normalize_against_anchors` failure at either site degrades
    to the lexical fallback (site 1: `run_semantic_recall` returns None;
    site 2: `search_memories(mode="semantic")` falls back to
    `mode == "lexical"`), never raises.

All offline: `FakeRerankerProvider`/a recording wrapper around it, and a
hand-scripted `EmbeddingProvider`, never a real model or the live Phoebe DB
(I11).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from brain.bridge import model_tier
from brain.memory.embeddings import EmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.reranker import ANCHOR_POOL, FakeRerankerProvider, RerankerProvider
from brain.memory.semantic_recall import run_semantic_recall
from brain.memory.store import Memory, MemoryStore
from brain.tools.dispatch import dispatch

_EMBED_DIM = 384
_TEST_MODEL_ID = "f2b-wiring-test-model"
# An arbitrary reference floor for these tests — no production significance,
# chosen only so the "below"/"above" scores below sit cleanly on either
# side of it.
_FLOOR = 0.0


# ---------------------------------------------------------------------------
# Shared scaffolding: 5 candidates (the S5 rerank minimum) with strictly
# descending cosine similarity -> deterministic coarse-cut order ->
# deterministic width/k (first rerank of a process: width 5, k 2).
# ---------------------------------------------------------------------------


class _QueryOnlyEmbeddingProvider(EmbeddingProvider):
    """Returns a hand-chosen vector for the LITERAL query string only —
    nothing else is embedded live in these tests (every memory row's vector
    is written directly onto the row, bypassing the provider entirely, same
    technique `test_semantic_recall.py`'s `_seed_row_vector` uses)."""

    def __init__(self, query: str, vec: np.ndarray) -> None:
        self._query = query
        self._vec = vec.astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        if text == self._query:
            return self._vec
        return np.zeros(_EMBED_DIM, dtype=np.float32)

    def embedding_dim(self) -> int:
        return _EMBED_DIM

    def model_id(self) -> str:
        return _TEST_MODEL_ID


def _unit_vec_with_cosine(score: float) -> np.ndarray:
    """A `_EMBED_DIM`-wide vector whose cosine similarity against
    `_query_unit_vec()` is exactly `score` — mirrors
    `test_search_memories_mode.py`'s identical helper."""
    vec = np.zeros(_EMBED_DIM, dtype=np.float32)
    vec[0] = score
    vec[1] = math.sqrt(max(0.0, 1.0 - score * score))
    return vec


def _query_unit_vec() -> np.ndarray:
    vec = np.zeros(_EMBED_DIM, dtype=np.float32)
    vec[0] = 1.0
    return vec


def _mem(store: MemoryStore, content: str) -> Memory:
    m = Memory.create_new(content=content, memory_type="event", domain="d")
    store.create(m)
    return m


def _seed_row_vector(store: MemoryStore, memory_id: str, vec: np.ndarray) -> None:
    store._conn.execute(  # noqa: SLF001
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (np.asarray(vec, dtype=np.float32).tobytes(), _TEST_MODEL_ID, memory_id),
    )
    store._conn.commit()  # noqa: SLF001


def _patch_query_embedding(monkeypatch: pytest.MonkeyPatch, query: str) -> None:
    monkeypatch.setattr(
        "brain.memory.embeddings.build_embedding_provider",
        lambda: _QueryOnlyEmbeddingProvider(query, _query_unit_vec()),
    )
    # Align model_tier's embedding tier to the model id `_seed_row_vector`
    # stamps rows with — see `test_semantic_recall.py::_align_embedding_tier`'s
    # docstring for why this must match or the matrix's lazy-build filters
    # the seeded rows straight out.
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, _TEST_MODEL_ID)


def _seed_floor(store: MemoryStore, floor: float = _FLOOR) -> None:
    """`FakeRerankerProvider`/`_RecordingProvider().model_id()` is always
    the literal "fake-reranker" string — the model_id both call sites key
    their `store.get_reranker_floor` lookup on."""
    store.write_reranker_floor(
        "fake-reranker", floor=floor, raw_fit_floor=floor, sample_pairs=10, is_cold_start=False
    )


class _RecordingProvider(RerankerProvider):
    """Wraps a `FakeRerankerProvider` and records the exact `documents` list
    passed to each `rerank()` call — mirrors `test_reranker.py`'s identical
    helper, reused here at the CALL-SITE level (through `run_semantic_
    recall`/`_semantic_top_k`, not `normalize_against_anchors` directly)."""

    def __init__(self, scores: dict[str, float], default: float = -1_000.0) -> None:
        self._fake = FakeRerankerProvider(scores=scores, default=default)
        self.calls: list[list[str]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append(list(documents))
        return self._fake.rerank(query, documents)

    def model_id(self) -> str:
        return "fake-reranker"


def _seed_candidates(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, query: str
) -> tuple[Memory, Memory, Memory, Memory, Memory]:
    """Seeds 5 memories with strictly descending cosine similarity to
    `query` (real-A > real-B > filler-C > filler-D > filler-E), so `coarse`
    is deterministically ordered [real-A, real-B, filler-C, filler-D,
    filler-E]. Name-recall fix R1: the first rerank of a process runs at the
    S5 minimum of 5 real candidates (S24), so all 5 are scored, with
    `k = min(P=8, 5 // 2) == 2` anchors appended ON TOP (7 documents, S23).
    The fillers are unscripted, so the recording provider scores them at its
    far-below-any-floor default and they never surface."""
    every_content_shares_this_token = "wiring"  # lets a lexical fallback find these too (fail-soft tests)
    real_a = _mem(store, f"F2b {every_content_shares_this_token} test real candidate A")
    real_b = _mem(store, f"F2b {every_content_shares_this_token} test real candidate B")
    filler_c = _mem(store, f"F2b {every_content_shares_this_token} test filler candidate C")
    filler_d = _mem(store, f"F2b {every_content_shares_this_token} test filler candidate D")
    filler_e = _mem(store, f"F2b {every_content_shares_this_token} test filler candidate E")
    _patch_query_embedding(monkeypatch, query)
    _seed_row_vector(store, real_a.id, _unit_vec_with_cosine(0.99))
    _seed_row_vector(store, real_b.id, _unit_vec_with_cosine(0.90))
    _seed_row_vector(store, filler_c.id, _unit_vec_with_cosine(0.50))
    _seed_row_vector(store, filler_d.id, _unit_vec_with_cosine(0.40))
    _seed_row_vector(store, filler_e.id, _unit_vec_with_cosine(0.30))
    return real_a, real_b, filler_c, filler_d, filler_e


_QUERY = "wiring"  # shares a token with every seeded memory (see _seed_candidates)


def _below_to_above_scores(real_a_content: str, real_b_content: str) -> dict[str, float]:
    """real-A: raw score sits BELOW the floor, but its NORMALIZED score
    sits ABOVE it. real-B: below the floor on BOTH scales (stays excluded
    either way — proves the gate stays genuinely selective, not merely
    "everything passes because the window shifted")."""
    return {
        ANCHOR_POOL[0]: -10.0,
        ANCHOR_POOL[1]: -10.0,  # median(anchor_scores) == -10.0
        real_a_content: -5.0,  # raw -5.0 < floor 0.0; normalized -5.0-(-10.0) = 5.0 > floor
        real_b_content: -20.0,  # raw -20.0 < floor; normalized -20.0-(-10.0) = -10.0 < floor
    }


def _above_to_below_scores(real_a_content: str, real_b_content: str) -> dict[str, float]:
    """real-A: raw score sits ABOVE the floor, but its NORMALIZED score
    sits BELOW it (the converse of the above). real-B: above the floor on
    BOTH scales (surfaces either way — a bug that gated on the raw score
    instead of normalized would surface BOTH real-A and real-B; the correct
    normalized gate surfaces only real-B)."""
    return {
        ANCHOR_POOL[0]: 10.0,
        ANCHOR_POOL[1]: 10.0,  # median(anchor_scores) == 10.0
        real_a_content: 5.0,  # raw 5.0 > floor 0.0; normalized 5.0-10.0 = -5.0 < floor
        real_b_content: 15.0,  # raw 15.0 > floor; normalized 15.0-10.0 = 5.0 > floor
    }


def _ctx2(tmp_path: Path, store: MemoryStore) -> dict:
    """Site-2 dispatch context (mirrors `test_search_memories_mode.py`'s
    `_ctx`, but takes an already-constructed `store` so callers can seed it
    with the SAME helpers site 1's tests use)."""
    return {"store": store, "hebbian": HebbianMatrix(":memory:"), "persona_dir": tmp_path}


# ---------------------------------------------------------------------------
# AC4 + AC5 — compose-under-the-floor, proven identically at BOTH sites via
# one parametrized scenario.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["below_to_above", "above_to_below"])
def test_ac4_run_semantic_recall_floor_follows_normalized_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, direction: str
) -> None:
    """Site 1 (`run_semantic_recall`) half of AC4/AC5."""
    store = MemoryStore(tmp_path / "memories.db")
    real_a, real_b, *_ = _seed_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)

    if direction == "below_to_above":
        scores = _below_to_above_scores(real_a.content, real_b.content)
        expected_present, expected_absent = real_a, real_b
    else:
        scores = _above_to_below_scores(real_a.content, real_b.content)
        expected_present, expected_absent = real_b, real_a

    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider", lambda **kwargs: _RecordingProvider(scores)
    )

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None, "at least one candidate must clear the NORMALIZED floor"
    surfaced_ids = {m.id for m in result.full + result.snippet}
    assert expected_present.id in surfaced_ids, (
        f"{direction}: the candidate whose NORMALIZED score clears the floor must surface"
    )
    assert expected_absent.id not in surfaced_ids, (
        f"{direction}: the candidate whose NORMALIZED score does NOT clear the floor must "
        "never surface, regardless of its RAW score"
    )
    # `result.scores` covers every ACTUALLY-RERANKED candidate (per its own
    # docstring), not only the ones that cleared the floor — so both ids
    # are expected here; what matters is the VALUE is the normalized one.
    assert expected_present.id in result.scores
    assert expected_absent.id in result.scores


@pytest.mark.parametrize("direction", ["below_to_above", "above_to_below"])
def test_ac5_semantic_top_k_floor_follows_normalized_score_same_as_run_semantic_recall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, direction: str
) -> None:
    """Site 2 (`_semantic_top_k`, driven through `dispatch`) half of
    AC4/AC5 — the SAME scenario/parametrization as
    `test_ac4_run_semantic_recall_floor_follows_normalized_score` above,
    proving both call sites apply IDENTICAL normalize-then-gate composition
    (AC5's explicit "not just one" requirement)."""
    store = MemoryStore(tmp_path / "memories.db")
    real_a, real_b, *_ = _seed_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)

    if direction == "below_to_above":
        scores = _below_to_above_scores(real_a.content, real_b.content)
        expected_present, expected_absent = real_a, real_b
    else:
        scores = _above_to_below_scores(real_a.content, real_b.content)
        expected_present, expected_absent = real_b, real_a

    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider", lambda **kwargs: _RecordingProvider(scores)
    )

    res = dispatch("search_memories", {"query": _QUERY, "mode": "semantic"}, **_ctx2(tmp_path, store))

    assert res["mode"] == "semantic", "at least one candidate must clear the NORMALIZED floor"
    surfaced_ids = {m["id"] for m in res["memories"]}
    assert expected_present.id in surfaced_ids, (
        f"{direction}: the candidate whose NORMALIZED score clears the floor must surface"
    )
    assert expected_absent.id not in surfaced_ids, (
        f"{direction}: the candidate whose NORMALIZED score does NOT clear the floor must "
        "never surface, regardless of its RAW score"
    )


# ---------------------------------------------------------------------------
# AC1 (integration) — the combined document list a call site sends is the
# fitted real candidates with the k anchors ON TOP (R1, S23).
# ---------------------------------------------------------------------------


def test_ac1_run_semantic_recall_sends_real_candidates_with_anchors_on_top(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    real_a, real_b, filler_c, filler_d, filler_e = _seed_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)
    provider = _RecordingProvider({}, default=0.0)
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kwargs: provider)

    run_semantic_recall(store, tmp_path, _QUERY)

    # The two process warm-up reranks are single-document; only the real
    # combined normalize_against_anchors call sends more than one document.
    combined_calls = [c for c in provider.calls if len(c) > 1]
    assert len(combined_calls) == 1, "exactly one combined rerank() call for real+anchor documents"
    (sent_docs,) = combined_calls
    real_contents = [m.content for m in (real_a, real_b, filler_c, filler_d, filler_e)]
    assert len(sent_docs) == 5 + 2, "width 5 real candidates + k = min(8, 5 // 2) = 2 anchors ON TOP"
    assert sent_docs[:5] == real_contents, (
        "every fitted real candidate is scored, in coarse rank order, at the FRONT"
    )
    assert sent_docs[5:] == list(ANCHOR_POOL[:2]), (
        "anchors are the ANCHOR_POOL prefix, appended after the real candidates"
    )


def test_ac1_semantic_top_k_sends_real_candidates_with_anchors_on_top(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    real_a, real_b, filler_c, filler_d, filler_e = _seed_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)
    provider = _RecordingProvider({}, default=0.0)
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kwargs: provider)

    dispatch("search_memories", {"query": _QUERY, "mode": "semantic"}, **_ctx2(tmp_path, store))

    combined_calls = [c for c in provider.calls if len(c) > 1]
    assert len(combined_calls) == 1, "exactly one combined rerank() call for real+anchor documents"
    (sent_docs,) = combined_calls
    real_contents = [m.content for m in (real_a, real_b, filler_c, filler_d, filler_e)]
    assert len(sent_docs) == 5 + 2
    assert sent_docs[:5] == real_contents
    assert sent_docs[5:] == list(ANCHOR_POOL[:2])


# ---------------------------------------------------------------------------
# AC2 — anchors never surface, even seeded to score astronomically high.
# ---------------------------------------------------------------------------


def _high_anchor_scores(real_a_content: str, real_b_content: str) -> dict[str, float]:
    return {
        ANCHOR_POOL[0]: 1_000_000.0,
        ANCHOR_POOL[1]: 1_000_000.0,  # median == 1_000_000.0
        # real-A's RAW score is also astronomically high (higher, even,
        # than each individual anchor score) so it still clears the floor
        # AFTER normalization (1_000_050.0 - 1_000_000.0 == 50.0) — proving
        # something genuinely surfaces here, so "no anchor content in the
        # result" is a meaningful assertion, not vacuously true because
        # nothing surfaced at all.
        real_a_content: 1_000_050.0,
        real_b_content: -1_000_000.0,  # stays excluded on any scale
    }


def test_ac2_anchor_never_surfaces_in_run_semantic_recall_even_at_very_high_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    real_a, real_b, *_ = _seed_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)
    scores = _high_anchor_scores(real_a.content, real_b.content)
    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider", lambda **kwargs: _RecordingProvider(scores)
    )

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    surfaced_ids = {m.id for m in result.full + result.snippet}
    assert real_a.id in surfaced_ids, "test precondition: something must genuinely surface"
    assert real_b.id not in surfaced_ids
    returned_contents = {m.content for m in result.full + result.snippet}
    assert not returned_contents & set(ANCHOR_POOL), "no anchor content may ever appear in the surfaced result"
    seeded_ids = {m.id for m in store.list_active()}
    assert set(result.scores.keys()) <= seeded_ids, (
        "scores must never contain any id beyond the scored real candidates "
        "(an anchor has no memory id, so this also catches any content-vs-id confusion)"
    )


def test_ac2_anchor_never_surfaces_in_semantic_top_k_even_at_very_high_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    real_a, real_b, *_ = _seed_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)
    scores = _high_anchor_scores(real_a.content, real_b.content)
    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider", lambda **kwargs: _RecordingProvider(scores)
    )

    res = dispatch("search_memories", {"query": _QUERY, "mode": "semantic"}, **_ctx2(tmp_path, store))

    assert res["mode"] == "semantic"
    surfaced_ids = {m["id"] for m in res["memories"]}
    assert real_a.id in surfaced_ids, "test precondition: something must genuinely surface"
    assert real_b.id not in surfaced_ids
    returned_contents = {m["content"] for m in res["memories"]}
    assert not returned_contents & set(ANCHOR_POOL), "no anchor content may ever appear in the surfaced result"


# ---------------------------------------------------------------------------
# AC8 — run_semantic_recall's calibration-log write is re-pointed at the
# NORMALIZED score + the scored real ids (site 2 has no log write).
# ---------------------------------------------------------------------------


def test_ac8_calibration_log_records_normalized_score_and_scored_real_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    real_a, real_b, filler_c, filler_d, filler_e = _seed_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)
    scores = _below_to_above_scores(real_a.content, real_b.content)
    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider", lambda **kwargs: _RecordingProvider(scores)
    )

    result = run_semantic_recall(store, tmp_path, _QUERY)
    assert result is not None

    row = store._conn.execute(  # noqa: SLF001
        "SELECT candidate_ids, reranker_scores, score_scale FROM calibration_log"
    ).fetchone()
    candidate_ids = json.loads(row["candidate_ids"])
    logged_scores = json.loads(row["reranker_scores"])

    assert candidate_ids == [real_a.id, real_b.id, filler_c.id, filler_d.id, filler_e.id], (
        "logged candidate_ids must be exactly the scored real candidates, in scored order — "
        "never an anchor"
    )
    # fillers: raw -1000 (the recording provider's default), anchor median -10.
    assert logged_scores == pytest.approx([5.0, -10.0, -990.0, -990.0, -990.0]), (
        "logged scores must be the NORMALIZED values (matching what the floor gate actually "
        "compared), never the pre-normalization raw scores (-5.0, -20.0, -1000.0 ...)"
    )
    assert row["score_scale"] == "normalized"

    # Cross-check: this is genuinely the SAME value the floor gate used —
    # real-A (normalized 5.0) surfaced, real-B (normalized -10.0) did not.
    surfaced_ids = {m.id for m in result.full + result.snippet}
    assert real_a.id in surfaced_ids
    assert real_b.id not in surfaced_ids


# ---------------------------------------------------------------------------
# Fail-soft: a normalize_against_anchors failure never crashes a turn at
# BOTH sites. Name-recall fix R2: it makes the turn a cosine-path turn (gated
# by the cosine floor), not a lexical demotion.
# ---------------------------------------------------------------------------


def _seed_cosine_floor(store: MemoryStore, floor: float) -> None:
    store.write_cosine_floor(
        _TEST_MODEL_ID, floor=floor, raw_fit_floor=floor, sample_pairs=10, is_cold_start=False
    )


def _boom_normalize(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated normalize_against_anchors failure")

    # `semantic_recall.py` reaches `reranker_mod.normalize_against_anchors(...)`
    # through `rerank_for_recall`, a dynamic module-attribute lookup at call
    # time, so patching the attribute on the reranker module itself is honored.
    monkeypatch.setattr("brain.memory.reranker.normalize_against_anchors", _boom)


def test_normalization_error_takes_the_cosine_path_in_run_semantic_recall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    real_a, real_b, *_ = _seed_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)
    _seed_cosine_floor(store, 0.6)  # real-A 0.99 and real-B 0.90 clear it; the 0.5, 0.4, 0.3 fillers do not
    _boom_normalize(monkeypatch)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == "cosine", "not raised, not demoted to keyword-only"
    assert [m.id for m in result.full] == [real_a.id, real_b.id]


def test_normalization_error_with_no_clearing_cosine_floor_is_the_lexical_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    _seed_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)
    _boom_normalize(monkeypatch)  # the suite default cosine floor (2.0) never clears

    assert run_semantic_recall(store, tmp_path, _QUERY) is None


def test_normalization_error_takes_the_cosine_path_at_semantic_top_k(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    real_a, real_b, *_ = _seed_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)
    _seed_cosine_floor(store, 0.6)
    _boom_normalize(monkeypatch)

    res = dispatch("search_memories", {"query": _QUERY, "mode": "semantic"}, **_ctx2(tmp_path, store))

    assert res["mode"] == "semantic"
    assert [mm["id"] for mm in res["memories"]] == [real_a.id, real_b.id]


def test_normalization_error_with_no_clearing_cosine_floor_falls_back_to_lexical_at_semantic_top_k(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    real_a, *_ = _seed_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)
    _boom_normalize(monkeypatch)

    # `_QUERY` ("wiring") shares a token with every seeded memory's content
    # (see _seed_candidates), so the lexical fallback has something real to find.
    res = dispatch("search_memories", {"query": _QUERY, "mode": "semantic"}, **_ctx2(tmp_path, store))

    assert res["mode"] == "lexical"
    assert real_a.id in {mm["id"] for mm in res["memories"]}


@pytest.mark.parametrize("site", ["run_semantic_recall", "search_memories"])
def test_pool_below_the_minimum_hands_off_at_both_call_sites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, site: str
) -> None:
    """R1 (S5/S23): fewer than 5 real candidates -> no rerank at all, and the
    call site takes its explicit hand-off branch (logged at info), not the
    broad fail-soft `except` (which would log a warning traceback). R2: the
    branch is the cosine path (`test_no_rerank_cosine_path.py` covers its
    results); under the suite's never-clearing cosine floor it yields no
    semantic result here."""
    import logging

    store = MemoryStore(tmp_path / "memories.db")
    real_a, *_, filler_e = _seed_candidates(store, monkeypatch, _QUERY)
    store.deactivate(filler_e.id)  # 4 candidates left in the pool
    _seed_floor(store)
    provider = _RecordingProvider({real_a.content: 50.0}, default=0.0)
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kwargs: provider)

    with caplog.at_level(logging.INFO):
        if site == "run_semantic_recall":
            assert run_semantic_recall(store, tmp_path, _QUERY) is None
        else:
            res = dispatch("search_memories", {"query": _QUERY, "mode": "semantic"}, **_ctx2(tmp_path, store))
            assert res["mode"] == "lexical"

    assert provider.calls == [], "no warm-up and no rerank below the minimum"
    assert any("no rerank (pool" in r.getMessage() for r in caplog.records), "the explicit hand-off branch ran"
    assert not any(r.levelno >= logging.WARNING for r in caplog.records), "not the fail-soft except"
