"""F2b (#276) increment 2 — call-site WIRING tests.

Covers the acceptance criteria that only make sense at the call-site level
(the isolated `normalize_against_anchors` mechanism itself is already fully
covered, offline, in `test_reranker.py`'s "normalize_against_anchors — F2b"
section — this file does NOT re-test that arithmetic):

  - AC1 (integration): the combined document list a call site hands the
    reranker provider has length exactly `width` and includes anchor
    content, at BOTH `run_semantic_recall` (site 1) and `_semantic_top_k`
    (site 2, driven through `dispatch`).
  - AC2: an anchor — even one seeded to score astronomically high — never
    appears in the surfaced result, at either site.
  - AC4 (compose-under-the-floor) + AC5 (both sites identical): a raw score
    below the floor whose NORMALIZED score is above it (and the converse)
    is gated on the NORMALIZED value, proven at BOTH call sites with the
    SAME parametrized scenario — this IS the "shared/parametrized test
    proving both, not just one" AC5 calls for.
  - AC8: `run_semantic_recall`'s calibration-log write (site 1's only log
    write — site 2 has none) records the NORMALIZED score and the
    `real_width`-sized id list, not the raw score or the full `width` list.
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
# Shared scaffolding: 4 candidates with strictly descending cosine
# similarity -> deterministic coarse-cut order -> deterministic width/k.
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


def _seed_four_candidates(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, query: str
) -> tuple[Memory, Memory, Memory, Memory]:
    """Seeds 4 memories with strictly descending cosine similarity to
    `query` (real-A > real-B > filler-C > filler-D) — `coarse` (and so
    `rerank_ids`) is therefore deterministically ordered [real-A, real-B,
    filler-C, filler-D]. With `pool_size=4` and a near-instant
    `_RecordingProvider`/`FakeRerankerProvider` (measured per-doc latency
    ~0 -> `get_rerank_width` is uncapped by the latency budget, so it's
    capped only by `pool_size`/`CANDIDATE_POOL`), `width == 4` ->
    `k = min(P=8, 4 // ANCHOR_SPLIT_DIVISOR=2) == 2` -> `real_width == 2`:
    ONLY real-A/real-B are ever actually sent to the reranker. filler-C/D
    exist purely to pad the coarse-cut pool up to `width == 4` and are
    NEVER scored — proving, as a side effect of every test that uses this
    fixture, that anchors are reserved OUT of `width`, never appended on
    top of it (I6)."""
    every_content_shares_this_token = "wiring"  # lets a lexical fallback find these too (fail-soft tests)
    real_a = _mem(store, f"F2b {every_content_shares_this_token} test real candidate A")
    real_b = _mem(store, f"F2b {every_content_shares_this_token} test real candidate B")
    filler_c = _mem(store, f"F2b {every_content_shares_this_token} test filler candidate C")
    filler_d = _mem(store, f"F2b {every_content_shares_this_token} test filler candidate D")
    _patch_query_embedding(monkeypatch, query)
    _seed_row_vector(store, real_a.id, _unit_vec_with_cosine(0.99))
    _seed_row_vector(store, real_b.id, _unit_vec_with_cosine(0.90))
    _seed_row_vector(store, filler_c.id, _unit_vec_with_cosine(0.50))
    _seed_row_vector(store, filler_d.id, _unit_vec_with_cosine(0.40))
    return real_a, real_b, filler_c, filler_d


_QUERY = "wiring"  # shares a token with every seeded memory (see _seed_four_candidates)


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
    real_a, real_b, _, _ = _seed_four_candidates(store, monkeypatch, _QUERY)
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
    real_a, real_b, _, _ = _seed_four_candidates(store, monkeypatch, _QUERY)
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
# AC1 (integration) — the combined document list a call site sends has
# length == width and includes anchor content; anchors are RESERVED out of
# width (the dropped filler candidates never reach the reranker at all).
# ---------------------------------------------------------------------------


def test_ac1_run_semantic_recall_sends_one_combined_call_of_length_width(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    real_a, real_b, filler_c, filler_d = _seed_four_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)
    provider = _RecordingProvider({}, default=0.0)
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kwargs: provider)

    run_semantic_recall(store, tmp_path, _QUERY)

    # The latency-calibration warm-up/measure calls (get_rerank_width) are
    # all single-document; only the real combined normalize_against_anchors
    # call sends more than one document at once.
    combined_calls = [c for c in provider.calls if len(c) > 1]
    assert len(combined_calls) == 1, "exactly one combined rerank() call for real+anchor documents"
    (sent_docs,) = combined_calls
    assert len(sent_docs) == 4, "combined call length must equal width exactly"
    assert sent_docs[:2] == [real_a.content, real_b.content], (
        "real candidates (the top real_width by coarse rank) occupy the FRONT of the combined list"
    )
    assert sent_docs[2:] == list(ANCHOR_POOL[:2]), (
        "anchors are the ANCHOR_POOL prefix, appended after the real candidates"
    )
    assert filler_c.content not in sent_docs and filler_d.content not in sent_docs, (
        "anchors are RESERVED OUT of width, never appended on top of it — the filler "
        "candidates dropped by the real_width narrowing never reach the reranker at all"
    )


def test_ac1_semantic_top_k_sends_one_combined_call_of_length_width(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    real_a, real_b, filler_c, filler_d = _seed_four_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)
    provider = _RecordingProvider({}, default=0.0)
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kwargs: provider)

    dispatch("search_memories", {"query": _QUERY, "mode": "semantic"}, **_ctx2(tmp_path, store))

    combined_calls = [c for c in provider.calls if len(c) > 1]
    assert len(combined_calls) == 1, "exactly one combined rerank() call for real+anchor documents"
    (sent_docs,) = combined_calls
    assert len(sent_docs) == 4
    assert sent_docs[:2] == [real_a.content, real_b.content]
    assert sent_docs[2:] == list(ANCHOR_POOL[:2])
    assert filler_c.content not in sent_docs and filler_d.content not in sent_docs


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
    real_a, real_b, _, _ = _seed_four_candidates(store, monkeypatch, _QUERY)
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
    assert set(result.scores.keys()) <= {real_a.id, real_b.id}, (
        "scores must never contain any id beyond the real_width-scored real candidates "
        "(an anchor has no memory id, so this also catches any content-vs-id confusion)"
    )


def test_ac2_anchor_never_surfaces_in_semantic_top_k_even_at_very_high_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    real_a, real_b, _, _ = _seed_four_candidates(store, monkeypatch, _QUERY)
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
# NORMALIZED score + the real_width-sized id list (site 2 has no log write).
# ---------------------------------------------------------------------------


def test_ac8_calibration_log_records_normalized_score_and_real_width_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    real_a, real_b, filler_c, filler_d = _seed_four_candidates(store, monkeypatch, _QUERY)
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

    assert candidate_ids == [real_a.id, real_b.id], (
        "logged candidate_ids must be exactly the real_width-scored ids, in scored order — "
        "never the full width's ids, and never including the dropped filler/anchor ids"
    )
    assert filler_c.id not in candidate_ids and filler_d.id not in candidate_ids
    assert logged_scores == pytest.approx([5.0, -10.0]), (
        "logged scores must be the NORMALIZED values (matching what the floor gate actually "
        "compared), never the pre-normalization raw scores (-5.0, -20.0)"
    )
    assert row["score_scale"] == "normalized"

    # Cross-check: this is genuinely the SAME value the floor gate used —
    # real-A (normalized 5.0) surfaced, real-B (normalized -10.0) did not.
    surfaced_ids = {m.id for m in result.full + result.snippet}
    assert real_a.id in surfaced_ids
    assert real_b.id not in surfaced_ids


# ---------------------------------------------------------------------------
# Fail-soft: a normalize_against_anchors failure degrades to lexical at
# BOTH sites, never crashes a turn.
# ---------------------------------------------------------------------------


def test_fail_soft_normalization_error_degrades_run_semantic_recall_to_lexical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    _seed_four_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated normalize_against_anchors failure")

    # `semantic_recall.py` calls `reranker_mod.normalize_against_anchors(...)`
    # — a dynamic module-attribute lookup at call time, so patching the
    # attribute on the reranker module itself is honored (mirrors this
    # suite's existing `_BoomReranker`/build_reranker_provider patch
    # pattern in test_semantic_recall.py).
    monkeypatch.setattr("brain.memory.reranker.normalize_against_anchors", _boom)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is None, (
        "a normalize_against_anchors failure must demote this turn to the lexical fallback, not raise"
    )


def test_fail_soft_normalization_error_degrades_semantic_top_k_to_lexical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    real_a, _, _, _ = _seed_four_candidates(store, monkeypatch, _QUERY)
    _seed_floor(store)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated normalize_against_anchors failure")

    monkeypatch.setattr("brain.memory.reranker.normalize_against_anchors", _boom)

    # `_QUERY` ("wiring") shares a token with every seeded memory's content
    # (see _seed_four_candidates), so the lexical fallback has something
    # real to find — proving this returns a usable result, not just an
    # empty-but-non-erroring response.
    res = dispatch("search_memories", {"query": _QUERY, "mode": "semantic"}, **_ctx2(tmp_path, store))

    assert res["mode"] == "lexical", (
        "a normalize_against_anchors failure must demote this call to the lexical fallback, not raise"
    )
    ids = {mm["id"] for mm in res["memories"]}
    assert real_a.id in ids
