"""Integration tests for Stage 3 (semantic-PRIMARY recall) + #231 RERANKER
RE-ARCHITECTURE (floor-gated reranker surfacing) driven through the REAL
`_build_recall_block` — not just the raw `brain.memory.semantic_recall`
functions (those are covered directly in
`tests/unit/brain/memory/test_semantic_recall.py`).

Uses a small scripted `EmbeddingProvider` (deterministic, hand-chosen cosine
relationships) so the semantic candidate POOL is populated and coarse-cut
predictably, and a scripted `FakeRerankerProvider` (deterministic,
per-content-text scores) to control which candidates clear the calibrated
reranker floor (F2a inc8: read live via `store.get_reranker_floor`, seeded
directly in this suite via `_seed_floor`) and in what order — since #231 the
reranker score, not cosine, decides surfacing. Fake's hash-seeded default
embedding vectors are ~orthogonal for
any two distinct strings (fine for mechanical plumbing) and
FakeRerankerProvider's default score for an unscripted document sits far
below any floor (see that class's docstring) — neither can exercise a
SPECIFIC semantic relationship like "paraphrase beats keyword-overlap
decoy" on its own, which is exactly what the #88 acceptance case needs.

Vectors are seeded directly onto each memory row's `embedding` /
`embedding_model_id` columns (F1 #259: the warm matrix reads these row
columns, not the old content-hash `embeddings.db` cache), mirroring "the
embed-on-write / idle backfill already ran". `brain.memory.embeddings.
build_embedding_provider` / `brain.memory.semantic_recall.
build_reranker_provider` are monkeypatched to scripted providers so the
query embed / rerank call issued at recall time share the seeded pool.
"""

from __future__ import annotations

import math
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from brain.bridge import model_tier
from brain.chat.prompt import _build_recall_block
from brain.memory.embeddings import EmbeddingProvider
from brain.memory.relevance import SNIPPET_COUNT
from brain.memory.reranker import FakeRerankerProvider
from brain.memory.store import Memory, MemoryStore

_SCRIPTED_MODEL_ID = "scripted-test"
# F2a inc8 cutover: no more module-level `RERANK_FLOOR` — the floor is read
# live from `store.get_reranker_floor(reranker_provider.model_id())`. This
# suite seeds it directly via `_seed_floor` below; the numeric value here has
# no production significance (an arbitrary reference point for test data).
_TEST_FLOOR = -9.25
_FAKE_RERANKER_MODEL_ID = "fake-reranker"
# Row vectors now flow through EmbeddingMatrix. As of #259 inc7 red-team F1,
# EmbeddingMatrix no longer enforces any fixed expected-dim at decode time
# (each row decodes to its own stored byte-length) — a 2-dim test vector
# WOULD now appear in a `matrix.snapshot()` just fine. This suite still pads
# every scripted vector to 384 dims (see `_unit_vec_with_cosine` /
# `_query_unit_vec`) purely to look production-realistic and to keep every
# row in one matrix sharing a uniform shape (needed elsewhere, e.g.
# clustering's `np.stack`), not because a shorter vector would be dropped.
_EMBED_DIM = 384


class _ScriptedProvider(EmbeddingProvider):
    """Returns a HAND-CHOSEN vector for each scripted text; anything else
    embeds to an all-zero vector (cosine 0.0 against everything — a neutral
    "background" score, not accidentally a match)."""

    def __init__(self, vectors: dict[str, np.ndarray], *, dim: int) -> None:
        self._vectors = vectors
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        if text in self._vectors:
            return self._vectors[text].astype(np.float32)
        return np.zeros(self._dim, dtype=np.float32)

    def embedding_dim(self) -> int:
        return self._dim

    def model_id(self) -> str:
        return _SCRIPTED_MODEL_ID


def _unit_vec_with_cosine(score: float) -> np.ndarray:
    """A `_EMBED_DIM`-wide vector whose cosine similarity against
    `_query_unit_vec()` is exactly `score` (for |score| <= 1) — only the
    first two components are non-zero; the zero padding contributes nothing
    to either the dot product or the norm, so it never perturbs the
    hand-chosen cosine relationship."""
    vec = np.zeros(_EMBED_DIM, dtype=np.float32)
    vec[0] = score
    vec[1] = math.sqrt(max(0.0, 1.0 - score * score))
    return vec


def _query_unit_vec() -> np.ndarray:
    """The `_EMBED_DIM`-wide vector `_unit_vec_with_cosine`'s cosine scores
    are measured against — the scripted "query" vector."""
    vec = np.zeros(_EMBED_DIM, dtype=np.float32)
    vec[0] = 1.0
    return vec


def _store(tmp_path: Path) -> MemoryStore:
    """A REAL memories.db file under `persona_dir` — NOT `:memory:`. The
    warm matrix `run_semantic_recall` builds reads from `store.db_path`
    directly (its own separate sqlite3 connection); an in-memory store's
    writes would be invisible to it (two independent `:memory:` databases).
    `persona_dir` IS `tmp_path` throughout this file (every
    `_build_recall_block` call below passes `persona_dir=tmp_path`), so the
    store's backing file lives at the same path the matrix reads."""
    return MemoryStore(tmp_path / "memories.db")


def _mem(store: MemoryStore, content: str, *, importance: float = 1.0) -> Memory:
    m = Memory(
        id=str(uuid.uuid4()),
        content=content,
        memory_type="event",
        domain="d",
        created_at=datetime.now(UTC) - timedelta(seconds=1),
        importance=importance,
    )
    store.create(m)
    return m


def _seed_vectors(store: MemoryStore, vectors: dict[str, np.ndarray], *, contents_by_id: dict[str, str]) -> None:
    """Write a vector directly onto each memory row's `embedding` /
    `embedding_model_id` columns. `contents_by_id` maps memory id -> its
    content, used to look the right vector up in `vectors` (keyed by
    content, matching `_patch_provider`'s scripting)."""
    for memory_id, content in contents_by_id.items():
        vec = vectors[content]
        store._conn.execute(  # noqa: SLF001
            "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
            (np.asarray(vec, dtype=np.float32).tobytes(), _SCRIPTED_MODEL_ID, memory_id),
        )
    store._conn.commit()  # noqa: SLF001


def _patch_provider(monkeypatch: pytest.MonkeyPatch, vectors: dict[str, np.ndarray], *, dim: int) -> None:
    monkeypatch.setattr(
        "brain.memory.embeddings.build_embedding_provider",
        lambda: _ScriptedProvider(vectors, dim=dim),
    )
    # `run_semantic_recall` sources its candidate pool via
    # `build_embedding_matrix`, which derives the matrix's filter model id
    # from `model_tier.model_for_tier(TIER_EMBEDDING)` (F1 #259 step 0) —
    # NOT from whichever provider `build_embedding_provider` is patched to
    # above. Align the two so the matrix's lazy-build filter matches what
    # `_seed_vectors` stamped on the rows; otherwise the first matrix read
    # reloads from disk filtered to the real production model id, finds
    # nothing, and silently discards the seeded vectors.
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, _SCRIPTED_MODEL_ID)


def _patch_reranker(monkeypatch: pytest.MonkeyPatch, scores: dict[str, float]) -> None:
    """Script the RERANKER score per memory-content text (#231: reranker
    score, not cosine, decides surfacing). Any content NOT listed here falls
    back to FakeRerankerProvider's default — far below any plausible floor,
    so it never accidentally clears it."""
    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider",
        lambda **kwargs: FakeRerankerProvider(scores=scores),
    )


def _seed_floor(
    store: MemoryStore, floor: float = _TEST_FLOOR, *, model_id: str = _FAKE_RERANKER_MODEL_ID
) -> None:
    """F2a inc8: write a calibrated floor row directly so the real
    `run_semantic_recall` (driven here through `_build_recall_block`) has
    something to read via `store.get_reranker_floor`."""
    store.write_reranker_floor(
        model_id, floor=floor, raw_fit_floor=floor, sample_pairs=10, is_cold_start=False
    )


def _rc(store: MemoryStore, mid: str) -> float:
    return store._conn.execute(  # noqa: SLF001
        "SELECT recall_count FROM memories WHERE id = ?", (mid,)
    ).fetchone()[0]


def _active_lines(block: str) -> list[str]:
    return [ln for ln in block.splitlines() if re.match(r'^    - \S+: "', ln)]


def _display_ids(block: str) -> list[str]:
    ids = []
    for line in _active_lines(block):
        m = re.match(r'^    - (\S+): "', line)
        if m:
            ids.append(m.group(1))
    return ids


# ---------------------------------------------------------------------------
# #88 case: paraphrase (no shared keyword) beats a keyword-overlap decoy —
# the reranker floor-gates the decoy out, full-inject tier (1 standout).
# ---------------------------------------------------------------------------


def test_88_paraphrase_beats_keyword_overlap_decoy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    query = "how do I calm down when everything feels like too much"
    target = "deep breathing helps when you are feeling anxious"
    decoy = "too much of a flood of party invitations this week"

    dim = _EMBED_DIM
    vectors = {
        query: _query_unit_vec(),
        target: _unit_vec_with_cosine(0.95),
        decoy: _unit_vec_with_cosine(0.60),
    }

    store = _store(tmp_path)
    m_target = _mem(store, target)
    m_decoy = _mem(store, decoy)

    _seed_vectors(store, vectors, contents_by_id={m_target.id: target, m_decoy.id: decoy})
    _patch_provider(monkeypatch, vectors, dim=dim)
    _seed_floor(store)
    _patch_reranker(
        monkeypatch,
        {target: _TEST_FLOOR + 5.0, decoy: _TEST_FLOOR - 2.0},  # decoy scored but below floor
    )

    before_target, before_decoy = _rc(store, m_target.id), _rc(store, m_decoy.id)
    block = _build_recall_block(store, query, persona_dir=tmp_path)

    assert target in block, "the semantically-matching paraphrase memory surfaces"
    assert decoy not in block, "the keyword-overlap-but-semantically-wrong decoy does NOT surface"
    assert _rc(store, m_target.id) - before_target == pytest.approx(1.0), "sole standout gets a FULL tick"
    assert _rc(store, m_decoy.id) == before_decoy, "the excluded decoy is never bumped"


# ---------------------------------------------------------------------------
# Exact-name / proper-noun recall NOT regressed: semantic infra is live and
# non-empty (a real, populated candidate pool) but INCONCLUSIVE for this
# query — nothing scripted for the reranker to prefer, so every candidate
# falls back to FakeRerankerProvider's below-floor default — the lexical
# fallback still catches a proper noun that was never even a semantic
# candidate (no cached vector for it at all).
# ---------------------------------------------------------------------------


def test_exact_name_recall_not_regressed_when_semantic_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dim = _EMBED_DIM
    unrelated_a = "the weather was pleasant that afternoon"
    unrelated_b = "a quiet evening with nothing much happening"
    _vec_up = np.zeros(_EMBED_DIM, dtype=np.float32)
    _vec_up[1] = 1.0
    vectors = {
        unrelated_a: _vec_up,
        unrelated_b: _vec_up,
        # "Zoraida" (the query below) is deliberately NOT scripted here, so
        # it embeds to an all-zero vector at recall time — cosine 0.0
        # against everything. No reranker score is scripted for either
        # candidate either, so both fall to FakeRerankerProvider's
        # below-floor default — guaranteeing an INCONCLUSIVE result even
        # though the candidate pool itself is non-empty.
    }

    store = _store(tmp_path)
    m_unrelated_a = _mem(store, unrelated_a)
    m_unrelated_b = _mem(store, unrelated_b)
    proper_noun_mem = _mem(store, "Zoraida visited the old lighthouse last spring")

    _seed_vectors(
        store, vectors, contents_by_id={m_unrelated_a.id: unrelated_a, m_unrelated_b.id: unrelated_b}
    )
    _patch_provider(monkeypatch, vectors, dim=dim)

    # A bare "Zoraida" (rather than "tell me about Zoraida") keeps every
    # extracted token lexically findable — "tell"/"about" would otherwise
    # ALSO be extracted as content tokens and correctly render under
    # "not recognised" (zero lexical hits of their own), which is unrelated
    # to what this test is checking and would make the assertion below a
    # false negative.
    block = _build_recall_block(store, "Zoraida", persona_dir=tmp_path)
    assert block.strip() != ""
    assert proper_noun_mem.id in _display_ids(block), "the proper noun still surfaces via the lexical fallback"
    assert "not recognised" not in block.lower(), "a real lexical hit must not read as unrecognised"


# ---------------------------------------------------------------------------
# Warm-up: an empty embeddings.db (nothing embedded yet) must not block
# recall — semantic returns nothing, lexical fallback runs exactly as before
# this stage.
# ---------------------------------------------------------------------------


def test_warmup_empty_vector_store_falls_back_to_lexical(tmp_path: Path) -> None:
    store = _store(tmp_path)
    m = _mem(store, "jordan was here that summer")

    # No embeddings.db seeded at all — cold-start / backfill hasn't run yet.
    block = _build_recall_block(store, "what about jordan", persona_dir=tmp_path)

    assert block.strip() != ""
    assert m.id in _display_ids(block)


# ---------------------------------------------------------------------------
# 6-9 standout tier: top 5 full (full tick) + trailing snippet (fractional
# tick); a below-floor candidate that was RERANKED but never surfaced stays
# unbumped (scoring must not tick the counter).
# ---------------------------------------------------------------------------


def test_6_standouts_top5_full_plus_one_snippet_with_correct_ticks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    query = "target query"
    dim = _EMBED_DIM
    contents = [f"standout memory number {i}" for i in range(6)]
    noise_content = "noise memory scored but never surfaced"

    # Cosine vectors just need to be nonzero and distinct enough to populate
    # + coarse-cut the pool — the reranker score (below) is what actually
    # decides tiers under #231.
    vectors = {query: _query_unit_vec()}
    for i, content in enumerate(contents):
        vectors[content] = _unit_vec_with_cosine(0.9 - 0.02 * i)
    vectors[noise_content] = _unit_vec_with_cosine(0.5)

    store = _store(tmp_path)
    mems = [_mem(store, c) for c in contents]
    noise_mem = _mem(store, noise_content)
    before = {m.id: _rc(store, m.id) for m in mems}
    before_noise = _rc(store, noise_mem.id)

    contents_by_id = {m.id: c for m, c in zip(mems, contents, strict=True)}
    contents_by_id[noise_mem.id] = noise_content
    _seed_vectors(store, vectors, contents_by_id=contents_by_id)
    _patch_provider(monkeypatch, vectors, dim=dim)
    _seed_floor(store)
    _patch_reranker(
        monkeypatch,
        {
            **{content: _TEST_FLOOR + 6.0 - i for i, content in enumerate(contents)},  # 6 standouts, strictly descending
            noise_content: _TEST_FLOOR - 1.0,  # scored but below floor
        },
    )

    block = _build_recall_block(store, query, persona_dir=tmp_path)

    ids = _display_ids(block)
    assert len(ids) == 6, "5 full + 1 snippet == 6 rendered bullets"
    assert noise_mem.id not in ids, "the below-floor candidate never surfaces"

    # Ranks 0-4 (the 5 highest-scored contents) are the full tier -> +1.0 each.
    top5 = [m for m in mems if _rc(store, m.id) - before[m.id] == pytest.approx(1.0)]
    assert len(top5) == 5, "exactly 5 memories get the full (+1.0) tick"

    # Rank 5 is the lone snippet -> +0.8 (n_snip == 1 -> top amount).
    snippet_mem = [m for m in mems if m not in top5][0]
    assert _rc(store, snippet_mem.id) - before[snippet_mem.id] == pytest.approx(0.8)

    # The noise candidate was reranked (part of the pool) but never
    # selected — must not be bumped at all.
    assert _rc(store, noise_mem.id) == before_noise


# ---------------------------------------------------------------------------
# #231 correction: 10+ candidates ALL clearing the floor is capped at
# MAX_STANDOUT_COUNT (top 5 full + 4 snippet) — NOT demoted to the lexical
# fallback. The old cosine-era "bunched clump" bucket is gone: a
# trustworthy per-candidate reranker floor means 10+ above-floor candidates
# are 10+ genuinely relevant results.
# ---------------------------------------------------------------------------


def test_10_or_more_standouts_caps_at_9_not_lexical_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    query = "capped query"
    dim = _EMBED_DIM
    contents = [f"clearly relevant memory number {i}" for i in range(12)]

    vectors = {query: _query_unit_vec()}
    for i, content in enumerate(contents):
        vectors[content] = _unit_vec_with_cosine(0.9 - 0.01 * i)

    store = _store(tmp_path)
    mems = [_mem(store, c) for c in contents]

    _seed_vectors(store, vectors, contents_by_id={m.id: c for m, c in zip(mems, contents, strict=True)})
    _patch_provider(monkeypatch, vectors, dim=dim)
    _seed_floor(store)
    _patch_reranker(
        monkeypatch,
        {content: _TEST_FLOOR + 12.0 - i for i, content in enumerate(contents)},  # all 12 clear the floor
    )

    block = _build_recall_block(store, query, persona_dir=tmp_path)

    ids = _display_ids(block)
    assert len(ids) == 9, "capped at MAX_STANDOUT_COUNT (5 full + 4 snippet), not the lexical fallback's top-8"
    # The 9 surfaced are the 9 HIGHEST-reranked (ranks 0-8), not an
    # arbitrary/lexical selection.
    expected_top9_ids = {m.id for m in mems[:9]}
    assert set(ids) == expected_top9_ids


# ---------------------------------------------------------------------------
# Nothing clears the floor: the (already-running, per Stage-3 Defect-3)
# lexical retrieval's own blended-relevance order governs — NOT reranker
# order. Proven the same way the old "clump" test proved the blend orders
# by relevance, not raw cosine: a short, keyword-dense memory with a LOW
# cosine/reranker score outranks a long, keyword-sparse memory with a HIGH
# cosine/reranker score.
# ---------------------------------------------------------------------------


def test_nothing_clears_floor_engages_lexical_fallback_ordered_by_blend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    query = "harbor query"
    dim = _EMBED_DIM

    # Strong lexical match (many mentions, short body).
    lexical_strong = "harbor harbor harbor harbor harbor"
    # Weak lexical match (one mention buried in filler).
    lexical_weak = (
        "harbor mentioned once among a great many other filler words "
        "padding here around it and further padding text besides"
    )
    contents = [lexical_strong, lexical_weak] + [f"harbor filler memory {i}" for i in range(6)]

    vectors = {query: _query_unit_vec()}
    for content in contents:
        vectors[content] = _unit_vec_with_cosine(0.5)  # populates the pool; irrelevant to the outcome

    store = _store(tmp_path)
    mem_strong = _mem(store, lexical_strong)
    mem_weak = _mem(store, lexical_weak)
    other_mems = [_mem(store, content) for content in contents[2:]]

    contents_by_id = {mem_strong.id: lexical_strong, mem_weak.id: lexical_weak}
    contents_by_id.update({m.id: c for m, c in zip(other_mems, contents[2:], strict=True)})
    _seed_vectors(store, vectors, contents_by_id=contents_by_id)
    _patch_provider(monkeypatch, vectors, dim=dim)
    # No reranker scores scripted at all — every candidate falls to
    # FakeRerankerProvider's below-floor default, so NOTHING clears the
    # floor and run_semantic_recall returns None. No floor row is even
    # seeded here — either way (no row, or a row nothing clears) the
    # outcome is the same INCONCLUSIVE None.

    block = _build_recall_block(store, query, persona_dir=tmp_path)

    assert len(_active_lines(block)) == SNIPPET_COUNT, "fallback surfaces its own top-8, unchanged from today"
    i_strong = block.find(mem_strong.id)
    i_weak = block.find(mem_weak.id)
    assert i_strong != -1 and i_weak != -1, "both the strong- and weak-lexical-match memories surface"
    assert i_strong < i_weak, (
        "blended-relevance order (strong lexical match first) governs presentation — "
        "the reranker never even ran a conclusive selection here"
    )
