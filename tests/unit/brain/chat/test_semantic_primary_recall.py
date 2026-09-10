"""Integration tests for Stage 3 (semantic-PRIMARY recall + option-4
surfacing) driven through the REAL `_build_recall_block` — not just the raw
`brain.memory.semantic_recall` functions (those are covered directly in
`tests/unit/brain/memory/test_semantic_recall.py`).

Uses a small scripted `EmbeddingProvider` (deterministic, hand-chosen cosine
relationships) rather than the suite-wide `FakeEmbeddingProvider` — Fake's
hash-seeded vectors are ~orthogonal for any two distinct strings, which is
fine for mechanical plumbing but cannot exercise a SPECIFIC semantic
relationship like "paraphrase beats keyword-overlap decoy", which is exactly
what the #88 acceptance case needs.

Vectors are seeded into `<persona_dir>/embeddings.db` directly (mirroring
"the idle backfill already ran"), and
`brain.memory.embeddings.build_embedding_provider` is monkeypatched to the
same scripted provider so the query embed issued at recall time shares its
model_id/vectors with the seeded pool.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from brain.chat.prompt import _build_recall_block
from brain.memory.embeddings import EmbeddingCache, EmbeddingProvider
from brain.memory.relevance import SNIPPET_COUNT
from brain.memory.store import Memory, MemoryStore


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
        return "scripted-test"


def _unit_vec_with_cosine(score: float) -> np.ndarray:
    """A 2-D unit vector whose cosine similarity against [1.0, 0.0] is
    exactly `score` (for |score| <= 1)."""
    return np.array([score, math.sqrt(max(0.0, 1.0 - score * score))], dtype=np.float32)


def _store() -> MemoryStore:
    return MemoryStore(":memory:")


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


def _seed_vectors(persona_dir: Path, vectors: dict[str, np.ndarray], *, dim: int, contents: list[str]) -> None:
    """Pre-populate embeddings.db as if the idle backfill already embedded
    `contents` (memory bodies only — never the query text itself)."""
    cache = EmbeddingCache(persona_dir / "embeddings.db", _ScriptedProvider(vectors, dim=dim))
    try:
        for content in contents:
            cache.get_or_compute(content)
    finally:
        cache.close()


def _patch_provider(monkeypatch: pytest.MonkeyPatch, vectors: dict[str, np.ndarray], *, dim: int) -> None:
    monkeypatch.setattr(
        "brain.memory.embeddings.build_embedding_provider",
        lambda: _ScriptedProvider(vectors, dim=dim),
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
# semantic surfaces the meaning-matching memory, full-inject tier (1 standout).
# ---------------------------------------------------------------------------


def test_88_paraphrase_beats_keyword_overlap_decoy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    query = "how do I calm down when everything feels like too much"
    target = "deep breathing helps when you are feeling anxious"
    decoy = "too much of a flood of party invitations this week"

    dim = 2
    vectors = {
        query: np.array([1.0, 0.0], dtype=np.float32),
        target: _unit_vec_with_cosine(0.95),  # clear standout
        decoy: _unit_vec_with_cosine(0.60),  # passes the floor but well below the cliff
    }

    store = _store()
    m_target = _mem(store, target)
    m_decoy = _mem(store, decoy)

    _seed_vectors(tmp_path, vectors, dim=dim, contents=[target, decoy])
    _patch_provider(monkeypatch, vectors, dim=dim)

    before_target, before_decoy = _rc(store, m_target.id), _rc(store, m_decoy.id)
    block = _build_recall_block(store, query, persona_dir=tmp_path)

    assert target in block, "the semantically-matching paraphrase memory surfaces"
    assert decoy not in block, "the keyword-overlap-but-semantically-wrong decoy does NOT surface"
    assert _rc(store, m_target.id) - before_target == pytest.approx(1.0), "sole standout gets a FULL tick"
    assert _rc(store, m_decoy.id) == before_decoy, "the excluded decoy is never bumped"


# ---------------------------------------------------------------------------
# Exact-name / proper-noun recall NOT regressed: semantic infra is live and
# non-empty (a real, populated candidate pool) but INCONCLUSIVE for this
# query — the lexical fallback still catches a proper noun that was never
# even a semantic candidate (no cached vector for it at all).
# ---------------------------------------------------------------------------


def test_exact_name_recall_not_regressed_when_semantic_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dim = 2
    unrelated_a = "the weather was pleasant that afternoon"
    unrelated_b = "a quiet evening with nothing much happening"
    vectors = {
        unrelated_a: np.array([0.0, 1.0], dtype=np.float32),
        unrelated_b: np.array([0.0, 1.0], dtype=np.float32),
        # "Zoraida" (the query below) is deliberately NOT scripted here, so
        # it embeds to an all-zero vector at recall time — cosine 0.0
        # against everything, guaranteeing an INCONCLUSIVE semantic shape
        # ("none") even though the candidate pool itself is non-empty.
    }

    store = _store()
    _mem(store, unrelated_a)
    _mem(store, unrelated_b)
    proper_noun_mem = _mem(store, "Zoraida visited the old lighthouse last spring")

    _seed_vectors(tmp_path, vectors, dim=dim, contents=[unrelated_a, unrelated_b])
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
    store = _store()
    m = _mem(store, "jordan was here that summer")

    # No embeddings.db seeded at all — cold-start / backfill hasn't run yet.
    block = _build_recall_block(store, "what about jordan", persona_dir=tmp_path)

    assert block.strip() != ""
    assert m.id in _display_ids(block)


# ---------------------------------------------------------------------------
# Stage 4 wiring: `_build_recall_block` must load THIS persona's persisted
# semantic_calibration.json (brain.memory.semantic_calibration) rather than
# always using SemanticCalibration.bootstrap() — proven by a candidate whose
# cosine clears a persisted (looser) floor but NOT the Stage-3 bootstrap
# floor, with no shared keyword either (so a bootstrap-default run can't
# accidentally pass via the lexical fallback and mask the wiring gap).
# ---------------------------------------------------------------------------


def test_recall_block_uses_persisted_per_persona_calibration_not_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    query = "typhoon relief efforts continue"
    target = "the community rebuilds together slowly"  # no shared keyword with the query

    dim = 2
    vectors = {
        query: np.array([1.0, 0.0], dtype=np.float32),
        target: _unit_vec_with_cosine(0.35),  # below the 0.45 bootstrap floor
    }

    store = _store()
    m_target = _mem(store, target)
    _seed_vectors(tmp_path, vectors, dim=dim, contents=[target])
    _patch_provider(monkeypatch, vectors, dim=dim)

    # 1. WITHOUT a persisted calibration file, the bootstrap default (floor
    #    0.45) excludes the 0.35-cosine candidate, and there's no lexical
    #    overlap either -> it must not surface at all.
    before = _rc(store, m_target.id)
    bootstrap_block = _build_recall_block(store, query, persona_dir=tmp_path)
    assert target not in bootstrap_block, "0.35 cosine must NOT clear the bootstrap floor (0.45)"
    assert _rc(store, m_target.id) == before

    # 2. WITH a persisted per-persona calibration (a looser floor derived —
    #    in production — from THIS persona's own recalibration pass), the
    #    same 0.35-cosine candidate clears it and surfaces as a full-inject
    #    semantic standout.
    (tmp_path / "semantic_calibration.json").write_text(
        json.dumps({"floor": 0.2, "gap": 0.05, "sample_count": 60, "updated_at": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    calibrated_block = _build_recall_block(store, query, persona_dir=tmp_path)
    assert target in calibrated_block, "the persisted (looser) calibration must be the one actually used"
    assert _rc(store, m_target.id) - before == pytest.approx(1.0), "sole standout gets a FULL tick"


# ---------------------------------------------------------------------------
# 6-9 standout tier: top 5 full (full tick) + trailing snippet (fractional
# tick); a below-floor candidate that was SCORED but never surfaced stays
# unbumped (scoring must not tick the counter).
# ---------------------------------------------------------------------------


def test_6_standouts_top5_full_plus_one_snippet_with_correct_ticks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    query = "target query"
    dim = 2
    standout_scores = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65]  # 6 standouts, 0.05 apart (< 0.08 gap)
    contents = [f"standout memory number {i}" for i in range(6)]
    noise_content = "noise memory scored but never surfaced"

    vectors = {query: np.array([1.0, 0.0], dtype=np.float32)}
    for content, score in zip(contents, standout_scores, strict=True):
        vectors[content] = _unit_vec_with_cosine(score)
    vectors[noise_content] = _unit_vec_with_cosine(0.10)  # below the 0.45 floor

    store = _store()
    mems = [_mem(store, c) for c in contents]
    noise_mem = _mem(store, noise_content)
    before = {m.id: _rc(store, m.id) for m in mems}
    before_noise = _rc(store, noise_mem.id)

    _seed_vectors(tmp_path, vectors, dim=dim, contents=[*contents, noise_content])
    _patch_provider(monkeypatch, vectors, dim=dim)

    block = _build_recall_block(store, query, persona_dir=tmp_path)

    ids = _display_ids(block)
    assert len(ids) == 6, "5 full + 1 snippet == 6 rendered bullets"
    assert noise_mem.id not in ids, "the below-floor candidate never surfaces"

    # Ranks 0-4 (scores 0.90..0.70) are the full tier -> +1.0 each.
    top5 = [m for m in mems if _rc(store, m.id) - before[m.id] == pytest.approx(1.0)]
    assert len(top5) == 5, "exactly 5 memories get the full (+1.0) tick"

    # Rank 5 (score 0.65) is the lone snippet -> +0.8 (n_snip == 1 -> top amount).
    snippet_mem = [m for m in mems if m not in top5][0]
    assert _rc(store, snippet_mem.id) - before[snippet_mem.id] == pytest.approx(0.8)

    # The noise candidate was scored (cosine computed, part of the pool) but
    # never selected — must not be bumped at all.
    assert _rc(store, noise_mem.id) == before_noise


# ---------------------------------------------------------------------------
# Clump (bunched, no clear standouts): the lexical FALLBACK engages and
# surfaces its OWN top-8 in blended-relevance order — NOT a re-rank of the
# semantic clump. Proven the same way test_c22 proves the existing blend
# orders by relevance, not raw cosine: a short, keyword-dense memory with a
# LOW cosine score outranks a long, keyword-sparse memory with a HIGH cosine
# score. A cosine re-rank would have put them in the opposite order.
# ---------------------------------------------------------------------------


def test_clump_engages_fresh_lexical_fallback_not_a_semantic_rerank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    query = "harbor query"
    dim = 2

    # Strong lexical match (many mentions, short body) but the WORST cosine
    # of the clump.
    lexical_strong = "harbor harbor harbor harbor harbor"
    # Weak lexical match (one mention buried in filler) but the BEST cosine
    # of the clump.
    lexical_weak = (
        "harbor mentioned once among a great many other filler words "
        "padding here around it and further padding text besides"
    )

    # Exactly SNIPPET_COUNT (8) candidates total — strong + weak + 6 filler —
    # so ALL of them fit within the fallback's own top-8 cutoff with no
    # ranked-9th-or-10th casualty; the discriminating assertion is about
    # ORDER (blend vs cosine), not about which subset survives a cutoff.
    contents = [lexical_strong, lexical_weak] + [f"harbor filler memory {i}" for i in range(6)]
    # 8 candidates, stepped 0.03 apart from 0.90 down to 0.69 — all
    # comfortably clear of the 0.45 floor (a step landing exactly ON the
    # floor is fragile: the float32 round-trip through a constructed unit
    # vector and back through cosine_similarity can land a hair under it)
    # and with gaps well under the 0.08 cliff threshold, so there is no
    # cliff anywhere in the scan window == a clump.
    scores = [0.90 - 0.03 * i for i in range(8)]
    # lexical_weak gets the TOP cosine score; lexical_strong gets the WORST.
    # The 6 filler memories take the scores left over in between.
    score_by_content = dict(zip(contents[2:], scores[1:7], strict=True))
    score_by_content[lexical_weak] = scores[0]
    score_by_content[lexical_strong] = scores[7]

    vectors = {query: np.array([1.0, 0.0], dtype=np.float32)}
    for content, score in score_by_content.items():
        vectors[content] = _unit_vec_with_cosine(score)

    store = _store()
    mem_strong = _mem(store, lexical_strong)
    mem_weak = _mem(store, lexical_weak)
    for content in contents[2:]:
        _mem(store, content)

    _seed_vectors(tmp_path, vectors, dim=dim, contents=contents)
    _patch_provider(monkeypatch, vectors, dim=dim)

    block = _build_recall_block(store, query, persona_dir=tmp_path)

    assert len(_active_lines(block)) == SNIPPET_COUNT, "fallback surfaces its own top-8, unchanged from today"
    i_strong = block.find(mem_strong.id)
    i_weak = block.find(mem_weak.id)
    assert i_strong != -1 and i_weak != -1, "both the strong- and weak-lexical-match memories surface"
    assert i_strong < i_weak, (
        "blended-relevance order (strong lexical match first) governs presentation — "
        "a cosine re-rank would have put the higher-cosine (lexical_weak) memory first instead"
    )
