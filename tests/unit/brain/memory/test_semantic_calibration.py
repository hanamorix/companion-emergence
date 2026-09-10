"""Tests for brain.memory.semantic_calibration — Stage 4 of the local
semantic-retrieval build (spec decision 5, "Calibration"): the per-persona,
self-recalibrating floor/gap derived from the persona's OWN embedded-memory
similarity distribution using plain statistics, persisted per-persona, and
plugged into `brain.memory.semantic_recall.SemanticCalibration` in place of
the Stage-3 bootstrap default.

Core acceptance (per the build brief): the AUTO-DERIVED floor+gap must
reproduce the #88 case (a paraphrase/no-shared-keyword match surfaces) AND
keep a true-no-match query quiet — using calibration values the algorithm
itself derived from a representative sample corpus, not hand-picked
constants. See `test_auto_derived_calibration_reproduces_88_case_and_stays_
quiet_on_no_match` below.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from brain.memory.embeddings import EmbeddingCache, EmbeddingProvider
from brain.memory.semantic_calibration import (
    CALIBRATION_EMA_ALPHA,
    MIN_CORPUS_FOR_CALIBRATION,
    DerivedCalibration,
    derive_calibration_from_vectors,
    load_persisted_calibration,
    load_semantic_calibration,
    recalibrate_persona,
)
from brain.memory.semantic_recall import (
    SemanticCalibration,
    classify_semantic_shape,
    cosine_similarity,
    surfacing_tiers,
)

DIM = 64


class _ScriptedProvider(EmbeddingProvider):
    """Hand-chosen vectors per text, mirroring
    tests/unit/brain/chat/test_semantic_primary_recall.py's helper — gives
    exact, reproducible cosine relationships instead of hash-seeded noise."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors

    def embed(self, text: str) -> np.ndarray:
        return self._vectors[text].astype(np.float32)

    def embedding_dim(self) -> int:
        return DIM

    def model_id(self) -> str:
        return "scripted-calibration-test"


# ---------------------------------------------------------------------------
# Deterministic corpus construction helpers
# ---------------------------------------------------------------------------


def _background_corpus(n: int = 50, *, seed: int = 7, idio: float = 0.15) -> list[np.ndarray]:
    """`n` vectors sharing one common "bias" direction plus per-vector
    idiosyncratic noise — mimics real sentence embeddings not being
    uniformly distributed on the sphere (most saved memories share some
    generic "everyday sentence" direction), giving a background pairwise-
    similarity distribution with a real median/spread to derive from,
    rather than the near-orthogonal noise a plain `standard_normal` draw
    would give at this dimension.
    """
    bias = np.random.default_rng(1).standard_normal(DIM)
    bias = bias / np.linalg.norm(bias)
    rng = np.random.default_rng(seed)
    vecs = []
    for _ in range(n):
        v = bias + idio * rng.standard_normal(DIM)
        vecs.append((v / np.linalg.norm(v)).astype(np.float32))
    return vecs


def _orthonormal_pair(seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(DIM)
    a = a / np.linalg.norm(a)
    b = rng.standard_normal(DIM)
    b = b - np.dot(b, a) * a
    b = b / np.linalg.norm(b)
    return a, b


def _vec_with_cosine(e1: np.ndarray, e2: np.ndarray, cos_val: float) -> np.ndarray:
    """A vector in the span of orthonormal (e1, e2) whose cosine similarity
    against e1 is exactly `cos_val` — the same trick
    test_semantic_primary_recall.py uses in 2-D, generalised to any
    orthonormal pair so it composes with the higher-dim background corpus."""
    s = float(np.sqrt(max(0.0, 1.0 - cos_val * cos_val)))
    return (cos_val * e1 + s * e2).astype(np.float32)


# ---------------------------------------------------------------------------
# derive_calibration_from_vectors — cold-start gate, determinism, bounds
# ---------------------------------------------------------------------------


def test_cold_start_below_threshold_returns_none() -> None:
    vecs = _background_corpus(n=MIN_CORPUS_FOR_CALIBRATION - 1)
    assert derive_calibration_from_vectors(vecs) is None


def test_at_threshold_returns_a_result() -> None:
    vecs = _background_corpus(n=MIN_CORPUS_FOR_CALIBRATION)
    result = derive_calibration_from_vectors(vecs)
    assert result is not None
    assert result.sample_count == MIN_CORPUS_FOR_CALIBRATION


def test_derive_is_deterministic_given_the_same_corpus() -> None:
    """Same corpus -> same derived (floor, gap), including when the input
    list arrives in a different order (the deterministic byte-sorted sample
    inside derive_calibration_from_vectors must not depend on input order)."""
    vecs = _background_corpus(n=40)
    first = derive_calibration_from_vectors(vecs)
    shuffled = list(reversed(vecs))
    second = derive_calibration_from_vectors(shuffled)
    assert first == second


def test_derived_values_are_within_the_clamped_range() -> None:
    vecs = _background_corpus(n=60)
    result = derive_calibration_from_vectors(vecs)
    assert result is not None
    assert 0.15 <= result.floor <= 0.85
    assert 0.03 <= result.gap <= 0.25


def test_a_near_duplicate_corpus_does_not_derive_a_degenerate_gap() -> None:
    """A corpus where every vector is nearly identical (background spread
    ~0) must still clamp to a usable, non-zero gap — a raw (p90-p50)/2 of
    ~0 would otherwise make the classifier call ANY score drop a cliff."""
    base = np.random.default_rng(3).standard_normal(DIM)
    base = base / np.linalg.norm(base)
    vecs = [
        (base + 1e-4 * np.random.default_rng(i).standard_normal(DIM))
        for i in range(40)
    ]
    vecs = [(v / np.linalg.norm(v)).astype(np.float32) for v in vecs]
    result = derive_calibration_from_vectors(vecs)
    assert result is not None
    assert result.gap >= 0.03  # clamped floor, never a degenerate near-0 gap


# ---------------------------------------------------------------------------
# Core acceptance: auto-derived floor+gap reproduce the #88 case AND keep a
# true-no-match query quiet — using values the algorithm derived from a
# representative sample corpus, not hand-picked constants.
# ---------------------------------------------------------------------------


def test_auto_derived_calibration_reproduces_88_case_and_stays_quiet_on_no_match() -> None:
    # 1. Build a representative sample corpus (this persona's existing
    #    embedded memories) and derive floor/gap from ITS distribution.
    background = _background_corpus(n=50)
    derived = derive_calibration_from_vectors(background)
    assert derived is not None
    calibration = SemanticCalibration(floor=derived.floor, gap=derived.gap)

    # 2. The #88 case: a query with a clear semantic (paraphrase) match, a
    #    keyword-overlap-but-wrong decoy, and a genuinely unrelated memory —
    #    same shape as the spec's own empirical sanity numbers (unrelated
    #    0.424 < decoy 0.602 < paraphrase 0.749), constructed here with
    #    EXACT cosines via the orthonormal-pair trick so the assertion isn't
    #    sensitive to incidental noise.
    e1, e2 = _orthonormal_pair()
    query = e1
    target = _vec_with_cosine(e1, e2, 0.90)  # clear paraphrase standout
    decoy = _vec_with_cosine(e1, e2, 0.60)  # passes the floor, but not a standout
    unrelated = _vec_with_cosine(e1, e2, 0.30)  # below the derived floor

    scored = sorted(
        [
            ("target", cosine_similarity(query, target)),
            ("decoy", cosine_similarity(query, decoy)),
            ("unrelated", cosine_similarity(query, unrelated)),
        ],
        key=lambda pair: -pair[1],
    )
    shape = classify_semantic_shape(scored, calibration=calibration)
    assert shape.kind == "standouts", f"paraphrase must clear the auto-derived floor (calibration={calibration})"
    assert shape.standout_count == 1, "only the paraphrase target is a standout — the decoy must NOT ride along"
    tiers = surfacing_tiers(scored, shape)
    assert tiers is not None
    assert tiers.full_ids == ["target"]
    assert "decoy" not in tiers.full_ids and "decoy" not in tiers.snippet_ids
    assert "unrelated" not in tiers.full_ids and "unrelated" not in tiers.snippet_ids

    # 3. True-no-match: every candidate sits well below the derived floor —
    #    the shape must be "none" (quiet), never a spurious standout.
    noise = [
        _vec_with_cosine(e1, e2, cos_val) for cos_val in (0.25, 0.20, 0.15, 0.10)
    ]
    scored_noise = sorted(
        [(f"n{i}", cosine_similarity(query, v)) for i, v in enumerate(noise)],
        key=lambda pair: -pair[1],
    )
    shape_noise = classify_semantic_shape(scored_noise, calibration=calibration)
    assert shape_noise.kind == "none", (
        f"a true no-match query must stay quiet under the auto-derived calibration "
        f"(calibration={calibration}, top score={scored_noise[0][1]:.3f})"
    )
    assert surfacing_tiers(scored_noise, shape_noise) is None


# ---------------------------------------------------------------------------
# Stability: EMA smoothing damps a single pass from swinging the persisted
# value, given similar (not wildly different) corpora pass to pass.
# ---------------------------------------------------------------------------


def _seed_cache(persona_dir: Path, vectors: dict[str, np.ndarray]) -> EmbeddingCache:
    persona_dir.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(persona_dir / "embeddings.db", _ScriptedProvider(vectors))
    for content in vectors:
        cache.get_or_compute(content)
    return cache


def test_recalibration_smoothing_damps_a_single_pass_swing(tmp_path: Path) -> None:
    """Two passes on broadly SIMILAR corpora (corpus B is corpus A with a
    handful of vectors swapped, not a wholesale change) must not swing the
    persisted floor/gap by the full raw delta — the EMA-smoothed result
    must land strictly between the first pass's persisted value and what an
    UNSMOOTHED second pass would have derived on its own."""
    corpus_a = _background_corpus(n=40, seed=11, idio=0.15)
    corpus_b = _background_corpus(n=40, seed=12, idio=0.22)  # noisier -> different distribution

    vectors_a = {f"mem-a-{i}": v for i, v in enumerate(corpus_a)}
    cache_a = _seed_cache(tmp_path, vectors_a)
    try:
        first = recalibrate_persona(tmp_path, cache_a, now=datetime(2026, 1, 1, tzinfo=UTC))
    finally:
        cache_a.close()
    assert first is not None

    raw_second = derive_calibration_from_vectors(corpus_b)
    assert raw_second is not None

    # Re-seed a FRESH cache under the same persona dir with corpus B (a
    # second EmbeddingCache instance mirrors what a later recalibration pass
    # actually does: it just reads back the persona's now-different set of
    # embedded vectors at that point in time).
    vectors_b = {f"mem-b-{i}": v for i, v in enumerate(corpus_b)}
    cache_b = _seed_cache(tmp_path / "b", vectors_b)  # separate embeddings.db is fine here
    try:
        second = recalibrate_persona(tmp_path, cache_b, now=datetime(2026, 1, 8, tzinfo=UTC))
    finally:
        cache_b.close()
    assert second is not None

    # The persisted result after smoothing must sit strictly between the
    # first pass's value and the raw (unsmoothed) second derivation —
    # proving the smoothing actually pulled it back, not just repeated one
    # or the other value.
    lo, hi = sorted([first.floor, raw_second.floor])
    if lo != hi:  # only meaningful when the two corpora actually differ
        assert lo <= second.floor <= hi

    # And it must match the documented EMA formula exactly (determinism —
    # not just "somewhere in between").
    expected_floor = CALIBRATION_EMA_ALPHA * raw_second.floor + (1 - CALIBRATION_EMA_ALPHA) * first.floor
    expected_gap = CALIBRATION_EMA_ALPHA * raw_second.gap + (1 - CALIBRATION_EMA_ALPHA) * first.gap
    assert second.floor == pytest.approx(expected_floor)
    assert second.gap == pytest.approx(expected_gap)


def test_first_ever_recalibration_has_nothing_to_smooth_against(tmp_path: Path) -> None:
    """With no prior persisted state, the first pass writes the raw derived
    value directly (nothing to EMA against yet)."""
    corpus = _background_corpus(n=40)
    vectors = {f"mem-{i}": v for i, v in enumerate(corpus)}
    cache = _seed_cache(tmp_path, vectors)
    try:
        result = recalibrate_persona(tmp_path, cache)
    finally:
        cache.close()
    raw = derive_calibration_from_vectors(corpus)
    assert result is not None and raw is not None
    assert result.floor == pytest.approx(raw.floor)
    assert result.gap == pytest.approx(raw.gap)


# ---------------------------------------------------------------------------
# recalibrate_persona — cold start no-op, persistence round trip
# ---------------------------------------------------------------------------


def test_recalibrate_persona_below_threshold_is_a_noop_no_file_written(tmp_path: Path) -> None:
    corpus = _background_corpus(n=MIN_CORPUS_FOR_CALIBRATION - 5)
    vectors = {f"mem-{i}": v for i, v in enumerate(corpus)}
    cache = _seed_cache(tmp_path, vectors)
    try:
        result = recalibrate_persona(tmp_path, cache)
    finally:
        cache.close()
    assert result is None
    assert not (tmp_path / "semantic_calibration.json").exists()
    # And the caller-facing loader keeps serving the bootstrap default.
    assert load_semantic_calibration(tmp_path) == SemanticCalibration.bootstrap()


def test_recalibrate_persona_writes_state_and_load_reads_it_back(tmp_path: Path) -> None:
    corpus = _background_corpus(n=40)
    vectors = {f"mem-{i}": v for i, v in enumerate(corpus)}
    cache = _seed_cache(tmp_path, vectors)
    try:
        result = recalibrate_persona(tmp_path, cache, now=datetime(2026, 3, 1, tzinfo=UTC))
    finally:
        cache.close()
    assert result is not None

    on_disk = json.loads((tmp_path / "semantic_calibration.json").read_text())
    assert on_disk["floor"] == pytest.approx(result.floor)
    assert on_disk["gap"] == pytest.approx(result.gap)
    assert on_disk["sample_count"] == 40

    loaded = load_semantic_calibration(tmp_path)
    assert loaded.floor == pytest.approx(result.floor)
    assert loaded.gap == pytest.approx(result.gap)
    assert loaded != SemanticCalibration.bootstrap()


# ---------------------------------------------------------------------------
# load_semantic_calibration — fail-open fallback contract
# ---------------------------------------------------------------------------


def test_load_falls_back_to_bootstrap_on_missing_file(tmp_path: Path) -> None:
    assert load_persisted_calibration(tmp_path) is None
    assert load_semantic_calibration(tmp_path) == SemanticCalibration.bootstrap()


def test_load_falls_back_to_bootstrap_on_corrupt_json(tmp_path: Path) -> None:
    (tmp_path / "semantic_calibration.json").write_text("{not json", encoding="utf-8")
    assert load_persisted_calibration(tmp_path) is None
    assert load_semantic_calibration(tmp_path) == SemanticCalibration.bootstrap()


def test_load_falls_back_to_bootstrap_on_missing_fields(tmp_path: Path) -> None:
    (tmp_path / "semantic_calibration.json").write_text(
        json.dumps({"gap": 0.1}), encoding="utf-8"
    )
    assert load_persisted_calibration(tmp_path) is None
    assert load_semantic_calibration(tmp_path) == SemanticCalibration.bootstrap()


def test_load_falls_back_to_bootstrap_on_non_object_json(tmp_path: Path) -> None:
    (tmp_path / "semantic_calibration.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert load_semantic_calibration(tmp_path) == SemanticCalibration.bootstrap()


def test_derived_calibration_frozen_and_comparable() -> None:
    a = DerivedCalibration(floor=0.5, gap=0.1, sample_count=10)
    b = DerivedCalibration(floor=0.5, gap=0.1, sample_count=10)
    assert a == b
