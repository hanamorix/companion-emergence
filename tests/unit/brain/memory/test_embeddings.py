"""Tests for brain.memory.embeddings — provider abstraction.

F1 (#259) increment 8: the old SQLite content-hash cache (`EmbeddingCache`,
`embeddings.db`) that used to sit in front of these providers is removed —
every memory's embedding now lives on its own `memories` row, and the one
remaining transient use (the per-recall query embed) calls
`build_embedding_provider().embed()` directly (covered in
test_semantic_recall.py / test_search_memories_mode.py, not here). This file
now covers only the provider abstraction itself: FakeEmbeddingProvider,
FastEmbedProvider, cosine_similarity, and build_embedding_provider's
process-wide caching.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from brain.memory.embeddings import (
    EmbeddingProvider,
    FakeEmbeddingProvider,
    FastEmbedProvider,
    build_embedding_provider,
    cosine_similarity,
)


@pytest.fixture
def provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


def test_fake_provider_produces_unit_vector(provider: FakeEmbeddingProvider) -> None:
    """FakeEmbeddingProvider returns a unit-norm vector."""
    vec = provider.embed("anything")
    assert isinstance(vec, np.ndarray)
    assert math.isclose(float(np.linalg.norm(vec)), 1.0, rel_tol=1e-6)


def test_fake_provider_embedding_dim_is_256(provider: FakeEmbeddingProvider) -> None:
    """Default embedding dim is 256."""
    vec = provider.embed("x")
    assert vec.shape == (256,)
    assert provider.embedding_dim() == 256


def test_fake_provider_deterministic_same_text(provider: FakeEmbeddingProvider) -> None:
    """Same text → identical vector every time."""
    a = provider.embed("the cold coffee")
    b = provider.embed("the cold coffee")
    np.testing.assert_array_equal(a, b)


def test_fake_provider_different_text_different_vectors(
    provider: FakeEmbeddingProvider,
) -> None:
    """Different text produces different vectors (not identical)."""
    a = provider.embed("hello")
    b = provider.embed("goodbye")
    assert not np.array_equal(a, b)


def test_cosine_similarity_self_is_one() -> None:
    """cosine_similarity(v, v) == 1.0."""
    v = np.array([1.0, 0.0, 0.0])
    assert math.isclose(cosine_similarity(v, v), 1.0, rel_tol=1e-6)


def test_cosine_similarity_orthogonal_is_zero() -> None:
    """Orthogonal vectors have cosine similarity 0."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert math.isclose(cosine_similarity(a, b), 0.0, abs_tol=1e-6)


def test_cosine_similarity_antiparallel_is_negative_one() -> None:
    """Anti-parallel vectors have cosine similarity -1."""
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert math.isclose(cosine_similarity(a, b), -1.0, rel_tol=1e-6)


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    """Zero-norm input returns 0.0 without dividing by zero."""
    zero = np.zeros(3)
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(zero, v) == 0.0
    assert cosine_similarity(v, zero) == 0.0
    assert cosine_similarity(zero, zero) == 0.0


# ---------------------------------------------------------------------------
# model_id — dim-qualified so two providers of different dims never collide
# (e.g. if some future caller ever keys anything by model_id again).
# ---------------------------------------------------------------------------


def test_fake_provider_model_id_is_dim_qualified() -> None:
    """Two Fake providers of different dims must never share a model_id."""
    assert FakeEmbeddingProvider(dim=256).model_id() != FakeEmbeddingProvider(dim=384).model_id()


# ---------------------------------------------------------------------------
# FastEmbedProvider — real local provider. Construction wired against a stub
# fastembed.TextEmbedding so these tests never touch the network or download
# a model file; the real download/inference path is covered by a manual
# smoke check (see the Stage-1 report), not the automated suite.
# ---------------------------------------------------------------------------


class _StubTextEmbedding:
    """Stand-in for fastembed.TextEmbedding — records constructor args,
    returns a deterministic zero vector so shape/plumbing can be asserted
    without any model download or ONNX inference."""

    def __init__(self, model_name: str, cache_dir: str, lazy_load: bool = False, **kwargs) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.lazy_load = lazy_load

    def embed(self, texts):
        for _ in texts:
            yield np.ones(384, dtype=np.float32)


def test_fastembed_provider_wires_model_name_and_cache_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)
    provider = FastEmbedProvider(model_id="BAAI/bge-small-en-v1.5", cache_dir=tmp_path, dim=384)
    assert isinstance(provider, EmbeddingProvider)
    assert provider.model_id() == "BAAI/bge-small-en-v1.5"
    assert provider.embedding_dim() == 384
    assert provider._model.model_name == "BAAI/bge-small-en-v1.5"  # noqa: SLF001
    assert provider._model.cache_dir == str(tmp_path)  # noqa: SLF001
    assert provider._model.lazy_load is True  # noqa: SLF001 — never blocks construction on a download


def test_fastembed_provider_embed_returns_the_declared_dim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)
    provider = FastEmbedProvider(model_id="some/model", cache_dir=tmp_path, dim=384)
    vec = provider.embed("hello")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


# ---------------------------------------------------------------------------
# #259 inc7 red-team F1 — embedding_dim() must derive the REAL model output,
# never echo the constructor's declared `dim=` (a stale MODEL_EMBEDDING_DIM
# after an un-updated model swap). This is the decisive unit-level proof for
# FastEmbedProvider itself; the end-to-end proof (embed_row -> warm matrix ->
# clustering, with no MODEL_EMBEDDING_DIM edit) lives in
# test_embedding_dimension_one_touch_swap.py.
# ---------------------------------------------------------------------------


def _make_stub_text_embedding(output_dim: int):
    """Factory for a fastembed.TextEmbedding stub whose embed() always
    yields vectors of `output_dim` — used to simulate the model's REAL
    output being a DIFFERENT dim than whatever `dim=` a caller declares to
    FastEmbedProvider's constructor."""

    class _Stub:
        def __init__(self, model_name: str, cache_dir: str, lazy_load: bool = False, **kwargs) -> None:
            self.model_name = model_name
            self.cache_dir = cache_dir
            self.lazy_load = lazy_load

        def embed(self, texts):
            for _ in texts:
                yield np.ones(output_dim, dtype=np.float32)

    return _Stub


def test_fastembed_provider_embedding_dim_reflects_the_real_model_output_not_the_declared_dim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Construct with a DECLARED `dim=384` (what a stale MODEL_EMBEDDING_DIM
    would pass after an un-updated model swap), but wire the underlying
    model to REALLY produce 1024-dim vectors. `embedding_dim()` must report
    1024 (the real output), never 384 (the declared value). Fails pre-fix:
    the old `embedding_dim()` just returned the constructor's `dim` — this
    test would have asserted 384."""
    monkeypatch.setattr("fastembed.TextEmbedding", _make_stub_text_embedding(1024))
    provider = FastEmbedProvider(model_id="some/other-model", cache_dir=tmp_path, dim=384)
    assert provider.embedding_dim() == 1024


def test_fastembed_provider_embedding_dim_probes_once_and_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`embedding_dim()` works even before any real `embed()` call has
    happened — it probes the model itself exactly once (never at
    construction — construction must stay network/inference-free), then
    reuses the cached real dim on every later call without re-probing."""
    calls = {"n": 0}
    base_stub = _make_stub_text_embedding(1024)

    class _CountingStub(base_stub):
        def embed(self, texts):
            calls["n"] += 1
            yield from super().embed(texts)

    monkeypatch.setattr("fastembed.TextEmbedding", _CountingStub)
    provider = FastEmbedProvider(model_id="some/other-model", cache_dir=tmp_path, dim=384)
    assert calls["n"] == 0  # construction never embeds

    assert provider.embedding_dim() == 1024
    assert calls["n"] == 1  # one probe embed

    assert provider.embedding_dim() == 1024
    assert calls["n"] == 1  # cached — no second probe


def test_fastembed_provider_logs_loud_error_on_declared_vs_real_dim_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A mismatch between the DECLARED dim (MODEL_EMBEDDING_DIM in
    production) and the REAL probed dim must be logged loudly (ERROR) the
    first time it's discovered — a stale constant is caught rather than
    silently ignored, even though it is not fatal (the real dim is what
    actually drives embed/decode/cluster)."""
    monkeypatch.setattr("fastembed.TextEmbedding", _make_stub_text_embedding(1024))
    provider = FastEmbedProvider(model_id="some/other-model", cache_dir=tmp_path, dim=384)

    with caplog.at_level(logging.ERROR, logger="brain.memory.embeddings"):
        provider.embed("hello")

    assert "1024" in caplog.text
    assert "384" in caplog.text


def test_fastembed_provider_no_mismatch_log_when_declared_dim_matches_real_dim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No spurious loud log when the declared dim genuinely matches the
    real one — the common, correctly-configured case stays quiet."""
    monkeypatch.setattr("fastembed.TextEmbedding", _make_stub_text_embedding(384))
    provider = FastEmbedProvider(model_id="some/other-model", cache_dir=tmp_path, dim=384)

    with caplog.at_level(logging.ERROR, logger="brain.memory.embeddings"):
        provider.embed("hello")

    assert caplog.text == ""


def test_build_embedding_provider_resolves_model_id_from_model_tier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """build_embedding_provider() must source the model id from
    model_tier.model_for_tier(TIER_EMBEDDING) — never a hardcoded literal in
    embeddings.py — and cache into the shared get_cache_dir()."""
    from brain.bridge.model_tier import MODEL_EMBEDDING, MODEL_EMBEDDING_DIM

    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: tmp_path)

    provider = build_embedding_provider()
    assert provider.model_id() == MODEL_EMBEDDING
    assert provider.embedding_dim() == MODEL_EMBEDDING_DIM
    assert provider._model.cache_dir == str(tmp_path)  # noqa: SLF001


def test_build_embedding_provider_repointed_by_reassigning_model_tier_constant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """N-model-extensible: repointing MODEL_EMBEDDING (or TIER_MODEL's
    embedding entry) is a model_tier.py-only edit — build_embedding_provider
    picks it up with no change to embeddings.py itself."""
    from brain.bridge import model_tier

    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: tmp_path)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, "some/other-model")

    provider = build_embedding_provider()
    assert provider.model_id() == "some/other-model"


# ---------------------------------------------------------------------------
# build_embedding_provider() process-wide caching — the Stage-3 hot-path-
# latency fix (was: a fresh FastEmbedProvider/TextEmbedding, i.e. a fresh
# ONNX session, built on EVERY recall call; now: built once per process per
# model_id and reused).
# ---------------------------------------------------------------------------


def test_build_embedding_provider_constructs_the_model_only_once_across_n_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test for the Stage-3 hot-path-latency defect: calling
    build_embedding_provider() N times with the same model_id must construct
    the underlying TextEmbedding exactly ONCE — every call after the first
    must return the SAME cached instance, not pay the ~300-450ms
    construction cost again."""
    construct_count = {"n": 0}

    class _CountingStubTextEmbedding(_StubTextEmbedding):
        def __init__(self, *args, **kwargs) -> None:
            construct_count["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("fastembed.TextEmbedding", _CountingStubTextEmbedding)
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: tmp_path)

    first = build_embedding_provider()
    for _ in range(4):
        again = build_embedding_provider()
        assert again is first, "every call after the first must return the SAME cached provider"

    assert construct_count["n"] == 1, "the model must be constructed exactly once across 5 calls"


def test_build_embedding_provider_cache_is_keyed_by_model_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two different model_ids must each get their OWN cached provider —
    caching by model_id (not a bare singleton) must not silently serve one
    model's provider for a different model_id."""
    from brain.bridge import model_tier

    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: tmp_path)

    first = build_embedding_provider()
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, "some/other-model")
    second = build_embedding_provider()

    assert first is not second
    assert first.model_id() != second.model_id()


def test_build_embedding_provider_cache_reset_hook_forces_a_fresh_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The test-only reset hook (wired into conftest's autouse fixture)
    actually clears the cache — proves the isolation mechanism the conftest
    fixture relies on is real, not a no-op."""
    from brain.memory.embeddings import _reset_embedding_provider_cache

    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: tmp_path)

    first = build_embedding_provider()
    _reset_embedding_provider_cache()
    second = build_embedding_provider()

    assert first is not second, "resetting the cache must force a fresh construction"


class _LazyLoadRaceStubTextEmbedding:
    """Stand-in for fastembed.TextEmbedding that reproduces the ACTUAL race
    pattern found in fastembed's real `OnnxTextModel._embed_documents`:

        if not hasattr(self, "model") or self.model is None:
            self.load_onnx_model()

    an unguarded check-then-act on first use. A `time.sleep` between the
    check and the "load" widens the race window so that two threads calling
    `embed()` on the SAME instance without any external serialization would
    both observe "not yet loaded" and both "load" concurrently —
    `max_concurrent_loads` would exceed 1. Used to prove
    FastEmbedProvider.embed()'s own lock actually serializes against this,
    rather than merely trusting it by inspection.
    """

    _cls_lock = threading.Lock()

    def __init__(self, model_name: str, cache_dir: str, lazy_load: bool = False, **kwargs) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.lazy_load = lazy_load
        self._loaded = False
        self.concurrent_loads = 0
        self.max_concurrent_loads = 0

    def embed(self, texts):
        if not self._loaded:
            with self._cls_lock:
                self.concurrent_loads += 1
                self.max_concurrent_loads = max(self.max_concurrent_loads, self.concurrent_loads)
            time.sleep(0.05)  # widen the race window past a normal context switch
            self._loaded = True
            with self._cls_lock:
                self.concurrent_loads -= 1
        for _ in texts:
            yield np.ones(384, dtype=np.float32)


def test_fastembed_provider_embed_serializes_the_lazy_load_race_across_threads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two threads calling FastEmbedProvider.embed() concurrently on the SAME
    (shared/cached) instance must not race fastembed's own unguarded
    check-then-act lazy-load — proven by `max_concurrent_loads` staying at 1
    under FastEmbedProvider's lock, using a stub that widens the real race
    window (see _LazyLoadRaceStubTextEmbedding). Also asserts no
    exception/deadlock and that every thread gets a correct, consistent
    vector back."""
    monkeypatch.setattr("fastembed.TextEmbedding", _LazyLoadRaceStubTextEmbedding)
    provider = FastEmbedProvider(model_id="some/model", cache_dir=tmp_path, dim=384)

    results: list[np.ndarray] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()

    def worker(text: str) -> None:
        try:
            vec = provider.embed(text)
            with results_lock:
                results.append(vec)
        except BaseException as exc:  # noqa: BLE001 — capture for the assertion, not swallow
            with results_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"text-{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"concurrent embed() calls raised: {errors}"
    assert len(results) == 8
    for vec in results:
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    assert provider._model.max_concurrent_loads == 1, (  # noqa: SLF001
        "FastEmbedProvider.embed() must serialize calls so fastembed's own "
        "unguarded lazy-load check-then-act never overlaps across threads"
    )


def test_stub_race_detector_actually_detects_the_race_when_unserialized(tmp_path: Path) -> None:
    """Control test: proves _LazyLoadRaceStubTextEmbedding's race window is
    real (not a test that could never fail) by calling the stub's embed()
    DIRECTLY from multiple threads with no FastEmbedProvider lock in the
    way — max_concurrent_loads must exceed 1 in that unserialized case,
    which is exactly what FastEmbedProvider.embed()'s lock prevents."""
    stub = _LazyLoadRaceStubTextEmbedding(model_name="some/model", cache_dir=str(tmp_path))

    def worker() -> None:
        list(stub.embed(["hi"]))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert stub.max_concurrent_loads > 1, (
        "the race stub itself must race when nothing external serializes calls to it "
        "-- otherwise the serialization test above wouldn't actually be testing anything"
    )
