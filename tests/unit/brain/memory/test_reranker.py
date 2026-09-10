"""Tests for brain.memory.reranker — the #231 cross-encoder reranker
provider + process-wide caches + auto-scaling rerank width.

All OFFLINE (FakeRerankerProvider / a scripted timing stub) — no real model
download. A real-model validation test lives in
tests/unit/brain/memory/test_reranker_real_model.py, marked BOTH
@pytest.mark.requires_network (opts out of the autouse fake-provider
fixture) and @pytest.mark.integration (the marker actually deselected by
the local pre-check gate, -m "not live and not requires_claude_cli and not
integration" — requires_network alone is not part of that expression).
"""

from __future__ import annotations

import pytest

import brain.memory.reranker as reranker_mod
from brain.memory.relevance import CANDIDATE_POOL
from brain.memory.reranker import (
    FakeRerankerProvider,
    RerankerProvider,
    _reset_latency_cache,
    _reset_reranker_provider_cache,
    build_reranker_provider,
    get_rerank_width,
)

# ---------------------------------------------------------------------------
# FakeRerankerProvider — scriptable, offline
# ---------------------------------------------------------------------------


def test_fake_reranker_returns_scripted_scores_positionally_aligned() -> None:
    provider = FakeRerankerProvider(scores={"a": 1.0, "b": 2.0, "c": 3.0})
    scores = list(provider.rerank("any query", ["c", "a", "b"]))
    assert scores == [3.0, 1.0, 2.0], "scores must align POSITIONALLY with the input documents"


def test_fake_reranker_unscripted_document_gets_the_default_far_below_floor() -> None:
    from brain.memory.semantic_recall import RERANK_FLOOR

    provider = FakeRerankerProvider(scores={"scripted": 5.0})
    scores = list(provider.rerank("query", ["scripted", "never scripted"]))
    assert scores[0] == 5.0
    assert scores[1] < RERANK_FLOOR, "the unscripted default must sit below any plausible floor"


def test_fake_reranker_custom_default_is_honored() -> None:
    provider = FakeRerankerProvider(scores={}, default=42.0)
    assert list(provider.rerank("q", ["unscripted"])) == [42.0]


def test_fake_reranker_model_id_is_stable_and_distinct() -> None:
    assert FakeRerankerProvider().model_id() == "fake-reranker"


# ---------------------------------------------------------------------------
# build_reranker_provider — process-wide cache (mirrors test_embeddings.py's
# coverage of build_embedding_provider's own cache).
# ---------------------------------------------------------------------------


def test_build_reranker_provider_is_process_cached_by_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_reranker_provider_cache()
    calls = {"n": 0}

    class _CountingFake(FakeRerankerProvider):
        def __init__(self, model_id: str, cache_dir) -> None:
            super().__init__()
            calls["n"] += 1

    monkeypatch.setattr(reranker_mod, "CrossEncoderProvider", _CountingFake)
    monkeypatch.setattr(
        "brain.bridge.model_tier.model_for_tier", lambda tier: "fake-model-id"
    )
    monkeypatch.setattr(
        "brain.paths.get_cache_dir", lambda: "/tmp/fake-cache-dir"
    )

    p1 = build_reranker_provider()
    p2 = build_reranker_provider()
    assert p1 is p2, "same model_id must return the SAME cached provider instance"
    assert calls["n"] == 1, "construction must happen exactly once, not per call"
    _reset_reranker_provider_cache()


def test_reset_reranker_provider_cache_forces_reconstruction(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_reranker_provider_cache()
    calls = {"n": 0}

    class _CountingFake(FakeRerankerProvider):
        def __init__(self, model_id: str, cache_dir) -> None:
            super().__init__()
            calls["n"] += 1

    monkeypatch.setattr(reranker_mod, "CrossEncoderProvider", _CountingFake)
    monkeypatch.setattr("brain.bridge.model_tier.model_for_tier", lambda tier: "fake-model-id")
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: "/tmp/fake-cache-dir")

    build_reranker_provider()
    _reset_reranker_provider_cache()
    build_reranker_provider()
    assert calls["n"] == 2, "a reset must force the next call to construct again"
    _reset_reranker_provider_cache()


# ---------------------------------------------------------------------------
# get_rerank_width — the spec formula:
#   max(1, min(pool_size, CANDIDATE_POOL, floor(budget / warm_per_doc)))
# ---------------------------------------------------------------------------


class _InstantProvider(RerankerProvider):
    """A reranker whose rerank() returns instantly — used to prove the width
    formula's SHAPE (caps) without depending on real timing noise for the
    "budget generously covers pool_size" cases."""

    def rerank(self, query: str, documents: list[str]):
        return [0.0 for _ in documents]

    def model_id(self) -> str:
        return "instant-test-provider"


def test_width_capped_by_pool_size_when_budget_is_generous() -> None:
    _reset_latency_cache()
    provider = _InstantProvider()
    width = get_rerank_width(5, provider)
    assert width == 5, "an instant provider's per-doc latency is ~0 -> budget never binds -> width = pool_size"
    _reset_latency_cache()


def test_width_capped_by_candidate_pool_when_pool_exceeds_it() -> None:
    _reset_latency_cache()
    provider = _InstantProvider()
    width = get_rerank_width(CANDIDATE_POOL + 25, provider)
    assert width == CANDIDATE_POOL, "width must never exceed relevance.CANDIDATE_POOL"
    _reset_latency_cache()


def test_width_zero_for_empty_pool() -> None:
    _reset_latency_cache()
    assert get_rerank_width(0, _InstantProvider()) == 0
    _reset_latency_cache()


class _SlowProvider(RerankerProvider):
    """A reranker whose rerank() call is timed via a monkeypatched clock —
    used to prove the width formula actually NARROWS under a tight budget,
    without a real sleep (flaky under load) or a real model (network)."""

    def __init__(self, seconds_per_call: float) -> None:
        self._seconds_per_call = seconds_per_call

    def rerank(self, query: str, documents: list[str]):
        return [0.0 for _ in documents]

    def model_id(self) -> str:
        return "slow-test-provider"


def test_width_narrows_under_a_tight_latency_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fakes time.monotonic() so the measured per-doc latency is a KNOWN,
    exact value (0.1s/doc) with no real sleeping — proving the auto-scaler's
    formula narrows the width on a "slow host" the same way a real slow
    (no-AVX2) machine would, per the spec's fast-vs-slow-host acceptance
    criterion, without an actual timing-dependent test."""
    _reset_latency_cache()
    provider = _SlowProvider(seconds_per_call=0.1)

    clock = {"t": 0.0}

    def fake_monotonic() -> float:
        clock["t"] += 0.1  # each call to monotonic() advances by one "call" worth
        return clock["t"]

    monkeypatch.setattr(reranker_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(reranker_mod, "LATENCY_BUDGET_SECONDS", 0.5)

    width = get_rerank_width(50, provider)
    # budget 0.5s / 0.1s-per-doc = 5.
    assert width == 5
    _reset_latency_cache()


def test_width_latency_is_cached_not_remeasured_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_latency_cache()
    calls = {"n": 0}

    class _CountingProvider(RerankerProvider):
        def rerank(self, query: str, documents: list[str]):
            calls["n"] += 1
            return [0.0 for _ in documents]

        def model_id(self) -> str:
            return "counting-test-provider"

    provider = _CountingProvider()
    get_rerank_width(3, provider)
    calls_after_first = calls["n"]
    get_rerank_width(3, provider)
    assert calls["n"] == calls_after_first, "a second call within the recompute interval must not re-measure"
    _reset_latency_cache()


def test_width_measurement_failure_is_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    """A measurement (calibration) failure must not raise into the caller —
    it degrades to 'no latency signal', capping width by pool/CANDIDATE_POOL
    alone (spec: never break recall over a calibration failure)."""
    _reset_latency_cache()

    class _BoomProvider(RerankerProvider):
        def rerank(self, query: str, documents: list[str]):
            raise RuntimeError("simulated calibration failure")

        def model_id(self) -> str:
            return "boom-test-provider"

    width = get_rerank_width(4, _BoomProvider())
    assert width == 4, "measurement failure -> unthrottled (pool_size-capped) width, never an exception"
    _reset_latency_cache()
