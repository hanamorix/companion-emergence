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
    """Scoped to the OUTER provider-caching machinery only — the fp16-vs-
    fp32 precision DECISION (F2a inc2, #250 §2) is a separate concern with
    its own dedicated tests below (`_choose_reranker_model_id` /
    `_run_precision_selfcheck`), so that self-check is short-circuited here
    via a passthrough stub (mirrors how `_fake_reranker_provider_by_default`
    intercepts the whole factory for the rest of the suite)."""
    _reset_reranker_provider_cache()
    calls = {"n": 0}

    class _CountingFake(FakeRerankerProvider):
        def __init__(self, model_id: str, cache_dir) -> None:
            super().__init__()
            calls["n"] += 1

    monkeypatch.setattr(reranker_mod, "CrossEncoderProvider", _CountingFake)
    monkeypatch.setattr(
        reranker_mod, "_choose_reranker_model_id", lambda fp32_id, fp16_id, cache_dir: fp32_id
    )
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
    """Scoped to the OUTER provider-caching machinery only — see the
    docstring on `test_build_reranker_provider_is_process_cached_by_model_id`
    above for why the precision self-check is stubbed to a passthrough
    here."""
    _reset_reranker_provider_cache()
    calls = {"n": 0}

    class _CountingFake(FakeRerankerProvider):
        def __init__(self, model_id: str, cache_dir) -> None:
            super().__init__()
            calls["n"] += 1

    monkeypatch.setattr(reranker_mod, "CrossEncoderProvider", _CountingFake)
    monkeypatch.setattr(
        reranker_mod, "_choose_reranker_model_id", lambda fp32_id, fp16_id, cache_dir: fp32_id
    )
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


# ---------------------------------------------------------------------------
# #231-fix regression: calibration on REAL candidate-pool documents.
#
# The live no-AVX2 defect: `_measure_warm_per_doc_latency` calibrated using
# a fixed 17-word synthetic placeholder (~7-8ms/doc measured), while REAL
# corpus-length documents reranked in production cost ~128ms/doc — so
# `get_rerank_width` always computed floor(1.5/0.0075)=200, clamped to
# min(pool, CANDIDATE_POOL, 200) = CANDIDATE_POOL = 50, EVERY time,
# regardless of true per-doc cost. The fix threads real candidate-pool
# documents into calibration (`sample_documents`) so the measured per-doc
# figure reflects what actually gets reranked.
# ---------------------------------------------------------------------------


class _PlaceholderVsRealisticProvider(RerankerProvider):
    """Simulates a REAL cross-encoder where the fixed short synthetic
    calibration placeholder reranks fast but a corpus-realistic-length
    document reranks much slower — the actual shape of the #231 defect.
    Advances a shared fake monotonic clock by an amount that depends on the
    CONTENT of the document passed to `rerank()` (not a fixed per-call
    constant), so a test can prove calibration actually reads the sampled
    document rather than ignoring it."""

    PLACEHOLDER_SECONDS = 0.0075  # ~7.5ms/doc — matches the live no-AVX2 measurement of the old 17-word stub
    REALISTIC_SECONDS = 0.128  # ~128ms/doc — matches the live no-AVX2 measurement of real corpus documents

    def __init__(self, clock: dict, placeholder_text: str) -> None:
        self._clock = clock
        self._placeholder_text = placeholder_text

    def rerank(self, query: str, documents: list[str]):
        for doc in documents:
            seconds = (
                self.PLACEHOLDER_SECONDS
                if doc == self._placeholder_text
                else self.REALISTIC_SECONDS
            )
            self._clock["t"] += seconds
        return [0.0 for _ in documents]

    def model_id(self) -> str:
        return "placeholder-vs-realistic-test-provider"


def _install_fake_clock(monkeypatch: pytest.MonkeyPatch) -> dict:
    clock = {"t": 0.0}
    monkeypatch.setattr(reranker_mod.time, "monotonic", lambda: clock["t"])
    return clock


def test_get_rerank_width_throttles_when_calibrated_on_realistic_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE #231 REGRESSION TEST: with a per-doc latency representative of
    real corpus documents (~128ms/doc, the live-measured figure), the width
    must narrow BELOW CANDIDATE_POOL — proving the throttle actually
    engages, instead of the pre-fix structural no-op that always returned
    min(pool, CANDIDATE_POOL) = 50 regardless of true per-doc cost.

    floor(LATENCY_BUDGET_SECONDS / REALISTIC_SECONDS) = floor(1.5 / 0.128)
    = 11.

    Confirmed to FAIL against the pre-fix code: the pre-fix
    `get_rerank_width(pool_size, provider)` took no third argument at all,
    so calling it with `sample_documents` raises `TypeError` under that
    signature — and even reasoning from the unchanged formula, the
    pre-fix `_measure_warm_per_doc_latency` always measured
    `_MEASURE_DOCUMENT` (the fast placeholder) regardless of what real
    documents existed, so it could never have produced the 128ms/doc figure
    this test calibrates against. Verified directly by running this test
    against the pre-fix revision (see the session report for the exact
    command/output).
    """
    _reset_latency_cache()
    clock = _install_fake_clock(monkeypatch)
    monkeypatch.setattr(reranker_mod, "LATENCY_BUDGET_SECONDS", 1.5)
    provider = _PlaceholderVsRealisticProvider(clock, reranker_mod._MEASURE_DOCUMENT)

    realistic_doc = "x" * 300  # any content that is NOT the placeholder string
    width = get_rerank_width(CANDIDATE_POOL, provider, [realistic_doc])

    assert width == 11, f"expected floor(1.5/0.128)=11, got {width}"
    assert width < CANDIDATE_POOL, "the whole point of the fix: width must narrow below the pool cap"
    _reset_latency_cache()


def test_get_rerank_width_without_real_docs_reproduces_the_pre_fix_bug_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirms the PRE-FIX failure mode directly on the SAME simulated
    host: with no `sample_documents` supplied, calibration falls back to
    the fixed placeholder — which measures the fast PLACEHOLDER_SECONDS
    figure and therefore never narrows the width below CANDIDATE_POOL. This
    is the #231 defect this module fixes: calibrating on an
    unrepresentative document is a structural no-op, and this test proves
    the fallback path (used by any caller that can't supply real docs)
    still reproduces exactly that no-op shape — it is not itself a fix,
    only `sample_documents` is.
    """
    _reset_latency_cache()
    clock = _install_fake_clock(monkeypatch)
    monkeypatch.setattr(reranker_mod, "LATENCY_BUDGET_SECONDS", 1.5)
    provider = _PlaceholderVsRealisticProvider(clock, reranker_mod._MEASURE_DOCUMENT)

    width = get_rerank_width(CANDIDATE_POOL, provider)  # no sample_documents

    assert width == CANDIDATE_POOL, (
        "calibrating on the placeholder alone measures the FAST synthetic "
        "figure, so the throttle never engages — this is the reproduced bug shape"
    )
    _reset_latency_cache()


def test_measure_warm_per_doc_latency_cycles_through_sample_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration must actually use EACH supplied sample document (cycling
    across the warmup + measured calls), not just repeat the first one —
    otherwise a caller sampling several real docs for variety would gain
    nothing over sampling one."""
    from brain.memory.reranker import _measure_warm_per_doc_latency

    calls: list[str] = []

    class _RecordingProvider(RerankerProvider):
        def rerank(self, query: str, documents: list[str]):
            calls.extend(documents)
            return [0.0 for _ in documents]

        def model_id(self) -> str:
            return "recording-test-provider"

    sample_docs = ["doc-a", "doc-b"]
    _measure_warm_per_doc_latency(_RecordingProvider(), sample_docs)

    total_calls = reranker_mod._WARMUP_RERANKS + reranker_mod._MEASURE_RERANKS
    expected = [sample_docs[i % len(sample_docs)] for i in range(reranker_mod._WARMUP_RERANKS)]
    expected += [sample_docs[i % len(sample_docs)] for i in range(reranker_mod._MEASURE_RERANKS)]

    assert set(calls) == {"doc-a", "doc-b"}, "both sample docs must be exercised, not just the first"
    assert calls == expected, (
        f"{total_calls} total calls ({reranker_mod._WARMUP_RERANKS} warmup + "
        f"{reranker_mod._MEASURE_RERANKS} measured), cycling through the 2 sample docs in order"
    )


def test_measured_calls_cover_a_full_calibration_sample_size_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#231 widen-to-5 regression: real callers build their sample as
    `coarse[:CALIBRATION_SAMPLE_SIZE]`, so a sample sized exactly at
    CALIBRATION_SAMPLE_SIZE must have EVERY one of its documents appear
    among the MEASURED (post-warmup) rerank() calls that determine
    warm_per_doc, not merely among the discarded warmup calls. A measured
    count smaller than the sample size would silently leave the tail of a
    widened sample never actually averaged over."""
    from brain.memory.reranker import _measure_warm_per_doc_latency

    measured_calls: list[str] = []
    call_index = {"n": 0}

    class _PhaseRecordingProvider(RerankerProvider):
        def rerank(self, query: str, documents: list[str]):
            call_index["n"] += 1
            if call_index["n"] > reranker_mod._WARMUP_RERANKS:
                measured_calls.extend(documents)
            return [0.0 for _ in documents]

        def model_id(self) -> str:
            return "phase-recording-test-provider"

    sample_docs = [f"doc-{i}" for i in range(reranker_mod.CALIBRATION_SAMPLE_SIZE)]
    _measure_warm_per_doc_latency(_PhaseRecordingProvider(), sample_docs)

    assert set(measured_calls) == set(sample_docs), (
        "every document in a CALIBRATION_SAMPLE_SIZE-sized sample must be "
        "exercised by a MEASURED call, not just by warmup"
    )
    assert len(measured_calls) == reranker_mod._MEASURE_RERANKS


def test_width_with_sample_docs_is_cached_not_remeasured_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caching (#231's existing behavior) must be preserved when sample
    documents are supplied too — a second call within the recompute
    interval must not re-measure, even though real docs were passed."""
    _reset_latency_cache()
    calls = {"n": 0}

    class _CountingProvider(RerankerProvider):
        def rerank(self, query: str, documents: list[str]):
            calls["n"] += 1
            return [0.0 for _ in documents]

        def model_id(self) -> str:
            return "counting-sample-test-provider"

    provider = _CountingProvider()
    get_rerank_width(3, provider, ["real doc one", "real doc two"])
    calls_after_first = calls["n"]
    assert calls_after_first > 0
    get_rerank_width(3, provider, ["real doc one", "real doc two"])
    assert calls["n"] == calls_after_first, "a second call within the recompute interval must not re-measure"
    _reset_latency_cache()


def test_width_measurement_failure_with_sample_docs_is_fail_soft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fail-soft contract (calibration failure -> 0.0 -> unthrottled
    width) must hold on the NEW code path too, when `sample_documents` is
    supplied and the provider still raises."""
    _reset_latency_cache()

    class _BoomProvider(RerankerProvider):
        def rerank(self, query: str, documents: list[str]):
            raise RuntimeError("simulated calibration failure")

        def model_id(self) -> str:
            return "boom-sample-test-provider"

    width = get_rerank_width(4, _BoomProvider(), ["a real candidate-pool document"])
    assert width == 4, "measurement failure -> unthrottled (pool_size-capped) width, never an exception"
    _reset_latency_cache()


# ---------------------------------------------------------------------------
# fp16-vs-fp32 accuracy self-check (F2a inc2, #250 §2) — all OFFLINE via a
# scripted stub provider + a fake monotonic clock (same techniques the
# latency-auto-calibration tests above already use), no real model/network.
# ---------------------------------------------------------------------------


class _PrecisionTimingProvider(RerankerProvider):
    """Deterministic stub standing in for `CrossEncoderProvider`: scores are
    keyed by (query, doc) and shared across whichever model_id is asked for
    (a test overrides per-model_id via the `scores` dict it's constructed
    with), and each `rerank()` call advances a SHARED fake clock by a
    per-instance fixed amount — the same "advance the clock inside
    rerank()" trick `_PlaceholderVsRealisticProvider` above uses, so
    `_measure_warm_per_doc_latency`'s `time.monotonic()` before/after
    bracketing reads back an exact, known per-doc latency with no real
    sleeping."""

    def __init__(
        self,
        model_id: str,
        scores: dict[tuple[str, str], float],
        clock: dict[str, float],
        seconds_per_call: float,
    ) -> None:
        self._model_id = model_id
        self._scores = scores
        self._clock = clock
        self._seconds_per_call = seconds_per_call

    def rerank(self, query: str, documents: list[str]):
        out = []
        for doc in documents:
            self._clock["t"] += self._seconds_per_call
            out.append(self._scores.get((query, doc), -1_000.0))
        return out

    def model_id(self) -> str:
        return self._model_id


def _install_precision_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fp32_id: str,
    fp16_id: str,
    fp32_scores: dict[tuple[str, str], float],
    fp16_scores: dict[tuple[str, str], float],
    fp32_seconds_per_call: float,
    fp16_seconds_per_call: float,
) -> dict[str, float]:
    """Wires a fake clock + a `CrossEncoderProvider` stub that returns a
    `_PrecisionTimingProvider` scripted per model_id, and no-ops the real
    fastembed registration call (metadata-only in production, but this
    keeps these tests hermetic and independent of fastembed's own
    registry state). Returns the shared clock dict."""
    clock = {"t": 0.0}
    monkeypatch.setattr(reranker_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(reranker_mod, "_register_fp16_reranker_model", lambda *a, **k: None)

    def _fake_ctor(model_id: str, cache_dir):
        if model_id == fp16_id:
            return _PrecisionTimingProvider(model_id, fp16_scores, clock, fp16_seconds_per_call)
        return _PrecisionTimingProvider(model_id, fp32_scores, clock, fp32_seconds_per_call)

    monkeypatch.setattr(reranker_mod, "CrossEncoderProvider", _fake_ctor)
    return clock


def test_precision_selfcheck_ships_fp16_on_agreement_and_a_measured_speed_win(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub scenario 1: fp16 and fp32 agree on every bundled surface/abstain
    decision, AND fp16 measures faster on this (simulated) host -> the gate
    ships fp16."""
    from brain.memory.reranker import (
        _FP16_GATE_PAIRS,
        _choose_reranker_model_id,
        _reset_precision_decision_cache,
    )

    _reset_precision_decision_cache()
    fp32_id, fp16_id = "fake-fp32-agree-fast", "fake-fp16-agree-fast"
    # Every bundled pair scores well above RERANK_FLOOR for BOTH precisions
    # -> every "surfaced" decision agrees.
    agree_scores = dict.fromkeys(_FP16_GATE_PAIRS, 5.0)
    _install_precision_stubs(
        monkeypatch,
        fp32_id=fp32_id,
        fp16_id=fp16_id,
        fp32_scores=agree_scores,
        fp16_scores=agree_scores,
        fp32_seconds_per_call=0.02,
        fp16_seconds_per_call=0.01,  # fp16 measurably faster
    )

    chosen = _choose_reranker_model_id(fp32_id, fp16_id, "/tmp/fake-cache-dir")
    assert chosen == fp16_id, "agreement + a real speed win must ship fp16"
    _reset_precision_decision_cache()


def test_precision_selfcheck_ships_fp32_on_any_decision_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub scenario 2: fp16 flips the surface/abstain decision on exactly
    ONE bundled pair (fp32 keeps it, fp16 would drop it) -> the gate ships
    fp32, even though fp16 would otherwise be faster. Proves the mechanical
    bar is ZERO-tolerance, not a percentage — a single flip fails it."""
    from brain.memory.reranker import (
        _FP16_GATE_PAIRS,
        _choose_reranker_model_id,
        _reset_precision_decision_cache,
    )

    _reset_precision_decision_cache()
    fp32_id, fp16_id = "fake-fp32-disagree", "fake-fp16-disagree"
    fp32_scores = dict.fromkeys(_FP16_GATE_PAIRS, 5.0)  # fp32 keeps everything
    fp16_scores = dict(fp32_scores)
    flipped_pair = _FP16_GATE_PAIRS[0]
    fp16_scores[flipped_pair] = -1_000.0  # fp16 alone drops this one

    _install_precision_stubs(
        monkeypatch,
        fp32_id=fp32_id,
        fp16_id=fp16_id,
        fp32_scores=fp32_scores,
        fp16_scores=fp16_scores,
        fp32_seconds_per_call=0.02,
        fp16_seconds_per_call=0.01,  # fp16 would be faster, but must not matter here
    )

    chosen = _choose_reranker_model_id(fp32_id, fp16_id, "/tmp/fake-cache-dir")
    assert chosen == fp32_id, "any single flipped keep/drop decision must fail the gate -> fp32"
    _reset_precision_decision_cache()


def test_precision_selfcheck_ships_fp32_when_agreement_holds_but_fp16_is_not_faster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub scenario 3: fp16 agrees with fp32 on every bundled decision, but
    measures NO FASTER on this (simulated, no-AVX2-potato-like) host -> the
    gate ships fp32 anyway (agreement alone is not sufficient — §2 FORK 2's
    rationale (iii): a potato CPU may not accelerate fp16, and shipping it
    without a real latency win is a pure downside)."""
    from brain.memory.reranker import (
        _FP16_GATE_PAIRS,
        _choose_reranker_model_id,
        _reset_precision_decision_cache,
    )

    _reset_precision_decision_cache()
    fp32_id, fp16_id = "fake-fp32-not-faster", "fake-fp16-not-faster"
    agree_scores = dict.fromkeys(_FP16_GATE_PAIRS, 5.0)
    _install_precision_stubs(
        monkeypatch,
        fp32_id=fp32_id,
        fp16_id=fp16_id,
        fp32_scores=agree_scores,
        fp16_scores=agree_scores,
        fp32_seconds_per_call=0.01,
        fp16_seconds_per_call=0.02,  # fp16 SLOWER on this simulated host
    )

    chosen = _choose_reranker_model_id(fp32_id, fp16_id, "/tmp/fake-cache-dir")
    assert chosen == fp32_id, "agreement without a measured fp16 speed win must still ship fp32"
    _reset_precision_decision_cache()


def test_precision_selfcheck_decision_is_cached_not_recomputed_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The self-check is a cached FIRST-USE cost, not a per-recall one: a
    second call for the SAME (fp32_model_id, fp16_model_id) pair must be a
    pure cache hit — no additional provider construction."""
    from brain.memory.reranker import (
        _FP16_GATE_PAIRS,
        _choose_reranker_model_id,
        _reset_precision_decision_cache,
    )

    _reset_precision_decision_cache()
    fp32_id, fp16_id = "fake-fp32-cached", "fake-fp16-cached"
    agree_scores = dict.fromkeys(_FP16_GATE_PAIRS, 5.0)

    clock = {"t": 0.0}
    monkeypatch.setattr(reranker_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(reranker_mod, "_register_fp16_reranker_model", lambda *a, **k: None)

    construct_calls = {"n": 0}

    def _fake_ctor(model_id: str, cache_dir):
        construct_calls["n"] += 1
        seconds = 0.01 if model_id == fp16_id else 0.02
        return _PrecisionTimingProvider(model_id, agree_scores, clock, seconds)

    monkeypatch.setattr(reranker_mod, "CrossEncoderProvider", _fake_ctor)

    first = _choose_reranker_model_id(fp32_id, fp16_id, "/tmp/fake-cache-dir")
    calls_after_first = construct_calls["n"]
    assert calls_after_first > 0, "the first (uncached) call must actually run the self-check"

    second = _choose_reranker_model_id(fp32_id, fp16_id, "/tmp/fake-cache-dir")
    assert second == first, "a cache hit must return the same decision"
    assert construct_calls["n"] == calls_after_first, (
        "a second call for the same (fp32, fp16) pair must be a pure cache hit — "
        "the expensive check runs once, not per call"
    )
    _reset_precision_decision_cache()


def test_precision_selfcheck_registration_failure_is_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    """If registering the fp16 export itself fails (e.g. a bad model
    description, an incompatible fastembed version), the self-check must
    still resolve to fp32 rather than raising into a recall."""
    from brain.memory.reranker import _choose_reranker_model_id, _reset_precision_decision_cache

    _reset_precision_decision_cache()
    fp32_id, fp16_id = "fake-fp32-regfail", "fake-fp16-regfail"

    def _boom_register(*args, **kwargs):
        raise RuntimeError("simulated fastembed registration failure")

    monkeypatch.setattr(reranker_mod, "_register_fp16_reranker_model", _boom_register)

    chosen = _choose_reranker_model_id(fp32_id, fp16_id, "/tmp/fake-cache-dir")
    assert chosen == fp32_id, "a registration failure must fail-soft to fp32, never raise"
    _reset_precision_decision_cache()


def test_precision_selfcheck_load_failure_is_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    """If constructing a provider for either candidate fails (e.g. the fp16
    onnx file doesn't actually exist on the HF repo, or a load error), the
    self-check must still resolve to fp32 rather than raising."""
    from brain.memory.reranker import _choose_reranker_model_id, _reset_precision_decision_cache

    _reset_precision_decision_cache()
    fp32_id, fp16_id = "fake-fp32-loadfail", "fake-fp16-loadfail"
    monkeypatch.setattr(reranker_mod, "_register_fp16_reranker_model", lambda *a, **k: None)

    def _boom_ctor(model_id: str, cache_dir):
        raise RuntimeError("simulated onnx load failure")

    monkeypatch.setattr(reranker_mod, "CrossEncoderProvider", _boom_ctor)

    chosen = _choose_reranker_model_id(fp32_id, fp16_id, "/tmp/fake-cache-dir")
    assert chosen == fp32_id, "a provider construction/load failure must fail-soft to fp32, never raise"
    _reset_precision_decision_cache()


def test_precision_selfcheck_cache_key_invalidates_on_model_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cached decision is keyed by the (fp32_model_id, fp16_model_id)
    PAIR, so a model swap on either side (a mini-model registration change)
    must not serve a stale decision computed for the OLD pair."""
    from brain.memory.reranker import (
        _FP16_GATE_PAIRS,
        _choose_reranker_model_id,
        _reset_precision_decision_cache,
    )

    _reset_precision_decision_cache()
    agree_scores = dict.fromkeys(_FP16_GATE_PAIRS, 5.0)
    clock = {"t": 0.0}
    monkeypatch.setattr(reranker_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(reranker_mod, "_register_fp16_reranker_model", lambda *a, **k: None)

    construct_calls = {"n": 0}

    def _fake_ctor(model_id: str, cache_dir):
        construct_calls["n"] += 1
        return _PrecisionTimingProvider(model_id, agree_scores, clock, 0.01)

    monkeypatch.setattr(reranker_mod, "CrossEncoderProvider", _fake_ctor)

    _choose_reranker_model_id("fp32-v1", "fp16-v1", "/tmp/fake-cache-dir")
    calls_after_first_pair = construct_calls["n"]

    # A different fp32/fp16 pair (simulating a model swap) must re-run the
    # self-check, not reuse the old pair's cached decision.
    _choose_reranker_model_id("fp32-v2", "fp16-v2", "/tmp/fake-cache-dir")
    assert construct_calls["n"] > calls_after_first_pair, (
        "a different (fp32, fp16) model-id pair must invalidate the cache and re-run the self-check"
    )
    _reset_precision_decision_cache()


def test_ac3_no_int8_quantization_code_path() -> None:
    """AC#3: int8 quantization was explicitly dropped (Roy's catch: a 278M
    model has less redundancy to absorb int8's accuracy hit than fp16 —
    fp16 is the one quantization lever). Mechanical grep-level check: the
    reranker module (and the model_tier registrations it reads) must
    contain no "int8" code path at all."""
    import inspect

    from brain.bridge import model_tier as model_tier_mod

    assert "int8" not in inspect.getsource(reranker_mod).lower(), (
        "no int8 quantization code path may exist in brain/memory/reranker.py"
    )
    assert "int8" not in inspect.getsource(model_tier_mod).lower(), (
        "no int8 quantization code path may exist in brain/bridge/model_tier.py"
    )
