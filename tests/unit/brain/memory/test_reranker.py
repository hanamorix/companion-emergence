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

import json
import sys
from unittest.mock import mock_open

import pytest

import brain.memory.reranker as reranker_mod
from brain.memory.relevance import CANDIDATE_POOL
from brain.memory.reranker import (
    FakeRerankerProvider,
    RerankerProvider,
    _default_latency_budget_seconds,
    _detect_avx2,
    _reset_latency_cache,
    _reset_memory_cache,
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
    """F2a inc8: no more module-level `RERANK_FLOOR` to compare against — the
    default is `_DEFAULT_UNSCORED` (a large negative sentinel), so a plain
    sanity bound well below any plausible calibrated floor value (jina's raw
    logits, per the ledger, range roughly -1 digit to small positive) proves
    the same "never accidentally clears a real floor" property."""
    provider = FakeRerankerProvider(scores={"scripted": 5.0})
    scores = list(provider.rerank("query", ["scripted", "never scripted"]))
    assert scores[0] == 5.0
    assert scores[1] < -100.0, "the unscripted default must sit below any plausible floor"


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
    """Scoped to the OUTER provider-caching machinery only — the fp16
    registration side effect (`_register_fp16_reranker_model`, invoked for
    the pinned-fp16 default per the pre-flip revision's Change 2) is stubbed
    to a no-op here so this test never touches fastembed's real model
    registry; which precision id gets resolved is a separate concern with
    its own dedicated tests below."""
    _reset_reranker_provider_cache()
    calls = {"n": 0}

    class _CountingFake(FakeRerankerProvider):
        def __init__(self, model_id: str, cache_dir) -> None:
            super().__init__()
            calls["n"] += 1

    monkeypatch.setattr(reranker_mod, "CrossEncoderProvider", _CountingFake)
    monkeypatch.setattr(reranker_mod, "_register_fp16_reranker_model", lambda *a, **k: None)
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
    above for why fp16 registration is stubbed to a no-op here."""
    _reset_reranker_provider_cache()
    calls = {"n": 0}

    class _CountingFake(FakeRerankerProvider):
        def __init__(self, model_id: str, cache_dir) -> None:
            super().__init__()
            calls["n"] += 1

    monkeypatch.setattr(reranker_mod, "CrossEncoderProvider", _CountingFake)
    monkeypatch.setattr(reranker_mod, "_register_fp16_reranker_model", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.model_tier.model_for_tier", lambda tier: "fake-model-id")
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: "/tmp/fake-cache-dir")

    build_reranker_provider()
    _reset_reranker_provider_cache()
    build_reranker_provider()
    assert calls["n"] == 2, "a reset must force the next call to construct again"
    _reset_reranker_provider_cache()


# ---------------------------------------------------------------------------
# fp16 pinned precision (pre-flip revision Change 2) — supersedes the F2a
# inc2 dual-load fp16-vs-fp32 self-check removed above (see the module
# docstring / RERANKER_PRECISION's own comment in reranker.py). Testing's
# matched-width A/B proved fp16-vs-fp32 agreement is a property of the
# bundled model weights, not something a per-box runtime probe needs to
# establish, so production now pins fp16 as a config default, overridable
# to fp32 via the SAME tunable-override shape LATENCY_BUDGET_SECONDS uses.
# ---------------------------------------------------------------------------


def test_ac1_reranker_precision_defaults_to_fp16() -> None:
    """AC1: the reranker's precision configuration DEFAULTS to fp16 —
    asserted by READING the tunable/config directly, not by inferring it
    from provider-construction behavior."""
    assert reranker_mod.RERANKER_PRECISION == reranker_mod.RERANKER_PRECISION_FP16 == "fp16"
    # No override on disk in this test's environment -> get_tunable must
    # resolve back to that same registered default.
    assert (
        reranker_mod.tunables.get_tunable("reranker.precision", reranker_mod.RERANKER_PRECISION)
        == "fp16"
    )


def test_ac2_build_reranker_provider_constructs_exactly_one_onnx_export_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC2 (single-load proof): `build_reranker_provider()` must construct
    EXACTLY ONE `CrossEncoderProvider` — the fp16 export, by default. No
    code path may construct both a fp16 AND an fp32 `CrossEncoderProvider`
    in the same process lifetime — this is the property the deleted F2a
    inc2 self-check violated (it always warmed BOTH candidates to compare
    them), and the whole reason Change 2 removed it (a confirmed ~1.16 GiB
    one-time dual-load memory spike). This test would have FAILED against
    the old self-check: that code path always appended both `fake-fp32-id`
    AND `fake-fp16-id` to `constructed` below."""
    _reset_reranker_provider_cache()
    constructed: list[str] = []

    class _CountingFake(FakeRerankerProvider):
        def __init__(self, model_id: str, cache_dir) -> None:
            super().__init__()
            constructed.append(model_id)

    monkeypatch.setattr(reranker_mod, "CrossEncoderProvider", _CountingFake)
    registration_calls = {"n": 0}
    monkeypatch.setattr(
        reranker_mod,
        "_register_fp16_reranker_model",
        lambda *a, **k: registration_calls.__setitem__("n", registration_calls["n"] + 1),
    )
    monkeypatch.setattr("brain.bridge.model_tier.model_for_tier", lambda tier: "fake-fp32-id")
    monkeypatch.setattr("brain.bridge.model_tier.MODEL_RERANKER_FP16", "fake-fp16-id")
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: "/tmp/fake-cache-dir")

    provider = build_reranker_provider()

    assert constructed == ["fake-fp16-id"], (
        f"expected exactly one construction, the fp16 default -- got {constructed!r}"
    )
    assert isinstance(provider, FakeRerankerProvider)
    assert registration_calls["n"] == 1, "the fp16 export must be registered exactly once"
    _reset_reranker_provider_cache()


def test_ac3_reranker_precision_override_loads_fp32_instead(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """AC3: with `reranker.precision` explicitly overridden to fp32 (the
    SAME tunables.json override mechanism `test_manual_override_wins_over_
    avx2_auto_detected_default` above exercises for the latency budget),
    the fp32 export loads instead of fp16 — proving the pin is a DEFAULT,
    not a hard removal of operator choice. fp16 registration must never
    even be attempted under this override."""
    import brain.tunables as tunables_mod

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables_mod._reset_for_tests()
    (tmp_path / "tunables.json").write_text(
        json.dumps({"defaults": {}, "overrides": {"reranker.precision": "fp32"}}),
        encoding="utf-8",
    )

    _reset_reranker_provider_cache()
    constructed: list[str] = []

    class _CountingFake(FakeRerankerProvider):
        def __init__(self, model_id: str, cache_dir) -> None:
            super().__init__()
            constructed.append(model_id)

    monkeypatch.setattr(reranker_mod, "CrossEncoderProvider", _CountingFake)
    registration_calls = {"n": 0}
    monkeypatch.setattr(
        reranker_mod,
        "_register_fp16_reranker_model",
        lambda *a, **k: registration_calls.__setitem__("n", registration_calls["n"] + 1),
    )
    monkeypatch.setattr("brain.bridge.model_tier.model_for_tier", lambda tier: "fake-fp32-id")
    monkeypatch.setattr("brain.bridge.model_tier.MODEL_RERANKER_FP16", "fake-fp16-id")
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: "/tmp/fake-cache-dir")

    provider = build_reranker_provider()

    assert constructed == ["fake-fp32-id"], (
        f"expected the fp32 override to load fp32 alone -- got {constructed!r}"
    )
    assert isinstance(provider, FakeRerankerProvider)
    assert registration_calls["n"] == 0, (
        "fp16 registration must never be attempted under an explicit fp32 override"
    )
    _reset_reranker_provider_cache()
    tunables_mod._reset_for_tests()


def test_ac4_no_precision_selfcheck_reference_remains_in_reranker_module() -> None:
    """AC4 (grep-clean): no reference to a cached first-use precision
    decision, an agreement bar, or a floor-write-triggered precision-cache
    invalidation remains in brain/memory/reranker.py or brain/bridge/
    supervisor.py — mechanical check, same shape as
    `test_ac3_no_int8_quantization_code_path` below."""
    import inspect

    from brain.bridge import supervisor as supervisor_mod

    forbidden = [
        "_precision_decision_cache",
        "_choose_reranker_model_id",
        "_run_precision_selfcheck",
        "reset_precision_decision_for_floor_change",
        "agreement bar",
    ]
    for mod in (reranker_mod, supervisor_mod):
        source = inspect.getsource(mod)
        for token in forbidden:
            assert token not in source, (
                f"stale precision-self-check reference {token!r} still present in {mod.__name__}"
            )


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


# ---------------------------------------------------------------------------
# F2a inc3 (#250 §3): startup AVX2 check -> AVX2-aware rerank latency-budget
# default, with manual-override precedence. Four paths: AVX2-detected -> 2s,
# no-AVX2 -> 4s, explicit override wins regardless of the detected default,
# and detection-error -> fail-soft conservative default (no crash).
# ---------------------------------------------------------------------------


def test_default_latency_budget_is_2s_when_avx2_detected() -> None:
    assert _default_latency_budget_seconds(True) == 2.0


def test_default_latency_budget_is_4s_when_avx2_not_detected() -> None:
    assert _default_latency_budget_seconds(False) == 4.0


def test_manual_override_wins_over_avx2_auto_detected_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A manual tunables.json override for reranker.latency_budget_seconds
    must win over the AVX2-auto-detected default — regardless of what that
    default resolved to — per #250 §3's "manual override takes precedence"
    requirement. Simulates an AVX2-detected host (default would be 2.0s,
    via the module-level LATENCY_BUDGET_SECONDS monkeypatch below) with an
    override of 0.3s, so the two are clearly distinguishable: only the
    override value can produce the asserted width."""
    import brain.tunables as tunables_mod

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables_mod._reset_for_tests()
    (tmp_path / "tunables.json").write_text(
        json.dumps({"defaults": {}, "overrides": {"reranker.latency_budget_seconds": 0.3}}),
        encoding="utf-8",
    )
    # The auto-detected default this process would otherwise use (as if
    # AVX2 were detected) — the override must win over THIS, not just over
    # some arbitrary fallback.
    monkeypatch.setattr(reranker_mod, "LATENCY_BUDGET_SECONDS", 2.0)

    _reset_latency_cache()
    provider = _SlowProvider(seconds_per_call=0.1)
    clock = {"t": 0.0}

    def fake_monotonic() -> float:
        clock["t"] += 0.1
        return clock["t"]

    monkeypatch.setattr(reranker_mod.time, "monotonic", fake_monotonic)

    width = get_rerank_width(50, provider)
    # override budget 0.3s / 0.1s-per-doc = 3, NOT floor(2.0/0.1)=20 (the
    # auto-detected-default figure) — proves the override, not the default,
    # drove the computation.
    assert width == 3, f"expected the override (0.3s) to win, got width={width}"
    _reset_latency_cache()
    tunables_mod._reset_for_tests()


def test_avx2_detection_error_is_fail_soft_and_conservative(monkeypatch: pytest.MonkeyPatch) -> None:
    """AVX2 detection failing outright (e.g. /proc/cpuinfo unreadable) must
    never raise into startup — it degrades to "no AVX2" (the conservative,
    larger-budget default), matching #250 §3's fail-soft requirement."""
    monkeypatch.setattr(sys, "platform", "linux")

    def _boom(*args, **kwargs):
        raise OSError("simulated /proc/cpuinfo read failure")

    monkeypatch.setattr("builtins.open", _boom)

    assert _detect_avx2() is False  # no raise


# ---------------------------------------------------------------------------
# F2a inc3 test-coverage gap (#250 §3 follow-up): the tests above never
# exercise the actual /proc/cpuinfo PARSING — they call
# _default_latency_budget_seconds(bool) directly or only force open() to
# raise. `_detect_avx2` reads the `flags` line and does a WHOLE-TOKEN match
# (`"avx2" in line.split(":", 1)[1].split()`), not a substring search — a
# regression to a naive `"avx2" in text` substring check would leave every
# test above green. These tests run the REAL `_detect_avx2()` against
# crafted /proc/cpuinfo content (via mock_open on builtins.open, matching
# the function's actual `open(...).read()`-by-line-iteration mechanism) to
# close that gap.
# ---------------------------------------------------------------------------


def test_detect_avx2_true_when_flags_line_has_the_bare_avx2_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A realistic multi-token `flags` line with `avx2` present among many
    other flags (including the related-but-distinct `avx`) must detect
    True."""
    monkeypatch.setattr(sys, "platform", "linux")
    cpuinfo = (
        "processor\t: 0\n"
        "vendor_id\t: GenuineIntel\n"
        "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca "
        "cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb "
        "rdtscp lm constant_tsc rep_good nopl xtopology cpuid tsc_known_freq "
        "pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave "
        "avx f16c rdrand avx2 bmi1 bmi2\n"
        "bogomips\t: 4800.00\n"
    )
    monkeypatch.setattr("builtins.open", mock_open(read_data=cpuinfo))

    assert _detect_avx2() is True


def test_detect_avx2_false_when_flags_line_has_avx_but_not_avx2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the avx/avx2 PREFIX confusion: `avx` present, `avx2` absent ->
    must be False, not True from a loose `avx` prefix match."""
    monkeypatch.setattr(sys, "platform", "linux")
    cpuinfo = "processor\t: 0\nflags\t\t: fpu vme de pse avx f16c rdrand bmi1 bmi2\n"
    monkeypatch.setattr("builtins.open", mock_open(read_data=cpuinfo))

    assert _detect_avx2() is False


def test_detect_avx2_false_for_decoy_substring_token_without_bare_avx2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE SUBSTRING-REGRESSION GUARD: the flags line contains a token that
    CONTAINS "avx2" as a substring (`avx2_vnni`, the real Intel flag name
    for AVX2-VNNI-INT8 support) but no bare `avx2` token. `_detect_avx2`'s
    real implementation does a whole-token match
    (`"avx2" in line.split(":", 1)[1].split()`), so this must be False. If
    the code were ever regressed to a naive `"avx2" in text` substring
    check, this test would flip to True and fail — that's the point: it
    fails under the regression this whole test class exists to catch.
    Verified by construction: `"avx2" in "avx2_vnni"` is True (substring),
    but `"avx2" in "avx2_vnni".split()` is False (whole-token: split()
    produces the single token "avx2_vnni", not "avx2")."""
    assert "avx2" in "avx2_vnni"  # sanity: the decoy IS a substring match
    assert "avx2" not in "avx2_vnni".split()  # but NOT a whole-token match

    monkeypatch.setattr(sys, "platform", "linux")
    cpuinfo = "processor\t: 0\nflags\t\t: fpu vme de pse avx2_vnni bmi1 bmi2\n"
    monkeypatch.setattr("builtins.open", mock_open(read_data=cpuinfo))

    assert _detect_avx2() is False


def test_detect_avx2_false_for_arm_style_cpuinfo_with_no_flags_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ARM-style /proc/cpuinfo has a `Features:` line, not `flags` — the
    loop never finds a line starting with "flags" and falls through to the
    trailing `return False`."""
    monkeypatch.setattr(sys, "platform", "linux")
    cpuinfo = (
        "processor\t: 0\n"
        "model name\t: ARMv8 Processor rev 1 (v8l)\n"
        "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
        "CPU implementer\t: 0x41\n"
    )
    monkeypatch.setattr("builtins.open", mock_open(read_data=cpuinfo))

    assert _detect_avx2() is False


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
# get_rerank_width — pre-flip revision Change 3 (RAM-and-time-aware rerank
# width): a MEMORY bound added as an additional term in the same min(). All
# offline: `_warm_per_doc_memory` / `_available_ram_headroom_bytes` are
# monkeypatched directly on the module — exactly the same seam the tests
# above use to script the TIME term (`reranker_mod.time.monotonic` /
# `reranker_mod.LATENCY_BUDGET_SECONDS`) — no real /proc or /sys reads, no
# real RSS measurement, no real model.
# ---------------------------------------------------------------------------


def _script_memory_term(
    monkeypatch: pytest.MonkeyPatch, *, per_doc_memory: float, headroom: float | None
) -> None:
    """Force `get_rerank_width`'s memory term to a known, deterministic
    value by monkeypatching the two functions it reads — `_warm_per_doc_
    memory` (the cached, measured-once-warm figure) and `_available_ram_
    headroom_bytes` (the cheap per-call host read) — exactly the seam
    production code reads through."""
    monkeypatch.setattr(
        reranker_mod, "_warm_per_doc_memory", lambda provider, sample_docs=None: per_doc_memory
    )
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: headroom)


def test_memory_term_bites_alone_with_tight_memory_and_generous_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC1: a scripted tight per_doc_MEMORY/headroom ratio, with a generous
    latency budget/pool (an instant provider -> the time term never binds),
    must bound width by the MEMORY term — strictly smaller than what the
    time-only formula would have returned.

    Bite-checked: with the memory-term `min()` argument removed from
    `get_rerank_width`, this assertion (`width == 10`) fails — the function
    instead returns 50 (the pool/CANDIDATE_POOL cap), same as
    `time_only_width` below. Confirmed by temporarily deleting that term and
    re-running this test before landing the change; restored afterward."""
    _reset_latency_cache()
    _reset_memory_cache()
    provider = _InstantProvider()  # ~0 per-doc TIME -> time term never binds

    time_only_width = get_rerank_width(50, provider)
    assert time_only_width == 50, "sanity: with no memory term, an instant provider's width is pool-capped"

    _reset_latency_cache()
    _reset_memory_cache()
    # headroom=100 bytes / per_doc_MEMORY=10 bytes -> memory term = 10, well
    # below the 50 the time-only formula produced above.
    _script_memory_term(monkeypatch, per_doc_memory=10.0, headroom=100.0)
    width = get_rerank_width(50, provider)

    assert width == 10, f"expected the memory term (floor(100/10)=10) to bind, got {width}"
    assert width < time_only_width, "the memory bound must be strictly smaller than the time-only result"
    _reset_latency_cache()
    _reset_memory_cache()


def test_time_term_still_bites_alone_with_generous_memory_and_tight_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC2 (regression guard): generous memory (headroom vastly exceeds any
    plausible per-doc memory cost) + a tight latency budget must reproduce
    EXACTLY what the pre-Change-3 time-only formula returns for these same
    inputs (see `test_width_narrows_under_a_tight_latency_budget` above,
    identical timing setup, width=5) — adding the memory term must not
    change behavior when memory isn't the binding constraint.

    Bite-checked: with the memory-term `min()` argument NOT gated behind
    `per_doc_memory > 0.0 and headroom is not None` (i.e. always appended,
    or the whole term removed so this test degenerates to a tautology), a
    deliberately tiny scripted headroom would flip this assertion — this
    test uses a huge headroom specifically so the memory term, if present,
    still does not bind, isolating the regression-guard property."""
    _reset_latency_cache()
    _reset_memory_cache()
    provider = _SlowProvider(seconds_per_call=0.1)
    clock = {"t": 0.0}

    def fake_monotonic() -> float:
        clock["t"] += 0.1
        return clock["t"]

    monkeypatch.setattr(reranker_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(reranker_mod, "LATENCY_BUDGET_SECONDS", 0.5)
    # headroom vastly exceeds any plausible per-doc memory cost -> memory
    # term never binds.
    _script_memory_term(monkeypatch, per_doc_memory=1.0, headroom=1e12)

    width = get_rerank_width(50, provider)
    assert width == 5, "must match the pre-change time-only formula's result exactly (floor(0.5/0.1)=5)"
    _reset_latency_cache()
    _reset_memory_cache()


def test_oom_repro_reversed_fast_time_tight_memory_caps_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC3: reproduces the fp16 natural-width OOM scenario in reverse — a
    fast measured per-doc TIME (as fp16's extra speed produced) under a
    tight memory ceiling. The pre-change time-only formula would have
    poured that speed entirely into a wide width (CANDIDATE_POOL, the
    exact OOM shape); the memory term must now cap it well below that."""
    _reset_latency_cache()
    _reset_memory_cache()
    provider = _InstantProvider()  # fast per-doc TIME -> old formula picks CANDIDATE_POOL

    old_style_width = get_rerank_width(CANDIDATE_POOL + 25, provider)
    assert old_style_width == CANDIDATE_POOL, "sanity: fast time alone would pick the CANDIDATE_POOL cap"

    _reset_latency_cache()
    _reset_memory_cache()
    # A tight memory ceiling: only enough headroom for 6 candidates at the
    # scripted per-doc cost.
    _script_memory_term(monkeypatch, per_doc_memory=1_000_000.0, headroom=6_000_000.0)
    width = get_rerank_width(CANDIDATE_POOL + 25, provider)

    assert width == 6, f"expected the memory ceiling (floor(6e6/1e6)=6) to cap width, got {width}"
    assert width < old_style_width, "must stay bounded well below what the old time-only formula would have exceeded"
    _reset_latency_cache()
    _reset_memory_cache()


def test_hardware_adaptive_potato_vs_capable_box(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC4: a potato scenario (tight latency budget AND tight memory
    ceiling, on top of a slow measured per-doc time) narrows width; a
    capable-box scenario (generous latency budget AND generous memory, on
    top of a fast measured per-doc time) widens width toward
    CANDIDATE_POOL — both purely from measured inputs, no hardcoded tier
    logic."""
    _reset_latency_cache()
    _reset_memory_cache()
    clock = {"t": 0.0}

    def fake_monotonic() -> float:
        clock["t"] += 0.1
        return clock["t"]

    monkeypatch.setattr(reranker_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(reranker_mod, "LATENCY_BUDGET_SECONDS", 0.4)
    _script_memory_term(monkeypatch, per_doc_memory=1_000_000.0, headroom=3_000_000.0)
    potato_width = get_rerank_width(CANDIDATE_POOL, _SlowProvider(seconds_per_call=0.1))
    assert potato_width < CANDIDATE_POOL, "potato: tight time AND tight memory must narrow width well below the pool cap"

    _reset_latency_cache()
    _reset_memory_cache()
    monkeypatch.setattr(reranker_mod, "LATENCY_BUDGET_SECONDS", 10.0)
    _script_memory_term(monkeypatch, per_doc_memory=1.0, headroom=1e12)
    capable_width = get_rerank_width(CANDIDATE_POOL, _InstantProvider())
    assert capable_width == CANDIDATE_POOL, "capable box: generous time AND generous memory must widen to the pool cap"

    assert capable_width > potato_width
    _reset_latency_cache()
    _reset_memory_cache()


def test_get_rerank_width_skips_memory_term_when_headroom_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safe fallback (I6 / spec Open Reconfirmation): when available_RAM_
    headroom cannot be determined at all (`None` — unsupported platform, or
    every read source failed), get_rerank_width must degrade to the pre-
    Change-3 time-only bound — NEVER assume unlimited headroom, and never
    crash on a `None / per_doc_memory` division."""
    _reset_latency_cache()
    _reset_memory_cache()
    provider = _InstantProvider()
    _script_memory_term(monkeypatch, per_doc_memory=1.0, headroom=None)  # tiny per-doc cost, headroom unknown

    width = get_rerank_width(50, provider)
    assert width == 50, "unknown headroom must skip the memory term entirely, not narrow width"
    _reset_latency_cache()
    _reset_memory_cache()


def test_get_rerank_width_skips_memory_term_when_per_doc_memory_measurement_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-soft: a per-doc memory measurement failure (mirrors `_warm_per_
    doc_latency`'s 0.0 fail-soft contract) must skip the memory term, never
    raise or divide by zero."""
    _reset_latency_cache()
    _reset_memory_cache()
    provider = _InstantProvider()
    _script_memory_term(monkeypatch, per_doc_memory=0.0, headroom=1.0)  # tight headroom, but no memory signal

    width = get_rerank_width(50, provider)
    assert width == 50, "a zero/unmeasured per-doc memory figure must skip the memory term, not divide by zero"
    _reset_latency_cache()
    _reset_memory_cache()


def test_ac5_get_rerank_width_introduces_no_new_numeric_literal() -> None:
    """AC5 (mechanical, AST-based): `get_rerank_width`'s own body
    must contain no numeric literal beyond the two PRE-EXISTING, structural
    ones (`0` for the empty-pool guard, `1` for the `max(1, ...)` floor) and
    the `0.0` "no signal" sentinel comparison (already present pre-Change-3
    for the time term; reused, not duplicated, for the memory term) —
    Change 3's memory bound must be assembled purely from `CANDIDATE_POOL`
    (pre-existing, imported) and the runtime-measured `per_doc_TIME`/
    `per_doc_MEMORY`/`available_RAM_headroom`, never a new inline ratio or
    threshold constant."""
    import ast
    import inspect

    source = inspect.getsource(reranker_mod.get_rerank_width)
    tree = ast.parse(source)
    numeric_literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ]
    assert set(numeric_literals) <= {0, 1}, (
        "get_rerank_width must contain no numeric literal beyond the pre-existing "
        f"empty-pool guard (0), max(1, ...) floor, and 0.0 no-signal sentinel — found: {numeric_literals}"
    )


# --- direct coverage of the new measurement/read helpers --------------------


def test_warm_per_doc_memory_is_cached_not_remeasured_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_memory_cache()
    calls = {"n": 0}

    def _fake_measure(provider, sample_docs=None):
        calls["n"] += 1
        return 5.0

    monkeypatch.setattr(reranker_mod, "_measure_warm_per_doc_memory", _fake_measure)
    provider = _InstantProvider()

    first = reranker_mod._warm_per_doc_memory(provider)
    second = reranker_mod._warm_per_doc_memory(provider)
    assert first == second == 5.0
    assert calls["n"] == 1, "a second call within the recompute interval must not re-measure"
    _reset_memory_cache()


def test_warm_per_doc_memory_measurement_failure_is_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_memory_cache()

    def _boom_measure(provider, sample_docs=None):
        raise RuntimeError("simulated memory measurement failure")

    monkeypatch.setattr(reranker_mod, "_measure_warm_per_doc_memory", _boom_measure)
    assert reranker_mod._warm_per_doc_memory(_InstantProvider()) == 0.0
    _reset_memory_cache()


def test_measure_warm_per_doc_memory_divides_rss_delta_by_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """The measured figure must be `(RSS after the measured batch - RSS
    before it) / _MEMORY_MEASURE_BATCH_SIZE` — proving the per-doc figure is
    derived from a batch RSS delta, not some other arithmetic. Only TWO
    `_current_rss_bytes()` reads are taken (before/after the MEASURED batch
    — the warmup batch is not RSS-bracketed), so a 2-element sequence fully
    determines the result."""
    batch_size = reranker_mod._MEMORY_MEASURE_BATCH_SIZE
    rss_sequence = iter([1_000_000.0, 1_000_000.0 + 500.0 * batch_size])
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: next(rss_sequence))

    result = reranker_mod._measure_warm_per_doc_memory(_InstantProvider())
    assert result == 500.0


def test_measure_warm_per_doc_memory_clamps_negative_delta_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """RSS can legitimately DROP between the before/after read (a GC pass,
    another thread freeing memory) — a negative delta must clamp to 0.0
    ('no memory signal') rather than a negative per-doc figure that would
    make the width formula's floor() division nonsensical."""
    rss_sequence = iter([2_000_000.0, 1_000_000.0])
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: next(rss_sequence))
    assert reranker_mod._measure_warm_per_doc_memory(_InstantProvider()) == 0.0


def test_measure_warm_per_doc_memory_returns_zero_when_rss_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: None)
    assert reranker_mod._measure_warm_per_doc_memory(_InstantProvider()) == 0.0


def test_current_rss_bytes_parses_vmrss_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    status = "VmPeak:\t   50000 kB\nVmRSS:\t   12345 kB\nVmData:\t   9999 kB\n"
    monkeypatch.setattr("builtins.open", mock_open(read_data=status))
    assert reranker_mod._current_rss_bytes() == 12345 * 1024.0


def test_current_rss_bytes_none_on_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert reranker_mod._current_rss_bytes() is None


def test_current_rss_bytes_fail_soft_on_read_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    def _boom(*args, **kwargs):
        raise OSError("simulated /proc/self/status read failure")

    monkeypatch.setattr("builtins.open", _boom)
    assert reranker_mod._current_rss_bytes() is None  # no raise


class _FakeTextFile:
    """Minimal context-manager stand-in for an open text file, used by
    `_fake_open_dispatcher` below — `mock_open` only serves ONE file's
    content per patch, but the cgroup readers open two different paths
    (limit/max then usage/current) per call, so this dispatches by path."""

    def __init__(self, content: str) -> None:
        self._content = content

    def __enter__(self) -> _FakeTextFile:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> str:
        return self._content


def _fake_open_dispatcher(contents: dict[str, str]):
    def _open(path, *args, **kwargs):
        if path not in contents:
            raise FileNotFoundError(path)
        return _FakeTextFile(contents[path])

    return _open


def test_cgroup_v2_headroom_reads_limit_minus_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Process is at the v2 root itself (`0::/`) -> the root's own
    memory.max/memory.current are the ones read."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "builtins.open",
        _fake_open_dispatcher(
            {
                "/proc/self/cgroup": "0::/\n",
                "/sys/fs/cgroup/memory.max": "6000000000\n",
                "/sys/fs/cgroup/memory.current": "1000000000\n",
            }
        ),
    )
    assert reranker_mod._cgroup_v2_memory_headroom_bytes() == 5_000_000_000.0


def test_cgroup_v2_nested_cgroup_reads_process_own_limit_not_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BUG FIX / BITE-CHECK: a real process lives in a NESTED cgroup, not
    at the v2 root — the root has no memory.max/memory.current of its
    own. This fixture deliberately provides NO `/sys/fs/cgroup/memory.max`
    (the fixed-root path the old, buggy reader used) — only the process's
    own nested cgroup's files. The old fixed-root reader would hit
    FileNotFoundError on `/sys/fs/cgroup/memory.max` and return None here;
    the corrected reader resolves the process's own cgroup via
    `/proc/self/cgroup` and finds its cap."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "builtins.open",
        _fake_open_dispatcher(
            {
                "/proc/self/cgroup": "0::/user.slice/app.scope\n",
                "/sys/fs/cgroup/user.slice/app.scope/memory.max": "6000000000\n",
                "/sys/fs/cgroup/user.slice/app.scope/memory.current": "1000000000\n",
            }
        ),
    )
    assert reranker_mod._cgroup_v2_memory_headroom_bytes() == 5_000_000_000.0


def test_cgroup_v2_walk_up_uses_ancestor_cap_when_own_level_is_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BUG FIX / BITE-CHECK: the process's OWN cgroup reads "max"
    (unbounded at that level), but a PARENT cgroup sets a real numeric
    cap. The effective limit must be the parent's cap, found by walking
    up the chain — not "unbounded" (which the old fixed-root reader could
    never even see, since it never looked at the process's own cgroup at
    all) and not the root (not provided here, so it would raise
    FileNotFoundError while walking, which the walk must tolerate)."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "builtins.open",
        _fake_open_dispatcher(
            {
                "/proc/self/cgroup": "0::/user.slice/app.scope\n",
                "/sys/fs/cgroup/user.slice/app.scope/memory.max": "max\n",
                "/sys/fs/cgroup/user.slice/app.scope/memory.current": "1000000000\n",
                "/sys/fs/cgroup/user.slice/memory.max": "4000000000\n",
                # root's memory.max deliberately absent -> must be tolerated
            }
        ),
    )
    assert reranker_mod._cgroup_v2_memory_headroom_bytes() == 3_000_000_000.0


def test_cgroup_v2_all_max_up_chain_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every level on the chain (own cgroup, parent, root) reads "max" ->
    no numeric cap anywhere -> None, and usage is never even read since
    there is nothing to subtract it from."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "builtins.open",
        _fake_open_dispatcher(
            {
                "/proc/self/cgroup": "0::/user.slice/app.scope\n",
                "/sys/fs/cgroup/user.slice/app.scope/memory.max": "max\n",
                "/sys/fs/cgroup/user.slice/memory.max": "max\n",
                "/sys/fs/cgroup/memory.max": "max\n",
            }
        ),
    )
    assert reranker_mod._cgroup_v2_memory_headroom_bytes() is None


def test_cgroup_v2_unbounded_max_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "builtins.open",
        _fake_open_dispatcher(
            {
                "/proc/self/cgroup": "0::/\n",
                "/sys/fs/cgroup/memory.max": "max\n",
            }
        ),
    )
    assert reranker_mod._cgroup_v2_memory_headroom_bytes() is None


def test_cgroup_v2_missing_proc_self_cgroup_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-soft: `/proc/self/cgroup` itself absent (e.g. non-Linux-like
    sandbox) -> None, never a crash."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("builtins.open", _fake_open_dispatcher({}))
    assert reranker_mod._cgroup_v2_memory_headroom_bytes() is None


def test_cgroup_v2_garbage_memory_max_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-soft: an unparsable memory.max value -> None, never a crash."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "builtins.open",
        _fake_open_dispatcher(
            {
                "/proc/self/cgroup": "0::/\n",
                "/sys/fs/cgroup/memory.max": "not-a-number\n",
            }
        ),
    )
    assert reranker_mod._cgroup_v2_memory_headroom_bytes() is None


def test_cgroup_v1_headroom_reads_limit_minus_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Process is at the v1 memory-controller root itself -> the root's
    own limit_in_bytes/usage_in_bytes are the ones read."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "builtins.open",
        _fake_open_dispatcher(
            {
                "/proc/self/cgroup": "5:memory:/\n",
                "/sys/fs/cgroup/memory/memory.limit_in_bytes": "5500000000\n",
                "/sys/fs/cgroup/memory/memory.usage_in_bytes": "5000000000\n",
            }
        ),
    )
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 500_000_000.0


def test_cgroup_v1_nested_cgroup_reads_process_own_limit_not_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BUG FIX / BITE-CHECK: a real process lives in a NESTED v1 cgroup.
    This fixture deliberately provides NO `/sys/fs/cgroup/memory/memory.
    limit_in_bytes` (the fixed-root path the old, buggy reader used) —
    only the process's own nested cgroup's files. The old fixed-root
    reader would hit FileNotFoundError there and return None; the
    corrected reader resolves the process's own cgroup via `/proc/self/
    cgroup`'s `memory` controller line and finds its cap."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "builtins.open",
        _fake_open_dispatcher(
            {
                "/proc/self/cgroup": "5:memory:/user.slice/app.scope\n",
                "/sys/fs/cgroup/memory/user.slice/app.scope/memory.limit_in_bytes": "5500000000\n",
                "/sys/fs/cgroup/memory/user.slice/app.scope/memory.usage_in_bytes": "5000000000\n",
            }
        ),
    )
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 500_000_000.0


def test_cgroup_v1_hybrid_proc_self_cgroup_picks_memory_controller_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A combined-controller hierarchy line (e.g. `cpu,memory`) and other,
    unrelated controller lines are both present in `/proc/self/cgroup` —
    the v1 reader must pick the line that actually lists `memory`."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "builtins.open",
        _fake_open_dispatcher(
            {
                "/proc/self/cgroup": (
                    "11:pids:/user.slice/app.scope\n7:cpu,memory:/user.slice/app.scope\n"
                ),
                "/sys/fs/cgroup/memory/user.slice/app.scope/memory.limit_in_bytes": "5500000000\n",
                "/sys/fs/cgroup/memory/user.slice/app.scope/memory.usage_in_bytes": "5000000000\n",
            }
        ),
    )
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 500_000_000.0


def test_cgroup_v1_unbounded_sentinel_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1 has no explicit "unbounded" marker like v2's "max" — an
    unbounded limit reads back as the kernel's own huge sentinel value,
    which must be treated as unbounded -> None, never a fabricated huge
    headroom."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "builtins.open",
        _fake_open_dispatcher(
            {
                "/proc/self/cgroup": "5:memory:/\n",
                "/sys/fs/cgroup/memory/memory.limit_in_bytes": (
                    f"{reranker_mod._CGROUP_V1_UNBOUNDED_SENTINEL}\n"
                ),
            }
        ),
    )
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() is None


def test_cgroup_v1_missing_proc_self_cgroup_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-soft: `/proc/self/cgroup` itself absent -> None, never a crash."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("builtins.open", _fake_open_dispatcher({}))
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() is None


def test_cgroup_v1_garbage_limit_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-soft: an unparsable limit_in_bytes value -> None, never a crash."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "builtins.open",
        _fake_open_dispatcher(
            {
                "/proc/self/cgroup": "5:memory:/\n",
                "/sys/fs/cgroup/memory/memory.limit_in_bytes": "garbage\n",
            }
        ),
    )
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() is None


def test_proc_meminfo_available_bytes_parses_kb_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    meminfo = "MemTotal:       16333000 kB\nMemFree:         2000000 kB\nMemAvailable:    5000000 kB\n"
    monkeypatch.setattr("builtins.open", mock_open(read_data=meminfo))
    assert reranker_mod._proc_meminfo_available_bytes() == 5_000_000 * 1024.0


def test_proc_meminfo_available_bytes_none_on_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert reranker_mod._proc_meminfo_available_bytes() is None


def test_available_ram_headroom_prefers_cgroup_v2_over_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: 111.0)
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: 222.0)
    monkeypatch.setattr(reranker_mod, "_proc_meminfo_available_bytes", lambda: 333.0)
    assert reranker_mod._available_ram_headroom_bytes() == 111.0


def test_available_ram_headroom_falls_back_to_cgroup_v1_when_v2_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: 222.0)
    monkeypatch.setattr(reranker_mod, "_proc_meminfo_available_bytes", lambda: 333.0)
    assert reranker_mod._available_ram_headroom_bytes() == 222.0


def test_available_ram_headroom_falls_back_to_meminfo_when_no_cgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_proc_meminfo_available_bytes", lambda: 333.0)
    assert reranker_mod._available_ram_headroom_bytes() == 333.0


def test_available_ram_headroom_none_when_everything_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safe-degrade case: no cgroup limit and an unreadable/absent
    /proc/meminfo (or a non-Linux platform) -> None, which get_rerank_width
    must treat as 'skip the memory term', never as unlimited headroom."""
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_proc_meminfo_available_bytes", lambda: None)
    assert reranker_mod._available_ram_headroom_bytes() is None


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
