"""Tests for brain.memory.reranker — the #231 cross-encoder reranker
provider + process-wide caches + anchor normalization (the per-message
width fit is covered in test_rerank_cost_model.py).

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
from brain.memory.reranker import (
    ANCHOR_POOL,
    ANCHOR_SPLIT_DIVISOR,
    K_MIN,
    AnchorNormalizationResult,
    FakeRerankerProvider,
    P,
    RerankerProvider,
    _default_latency_budget_seconds,
    _detect_avx2,
    _reset_reranker_provider_cache,
    build_reranker_provider,
    normalize_against_anchors,
    normalize_bundled_pairs_against_anchors,
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
    override of 0.25s, so the two are clearly distinguishable: only the
    override value can produce the asserted width."""
    import brain.tunables as tunables_mod

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables_mod._reset_for_tests()
    (tmp_path / "tunables.json").write_text(
        json.dumps({"defaults": {}, "overrides": {"reranker.latency_budget_seconds": 0.25}}),
        encoding="utf-8",
    )
    # The auto-detected default this process would otherwise use (as if
    # AVX2 were detected) — the override must win over THIS, not just over
    # some arbitrary fallback.
    monkeypatch.setattr(reranker_mod, "LATENCY_BUDGET_SECONDS", 2.0)

    # Name-recall fix R1: the budget feeds the per-message width fit. A
    # measured cost of 1/1024 s per padded token (overhead 0) and pairs
    # of 8 tokens make each document cost 1/128 s, so the 0.25 s
    # override fits 32 documents (24 real + 8 anchors) where the 2.0 s
    # default would fit the whole 50-candidate pool.
    class _EightTokenPairs(FakeRerankerProvider):
        def pair_token_lengths(self, query, documents):
            return [8 for _ in documents]

        def model_id(self) -> str:
            return "override-test-provider"

    reranker_mod._record_rerank_cost("override-test-provider", 1024, 1.0, None)

    outcome = reranker_mod.rerank_for_recall(
        _EightTokenPairs(default=0.0), "q", [f"doc-{i}" for i in range(50)]
    )
    # 24 real + min(8, 24 // 2) = 8 anchors = 32 documents = 0.25 s; 25 real
    # would be 33 documents. The default budget would have given 50.
    assert outcome.width == 24, f"expected the override (0.25s) to win, got width={outcome.width}"
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
    /proc/meminfo (or a non-Linux platform) -> None, which the width fit
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


# ---------------------------------------------------------------------------
# normalize_against_anchors — F2b (#276) per-query anchor-median
# normalization. Since name-recall fix R1 (S23) the anchors come ON TOP of
# the real candidates: the caller passes exactly the fitted real prefix,
# every one of those documents is scored, and `k = min(P, real // 2)`
# anchors are appended after them. All offline, FakeRerankerProvider.
# ---------------------------------------------------------------------------


class _RecordingProvider(RerankerProvider):
    """Wraps a FakeRerankerProvider and records the exact `documents` list
    passed to each `rerank()` call — used to prove the ONE-combined-call
    shape and that no anchor content leaks outside that one call."""

    def __init__(self, scores: dict[str, float], default: float = 0.0) -> None:
        self._fake = FakeRerankerProvider(scores=scores, default=default)
        self.calls: list[list[str]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append(list(documents))
        return self._fake.rerank(query, documents)

    def model_id(self) -> str:
        return "recording-fake"


def _real_docs(n: int) -> list[str]:
    """`n` distinct, deterministic placeholder real-candidate documents."""
    return [f"real-candidate-{i}" for i in range(n)]


def test_k_min_and_p_are_named_module_constants() -> None:
    """Sanity/I7 check: P is DERIVED from ANCHOR_POOL's own length (never a
    re-typed literal), K_MIN is the named meaningful-median floor, and the
    "anchors at most half" split ratio is its own named divisor."""
    assert K_MIN == 2
    assert P == len(ANCHOR_POOL) == 8
    assert ANCHOR_SPLIT_DIVISOR == 2
    assert reranker_mod.P == P, "module-level P must be the single source of truth"


def test_normalize_against_anchors_returns_an_anchor_normalization_result() -> None:
    provider = _RecordingProvider({}, default=0.0)
    result = normalize_against_anchors(provider, "q", _real_docs(5))
    assert isinstance(result, AnchorNormalizationResult)


def test_normalized_score_equals_raw_minus_median_of_anchor_scores() -> None:
    real = _real_docs(5)  # k = min(8, 5 // 2) = 2 anchors on top
    scores = {
        real[0]: 5.0,
        real[1]: 3.0,
        ANCHOR_POOL[0]: 10.0,
        ANCHOR_POOL[1]: 12.0,  # median(10.0, 12.0) == 11.0
    }
    provider = _RecordingProvider(scores, default=0.0)

    result = normalize_against_anchors(provider, "some query", real)

    assert result.did_normalize is True
    assert result.real_width == 5
    assert result.anchor_count == 2
    assert result.scores == [5.0 - 11.0, 3.0 - 11.0, -11.0, -11.0, -11.0], (
        "normalized_score must equal raw_score - median(anchor_scores), for EVERY real document"
    )


def test_same_query_different_anchor_scores_yields_different_normalized_value() -> None:
    """The offset is computed FRESH on every call, never cached or reused,
    even for the identical query string."""
    real = _real_docs(5)
    provider_a = _RecordingProvider({real[0]: 5.0, ANCHOR_POOL[0]: 10.0, ANCHOR_POOL[1]: 12.0})
    provider_b = _RecordingProvider({real[0]: 5.0, ANCHOR_POOL[0]: 1.0, ANCHOR_POOL[1]: 3.0})

    result_a = normalize_against_anchors(provider_a, "identical query text", real)
    result_b = normalize_against_anchors(provider_b, "identical query text", real)

    assert result_a.scores != result_b.scores
    assert result_b.scores[0] == 5.0 - 2.0  # median(1.0, 3.0) == 2.0


def test_one_combined_call_sends_every_real_document_plus_k_anchors_on_top() -> None:
    real = _real_docs(7)  # k = min(8, 7 // 2) = 3
    provider = _RecordingProvider({}, default=0.0)

    result = normalize_against_anchors(provider, "q", real)

    assert len(provider.calls) == 1, "must be exactly ONE combined rerank() call, not two"
    (sent_docs,) = provider.calls
    assert len(sent_docs) == 7 + 3, "anchors come ON TOP of the real documents (S23), never out of them"
    assert sent_docs[:7] == real, "every real document is scored, in the caller's order, at the FRONT"
    assert sent_docs[7:] == ANCHOR_POOL[:3], "the k anchors are the ANCHOR_POOL prefix, appended after"
    assert not set(ANCHOR_POOL[3:]) & set(sent_docs), "only the k-sized anchor prefix may appear"
    assert result.real_width == 7
    assert len(result.scores) == 7, "one score per real document; anchor scores are never returned"


@pytest.mark.parametrize("real_count", [1, 2, 3])
def test_near_degenerate_real_count_noops_to_raw_scores(real_count: int) -> None:
    """Defensive no-op (recall callers never reach it: they rerank at least
    5 real candidates, so k >= 2): too few real documents for an anchor
    median -> raw scores, no anchors, `did_normalize=False`."""
    real = _real_docs(real_count)
    provider = _RecordingProvider({doc: float(i) for i, doc in enumerate(real)})

    result = normalize_against_anchors(provider, "q", real)

    assert result.did_normalize is False
    assert result.anchor_count == 0
    assert result.real_width == real_count
    (sent_docs,) = provider.calls
    assert sent_docs == real, "no-op call must be real-candidates-only"
    assert result.scores == [float(i) for i in range(real_count)], "no-op scores must be RAW"


@pytest.mark.parametrize("real_count", [4, 5, 7, 20, 50])
def test_k_equals_min_pool_cap_and_floor_half_real_count(real_count: int) -> None:
    """`k = min(P, real // ANCHOR_SPLIT_DIVISOR)` (S23, the spec's own
    arithmetic, independent of the implementation), anchors on top: the
    call sends real + k documents."""
    provider = _RecordingProvider({}, default=0.0)

    result = normalize_against_anchors(provider, "q", _real_docs(real_count))

    formula_k = min(P, real_count // ANCHOR_SPLIT_DIVISOR)
    (sent_docs,) = provider.calls
    assert formula_k >= K_MIN
    assert result.did_normalize is True
    assert result.anchor_count == formula_k == reranker_mod.anchor_count(real_count)
    assert len(sent_docs) == real_count + formula_k
    assert result.real_width == real_count


def test_k_grows_monotonically_with_real_count() -> None:
    counts = [1, 2, 3, 4, 5, 6, 7, 10, 13, 16, 20, 30, 50]
    ks = []
    for n in counts:
        provider = _RecordingProvider({}, default=0.0)
        normalize_against_anchors(provider, "q", _real_docs(n))
        (sent_docs,) = provider.calls
        ks.append(len(sent_docs) - n)
    assert ks == sorted(ks), f"k must be non-decreasing as the real count grows: {ks}"
    assert ks[0] == 0, "the smallest count must no-op (k below K_MIN)"
    assert ks[-1] == P, "a wide-enough real count must saturate k at the curated pool size P"


def test_result_carries_the_rerank_timed_measurements() -> None:
    """The combined call goes through `rerank_timed` (a recall-time rerank,
    S24), and its seconds / RSS delta come back on the result for the cost
    model."""

    class _Timed(_RecordingProvider):
        def rerank_timed(self, query, documents):
            return self.rerank(query, documents), 0.25, 4096.0

    result = normalize_against_anchors(_Timed({}, default=0.0), "q", _real_docs(5))

    assert result.seconds == 0.25
    assert result.rss_delta_bytes == 4096.0


def test_ac11_normalize_against_anchors_has_no_bare_numeric_literal() -> None:
    """AC11 (mechanical, AST-level): the mechanism function contains NO
    numeric literal at all — `k` is derived from the named `P` / `K_MIN` /
    `ANCHOR_SPLIT_DIVISOR` module constants (via `anchor_count`), never a
    re-typed number."""
    import ast
    import inspect

    for fn in (reranker_mod.normalize_against_anchors, reranker_mod.anchor_count):
        tree = ast.parse(inspect.getsource(fn))
        numeric_literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ]
        assert numeric_literals == [], f"{fn.__name__}: found literal(s) {numeric_literals}"


# ---------------------------------------------------------------------------
# normalize_bundled_pairs_against_anchors — F2b §5b (#276 inc3): the
# off-hot-path bundled-pair normalization used by floor_calibration.py's
# cold-start/bootstrap sources. Scores
# the FULL curated anchor pool (never a `k`-subset) once per pair, since a
# different query means the pairs cannot be batched into one rerank() call.
# ---------------------------------------------------------------------------


def test_normalize_bundled_pairs_against_anchors_scores_full_pool_one_call_per_pair() -> None:
    pairs = [("query-a", "doc-a"), ("query-b", "doc-b")]
    scores = {"doc-a": 5.0, "doc-b": -2.0, **dict.fromkeys(ANCHOR_POOL, 1.0)}
    provider = _RecordingProvider(scores)

    result = normalize_bundled_pairs_against_anchors(provider, pairs)

    assert len(provider.calls) == 2, "one combined rerank() call PER pair — different query each time"
    for sent_docs, (_query, doc) in zip(provider.calls, pairs, strict=True):
        assert sent_docs[0] == doc, "the real doc occupies the FRONT of each combined call"
        assert sent_docs[1:] == ANCHOR_POOL, (
            "must score the FULL anchor pool (never a k-subset) — off the hot path, no latency budget"
        )
    assert result == [5.0 - 1.0, -2.0 - 1.0], "normalized_score = raw_score - median(anchor_scores)"


def test_normalize_bundled_pairs_against_anchors_never_leaks_anchor_scores_into_the_result() -> None:
    """Only ONE normalized score per pair is ever returned — the anchor
    scores feed the median offset and nothing else."""
    pairs = [("q", "the-real-doc")]
    scores = {"the-real-doc": 10.0, **dict.fromkeys(ANCHOR_POOL, -1.0)}
    provider = _RecordingProvider(scores)

    result = normalize_bundled_pairs_against_anchors(provider, pairs)

    assert len(result) == 1
    assert result[0] == pytest.approx(10.0 - (-1.0))
