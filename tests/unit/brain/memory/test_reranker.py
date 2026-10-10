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
from brain.memory.store import MemoryStore

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

    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    reranker_mod._record_rerank_cost("override-test-provider", store, 1024, 1.0, None)

    outcome = reranker_mod.rerank_for_recall(
        _EightTokenPairs(default=0.0), "q", [f"doc-{i}" for i in range(50)], store=store
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


def test_the_third_timed_value_is_the_peak_growth_read_by_the_peak_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """S220: `rerank_timed` returns the call's PEAK growth (peak_rss), never an RSS difference."""
    from brain.memory import peak_rss

    events: list[str] = []
    monkeypatch.setattr(peak_rss, "begin", lambda: events.append("begin") or 1000.0)
    monkeypatch.setattr(peak_rss, "growth_since", lambda before: events.append(f"growth since {before}") or 4096.0)
    provider = FakeRerankerProvider({})
    original = provider.rerank
    provider.rerank = lambda q, d: (events.append("rerank"), original(q, d))[1]
    scores, seconds, growth = provider.rerank_timed("q", ["a", "b"])
    assert growth == 4096.0 and events == ["begin", "rerank", "growth since 1000.0"]
    assert len(scores) == 2 and seconds >= 0.0


def test_an_unreadable_peak_is_none_in_the_timed_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.memory import peak_rss

    monkeypatch.setattr(peak_rss, "begin", lambda: None)
    assert FakeRerankerProvider({}).rerank_timed("q", ["a"])[2] is None


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


# ---- S233 + S235: the cgroup headroom counts clean unmapped file cache as reclaimable and reads
# every limited level of the chain against that level's own usage ----

_OWN = "user.slice/app.scope"
_PAR = "user.slice"
_GRAND = ""  # the root level of the synthetic chain
_V2_ROOT = "/sys/fs/cgroup"
_V1_OWN = "/sys/fs/cgroup/memory/user.slice/app.scope"
_REMOVE = object()
# Lines that would give a DIFFERENT figure if the wrong key were read.
_V2_DECOYS = "inactive_file 111\nactive_file 222\nanon 333\ntotal_cache 444\nfile_thp 555\nshmem_thp 666\nfile_mapped_x 777\n"
_V2_KEYS = ("file", "file_mapped", "file_dirty", "file_writeback", "shmem")
_V1_KEYS = ("total_cache", "total_mapped_file", "total_dirty", "total_writeback", "total_shmem")
# base page-cache terms: reclaimable = 700 - 50 - 30 - 20 - 100 = 500 MB
_BASE = {"cache": 700_000_000, "mapped": 50_000_000, "dirty": 30_000_000, "writeback": 20_000_000, "shmem": 100_000_000}


def _stat(
    cache: int = 700_000_000,
    mapped: int = 50_000_000,
    dirty: int = 30_000_000,
    writeback: int = 20_000_000,
    shmem: int = 100_000_000,
    *,
    keys: tuple[str, ...] = _V2_KEYS,
    decoys: str = _V2_DECOYS,
    override: dict | None = None,
) -> str:
    """A memory.stat with the five page-cache lines (in `keys` names), decoys, and optional per-key
    raw overrides (`_REMOVE` drops the line)."""
    out = decoys
    for key, value in zip(keys, (cache, mapped, dirty, writeback, shmem), strict=True):
        if override and key in override:
            if override[key] is _REMOVE:
                continue
            out += f"{key} {override[key]}\n".replace(" \n", "\n")
        else:
            out += f"{key} {value}\n"
    return out


def _lvl(max_: str | None = None, current: str | int | None = None, stat: str | None = None) -> dict:
    return {"max": max_, "current": current, "stat": stat}


def _tree(levels: dict[str, dict], *, own: str = _OWN) -> dict[str, str]:
    """Synthetic v2 files for a chain; `levels` maps a relative directory to its max/current/stat (None = file absent)."""
    files = {"/proc/self/cgroup": f"0::/{own}\n"}
    for rel, lv in levels.items():
        base = f"{_V2_ROOT}/{rel}" if rel else _V2_ROOT
        if lv["max"] is not None:
            files[f"{base}/memory.max"] = f"{lv['max']}\n"
        if lv["current"] is not None:
            files[f"{base}/memory.current"] = f"{lv['current']}\n"
        if lv["stat"] is not None:
            files[f"{base}/memory.stat"] = lv["stat"]
    return files


def _one_level(stat: str | None, *, limit: str = "6000000000", current: str = "1000000000") -> dict[str, str]:
    return _tree({_OWN: _lvl(limit, current, stat)})


def _v1_files(stat: str | None, *, limit: str = "5500000000", usage: str = "5000000000") -> dict[str, str]:
    files = {
        "/proc/self/cgroup": "5:memory:/user.slice/app.scope\n",
        f"{_V1_OWN}/memory.limit_in_bytes": f"{limit}\n",
        f"{_V1_OWN}/memory.usage_in_bytes": f"{usage}\n",
    }
    if stat is not None:
        files[f"{_V1_OWN}/memory.stat"] = stat
    return files


_V1_DECOYS = "cache 111\nmapped_file 222\ndirty 333\nwriteback 444\nshmem 555\ninactive_file 666\ntotal_inactive_file 777\n"


def _v1_stat(**kw) -> str:
    return _stat(keys=_V1_KEYS, decoys=_V1_DECOYS, **kw)


def _use_files(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, str],
    *,
    raising: tuple[str, ...] = (),
    opened: list[str] | None = None,
) -> None:
    """Install the fake `open`; a path ending with any of `raising` raises PermissionError; every opened path is recorded."""
    inner = _fake_open_dispatcher(files)

    def _open(path, *args, **kwargs):
        if opened is not None:
            opened.append(str(path))
        if any(str(path).endswith(r) for r in raising):
            raise PermissionError(path)
        return inner(path, *args, **kwargs)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("builtins.open", _open)


def _v2() -> float | None:
    return reranker_mod._cgroup_v2_memory_headroom_bytes()


def _warnings(caplog: pytest.LogCaptureFixture, needle: str) -> list:
    return [r for r in caplog.records if r.levelname == "WARNING" and needle in r.getMessage()]


# ---- G1/G2/G3: the formula, what is and is not reclaimable, the clamps (one level) ----


def test_g1_v2_headroom_subtracts_cache_minus_mapped_dirty_writeback_shmem(monkeypatch: pytest.MonkeyPatch) -> None:
    """limit 6.0 GB, current 1.0 GB, reclaimable 700-50-30-20-100 = 500 MB -> 5.5 GB; decoy lines would change it."""
    _use_files(monkeypatch, _one_level(_stat()))
    assert _v2() == 5_500_000_000.0


@pytest.mark.parametrize(
    ("which", "sign"), [("cache", 1), ("mapped", -1), ("dirty", -1), ("writeback", -1), ("shmem", -1)]
)
def test_g1_each_of_the_five_terms_moves_the_figure_by_exactly_its_size(
    monkeypatch: pytest.MonkeyPatch, which: str, sign: int
) -> None:
    """Bump one term by 10 MB: more cached file data lifts the headroom, more mapped/dirty/writeback/shmem lowers it."""
    kw = dict(_BASE)
    kw[which] += 10_000_000
    _use_files(
        monkeypatch,
        _one_level(
            _stat(
                cache=kw["cache"], mapped=kw["mapped"], dirty=kw["dirty"], writeback=kw["writeback"], shmem=kw["shmem"]
            )
        ),
    )
    assert _v2() == 5_500_000_000.0 + sign * 10_000_000


def test_g2_clean_unmapped_active_cache_counts_as_reclaimable(monkeypatch: pytest.MonkeyPatch) -> None:
    """800 MB of clean, unmapped cache the kernel has marked ACTIVE (inactive_file 0): all reclaimable."""
    stat = _stat(cache=800_000_000, mapped=0, dirty=0, writeback=0, shmem=0, decoys="inactive_file 0\nactive_file 800000000\n")
    _use_files(monkeypatch, _one_level(stat))
    assert _v2() == 5_800_000_000.0


def test_g2_a_mapped_model_stays_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_files(monkeypatch, _one_level(_stat(cache=900_000_000, mapped=900_000_000, dirty=0, writeback=0, shmem=0)))
    assert _v2() == 5_000_000_000.0


@pytest.mark.parametrize("term", ["dirty", "writeback", "shmem"])
def test_g2_dirty_writeback_and_tmpfs_pages_stay_counted(monkeypatch: pytest.MonkeyPatch, term: str) -> None:
    """500 MB cached, all of it in one not-reclaimable term -> nothing is given back."""
    kw = {"mapped": 0, "dirty": 0, "writeback": 0, "shmem": 0, term: 500_000_000}
    _use_files(monkeypatch, _one_level(_stat(cache=500_000_000, **kw)))
    assert _v2() == 5_000_000_000.0


def test_g3_reclaimable_larger_than_the_usage_is_capped_at_the_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_files(monkeypatch, _one_level(_stat(cache=5_000_000_000, mapped=0, dirty=0, writeback=0, shmem=0)))
    assert _v2() == 6_000_000_000.0


def test_g3_terms_larger_than_the_cache_floor_the_reclaimable_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """mapped+dirty+writeback+shmem above `file` must not add to the usage (a negative reclaimable)."""
    _use_files(monkeypatch, _one_level(_stat(cache=100_000_000, mapped=300_000_000)))
    assert _v2() == 5_000_000_000.0


def test_g3_usage_beyond_the_limit_after_the_subtraction_is_zero_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    stat = _stat(cache=1_000_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    _use_files(monkeypatch, _one_level(stat, limit="1000000000", current="3000000000"))
    assert _v2() == 0.0


def test_g3_the_clamps_hold_at_an_ancestor_level(monkeypatch: pytest.MonkeyPatch) -> None:
    big = _stat(cache=9_000_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    tree = _tree(
        {_OWN: _lvl("max", 1_000_000_000, _stat()), _PAR: _lvl(4_000_000_000, 3_000_000_000, big)}
    )
    _use_files(monkeypatch, tree)
    assert _v2() == 4_000_000_000.0  # reclaimable capped at the ancestor's own usage: headroom = its limit
    tree = _tree({_OWN: _lvl("max", 1_000_000_000, _stat()), _PAR: _lvl(4_000_000_000, 5_000_000_000, _stat(cache=0, mapped=0, dirty=0, writeback=0, shmem=0))})
    _use_files(monkeypatch, tree)
    assert _v2() == 0.0  # the ancestor is over its limit: never negative


# ---- G4/G5: degrade to "subtract nothing", exact keys ----

_BAD_VALUES = ["abc", "-5", "+5", "1_000", "nan", "inf", "infinity", "", "²", "9" * 400, "1.5"]


@pytest.mark.parametrize("key", _V2_KEYS)
@pytest.mark.parametrize("raw", [*_BAD_VALUES, _REMOVE])
def test_g4_any_unusable_page_cache_line_subtracts_nothing(monkeypatch: pytest.MonkeyPatch, key: str, raw: object) -> None:
    """Every degrade input, on any ONE of the five lines, gives limit - current (5.0 GB): never None, never a raise."""
    _use_files(monkeypatch, _one_level(_stat(override={key: raw})))
    assert _v2() == 5_000_000_000.0


@pytest.mark.parametrize("stat", [None, ""])
def test_g4_absent_or_empty_stat_subtracts_nothing(monkeypatch: pytest.MonkeyPatch, stat: str | None) -> None:
    _use_files(monkeypatch, _one_level(stat))
    assert _v2() == 5_000_000_000.0


@pytest.mark.parametrize("key", _V2_KEYS)
def test_g5_a_look_alike_key_is_never_read_as_one_of_the_five(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    """The line for `key` is replaced by a look-alike (suffix, prefix): the term is missing -> nothing subtracted."""
    for look_alike in (f"\n{key}_x ", f"\ntotal_{key} "):
        stat = _stat().replace(f"\n{key} ", look_alike, 1)
        assert stat != _stat()
        _use_files(monkeypatch, _one_level(stat))
        assert _v2() == 5_000_000_000.0


def test_g5_the_first_line_of_a_repeated_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_files(monkeypatch, _one_level(_stat() + "file 1\nfile_mapped 99999999999\n"))
    assert _v2() == 5_500_000_000.0


def test_g4_unreadable_own_stat_subtracts_nothing_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _use_files(monkeypatch, _one_level(_stat()), raising=("/memory.stat",))
    with caplog.at_level("DEBUG"):
        assert _v2() == 5_000_000_000.0
        assert _v2() == 5_000_000_000.0
    warns = _warnings(caplog, "memory.stat")
    assert len(warns) == 1
    assert warns[0].exc_info is not None  # the traceback comes with the first warning


def test_g4_absent_stat_file_is_silent(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _use_files(monkeypatch, _one_level(None))
    with caplog.at_level("DEBUG"):
        assert _v2() == 5_000_000_000.0
    assert [r for r in caplog.records if "memory.stat" in r.getMessage()] == []


def test_g4_a_stat_missing_a_line_or_with_garbage_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _use_files(monkeypatch, _one_level(_stat(override={"shmem": _REMOVE})))
    with caplog.at_level("DEBUG"):
        _v2()
        _v2()
    assert len(_warnings(caplog, "memory.stat")) == 1
    caplog.clear()
    reranker_mod._ram_figure_warned.clear()
    _use_files(monkeypatch, _one_level(_stat(override={"file": "abc"})))
    with caplog.at_level("DEBUG"):
        _v2()
        _v2()
    assert len(_warnings(caplog, "memory.stat")) == 1


def test_g4_an_ancestor_stat_that_cannot_be_used_subtracts_nothing_and_does_not_borrow_the_own_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Own stat gives back 500 MB; the parent's stat is garbage -> the parent level is limit - current (0.8 GB), not 1.3 GB."""
    for parent_stat in (None, _stat(override={"file": "abc"}), _stat(override={"shmem": _REMOVE})):
        tree = _tree(
            {_OWN: _lvl("max", 1_000_000_000, _stat()), _PAR: _lvl(4_000_000_000, 3_200_000_000, parent_stat)}
        )
        _use_files(monkeypatch, tree)
        assert _v2() == 800_000_000.0
    tree = _tree({_OWN: _lvl("max", 1_000_000_000, _stat()), _PAR: _lvl(4_000_000_000, 3_200_000_000, _stat())})
    _use_files(monkeypatch, tree, raising=(f"{_PAR}/memory.stat",))
    assert _v2() == 800_000_000.0


# ---- G6/G7/G8: which level binds, and with whose usage ----


def test_g6_the_own_level_binds_with_its_own_counters_and_the_other_levels_stats_are_decoys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _tree(
        {
            _OWN: _lvl(3_000_000_000, 1_000_000_000, _stat()),  # H = 3.0 - (1.0 - 0.5) = 2.5 GB
            _PAR: _lvl(6_000_000_000, 1_500_000_000, _stat(cache=900_000_000, mapped=0, dirty=0, writeback=0, shmem=0)),
            _GRAND: _lvl(None, None, _stat(cache=999_000_000, mapped=0, dirty=0, writeback=0, shmem=0)),
        }
    )
    _use_files(monkeypatch, tree)
    assert _v2() == 2_500_000_000.0


def test_g7_an_ancestor_that_holds_the_cap_is_measured_by_its_own_usage_and_stat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Own cap "max"; the parent caps at 4.0 GB and its usage (siblings included) is 3.2 GB with 400 MB reclaimable:
    4.0 - (3.2 - 0.4) = 1.2 GB. The own level (1.0 GB used, 300 MB reclaimable) would give 3.3 GB."""
    parent = _stat(cache=400_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    own = _stat(cache=300_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    root = _stat(cache=800_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    tree = _tree(
        {
            _OWN: _lvl("max", 1_000_000_000, own),
            _PAR: _lvl(4_000_000_000, 3_200_000_000, parent),
            _GRAND: _lvl(None, None, root),
        }
    )
    _use_files(monkeypatch, tree)
    assert _v2() == 1_200_000_000.0


def test_g7b_a_real_format_memory_stat_is_read_at_the_own_and_at_an_ancestor_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """All 71 key names of a real cgroup-v2 memory.stat, in the kernel's order, numbers synthetic."""
    real_keys = (
        "anon file kernel kernel_stack pagetables sec_pagetables percpu sock vmalloc shmem zswap zswapped file_mapped "
        "file_dirty file_writeback swapcached anon_thp file_thp shmem_thp inactive_anon active_anon inactive_file "
        "active_file unevictable slab_reclaimable slab_unreclaimable slab hugetlb workingset_refault_anon "
        "workingset_refault_file workingset_activate_anon workingset_activate_file workingset_restore_anon "
        "workingset_restore_file workingset_nodereclaim pgdemote_kswapd pgdemote_direct pgdemote_khugepaged "
        "pgdemote_proactive pgpromote_success pgscan pgsteal pswpin pswpout pgscan_kswapd pgscan_direct "
        "pgscan_khugepaged pgscan_proactive pgsteal_kswapd pgsteal_direct pgsteal_khugepaged pgsteal_proactive "
        "pgfault pgmajfault pgrefill pgactivate pgdeactivate pglazyfree pglazyfreed swpin_zero swpout_zero zswpin "
        "zswpout zswpwb thp_fault_alloc thp_collapse_alloc thp_swpout thp_swpout_fallback numa_pages_migrated "
        "numa_pte_updates numa_hint_faults"
    ).split()
    assert len(real_keys) == 71

    def real(cache: int, mapped: int, dirty: int, wb: int, shmem: int) -> str:
        fixed = {"file": cache, "file_mapped": mapped, "file_dirty": dirty, "file_writeback": wb, "shmem": shmem}
        return "".join(f"{k} {fixed.get(k, 7_000_000 + i)}\n" for i, k in enumerate(real_keys))

    _use_files(monkeypatch, _one_level(real(817_922_048, 197_251_072, 4096, 0, 45_297_664), current="1500000000"))
    assert _v2() == 6_000_000_000.0 - (1_500_000_000.0 - (817_922_048 - 197_251_072 - 4096 - 45_297_664))
    tree = _tree(
        {
            _OWN: _lvl("max", 1_000_000_000, real(1, 1, 1, 1, 1)),
            _PAR: _lvl(4_000_000_000, 3_000_000_000, real(900_000_000, 100_000_000, 0, 0, 50_000_000)),
        }
    )
    _use_files(monkeypatch, tree)
    assert _v2() == 4_000_000_000.0 - (3_000_000_000.0 - 750_000_000.0)


def test_g8_a_smaller_limit_that_also_has_less_room_binds(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _tree({_OWN: _lvl(6_000_000_000, 2_000_000_000, None), _PAR: _lvl(4_000_000_000, 3_500_000_000, None)})
    _use_files(monkeypatch, tree)
    assert _v2() == 500_000_000.0


def test_g8_a_larger_limit_that_siblings_have_filled_binds_over_a_smaller_limit_with_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Own cap 3.0 GB with 1.0 GB used (2.0 GB left); the parent's cap is 5.0 GB but its usage is 4.5 GB: 0.5 GB left.
    Choosing the level with the smallest LIMIT would say 2.0 GB."""
    tree = _tree({_OWN: _lvl(3_000_000_000, 1_000_000_000, None), _PAR: _lvl(5_000_000_000, 4_500_000_000, None)})
    _use_files(monkeypatch, tree)
    assert _v2() == 500_000_000.0


def test_g8_the_own_level_binds_when_the_parent_has_room(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _tree({_OWN: _lvl(3_000_000_000, 1_000_000_000, None), _PAR: _lvl(5_000_000_000, 1_200_000_000, None)})
    _use_files(monkeypatch, tree)
    assert _v2() == 2_000_000_000.0


def test_g8_three_limited_levels_each_with_its_own_counters_take_the_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    """own: 4.0 - (1.0 - 0.2) = 3.2; parent: 6.0 - (3.0 - 0.5) = 3.5; grandparent: 8.0 - (7.4 - 0.9) = 1.5 (GB)."""

    def st(n: int) -> str:
        return _stat(cache=n, mapped=0, dirty=0, writeback=0, shmem=0)

    tree = _tree(
        {
            _OWN: _lvl(4_000_000_000, 1_000_000_000, st(200_000_000)),
            _PAR: _lvl(6_000_000_000, 3_000_000_000, st(500_000_000)),
            _GRAND: _lvl(8_000_000_000, 7_400_000_000, st(900_000_000)),
        }
    )
    # the synthetic chain is own -> parent (user.slice) -> the cgroup root, here given a numeric limit as the third level
    _use_files(monkeypatch, tree)
    assert _v2() == 1_500_000_000.0
    # raise the root's room: the parent (3.5) vs own (3.2) then decide -> own binds
    tree = _tree(
        {
            _OWN: _lvl(4_000_000_000, 1_000_000_000, st(200_000_000)),
            _PAR: _lvl(6_000_000_000, 3_000_000_000, st(500_000_000)),
            _GRAND: _lvl(8_000_000_000, 2_000_000_000, st(900_000_000)),
        }
    )
    _use_files(monkeypatch, tree)
    assert _v2() == 3_200_000_000.0


def test_g8_equal_limits_take_the_smaller_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _tree({_OWN: _lvl(4_000_000_000, 1_000_000_000, None), _PAR: _lvl(4_000_000_000, 3_900_000_000, None)})
    _use_files(monkeypatch, tree)
    assert _v2() == 100_000_000.0


# ---- G9: an ancestor whose usage cannot be used stands in with the own level ----


@pytest.mark.parametrize("mode", ["absent", "garbage", "unreadable", "negative"])
def test_g9_an_ancestor_without_a_usable_usage_uses_the_own_usage_and_reclaimable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, mode: str
) -> None:
    """Parent cap 4.0 GB; own usage 1.0 GB with 300 MB reclaimable: 4.0 - (1.0 - 0.3) = 3.3 GB, never None."""
    own = _stat(cache=300_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    current = {"absent": None, "garbage": "abc", "unreadable": 3_000_000_000, "negative": "-5"}[mode]
    tree = _tree({_OWN: _lvl("max", 1_000_000_000, own), _PAR: _lvl(4_000_000_000, current, _stat())})
    raising = (f"{_PAR}/memory.current",) if mode == "unreadable" else ()
    _use_files(monkeypatch, tree, raising=raising)
    with caplog.at_level("DEBUG"):
        assert _v2() == 3_300_000_000.0
        assert _v2() == 3_300_000_000.0
    assert len(_warnings(caplog, "memory.current")) == (0 if mode == "absent" else 1)


def test_g9_the_fallback_reclaimable_is_capped_at_the_own_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parent cap 4.0 GB without a usable usage; own usage 100 MB but an own stat claiming 900 MB reclaimable
    (skewed files): the stand-in is capped at the own usage, so the headroom is the parent's limit, not more."""
    own = _stat(cache=900_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    tree = _tree({_OWN: _lvl("max", 100_000_000, own), _PAR: _lvl(4_000_000_000, None, _stat())})
    _use_files(monkeypatch, tree)
    assert _v2() == 4_000_000_000.0


# ---- G10: the None cases, the files that are (not) opened ----


def test_g10_ancestor_levels_without_a_limit_are_never_read_for_usage_or_stat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chain inner -> app.scope -> user.slice -> root: only `inner` has a numeric limit (3.5 GB headroom: 4.0 - (1.0 - 0.5));
    the unlimited ancestors' memory.current and memory.stat (which would give different numbers) are never opened."""
    inner = f"{_OWN}/inner"
    tree = _tree(
        {
            inner: _lvl(4_000_000_000, 1_000_000_000, _stat()),
            _OWN: _lvl("max", 2_000_000_000, _stat()),
            _PAR: _lvl("max", 2_500_000_000, _stat()),
            _GRAND: _lvl(None, 9_000_000_000, _stat()),
        },
        own=inner,
    )
    opened: list[str] = []
    _use_files(monkeypatch, tree, opened=opened)
    assert _v2() == 3_500_000_000.0
    for unlimited in (_OWN, _PAR, ""):
        base = f"{_V2_ROOT}/{unlimited}" if unlimited else _V2_ROOT
        assert f"{base}/memory.current" not in opened
        assert f"{base}/memory.stat" not in opened
    assert f"{_V2_ROOT}/{inner}/memory.current" in opened


def test_g10_none_cases_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    # no numeric cap anywhere
    _use_files(monkeypatch, _tree({_OWN: _lvl("max", 1, _stat()), _PAR: _lvl("max", 1, None)}))
    assert _v2() is None
    # garbage memory.max
    _use_files(monkeypatch, _tree({_OWN: _lvl("garbage", 1, None)}))
    assert _v2() is None
    # unreadable memory.max
    _use_files(monkeypatch, _one_level(_stat()), raising=("/memory.max",))
    assert _v2() is None
    # own memory.current garbage / unreadable with a cap
    _use_files(monkeypatch, _one_level(_stat(), current="abc"))
    assert _v2() is None
    _use_files(monkeypatch, _one_level(_stat()), raising=(f"{_OWN}/memory.current",))
    assert _v2() is None


def test_g10_garbage_own_usage_is_none_even_with_a_capped_ancestor_and_no_stat_is_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    tree = _tree({_OWN: _lvl("max", "abc", _stat()), _PAR: _lvl(4_000_000_000, 3_000_000_000, _stat())})
    _use_files(monkeypatch, tree, opened=opened)
    assert _v2() is None
    assert not any(p.endswith("/memory.stat") for p in opened)


def test_g10_no_own_usage_file_and_a_capped_ancestor_is_measured_by_the_ancestor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Own cgroup exposes no memory files; the parent caps at 4.0 GB with 3.2 GB used, 400 MB reclaimable: 1.2 GB.
    The own stat is not opened (nothing needs it)."""
    opened: list[str] = []
    parent = _stat(cache=400_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    tree = _tree({_OWN: _lvl(None, None, _stat()), _PAR: _lvl(4_000_000_000, 3_200_000_000, parent)})
    _use_files(monkeypatch, tree, opened=opened)
    assert _v2() == 1_200_000_000.0
    assert not any(p.startswith(f"{_V2_ROOT}/{_OWN}/") and p.endswith("/memory.stat") for p in opened)


def test_g10_no_own_usage_file_and_nothing_to_measure_with_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # the own level is itself capped but has no usage
    _use_files(monkeypatch, _tree({_OWN: _lvl(6_000_000_000, None, _stat())}))
    assert _v2() is None
    # an ancestor is capped but has no usable usage either, and the own level has none to stand in
    _use_files(monkeypatch, _tree({_OWN: _lvl(None, None, _stat()), _PAR: _lvl(4_000_000_000, None, _stat())}))
    assert _v2() is None


def test_g10_with_no_own_usage_one_unusable_capped_ancestor_makes_the_whole_figure_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Own cgroup exposes no memory files; the parent is capped and readable, the root-level cap has garbage usage:
    no figure can be given for the second level, so the reader returns None (the caller falls back) rather than a
    figure that ignores a cap it could not measure. Pinned as the fail-soft contract (None = fall back)."""
    tree = _tree(
        {
            _OWN: _lvl(None, None, None),
            _PAR: _lvl(4_000_000_000, 3_200_000_000, _stat()),
            _GRAND: _lvl(8_000_000_000, "abc", _stat()),
        }
    )
    _use_files(monkeypatch, tree)
    assert _v2() is None


def test_g10_nothing_of_the_own_level_is_opened_when_no_level_has_a_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    _use_files(monkeypatch, _tree({_OWN: _lvl("max", 1, _stat()), _PAR: _lvl("max", 1, _stat())}), opened=opened)
    assert _v2() is None
    assert not any(p.endswith(("/memory.current", "/memory.stat")) for p in opened)


def test_g10_the_stat_is_only_opened_after_the_own_usage_was_read(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    _use_files(monkeypatch, _one_level(_stat()), opened=opened)
    _v2()
    own_current = opened.index(f"{_V2_ROOT}/{_OWN}/memory.current")
    own_stat = opened.index(f"{_V2_ROOT}/{_OWN}/memory.stat")
    assert own_current < own_stat


# ---- G11: cgroup v1 ----


def test_g11_v1_headroom_subtracts_the_hierarchical_cache_minus_mapped_dirty_writeback_shmem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """limit 5.5 GB, usage 5.0 GB, reclaimable 700-50-30-20-100 = 500 MB -> 1.0 GB (the non-total_ lines are decoys)."""
    _use_files(monkeypatch, _v1_files(_v1_stat()))
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 1_000_000_000.0


@pytest.mark.parametrize(
    ("which", "sign"), [("cache", 1), ("mapped", -1), ("dirty", -1), ("writeback", -1), ("shmem", -1)]
)
def test_g11_v1_each_term_moves_the_figure_by_exactly_its_size(
    monkeypatch: pytest.MonkeyPatch, which: str, sign: int
) -> None:
    kw = {"cache": 700_000_000, "mapped": 50_000_000, "dirty": 30_000_000, "writeback": 20_000_000, "shmem": 100_000_000}
    kw[which] += 10_000_000
    _use_files(monkeypatch, _v1_files(_v1_stat(**kw)))
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 1_000_000_000.0 + sign * 10_000_000


def test_g11_v1_active_unmapped_cache_counts_and_mapped_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_files(monkeypatch, _v1_files(_v1_stat(cache=800_000_000, mapped=0, dirty=0, writeback=0, shmem=0)))
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 1_300_000_000.0
    _use_files(monkeypatch, _v1_files(_v1_stat(cache=800_000_000, mapped=800_000_000, dirty=0, writeback=0, shmem=0)))
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 500_000_000.0


def test_g11_v1_clamps_and_floors(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_files(monkeypatch, _v1_files(_v1_stat(cache=9_000_000_000, mapped=0, dirty=0, writeback=0, shmem=0)))
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 5_500_000_000.0  # capped at usage: headroom = limit
    _use_files(monkeypatch, _v1_files(_v1_stat(cache=100, mapped=300_000_000)))
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 500_000_000.0  # floored: nothing subtracted
    _use_files(monkeypatch, _v1_files(_v1_stat(cache=0, mapped=0, dirty=0, writeback=0, shmem=0), usage="9000000000"))
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 0.0


@pytest.mark.parametrize("key", _V1_KEYS)
@pytest.mark.parametrize("raw", [*_BAD_VALUES, _REMOVE])
def test_g11_v1_any_unusable_page_cache_line_subtracts_nothing(
    monkeypatch: pytest.MonkeyPatch, key: str, raw: object
) -> None:
    _use_files(monkeypatch, _v1_files(_v1_stat(override={key: raw})))
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 500_000_000.0


@pytest.mark.parametrize("stat", [None, ""])
def test_g11_v1_absent_or_empty_stat_subtracts_nothing(monkeypatch: pytest.MonkeyPatch, stat: str | None) -> None:
    _use_files(monkeypatch, _v1_files(stat))
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 500_000_000.0


def test_g11_v1_uses_only_total_keys_not_the_non_hierarchical_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    """A v1 file with only the own-cgroup (non-total) lines has none of the five terms: nothing is subtracted."""
    stat = "cache 700000000\nmapped_file 50000000\ndirty 30000000\nwriteback 20000000\nshmem 100000000\n"
    _use_files(monkeypatch, _v1_files(stat))
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 500_000_000.0


def test_g11_v1_reads_only_the_own_cgroup_not_a_parent_or_the_root(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _v1_files(_v1_stat())
    files["/sys/fs/cgroup/memory/user.slice/memory.limit_in_bytes"] = "1000000000\n"
    files["/sys/fs/cgroup/memory/user.slice/memory.usage_in_bytes"] = "999000000\n"
    files["/sys/fs/cgroup/memory/memory.stat"] = _v1_stat(cache=9_000_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    _use_files(monkeypatch, files)
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 1_000_000_000.0


# ---- G12/G13: text and consumers ----


def test_g13_available_headroom_carries_the_v2_figure_and_does_not_reach_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_files(monkeypatch, _one_level(_stat()))
    monkeypatch.setattr(
        reranker_mod, "_psutil_available_bytes", lambda: pytest.fail("psutil must not be read under a cgroup cap")
    )
    assert reranker_mod._available_ram_headroom_bytes() == 5_500_000_000.0


def test_g13_available_headroom_v1_fallback_carries_the_v1_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_files(monkeypatch, _v1_files(_v1_stat()))
    monkeypatch.setattr(
        reranker_mod, "_psutil_available_bytes", lambda: pytest.fail("psutil must not be read under a cgroup cap")
    )
    assert reranker_mod._available_ram_headroom_bytes() == 1_000_000_000.0


def test_g13_the_ancestor_only_cap_is_not_dropped_to_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _tree({_OWN: _lvl(None, None, None), _PAR: _lvl(4_000_000_000, 3_200_000_000, _stat())})
    _use_files(monkeypatch, tree)
    monkeypatch.setattr(
        reranker_mod, "_psutil_available_bytes", lambda: pytest.fail("psutil must not be read under a cgroup cap")
    )
    assert reranker_mod._available_ram_headroom_bytes() == 4_000_000_000.0 - (3_200_000_000.0 - 500_000_000.0)


def _py_files_mentioning(root, token: str) -> list[str]:
    return [str(p) for p in root.rglob("*.py") if token in p.read_text(encoding="utf-8")]


def test_g12_the_docstrings_state_the_new_formula_and_the_v1_claim_is_corrected() -> None:
    v2 = reranker_mod._cgroup_v2_memory_headroom_bytes.__doc__ or ""
    v1 = reranker_mod._cgroup_v1_memory_headroom_bytes.__doc__ or ""
    helper = reranker_mod._cgroup_reclaimable_file_bytes.__doc__ or ""
    avail = reranker_mod._available_ram_headroom_bytes.__doc__ or ""
    for doc in (helper, avail):
        assert "file_mapped" in doc and "tmpfs" in doc
    assert "file_writeback" in helper and "shmem" in helper
    assert "MINIMUM" in v2 and "own usage" in v2 and "siblings" in v2 and "file_mapped" in v2
    assert "total_cache" in v1 and "total_mapped_file" in v1 and "tmpfs" in v1
    assert "OWN cgroup" in v1 and "NOT visible" in v1 and "own setting" in v1
    assert "already hierarchical" not in v1 and "no walk-up is needed" not in v1
    for doc in (v2, v1, helper, avail):
        assert "matches psutil" not in doc.lower() and "matches the host" not in doc.lower()


def test_g12_no_text_in_brain_still_calls_inactive_file_the_reclaimable_term(tmp_path) -> None:
    from pathlib import Path

    root = Path(reranker_mod.__file__).resolve().parents[1]
    assert (root / "memory" / "reranker.py").is_file()
    assert _py_files_mentioning(root, "inactive_file") == []
    # the sweep can fail: it flags a known-bad file placed in a directory it walks
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "stale.py").write_text('"""reclaimable = memory.stat inactive_file"""\n', encoding="utf-8")
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    assert _py_files_mentioning(tmp_path, "inactive_file") == [str(tmp_path / "sub" / "stale.py")]


def test_psutil_available_bytes_reads_the_host_available_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """S211: the host free-memory fallback is psutil's `available`, the same call on every platform."""
    import psutil

    class _Mem:
        available = 5_000_000 * 1024

    for platform in ("linux", "darwin", "win32"):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(psutil, "virtual_memory", lambda: _Mem())
        assert reranker_mod._psutil_available_bytes() == 5_000_000 * 1024.0


def test_psutil_available_bytes_none_when_psutil_fails_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import psutil

    def _boom():
        raise OSError("no access")

    monkeypatch.setattr(psutil, "virtual_memory", _boom)
    reranker_mod._ram_figure_warned.clear()
    with caplog.at_level("DEBUG"):
        assert reranker_mod._psutil_available_bytes() is None
        assert reranker_mod._psutil_available_bytes() is None
    warns = [r for r in caplog.records if r.levelname == "WARNING" and "free memory could not be read" in r.getMessage()]
    assert len(warns) == 1


def test_headroom_is_read_on_a_non_linux_platform_through_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """S211: on macOS/Windows the cgroup readers answer None and the headroom is still psutil's."""
    import psutil

    class _Mem:
        available = 7e9

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Mem())
    assert reranker_mod._available_ram_headroom_bytes() == 7e9


def test_available_ram_headroom_prefers_cgroup_v2_over_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: 111.0)
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: 222.0)
    monkeypatch.setattr(reranker_mod, "_psutil_available_bytes", lambda: 333.0)
    assert reranker_mod._available_ram_headroom_bytes() == 111.0


def test_available_ram_headroom_falls_back_to_cgroup_v1_when_v2_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: 222.0)
    monkeypatch.setattr(reranker_mod, "_psutil_available_bytes", lambda: 333.0)
    assert reranker_mod._available_ram_headroom_bytes() == 222.0


def test_available_ram_headroom_falls_back_to_psutil_when_no_cgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_psutil_available_bytes", lambda: 333.0)
    assert reranker_mod._available_ram_headroom_bytes() == 333.0


def test_available_ram_headroom_none_when_everything_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safe-degrade case: no cgroup limit and an unreadable
    psutil memory read -> None, which the width fit
    must treat as 'skip the memory term', never as unlimited headroom."""
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_psutil_available_bytes", lambda: None)
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
    assert result.peak_growth_bytes == 4096.0


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
