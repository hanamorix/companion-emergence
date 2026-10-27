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

import ast as _ast
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
from tests.source_tree import brain_sources, brain_trees

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
    SAME tunables.toml setting mechanism `test_manual_override_wins_over_
    avx2_auto_detected_default` above exercises for the latency budget),
    the fp32 export loads instead of fp16 — proving the pin is a DEFAULT,
    not a hard removal of operator choice. fp16 registration must never
    even be attempted under this override."""
    import brain.tunables as tunables_mod

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables_mod._reset_for_tests()
    (tmp_path / "tunables.toml").write_text('"reranker.precision" = "fp32"\n', encoding="utf-8")

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
    """A manual tunables.toml value for reranker.latency_budget_seconds
    must win over the AVX2-auto-detected default — regardless of what that
    default resolved to — per #250 §3's "manual override takes precedence"
    requirement. Simulates an AVX2-detected host (default would be 2.0s,
    via the module-level LATENCY_BUDGET_SECONDS monkeypatch below) with an
    override of 0.25s, so the two are clearly distinguishable: only the
    override value can produce the asserted width."""
    import brain.tunables as tunables_mod

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables_mod._reset_for_tests()
    (tmp_path / "tunables.toml").write_text('"reranker.latency_budget_seconds" = 0.25\n', encoding="utf-8")
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
    FileNotFoundError while walking, which the walk must tolerate).
    S240: the capped parent has its own memory.current (a real kernel always has one); without it the
    parent would count 0 (the second assertion), where R13 borrowed the process's own usage."""
    files = {
        "/proc/self/cgroup": "0::/user.slice/app.scope\n",
        "/sys/fs/cgroup/user.slice/app.scope/memory.max": "max\n",
        "/sys/fs/cgroup/user.slice/app.scope/memory.current": "1000000000\n",
        "/sys/fs/cgroup/user.slice/memory.max": "4000000000\n",
        "/sys/fs/cgroup/user.slice/memory.current": "1000000000\n",
        # root's memory.max deliberately absent -> must be tolerated
    }
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("builtins.open", _fake_open_dispatcher(files))
    assert reranker_mod._cgroup_v2_memory_headroom_bytes() == 3_000_000_000.0
    del files["/sys/fs/cgroup/user.slice/memory.current"]
    assert reranker_mod._cgroup_v2_memory_headroom_bytes() == 0.0  # R13: 3.0 GB, from the process's own usage


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
    assert warns[0].exc_info and warns[0].exc_info[0] is PermissionError  # the traceback comes with the first warning


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


# ---- G9: a capped ancestor whose usage cannot be used counts 0 and borrows nothing (S240) ----


@pytest.mark.parametrize("mode", ["absent", "garbage", "unreadable", "negative"])
def test_g9_a_capped_ancestor_without_a_usable_usage_has_no_headroom_and_borrows_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, mode: str
) -> None:
    """S240 (R13 stood in with the own usage and gave 3.3 GB here): the parent is capped at 4.0 GB and its
    usage is unusable, so its headroom is 0.0 (the own usage 1.0 GB and the own 300 MB reclaimable are never
    borrowed), one warning per (file, kind) including the absent file, and no memory.stat is opened."""
    own = _stat(cache=300_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    current = {"absent": None, "garbage": "abc", "unreadable": 3_000_000_000, "negative": "-5"}[mode]
    tree = _tree({_OWN: _lvl("max", 1_000_000_000, own), _PAR: _lvl(4_000_000_000, current, _stat())})
    raising = (f"{_PAR}/memory.current",) if mode == "unreadable" else ()
    opened: list[str] = []
    _use_files(monkeypatch, tree, raising=raising, opened=opened)
    with caplog.at_level("DEBUG"):
        assert _v2() == 0.0
        assert _v2() == 0.0
    found = _warnings(caplog, "memory.current")
    assert len(found) == 1
    if mode == "unreadable":
        assert found[0].exc_info and found[0].exc_info[0] is PermissionError
    assert not any(p.endswith("/memory.stat") for p in opened)


def test_g9_the_own_counters_are_never_borrowed_for_a_capped_ancestor(monkeypatch: pytest.MonkeyPatch) -> None:
    """S240 (R13 capped the stand-in's reclaimable at the own usage and gave the parent's limit, 4.0 GB): the
    parent is capped at 4.0 GB with no usable usage while the own level has 100 MB used and a skewed stat
    claiming 900 MB reclaimable. Nothing of the own level is used: the parent counts 0.0."""
    own = _stat(cache=900_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    tree = _tree({_OWN: _lvl("max", 100_000_000, own), _PAR: _lvl(4_000_000_000, None, _stat())})
    _use_files(monkeypatch, tree)
    assert _v2() == 0.0


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
    # own memory.current garbage / unreadable under a cap: S240, headroom 0.0 (R13: None)
    _use_files(monkeypatch, _one_level(_stat(), current="abc"))
    assert _v2() == 0.0
    _use_files(monkeypatch, _one_level(_stat()), raising=(f"{_OWN}/memory.current",))
    assert _v2() == 0.0


def test_g10_garbage_own_usage_with_a_capped_ancestor_is_measured_by_the_ancestor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S238 (R12 returned None here): the own level is unlimited with an unusable usage, the parent is capped at
    4.0 GB with 3.0 GB used and 500 MB reclaimable: 1.5 GB, from the parent's own counters. The own stat is not
    opened (nothing needs it)."""
    opened: list[str] = []
    tree = _tree({_OWN: _lvl("max", "abc", _stat()), _PAR: _lvl(4_000_000_000, 3_000_000_000, _stat())})
    _use_files(monkeypatch, tree, opened=opened)
    assert _v2() == 1_500_000_000.0
    assert f"{_V2_ROOT}/{_PAR}/memory.stat" in opened
    assert f"{_V2_ROOT}/{_OWN}/memory.stat" not in opened


def test_g10_no_own_usage_file_and_a_capped_ancestor_is_measured_by_the_ancestor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Own cgroup exposes no memory files; the parent caps at 4.0 GB with 3.2 GB used, 400 MB reclaimable: 1.2 GB.
    The own stat is not opened (nothing needs it)."""
    opened: list[str] = []
    parent = _stat(cache=400_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    tree = _tree({_OWN: _lvl(None, None, _stat()), _PAR: _lvl(4_000_000_000, 3_200_000_000, parent)})
    _use_files(monkeypatch, tree, opened=opened)
    assert _v2() == 1_200_000_000.0
    assert not any(p.startswith(f"{_V2_ROOT}/{_OWN}/") and p.endswith("/memory.stat") for p in opened)


def test_g10_a_capped_level_without_a_usage_file_has_zero_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    """S240 (R13: None for both): the own level is itself capped but has no usage; and an ancestor is capped but
    has no usable usage while the own level (uncapped) has none to lend."""
    _use_files(monkeypatch, _tree({_OWN: _lvl(6_000_000_000, None, _stat())}))
    assert _v2() == 0.0
    _use_files(monkeypatch, _tree({_OWN: _lvl(None, None, _stat()), _PAR: _lvl(4_000_000_000, None, _stat())}))
    assert _v2() == 0.0


def test_g10_with_no_own_usage_one_unusable_capped_ancestor_binds_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """S240 (R13 skipped the unusable level and gave 1.3 GB): the own cgroup exposes no memory files; the parent
    is capped and readable (4.0 GB, 3.2 GB used, 500 MB reclaimable: 1.3 GB); the root-level cap has garbage
    usage, so it counts 0.0 and the minimum is 0.0."""
    tree = _tree(
        {
            _OWN: _lvl(None, None, None),
            _PAR: _lvl(4_000_000_000, 3_200_000_000, _stat()),
            _GRAND: _lvl(8_000_000_000, "abc", _stat()),
        }
    )
    _use_files(monkeypatch, tree)
    assert _v2() == 0.0
    tree[f"{_V2_ROOT}/memory.current"] = "5000000000\n"  # the same tree with a usable root usage: the parent binds
    _use_files(monkeypatch, tree)
    assert _v2() == 1_300_000_000.0


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


def test_g11_v1_a_tighter_parent_binds_and_the_roots_stat_is_a_decoy(monkeypatch: pytest.MonkeyPatch) -> None:
    """S237 (R12 pinned the opposite): the parent's cap now counts, with the parent's own usage (no stat file: nothing
    subtracted) - 1.0 GB limit, 999 MB used = 1 MB; the root has no limit file, so its stat is never used."""
    files = _v1_files(_v1_stat())
    files["/sys/fs/cgroup/memory/user.slice/memory.limit_in_bytes"] = "1000000000\n"
    files["/sys/fs/cgroup/memory/user.slice/memory.usage_in_bytes"] = "999000000\n"
    files["/sys/fs/cgroup/memory/memory.stat"] = _v1_stat(cache=9_000_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    _use_files(monkeypatch, files)
    assert reranker_mod._cgroup_v1_memory_headroom_bytes() == 1_000_000.0


# ---- G12/G13: text and consumers ----


def test_g13_available_headroom_carries_the_v2_figure_when_psutil_reads_more(monkeypatch: pytest.MonkeyPatch) -> None:
    """S241 (R13: psutil must not be read under a cgroup cap): psutil is read once and the smaller figure wins."""
    _use_files(monkeypatch, _one_level(_stat()))
    reads = _psutil_reads(monkeypatch, 9_000_000_000.0)
    assert reranker_mod._available_ram_headroom_bytes() == 5_500_000_000.0
    assert len(reads) == 1


def test_g13_available_headroom_v1_fallback_carries_the_v1_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_files(monkeypatch, _v1_files(_v1_stat()))
    reads = _psutil_reads(monkeypatch, 9_000_000_000.0)
    assert reranker_mod._available_ram_headroom_bytes() == 1_000_000_000.0
    assert len(reads) == 1


def test_g13_the_ancestor_only_cap_is_not_dropped_to_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _tree({_OWN: _lvl(None, None, None), _PAR: _lvl(4_000_000_000, 3_200_000_000, _stat())})
    _use_files(monkeypatch, tree)
    reads = _psutil_reads(monkeypatch, 9_000_000_000.0)
    assert reranker_mod._available_ram_headroom_bytes() == 4_000_000_000.0 - (3_200_000_000.0 - 500_000_000.0)
    assert len(reads) == 1


# ---- R13: S239 memory.high and the strict number parse, S238 an unusable level never ends the read,
# S237 the v1 reader walks its ancestors ----

_ZERO = _stat(cache=0, mapped=0, dirty=0, writeback=0, shmem=0)  # reclaimable 0
_R400 = _stat(cache=400_000_000, mapped=0, dirty=0, writeback=0, shmem=0)  # reclaimable 400 MB
_V1_ROOT = "/sys/fs/cgroup/memory"
_V1_PAR = f"{_V1_ROOT}/user.slice"
_SENTINEL = str(reranker_mod._CGROUP_V1_UNBOUNDED_SENTINEL)


def _hi(tree: dict[str, str], rel: str, value: str) -> dict[str, str]:
    """Add a memory.high file at a level of a v2 tree."""
    base = f"{_V2_ROOT}/{rel}" if rel else _V2_ROOT
    tree[f"{base}/memory.high"] = f"{value}\n"
    return tree


def _v1lvl(limit: str | None = None, usage: str | None = None, stat: str | None = None, hier: str | None = None) -> dict:
    return {"limit": limit, "usage": usage, "stat": stat, "hier": hier}


def _v1tree(levels: dict[str, dict], *, own: str = _OWN) -> dict[str, str]:
    """Synthetic v1 files for a chain; `levels` maps a relative directory ("" = the controller root) to its files."""
    files = {"/proc/self/cgroup": f"5:memory:/{own}\n"}
    for rel, lv in levels.items():
        base = f"{_V1_ROOT}/{rel}" if rel else _V1_ROOT
        for key, name in (
            ("limit", "memory.limit_in_bytes"),
            ("usage", "memory.usage_in_bytes"),
            ("hier", "memory.use_hierarchy"),
        ):
            if lv[key] is not None:
                files[f"{base}/{name}"] = f"{lv[key]}\n"
        if lv["stat"] is not None:
            files[f"{base}/memory.stat"] = lv["stat"]
    return files


def _v1() -> float | None:
    return reranker_mod._cgroup_v1_memory_headroom_bytes()


def _v1s(**kw) -> str:
    return _v1_stat(**kw)


_V1_ZERO = _v1s(cache=0, mapped=0, dirty=0, writeback=0, shmem=0)
_V1_R400 = _v1s(cache=400_000_000, mapped=0, dirty=0, writeback=0, shmem=0)


def _no_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_psutil_available_bytes", lambda: pytest.fail("psutil must not be read"))


def _psutil_reads(monkeypatch: pytest.MonkeyPatch, value: float | None) -> list[float | None]:
    """Patch psutil's available figure to `value`; the returned list records every read (S241: one per call)."""
    reads: list[float | None] = []

    def _read() -> float | None:
        reads.append(value)
        return value

    monkeypatch.setattr(reranker_mod, "_psutil_available_bytes", _read)
    return reads


# ---- H1: memory.high counts as a limit (S239) ----


@pytest.mark.parametrize(
    ("max_", "high", "expected"),
    [
        ("6000000000", "4000000000", 3_500_000_000.0),  # the high is lower: it binds (max-only would say 5.5e9)
        ("6000000000", "8000000000", 5_500_000_000.0),  # the max is lower: it binds
        ("max", "4000000000", 3_500_000_000.0),  # a high alone limits
        ("6000000000", "max", 5_500_000_000.0),  # "max" high imposes nothing
        ("6000000000", None, 5_500_000_000.0),  # an absent high imposes nothing
        ("6000000000", "0", 0.0),  # a high of 0 is a limit of 0, not "no limit"
    ],
)
def test_h1_the_limit_of_a_level_is_the_lower_of_memory_max_and_memory_high(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, max_: str, high: str | None, expected: float
) -> None:
    tree = _one_level(_stat(), limit=max_)
    if high is not None:
        _hi(tree, _OWN, high)
    _use_files(monkeypatch, tree)
    with caplog.at_level("DEBUG"):
        assert _v2() == expected
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_h1_a_memory_high_at_an_ancestor_is_measured_by_that_ancestors_own_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Own level unlimited, the parent has only a memory.high of 4.0 GB: its usage 3.2 GB with 400 MB reclaimable
    gives 1.2 GB (the own counters would give 4.0 - (1.0 - 0.3) = 3.3 GB)."""
    tree = _tree(
        {
            _OWN: _lvl("max", 1_000_000_000, _stat(cache=300_000_000, mapped=0, dirty=0, writeback=0, shmem=0)),
            _PAR: _lvl("max", 3_200_000_000, _R400),
        }
    )
    _hi(tree, _PAR, "4000000000")
    _use_files(monkeypatch, tree)
    assert _v2() == 1_200_000_000.0


@pytest.mark.parametrize("bad", ["abc", "-5", "1.5", "1e9", "inf", "+5", ""])
def test_h1_a_memory_high_that_is_not_a_plain_integer_degrades_the_level_to_zero_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad: str
) -> None:
    """S240 (R13 ignored the file and let the max stand, 5.5 GB): a numeric max beside a high we cannot use is a
    capped level with an unknown second limit: headroom 0.0."""
    tree = _one_level(_stat())
    _hi(tree, _OWN, bad)
    _use_files(monkeypatch, tree)
    with caplog.at_level("DEBUG"):
        assert _v2() == 0.0
        assert _v2() == 0.0
    assert len(_warnings(caplog, "memory.high")) == 1


def test_h1_an_unreadable_memory_high_degrades_the_level_to_zero_and_warns_once_with_the_traceback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tree = _one_level(_stat())
    _hi(tree, _OWN, "4000000000")
    _use_files(monkeypatch, tree, raising=("memory.high",))
    with caplog.at_level("DEBUG"):
        assert _v2() == 0.0  # R13: 5.5 GB, the max alone
        assert _v2() == 0.0
    found = _warnings(caplog, "memory.high")
    assert len(found) == 1
    assert found[0].exc_info and found[0].exc_info[0] is PermissionError


@pytest.mark.parametrize("mode", ["garbage", "unreadable"])
def test_h1_a_bad_memory_max_beside_a_numeric_memory_high_degrades_the_level_to_zero(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """S240 (R13 let the readable high stand, 3.5 GB, and disclosed the over-read): the level is capped (the high
    is a number) and its max cannot be used, so its limit is unknown: headroom 0.0."""
    tree = _one_level(_stat(), limit="abc" if mode == "garbage" else "6000000000")
    _hi(tree, _OWN, "4000000000")
    _use_files(monkeypatch, tree, raising=("memory.max",) if mode == "unreadable" else ())
    assert _v2() == 0.0


def test_h1_memory_high_is_read_at_every_level_and_an_unlimited_levels_counters_are_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    levels = {_OWN: _lvl("max", 1, _stat()), _PAR: _lvl("max", 1, _stat()), _GRAND: _lvl(None, 1, _stat())}
    _use_files(monkeypatch, _tree(levels), opened=opened)
    assert _v2() is None
    for rel in (_OWN, _PAR, _GRAND):
        base = f"{_V2_ROOT}/{rel}" if rel else _V2_ROOT
        assert f"{base}/memory.high" in opened and f"{base}/memory.max" in opened
        assert f"{base}/memory.current" not in opened and f"{base}/memory.stat" not in opened


# ---- H2: the strict number parse (S239) ----

_BAD_NUMBERS = [
    "1e9", "1.5e9", "-5", "+5", "1_000", "nan", "inf", "", " 12 3", "\u0661\u0662\u0663", "\u00b2", "9" * 400, "abc", "12\u00a0",
]


@pytest.mark.parametrize("bad", _BAD_NUMBERS)
def test_h2_an_own_usage_that_is_not_a_plain_integer_is_rejected_and_warned_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad: str
) -> None:
    _use_files(monkeypatch, _one_level(_stat(), current=bad))
    with caplog.at_level("DEBUG"):
        assert _v2() == 0.0  # S240: the capped level counts 0 (R13: None, the level did not resolve)
        assert _v2() == 0.0
    assert len(_warnings(caplog, "memory.current")) == 1


@pytest.mark.parametrize("bad", ["1e9", "-5", "1.5", "nan", "inf", "9" * 400, "\u0661\u0662\u0663"])
def test_h2_a_bad_own_usage_does_not_stop_a_good_ancestor_from_measuring_the_chain(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    """An UNCAPPED own level's usage is never read, so a bad one cannot stop the capped parent from measuring
    (1.2 GB from the parent's own counters). With the own level capped the same bad usage counts 0.0 (S240; R13
    gave the parent's 1.2 GB)."""
    tree = _tree({_OWN: _lvl("max", bad, _ZERO), _PAR: _lvl(4_000_000_000, 3_200_000_000, _R400)})
    _use_files(monkeypatch, tree)
    assert _v2() == 1_200_000_000.0
    tree = _tree({_OWN: _lvl(6_000_000_000, bad, _ZERO), _PAR: _lvl(4_000_000_000, 3_200_000_000, _R400)})
    _use_files(monkeypatch, tree)
    assert _v2() == 0.0


@pytest.mark.parametrize("raw", ["1000000000\n", "1000000000", "  1000000000  \n"])
def test_h2_a_plain_integer_parses(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    _use_files(monkeypatch, _one_level(_stat(), current=raw.rstrip("\n")))
    assert _v2() == 5_500_000_000.0


def test_h2_zero_is_a_plain_integer() -> None:
    assert reranker_mod._cgroup_plain_bytes("0\n") == 0.0
    assert reranker_mod._cgroup_plain_bytes("000123") == 123.0


@pytest.mark.parametrize("bad", ["1e9", "-5", "nan", "9" * 400])
def test_h2_a_v1_usage_that_is_not_a_plain_integer_is_rejected(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    _use_files(monkeypatch, _v1_files(_v1_stat(), usage=bad))
    assert _v1() == 0.0  # S240: the capped level counts 0 (R13: None)


@pytest.mark.parametrize("bad", ["1e9", "-5", "inf", "9" * 400, "0x10"])
def test_h2_a_limit_that_is_not_a_plain_integer_is_skipped_at_both_versions(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    _use_files(monkeypatch, _one_level(_stat(), limit=bad))
    assert _v2() is None  # an "infinite" or negative limit is not read as a number
    _use_files(monkeypatch, _v1_files(_v1_stat(), limit=bad))
    assert _v1() is None


# ---- H3: one unusable level never ends the read (S238) ----


@pytest.mark.parametrize("mode", ["garbage", "unreadable"])
def test_h3_an_unusable_own_usage_degrades_the_own_level_to_zero_whatever_the_parent_says(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, mode: str
) -> None:
    """S240 (R13 returned the parent's 1.2 GB here and R12 returned None): the own level is capped at 6.0 GB and
    its usage cannot be used, so it counts 0.0 and the minimum is 0.0, however roomy the parent is."""
    own_current = "abc" if mode == "garbage" else 900_000_000
    tree = _tree({_OWN: _lvl(6_000_000_000, own_current, _stat()), _PAR: _lvl(4_000_000_000, 3_200_000_000, _R400)})
    _use_files(monkeypatch, tree, raising=(f"{_OWN}/memory.current",) if mode == "unreadable" else ())
    with caplog.at_level("DEBUG"):
        assert _v2() == 0.0
    found = _warnings(caplog, "memory.current")
    assert len(found) == 1
    if mode == "unreadable":
        assert found[0].exc_info and found[0].exc_info[0] is PermissionError


@pytest.mark.parametrize("mode", ["garbage", "unreadable"])
def test_h3_an_unusable_own_memory_max_still_lets_the_capped_parent_bind(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    tree = _tree(
        {
            _OWN: _lvl("abc" if mode == "garbage" else 6_000_000_000, 1_000_000_000, _stat()),
            _PAR: _lvl(4_000_000_000, 3_200_000_000, _R400),
        }
    )
    _use_files(monkeypatch, tree, raising=(f"{_OWN}/memory.max",) if mode == "unreadable" else ())
    assert _v2() == 1_200_000_000.0


def test_h3_with_no_own_usage_a_garbled_near_ancestor_counts_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """S240 (R13 skipped the garbled parent and gave the root level's 4.0 GB): the parent is capped with garbage
    usage, so it counts 0.0; the root level (capped 8.0 GB, usage 5.0 GB, 1.0 GB reclaimable: 4.0 GB) does not
    matter."""
    root_stat = _stat(cache=1_000_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    tree = _tree(
        {
            _OWN: _lvl(None, None, None),
            _PAR: _lvl(4_000_000_000, "abc", _stat()),
            _GRAND: _lvl(8_000_000_000, 5_000_000_000, root_stat),
        }
    )
    _use_files(monkeypatch, tree)
    assert _v2() == 0.0


def test_h3_the_minimum_includes_a_degraded_levels_zero_and_ignores_an_uncapped_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S240 (R13: the unresolved levels neither lowered nor raised the result, 0.5 GB). The own level is uncapped
    with a garbage usage (never read, it changes nothing); the parent resolves to 0.5 GB; the root level has a
    tiny limit and garbage usage: it counts 0.0, so the minimum is 0.0. Without the root level it is 0.5 GB."""
    tree = _tree(
        {
            _OWN: _lvl("max", "abc", _stat()),
            _PAR: _lvl(4_000_000_000, 3_500_000_000, _ZERO),
            _GRAND: _lvl(100_000_000, "abc", _stat()),
        }
    )
    _use_files(monkeypatch, tree)
    assert _v2() == 0.0
    del tree[f"{_V2_ROOT}/memory.max"]
    _use_files(monkeypatch, tree)
    assert _v2() == 500_000_000.0


@pytest.mark.parametrize(
    "own_files",
    [
        {"max": "abc"},  # a memory.max we cannot use, no memory.high
        {"max": "abc", "high": "def"},  # both unusable
        {"high": "abc"},  # an unusable high, no memory.max
    ],
)
def test_h3_pinned_exposure_a_level_with_no_usable_limit_file_lets_a_looser_ancestor_bind(
    monkeypatch: pytest.MonkeyPatch, own_files: dict[str, str]
) -> None:
    """The residual exposure of S240 as ordered ("a readable memory.max or memory.high"), pinned on purpose and
    a disclosed owner question: a level none of whose limit files is usable gives no evidence of a cap, so it is
    skipped, and the looser parent (8.0 GB, 3.0 GB used: 5.0 GB) binds. A future change that degrades such a
    level must edit this test."""
    tree = _tree({_OWN: _lvl(own_files.get("max"), 1_000_000_000, _stat()), _PAR: _lvl(8_000_000_000, 3_000_000_000, _ZERO)})
    if "high" in own_files:
        _hi(tree, _OWN, own_files["high"])
    _use_files(monkeypatch, tree)
    assert _v2() == 5_000_000_000.0


def test_h3_the_own_usage_of_an_uncapped_own_level_is_never_read(monkeypatch: pytest.MonkeyPatch) -> None:
    own_current = f"{_V2_ROOT}/{_OWN}/memory.current"
    opened: list[str] = []
    # own unlimited, the capped parent has its own usage: the own usage is never opened
    tree = _tree({_OWN: _lvl("max", 1_000_000_000, _stat()), _PAR: _lvl(4_000_000_000, 3_000_000_000, _ZERO)})
    _use_files(monkeypatch, tree, opened=opened)
    assert _v2() == 1_000_000_000.0
    assert own_current not in opened
    # S240 (R13 lent the own usage to both ancestors: 3.0 GB): two capped ancestors without a usage of their own
    # count 0.0 each, and the own usage is still not opened.
    opened.clear()
    tree = _tree(
        {
            _OWN: _lvl("max", 1_000_000_000, _ZERO),
            _PAR: _lvl(4_000_000_000, None, _stat()),
            _GRAND: _lvl(6_000_000_000, None, _stat()),
        }
    )
    _use_files(monkeypatch, tree, opened=opened)
    assert _v2() == 0.0
    assert own_current not in opened


def test_h3_different_files_and_kinds_warn_under_separate_keys(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tree = _one_level(_stat(), limit="abc")
    _hi(tree, _OWN, "def")
    _use_files(monkeypatch, tree)
    with caplog.at_level("DEBUG"):
        assert _v2() is None
        assert _v2() is None
    assert len(_warnings(caplog, "memory.max")) == 1 and len(_warnings(caplog, "memory.high")) == 1


def test_h3_a_garbled_own_usage_and_an_unreadable_ancestor_usage_warn_separately(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tree = _tree({_OWN: _lvl(6_000_000_000, "abc", _stat()), _PAR: _lvl(4_000_000_000, 3_000_000_000, _stat())})
    _use_files(monkeypatch, tree, raising=(f"{_PAR}/memory.current",))
    with caplog.at_level("DEBUG"):
        assert _v2() == 0.0  # both capped levels are unmeasurable: S240 counts 0 (R13: None)
        assert _v2() == 0.0
    found = _warnings(caplog, "memory.current")
    assert len(found) == 2
    assert sorted(bool(r.exc_info) for r in found) == [False, True]


def test_h3_no_level_known_to_be_capped_falls_back_to_psutil_not_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No level with a usable numeric limit: the cgroup figure is absent and psutil's reading stands alone
    (S241: it is read on every call). R13's second fixture (a capped parent with garbage usage) is a capped
    level that cannot be measured and now counts 0.0 (test_h3_an_unusable_own_usage_...)."""
    for tree in (
        _tree({_OWN: _lvl("abc", 1_000_000_000, _stat())}),
        _tree({_OWN: _lvl(None, None, None), _PAR: _lvl("abc", 3_000_000_000, _stat())}),
    ):
        _use_files(monkeypatch, tree)
        reads = _psutil_reads(monkeypatch, 777.0)
        assert _v2() is None
        assert reranker_mod._available_ram_headroom_bytes() == 777.0
        assert len(reads) == 1


def test_h3_a_partial_chain_returns_the_smaller_of_the_cgroup_figure_and_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """The own level is uncapped with a garbage usage; the capped parent gives 1.2 GB from its own counters.
    S241 (R13: psutil never read): psutil is read once and the smaller figure is returned."""
    tree = _tree({_OWN: _lvl("max", "abc", _stat()), _PAR: _lvl(4_000_000_000, 3_200_000_000, _R400)})
    _use_files(monkeypatch, tree)
    reads = _psutil_reads(monkeypatch, 9_000_000_000.0)
    assert reranker_mod._available_ram_headroom_bytes() == 1_200_000_000.0
    assert len(reads) == 1
    _psutil_reads(monkeypatch, 800_000_000.0)
    assert reranker_mod._available_ram_headroom_bytes() == 800_000_000.0


def test_h3_the_only_remaining_none_is_no_capped_level_and_psutil_failing_too(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_files(monkeypatch, _tree({_OWN: _lvl("max", 1_000_000_000, _stat())}))
    monkeypatch.setattr(reranker_mod, "_psutil_available_bytes", lambda: None)
    assert reranker_mod._available_ram_headroom_bytes() is None
    _use_files(monkeypatch, _one_level(_stat()))  # a capped level: its figure stands when psutil fails
    assert reranker_mod._available_ram_headroom_bytes() == 5_500_000_000.0


def test_h3_no_limit_anywhere_is_still_none_and_opens_no_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    _use_files(monkeypatch, _tree({_OWN: _lvl("max", 1, _stat()), _PAR: _lvl("max", 1, _stat())}), opened=opened)
    assert _v2() is None
    assert not any(p.endswith(("/memory.current", "/memory.stat")) for p in opened)


# ---- H4: the v1 reader walks its ancestors (S237) ----


def _v1_chain(own: dict, parent: dict | None = None, root: dict | None = None) -> dict[str, str]:
    levels = {_OWN: own}
    if parent is not None:
        levels[_PAR] = parent
    if root is not None:
        levels[""] = root
    return _v1tree(levels)


def test_h4_a_cap_that_only_the_parent_holds_is_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    """R12 returned None here (and the caller used psutil). The parent's own counters: 4.0 - (3.2 - 0.4) = 1.2 GB.
    S241: psutil is read once through the public function and the smaller figure wins."""
    tree = _v1_chain(
        _v1lvl(_SENTINEL, 500_000_000, _V1_ZERO), _v1lvl(4_000_000_000, 3_200_000_000, _V1_R400)
    )
    _use_files(monkeypatch, tree)
    reads = _psutil_reads(monkeypatch, 9_000_000_000.0)
    assert _v1() == 1_200_000_000.0
    assert reranker_mod._available_ram_headroom_bytes() == 1_200_000_000.0
    assert len(reads) == 1


def test_h4_the_smaller_headroom_of_own_and_parent_binds(monkeypatch: pytest.MonkeyPatch) -> None:
    own = _v1lvl(5_500_000_000, 5_000_000_000, _v1s())  # headroom 1.0 GB
    _use_files(monkeypatch, _v1_chain(own, _v1lvl(4_000_000_000, 3_500_000_000, _V1_ZERO)))
    assert _v1() == 500_000_000.0
    _use_files(monkeypatch, _v1_chain(own, _v1lvl(20_000_000_000, 1_000_000_000, _V1_ZERO)))
    assert _v1() == 1_000_000_000.0


def test_h4_a_larger_limit_that_siblings_filled_binds_over_a_smaller_limit_with_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    own = _v1lvl(3_000_000_000, 1_000_000_000, _V1_ZERO)  # headroom 2.0 GB
    _use_files(monkeypatch, _v1_chain(own, _v1lvl(5_000_000_000, 4_500_000_000, _V1_ZERO)))
    assert _v1() == 500_000_000.0  # the smallest-limit reading would say 2.0 GB


def test_h4_three_levels_take_the_minimum_each_with_its_own_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    def st(n: int) -> str:
        return _v1s(cache=n, mapped=0, dirty=0, writeback=0, shmem=0)

    own = _v1lvl(6_000_000_000, 2_000_000_000, st(300_000_000))  # 6.0 - 1.7 = 4.3 GB
    par = _v1lvl(5_000_000_000, 4_000_000_000, st(200_000_000))  # 5.0 - 3.8 = 1.2 GB
    root = _v1lvl(9_000_000_000, 8_000_000_000, st(500_000_000))  # 9.0 - 7.5 = 1.5 GB
    _use_files(monkeypatch, _v1_chain(own, par, root))
    assert _v1() == 1_200_000_000.0  # one level's counters applied to all would say 3.3 GB
    # the controller root binds
    root = _v1lvl(2_000_000_000, 1_900_000_000, st(100_000_000))  # 2.0 - 1.8 = 0.2 GB
    _use_files(monkeypatch, _v1_chain(own, par, root))
    assert _v1() == 200_000_000.0


def test_h4_levels_without_a_limit_are_never_read_for_usage_or_stat(monkeypatch: pytest.MonkeyPatch) -> None:
    levels = {
        f"{_OWN}/inner": _v1lvl(4_000_000_000, 1_000_000_000, _v1s()),  # 4.0 - 0.5 = 3.5 GB
        _OWN: _v1lvl(_SENTINEL, 9_000_000_000, _v1s()),
        _PAR: _v1lvl("garbage", 9_000_000_000, _v1s()),
        "": _v1lvl(None, 9_000_000_000, _v1s()),
    }
    opened: list[str] = []
    files = _v1tree(levels, own=f"{_OWN}/inner")
    _use_files(monkeypatch, files, opened=opened)
    assert _v1() == 3_500_000_000.0
    for unlimited in (_OWN, _PAR, ""):
        base = f"{_V1_ROOT}/{unlimited}" if unlimited else _V1_ROOT
        assert f"{base}/memory.usage_in_bytes" not in opened
        assert f"{base}/memory.stat" not in opened


@pytest.mark.parametrize("mode", ["absent", "garbage", "unreadable"])
def test_h4_a_capped_ancestor_with_an_unusable_usage_counts_zero_and_borrows_nothing(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """S240 (R13 stood in with the own usage and stat: 3.3 GB)."""
    own_stat = _v1s(cache=300_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    parent_usage = {"absent": None, "garbage": "abc", "unreadable": 3_000_000_000}[mode]
    tree = _v1_chain(_v1lvl(_SENTINEL, 1_000_000_000, own_stat), _v1lvl(4_000_000_000, parent_usage, _v1s()))
    _use_files(monkeypatch, tree, raising=(f"{_PAR}/memory.usage_in_bytes",) if mode == "unreadable" else ())
    assert _v1() == 0.0


def test_h4_an_unusable_own_usage_counts_zero_for_a_capped_own_level_and_leaves_an_uncapped_one_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _v1lvl(4_000_000_000, 3_200_000_000, _V1_R400)
    _use_files(monkeypatch, _v1_chain(_v1lvl(6_000_000_000, "abc", _v1s()), parent))
    assert _v1() == 0.0  # S240 (R13: the parent's 1.2 GB)
    _use_files(monkeypatch, _v1_chain(_v1lvl(_SENTINEL, "abc", _v1s()), parent))
    assert _v1() == 1_200_000_000.0  # an uncapped own level's usage is never read


def test_h4_the_clamps_hold_at_an_ancestor_level(monkeypatch: pytest.MonkeyPatch) -> None:
    big = _v1s(cache=900_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    own = _v1lvl(_SENTINEL, 1_000_000_000, _V1_ZERO)
    _use_files(monkeypatch, _v1_chain(own, _v1lvl(4_000_000_000, 100_000_000, big)))
    assert _v1() == 4_000_000_000.0  # reclaimable capped at the level's usage (100 MB)
    _use_files(monkeypatch, _v1_chain(own, _v1lvl(1_000_000_000, 3_000_000_000, _V1_ZERO)))
    assert _v1() == 0.0  # usage beyond the limit


def test_h4_no_limit_anywhere_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _v1_chain(_v1lvl(_SENTINEL, 1, _v1s()), _v1lvl(None, 1, None), _v1lvl("garbage", 1, None))
    _use_files(monkeypatch, tree)
    assert _v1() is None


def test_h4_only_the_total_keys_are_used_at_an_ancestor(monkeypatch: pytest.MonkeyPatch) -> None:
    plain = "cache 700000000\nmapped_file 50000000\ndirty 30000000\nwriteback 20000000\nshmem 100000000\n"
    tree = _v1_chain(_v1lvl(_SENTINEL, 1_000_000_000, _v1s()), _v1lvl(4_000_000_000, 3_000_000_000, plain))
    _use_files(monkeypatch, tree)
    assert _v1() == 1_000_000_000.0  # nothing is subtracted without the total_ lines


@pytest.mark.parametrize(
    ("raw", "is_limit"),
    [
        (0x7FFFFFFFFFFFF000, False),  # 4 KiB pages
        (0x7FFFFFFFFFFFC000, False),  # 16 KiB pages
        (0x7FFFFFFFFFFF0000, False),  # 64 KiB pages
        (0x7FFFFFFFFFFE0000, True),  # not a kernel value: below the sentinel it is a (huge) limit
    ],
)
def test_h4_the_unbounded_sentinel_is_page_size_independent(
    monkeypatch: pytest.MonkeyPatch, raw: int, is_limit: bool
) -> None:
    _use_files(monkeypatch, _v1_files(_v1_stat(), limit=str(raw), usage="1000000000"))
    assert (_v1() is not None) == is_limit


def test_h4_an_ancestor_reading_use_hierarchy_0_is_not_walked(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1 charges a cgroup into its parent only while the PARENT has hierarchy accounting on: the parent's own flag
    gates the parent, the child's flag (here "1", a decoy) does not."""
    parent = _v1lvl(4_000_000_000, 3_900_000_000, _V1_ZERO, hier="0")
    # own unlimited: the parent's cap is not this cgroup's -> no limit seen
    _use_files(monkeypatch, _v1_chain(_v1lvl(_SENTINEL, 1_000_000_000, _V1_ZERO, hier="1"), parent))
    assert _v1() is None
    # own capped, parent tighter but not charged: the own figure
    _use_files(monkeypatch, _v1_chain(_v1lvl(6_000_000_000, 1_000_000_000, _V1_ZERO, hier="1"), parent))
    assert _v1() == 5_000_000_000.0


@pytest.mark.parametrize("hier", ["1", None, "garbage", "unreadable"])
def test_h4_any_other_use_hierarchy_keeps_walking(monkeypatch: pytest.MonkeyPatch, hier: str | None) -> None:
    """The parent's flag is "1", absent, garbage or unreadable: it is walked; the child's flag "0" (a decoy) is ignored."""
    unreadable = hier == "unreadable"
    own = _v1lvl(_SENTINEL, 1_000_000_000, _V1_ZERO, hier="0")
    parent = _v1lvl(4_000_000_000, 3_900_000_000, _V1_ZERO, hier=None if unreadable else hier)
    tree = _v1_chain(own, parent)
    if unreadable:
        tree[f"{_V1_PAR}/memory.use_hierarchy"] = "0\n"  # would end the walk, but it cannot be read
    _use_files(monkeypatch, tree, raising=(f"{_V1_PAR}/memory.use_hierarchy",) if unreadable else ())
    assert _v1() == 100_000_000.0


def test_h4_use_hierarchy_is_read_once_per_ancestor_visited_including_the_root(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    tree = _v1_chain(
        _v1lvl(_SENTINEL, 1, _V1_ZERO, hier="1"), _v1lvl(_SENTINEL, 1, _V1_ZERO, hier="1"), _v1lvl(None, 1, None, hier="1")
    )
    _use_files(monkeypatch, tree, opened=opened)
    assert _v1() is None
    assert [p for p in opened if p.endswith("memory.use_hierarchy")] == [
        f"{_V1_PAR}/memory.use_hierarchy",
        f"{_V1_ROOT}/memory.use_hierarchy",
    ]


def test_h4_a_use_hierarchy_0_at_the_root_ends_the_walk_there(monkeypatch: pytest.MonkeyPatch) -> None:
    own = _v1lvl(_SENTINEL, 1_000_000_000, _V1_ZERO)
    parent = _v1lvl(4_000_000_000, 3_000_000_000, _V1_ZERO, hier="1")
    root = _v1lvl(1_100_000_000, 1_000_000_000, _V1_ZERO, hier="0")  # the parent is not charged into the root
    _use_files(monkeypatch, _v1_chain(own, parent, root))
    assert _v1() == 1_000_000_000.0  # the parent's cap counts; the root's (0.1 GB) does not


def test_h4_the_fallback_order_is_unchanged_with_a_walking_v1_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _v1_chain(_v1lvl(_SENTINEL, 1, _V1_ZERO), _v1lvl(None, 1, None))
    _use_files(monkeypatch, tree)
    monkeypatch.setattr(reranker_mod, "_psutil_available_bytes", lambda: 4242.0)
    assert reranker_mod._available_ram_headroom_bytes() == 4242.0


# ---- R14: S240 an unmeasurable capped level counts 0, S241 the smaller of cgroup and psutil, S243 container layouts ----


@pytest.mark.parametrize("mode", ["absent", "garbage", "unreadable", "negative"])
def test_j1_a_capped_own_level_with_an_unusable_usage_counts_zero_and_a_roomy_parent_cannot_loosen_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, mode: str
) -> None:
    """The R13-pinned exposure (R6a), removed by S240: the own level is capped at 6.0 GB, its usage cannot be
    used, and the parent (8.0 GB, 3.0 GB used, nothing reclaimable) would have given 5.0 GB. Now 0.0."""
    current = {"absent": None, "garbage": "abc", "unreadable": 900_000_000, "negative": "-5"}[mode]
    tree = _tree({_OWN: _lvl(6_000_000_000, current, _stat()), _PAR: _lvl(8_000_000_000, 3_000_000_000, _ZERO)})
    opened: list[str] = []
    _use_files(monkeypatch, tree, raising=(f"{_OWN}/memory.current",) if mode == "unreadable" else (), opened=opened)
    with caplog.at_level("DEBUG"):
        assert _v2() == 0.0
        assert _v2() == 0.0
    found = _warnings(caplog, "memory.current")
    assert len(found) == 1 and ("absent" in found[0].getMessage()) == (mode == "absent")
    if mode == "unreadable":
        assert found[0].exc_info and found[0].exc_info[0] is PermissionError
    assert f"{_V2_ROOT}/{_OWN}/memory.stat" not in opened, "an unusable usage never opens that level's stat"


@pytest.mark.parametrize(
    ("max_", "high", "raising", "expected"),
    [
        ("6000000000", "abc", (), 0.0),  # a numeric max, an unusable high
        ("6000000000", "4000000000", ("memory.high",), 0.0),  # the high cannot be read
        ("abc", "4000000000", (), 0.0),  # an unusable max, a numeric high
        ("max", "abc", (), 0.0),  # a usable "max" and an unusable high: capped with an unknown second limit
        ("max", "4000000000", ("memory.high",), 0.0),
        ("abc", "max", (), 0.0),  # the reverse
        ("6000000000", "4000000000", ("memory.max",), 0.0),
        ("6000000000", None, (), 5_500_000_000.0),  # an ABSENT high is no limit, silent
        ("6000000000", "max", (), 5_500_000_000.0),
        ("max", "max", (), None),  # nothing capped anywhere
    ],
)
def test_j1_a_limit_file_that_is_present_but_unusable_degrades_a_level_with_a_usable_limit_file(
    monkeypatch: pytest.MonkeyPatch, max_: str, high: str | None, raising: tuple[str, ...], expected: float | None
) -> None:
    tree = _one_level(_stat(), limit=max_)
    if high is not None:
        _hi(tree, _OWN, high)
    _use_files(monkeypatch, tree, raising=raising)
    assert _v2() == expected


def test_j1_an_ancestor_with_an_unusable_limit_file_is_degraded_too(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _tree({_OWN: _lvl("max", 1_000_000_000, _stat()), _PAR: _lvl(4_000_000_000, 1_000_000_000, _ZERO)})
    _use_files(monkeypatch, tree)
    assert _v2() == 3_000_000_000.0
    _hi(tree, _PAR, "abc")
    _use_files(monkeypatch, tree)
    assert _v2() == 0.0


def test_j1_a_degraded_level_opens_neither_its_usage_nor_its_stat(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    tree = _one_level(_stat())
    _hi(tree, _OWN, "abc")
    _use_files(monkeypatch, tree, opened=opened)
    assert _v2() == 0.0
    assert not any(p.endswith(("/memory.current", "/memory.stat")) for p in opened)


def test_j1_the_warn_once_key_is_the_file_and_the_kind(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Absent, not plain and unreadable are three kinds of one file: three warnings, each once per process."""
    with caplog.at_level("DEBUG"):
        _use_files(monkeypatch, _one_level(_stat(), current="abc"))
        assert _v2() == 0.0 and _v2() == 0.0
        assert len(_warnings(caplog, "memory.current")) == 1
        tree = _one_level(_stat())
        del tree[f"{_V2_ROOT}/{_OWN}/memory.current"]
        _use_files(monkeypatch, tree)
        assert _v2() == 0.0 and _v2() == 0.0
        assert len(_warnings(caplog, "memory.current")) == 2
        _use_files(monkeypatch, _one_level(_stat()), raising=(f"{_OWN}/memory.current",))
        assert _v2() == 0.0 and _v2() == 0.0
        assert len(_warnings(caplog, "memory.current")) == 3


def test_j1_a_level_with_no_usable_limit_file_and_an_absent_second_file_is_silent_about_the_absent_one(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An absent memory.high or memory.stat is never warned; only the unusable file is."""
    tree = _one_level(None)
    _use_files(monkeypatch, tree)
    with caplog.at_level("DEBUG"):
        assert _v2() == 5_000_000_000.0
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_j2_v1_a_capped_level_with_an_unusable_usage_counts_zero_and_the_minimum_includes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = _v1lvl(5_500_000_000, 5_000_000_000, _v1s())  # headroom 1.0 GB
    _use_files(monkeypatch, _v1_chain(good, _v1lvl(8_000_000_000, 3_000_000_000, _V1_ZERO)))
    assert _v1() == 1_000_000_000.0
    _use_files(monkeypatch, _v1_chain(good, _v1lvl(8_000_000_000, "abc", _V1_ZERO)))
    assert _v1() == 0.0
    opened: list[str] = []
    _use_files(monkeypatch, _v1_chain(good, _v1lvl(8_000_000_000, "abc", _V1_ZERO)), opened=opened)
    _v1()
    assert f"{_V1_PAR}/memory.stat" not in opened


def test_j2_v1_an_unreadable_limit_file_is_not_a_cap_and_the_level_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The v1 counterpart of the pinned exposure: a limit file that cannot be read gives no evidence of a cap."""
    tree = _v1_chain(_v1lvl(5_500_000_000, 5_000_000_000, _v1s()), _v1lvl(4_000_000_000, 3_200_000_000, _V1_R400))
    _use_files(monkeypatch, tree, raising=(f"{_V1_OWN}/memory.limit_in_bytes",))
    assert _v1() == 1_200_000_000.0


def test_j1_an_unreadable_limit_file_beside_a_usable_one_degrades_and_the_unusable_only_level_is_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The unreadable (PermissionError) variants of the limit-file rules, at an ancestor and at the own level."""
    # an ancestor: numeric max, memory.high unreadable -> that ancestor is degraded
    tree = _tree({_OWN: _lvl("max", 1_000_000_000, _stat()), _PAR: _lvl(4_000_000_000, 1_000_000_000, _ZERO)})
    _hi(tree, _PAR, "3000000000")
    _use_files(monkeypatch, tree, raising=(f"{_PAR}/memory.high",))
    with caplog.at_level("DEBUG"):
        assert _v2() == 0.0
        assert _v2() == 0.0
    found = _warnings(caplog, "memory.high")
    assert len(found) == 1 and found[0].exc_info and found[0].exc_info[0] is PermissionError
    # the own level: memory.max absent and memory.high unreadable -> no usable limit file, skipped, the parent binds
    tree = _tree({_OWN: _lvl(None, 1_000_000_000, _stat()), _PAR: _lvl(8_000_000_000, 3_000_000_000, _ZERO)})
    _hi(tree, _OWN, "2000000000")
    _use_files(monkeypatch, tree, raising=(f"{_OWN}/memory.high",))
    assert _v2() == 5_000_000_000.0


def test_j2_v1_a_capped_own_level_without_a_usage_file_counts_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_files(monkeypatch, _v1_chain(_v1lvl(5_500_000_000, None, _v1s())))
    assert _v1() == 0.0


def test_j1_a_chain_with_no_capped_level_has_no_figure_of_its_own() -> None:
    """The helper itself never turns "nothing capped" into a number: None, so the caller falls back (a 0.0 here
    would be read as a fully used cap)."""
    assert reranker_mod._cgroup_chain_headroom([], "memory.current", reranker_mod._RECLAIM_KEYS_V2) is None  # noqa: SLF001


def test_j2_v1_an_unreadable_use_hierarchy_warns_that_the_walk_continues_not_that_a_level_counts_zero(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tree = _v1_chain(_v1lvl(5_500_000_000, 5_000_000_000, _v1s()), _v1lvl(4_000_000_000, 3_500_000_000, _V1_ZERO, "0"))
    _use_files(monkeypatch, tree, raising=(f"{_V1_PAR}/memory.use_hierarchy",))
    with caplog.at_level("DEBUG"):
        assert _v1() == 500_000_000.0  # the unreadable flag keeps the walk going: the parent binds
    found = _warnings(caplog, "memory.use_hierarchy")
    assert len(found) == 1 and "keeps going" in found[0].getMessage() and "no headroom" not in found[0].getMessage()


# ---- J3: the headroom is the smaller of the cgroup figure and psutil (S241) ----


@pytest.mark.parametrize(
    ("cgroup", "host", "expected"),
    [(1e9, 2e9, 1e9), (3e9, 2e9, 2e9), (2e9, 2e9, 2e9), (0.0, 2e9, 0.0), (1e9, 0.0, 0.0), (1e9, None, 1e9), (None, 2e9, 2e9), (None, None, None)],
)
def test_j3_the_headroom_is_the_smaller_of_the_cgroup_figure_and_psutil(
    monkeypatch: pytest.MonkeyPatch, cgroup: float | None, host: float | None, expected: float | None
) -> None:
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: cgroup)
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: None)
    reads = _psutil_reads(monkeypatch, host)
    assert reranker_mod._available_ram_headroom_bytes() == expected
    assert len(reads) == 1, "psutil is read exactly once per call"


def test_j3_a_v1_figure_is_bounded_by_psutil_too_and_v2_is_tried_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: 5e9)
    _psutil_reads(monkeypatch, 2e9)
    assert reranker_mod._available_ram_headroom_bytes() == 2e9
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: 4e9)
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: pytest.fail("v1 after a v2 figure"))
    _psutil_reads(monkeypatch, 9e9)
    assert reranker_mod._available_ram_headroom_bytes() == 4e9


def test_j3_a_cap_above_the_hosts_free_memory_gives_no_extra_memory_through_the_real_readers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real chain (limit 64 GB, 1 GB used: 63 GB) against a host with 2 GB available."""
    tree = _tree({_OWN: _lvl(64_000_000_000, 1_000_000_000, _ZERO)})
    _use_files(monkeypatch, tree)
    _psutil_reads(monkeypatch, 2_000_000_000.0)
    assert _v2() == 63_000_000_000.0
    assert reranker_mod._available_ram_headroom_bytes() == 2_000_000_000.0


def test_j3_a_degraded_level_gives_zero_through_the_public_function_whatever_psutil_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_files(monkeypatch, _one_level(_stat(), current="abc"))
    _psutil_reads(monkeypatch, 9_000_000_000.0)
    assert reranker_mod._available_ram_headroom_bytes() == 0.0


def test_j3_a_zero_headroom_from_the_public_function_narrows_the_width_to_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_files(monkeypatch, _one_level(_stat(), current="abc"))
    _psutil_reads(monkeypatch, 9_000_000_000.0)
    figure = reranker_mod._available_ram_headroom_bytes()
    est = reranker_mod.RerankCostEstimate(0.0, 1e-5, _per_token_fit(), measured_batches=3)
    assert reranker_mod.fit_rerank_width([400] * 50, [50] * len(ANCHOR_POOL), est, 4.0, figure) == 0


# ---- J5: S243, the path /proc/self/cgroup names is absent under the mounted tree (container layout) ----

_CONTAINER = "docker/abc"


def test_j5_v2_in_the_container_layout_the_mount_root_is_read_with_its_own_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_stat = _stat(cache=300_000_000, mapped=0, dirty=0, writeback=0, shmem=0)
    opened: list[str] = []
    _use_files(monkeypatch, _tree({_GRAND: _lvl(2_000_000_000, 1_500_000_000, root_stat)}, own=_CONTAINER), opened=opened)
    assert _v2() == 800_000_000.0
    # root unbounded: nothing capped, the v2 reader is None and the public function is psutil's figure
    _use_files(monkeypatch, _tree({_GRAND: _lvl("max", 1_500_000_000, root_stat)}, own=_CONTAINER))
    _psutil_reads(monkeypatch, 4_000_000_000.0)
    assert _v2() is None
    assert reranker_mod._available_ram_headroom_bytes() == 4_000_000_000.0
    # no cgroup file at all
    _use_files(monkeypatch, {"/proc/self/cgroup": f"0::/{_CONTAINER}\n"})
    assert _v2() is None


@pytest.mark.parametrize("hier", ["0", "1", None])
def test_j5_v1_in_the_container_layout_the_mount_root_is_the_own_level_whatever_its_use_hierarchy(
    monkeypatch: pytest.MonkeyPatch, hier: str | None
) -> None:
    """R13 treated the root as an ancestor and dropped it when its use_hierarchy read "0": the container cap was
    invisible. Now the first present level (here the root) is the process's own level and is never gated."""
    root = _v1lvl(2_000_000_000, 1_500_000_000, _v1s(cache=300_000_000, mapped=0, dirty=0, writeback=0, shmem=0), hier)
    opened: list[str] = []
    _use_files(monkeypatch, _v1tree({"": root}, own=_CONTAINER), opened=opened)
    assert _v1() == 800_000_000.0
    probes = [p for p in opened if p.startswith(f"{_V1_ROOT}/docker")]
    assert probes == [f"{_V1_ROOT}/{_CONTAINER}/memory.limit_in_bytes", f"{_V1_ROOT}/docker/memory.limit_in_bytes"]


def test_j5_v1_the_first_present_level_is_the_own_level_and_a_later_level_is_walked_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _v1lvl(4_000_000_000, 3_200_000_000, _V1_R400, "0")
    _use_files(monkeypatch, _v1tree({"docker": parent, "": _v1lvl(_SENTINEL, 1, None)}, own=_CONTAINER))
    assert _v1() == 1_200_000_000.0  # the parent's use_hierarchy "0" does not gate the first present level
    # the first present level (here "docker", unbounded) is the own level; the capped root above it is walked
    tree = _v1tree(
        {"docker": _v1lvl(_SENTINEL, 1, None, "1"), "": _v1lvl(3_000_000_000, 2_000_000_000, _V1_ZERO)}, own=_CONTAINER
    )
    _use_files(monkeypatch, tree)
    assert _v1() == 1_000_000_000.0


def test_j5_v1_nothing_present_or_an_unbounded_root_is_none_and_psutil_stands(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_files(monkeypatch, {"/proc/self/cgroup": f"5:memory:/{_CONTAINER}\n"})
    assert _v1() is None
    _use_files(monkeypatch, _v1tree({"": _v1lvl(_SENTINEL, 1_500_000_000, _V1_ZERO, "0")}, own=_CONTAINER))
    assert _v1() is None
    _psutil_reads(monkeypatch, 4_000_000_000.0)
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: None)
    assert reranker_mod._available_ram_headroom_bytes() == 4_000_000_000.0


def test_j5_v1_a_permission_error_on_the_probe_is_not_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    """The own level exists but its limit file cannot be read: it is walked normally (skipped as not known to be
    capped), so the parent stays an ANCESTOR and its use_hierarchy "0" ends the walk before it (None). Treating the
    permission error as absence would make the parent the own level and give 1.2 GB."""
    tree = _v1_chain(_v1lvl(5_500_000_000, 5_000_000_000, _v1s()), _v1lvl(4_000_000_000, 3_200_000_000, _V1_R400, "0"))
    _use_files(monkeypatch, tree, raising=(f"{_V1_OWN}/memory.limit_in_bytes",))
    assert _v1() is None
    _use_files(monkeypatch, tree)  # control: the own level present, its cap read, the parent gated by its "0"
    assert _v1() == 1_000_000_000.0


def test_j5_v1_on_real_files_the_container_layout_and_a_real_permission_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Real open(), real FileNotFoundError and a real PermissionError (chmod 000) on a temporary v1 tree."""
    import os

    root = tmp_path / "memory"
    root.mkdir()
    (root / "memory.limit_in_bytes").write_text("2000000000\n")
    (root / "memory.usage_in_bytes").write_text("1500000000\n")
    (root / "memory.use_hierarchy").write_text("0\n")
    (root / "memory.stat").write_text(_v1s(cache=300_000_000, mapped=0, dirty=0, writeback=0, shmem=0))
    monkeypatch.setattr(reranker_mod, "_CGROUP_V1_ROOT", str(root))
    monkeypatch.setattr(reranker_mod, "_read_proc_self_cgroup_v1_memory_path", lambda: "docker/abc")
    monkeypatch.setattr(sys, "platform", "linux")
    assert _v1() == 800_000_000.0  # the path is absent under the tree: the root (use_hierarchy 0) is the own level
    own = root / "docker" / "abc"
    own.mkdir(parents=True)
    (own / "memory.limit_in_bytes").write_text(_SENTINEL + "\n")
    (own / "memory.usage_in_bytes").write_text("1\n")
    assert _v1() is None  # the own level exists and is unbounded; the root's use_hierarchy 0 ends the walk
    # The permission half needs POSIX permission bits and a non-root user: Windows has no `os.geteuid` and
    # `chmod(0)` there does not deny reads. cgroups are Linux-only, so this is a guard on a facility that only
    # exists where the bits do; the container-layout half above runs on every OS.
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        (own / "memory.limit_in_bytes").chmod(0)
        assert _v1() is None  # a permission error is not absence: the root is not promoted to the own level
        (own / "memory.limit_in_bytes").chmod(0o644)


def test_j7_the_fallback_docstrings_state_the_minimum_with_psutil_and_the_s240_s242_s243_rules() -> None:
    avail = " ".join((reranker_mod._available_ram_headroom_bytes.__doc__ or "").split())
    v2 = " ".join((reranker_mod._cgroup_v2_memory_headroom_bytes.__doc__ or "").split())
    v1 = " ".join((reranker_mod._cgroup_v1_memory_headroom_bytes.__doc__ or "").split())
    chain = " ".join((reranker_mod._cgroup_chain_headroom.__doc__ or "").split())
    width = " ".join((reranker_mod.fit_rerank_width.__doc__ or "").split())
    assert "SMALLER of two figures" in avail and "only when no cgroup level could be measured" not in avail
    assert "has headroom 0" in v2 and "never borrowed" in v2 and "mount root" in v2
    assert "first existing level" in v1 and "headroom 0" in v1
    assert "never borrowed" in chain or "never borrow" in chain
    assert "within_measured" in width and "S242" in width
    from brain.memory import ram_fit

    fit_doc = " ".join((ram_fit.RamFit.within_measured.__doc__ or "").split())
    assert "no more pairs" in fit_doc and "no longer a padded length" in fit_doc


# ---- H6: the callers see the same float | None ----


def _per_token_fit():
    from brain.memory import ram_fit

    return ram_fit.RamFit(a=0.0, b=1_000.0, c=0.0, shapes=4, samples=4, max_pairs=2, max_length=1024)


def test_h6_a_partial_chain_figure_narrows_the_width_exactly_like_the_same_plain_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The own level is uncapped with a garbage usage; the capped parent leaves 5.0 MB. S241: psutil is read
    (patched to a larger figure) and the cgroup figure wins."""
    tree = _tree({_OWN: _lvl("max", "abc", _stat()), _PAR: _lvl(8_000_000_000, 7_995_000_000, _ZERO)})
    _use_files(monkeypatch, tree)
    _psutil_reads(monkeypatch, 9_000_000_000.0)
    figure = reranker_mod._available_ram_headroom_bytes()
    assert figure == 5_000_000.0
    est = reranker_mod.RerankCostEstimate(0.0, 1e-5, _per_token_fit(), measured_batches=3)
    cand, anchors = [400] * 50, [50] * len(ANCHOR_POOL)
    narrowed = reranker_mod.fit_rerank_width(cand, anchors, est, 4.0, figure)
    assert narrowed == reranker_mod.fit_rerank_width(cand, anchors, est, 4.0, 5_000_000.0)
    assert 0 < narrowed < reranker_mod.fit_rerank_width(cand, anchors, est, 4.0, None)


def test_h6_a_headroom_of_exactly_zero_allows_no_rerank_width() -> None:
    est = reranker_mod.RerankCostEstimate(0.0, 1e-5, _per_token_fit(), measured_batches=3)
    cand, anchors = [400] * 50, [50] * len(ANCHOR_POOL)
    assert reranker_mod.fit_rerank_width(cand, anchors, est, 4.0, None) >= reranker_mod.RERANK_MIN_REAL_CANDIDATES
    assert reranker_mod.fit_rerank_width(cand, anchors, est, 4.0, 0.0) == 0


def _single_pair_fit(max_pairs: int = 1):
    from brain.memory import ram_fit

    return ram_fit.RamFit(a=0.0, b=1_000.0, c=0.0, shapes=4, samples=4, max_pairs=max_pairs, max_length=1024)


def test_j4_a_fit_with_single_pairs_only_sizes_no_batch_of_the_width_fit_even_with_endless_room() -> None:
    """S242: the same numbers with a measured 2-pair batch (cap lifted) give the time-bound width; with single
    pairs only the width is below the S5 minimum, so the caller takes the cosine path. No headroom reading means
    no RAM term and no cap (the time-only width `_is_stuck` relies on)."""
    cand, anchors = [400] * 50, [50] * len(ANCHOR_POOL)
    capped = reranker_mod.RerankCostEstimate(0.0, 1e-5, _single_pair_fit(1), measured_batches=3)
    lifted = reranker_mod.RerankCostEstimate(0.0, 1e-5, _single_pair_fit(2), measured_batches=3)
    minimum = reranker_mod.RERANK_MIN_REAL_CANDIDATES
    assert reranker_mod.fit_rerank_width(cand, anchors, capped, 4.0, 1e15) < minimum
    assert reranker_mod.fit_rerank_width(cand, anchors, lifted, 4.0, 1e15) >= minimum
    assert reranker_mod.fit_rerank_width(cand, anchors, capped, 4.0, None) >= minimum
    assert reranker_mod.fit_rerank_width(cand, anchors, capped, 4.0, None) == reranker_mod.fit_rerank_width(
        cand, anchors, lifted, 4.0, None
    )


def test_j4_the_cap_refuses_a_single_pair_longer_than_any_measured_and_allows_one_within_it() -> None:
    capped = reranker_mod.RerankCostEstimate(0.0, 1e-5, _single_pair_fit(1), measured_batches=3)
    anchors = [50] * len(ANCHOR_POOL)
    assert reranker_mod.fit_rerank_width([1024], anchors, capped, 4.0, 1e15, max_real=1) == 1
    assert reranker_mod.fit_rerank_width([1025], anchors, capped, 4.0, 1e15, max_real=1) == 0


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
    assert "MINIMUM" in v1 and "own setting" in v1 and "use_hierarchy" in v1 and "ancestor" in v1
    assert "NOT visible" not in v1 and "Only the process's OWN cgroup is read" not in v1
    assert "already hierarchical" not in v1 and "no walk-up is needed" not in v1
    assert "memory.high" in v2 and "plain" in v2 and "has a usable numeric limit" in " ".join(v2.split())
    assert "psutil" in avail and "memory.high" in avail
    for doc in (v2, v1, helper, avail):
        assert "matches psutil" not in doc.lower() and "matches the host" not in doc.lower()


def test_g12_no_text_in_brain_still_calls_inactive_file_the_reclaimable_term(tmp_path) -> None:
    assert "brain/memory/reranker.py" in brain_sources()
    assert [rel for rel, text in brain_sources().items() if "inactive_file" in text] == []
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


def test_available_ram_headroom_takes_cgroup_v2_before_v1_and_the_smaller_of_it_and_psutil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


# ---------------------------------------------------------------------------
# S247: every rerank call site goes through the RAM bound. This guards ADDITIONS: a new `.rerank(` or
# `.rerank_timed(` call anywhere in brain/ fails here until its bound is decided and listed in
# the records' call-sites.md (R15), with the test that proves it.
# ---------------------------------------------------------------------------

_RERANK_CALL_SITES = {
    # leaf provider methods: the calls being bounded, not call sites that decide
    ("brain/memory/reranker.py", "RerankerProvider.rerank_timed"): 1,
    ("brain/memory/reranker.py", "CrossEncoderProvider.rerank"): 1,
    ("brain/memory/reranker.py", "CrossEncoderProvider.rerank_timed"): 1,
    # the ladder's rungs (S245: headroom checked before every rung, the first included)
    ("brain/memory/reranker.py", "_run_rung"): 1,
    # the warm-up reranks (the ladder's own: S245 check first; the recall callers: S247 `bounded=True`)
    ("brain/memory/reranker.py", "_warm_up_once"): 1,
    # the floor calibration's bundled pairs (S247: the kept memory model and the live headroom)
    ("brain/memory/reranker.py", "normalize_bundled_pairs_against_anchors"): 1,
    # the per-recall gate normalization: reached only through `_measured_rerank` (the width fit's RAM term,
    # the seed bound, the re-measure job's explicit check); two calls (the no-anchor no-op and the anchored one)
    ("brain/memory/reranker.py", "normalize_against_anchors"): 2,
}



class _RerankCallFinder(_ast.NodeVisitor):
    """The (enclosing class.function) of every `.rerank(` / `.rerank_timed(` call in one module."""

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.scopes: list[str] = []

    def _scope(self, node) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: _ast.ClassDef) -> None:
        self._scope(node)

    def visit_FunctionDef(self, node: _ast.FunctionDef) -> None:
        self._scope(node)

    def visit_AsyncFunctionDef(self, node: _ast.AsyncFunctionDef) -> None:
        self._scope(node)

    def visit_Call(self, node: _ast.Call) -> None:
        func = node.func
        if isinstance(func, _ast.Attribute) and func.attr in ("rerank", "rerank_timed"):
            self.scopes.append(".".join(self.stack))
        self.generic_visit(node)


def _rerank_call_sites(root=None) -> dict[tuple[str, str], int]:
    """(file, enclosing function) -> how many `.rerank(` / `.rerank_timed(` calls it holds.

    ``root`` None scans this repository's brain/ from the process-wide parsed copy; a path scans ``root / "brain"``."""
    if root is None:
        modules = brain_trees().items()
    else:
        modules = (
            (path.relative_to(root).as_posix(), _ast.parse(path.read_text(encoding="utf-8")))
            for path in sorted((root / "brain").rglob("*.py"))
        )
    found: dict[tuple[str, str], int] = {}
    for rel, tree in modules:
        finder = _RerankCallFinder()
        finder.visit(tree)
        for scope in finder.scopes:
            key = (rel, scope)
            found[key] = found.get(key, 0) + 1
    return found


def test_the_rerank_call_sites_are_exactly_the_audited_set() -> None:
    found = _rerank_call_sites()
    assert found == _RERANK_CALL_SITES, (
        "S247: a rerank call site was added or removed. Decide how the new site goes through the RAM bound "
        f"and update this set and the call-sites record. Found: {sorted(found.items())}"
    )


def test_the_call_site_scan_sees_a_call_it_has_not_been_told_about(tmp_path) -> None:
    """The scan can fail: a throwaway tree with a rerank call is reported, enclosing function included."""
    (tmp_path / "brain" / "memory").mkdir(parents=True)
    (tmp_path / "brain" / "memory" / "stray.py").write_text(
        "class Holder:\n    def go(self, p):\n        return p.rerank('q', ['d']) + p.rerank_timed('q', ['d'])\n"
    )
    assert _rerank_call_sites(tmp_path) == {("brain/memory/stray.py", "Holder.go"): 2}


def test_the_ram_bound_texts_state_the_s245_and_s247_rules() -> None:
    from brain.memory import floor_calibration

    def flat(obj) -> str:
        return " ".join((obj.__doc__ or "").split())

    measure = flat(reranker_mod.measure_ram_figure)
    assert "records the completion marker ONLY IF the window then holds a measured batch of more than one pair" in measure
    assert "before the warm-up and before EVERY rung, the first included" in measure
    assert "without a headroom check" not in measure
    assert "BEFORE the provider is built" in flat(reranker_mod.run_ram_figure)
    assert "`RerankDeferred` is raised BEFORE that pair's call" in flat(reranker_mod.normalize_bundled_pairs_against_anchors)
    assert "RerankDeferred" in flat(floor_calibration.get_bootstrap_floor)
    assert "seed_ram_fit" in flat(reranker_mod.fit_rerank_width)
    assert "S247" in flat(reranker_mod.RerankDeferred) and "S247" in flat(reranker_mod._batch_fits_ram_bound)  # noqa: SLF001
    assert "S247" in flat(reranker_mod._warm_up_once) and "bounded" in flat(reranker_mod._warm_up_once)  # noqa: SLF001


# ---------------------------------------------------------------------------
# S249: has fastembed's construction got the model file in the cache, or does it download it?
# ---------------------------------------------------------------------------


def _hf_cache(root, repo: str, files: list[str]) -> None:
    folder = root / ("models--" + repo.replace("/", "--"))
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text("abc123")
    for name in files:
        path = folder / "snapshots" / "abc123" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


def _production_fp16_id() -> str:
    from brain.bridge.model_tier import MODEL_RERANKER, MODEL_RERANKER_FP16

    reranker_mod._register_fp16_reranker_model(MODEL_RERANKER_FP16, MODEL_RERANKER)  # noqa: SLF001
    return MODEL_RERANKER_FP16


def test_the_cached_check_follows_the_registered_repo_and_file_not_the_model_id(tmp_path) -> None:
    from brain.bridge.model_tier import MODEL_RERANKER

    model_id = _production_fp16_id()
    assert model_id != MODEL_RERANKER
    assert reranker_mod._model_file_cached(model_id, tmp_path) is False, "an empty cache: construction downloads"  # noqa: SLF001
    _hf_cache(tmp_path, MODEL_RERANKER, ["onnx/model.onnx"])
    assert reranker_mod._model_file_cached(model_id, tmp_path) is False, "only the fp32 file is cached: the fp16 export is fetched"  # noqa: SLF001
    assert reranker_mod._model_file_cached(MODEL_RERANKER, tmp_path) is True  # noqa: SLF001
    (tmp_path / ("models--" + MODEL_RERANKER.replace("/", "--")) / "snapshots" / "abc123" / "onnx" / "model_fp16.onnx").write_text("x")
    assert reranker_mod._model_file_cached(model_id, tmp_path) is True  # noqa: SLF001


def test_the_cached_check_cannot_decide_for_an_unknown_model_or_on_an_error(tmp_path, monkeypatch, caplog) -> None:
    assert reranker_mod._model_file_cached("nobody/unknown-reranker", tmp_path) is None  # noqa: SLF001
    import huggingface_hub

    def _boom(*args, **kwargs):
        raise OSError("cache unreadable")

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _boom)
    with caplog.at_level("WARNING"):
        assert reranker_mod._model_file_cached(_production_fp16_id(), tmp_path) is None  # noqa: SLF001
    assert any("already cached" in r.getMessage() for r in caplog.records)
