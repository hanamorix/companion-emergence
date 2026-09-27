"""INC-5 C2/C3/C14 — judge lifetime + offline load (spec §2, S2/S11/S19/S27).

Real ``bge-reranker-v2-m3`` (torch), real release mechanism
(``relevance_judge.release_judge``), run in a FRESH SUBPROCESS per test
(mirrors ``test_relevance_judge.py``'s
``test_label_calibration_sample_with_persona_knob_does_not_import_torch``
convention, and ``test_search_via_bridge.py``'s real-process convention) —
NOT in-process, because this repo's ``tests/conftest.py`` autouse fixture
monkeypatches ``build_judge_provider`` to a zero-network fake for every
in-process test that isn't marked ``requires_network``, and this file
specifically needs the REAL factory. A fresh subprocess never loads that
conftest fixture at all, so no marker dance is needed to "opt out" of it —
the real judge is simply what a plain import gets.

Marked ``requires_models`` + ``integration`` (mirrors ``test_search_via_
bridge.py``'s convention exactly: the project's local pre-check gate is
``-m "not live and not requires_claude_cli and not integration"``, so
``integration`` alone already excludes this from it) — heavy (real ~2.1GB
torch model load), run locally by hand and in the dedicated 3-OS CI job
this increment adds (``.github/workflows/judge-release-rss.yml``), never in
the default suite.

Scope note (C2's self-tune "finish" arm): this file drives the self-tune
FINISH arm via ``judge_lora.build_lora_retrain_fn`` (the real weight-retrain
mechanism ``judge_selftune._run_weight_retrain`` calls) directly, rather than
driving the full weekly tick end-to-end (which would additionally require
>200 real Haiku-labeled ``calibration_log`` rows just to clear the tick's own
gate — a fixture-construction cost belonging to that gate's own tests,
already covered offline in ``test_judge_selftune*.py``). What THIS file
proves is the thing C2 is actually about: after the real judge-lifecycle
functions ``_run_weight_retrain`` calls (``build_judge_provider`` for the
champion, ``judge_lora``'s real LoRA training path for the challenger) go out
of scope, ``release_judge()`` — the exact function ``judge_selftune.
_run_judge_selftune_tick``'s ``finally`` calls — brings bridge-process RSS
back down. The gate/orchestration logic around this (200-decision gate,
tune-grade selection, champion/challenger accept/revert) is a SEPARATE
concern, already exercised offline elsewhere.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from brain.bridge.model_tier import TIER_RELEVANCE_JUDGE, model_for_tier
from brain.memory.relevance_judge import _hf_cached
from brain.paths import get_cache_dir

pytestmark = [pytest.mark.requires_models, pytest.mark.integration]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TIMEOUT_S = 300

# CI follow-up (2026-09-27): test.yml's own job runs `uv run pytest -v
# --tb=short` with NO `-m` filter at all (a repo-wide CI policy, out of this
# fix's scope to change) -- so this module's `requires_models`/`integration`
# markers alone did NOT keep it out of that job, and it downloaded the real
# ~2.1GB judge model there. This module's own dedicated CI job
# (`.github/workflows/judge-release-rss.yml`) warms the local HF cache
# BEFORE running these tests; test.yml's jobs never do. So: skip the whole
# module, without ever touching the network, unless the configured judge id
# is ALREADY present in the local HF cache -- `_hf_cached` is exactly the
# no-network cache probe `offline_load_kwargs`/production itself uses (S2/
# S19), so "skip" here means precisely "this run would have had to hit the
# network to get a real judge", never a false skip of a run that could have
# gone offline. A machine (this dev box, or the dedicated workflow after its
# warm step) that already has the model cached still runs these tests.
_JUDGE_MODEL_ID = model_for_tier(TIER_RELEVANCE_JUDGE)
if not _hf_cached(_JUDGE_MODEL_ID, get_cache_dir()):
    pytest.skip(
        f"{_JUDGE_MODEL_ID!r} not in the local HF cache ({get_cache_dir()}) -- "
        "skipping without a network request; this module needs the model "
        "pre-warmed (see .github/workflows/judge-release-rss.yml)",
        allow_module_level=True,
    )


def _run_subprocess_script(script: str) -> dict:
    """Run `script` in a fresh `python -P` subprocess (mirrors test_search_
    via_bridge.py's `-P` convention — no cwd-shadowing of `brain`), parse the
    LAST stdout line as the JSON result dict this file's scripts always print,
    and fail loudly (with full stdout/stderr) on any non-zero exit."""
    proc = subprocess.run(
        [sys.executable, "-P", "-c", script],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"no stdout at all — stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    import json

    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# Shared RSS-trace harness, embedded in every subprocess script: a background
# poller thread samples this PROCESS's own RSS (psutil, self-measurement —
# C2's "measured with psutil in-process") every 20ms for the duration of the
# job under test, so `peak` approximates "RSS right after the judge loaded"
# without needing test-only instrumentation inside production code.
# ---------------------------------------------------------------------------
_RSS_TRACE_PRELUDE = textwrap.dedent(
    """
    import json
    import os
    import threading
    import time

    import psutil

    _proc = psutil.Process(os.getpid())
    _trace = []
    _stop = threading.Event()

    def _poll():
        while not _stop.is_set():
            _trace.append(_proc.memory_info().rss)
            time.sleep(0.02)

    _poll_thread = threading.Thread(target=_poll, daemon=True)

    def _settled_rss(samples: int = 8, interval: float = 0.05) -> int:
        # Round-7 red-team MAJOR (this file's own cold review): a single
        # instantaneous psutil read right after release_judge() returns is a
        # single-shot measurement -- exactly this project's own documented
        # cold-cache-trap failure mode (single-shot-timing-cold-cache-trap.md),
        # here in the other direction: a transient upward blip (a GC-thread
        # bookkeeping allocation, an unrelated background allocation on a
        # shared/loaded CI runner) could inflate ONE reading enough to trip a
        # bound sitting close to the observed retention ratio. Sampling over a
        # short settle window and taking the MINIMUM is the direct fix for
        # THIS mechanism (a transient high reading), as opposed to repeating
        # the whole expensive job N times (which would guard against a
        # different failure mode -- run-to-run variance in how much the
        # judge itself retains -- at a real cost: C2b's real LoRA retrain
        # alone takes ~2-4 minutes, so 3 full repeats per OS per PR was
        # judged the wrong trade for this specific, timing-shaped risk).
        readings = []
        for _ in range(samples):
            time.sleep(interval)
            readings.append(_proc.memory_info().rss)
        return min(readings)

    def _warm_libs() -> None:
        # S81 owner ruling (RAM-spike-fix ledger, "CPU everywhere
        # (Recommended)"): every judge construction site now passes
        # `device="cpu"` explicitly (relevance_judge.py, judge_full_ft.py,
        # judge_lora.py) -- production never runs the judge on MPS, on any
        # platform. This test file previously monkeypatched
        # `torch.backends.mps.is_available` to force CPU here (GitHub's
        # macos-14 runners advertise MPS but only guarantee a small
        # unified-memory budget under it, which OOM'd loading this ~2.1GB
        # judge) -- that patch is no longer needed now that CPU is the
        # shipped path itself: this test now measures the real code, and
        # the macOS CI leg exercises the SAME device production actually
        # uses, closing the coverage gap an earlier round of this file
        # flagged as an open question (see relevance_judge.py's S81 comment
        # for the full mechanism/rationale).
        #
        # Warm the baseline. Round-2 CI red-team (C2 MAJOR): capturing
        # `rss_before` before `torch`/`sentence_transformers` are ever
        # imported means the libraries' own non-unloadable C-extension/
        # allocator-arena footprint (present for the rest of the process's
        # life regardless of `release_judge()`) got counted as part of the
        # judge's "load_delta", inflating the denominator the 25% bound is
        # measured against, and then counted AGAIN as "retained" once the
        # model itself was freed -- observed ubuntu/macOS CI retention of
        # ~26-27%, just over the 25% gate. Importing (not constructing) the
        # judge's own dependency chain here, before `rss_before` is
        # captured by each test, moves that library overhead into the
        # measurement's baseline, so `load_delta`/`rss_after` isolate the
        # MODEL's own weights -- what C2/S27 are actually about.
        from sentence_transformers import CrossEncoder  # noqa: F401 (import-only warm-up)
    """
)

# C2's retained-RSS bound, as a fraction of the judge's own load_delta
# (measured with the warmed baseline above, so it reflects only the model's
# own release, not torch/sentence-transformers' fixed library footprint).
#
# Round-2 CI follow-up (2026-09-27): kept at 1.5-criteria.md C2's own,
# already owner-ratified 25% -- NOT tightened to a new number. The root
# cause of the ~26-27% CI failures was the baseline placement (see
# `_warm_libs` above), not the bound: after that fix, local
# re-measurement on this machine (Linux, model cached) shows `test_c2a`
# load_delta=1603.2MB / retained=28.6MB (1.78%) and `test_c2b`
# load_delta=3380.3MB / retained=56.2MB (1.66%) -- both far under 25%, with
# no need to invent a tighter number. An earlier draft of this fix tightened
# this to 0.10 from only the Linux measurements above; a cold code red-team
# (agentId a604186324bde4711) correctly flagged that as generalizing a
# single-platform sample to macOS/Windows CI legs whose reclaim mechanism is
# explicitly documented (`judge-release-rss.yml`'s own header, S27) to
# differ from Linux's (`malloc_trim(0)` is Linux-only; macOS/Windows rely on
# the allocator's own free()-time reclaim) -- an assumption of cross-OS
# equivalence this criterion exists to test, not one to bake into its own
# gate. Restoring the criteria's original, already-negotiated 25% avoids
# that risk entirely while keeping the bite: it still fails at ~100%
# retained if `release_judge()` is removed, and the criteria doc
# (`1.5-criteria.md` C2) and this test's own docstrings now agree on one
# number instead of drifting to three.
_C2_RETAINED_FRACTION_BOUND = 0.25


def test_c2a_calibration_tick_finish_releases_judge_rss() -> None:
    """C2 (calibration finish arm): a real `_run_calibration_tick` call that
    builds the real torch judge (one unlabeled calibration_log row, a clearly
    non-ambiguous pair so no Haiku call is needed — a `FakeProvider` is
    injected anyway, purely so the tick never attempts `build_tier_provider`'s
    real Claude-CLI construction) ends with bridge-process RSS back down to
    <= (RSS before judge load) + 25% * (judge load delta) — S11/S27's
    release step (drop refs, gc.collect(), Linux malloc_trim(0)) actually
    frees the judge's RAM, not merely drops a Python reference.

    Fail-first (pre-INC-5): no `release_judge()` call exists at all — the
    cached judge in `_provider_cache` (relevance_judge.py:347) stays alive
    for the bridge's whole life, so RSS after the tick would sit at
    (RSS before) + (~100% of the load delta), failing the <=25% bound.
    """
    script = _RSS_TRACE_PRELUDE + textwrap.dedent(
        """
        import tempfile
        from pathlib import Path

        from brain.bridge.provider import FakeProvider
        from brain.bridge.supervisor import _run_calibration_tick
        from brain.memory.store import Memory, MemoryStore
        import brain.memory.relevance_judge as rj
        import brain.memory.reranker as reranker_mod

        # Isolate the JUDGE's own RSS delta from the tick's SEPARATE
        # floor-derivation step, which loads a genuinely different real
        # model (the jina reranker, via `build_reranker_provider`) that is
        # NOT released by this fix (S11: the embedder/reranker/matrix are
        # bridge-process singletons, kept for the bridge's whole life --
        # only the judge is finish/pause-released). Left unpatched, that
        # second real model load would inflate this test's `peak` (which
        # this test attributes entirely to the judge) while `release_judge`
        # correctly leaves the reranker resident, producing a false C2
        # failure that is actually this test's own measurement conflating
        # two different models, not a regression in the judge's release.
        # Failing the floor-derivation step is safe: `_run_calibration_tick`
        # wraps it in its own try/except specifically so a failure there
        # never blocks or undoes the judge-labeling step above it.
        def _no_reranker(*a, **kw):
            raise RuntimeError("reranker load disabled for this RSS-isolation test")

        reranker_mod.build_reranker_provider = _no_reranker

        persona_dir = Path(tempfile.mkdtemp())
        store = MemoryStore(persona_dir / "memories.db")
        # Same genuine pair as test_relevance_judge_real_model.py's
        # _GENUINE_PAIRS[0] (measured there, and independently confirmed
        # while building this file, to score ~-0.38 raw / ~0.41 sigmoid on
        # bge-reranker-v2-m3 -- outside the default [0.45, 0.55] ambiguous
        # band, so no Haiku tie-break call is needed).
        query = "how do I calm down when everything feels like too much"
        mem = Memory.create_new(
            content="deep breathing helps when you are feeling anxious",
            memory_type="conversation",
            domain="us",
        )
        store.create(mem)
        store.log_calibration_sample(
            query=query, candidate_ids=[mem.id], reranker_scores=[5.0], reranker_model_id="m",
        )
        store.close()

        _warm_libs()
        rss_before = _proc.memory_info().rss
        _poll_thread.start()
        _run_calibration_tick(persona_dir, provider=FakeProvider())
        _stop.set()
        _poll_thread.join(timeout=5)
        rss_after = _settled_rss()
        peak = max(_trace) if _trace else rss_after

        print(json.dumps({
            "rss_before": rss_before,
            "peak": peak,
            "rss_after": rss_after,
            "provider_cache_empty": rj._provider_cache == {},
        }))
        """
    )
    result = _run_subprocess_script(script)
    rss_before, peak, rss_after = result["rss_before"], result["peak"], result["rss_after"]
    load_delta = peak - rss_before
    bound = rss_before + _C2_RETAINED_FRACTION_BOUND * load_delta
    print(f"\nload_delta={load_delta / 1e6:.1f}MB rss_after-rss_before={(rss_after - rss_before) / 1e6:.1f}MB")
    assert load_delta > 200_000_000, (
        f"the judge doesn't look like it actually loaded (load_delta={load_delta / 1e6:.1f}MB) "
        "-- this test would pass vacuously if label_calibration_sample never built the real judge"
    )
    assert result["provider_cache_empty"], "release_judge() must clear the shared base-judge cache"
    assert rss_after <= bound, (
        f"rss_after={rss_after / 1e6:.1f}MB exceeds the C2 bound {bound / 1e6:.1f}MB "
        f"(rss_before={rss_before / 1e6:.1f}MB + {_C2_RETAINED_FRACTION_BOUND:.0%} of load_delta={load_delta / 1e6:.1f}MB)"
    )


def test_c2b_weight_retrain_finish_releases_judge_rss() -> None:
    """C2 (self-tune finish arm, scoped per this file's module docstring): a
    real champion build (`build_judge_provider()`) + a real, tiny LoRA
    weight-retrain (`judge_lora.build_lora_retrain_fn`, 1 epoch, 6 real
    triples -- the same mechanism `judge_selftune._run_weight_retrain` calls
    for the mid tier) followed by `release_judge()` brings RSS back down the
    same way as the calibration arm.

    Fail-first: identical mechanism to C2a -- without `release_judge()`,
    the cached base judge (used as the champion) stays resident.
    """
    script = _RSS_TRACE_PRELUDE + textwrap.dedent(
        """
        from brain.memory import judge_lora
        import brain.memory.relevance_judge as rj

        _warm_libs()
        rss_before = _proc.memory_info().rss
        _poll_thread.start()

        # Champion: the real base judge (mirrors _run_weight_retrain's
        # `current is None` branch, judge_selftune.py `base_judge = ...`).
        champion = rj.build_judge_provider()
        _ = champion.score("q", "d")
        champion_device = str(champion._model.device)

        # Challenger: a real, tiny LoRA retrain on a handful of real triples
        # (mirrors _run_weight_retrain's `_select_retrain` -> judge_lora path).
        triples = [
            ("how do I calm down", "deep breathing helps when anxious", "relevant"),
            ("how do I calm down", "the stock market closed higher today", "irrelevant"),
            ("what does Bob drink in the morning", "Bob starts his day with black coffee", "relevant"),
            ("what does Bob drink in the morning", "my cat knocked a glass off the counter", "irrelevant"),
            ("is it going to rain today", "the forecast shows a storm moving in", "relevant"),
            ("is it going to rain today", "the concert was rescheduled to next month", "irrelevant"),
        ]
        retrain_fn = judge_lora.build_lora_retrain_fn(
            champion.model_id(),
            target_modules=judge_lora.BGE_RERANKER_LORA_TARGET_MODULES,
            modules_to_save=judge_lora.BGE_RERANKER_LORA_MODULES_TO_SAVE,
            epochs=1,
        )
        label_fn = retrain_fn(triples)
        _ = label_fn(("a new query", "a new document"))

        del champion, retrain_fn, label_fn
        rj.release_judge()

        _stop.set()
        _poll_thread.join(timeout=5)
        rss_after = _settled_rss()
        peak = max(_trace) if _trace else rss_after

        print(json.dumps({
            "rss_before": rss_before,
            "peak": peak,
            "rss_after": rss_after,
            "provider_cache_empty": rj._provider_cache == {},
            "champion_device": champion_device,
        }))
        """
    )
    result = _run_subprocess_script(script)
    rss_before, peak, rss_after = result["rss_before"], result["peak"], result["rss_after"]
    load_delta = peak - rss_before
    bound = rss_before + _C2_RETAINED_FRACTION_BOUND * load_delta
    print(f"\nload_delta={load_delta / 1e6:.1f}MB rss_after-rss_before={(rss_after - rss_before) / 1e6:.1f}MB")
    assert load_delta > 200_000_000, (
        f"the champion+challenger don't look like they actually loaded (load_delta={load_delta / 1e6:.1f}MB)"
    )
    assert result["champion_device"] == "cpu", (
        f"expected the CPU-forcing patch to land the champion judge on cpu, got "
        f"{result['champion_device']!r} -- the torch.backends.mps.is_available patch may "
        "have been defeated (round-2 red-team finding 4: a separate lru_cache'd MPS check "
        "in transformers.utils.import_utils could lock in True before this patch runs)"
    )
    assert result["provider_cache_empty"], "release_judge() must clear the shared base-judge cache"
    assert rss_after <= bound, (
        f"rss_after={rss_after / 1e6:.1f}MB exceeds the C2 bound {bound / 1e6:.1f}MB "
        f"(rss_before={rss_before / 1e6:.1f}MB + {_C2_RETAINED_FRACTION_BOUND:.0%} of load_delta={load_delta / 1e6:.1f}MB)"
    )


def test_c3a_cached_judge_load_makes_zero_network_requests() -> None:
    """C3(a): with the configured judge id present in the local HF cache
    (this machine's real cache dir, per the module's own `get_cache_dir()`),
    building the judge via the production factory
    (`relevance_judge.build_judge_provider()`) makes ZERO socket connect()
    calls -- not merely "no full download": `offline_load_kwargs`'s module
    docstring documents a real transformers-library quirk this test guards
    against regressing (a `local_files_only=True` load that still issues one
    real HEAD request for `adapter_config.json` before falling back to the
    cache, unless `model_kwargs={"adapter_kwargs": {"local_files_only":
    True}}` is ALSO passed).

    Fail-first (pre-INC-5): `CrossEncoder(model_id, cache_folder=...)` with
    no offline kwargs at all issues real HEAD/GET requests to the Hub (O15)
    -> connect_calls > 0.
    """
    script = textwrap.dedent(
        """
        import json
        import socket

        _connect_calls = [0]
        _orig_connect = socket.socket.connect

        def _counting_connect(self, addr):
            _connect_calls[0] += 1
            raise OSError("network blocked in this test (C3a: expecting zero attempts)")

        socket.socket.connect = _counting_connect

        # Round-1 red-team (minor): 1.5-criteria.md C3 names TWO mechanisms
        # ("test blocks the network ... AND counts calls on huggingface_hub's
        # HTTP session") -- count huggingface_hub's own shared httpx.Client
        # calls too, not just the raw socket layer, so this test literally
        # matches the criterion rather than a reasonable-but-partial superset
        # of it.
        import huggingface_hub.utils._http as hf_http

        _hf_session_calls = [0]
        _orig_get_session = hf_http.get_session

        def _counting_get_session():
            session = _orig_get_session()
            if not getattr(session, "_rss_test_counted", False):
                _orig_request = session.request

                def _counting_request(*a, **kw):
                    _hf_session_calls[0] += 1
                    return _orig_request(*a, **kw)

                session.request = _counting_request
                session._rss_test_counted = True
            return session

        hf_http.get_session = _counting_get_session

        import brain.memory.relevance_judge as rj

        provider = rj.build_judge_provider()
        score = provider.score("hello", "world")

        print(json.dumps({
            "connect_calls": _connect_calls[0],
            "hf_session_calls": _hf_session_calls[0],
            "model_id": provider.model_id(),
            "score_is_float": isinstance(score, float),
            # S81 (owner ruling, "CPU everywhere (Recommended)"): every
            # judge construction site passes device="cpu" explicitly now
            # (relevance_judge.py), so this assertion measures the actual
            # shipped path rather than a test-only monkeypatch -- keep it
            # so a regression back to no device= (which would leave
            # sentence-transformers' own MPS-preferring default in force)
            # fails loudly here too, not just in the unit-level S81 bite
            # tests in test_relevance_judge.py.
            "device": str(provider._model.device),
        }))
        """
    )
    result = _run_subprocess_script(script)
    assert result["model_id"] == "BAAI/bge-reranker-v2-m3"
    assert result["score_is_float"]
    assert result["device"] == "cpu", (
        f"expected the S81 device='cpu' construction to land the judge on cpu, "
        f"got {result['device']!r}"
    )
    assert result["hf_session_calls"] == 0, (
        f"expected zero huggingface_hub HTTP-session calls for a cached judge load, "
        f"got {result['hf_session_calls']}"
    )
    assert result["connect_calls"] == 0, (
        f"expected zero network connects for a cached judge load, got {result['connect_calls']}"
    )


def test_c3b_uncached_judge_id_attempts_network_load() -> None:
    """C3(b): with a cache dir that does NOT have the judge cached (a fresh
    empty tmp dir passed as `cache_dir`), `offline_load_kwargs` returns `{}`
    (no offline kwargs at all) -- so the caller falls through to today's
    online-load behavior, which DOES attempt the network (unlike C3a's
    zero). This only asserts the ATTEMPT (a blocked-socket count >= 1, no
    real download), matching 1.5-criteria.md C3(b)'s own scope ("the test
    serves a local stub endpoint or asserts the attempted request, no real
    download").
    """
    script = textwrap.dedent(
        """
        import json
        import socket
        import tempfile

        _connect_calls = [0]

        def _counting_connect(self, addr):
            _connect_calls[0] += 1
            raise OSError("network blocked in this test (C3b: expecting >=1 attempt)")

        socket.socket.connect = _counting_connect

        import brain.memory.relevance_judge as rj

        empty_cache_dir = tempfile.mkdtemp()
        kwargs = rj.offline_load_kwargs("BAAI/bge-reranker-v2-m3", empty_cache_dir)

        # offline_load_kwargs itself must have already tried (and failed) the
        # local-only cache probe without raising -- confirm it decided "not
        # cached" (empty kwargs), THEN separately prove the real load path
        # (CrossEncoder with no offline kwargs) attempts the network.
        assert kwargs == {}, f"expected no offline kwargs for an uncached id, got {kwargs!r}"
        cached_check_attempted_no_network = _connect_calls[0] == 0

        try:
            from sentence_transformers import CrossEncoder
            CrossEncoder("BAAI/bge-reranker-v2-m3", cache_folder=empty_cache_dir, **kwargs)
        except Exception:
            pass  # expected -- the network is blocked; we only care that it TRIED

        print(json.dumps({
            "connect_calls_after_load_attempt": _connect_calls[0],
            "hf_cached_probe_itself_touched_network": not cached_check_attempted_no_network,
        }))
        """
    )
    result = _run_subprocess_script(script)
    assert not result["hf_cached_probe_itself_touched_network"], (
        "the _hf_cached probe (local_files_only=True) must decide 'not cached' without "
        "ever touching the network itself"
    )
    assert result["connect_calls_after_load_attempt"] >= 1, (
        "expected the online-load fallback to attempt at least one network connection "
        "for a judge id that isn't in the (empty) cache dir"
    )
