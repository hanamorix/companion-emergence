"""Cross-encoder reranker provider + process-wide caches.

#231 RERANKER RE-ARCHITECTURE (``~/.claude/plans/memory-dream-rework-
semantic-retrieval-brief.md``, "RERANKER RE-ARCHITECTURE" section). Replaces
the scrapped Stage-4 per-persona cosine-floor/gap auto-calibration
(``brain/memory/semantic_calibration.py``, deleted) — that auto-calibration
derived a floor/gap from the corpus's own pairwise cosine spread, which the
cold red-team proved doesn't generalize (breaks silently on tight/diffuse/
bimodal corpora, because the query-match cosine scale is MODEL-FIXED, not
corpus-shaped). A cross-encoder reads (query, memory) TOGETHER and scores
true relevance — that score is query-conditioned, so a floor on it is
trustworthy in a way a cosine floor never was. Originally a FIXED,
empirically-set module constant (``RERANK_FLOOR``); cut over by F2a inc8
(#250 §7/§8) to a DB-adaptive floor read live per call from
``MemoryStore.get_reranker_floor`` — see ``brain/memory/semantic_recall.py``.

Mirrors ``brain/memory/embeddings.py``'s ``FastEmbedProvider`` +
``build_embedding_provider`` PROCESS-CACHE pattern:
  - ``RerankerProvider`` ABC: two concrete providers, ``CrossEncoderProvider``
    (real, local, ONNX via fastembed's ``TextCrossEncoder`` — no network at
    rerank() time once the model file is cached) and ``FakeRerankerProvider``
    (deterministic/scriptable, zero network, used in tests).
  - ``build_reranker_provider()``: the production provider, PROCESS-WIDE
    cached (``_provider_cache``, keyed by model_id, double-checked locking
    via ``_provider_cache_lock``) so the expensive model/ONNX-session load
    happens once per process, not once per recall.
  - ``_reset_reranker_provider_cache()``: test-only reset hook, wired into
    ``tests/conftest.py``'s autouse fixture alongside the embedding one.

Also owns the AUTO-SCALING rerank-width calculation (spec point 3): the
number of candidates reranked self-derives from a MEASURED warm per-doc
rerank latency on the actual host vs a fixed latency budget — no per-corpus,
no per-persona, no operator tuning; only the budget itself
(``LATENCY_BUDGET_SECONDS``) is an ops tunable. Pre-flip revision Change 3
adds a second, MEMORY bound alongside the time one: a measured-once-warm-
and-cached per-doc memory cost vs a cheap per-call, cgroup-aware RAM-
headroom read — see ``get_rerank_width`` and the "Memory bound" section
above it.

F2a inc2 (#250 §2) originally owned a cached, first-use, on-box fp16-vs-fp32
accuracy SELF-CHECK here (dual-loading both ONNX exports at first use to
decide which to ship). The pre-flip revision's Change 2 REMOVES that
self-check entirely (a confirmed ~1.16 GiB one-time startup memory spike —
the leading suspect in two live VM crashes) and PINS fp16 as the shipped
default instead: Testing's matched-width fp16-vs-fp32 A/B reproduced fp32's
surface/abstain decisions on all 40 bundled queries, so fp16-vs-fp32
agreement is treated as a property of the bundled model weights (identical
on every box), not something a per-box runtime probe needs to establish.
See ``RERANKER_PRECISION`` (the tunable default, mirroring
``LATENCY_BUDGET_SECONDS``'s override shape above) and
``build_reranker_provider`` below.
"""

from __future__ import annotations

import logging
import math
import statistics
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from brain import tunables
from brain.memory.relevance import CANDIDATE_POOL

if TYPE_CHECKING:
    from brain.memory.store import MemoryStore

log = logging.getLogger(__name__)


def _detect_avx2() -> bool:
    """Best-effort startup AVX2 capability check (F2a inc3, #250 §3).

    Runs at process startup (this module's import time), not install time —
    a VM's AVX2 exposure depends on the host it boots on and can change
    between boots ([[dev-vm-avx2-depends-on-host]]), so baking the answer in
    at build/install time would go stale.

    Only Linux is actually probed, via /proc/cpuinfo's `flags` line — the
    one place a reliable answer is available with no new dependency (no
    py-cpuinfo, no numpy CPU-dispatch introspection — that answers "does
    numpy's own build support AVX2", not "does this CPU"). macOS, Windows,
    and any read/parse failure on Linux all fall back to "no AVX2": #250 §3
    pins the 2s/4s split but not a cross-platform detection METHOD, so this
    is resolved conservatively rather than guessed — an undetectable host is
    treated exactly like a confirmed no-AVX2 host, getting the larger, safer
    budget instead of silently assuming a fast one (keeps the potato
    baseline honest). Fail-soft throughout: this must never raise into
    startup.
    """
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags"):
                    return "avx2" in line.split(":", 1)[1].split()
        return False
    except Exception:  # noqa: BLE001 — fail-soft: a capability probe must never break startup
        log.exception("reranker: AVX2 detection failed — defaulting to the conservative no-AVX2 latency budget")
        return False


def _default_latency_budget_seconds(avx2_present: bool) -> float:
    """AVX2-aware rerank latency-budget default (#250 §3): 2s when the
    startup check found AVX2, 4s when it did not (or could not tell — see
    `_detect_avx2`) — build for the no-AVX2 potato baseline, AVX2 as a
    bonus, per the spec's design posture. Split out as its own pure
    function (rather than inlined where `LATENCY_BUDGET_SECONDS` is
    computed) so tests can exercise both branches directly, without needing
    to reload this module under a monkeypatched detector."""
    return 2.0 if avx2_present else 4.0


_AVX2_PRESENT = _detect_avx2()

# Latency budget for one rerank call (#250 §3): AVX2-aware default — 2s if
# this process's startup check found AVX2, 4s if not (see
# `_default_latency_budget_seconds`). An ops-clean knob (a latency/
# throughput setting), unlike the reranker abstention floor (physiology,
# fenced into semantic_recall.py/the DB-backed calibration table per
# tunables.py's own "physiology fenced out" rule, NOT a tunables.py entry).
# Registered here (the owning module) as the DEFAULT only; a manual
# override in tunables.json wins over this auto-detected value via
# tunables.get_tunable's existing override-precedence mechanism (see
# `get_rerank_width` below) — this AVX2-awareness only changes what the
# default resolves to, never the override behavior itself. Read at call
# time via tunables.get_tunable so a live override applies with no restart.
LATENCY_BUDGET_SECONDS: float = tunables.register(
    "reranker.latency_budget_seconds", _default_latency_budget_seconds(_AVX2_PRESENT)
)

# fp16-vs-fp32 reranker precision (pre-flip revision Change 2 — supersedes
# F2a inc2's runtime self-check, F2a spec §2, in full). fp16 is PINNED as
# the shipped default: Testing's matched-width A/B (2026-09-23) reproduced
# fp32's surface/abstain decisions on all 40 bundled queries (zero
# recoveries, zero regressions, zero rank-1 disagreements, deltas ~0.005
# mean / 0.023 max), so fp16-vs-fp32 agreement is treated as a property of
# the bundled model weights — identical on every box, unlike latency — that
# no longer needs a per-box runtime probe to establish. Registered here as
# a tunable DEFAULT, mirroring `LATENCY_BUDGET_SECONDS`'s override shape
# immediately above (I7: never hardcoded inline in `build_reranker_
# provider`) — an operator can still force fp32 by setting
# "reranker.precision" in tunables.json's "overrides" to "fp32".
RERANKER_PRECISION_FP16 = "fp16"
RERANKER_PRECISION_FP32 = "fp32"
RERANKER_PRECISION: str = tunables.register("reranker.precision", RERANKER_PRECISION_FP16)


class RerankerProvider(ABC):
    """Abstract reranker provider. Subclasses implement `rerank` and `model_id`."""

    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> Iterable[float]:
        """Score `documents` against `query`, ONE float per document,
        POSITIONALLY aligned with `documents` (NOT pre-sorted — the caller
        sorts). Higher = more relevant. Not a calibrated probability —
        cross-encoder scores are an empirically-cut relevance signal, not
        [0, 1]-normalized."""

    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier for the model producing these scores."""


class CrossEncoderProvider(RerankerProvider):
    """Real local cross-encoder reranker via `fastembed` (ONNX runtime, no torch).

    Production default. Model id comes from `model_tier.py`
    (`model_for_tier(TIER_RERANKER)`), never hardcoded here — same
    convention as `FastEmbedProvider` in `embeddings.py`.

    The model file is downloaded once (fastembed's own lazy-download-on-first-
    use behavior) into `cache_dir` and used fully offline after — no network
    call happens at rerank() time once the file is cached. Construction
    itself does NOT download; the download is deferred to fastembed's own
    internals on first `rerank()` call (`lazy_load=True`), matching
    `FastEmbedProvider`'s posture.
    """

    def __init__(self, model_id: str, cache_dir: str | Path) -> None:
        # Imported lazily (module-scoped, not top-level) so importing this
        # module never requires fastembed/onnxruntime to be installed unless
        # the real provider is actually constructed (tests exclusively use
        # FakeRerankerProvider). Confirmed path for fastembed 0.8.0: the
        # cross-encoder lives under `fastembed.rerank.cross_encoder`, NOT at
        # the top-level `fastembed` namespace.
        from fastembed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder

        self._model_id = model_id
        self._model = TextCrossEncoder(model_name=model_id, cache_dir=str(cache_dir), lazy_load=True)
        # Same lazy-load check-then-act race FastEmbedProvider.__init__
        # documents (fastembed's OnnxTextModel first-load path has no lock of
        # its own); a shared instance of this provider (the process-wide
        # cache below) can have .rerank() called concurrently from two
        # threads — the turn thread (passive/active recall) and, in
        # principle, a future background caller. Serialize the whole
        # rerank() call (construction + inference) behind one instance lock,
        # matching FastEmbedProvider's conservative choice.
        self._rerank_lock = threading.Lock()

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        with self._rerank_lock:
            return list(self._model.rerank(query, documents))

    def model_id(self) -> str:
        return self._model_id


class FakeRerankerProvider(RerankerProvider):
    """Deterministic, scriptable reranker for tests — zero network, zero model load.

    Mirrors `embeddings.py`'s `FakeEmbeddingProvider` in spirit (offline,
    zero-dependency stand-in for the real provider) but, per the offline-test
    discipline this build requires, is SCRIPTED rather than hash-based: tests
    that need to exercise the floor/tier logic must control the exact score
    per document, not merely get a consistent-but-arbitrary one. A document
    with no entry in `scores` falls back to `default` — deliberately a value
    far below any plausible calibrated floor, so a test that seeds unrelated
    filler content (the same "neutral, never-a-match" role
    `_ScriptedProvider.embed()`'s all-zero fallback plays for embeddings)
    never accidentally clears the floor and turns an intended-inconclusive
    case conclusive.
    """

    _DEFAULT_UNSCORED = -1_000.0

    def __init__(self, scores: dict[str, float] | None = None, *, default: float = _DEFAULT_UNSCORED) -> None:
        self._scores = scores or {}
        self._default = default

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [self._scores.get(doc, self._default) for doc in documents]

    def model_id(self) -> str:
        return "fake-reranker"


# Process-wide provider cache keyed by model_id (see build_reranker_provider).
# Kept at module scope, mirroring embeddings.py's _provider_cache, so
# `_reset_reranker_provider_cache` (test-only) can reach it and so
# monkeypatching the *function* fully controls behavior.
_provider_cache: dict[str, RerankerProvider] = {}
_provider_cache_lock = threading.Lock()


def build_reranker_provider(*, store: MemoryStore | None = None) -> RerankerProvider:
    """The production reranker provider: a `CrossEncoderProvider` pinned to
    `RERANKER_PRECISION`'s resolved id — `model_tier.MODEL_RERANKER_FP16`
    (the pinned default) or `model_tier.TIER_RERANKER`'s fp32 id (only when
    an operator override sets `reranker.precision` to `"fp32"` in
    tunables.json) — caching the model file in the shared `get_cache_dir()`
    (one download across every persona on the box — the model isn't
    persona-specific data, same reasoning as the embedding model).

    Pre-flip revision Change 2 removed the fp16-vs-fp32 first-use precision
    SELF-CHECK this function used to run (F2a spec §2, a confirmed ~1.16
    GiB one-time dual-load memory spike): exactly ONE ONNX export is ever
    constructed per resolved model_id now, never both. `store` is kept as a
    keyword-only parameter purely for call-site compatibility with every
    existing production caller (`semantic_recall.run_semantic_recall`,
    `search_memories._semantic_top_k`, `supervisor._run_calibration_tick`/
    `_run_deploy_recalibration_check`, `floor_calibration.py`) — it is no
    longer read by this function at all.

    PROCESS-WIDE CACHING, same rationale as `embeddings.build_embedding_
    provider`: constructing a CrossEncoderProvider builds a real ONNX
    inference session — expensive to redo every recall. Keyed by model_id
    (double-checked locking: unlocked fast-path read for the common
    already-cached case; the lock is only taken — then re-checked — the
    first time a given model_id needs constructing) for the same reasons
    documented on that function.

    TEST ISOLATION: `tests/conftest.py`'s autouse fixture clears this
    process-global dict before and after every test, mirroring the embedding
    provider's own isolation fixture.
    """
    from brain.bridge.model_tier import MODEL_RERANKER_FP16, TIER_RERANKER, model_for_tier
    from brain.paths import get_cache_dir

    fp32_model_id = model_for_tier(TIER_RERANKER)
    cache_dir = get_cache_dir()
    precision = tunables.get_tunable("reranker.precision", RERANKER_PRECISION)
    if precision == RERANKER_PRECISION_FP32:
        model_id = fp32_model_id
    else:
        if precision != RERANKER_PRECISION_FP16:
            log.warning(
                "reranker: unrecognized reranker.precision override %r — falling back to "
                "the pinned default %r",
                precision,
                RERANKER_PRECISION,
            )
        _register_fp16_reranker_model(MODEL_RERANKER_FP16, fp32_model_id)
        model_id = MODEL_RERANKER_FP16

    provider = _provider_cache.get(model_id)
    if provider is not None:
        return provider

    with _provider_cache_lock:
        provider = _provider_cache.get(model_id)  # re-check: lost the race?
        if provider is not None:
            return provider
        provider = CrossEncoderProvider(model_id=model_id, cache_dir=cache_dir)
        _provider_cache[model_id] = provider
        return provider


def _bootstrap_reranker_provider(model_id: str) -> RerankerProvider:
    """Raw provider construction for `model_id`, used ONLY by
    `floor_calibration.get_bootstrap_floor` (F2a inc8, #250 §7 UPDATED —
    Roy's 2026-09-18 bootstrap-floor ruling) to score the bundled cold-start
    pairs for a floor the caller already knows it needs, for a model_id it
    already knows it needs it for.

    Deliberately bypasses `build_reranker_provider` — that function resolves
    its OWN model_id from `RERANKER_PRECISION`'s tunable (ignoring any
    caller-specified id), whereas this function must construct a provider
    for the EXACT `model_id` it was handed (the specific id
    `get_bootstrap_floor`'s caller already knows it needs a floor for, which
    is not necessarily whatever `build_reranker_provider` would currently
    resolve to). This function never reads precision configuration and
    never touches `store` — it only constructs (or reuses) a plain
    `CrossEncoderProvider` for the exact `model_id` it was asked about.

    Reuses the shared `_provider_cache` (the SAME cache
    `build_reranker_provider` reads/writes, double-checked locking to
    match) so a caller that resolves to this same `model_id` elsewhere in
    the process reuses the already-loaded ONNX session instead of paying
    for a second one. In practice this is very often a cache HIT: every
    production call site that ends up asking `get_reranker_floor` a
    question (`run_semantic_recall`, `_semantic_top_k`) has ALREADY
    resolved/cached a provider for the exact model_id in question via
    `build_reranker_provider` by the time it does so.
    """
    cached = _provider_cache.get(model_id)
    if cached is not None:
        return cached
    with _provider_cache_lock:
        cached = _provider_cache.get(model_id)
        if cached is not None:
            return cached
        from brain.paths import get_cache_dir

        provider = CrossEncoderProvider(model_id=model_id, cache_dir=get_cache_dir())
        _provider_cache[model_id] = provider
        return provider


def _reset_reranker_provider_cache() -> None:
    """Test-only: clear the process-level provider cache.

    Wired into `tests/conftest.py`'s autouse fixture alongside
    `embeddings._reset_embedding_provider_cache` — same rationale (a test
    that calls the REAL `build_reranker_provider()` directly must not read
    or leak a provider a prior/later test's call happened to cache).
    """
    with _provider_cache_lock:
        _provider_cache.clear()


# ---------------------------------------------------------------------------
# Auto-scaling rerank width (spec point 3).
# ---------------------------------------------------------------------------
#
# The number of candidates actually sent through the (comparatively
# expensive) reranker self-derives from a MEASURED warm per-doc rerank
# latency on the actual host vs LATENCY_BUDGET_SECONDS — no per-corpus,
# per-persona, or operator tuning; a one-time hardware self-calibration,
# cached and periodically recomputed.

# Cold-cache timing trap ([[single-shot-timing-cold-cache-trap]]): the first
# rerank() call on a freshly-constructed provider pays ONNX session
# warm-up cost far above steady-state — discard this many calls before
# starting to time.
_WARMUP_RERANKS = 2

# Average over this many WARM calls (post-discard) rather than trusting a
# single sample, a lone reading can still jitter (scheduler noise, a
# concurrent backfill tick). Must stay >= CALIBRATION_SAMPLE_SIZE below,
# since `_doc_for`'s index restarts at 0 in each loop (warmup and measured
# are counted separately), a measured-call count smaller than the sample
# size would leave the tail of a wider sample never actually measured.
_MEASURE_RERANKS = 5

# Recompute the cached per-doc figure this often (startup + periodic, spec
# point 3) rather than trusting a single boot-time measurement forever — a
# long-lived process could see its host's effective throughput change
# (thermal throttling, a noisy neighbor). Not an operator tunable: this is
# an internal self-calibration cadence, not a latency/behavior knob.
_LATENCY_RECOMPUTE_INTERVAL_SECONDS = 3600.0

_MEASURE_QUERY = "warm-up latency calibration query"

# FALLBACK-ONLY synthetic calibration document, used only when no real
# candidate-pool documents are available to calibrate against (see
# CALIBRATION_SAMPLE_SIZE below for the preferred path). #231-fix: the
# ORIGINAL version of this string was a 17-word stub ("a short
# representative memory sentence used only to measure warm per-doc
# cross-encoder rerank latency on this host") — a live no-AVX2 run found
# that stub measures ~7-8ms/doc, while REAL corpus documents reranked in
# production cost ~128ms/doc: the auto-scale throttle this feeds
# (`get_rerank_width`) was a structural no-op because the calibration input
# was not representative of what actually gets reranked. Sized instead to
# match aggregate content-length stats pulled from a live production memory
# corpus (mean ~300 chars / ~45 words, median ~175 chars / ~25 words per
# memory) — this fallback targets the MEAN (the longer of the two), so a
# measurement that has to fall back to it errs toward under-throttling (a
# wider width) rather than over-throttling.
_MEASURE_DOCUMENT = (
    "a longer representative memory passage, sized to match a typical "
    "corpus document rather than a short placeholder, used only to measure "
    "warm per-document cross-encoder rerank latency realistically on this "
    "host so the auto-scaling width calculation reflects real production "
    "cost instead of an artificially cheap calibration figure"
)

# How many REAL candidate-pool documents callers should sample for
# calibration when they can supply them (the preferred path, see
# `get_rerank_width`'s `sample_documents` parameter). Calibration runs
# synchronously in-band on the first recall of the process (see
# `_warm_per_doc_latency`), so this is a tradeoff, not a free knob: a wider
# sample makes the measured per-doc figure more representative of the real
# corpus, but each extra sampled document is roughly one extra rerank() call
# added to that one-time first-recall calibration cost (kept in lockstep
# with `_MEASURE_RERANKS` above, see its comment).
CALIBRATION_SAMPLE_SIZE = 5

# model_id -> (per_doc_seconds, measured_at_monotonic). Process-wide, mirrors
# the provider cache above — one measurement per model_id, shared across
# every recall in the process.
_latency_cache: dict[str, tuple[float, float]] = {}
_latency_cache_lock = threading.Lock()


def _measure_warm_per_doc_latency(
    provider: RerankerProvider, sample_docs: list[str] | None = None
) -> float:
    """Time `_MEASURE_RERANKS` single-document rerank() calls AFTER
    discarding `_WARMUP_RERANKS` cold ones; return the mean seconds/doc.

    A single-document rerank isolates per-doc cost from batching effects —
    the width calculation multiplies this back out linearly
    (`floor(budget / per_doc)`), matching how `get_rerank_width` actually
    uses the figure.

    #231-fix: `sample_docs`, when non-empty, is REAL candidate-pool document
    content supplied by the caller (the preferred calibration input — see
    `get_rerank_width`) and is cycled through across the warmup + measured
    calls, one document per call, so the mean reflects real corpus document
    length rather than always the same document. Falls back to the fixed
    synthetic `_MEASURE_DOCUMENT` (see its own comment) only when no real
    docs are available — e.g. a caller that hasn't threaded them through, or
    an empty candidate pool.
    """
    docs = list(sample_docs) if sample_docs else [_MEASURE_DOCUMENT]

    def _doc_for(i: int) -> list[str]:
        return [docs[i % len(docs)]]

    for i in range(_WARMUP_RERANKS):
        list(provider.rerank(_MEASURE_QUERY, _doc_for(i)))

    samples: list[float] = []
    for i in range(_MEASURE_RERANKS):
        start = time.monotonic()
        list(provider.rerank(_MEASURE_QUERY, _doc_for(i)))
        samples.append(time.monotonic() - start)
    return sum(samples) / len(samples)


def _warm_per_doc_latency(
    provider: RerankerProvider, sample_docs: list[str] | None = None
) -> float:
    """Cached warm per-doc latency for `provider`'s model_id, measuring (and
    caching) on first use or once `_LATENCY_RECOMPUTE_INTERVAL_SECONDS` has
    elapsed since the last measurement.

    The first semantic recall of each process pays this calibration cost
    in-band (the warmup plus measured reranks above run synchronously before
    that recall's width is known). It is cached after that first call, so
    every later recall in the process reads the cached value instead —
    `sample_docs` passed on a later (cache-hit) call is simply ignored, same
    as it would be if the whole function signature hadn't changed; only the
    FIRST caller within `_LATENCY_RECOMPUTE_INTERVAL_SECONDS` actually
    supplies the docs a measurement uses.

    Fail-soft: a measurement failure (e.g. the real model errors on the
    calibration call) logs and returns 0.0 — `get_rerank_width` treats 0.0
    as "no latency signal, don't throttle by it," capping width by pool size
    / CANDIDATE_POOL alone instead. Never raises into a recall.
    """
    model_id = provider.model_id()
    now = time.monotonic()
    with _latency_cache_lock:
        cached = _latency_cache.get(model_id)
        if cached is not None and (now - cached[1]) < _LATENCY_RECOMPUTE_INTERVAL_SECONDS:
            return cached[0]

    # Measure OUTSIDE the lock — a real rerank call can take real time
    # (seconds), and holding the lock across it would serialize every
    # concurrent recall behind this one measurement.
    try:
        measured = _measure_warm_per_doc_latency(provider, sample_docs)
    except Exception:  # noqa: BLE001 — fail-soft: never break recall over a calibration failure
        log.exception("reranker: warm-latency measurement failed — width will not be latency-throttled")
        measured = 0.0

    with _latency_cache_lock:
        _latency_cache[model_id] = (measured, now)
    return measured


def _reset_latency_cache() -> None:
    """Test-only: clear the measured-latency cache."""
    with _latency_cache_lock:
        _latency_cache.clear()


# ---------------------------------------------------------------------------
# Memory bound for auto-scaling rerank width (pre-flip revision, Change 3).
# ---------------------------------------------------------------------------
#
# `get_rerank_width` was latency-derived only, memory-blind: fp16's extra
# speed, with no memory bound to check against, was poured entirely into a
# wider width and OOM'd at 5.36 GiB under a 5.5G cgroup cap (the fp16
# natural-width run, Testing 2026-09-23). This section adds a MEASURED
# memory term as an additional `min()` bound, mirroring the existing
# measure-once-warm-and-cache `_warm_per_doc_latency` pattern exactly:
# `per_doc_MEMORY` is measured once per model_id (RSS delta around a warm
# rerank batch of known size, divided by batch size) and cached, recomputed
# on the same cadence as the latency figure; `available_RAM_headroom` is a
# cheap per-call read (cgroup-limit-aware, falling back to host free memory),
# mirroring the `_detect_avx2` /proc-read shape: cheap, dependency-free,
# safely degrades on any read/parse error or unsupported platform, never
# crashes. On any failure to determine either figure, the memory term is
# SKIPPED entirely (never assume unlimited headroom) — `get_rerank_width`
# degrades to exactly the pre-Change-3 time-only bound.

# Batch size for the one-time warm RSS-delta memory measurement. Reuses
# `_MEASURE_RERANKS`'s value (rather than inventing a second, differently-
# derived integer): the same magnitude reasoning applies — large enough that
# the RSS delta from reranking the batch clears ordinary allocator/GC noise,
# small enough that the one-time measurement stays cheap.
_MEMORY_MEASURE_BATCH_SIZE = _MEASURE_RERANKS

# Recompute the cached per-doc memory figure on the SAME cadence as the
# per-doc latency figure above (`_LATENCY_RECOMPUTE_INTERVAL_SECONDS`) — a
# long-lived process's per-doc memory cost can drift for the same class of
# reasons the latency figure can (allocator fragmentation, a noisy
# neighbor's memory pressure), so this reuses that cadence rather than
# introducing a second, arbitrarily-different one.
_MEMORY_RECOMPUTE_INTERVAL_SECONDS = _LATENCY_RECOMPUTE_INTERVAL_SECONDS

# model_id -> (per_doc_bytes, measured_at_monotonic). Process-wide, mirrors
# `_latency_cache` above exactly — one measurement per model_id, shared
# across every recall in the process.
_memory_cache: dict[str, tuple[float, float]] = {}
_memory_cache_lock = threading.Lock()


def _current_rss_bytes() -> float | None:
    """This process's current resident set size, in bytes, via
    `/proc/self/status`'s `VmRSS` line — the same shape as `_detect_avx2`'s
    `/proc` read (Linux-only; fail-soft to `None` on any read/parse error or
    an unsupported platform, since there is no `/proc` on macOS/Windows).
    Never raises."""
    if sys.platform != "linux":
        return None
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # e.g. "VmRSS:\t   12345 kB\n" -> kB -> bytes.
                    return float(line.split()[1]) * 1024.0
        return None
    except Exception:  # noqa: BLE001 — fail-soft: an RSS probe must never break a recall
        log.exception("reranker: RSS read failed — per-doc memory measurement will be skipped")
        return None


def _measure_warm_per_doc_memory(
    provider: RerankerProvider, sample_docs: list[str] | None = None
) -> float:
    """RSS delta bracketing `_MEMORY_MEASURE_BATCH_SIZE` warm rerank() calls,
    divided by that batch size — the per-candidate memory cost `get_rerank_
    width`'s memory bound divides headroom by.

    Each bracketed call is SINGLE-document — one document per `rerank()`
    call, cycling through `sample_docs` exactly like `_measure_warm_per_doc_
    latency`'s `_doc_for` helper — rather than one N-document batch call.
    This mirrors that function's per-doc isolation shape (reusing it, not
    inventing a second one, per this change's own requirement) AND matters
    structurally: a caller-facing invariant elsewhere in this codebase
    (`test_f2b_anchor_wiring.py`'s "exactly one combined rerank() call"
    checks) identifies the real+anchor call specifically by it being the
    only MULTI-document `rerank()` call in a turn — a single N-document
    batch call here would collide with that and be mistaken for a second
    combined call. Keeping every calibration call length-1 avoids that
    collision entirely, the same way the existing latency measurement's
    single-document calls already do.

    `_WARMUP_RERANKS` warmup calls are run first and discarded before the
    measured sequence, matching `_measure_warm_per_doc_latency`'s
    cold-cache-trap avoidance ([[single-shot-timing-cold-cache-trap]]) —
    a cold ONNX session's first calls do real allocation work
    (session/tensor buffers) unrelated to steady-state per-doc memory cost.

    A negative delta (RSS can legitimately drop between the two reads — a
    GC pass, another thread freeing memory) or an unreadable RSS clamps to
    0.0 — the same "no signal" sentinel `_warm_per_doc_latency` uses for a
    measurement failure, which `get_rerank_width` already treats as "skip
    this term".
    """
    docs = list(sample_docs) if sample_docs else [_MEASURE_DOCUMENT]

    def _doc_for(i: int) -> list[str]:
        return [docs[i % len(docs)]]

    for i in range(_WARMUP_RERANKS):
        list(provider.rerank(_MEASURE_QUERY, _doc_for(i)))  # warmup, discarded

    before = _current_rss_bytes()
    for i in range(_MEMORY_MEASURE_BATCH_SIZE):
        list(provider.rerank(_MEASURE_QUERY, _doc_for(i)))
    after = _current_rss_bytes()

    if before is None or after is None:
        return 0.0
    delta = after - before
    if delta <= 0.0:
        return 0.0
    return delta / _MEMORY_MEASURE_BATCH_SIZE


def _warm_per_doc_memory(
    provider: RerankerProvider, sample_docs: list[str] | None = None
) -> float:
    """Cached warm per-doc memory cost for `provider`'s model_id — mirrors
    `_warm_per_doc_latency` exactly: measured (and cached) on first use or
    once `_MEMORY_RECOMPUTE_INTERVAL_SECONDS` has elapsed since the last
    measurement, measured OUTSIDE the lock (a real rerank batch takes real
    time, and holding the lock across it would serialize every concurrent
    recall behind this one measurement), fail-soft to 0.0 (`get_rerank_
    width` treats 0.0 as "no memory signal, don't throttle by it") on any
    measurement failure. Never raises into a recall."""
    model_id = provider.model_id()
    now = time.monotonic()
    with _memory_cache_lock:
        cached = _memory_cache.get(model_id)
        if cached is not None and (now - cached[1]) < _MEMORY_RECOMPUTE_INTERVAL_SECONDS:
            return cached[0]

    try:
        measured = _measure_warm_per_doc_memory(provider, sample_docs)
    except Exception:  # noqa: BLE001 — fail-soft: never break recall over a calibration failure
        log.exception("reranker: warm-memory measurement failed — width will not be memory-throttled")
        measured = 0.0

    with _memory_cache_lock:
        _memory_cache[model_id] = (measured, now)
    return measured


def _reset_memory_cache() -> None:
    """Test-only: clear the measured-memory cache."""
    with _memory_cache_lock:
        _memory_cache.clear()


_CGROUP_V2_ROOT = "/sys/fs/cgroup"

# The kernel's own "no limit" sentinel for cgroup v1's memory.limit_in_bytes
# (there is no explicit "unbounded" marker like v2's "max" string) — derived
# from PAGE_COUNTER_MAX on a 4KiB-page 64-bit kernel. A limit at or above
# this value is unbounded, never a real cap.
_CGROUP_V1_UNBOUNDED_SENTINEL = 0x7FFFFFFFFFFFF000


def _read_proc_self_cgroup_v2_path() -> str | None:
    """The CALLING PROCESS's own cgroup v2 path (the unified hierarchy,
    hierarchy id 0) from `/proc/self/cgroup`'s single `0::/<path>` line —
    this is what lets the v2 reader resolve the process's OWN cgroup
    directory instead of the fixed cgroup2 root (the root exposes no
    `memory.max`/`memory.current` of its own, so reading it directly can
    never see a real per-process cap). Returns the path with leading/
    trailing slashes stripped (empty string for the v2 root itself), or
    `None` on any read/parse failure or if no `0::` line is present —
    same fail-soft posture as every other reader here."""
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as f:
            content = f.read()
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if line.startswith("0::"):
                return line[len("0::") :].strip("/")
        return None  # no unified (v2) hierarchy line present
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 — fail-soft: a headroom probe must never break a recall
        log.exception("reranker: /proc/self/cgroup (v2) read failed")
        return None


def _cgroup_v2_ancestor_dirs(cgroup_path: str) -> list[str]:
    """Every cgroup v2 directory from the process's own cgroup UP TO the
    root, deepest first. Used to find the EFFECTIVE memory limit: a
    cgroup's own `memory.max` may read "max" (unbounded) while an
    ANCESTOR still caps it, so the effective limit is the MIN of every
    numeric `memory.max` along this chain, not just the process's own
    level."""
    segments = [s for s in cgroup_path.split("/") if s]
    dirs = []
    for depth in range(len(segments), -1, -1):
        sub = "/".join(segments[:depth])
        dirs.append(f"{_CGROUP_V2_ROOT}/{sub}" if sub else _CGROUP_V2_ROOT)
    return dirs


def _cgroup_v2_memory_headroom_bytes() -> float | None:
    """cgroup v2 memory headroom for the CALLING PROCESS's own cgroup
    (`effective_memory.max - memory.current`), resolved via `/proc/self/
    cgroup` (see `_read_proc_self_cgroup_v2_path`) rather than the fixed
    cgroup2 root — the root has no `memory.max`/`memory.current` of its
    own, so a root-only read can never see a real per-process cap.

    The effective limit WALKS UP the chain from the process's own cgroup
    to the root (`_cgroup_v2_ancestor_dirs`), taking the MIN of every
    numeric `memory.max` seen (skipping "max" and any level whose file is
    absent — an ancestor may not expose the controller file at all),
    since an ancestor can cap memory even when the process's own cgroup
    reads "max". Usage is read from the process's OWN cgroup only.

    Returns `None` (never a fabricated headroom) when: not Linux,
    `/proc/self/cgroup` is missing/unparsable, every `memory.max` on the
    chain is "max"/absent (no cap anywhere -> no need to even read
    usage), the process's own `memory.current` is missing/unparsable, or
    any other read/parse error (logged, then degrades to `None`) — every
    `None` case means "caller falls back to the next source", never
    "unlimited"."""
    if sys.platform != "linux":
        return None

    cgroup_path = _read_proc_self_cgroup_v2_path()
    if cgroup_path is None:
        return None
    dirs = _cgroup_v2_ancestor_dirs(cgroup_path)

    effective_limit: float | None = None
    for cgroup_dir in dirs:
        try:
            with open(f"{cgroup_dir}/memory.max", encoding="utf-8") as f:
                max_raw = f.read().strip()
        except FileNotFoundError:
            continue  # this level exposes no memory controller -> no cap here, keep walking
        except Exception:  # noqa: BLE001 — fail-soft
            log.exception("reranker: cgroup v2 memory.max read failed")
            return None

        if max_raw == "max":
            continue  # explicitly unbounded at this level -> keep walking up

        try:
            level_limit = float(max_raw)
        except Exception:  # noqa: BLE001 — fail-soft
            log.exception("reranker: cgroup v2 memory.max parse failed")
            return None
        effective_limit = level_limit if effective_limit is None else min(effective_limit, level_limit)

    if effective_limit is None:
        return None  # no numeric cap anywhere on the chain -> caller falls back

    try:
        with open(f"{dirs[0]}/memory.current", encoding="utf-8") as f:
            usage = float(f.read().strip())
    except FileNotFoundError:
        return None  # process's own cgroup exposes no memory controller
    except Exception:  # noqa: BLE001 — fail-soft
        log.exception("reranker: cgroup v2 memory.current read/parse failed")
        return None
    return max(0.0, effective_limit - usage)


def _read_proc_self_cgroup_v1_memory_path() -> str | None:
    """The CALLING PROCESS's own cgroup v1 path for the `memory`
    controller, from `/proc/self/cgroup`'s `N:<controllers>:/<path>`
    lines (`controllers` is a comma-separated list on a combined
    hierarchy, e.g. `cpu,memory`). Returns the path with leading/
    trailing slashes stripped (empty string for the v1 memory root), or
    `None` on any read/parse failure or if no line lists `memory`."""
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as f:
            content = f.read()
        for raw_line in content.splitlines():
            parts = raw_line.strip().split(":", 2)
            if len(parts) != 3:
                continue
            _hierarchy_id, controllers, path = parts
            if "memory" in controllers.split(","):
                return path.strip("/")
        return None  # no v1 memory controller line present
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 — fail-soft: a headroom probe must never break a recall
        log.exception("reranker: /proc/self/cgroup (v1) read failed")
        return None


def _cgroup_v1_memory_headroom_bytes() -> float | None:
    """cgroup v1 fallback for `_cgroup_v2_memory_headroom_bytes` (older
    kernels/distros without the unified v2 hierarchy): `memory.limit_in_
    bytes - memory.usage_in_bytes` read from the CALLING PROCESS's own
    cgroup path, resolved via `/proc/self/cgroup`'s `memory` controller
    line (`_read_proc_self_cgroup_v1_memory_path`) rather than the fixed
    `/sys/fs/cgroup/memory/` root. v1's limit is already hierarchical (a
    child's own `memory.limit_in_bytes` reflects any ancestor cap), so —
    unlike v2 — no walk-up is needed here; the process's own path is
    always the right one to read.

    An unbounded v1 limit reads back as the kernel's own huge sentinel
    value (there is no explicit "unbounded" marker like v2's `"max"`); a
    limit at or above `_CGROUP_V1_UNBOUNDED_SENTINEL` is treated as
    unbounded -> `None`, the same "caller falls back" signal as every
    other case here. Same fail-soft posture as the v2 reader: `None` on
    no v1 controller, unreadable, or unparsable — never a fabricated
    headroom."""
    if sys.platform != "linux":
        return None

    cgroup_path = _read_proc_self_cgroup_v1_memory_path()
    if cgroup_path is None:
        return None
    cgroup_dir = f"/sys/fs/cgroup/memory/{cgroup_path}" if cgroup_path else "/sys/fs/cgroup/memory"

    try:
        with open(f"{cgroup_dir}/memory.limit_in_bytes", encoding="utf-8") as f:
            limit = float(f.read().strip())
    except FileNotFoundError:
        return None  # no v1 memory controller for this process's cgroup
    except Exception:  # noqa: BLE001 — fail-soft
        log.exception("reranker: cgroup v1 memory.limit_in_bytes read failed")
        return None

    if limit >= _CGROUP_V1_UNBOUNDED_SENTINEL:
        return None  # kernel's "no limit" sentinel -> caller falls back, never treated as unlimited

    try:
        with open(f"{cgroup_dir}/memory.usage_in_bytes", encoding="utf-8") as f:
            usage = float(f.read().strip())
    except Exception:  # noqa: BLE001 — fail-soft
        log.exception("reranker: cgroup v1 memory.usage_in_bytes read failed")
        return None
    return max(0.0, limit - usage)


def _proc_meminfo_available_bytes() -> float | None:
    """Plain host free-memory fallback: `/proc/meminfo`'s `MemAvailable`
    line (kernel-computed "usable without swapping" estimate) — used only
    when no cgroup memory limit applies (see `_available_ram_headroom_
    bytes`). Same `_detect_avx2`-shaped fail-soft posture: `None` on
    non-Linux or any read/parse error, never a fabricated figure."""
    if sys.platform != "linux":
        return None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024.0
        return None
    except Exception:  # noqa: BLE001 — fail-soft: a headroom probe must never break a recall
        log.exception("reranker: /proc/meminfo read failed")
        return None


def _available_ram_headroom_bytes() -> float | None:
    """Cheap, per-call RAM-headroom read for `get_rerank_width`'s memory
    bound — cadence is a fresh read on every call (not cached like the
    per-doc TIME/MEMORY figures above), since this is inexpensive (a
    handful of small `/proc`-or-`/sys` reads) and lets width react to
    headroom changes mid-session (another process on the box growing), the
    same trade-off the spec's own Open Reconfirmation favors absent a
    concrete reason not to.

    cgroup-limit-aware, per the pre-flip revision's Change-3 requirement:
    Testing's own OOM reproduction ran under an explicit cgroup cap that
    the host's plain `/proc/meminfo` free-memory reading does NOT reflect,
    so a host-only read would fail to prevent the exact OOM this change
    fixes. Tries cgroup v2 first, then v1, then falls back to the host-wide
    `/proc/meminfo` reading only when no cgroup limit applies (or it's
    explicitly unbounded). Returns `None` — never a fabricated number —
    when every source is unavailable/unsupported/errors; `get_rerank_width`
    treats `None` as "skip the memory term", never as "unlimited
    headroom"."""
    headroom = _cgroup_v2_memory_headroom_bytes()
    if headroom is None:
        headroom = _cgroup_v1_memory_headroom_bytes()
    if headroom is None:
        headroom = _proc_meminfo_available_bytes()
    return headroom


def get_rerank_width(
    pool_size: int,
    provider: RerankerProvider,
    sample_documents: list[str] | None = None,
) -> int:
    """How many of the (cosine-coarse-cut) candidate pool to actually rerank.

    `= max(1, min(pool_size, CANDIDATE_POOL, floor(LATENCY_BUDGET_SECONDS /
    per_doc_TIME), floor(available_RAM_headroom / per_doc_MEMORY)))` — the
    pre-flip revision's Change 3 formula: the original time-only bound plus
    a MEMORY bound as an additional `min()` term. On fast (AVX2), spacious
    hardware the measured per-doc latency and memory cost are both small
    relative to budget/headroom, so both bounds allow a wide rerank; on slow
    and/or RAM-tight hardware either (or both) narrows automatically,
    staying within budget AND within a safe memory ceiling — the fp16
    natural-width OOM (extra speed poured into a wider width with no memory
    check) is exactly the case a time-only bound could never catch. An
    auto-scaler computing a width under 50 on a constrained host is intended
    (not a violation) — the whole point is that the count adapts to the
    host, not to a fixed target.

    The memory bound is SKIPPED (formula degrades to exactly the pre-
    Change-3 time-only bound) whenever either input is unavailable: `per_doc
    _MEMORY <= 0.0` (no measured signal — mirrors `per_doc_TIME`'s identical
    0.0-means-"don't throttle by it" contract) or `available_RAM_headroom is
    None` (the cheap per-call read found no usable source — see
    `_available_ram_headroom_bytes`). This never assumes unlimited headroom;
    it only ever narrows width when it has a genuine measured reason to.

    `sample_documents` (#231-fix): a small sample of REAL candidate-pool
    document content (see `CALIBRATION_SAMPLE_SIZE`), used to calibrate
    `warm_per_doc` against actual corpus documents instead of a synthetic
    placeholder — a fixed short stub massively under-measures real
    cross-encoder cost, which made this auto-scaler a structural no-op (it
    always computed a budget far in excess of `CANDIDATE_POOL`, so the min()
    always picked `CANDIDATE_POOL` regardless of true per-doc cost). Optional
    and purely additive: omitting it (or an empty list) falls back to the
    original fixed-placeholder calibration, unchanged. The same sample feeds
    BOTH the time and memory measurements.

    `pool_size <= 0` returns 0 (nothing to rerank — the caller's empty-pool
    case never reaches here in practice, but this stays well-defined).
    """
    if pool_size <= 0:
        return 0
    budget = tunables.get_tunable("reranker.latency_budget_seconds", LATENCY_BUDGET_SECONDS)
    per_doc_time = _warm_per_doc_latency(provider, sample_documents)
    if per_doc_time <= 0.0:
        time_width = pool_size
    else:
        time_width = math.floor(budget / per_doc_time)

    bounds = [pool_size, CANDIDATE_POOL, time_width]

    per_doc_memory = _warm_per_doc_memory(provider, sample_documents)
    headroom = _available_ram_headroom_bytes()
    if per_doc_memory > 0.0 and headroom is not None:
        bounds.append(math.floor(headroom / per_doc_memory))

    return max(1, min(bounds))


# ---------------------------------------------------------------------------
# Per-query anchor normalization (F2b, #276 — follow-on to #250/F2a).
# ---------------------------------------------------------------------------
#
# A cross-encoder's raw score carries a PER-QUERY constant offset (listwise-
# softmax translation invariance): one query's candidate scores can sit
# systematically higher or lower than another's for reasons unrelated to
# relevance, which makes ANY absolute floor (including F2a's daily-derived
# one, #250 §7/§8) less trustworthy than it looks. This section fixes that
# PER REQUEST, at rerank time: inject a small FIXED set of off-topic "anchor"
# documents into the SAME rerank() call as the real candidates, and recenter
# each real candidate's score against that call's own anchor-score median —
# `normalized_score = raw_score - median(anchor_scores)`. Median (not mean)
# so a single anomalous anchor score cannot swing the whole correction, the
# same robust-statistics posture F2a's own design favors generally.
#
# `k = min(P, floor(width / 2))` — the anchor count `k` is HARDWARE-DERIVED
# from `get_rerank_width`'s own output, never a hand-picked constant [OWNER
# 2026-09-22 "that, is a magic number constant. No. Have it scale based on
# the hardware capability like everything else"]. Anchors are RESERVED OUT
# OF `width`, never appended on top of it, so the one combined rerank() call
# scores exactly `real_width + k = width` documents — the same budget-
# derived count `get_rerank_width` already enforces, never more (I6: zero
# net latency added over today's one bounded per-turn rerank step).
#
# This section builds ONLY the isolated mechanism (helper + anchor pool).
# Wiring it into `semantic_recall.py` / `search_memories.py`'s floor-gate
# call sites, re-pointing F2a's calibration-log write at the normalized
# score, and the deploy-time one-time recalibration are LATER increments
# (spec §2/§5/§6) — see `~/.claude/plans/f2b-anchor-normalization-spec.md`.

# The meaningful-median floor: below 2 anchor scores, "median" degenerates
# (a single value, or an arbitrary pick between two with no robust middle)
# and offers no protection against one anomalous anchor swinging the whole
# correction — the same robust-statistics reasoning the median choice above
# rests on. SIZES the anchor mechanism (I3-clean — same precedent class as
# `CALIBRATION_SAMPLE_SIZE` above); it is NOT a relevance threshold — no
# memory/candidate score is ever compared against `K_MIN`.
K_MIN = 2

# The split ratio (spec §3's natural "anchors keep AT MOST HALF the rerank
# slots" rule): `k = min(P, width // ANCHOR_SPLIT_DIVISOR)`. A build-time
# tunable ratio, not a relevance threshold — SIZES the mechanism, same
# I3-clean class as `K_MIN`/`P`. Named (rather than an inline `2` in the
# formula) so the split ratio is a single, greppable, documented knob if it
# is ever retuned, per the Open-reconfirmations note that this ratio "may be
# tuned".
ANCHOR_SPLIT_DIVISOR = 2

# Curated, FIXED, genuinely off-topic anchor documents (ledger-settled: Roy
# adopted "old Option B", whose mechanism is a fixed off-topic anchor set —
# a non-self-calibrating pool is ledger-settled, not a smuggled I3 constant,
# per the spec's §4 I3 discussion). Eight short documents spanning eight
# DIVERSE, non-overlapping mundane domains (consumer-goods warranty
# legalese, shipping/logistics contract text, a hardware spec sheet, a
# facilities-maintenance procedure, software release notes, a
# building-management notice, weather-instrument telemetry, a library loan
# policy) so no single real query plausibly lands close to more than one or
# two of them. The reranker is multilingual (jina, F2a §1) and judges
# semantic RELATEDNESS rather than language match, so off-topicness is
# expected to transfer cross-lingually.
#
# ORDER IS MEANINGFUL, not incidental: `normalize_against_anchors` takes
# `ANCHOR_POOL[:k]` — a PREFIX — so on a small/potato width (small `k`) only
# the FRONT of this list participates, and a large `k` (fast host) is the
# only case where the tail ever enters the median. The pool is therefore
# ordered SAFEST-FIRST: the six clearly-inert bureaucratic/technical anchors
# (warranty, shipping, printer, irrigation, spreadsheet release notes, HOA
# parking) come first, so the least-robust k=2/k=3 calls (narrowest width,
# median of the fewest anchors — most exposed to a single anomalous score)
# draw only from them. The two entries below with faint topical overlap with
# this project's own real query patterns are placed LAST, deliberately,
# so they only enter the mix at large k (k=7/8, wide/fast-host calls) where
# a median over many anchors absorbs one elevated score:
#   - the weather-instrument entry: the project pervasively uses an
#     "emotional weather" framing elsewhere (a `weather_shift` anchor
#     detector, the emotion self-model's weather metaphors) that a
#     weather-themed real query could resonate with, even though this entry
#     itself is pure instrument telemetry, not conversational small talk;
#   - the library loan-policy entry: the persona has an author/book life, so
#     "book / loan period / renewals" carries faint topical overlap with
#     real book/manuscript queries.
# Do NOT read this ordering as arbitrary or reorder it without re-applying
# this same safest-first placement.
ANCHOR_POOL: list[str] = [
    "This appliance's warranty covers manufacturing defects for twelve "
    "months from the original purchase date and does not cover damage "
    "caused by misuse or unauthorized repair.",
    "Standard shipping terms require the buyer to inspect goods within "
    "five business days of delivery and report any discrepancy in writing "
    "to the carrier's claims department.",
    "The printer supports A4, Letter, and Legal paper sizes with a "
    "recommended margin of at least six millimeters on all sides to avoid "
    "print clipping.",
    "Quarterly maintenance of the irrigation valve assembly should include "
    "flushing the filter screen and checking the solenoid wiring for "
    "corrosion.",
    "The spreadsheet application's release notes for this version list "
    "improved handling of frozen panes and a fix for a rare crash when "
    "pasting merged cells.",
    "Residents are reminded that guest parking permits must be displayed "
    "on the dashboard and are valid only between six in the evening and "
    "eight in the morning on weekdays.",
    "Yesterday's weather station reading recorded a barometric pressure of "
    "1013 hectopascals with wind speeds averaging twelve kilometers per "
    "hour from the northwest.",
    "A public library's standard loan period for print books is three "
    "weeks, with up to two renewals allowed unless another patron has "
    "placed a hold.",
]

# The curated pool size CAPS `k` (`k = min(P, floor(width / 2))`) so a fast
# host (large `width`) never reranks an excessive anchor block — a robust
# median saturates well before 8 anchors, and 8 gives headroom across the
# realistic width range (potato ~7 through CANDIDATE_POOL-bound hosts). SIZES
# the mechanism, same I3-clean class as `K_MIN` above; DERIVED from
# `ANCHOR_POOL`'s own length, never re-typed as a separate literal.
P = len(ANCHOR_POOL)


def _median_normalize(real_scores: list[float], anchor_scores: list[float]) -> list[float]:
    """Shared median-normalization CORE (F2b §5b, #276 inc3): `real_score -
    median(anchor_scores)` for each real score, sharing statistics.median's
    robust-to-a-single-outlier-anchor property (see the module section
    header above).

    Used by BOTH normalization callers in this module: the per-recall gate
    (`normalize_against_anchors`, below — scores a hardware-derived `k`-
    subset of `ANCHOR_POOL`, latency-budget-limited) and the off-hot-path
    bundled-pair normalization (`normalize_bundled_pairs_against_anchors`,
    below — scores the FULL curated pool `P`, off the hot path, no latency
    budget). One shared arithmetic core; each caller only differs in HOW
    MANY anchors it scores and WHY (spec §5b: "the FULL pool P, NOT the
    per-recall k" for the floor-derivation sources — the gate stays
    `k`-limited for latency, §3)."""
    offset = statistics.median(anchor_scores)
    return [s - offset for s in real_scores]


def normalize_bundled_pairs_against_anchors(
    provider: RerankerProvider,
    pairs: list[tuple[str, str]],
) -> list[float]:
    """Per-bundled-pair anchor-median normalization (F2b §5b, #276 inc3) —
    used by `floor_calibration.py`'s cold-start (`_cold_start_pairs`, feeding
    both `derive_and_persist_floor`'s cold-start branch and
    `get_bootstrap_floor`), so EVERY floor-derivation source F2a computes
    lands on the SAME normalized scale the per-recall gate compares against
    (`normalize_against_anchors`).

    Unlike `normalize_against_anchors` (the per-RECALL gate path, latency-
    budget-limited to a hardware-derived `k`-subset of `ANCHOR_POOL`), this
    scores each bundled `(query, doc)` pair against the FULL curated anchor
    pool (`ANCHOR_POOL`, all `P` of them) — off the hot path (build-time /
    first-load-cached for the bootstrap, or the daily idle tick for
    cold-start) there is no latency budget to reserve slots out of, so the
    full pool gives the most stable median for these single-scalar-floor
    derivations (spec §5b: "the FULL pool P, NOT the per-recall k" — the two
    are deliberately different anchor-COUNT policies sharing the same
    `_median_normalize` arithmetic core above).

    One combined `rerank(query, [doc] + ANCHOR_POOL)` call PER PAIR (a
    different query each time, so pairs cannot be batched into one call) —
    `len(ANCHOR_POOL) + 1` documents scored per pair. Cheap and bounded: this
    only ever runs at build-time/first-load (bootstrap) or in the daily idle
    tick (cold-start) — never per recall (I6).

    Returns one normalized score per pair, positionally aligned with
    `pairs`. Anchor documents/scores are used ONLY to compute each pair's
    median offset and are never returned.
    """
    normalized: list[float] = []
    for query, doc in pairs:
        raw_scores = list(provider.rerank(query, [doc, *ANCHOR_POOL]))
        real_score, anchor_scores = raw_scores[0], raw_scores[1:]
        normalized.append(_median_normalize([real_score], anchor_scores)[0])
    return normalized


@dataclass(frozen=True)
class AnchorNormalizationResult:
    """Result of `normalize_against_anchors` — everything a (later-increment)
    caller needs to map scores back onto real candidate ids.

    `scores` — one float per ACTUALLY-SCORED real candidate, positionally
    aligned with the FRONT of the caller's `real_documents` (i.e.
    `real_documents[:real_width]` — see `real_width`). Median-normalized
    (`raw - median(anchor_scores)`) when `did_normalize` is True; raw,
    unmodified reranker scores when False (the no-op case — F2a-only
    behaviour for that call).

    `real_width` — how many of the caller's `real_documents` were actually
    sent to the reranker (`== len(scores)`). Equals `width` when
    `did_normalize` is False (no anchors reserved that call) and
    `width - k` when True.

    `did_normalize` — False on a near-degenerate `width` (`k < K_MIN`, i.e.
    `width < 4` at today's `K_MIN`/split values): no anchors were appended,
    `scores` are RAW. True whenever the median correction actually ran.
    """

    scores: list[float]
    real_width: int
    did_normalize: bool


def normalize_against_anchors(
    provider: RerankerProvider,
    query: str,
    real_documents: list[str],
    width: int,
) -> AnchorNormalizationResult:
    """Per-query anchor-median normalization (F2b, #276) — see the module
    section header above for the mechanism and the owner ruling that shaped
    `k`'s derivation.

    `real_documents` is the caller's coarse-ranked REAL candidate content,
    already ordered best-first (the same list a caller would otherwise slice
    to `width` and rerank directly) — this function does its OWN slicing
    (`real_documents[:real_width]`), so callers should NOT pre-slice to
    `width` themselves. `width` is exactly what `get_rerank_width(...)`
    returned for this call.

    Computes `k = min(P, width // ANCHOR_SPLIT_DIVISOR)` (anchors keep at
    most half the rerank slots, capped at the curated pool size) and, when
    `k >= K_MIN`, reserves
    `k` anchors OUT OF `width` (never on top of it — `real_width = width -
    k`), makes ONE combined `rerank(query, real_documents[:real_width] +
    ANCHOR_POOL[:k])` call, splits the returned scores positionally back
    into real vs. anchor, and returns `raw_real_scores - median(anchor_
    scores)`. Computed FRESH on every call — unlike `_latency_cache` /
    `_provider_cache` above (which memoize call-STABLE properties: warm
    per-doc timing, a loaded model), the anchor offset is a per-QUERY
    property, so caching it across calls would defeat the point of the
    correction.

    No-ops (raw scores, no anchors appended) when `k < K_MIN` — `width` too
    small for a meaningful anchor median while still leaving a real
    candidate slot; see `AnchorNormalizationResult.did_normalize`.

    Total documents ever sent to `provider.rerank()` is exactly `width` in
    both branches (`real_width + k` when normalizing, `real_width` alone —
    `== width` — on no-op): never more than the budget-derived count
    `get_rerank_width` already enforces (I6).

    Anchor documents/scores are used ONLY to compute the median offset and
    are NEVER returned — callers must not treat them as candidates.

    Fail-soft guard: in the anchor-reserving branch (`k >= K_MIN`), the
    invariant `len(real_documents) >= real_width` always holds for
    correctly-wired callers (`real_documents` is the coarse-ranked pool
    backing `width = min(pool_size, ...) <= pool_size <= len(real_documents)`,
    and `real_width <= width`), but is not itself enforced here. A future
    caller that violates it would otherwise misalign the positional
    real/anchor split — `real_documents[:real_width]` silently returns fewer
    than `real_width` documents, so the positional
    `raw_scores[:real_width]` / `raw_scores[real_width:]` split spills
    anchor-scored positions into `real_scores` and can leave `anchor_scores`
    EMPTY, calling `statistics.median([])`, which raises `StatisticsError`
    and crashes recall. This is hot-path-adjacent, so a violation degrades
    gracefully instead of raising: it logs a warning (so the real bug stays
    visible) and no-ops to RAW scores over whatever `real_documents` are
    actually available (`did_normalize=False`), same shape as the `k <
    K_MIN` no-op above.
    """
    k = min(P, width // ANCHOR_SPLIT_DIVISOR)
    if k < K_MIN:
        real_width = width
        scores = list(provider.rerank(query, real_documents[:real_width]))
        return AnchorNormalizationResult(scores=scores, real_width=real_width, did_normalize=False)

    real_width = width - k
    if len(real_documents) < real_width:
        log.warning(
            "normalize_against_anchors: invariant len(real_documents) >= "
            "real_width violated (len(real_documents)=%d, real_width=%d, "
            "width=%d) — degrading to raw scores over the available "
            "real_documents, no anchors appended",
            len(real_documents),
            real_width,
            width,
        )
        scores = list(provider.rerank(query, real_documents))
        return AnchorNormalizationResult(scores=scores, real_width=len(real_documents), did_normalize=False)

    combined = real_documents[:real_width] + ANCHOR_POOL[:k]
    raw_scores = list(provider.rerank(query, combined))
    real_scores = raw_scores[:real_width]
    anchor_scores = raw_scores[real_width:]
    normalized = _median_normalize(real_scores, anchor_scores)
    return AnchorNormalizationResult(scores=normalized, real_width=real_width, did_normalize=True)


# ---------------------------------------------------------------------------
# Bundled representative (query, doc) pairs (F2a inc2, #250 §2 origin).
#
# Originally fed a cached first-use fp16-vs-fp32 accuracy self-check that
# lived in this module — REMOVED by the pre-flip revision's Change 2 (a
# confirmed ~1.16 GiB one-time dual-load memory spike; fp16-vs-fp32
# agreement is a property of the bundled model weights, proven by Testing's
# matched-width A/B, so no per-box runtime probe is needed to establish it
# — see `RERANKER_PRECISION` above). `_FP16_GATE_PAIRS` itself SURVIVES:
# `floor_calibration.py`'s cold-start and bootstrap floor derivation
# (`_cold_start_pairs` / `get_bootstrap_floor`) still score this same
# bundled set when no real corpus-derived pairs are available yet — that
# consumer is untouched by Change 2.
# ---------------------------------------------------------------------------

# Small BUNDLED representative (query, doc) pairs (bundled because a fresh
# install has no corpus yet to draw pairs from). The first 3 + next 3 pairs
# are the genuine/decoy set tests/unit/brain/memory/test_reranker_real_
# model.py already carries (itself reused from #88/test_search_memories_
# mode.py) — the closest existing checked-in "representative pair set" the
# original F2a §2 spec's open reconfirmation said to reuse if one exists.
# The last 4 are additional borderline/weakly-related pairs (topically
# adjacent but not a direct match) added for the diversity that same
# reconfirmation called for ("diverse relevant/irrelevant/borderline cases,
# not an arbitrary handful").
_FP16_GATE_PAIRS: list[tuple[str, str]] = [
    # genuine (clearly relevant)
    (
        "how do I calm down when everything feels like too much",
        "deep breathing helps when you are feeling anxious",
    ),
    ("quiet evening", "a quiet evening with nothing much happening"),
    (
        "what does Bob like to drink in the morning",
        "Bob always starts his day with a strong cup of black coffee",
    ),
    # decoy (clearly irrelevant)
    (
        "too much of a flood of party invitations this week",
        "how do I calm down when everything feels like too much",
    ),
    ("what's the capital of France", "my cat knocked a glass off the kitchen counter this morning"),
    (
        "how do I calm down when everything feels like too much",
        "the stock market closed higher today on tech earnings",
    ),
    # borderline (topically adjacent, weakly related — neither a clean
    # match nor a clean miss)
    (
        "what does Bob like to drink in the morning",
        "Bob mentioned he used to drink tea before switching to coffee last year",
    ),
    ("quiet evening", "a busy weekend trip with friends and lots of noise"),
    (
        "how do I calm down when everything feels like too much",
        "sometimes taking a walk outside clears my head a little",
    ),
    ("quiet evening", "the kitchen sink has been leaking for a week"),
]


def _register_fp16_reranker_model(fp16_model_id: str, hf_repo: str) -> None:
    """Idempotently register the jina fp16 onnx export with fastembed's
    `TextCrossEncoder`, via `add_custom_model()` — the confirmed mechanism
    (checked against the installed fastembed 0.8.0): fp16 is NOT in
    fastembed's built-in registry (only the fp32 export is), but the SAME
    HF repo (`hf_repo` — in practice `model_tier.MODEL_RERANKER`, the fp32
    id, since it names the identical repo) also ships
    `onnx/model_fp16.onnx` (~557MB, confirmed present on the repo).

    Registration is metadata-only (no network, no download — the download
    happens lazily on the registered model's first real `rerank()` call,
    same as the fp32 model). Guarded against `add_custom_model`'s own
    "already registered" `ValueError` so a second call in the same process
    (e.g. `build_reranker_provider`'s pinned-fp16 default resolving again
    after a provider-cache reset, or a test re-running this) is a no-op,
    not a crash.
    """
    from fastembed.common.model_description import ModelSource
    from fastembed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder

    already_registered = {m["model"] for m in TextCrossEncoder.list_supported_models()}
    if fp16_model_id in already_registered:
        return
    TextCrossEncoder.add_custom_model(
        model=fp16_model_id,
        sources=ModelSource(hf=hf_repo),
        model_file="onnx/model_fp16.onnx",
        description="fp16 export of jina-reranker-v2-base-multilingual, the pinned default reranker precision (#250, pre-flip revision Change 2)",
        license="cc-by-nc-4.0",
        size_in_gb=0.56,
    )
