"""Cross-encoder reranker provider + process-wide caches.

#231 RERANKER RE-ARCHITECTURE (``~/.claude/plans/memory-dream-rework-
semantic-retrieval-brief.md``, "RERANKER RE-ARCHITECTURE" section). Replaces
the scrapped Stage-4 per-persona cosine-floor/gap auto-calibration
(``brain/memory/semantic_calibration.py``, deleted) — that auto-calibration
derived a floor/gap from the corpus's own pairwise cosine spread, which the
cold red-team proved doesn't generalize (breaks silently on tight/diffuse/
bimodal corpora, because the query-match cosine scale is MODEL-FIXED, not
corpus-shaped). A cross-encoder reads (query, memory) TOGETHER and scores
true relevance — that score is query-conditioned, so a FIXED, empirically-set
floor on it (``brain.memory.semantic_recall.RERANK_FLOOR``) is trustworthy in
a way a cosine floor never was.

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
(``LATENCY_BUDGET_SECONDS``) is an ops tunable.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from brain import tunables
from brain.memory.relevance import CANDIDATE_POOL

log = logging.getLogger(__name__)

# Latency budget for one rerank call (spec point 3: "~1-2s", MEASURED
# no-AVX2 numbers put ~50 docs at ~1.1s). The ONLY operator-tunable knob in
# this module — ops-clean (a latency/throughput knob), unlike RERANK_FLOOR
# (physiology, fenced into semantic_recall.py per tunables.py's own
# "physiology fenced out" rule). Registered here (the owning module), read
# at call time via tunables.get_tunable so a live override applies with no
# restart.
LATENCY_BUDGET_SECONDS: float = tunables.register("reranker.latency_budget_seconds", 1.5)


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
    far below any plausible `RERANK_FLOOR`, so a test that seeds unrelated
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


def build_reranker_provider() -> RerankerProvider:
    """The production reranker provider: CrossEncoderProvider pinned to
    `model_tier.TIER_RERANKER`'s model id, caching the model file in the
    shared `get_cache_dir()` (one download across every persona on the box —
    the model isn't persona-specific data, same reasoning as the embedding
    model).

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
    from brain.bridge.model_tier import TIER_RERANKER, model_for_tier
    from brain.paths import get_cache_dir

    model_id = model_for_tier(TIER_RERANKER)

    provider = _provider_cache.get(model_id)
    if provider is not None:
        return provider

    with _provider_cache_lock:
        provider = _provider_cache.get(model_id)  # re-check: lost the race?
        if provider is not None:
            return provider
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


def get_rerank_width(
    pool_size: int,
    provider: RerankerProvider,
    sample_documents: list[str] | None = None,
) -> int:
    """How many of the (cosine-coarse-cut) candidate pool to actually rerank.

    `= max(1, min(pool_size, CANDIDATE_POOL, floor(LATENCY_BUDGET_SECONDS /
    warm_per_doc)))` — the spec's formula exactly. On fast (AVX2) hardware
    the measured per-doc latency is small, so the budget allows a wide
    rerank; on slow (no-AVX2) hardware it narrows automatically, staying
    within budget. An auto-scaler computing a width under 50 on a slow host
    is intended (not a violation) — the whole point is that the count
    adapts to the host, not to a fixed target.

    `sample_documents` (#231-fix): a small sample of REAL candidate-pool
    document content (see `CALIBRATION_SAMPLE_SIZE`), used to calibrate
    `warm_per_doc` against actual corpus documents instead of a synthetic
    placeholder — a fixed short stub massively under-measures real
    cross-encoder cost, which made this auto-scaler a structural no-op (it
    always computed a budget far in excess of `CANDIDATE_POOL`, so the min()
    always picked `CANDIDATE_POOL` regardless of true per-doc cost). Optional
    and purely additive: omitting it (or an empty list) falls back to the
    original fixed-placeholder calibration, unchanged.

    `pool_size <= 0` returns 0 (nothing to rerank — the caller's empty-pool
    case never reaches here in practice, but this stays well-defined).
    """
    if pool_size <= 0:
        return 0
    budget = tunables.get_tunable("reranker.latency_budget_seconds", LATENCY_BUDGET_SECONDS)
    per_doc = _warm_per_doc_latency(provider, sample_documents)
    if per_doc <= 0.0:
        width = pool_size
    else:
        width = math.floor(budget / per_doc)
    return max(1, min(pool_size, CANDIDATE_POOL, width))
