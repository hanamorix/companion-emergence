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

F2a inc2 (#250 §2) additionally owns the fp16-vs-fp32 accuracy SELF-CHECK: a
cached, first-use, on-box check (mirroring the warm-latency auto-calibration
pattern above — measured once, cached, not per-recall) that registers the
jina reranker's fp16 onnx export via ``TextCrossEncoder.add_custom_model()``,
tests it against the fp32 export on a small bundled representative set of
(query, doc) pairs for surface/abstain DECISION agreement against the
current ``RERANK_FLOOR``, and — only if that agreement holds AND fp16
measures faster on this host — ships fp16; otherwise fp32 (the safe
default). See ``_choose_reranker_model_id`` / ``_run_precision_selfcheck``
near the bottom of this module.
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
    whichever model id the fp16-vs-fp32 self-check (F2a inc2, #250 §2)
    decides to serve — `model_tier.TIER_RERANKER`'s fp32 model id, or its
    fp16 export (`model_tier.MODEL_RERANKER_FP16`) if that self-check's
    cached decision picked it — caching the model file in the shared
    `get_cache_dir()` (one download across every persona on the box — the
    model isn't persona-specific data, same reasoning as the embedding
    model).

    PROCESS-WIDE CACHING, same rationale as `embeddings.build_embedding_
    provider`: constructing a CrossEncoderProvider builds a real ONNX
    inference session — expensive to redo every recall. Keyed by model_id
    (double-checked locking: unlocked fast-path read for the common
    already-cached case; the lock is only taken — then re-checked — the
    first time a given model_id needs constructing) for the same reasons
    documented on that function. `_choose_reranker_model_id` below may
    already have populated this cache for the winning model_id (it warms
    both candidates to run the self-check, so it stashes whichever provider
    it already built for the DECIDED model_id here to avoid a second ONNX
    session load on this very first call) — the block below is then a
    fast-path cache hit, not a redundant construction.

    TEST ISOLATION: `tests/conftest.py`'s autouse fixture clears this
    process-global dict before and after every test, mirroring the embedding
    provider's own isolation fixture.
    """
    from brain.bridge.model_tier import MODEL_RERANKER_FP16, TIER_RERANKER, model_for_tier
    from brain.paths import get_cache_dir

    fp32_model_id = model_for_tier(TIER_RERANKER)
    cache_dir = get_cache_dir()
    model_id = _choose_reranker_model_id(fp32_model_id, MODEL_RERANKER_FP16, cache_dir)

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


def _cache_provider(model_id: str, provider: RerankerProvider) -> None:
    """Stash an already-constructed provider into the process-wide cache
    under `model_id`, WITHOUT clobbering one a concurrent caller already
    cached (`setdefault`) — used by `_run_precision_selfcheck` so a provider
    it already warmed (and paid the real ONNX-session-load cost for) while
    running the self-check is reused by `build_reranker_provider`'s own
    cache lookup rather than being constructed a second time."""
    with _provider_cache_lock:
        _provider_cache.setdefault(model_id, provider)


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


# ---------------------------------------------------------------------------
# fp16-vs-fp32 accuracy self-check (F2a inc2, #250 §2): a cached, first-use,
# on-box check that decides whether production serves the jina reranker's
# fp16 export or stays on fp32 — mirrors the warm-latency auto-calibration
# above (measured once, cached, not per-recall), not a GH-Actions
# release-pipeline step (see the ledger's FORK 2 resolution: fp16-vs-fp32
# agreement is a property of the BUNDLED model, computable anywhere, so
# first-use is as valid as a release-time gate, and first-use additionally
# covers `uv run` source installs the release pipeline never touches, and
# lets the check confirm fp16 is genuinely faster on THIS box before
# preferring it — a no-AVX2 potato CPU may not accelerate fp16).
# ---------------------------------------------------------------------------

# Small BUNDLED representative (query, doc) pairs for the self-check below
# (bundled because a fresh install has no corpus yet to draw pairs from —
# fp16-vs-fp32 agreement is a property of the MODEL, not of any one corpus,
# per §2). The first 3 + next 3 pairs are the genuine/decoy set
# tests/unit/brain/memory/test_reranker_real_model.py already carries
# (itself reused from #88/test_search_memories_mode.py) — the closest
# existing checked-in "representative pair set" the spec's open
# reconfirmation says to reuse if one exists. The last 4 are additional
# borderline/weakly-related pairs (topically adjacent but not a direct
# match) added for the diversity that same reconfirmation calls for
# ("diverse relevant/irrelevant/borderline cases, not an arbitrary
# handful"). This set is NOT used to derive a relevance floor — that is
# F2a §7, a separate, later increment — only to compare fp16 against fp32
# on the SAME comparison bar (`semantic_recall.RERANK_FLOOR`, read live,
# never copied/pinned here) both would use at recall time. That floor is
# still MiniLM-scaled as of this commit (see MODEL_RERANKER's model_tier.py
# comment) — a known, already-documented, EXPECTED state, not something
# this increment fixes — so this check is naturally low-power against it
# until §7 re-derives the floor for jina's own scale; reading the floor
# live (not hardcoding it) means this same gate sharpens automatically the
# day that lands, with no code change needed here.
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

# model_id -> chosen model_id ("fp32" or "fp16" candidate id, whichever the
# self-check decided). Keyed by the (fp32_model_id, fp16_model_id) PAIR, not
# a single id, so a model swap on EITHER side (a mini-model registration
# change per model_tier.py's own caveat comment) invalidates the cached
# decision automatically rather than serving a stale one. Process-wide,
# mirrors `_latency_cache` above — but unlike that cache this one is NOT
# periodically recomputed: the spec frames this as a one-time first-use
# cost for the life of the process, not a recurring measurement.
_precision_decision_cache: dict[tuple[str, str], str] = {}
_precision_decision_cache_lock = threading.Lock()


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
    "already registered" `ValueError` so a second self-check in the same
    process (or a test re-running this) is a no-op, not a crash.
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
        description="fp16 export of jina-reranker-v2-base-multilingual, for the F2a fp16/fp32 accuracy self-check (#250)",
        license="cc-by-nc-4.0",
        size_in_gb=0.56,
    )


def _run_precision_selfcheck(fp32_model_id: str, fp16_model_id: str, cache_dir: str | Path) -> str:
    """The actual (uncached — see `_choose_reranker_model_id`) fp16-vs-fp32
    self-check: registers the fp16 export, loads both it and fp32, tests
    surface/abstain DECISION agreement on `_FP16_GATE_PAIRS` against the
    live `semantic_recall.RERANK_FLOOR`, and — only on full agreement AND a
    measured fp16 speed win on this host — returns `fp16_model_id`.
    Returns `fp32_model_id` in every other case, INCLUDING any failure
    along the way (fail-soft: fp32 is always the safe default; this must
    never raise into a recall).

    ANY single flipped keep/drop decision fails the gate (mechanical bar,
    not an arbitrary loss percentage — per §2's "principled,
    behavior-preserving" bar and its open reconfirmation that the number
    must not be picked ad hoc).

    Whichever provider(s) this function actually constructs get stashed
    into the process-wide provider cache (`_cache_provider`) under their
    own model_id, so the ONE that wins is already warm (its ONNX session
    already loaded via the `.rerank()` calls below) by the time
    `build_reranker_provider` goes to construct it — the real, non-trivial
    load cost is paid exactly once, not twice, for the winning model.
    """
    try:
        _register_fp16_reranker_model(fp16_model_id, fp32_model_id)
    except Exception:  # noqa: BLE001 — fail-soft: registration failure must not break recall
        log.exception("reranker fp16/fp32 gate: failed to register the fp16 export -> shipping fp32")
        return fp32_model_id

    try:
        fp32_provider = CrossEncoderProvider(model_id=fp32_model_id, cache_dir=cache_dir)
        fp16_provider = CrossEncoderProvider(model_id=fp16_model_id, cache_dir=cache_dir)
    except Exception:  # noqa: BLE001 — fail-soft: fp16 (or even fp32) construction failure must not break recall
        log.exception("reranker fp16/fp32 gate: failed to construct a provider -> shipping fp32")
        return fp32_model_id

    try:
        # Deferred import: semantic_recall.py imports THIS module at its own
        # module scope (`from brain.memory import reranker as reranker_mod`),
        # so importing semantic_recall back at reranker.py's module scope
        # would be circular. A function-scoped import here is safe — by the
        # time this function is actually CALLED, both modules are fully
        # loaded regardless of which one happened to import first.
        from brain.memory.semantic_recall import RERANK_FLOOR

        for query, doc in _FP16_GATE_PAIRS:
            (fp32_score,) = fp32_provider.rerank(query, [doc])
            (fp16_score,) = fp16_provider.rerank(query, [doc])
            if (fp32_score >= RERANK_FLOOR) != (fp16_score >= RERANK_FLOOR):
                log.info(
                    "reranker fp16/fp32 gate: surface/abstain decision disagreement "
                    "(fp32=%.4f fp16=%.4f floor=%.4f) on pair %r -> shipping fp32",
                    fp32_score,
                    fp16_score,
                    RERANK_FLOOR,
                    (query, doc),
                )
                _cache_provider(fp32_model_id, fp32_provider)
                return fp32_model_id

        # Agreement holds on every bundled pair. Before preferring fp16,
        # confirm it is genuinely FASTER on THIS box (rationale iii, §2's
        # FORK 2 resolution) — a no-AVX2 potato CPU may not accelerate fp16,
        # and shipping it anyway would be a pure downside (same accuracy
        # bar, no latency win). Reuses the SAME warm per-doc measurement the
        # auto-scaling width calculation uses above, sampled against the
        # bundled pairs' own documents (already-loaded content, no extra
        # network).
        sample_docs = [doc for _, doc in _FP16_GATE_PAIRS]
        fp32_per_doc = _measure_warm_per_doc_latency(fp32_provider, sample_docs)
        fp16_per_doc = _measure_warm_per_doc_latency(fp16_provider, sample_docs)

        if fp16_per_doc < fp32_per_doc:
            log.info(
                "reranker fp16/fp32 gate: fp16 agrees with fp32 on every bundled decision and "
                "measured faster on this host (%.4fs vs %.4fs/doc) -> shipping fp16",
                fp16_per_doc,
                fp32_per_doc,
            )
            _cache_provider(fp16_model_id, fp16_provider)
            return fp16_model_id

        log.info(
            "reranker fp16/fp32 gate: decisions agree but fp16 was not faster on this host "
            "(%.4fs vs %.4fs/doc) -> shipping fp32",
            fp16_per_doc,
            fp32_per_doc,
        )
        _cache_provider(fp32_model_id, fp32_provider)
        return fp32_model_id
    except Exception:  # noqa: BLE001 — fail-soft: any self-check failure must not break recall
        log.exception("reranker fp16/fp32 gate: self-check failed -> shipping fp32")
        return fp32_model_id


def _choose_reranker_model_id(fp32_model_id: str, fp16_model_id: str, cache_dir: str | Path) -> str:
    """Cached entry point for the fp16-vs-fp32 self-check: a cache HIT
    returns instantly; a cache MISS runs `_run_precision_selfcheck` (real,
    potentially slow — first-use ONNX loads of BOTH exports plus scoring —
    which is why it runs OUTSIDE the lock, mirroring `_warm_per_doc_
    latency`'s identical reasoning: a concurrent caller must not block
    behind this one-time cost) and caches the result keyed by the
    `(fp32_model_id, fp16_model_id)` pair so either model changing
    invalidates it. A rare race where two callers both miss and both run
    the self-check pays the one-time cost twice in the worst case, never
    more — `setdefault` on write means whichever finishes first is the
    decision every later caller (and the other racer) actually gets.
    """
    cache_key = (fp32_model_id, fp16_model_id)
    with _precision_decision_cache_lock:
        cached = _precision_decision_cache.get(cache_key)
    if cached is not None:
        return cached

    decision = _run_precision_selfcheck(fp32_model_id, fp16_model_id, cache_dir)

    with _precision_decision_cache_lock:
        _precision_decision_cache.setdefault(cache_key, decision)
        return _precision_decision_cache[cache_key]


def _reset_precision_decision_cache() -> None:
    """Test-only: clear the cached fp16-vs-fp32 decision."""
    with _precision_decision_cache_lock:
        _precision_decision_cache.clear()
