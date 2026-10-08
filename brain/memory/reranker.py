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

Also owns the PER-MESSAGE rerank-width fit (name-recall fix R1, spec §1,
S4/S23/S24/S32/S62/S67): the hourly width sample (one query's top-5 docs,
cached ~1 h, diagnosis H8) is gone. Every recall-time rerank is timed and
feeds a per-process, per-reranker-model cost model (a per-call overhead plus
a per-token rate over the batch's padded size in reranker tokens, separated by
least squares over running sums) and a RAM-per-token figure (RSS delta around
the same call); each message then fits its own width to the lengths of its
own candidates within the latency budget (``LATENCY_BUDGET_SECONDS``, the one
ops tunable) and the measured RAM bound. See the "Per-message rerank width"
section and ``rerank_for_recall``.

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
import statistics
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from brain import tunables
from brain.dev_constants import RERANK_MIN_REAL_CANDIDATES
from brain.memory import cost_series
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
# `rerank_for_recall` below) — this AVX2-awareness only changes what the
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

    def rerank_timed(
        self, query: str, documents: list[str]
    ) -> tuple[list[float], float, float | None]:
        """`rerank()` plus the measurements the per-message width fit learns
        from (name-recall fix R1, plan P-3): `(scores, seconds, rss_delta_bytes)`.

        `seconds` is the wall time of the scoring call, read from the module
        clock seam `_clock`; `rss_delta_bytes` is this process's RSS after
        minus before the call (`None` when RSS is unreadable, e.g. not Linux),
        the per-token RAM measurement S32 asks for. Only RECALL paths call
        this (passive recall and `search_memories`, via
        `normalize_against_anchors`); bootstrap-floor and calibration scoring
        call plain `rerank()` and so never feed the cost model (S24).

        `CrossEncoderProvider` overrides this to take both readings INSIDE its
        instance lock, so a concurrent recall waiting on the lock never
        inflates the measured time."""
        before = _current_rss_bytes()
        start = _clock()
        scores = list(self.rerank(query, documents))
        seconds = _clock() - start
        return scores, seconds, _rss_delta(before, _current_rss_bytes())

    def pair_token_lengths_timed(
        self, query: str, documents: Sequence[str]
    ) -> tuple[list[int] | None, float | None]:
        """`pair_token_lengths` plus its seconds measured inside the provider's
        lock (round-3 N3-1). This default measures nothing (`None`): the caller
        times the call itself (test doubles hold no lock)."""
        return self.pair_token_lengths(query, documents), None

    def pair_token_lengths(self, query: str, documents: Sequence[str]) -> list[int] | None:
        """Per document, the length in reranker TOKENS of the (query, document)
        pair as the model sees it, after truncation (S62, S75): the cost model's
        size unit, because the reranker pays per padded token, not per
        character. `None` means token counts are unavailable.

        This default has no tokenizer, so it counts characters as a stand-in
        (`len(query) + len(doc)`): a provider with no tokenizer is only a test
        double, where the unit is whatever the test scripts consistently.
        `CrossEncoderProvider` overrides it with the loaded model's own
        tokenizer."""
        return [len(query) + len(doc) for doc in documents]


class CrossEncoderProvider(RerankerProvider):
    """Real local cross-encoder reranker via `fastembed` (ONNX runtime, no torch).

    Production default. Model id comes from `model_tier.py`
    (`model_for_tier(TIER_RERANKER)`), never hardcoded here — same
    convention as `FastEmbedProvider` in `embeddings.py`.

    The model file is downloaded once into `cache_dir` and used fully offline
    after — no network call happens at rerank() time once the file is cached.
    Construction DOES download it when it is not cached: fastembed's
    `OnnxTextCrossEncoder.__init__` calls `download_model()` unconditionally,
    and `lazy_load=True` defers only building the ONNX session to the first
    `rerank()` call (the same posture as `FastEmbedProvider`). So never
    construct one just to read its id: use `resolve_reranker_model_id()`.
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

    def rerank_timed(
        self, query: str, documents: list[str]
    ) -> tuple[list[float], float, float | None]:
        """See `RerankerProvider.rerank_timed`. Both readings are taken while
        holding `_rerank_lock`, so time spent WAITING for another thread's
        rerank is never counted as this call's cost (plan P-3)."""
        with self._rerank_lock:
            before = _current_rss_bytes()
            start = _clock()
            scores = list(self._model.rerank(query, documents))
            seconds = _clock() - start
            after = _current_rss_bytes()
        return scores, seconds, _rss_delta(before, after)

    def pair_token_lengths(self, query: str, documents: Sequence[str]) -> list[int] | None:
        return self.pair_token_lengths_timed(query, documents)[0]

    def pair_token_lengths_timed(
        self, query: str, documents: Sequence[str]
    ) -> tuple[list[int] | None, float | None]:
        """`pair_token_lengths` with its seconds read inside `_rerank_lock`, so a
        wait behind another thread's rerank (e.g. the re-measure job's) is
        never timed as work (round-3 N3-1).

        Surviving pair lengths in tokens (S62, S75, plan P-32): each (query,
        document) pair is encoded with the loaded model's OWN tokenizer (the
        same `encode_batch` call fastembed makes before scoring, truncation at
        the model maximum already enabled on it) and counted with
        `surviving_pair_token_lengths`. Fail-soft: on any failure the result is
        `None` (logged), never a character count, so a different unit can never
        reach the token-based cost model; the caller then does not rerank that
        query (hand-off "sizes", `rerank_for_recall`): a batch whose size is
        unknown cannot be held to the latency budget or the RAM bound."""
        try:
            with self._rerank_lock:
                tokenizer = self._pair_tokenizer()
                start = _clock()
                encodings = tokenizer.encode_batch([(query, doc) for doc in documents])
                seconds = _clock() - start
            return surviving_pair_token_lengths(encodings), seconds
        except Exception:  # noqa: BLE001 — fail-soft: a length probe must never break recall
            # Warn with the traceback once per provider; later failures (the
            # same broken accessor, every message) log at debug level.
            first = not getattr(self, "_length_probe_failed", False)
            self._length_probe_failed = True
            (log.warning if first else log.debug)(
                "reranker: surviving pair token lengths unavailable — sizing the rerank without them",
                exc_info=first,
            )
            return None, None

    def _pair_tokenizer(self) -> Any:
        """The loaded fastembed cross-encoder's tokenizer (a PRIVATE fastembed
        attribute, `TextCrossEncoder.model.tokenizer`; the same class of
        coupling as the lazy-load notes above, kept behind this one accessor).
        The tokenizer only exists once the ONNX model has loaded
        (`lazy_load=True`), so this loads it the way fastembed's own first
        `rerank()` would. Caller holds `_rerank_lock` (the same lock that
        guards fastembed's unlocked first-load path)."""
        inner = self._model.model
        if getattr(inner, "model", None) is None:
            inner.load_onnx_model()
        tokenizer = inner.tokenizer
        if tokenizer is None:
            raise RuntimeError("reranker tokenizer not loaded")
        return tokenizer

    def model_id(self) -> str:
        return self._model_id


def surviving_pair_token_lengths(encodings: Sequence[Any]) -> list[int]:
    """Tokens of each (query, document) pair that the reranker actually runs,
    counted from each pair encoding after truncation (S62, S75): the tokens
    the tokenizer kept, special tokens included (the model pays for those
    too), padding excluded. The encoding's `attention_mask` is 1 for exactly
    those tokens; the batch's padding to its own longest pair (which the
    tokenizer adds on `encode_batch`) carries 0, so it is not counted. A
    pair truncated to the model maximum therefore counts at the maximum, and
    a short pair at its own length.

    Counted per pair, not per character: the same number of characters is a
    very different token count in Latin text and in CJK or emoji text."""
    return [sum(encoding.attention_mask) for encoding in encodings]


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


# Unrecognised `reranker.precision` values already warned about (the due checks
# call `resolve_reranker_model_id` on every idle pass: warn once per value).
_warned_precisions: set[object] = set()


def resolve_reranker_model_id() -> str:
    """The runtime reranker model id (the id `build_reranker_provider()`'s
    provider reports from `model_id()`): `model_tier.MODEL_RERANKER_FP16` (the
    pinned default) or `TIER_RERANKER`'s fp32 id when the `reranker.precision`
    tunable says "fp32". Pure and cheap: builds and loads nothing, so the
    floor-bootstrap due checks can use it every pass."""
    from brain.bridge.model_tier import MODEL_RERANKER_FP16, TIER_RERANKER, model_for_tier

    precision = tunables.get_tunable("reranker.precision", RERANKER_PRECISION)
    if precision == RERANKER_PRECISION_FP32:
        return model_for_tier(TIER_RERANKER)
    if precision != RERANKER_PRECISION_FP16 and precision not in _warned_precisions:
        _warned_precisions.add(precision)
        log.warning(
            "reranker: unrecognized reranker.precision override %r — falling back to "
            "the pinned default %r",
            precision,
            RERANKER_PRECISION,
        )
    return MODEL_RERANKER_FP16


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
    `search_memories._semantic_top_k`) — it is no longer read by this
    function at all. The supervisor's calibration tick and deploy
    recalibration job do not build a provider: they read the id from
    `resolve_reranker_model_id()`.

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
    model_id = resolve_reranker_model_id()
    if model_id == MODEL_RERANKER_FP16:
        _register_fp16_reranker_model(MODEL_RERANKER_FP16, fp32_model_id)

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
    for a second one. In production a cache HIT is the rule: the only
    caller path, `floor_startup.run_rerank_floor` (first need, the per-message
    retry, or the next-lull retry job; name-recall fix S91/S92), calls
    `build_reranker_provider()` first, which registers the fp16 model and
    caches the provider for the runtime model_id.
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
# Per-message rerank width: the cost model (name-recall fix R1, spec §1).
# ---------------------------------------------------------------------------
#
# Replaces the hourly width measurement (`get_rerank_width` and its
# `_latency_cache` / `_memory_cache`: one query's top-5 documents, timed once
# and cached for an hour, diagnosis H8; a message of long monologue traces
# at the top of the hour pinned every message of that hour to a width of 2
# to 4). Now EVERY recall-time rerank is a measurement (S24):
#
#   - A rerank batch of `pairs` documents whose longest (query, document)
#     pair is `L` reranker tokens costs `overhead + rate * x` seconds,
#     x = pairs * L,
#     because the reranker pads every pair in a batch to the longest one
#     (fastembed `preprocessor_utils.load_tokenizer` enables padding; S67).
#   - `overhead` and `rate` are separated by ordinary least squares over the
#     samples of the recall reranks (and the S100 re-measure job's), per
#     reranker model id (spec §1). The samples are the last 24 hours' (S146,
#     S114; `cost_series`), saved in memories.db so a restart continues from
#     them, with no averaging constant: every sample in the window weighs the
#     same.
#   - RAM per padded token of GROWTH (S32: RSS delta around recall-time
#     reranks). The ONNX runtime keeps the memory a batch needed and reuses
#     it, so a call whose padded size x is within the largest size already
#     run (the high water) shows no RSS growth, and only a call that goes
#     past the high water grows RSS, by about (x - high water) * RAM/token.
#     So RAM/token = sum of max(0, RSS delta) / sum of (x - high water) over
#     the calls that went past it, and a candidate batch is predicted to need
#     RAM/token * max(0, x - high water) more memory, checked against the
#     current headroom. (Averaging deltas over EVERY call would dilute the
#     figure towards 0 as calls within the high water accumulate, and the
#     bound would stop binding: stage-6 finding F1.)
#   - The size unit is reranker TOKENS (S75), counted per pair from the
#     reranker's own tokenizer after truncation (`pair_token_lengths`), not
#     characters: the reranker pays per padded token, and the same number
#     of characters is a very different token count in Latin text and in CJK
#     or emoji text (samples through the shipped tokenizer: about 0.2 tokens
#     per character in plain English words, about 0.5 in Chinese, about 1.0
#     in emoji).
#
# Only recall reranks feed it (`rerank_timed`, called by
# `normalize_against_anchors`); the two warm-up reranks a process runs first
# are discarded, as the hourly measurement discarded them (S24/S32).
#
# The time series and the RAM figure are rolling 24-hour windows saved in
# memories.db (S146, S171; see `cost_series`); the high water and the warm-up
# state stay process memory. The first measured recall rerank of a process per
# model id is sized to the S5 minimum only on the first-ever startup on this
# machine (no saved time estimate, or none with a RAM figure above 0 in its
# 24-hour window: a figure whose samples have all aged out is no figure, S178)
# where RSS is readable (S158, S171), because it measures the RAM growth. A process
# that has recorded a measured rerank keeps the last RAM figure it saw when that
# figure's window empties (S180: it never reranks without a RAM bound, and a
# plateaued process is exactly the one whose window empties, since a RAM sample
# exists only for a call past the high water); the next lull re-seeds it (below).
# Stuck-width recovery (S100, S113, S116, S180): the last query that
# went to the cosine path for lack of budget is kept per model id; while the
# estimate says its first 5 candidates do not fit the full latency budget (time
# term), or the KEPT RAM figure alone keeps them from fitting the headroom, the
# `rerank_width_remeasure` gated job re-measures them once per chat lull
# (`stuck_width_due`, `remeasure_stuck_width`); a re-measure whose figure is the
# kept one skips the RAM-headroom check, so its 5-candidate batch (past the high
# water, because the figure blocked it) records the sample that re-seeds the figure.

# Clock seam: `rerank_timed` reads elapsed time through this name so tests can
# inject a scripted clock (criterion C1a). Production: `time.monotonic`.
_clock: Callable[[], float] = time.monotonic

# Cold-cache timing trap ([[single-shot-timing-cold-cache-trap]]): the first
# rerank() calls on a freshly-constructed provider pay ONNX session warm-up
# cost far above steady state. The first recall rerank of a process (per
# reranker model id) is preceded by this many single-document reranks whose
# results and timings are discarded (S24/S32 "two warm-up reranks discarded
# as today").
_WARMUP_RERANKS = 2


def _current_rss_bytes() -> float | None:
    """This process's current resident set size, in bytes, via
    `/proc/self/status`'s `VmRSS` line — the same shape as `_detect_avx2`'s
    `/proc` read (Linux-only; fail-soft to `None` on any read/parse error or
    an unsupported platform, since there is no `/proc` on macOS/Windows).
    Never raises. `None` makes the RAM term of the width fit skip (the width
    is then time-bound only), never assume unlimited memory."""
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
        log.exception("reranker: RSS read failed — the rerank's RAM measurement will be skipped")
        return None


def _rss_delta(before: float | None, after: float | None) -> float | None:
    """RSS after minus before, or `None` when either read failed. May be
    negative (a GC pass, another thread freeing memory); the running sum
    clamps each delta at 0."""
    if before is None or after is None:
        return None
    return after - before


@dataclass(frozen=True)
class _RamState:
    """One reranker model id's process-local RAM state: the high water, the
    largest padded size this process has measured (a property of the ONNX arena
    this process holds, S74; a new process starts at 0). The RAM growth figure
    itself is a saved series (`_ram_cost`, S171), not part of this state. The
    model id having an entry means this process has recorded a measured rerank
    for it (S158). Immutable: an update builds a new instance and swaps it in
    under `_cost_lock`."""

    peak_x: int = 0  # largest padded size measured so far (the high water)

    def plus(self, padded_tokens: int) -> _RamState:
        return _RamState(peak_x=max(self.peak_x, padded_tokens))


@dataclass(frozen=True)
class RerankCostEstimate:
    """The fitted cost model for one reranker model id (see the section
    header). `ram_bytes_per_token` is RSS growth per padded token past
    `peak_padded_tokens` (the largest padded batch this process has measured);
    `None` when no call past a high water with a readable RSS delta is in the
    24-hour window, saved or buffered, and this process has no kept figure (S178:
    the RAM term of the width fit is then skipped, and so is the re-measure job's
    RAM-headroom check). `ram_figure_kept` is True when the window is empty and
    `ram_bytes_per_token` is the last figure this running process saw (S180): it
    bounds the width like a window figure, and a re-measure that finds it blocking
    re-seeds it. `measured_batches` = the measurements behind the time fit in effect."""

    overhead_seconds: float
    seconds_per_token: float
    ram_bytes_per_token: float | None
    measured_batches: int
    peak_padded_tokens: int = 0
    ram_figure_kept: bool = False


@dataclass(frozen=True)
class _BudgetMiss:
    """The last query per reranker model id that went to the cosine path for
    lack of budget (S113): its text, its first `RERANK_MIN_REAL_CANDIDATES`
    prefix candidates and their pair token sizes, and the anchors' pair sizes.
    Held until a rerank proves 5 fit (S116 a), the re-measure job clears or
    drops it (S116), or a newer miss replaces it. Holds one query text (a whole
    paste on the fallback path) and 5 memory texts per model id."""

    query: str
    documents: tuple[str, ...]
    candidate_tokens: tuple[int, ...]
    anchor_tokens: tuple[int, ...]


# model_id -> this process's high water (the RAM figure itself is the saved series
# `_ram_cost`), the model ids whose warm-up has run, the last query per
# model id that went to the cosine path for lack of budget (S113), the chat
# activity marker of each model id's last re-measure (S100), and the "too long"
# verdicts (S127: remembered until the cost estimate next changes, decided by
# which samples the window holds, not by comparing fitted floats; saved with the
# host-keyed series and read back once per process, S179). One lock
# guards them all (every read and every read-modify-write), so a concurrent
# passive recall, `search_memories` call (#297: the tool runs in a bridge
# worker thread) and the re-measure job never lose an update. The time series
# itself is `_rerank_cost` (rolling window, memories.db, `cost_series`).
_rerank_cost = cost_series.CostSeries("rerank")
# S171: RSS growth per padded token past the high water, saved like the time series
# (host-keyed, 24 h window); its estimate is the weighted ratio of the window's sums
# (`cost_series.THROUGH_ORIGIN_SERIES`) and it has none once the window is empty
# (S178: its carried row is not read); a running process then keeps the last figure it
# saw (`_ram_kept`, S180).
_ram_cost = cost_series.CostSeries(cost_series.RAM_SERIES)
_ram_state: dict[str, _RamState] = {}
# S180: (database, model id) -> the last RAM-per-token figure (above 0) an estimate call saw
# in the window. Used only by a process whose `_ram_state` holds the model id (it has recorded
# a measured rerank), and only while the window answers nothing. Process memory, never saved;
# guarded by `_cost_lock` with `_ram_state`.
_ram_kept: dict[tuple[str, str], float] = {}
_warmed_model_ids: set[str] = set()
_budget_misses: dict[str, _BudgetMiss] = {}
_remeasured_marker: dict[str, object] = {}


@dataclass(frozen=True)
class _Verdict:
    """S127: the wall time a "too long" verdict was reached at and how many
    window samples had been taken at or before it. The verdict holds while no
    sample was taken after that time and those samples are all still in the
    window."""

    taken_at: float
    before: int


_too_long: dict[str, _Verdict] = {}
# S179: the (database, model id) pairs whose saved verdict row this process has read
# (or written), so the row is read once per process, not at every due check.
_too_long_loaded: set[tuple[str, str]] = set()
_cost_lock = threading.Lock()

# Test-only seam (CONC-1): called inside an update, between reading the
# current RAM state and writing the new one. Always `None` in production.
_cost_update_hook: Callable[[], None] | None = None


def rerank_cost_estimate(model_id: str, store: MemoryStore) -> RerankCostEstimate | None:
    """The current cost model for `model_id`: the time fit of the last 24
    hours (S146; saved, so available from the first message after a restart),
    this process's high water and the saved RAM figure (S171). `None` when there
    is no time estimate (the first startup on this machine: the width is then
    the S5 minimum, S24). The high water is read BEFORE the RAM figure:
    `_record_rerank_cost` buffers a sample before it publishes the new high
    water, so a reader that sees the new high water also sees its sample.

    S180: the RAM figure is the window's while the window answers (a rate above
    0 is also remembered as this process's last figure; a rate of 0 is no figure
    and forgets it). When the window is empty and this process has recorded a
    measured rerank for `model_id`, it is the last figure an earlier call of
    this process saw (`ram_figure_kept`): a running process never reranks
    without a RAM bound. A process that has not measured (a fresh one) has no
    kept figure to use, so its first measured rerank is the width-5 seed (S178)."""
    fit = _rerank_cost.fit(store, model_id)
    if fit is None:
        return None
    with _cost_lock:
        ram = _ram_state.get(model_id, _RamState())
    ram_fit = _ram_cost.fit(store, model_id)
    key = (cost_series._store_key(store), model_id)
    rate: float | None = None if ram_fit is None else ram_fit.rate
    kept = False
    with _cost_lock:
        if ram_fit is not None:
            if ram_fit.rate > 0.0:
                _ram_kept[key] = ram_fit.rate
            else:
                _ram_kept.pop(key, None)
        elif model_id in _ram_state and key in _ram_kept:
            rate, kept = _ram_kept[key], True
    return RerankCostEstimate(
        overhead_seconds=fit.overhead,
        seconds_per_token=fit.rate,
        ram_bytes_per_token=rate,
        measured_batches=fit.samples,
        peak_padded_tokens=ram.peak_x,
        ram_figure_kept=kept,
    )


def _record_rerank_cost(
    model_id: str,
    store: MemoryStore,
    padded_tokens: int,
    seconds: float,
    rss_delta_bytes: float | None,
) -> None:
    """Add one measured rerank: its RAM growth (a call that went past this
    process's high water with a readable RSS delta) and its time to the buffered
    series (S164: nothing is written to the database here). The RAM sample is
    buffered BEFORE the new high water is published (`rerank_cost_estimate` reads
    them in the opposite order), both under `_cost_lock`; the buffer's own lock is
    only ever taken inside it."""
    with _cost_lock:
        current = _ram_state.get(model_id, _RamState())
        if _cost_update_hook is not None:
            _cost_update_hook()
        if rss_delta_bytes is not None and padded_tokens > current.peak_x:
            _ram_cost.record(
                store, model_id, padded_tokens - current.peak_x, max(0.0, rss_delta_bytes)
            )
        _ram_state[model_id] = current.plus(padded_tokens)
    _rerank_cost.record(store, model_id, padded_tokens, seconds)


def _remember_too_long(model_id: str, store: MemoryStore) -> None:
    """S127: remember that the job measured the stuck record's 5 candidates
    over the full budget. The job's own sample is already buffered, so it was
    taken at or before this moment and counts in the held total; the verdict
    then ends only when a sample arrives or one of those ages out. A failed read
    remembers nothing (the job simply runs again at the next lull)."""
    try:
        now = cost_series._wall_clock()
        before, _after = _rerank_cost.window_counts(store, model_id, now)
    except Exception:  # noqa: BLE001 - fail soft
        log.warning("rerank width re-measure: could not record a too-long verdict", exc_info=True)
        return
    marker = (cost_series._store_key(store), model_id)
    with _cost_lock:
        _too_long[model_id] = _Verdict(now, before)
        _too_long_loaded.add(marker)
    # S179: saved with the host-keyed series, so a restart does not run the
    # over-budget rerank again. Written here, by the lull job (never on the reply
    # path, S164); a failed write leaves the in-process verdict standing.
    try:
        _rerank_cost.save_verdict(store, model_id, now, before)
    except Exception as exc:  # noqa: BLE001 - fail soft; warned once, then debug
        cost_series._note_failure("verdict save", exc)


def _load_saved_verdict(model_id: str, store: MemoryStore) -> _Verdict | None:
    """The in-process verdict for `model_id`, first reading the saved row once per
    (database, model id) per process (S179: a verdict reached before a restart).
    A failed read means no verdict for now (the job may run; the read is tried
    again at the next consult): there is nothing in this process to keep, and a
    database that cannot answer here cannot supply the estimate that makes the
    job due either. The database is read outside `_cost_lock`."""
    marker = (cost_series._store_key(store), model_id)
    try:
        row = _rerank_cost.load_verdict(store, model_id)
    except Exception:  # noqa: BLE001 - fail soft: no verdict, tried again next time
        log.debug("rerank width re-measure: saved verdict read failed", exc_info=True)
        return None
    with _cost_lock:
        _too_long_loaded.add(marker)
        held = _too_long.get(model_id)
        if held is not None or row is None:
            return held
        loaded = _Verdict(row[0], row[1])
        _too_long[model_id] = loaded
        return loaded


def _end_verdict(model_id: str, store: MemoryStore, verdict: _Verdict) -> None:
    """The verdict is over (S175): forget it in this process and delete its saved
    row (only if it is still the one reached at `verdict.taken_at`; best effort)."""
    with _cost_lock:
        if _too_long.get(model_id) is verdict:
            del _too_long[model_id]
    try:
        _rerank_cost.clear_verdict(store, model_id, verdict.taken_at)
    except Exception:  # noqa: BLE001 - fail soft: a stale row only ends again at the next load
        log.debug("rerank width re-measure: could not delete the saved verdict", exc_info=True)


def _too_long_remembered(model_id: str, store: MemoryStore) -> bool:
    """Whether `model_id` has a remembered S127 "too long" verdict that still
    holds, in this process or saved by an earlier one on this machine (S179). A
    verdict whose window changed (a sample arrived after it, or one it counted
    aged out) is dropped here, when consulted, and its saved row deleted. A failed
    read of the window keeps the verdict (no over-budget re-run on a database
    hiccup)."""
    marker = (cost_series._store_key(store), model_id)  # outside the lock: it resolves a path
    with _cost_lock:
        verdict = _too_long.get(model_id)
        unread = verdict is None and marker not in _too_long_loaded
    if unread:
        verdict = _load_saved_verdict(model_id, store)
    if verdict is None:
        return False
    try:
        before, after = _rerank_cost.window_counts(store, model_id, verdict.taken_at)
    except Exception:  # noqa: BLE001 - fail soft: keep the verdict
        log.debug("rerank width re-measure: verdict read failed", exc_info=True)
        return True
    if after == 0 and before == verdict.before:
        return True
    _end_verdict(model_id, store, verdict)
    return False


def _ram_unseeded(model_id: str, estimate: RerankCostEstimate) -> bool:
    """S158/S171: this process has not yet recorded a measured rerank for
    `model_id`, RSS is readable (Linux) and no RAM figure greater than 0 is saved
    for this model id on this machine, so its first measured recall rerank must run
    at the S5 minimum to measure RAM growth, whatever the saved time estimate says.
    "Saved" means a sample in the 24-hour window (S178): a figure whose samples
    have all aged out is no figure, so a too-high one is re-seeded here instead of
    staying for good. With a figure (a restart soon after the first-ever startup
    on this host) it is False: the first rerank is sized by the estimate. A process
    that has already recorded a rerank is never unseeded (its RAM window can empty
    while it runs; then it keeps the last figure it saw, `ram_figure_kept`, S180, and
    the re-measure job re-seeds it at a lull if it blocks a batch). A process that
    has not recorded a rerank never uses a kept figure, so its estimate is the
    window's alone."""
    with _cost_lock:
        if model_id in _ram_state:
            return False
    figure = estimate.ram_bytes_per_token
    if figure is not None and figure > 0.0:
        return False
    return _current_rss_bytes() is not None


def _reset_rerank_cost_model() -> None:
    """Test-only: forget every model id's RAM state, warm-up state, budget
    misses, re-measure markers and "too long" verdicts. The saved series lives
    in the database and the process buffer (`cost_series._reset_pending`)."""
    with _cost_lock:
        _ram_state.clear()
        _ram_kept.clear()
        _warmed_model_ids.clear()
        _budget_misses.clear()
        _remeasured_marker.clear()
        _too_long.clear()
        _too_long_loaded.clear()


def _full_rerank_budget() -> float:
    """The full per-call rerank latency budget (`reranker.latency_budget_seconds`,
    read now): what "stuck" and the S116 clear are judged against (S113, S116),
    never a paragraph's share or a message's remainder."""
    return float(tunables.get_tunable("reranker.latency_budget_seconds", LATENCY_BUDGET_SECONDS))


def _is_stuck(miss: _BudgetMiss, estimate: RerankCostEstimate | None) -> bool:
    """S113: the estimate predicts that fewer than 5 real candidates of the
    miss fit the full budget (time term only; the RAM bound is not the learned
    cost). The re-measure job's other trigger, a kept RAM figure that blocks the
    miss (S180), is `_needs_remeasure`."""
    if estimate is None:
        return False
    width = fit_rerank_width(
        miss.candidate_tokens,
        miss.anchor_tokens,
        estimate,
        _full_rerank_budget(),
        None,
        RERANK_MIN_REAL_CANDIDATES,
    )
    return width < RERANK_MIN_REAL_CANDIDATES


def _needs_remeasure(miss: _BudgetMiss, estimate: RerankCostEstimate | None) -> bool:
    """Whether the re-measure job should time the miss's 5 candidates (S100, S113,
    S180): the time term says they are stuck (`_is_stuck`), or the RAM figure in use
    is the KEPT one (a running process whose window emptied) and it keeps them from
    fitting the current headroom. The second case is the re-seed: that figure is stale,
    and the batch it blocks is past the high water, so measuring it records the RAM
    sample that makes the figure present again. A figure still in the window is a
    legitimate bound and never triggers the job."""
    if estimate is None:
        return False
    if _is_stuck(miss, estimate):
        return True
    if not estimate.ram_figure_kept:
        return False
    width = fit_rerank_width(
        miss.candidate_tokens,
        miss.anchor_tokens,
        estimate,
        _full_rerank_budget(),
        _available_ram_headroom_bytes(),
        RERANK_MIN_REAL_CANDIDATES,
    )
    return width < RERANK_MIN_REAL_CANDIDATES


def _drop_miss(model_id: str, miss: _BudgetMiss) -> None:
    """Remove `model_id`'s miss iff it is still `miss` (a newer one survives,
    CONC-8)."""
    with _cost_lock:
        if _budget_misses.get(model_id) is miss:
            del _budget_misses[model_id]


def _job_state(model_id: str) -> tuple[_BudgetMiss | None, object, bool]:
    """(miss, the marker of the last re-measure or `_NEVER`, warmed) under the
    lock."""
    with _cost_lock:
        return (
            _budget_misses.get(model_id),
            _remeasured_marker.get(model_id, _NEVER),
            model_id in _warmed_model_ids,
        )


# "No re-measure yet" for a model id: distinct from every chat activity
# marker value, including the `-inf` of a process that has seen no message.
_NEVER = object()

# Test-only seam (CONC-8): called by `remeasure_stuck_width` between its
# measured rerank and its clear/drop. `None` in production.
_remeasure_hook: Callable[[], None] | None = None


def stuck_width_due(
    activity_marker: object, store_factory: Callable[[], AbstractContextManager[MemoryStore]]
) -> bool:
    """Whether the `rerank_width_remeasure` job is due (S100, S113, S180): the
    runtime reranker model id has a budget miss that is stuck under the
    current estimate (the time term says fewer than 5 candidates fit, or the
    kept RAM figure of a running process blocks them: `_needs_remeasure`), its warm-up has run in this process and its provider is
    cached (the job never builds or loads a model), no re-measure has run
    at this chat activity marker (once per lull), and no "too long" verdict is
    remembered (S127: the job's own measured 5 candidates exceeded the full
    budget, and no sample has arrived or aged out since, so a new record does
    not re-trigger the job until the estimate changes). A miss found not stuck
    is dropped. The in-memory checks come first; `store_factory` (a context
    manager yielding a store, opened only past them) supplies the saved series
    and the saved verdict, which this check may read (once per process) and delete
    when the verdict has ended (S179: the one write a due check makes; it runs in
    the lull job, never on the reply path). Takes `_cost_lock` only briefly,
    never nested."""
    model_id = resolve_reranker_model_id()
    miss, stamp, warmed = _job_state(model_id)
    if miss is None or not warmed or _provider_cache.get(model_id) is None:
        return False
    if stamp is not _NEVER and stamp == activity_marker:
        return False
    with store_factory() as store:
        if _too_long_remembered(model_id, store):
            return False
        if not _needs_remeasure(miss, rerank_cost_estimate(model_id, store)):
            _drop_miss(model_id, miss)
            return False
    return True


def remeasure_stuck_width(activity_marker: object, store: MemoryStore) -> bool:
    """The `rerank_width_remeasure` job's run (S100, S113, S116): one timed
    rerank of the stuck miss's 5 candidates (5 real + `anchor_count(5)`
    anchors) on the already-loaded provider, added to the cost sums. Then the
    miss is cleared when the measured seconds are within the full budget
    (S116 a: 5 fit) and dropped when they exceed it (S116 b: genuinely too
    long, not stuck), and that verdict is remembered until the estimate next
    changes (S127: a sample arrives or one it counted ages out). The marker
    is stamped after the verdict check and before the measurement, so a failing
    or RAM-skipped run still runs once per lull, while a run skipped by a held
    verdict stamps nothing. Skipped (miss kept) when the miss is no longer
    stuck (then dropped; stuck = the time term, or the kept RAM figure of a running
    process blocks it, S180), a "too long" verdict is remembered, or the RAM figure
    in the window predicts the batch does not fit the current headroom. A figure that
    is only the process's kept one (its window emptied, S180) does not skip the run:
    the 5-candidate batch is past the high water and re-seeds it. Returns True iff the
    rerank ran. Provider exceptions propagate (the central pass logs them)."""
    model_id = resolve_reranker_model_id()
    miss, _, warmed = _job_state(model_id)
    provider = _provider_cache.get(model_id)
    if miss is None or not warmed or provider is None:
        return False
    if _too_long_remembered(model_id, store):
        return False
    with _cost_lock:
        _remeasured_marker[model_id] = activity_marker
    estimate = rerank_cost_estimate(model_id, store)
    if not _needs_remeasure(miss, estimate):
        _drop_miss(model_id, miss)
        return False
    k = anchor_count(len(miss.documents))
    padded = (len(miss.documents) + k) * max([*miss.candidate_tokens, *miss.anchor_tokens[:k]])
    # S180: a KEPT figure is stale (its window emptied), so the re-measure is the re-seed
    # and runs without the RAM-headroom check, as a first-ever startup seeds; a figure in
    # the window still skips a batch that does not fit.
    ram_per_token = (
        estimate.ram_bytes_per_token if estimate is not None and not estimate.ram_figure_kept else None
    )
    if ram_per_token is not None and ram_per_token > 0.0:
        headroom = _available_ram_headroom_bytes()
        growth = max(0, padded - estimate.peak_padded_tokens)
        if headroom is not None and ram_per_token * growth > headroom:
            log.info("rerank width re-measure: skipped, the batch does not fit the RAM headroom")
            return False
    _, seconds = _measured_rerank(
        provider,
        model_id,
        store,
        miss.query,
        list(miss.documents),
        miss.candidate_tokens,
        miss.anchor_tokens,
    )
    if _remeasure_hook is not None:
        _remeasure_hook()
    budget = _full_rerank_budget()
    if seconds is not None:
        _drop_miss(model_id, miss)
        if seconds > budget:
            _remember_too_long(model_id, store)
    log.info(
        "rerank width re-measure: model=%s padded_tokens=%d seconds=%s budget=%s -> %s",
        model_id,
        padded,
        seconds,
        budget,
        "5 fit" if seconds is not None and seconds <= budget else "too long, dropped",
    )
    return True


def _measured_rerank(
    provider: RerankerProvider,
    model_id: str,
    store: MemoryStore,
    query: str,
    documents: list[str],
    candidate_tokens: Sequence[int],
    anchor_tokens: Sequence[int],
) -> tuple[AnchorNormalizationResult, float | None]:
    """Rerank `documents` with anchors on top and record the call in the cost
    model (x = (n + k) x the longest of the n candidates' and k anchors' pair
    sizes). Shared by `rerank_for_recall` and the re-measure job. Returns the
    normalization and the measured seconds (`None` when the provider gave
    none: nothing recorded)."""
    normalization = normalize_against_anchors(provider, query, documents)
    if normalization.seconds is None:
        return normalization, None
    k = normalization.anchor_count
    longest = max([*candidate_tokens[: len(documents)], *anchor_tokens[:k]])
    _record_rerank_cost(
        model_id,
        store,
        (len(documents) + k) * longest,
        normalization.seconds,
        normalization.rss_delta_bytes,
    )
    return normalization, normalization.seconds


def _warm_up_once(provider: RerankerProvider, query: str, documents: Sequence[str]) -> None:
    """Before the first measured recall rerank of this process for the
    provider's model id: `_WARMUP_RERANKS` single-document reranks on the
    current candidates' first documents, results and timings discarded
    (plan P-1). Plain `rerank()`, so nothing here reaches the cost model.
    Marked done only after they succeed; two threads racing here may both
    warm up, which costs time but never skews a measurement."""
    model_id = provider.model_id()
    with _cost_lock:
        if model_id in _warmed_model_ids:
            return
    for i in range(_WARMUP_RERANKS):
        list(provider.rerank(query, [documents[i % len(documents)]]))
    with _cost_lock:
        _warmed_model_ids.add(model_id)


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
    """Cheap, per-call RAM-headroom read for the width fit's memory
    bound (`rerank_for_recall`) — a fresh read on every message, since
    this is inexpensive (a
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
    when every source is unavailable/unsupported/errors; the width fit
    treats `None` as "skip the memory term", never as "unlimited
    headroom". Also imported by `judge_selftune.py`."""
    headroom = _cgroup_v2_memory_headroom_bytes()
    if headroom is None:
        headroom = _cgroup_v1_memory_headroom_bytes()
    if headroom is None:
        headroom = _proc_meminfo_available_bytes()
    return headroom


def anchor_count(real_width: int) -> int:
    """Anchors appended ON TOP of `real_width` real candidates (S23):
    `k = min(P, real_width // ANCHOR_SPLIT_DIVISOR)`, so a rerank sends
    `real_width + k` documents. With the S5 minimum of real candidates,
    `k >= K_MIN`."""
    return min(P, real_width // ANCHOR_SPLIT_DIVISOR)


def prefix_cost(estimate: RerankCostEstimate, pairs: int, longest_pair_tokens: int) -> float:
    """Predicted seconds for one rerank batch of `pairs` documents whose
    longest (query, document) pair is `longest_pair_tokens` reranker tokens
    (S67, S75: per-call overhead + per-token rate * padded size). The one
    place the cost model's shape lives."""
    return estimate.overhead_seconds + estimate.seconds_per_token * pairs * longest_pair_tokens


def fit_rerank_width(
    candidate_pair_tokens: Sequence[int],
    anchor_pair_tokens: Sequence[int],
    estimate: RerankCostEstimate | None,
    budget_seconds: float,
    headroom_bytes: float | None,
    max_real: int = CANDIDATE_POOL,
) -> int:
    """How many real candidates to rerank this message (spec §1).

    `candidate_pair_tokens` are the candidates' surviving pair lengths (S62)
    in the caller's prefix order (genuine first, then monologue-family, each
    by cosine score, S16/S28); `anchor_pair_tokens` the same for
    `ANCHOR_POOL` (at least `P` entries). Before the first measurement
    (`estimate is None`) the width is the S5 minimum (S24). Otherwise it is
    the longest prefix n <= min(len(candidates), max_real) such that the
    batch of n real + `anchor_count(n)` anchors fits the budget
    (`prefix_cost`, padded to its longest pair, anchors included in both
    factors) and, when a RAM figure and a headroom reading exist, the RSS
    growth it is predicted to need (RAM per token times its padded size
    past the high water, see the section header) fits the headroom (S32,
    S65: no chunking).
    Both factors of the padded size only grow with n, so the scan stops at
    the first prefix that does not fit.

    The result may be below the S5 minimum; the caller then does not rerank
    (S5/S23). Pure: no clock, no I/O."""
    limit = min(len(candidate_pair_tokens), max_real)
    if estimate is None:
        return min(RERANK_MIN_REAL_CANDIDATES, limit)
    ram_per_token = estimate.ram_bytes_per_token
    ram_bound = ram_per_token is not None and ram_per_token > 0.0 and headroom_bytes is not None
    width = 0
    longest_real = 0
    for n in range(1, limit + 1):
        longest_real = max(longest_real, candidate_pair_tokens[n - 1])
        k = anchor_count(n)
        longest = max([longest_real, *anchor_pair_tokens[:k]])
        pairs = n + k
        if prefix_cost(estimate, pairs, longest) > budget_seconds:
            break
        growth = max(0, pairs * longest - estimate.peak_padded_tokens)
        if ram_bound and ram_per_token * growth > headroom_bytes:
            break
        width = n
    return width


@dataclass(frozen=True)
class RecallRerank:
    """What `rerank_for_recall` did for one query.

    `width` — real candidates the fit allowed (the first `width` of the
    caller's list). `reranked` — True iff the rerank ran and its scores are
    anchor-normalized; `normalization` then holds them (`scores[i]` belongs
    to the caller's `i`-th candidate). When `reranked` is False the caller
    must NOT gate any score (`normalization` is then `None`): `hand_off`
    says why the no-rerank path takes over — "pool" (fewer than the S5
    minimum of candidates exist), "budget" (fewer than the minimum fit the latency budget or the RAM
    bound), "sizes" (the pairs' token lengths could not be read, so no width can be sized), or
    "normalization" (the anchor median could not be taken).
    `measured` — whether this call added a sample to the cost model."""

    width: int
    reranked: bool
    hand_off: str | None
    normalization: AnchorNormalizationResult | None
    measured: bool


@dataclass(frozen=True)
class PairSizes:
    """One query's pair token counts, computed ahead of its rerank
    (`recall_pair_sizes`) so a caller can time them apart from the rerank
    itself (name-recall fix R6, plan P-28: the pair-length computation is
    part of the per-paragraph time `T_p`, not of the rerank budget).

    `lengths` is `provider.pair_token_lengths(query, [*candidates, *ANCHOR_POOL])`
    over the query's first `CANDIDATE_POOL` candidates, or `None` when the
    tokenizer probe failed (the "sizes" hand-off). `seconds` is the probe's
    time measured inside the provider's lock (round-3 N3-1), or `None` when
    the provider measures none (the caller then times the call)."""

    lengths: list[int] | None
    seconds: float | None = None


def warm_up_for_recall(
    provider: RerankerProvider, query: str, candidate_documents: Sequence[str]
) -> None:
    """The two discarded warm-up reranks (plan P-1), run now if this
    process has not run them yet for the provider's model id. A caller that
    times its own pair-size probe (`recall_pair_sizes`) calls this first, so
    the one-time ONNX session load is never timed into `T_p`."""
    _warm_up_once(provider, query, list(candidate_documents[:CANDIDATE_POOL]))


def recall_pair_sizes(
    provider: RerankerProvider, query: str, candidate_documents: Sequence[str]
) -> PairSizes:
    """The pair token counts `rerank_for_recall` would compute for this
    query, computed now (see `PairSizes`). Warms up first (the real
    provider's tokenizer exists only once its model is loaded). Raises
    whatever the provider raises."""
    pool = list(candidate_documents[:CANDIDATE_POOL])
    _warm_up_once(provider, query, pool)
    lengths, seconds = provider.pair_token_lengths_timed(query, [*pool, *ANCHOR_POOL])
    return PairSizes(lengths=None if lengths is None else list(lengths), seconds=seconds)


def rerank_for_recall(
    provider: RerankerProvider,
    query: str,
    candidate_documents: Sequence[str],
    *,
    store: MemoryStore,
    budget_seconds: float | None = None,
    max_real: int = CANDIDATE_POOL,
    sizes: PairSizes | None = None,
) -> RecallRerank:
    """Fit this query's width, rerank that prefix with anchors on top, and
    record the call in the cost model (spec §1, S23/S24/S32/S62/S67/S75).

    `candidate_documents` are the query's candidates in prefix order (see
    `fit_rerank_width`); only the first `CANDIDATE_POOL` are considered.
    `store` is the caller's memory store: the saved time series is read from
    it and the measured call is buffered against it (S146, S164).
    `budget_seconds` defaults to the `reranker.latency_budget_seconds`
    tunable read now; `max_real` lets a caller give this query less than the
    design maximum (name-recall fix R6, S58: what is left of the message's
    50 real candidates; below the S5 minimum the hand-off is "allocation").
    `sizes`, when given, are this query's pair token counts computed ahead
    by `recall_pair_sizes` over the same documents (R6 times them into
    `T_p`); otherwise they are computed here. Raises whatever the provider
    raises (warm-up, lengths, scoring); callers treat that as a reranker
    failure."""
    pool = list(candidate_documents[:CANDIDATE_POOL])
    cap = min(max_real, CANDIDATE_POOL)
    if min(len(pool), cap) < RERANK_MIN_REAL_CANDIDATES:
        return RecallRerank(
            width=min(len(pool), max(cap, 0)),
            reranked=False,
            hand_off="pool" if len(pool) < RERANK_MIN_REAL_CANDIDATES else "allocation",
            normalization=None,
            measured=False,
        )

    if budget_seconds is None:
        budget_seconds = tunables.get_tunable(
            "reranker.latency_budget_seconds", LATENCY_BUDGET_SECONDS
        )
    model_id = provider.model_id()
    # Warm up before the length probe: on the real provider the first rerank
    # loads the ONNX session, and with it the tokenizer the probe uses.
    _warm_up_once(provider, query, pool)
    if sizes is None:
        lengths = provider.pair_token_lengths(query, [*pool, *ANCHOR_POOL])
    else:
        lengths = sizes.lengths
        if lengths is not None and len(lengths) != len(pool) + len(ANCHOR_POOL):
            raise ValueError(
                f"rerank_for_recall: {len(lengths)} precomputed pair sizes for "
                f"{len(pool)} candidates + {len(ANCHOR_POOL)} anchors"
            )
    if lengths is None:
        # Token sizes unavailable (the tokenizer probe failed): a width cannot
        # be sized against the latency budget or the RAM bound, and a batch of
        # unknown padded size must not run unbudgeted, so this query is not
        # reranked (the no-rerank path serves it). No sample is recorded either:
        # the cost model holds tokens only (S75).
        return RecallRerank(
            width=0, reranked=False, hand_off="sizes", normalization=None, measured=False
        )
    candidate_tokens, anchor_tokens = lengths[: len(pool)], lengths[len(pool) :]
    estimate = rerank_cost_estimate(model_id, store)
    if estimate is not None and _ram_unseeded(model_id, estimate):
        # S158/S171: a saved time estimate without a saved RAM figure would size
        # this process's first measured rerank past the S5 minimum, but that
        # rerank measures the RAM growth the width fit's RAM bound needs; it runs
        # at 5 (where RSS is readable).
        estimate = None
    headroom = _available_ram_headroom_bytes() if estimate is not None else None
    width = fit_rerank_width(candidate_tokens, anchor_tokens, estimate, budget_seconds, headroom, cap)

    if width < RERANK_MIN_REAL_CANDIDATES:
        # The MEASURED estimate says fewer than the S5 minimum fit this
        # query's budget, so it is not reranked and adds no measurement. It is
        # kept as the model id's last budget miss (S113); while the estimate
        # says its first 5 do not fit the FULL budget, the re-measure job
        # re-times them at a chat lull (S100), so an inflated estimate cannot
        # absorb every later message. (The RAM term alone lands here only for
        # batches larger than any measured so far; if the figure doing it is a
        # running process's kept one, the same job re-seeds it, S180.)
        miss = _BudgetMiss(
            query=query,
            documents=tuple(pool[:RERANK_MIN_REAL_CANDIDATES]),
            candidate_tokens=tuple(candidate_tokens[:RERANK_MIN_REAL_CANDIDATES]),
            anchor_tokens=tuple(anchor_tokens),
        )
        with _cost_lock:
            _budget_misses[model_id] = miss
        return RecallRerank(
            width=width, reranked=False, hand_off="budget", normalization=None, measured=False
        )

    normalization, seconds = _measured_rerank(
        provider, model_id, store, query, pool[:width], candidate_tokens, anchor_tokens
    )
    measured = seconds is not None
    if measured and seconds <= _full_rerank_budget():
        # S116 (a): a rerank of 5 or more real candidates ran within the full
        # budget, which proves 5 fit: the model id is not stuck.
        with _cost_lock:
            _budget_misses.pop(model_id, None)
    if not normalization.did_normalize:
        # Raw scores: never handed to a caller that could gate them.
        return RecallRerank(
            width=width,
            reranked=False,
            hand_off="normalization",
            normalization=None,
            measured=measured,
        )
    return RecallRerank(
        width=width, reranked=True, hand_off=None, normalization=normalization, measured=measured
    )


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
# `k = min(P, real_width // 2)` (`anchor_count`) — the anchor count is
# derived from the hardware-fitted width, never a hand-picked constant [OWNER
# 2026-09-22 "that, is a magic number constant. No. Have it scale based on
# the hardware capability like everything else"]. Name-recall fix R1 (S23):
# anchors now come ON TOP of the real candidates (they were reserved out of
# the width until R1): width counts real candidates, a rerank sends
# `real_width + k` documents, and the anchors' cost is inside the width fit
# (`fit_rerank_width` prices the whole batch, anchors included), so the
# per-turn rerank stays inside the latency budget (I6).

# The meaningful-median floor: below 2 anchor scores, "median" degenerates
# (a single value, or an arbitrary pick between two with no robust middle)
# and offers no protection against one anomalous anchor swinging the whole
# correction — the same robust-statistics reasoning the median choice above
# rests on. SIZES the anchor mechanism (I3-clean); it is NOT a relevance
# threshold — no
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
    """Result of `normalize_against_anchors`.

    `scores` — one float per real candidate, positionally aligned with the
    caller's `real_documents` (all of them are scored). Median-normalized
    (`raw - median(anchor_scores)`) when `did_normalize` is True; raw,
    unmodified reranker scores when False (the defensive no-op below), which
    a caller must never gate against the normalized floor.

    `real_width` — `len(real_documents)` (`== len(scores)`).

    `did_normalize` — False only when `anchor_count(real_width) < K_MIN`
    (fewer than 4 real documents): no anchors were appended, `scores` are
    RAW. Recall callers never get here (they rerank at least the S5 minimum
    of 5 real candidates, so k >= 2).

    `anchor_count` — anchors appended on top (`0` on the no-op).
    `seconds` / `rss_delta_bytes` — the scoring call's measurements from
    `RerankerProvider.rerank_timed` (feed the width fit's cost model).
    """

    scores: list[float]
    real_width: int
    did_normalize: bool
    anchor_count: int = 0
    seconds: float | None = None
    rss_delta_bytes: float | None = None


def normalize_against_anchors(
    provider: RerankerProvider,
    query: str,
    real_documents: Sequence[str],
) -> AnchorNormalizationResult:
    """Per-query anchor-median normalization (F2b, #276), anchors ON TOP
    (name-recall fix R1, S23) — see the section header above.

    Scores EVERY document in `real_documents` (the caller passes exactly the
    prefix it fitted, e.g. `rerank_for_recall`'s `width`), plus
    `k = anchor_count(len(real_documents))` anchors (`ANCHOR_POOL[:k]`)
    appended after them, in ONE combined `rerank_timed` call; splits the
    scores positionally back into real vs anchor and returns
    `raw_real_scores - median(anchor_scores)`. Computed FRESH on every call:
    the anchor offset is a per-QUERY property, so caching it across calls
    would defeat the point of the correction.

    Defensive no-op (raw scores, no anchors appended, `did_normalize=False`)
    when `k < K_MIN`: too few real documents for a meaningful anchor median.

    Anchor documents/scores are used ONLY to compute the median offset and
    are NEVER returned — callers must not treat them as candidates.
    """
    real = list(real_documents)
    real_width = len(real)
    k = anchor_count(real_width)
    if k < K_MIN:
        scores, seconds, rss_delta = provider.rerank_timed(query, real)
        return AnchorNormalizationResult(
            scores=scores,
            real_width=real_width,
            did_normalize=False,
            seconds=seconds,
            rss_delta_bytes=rss_delta,
        )

    raw_scores, seconds, rss_delta = provider.rerank_timed(query, real + ANCHOR_POOL[:k])
    real_scores = raw_scores[:real_width]
    anchor_scores = raw_scores[real_width:]
    normalized = _median_normalize(real_scores, anchor_scores)
    return AnchorNormalizationResult(
        scores=normalized,
        real_width=real_width,
        did_normalize=True,
        anchor_count=k,
        seconds=seconds,
        rss_delta_bytes=rss_delta,
    )


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
