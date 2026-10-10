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
least squares over running sums) and a memory model over call shapes
(``a + b*T + c*B*L^2`` of the peak growth of the same calls, ``ram_fit``); each
message then fits its own width to the lengths of its own candidates within
the latency budget (``LATENCY_BUDGET_SECONDS``, the one ops tunable) and the
measured RAM bound. See the "Per-message rerank width"
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

import contextlib
import logging
import math
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
from brain.memory import cost_series, peak_rss, ram_fit
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


class RerankTerminated(Exception):  # noqa: N818 - a control signal, not an error
    """S223: a scoring call was stopped on purpose (the stop returned by `arm_terminate` was
    called). Raised only to a thread that armed a stop; the RAM ladder maps it to
    `RamMeasurementAbandoned`."""


def _noop() -> None:
    return None


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

    def arm_terminate(self) -> Callable[[], None]:
        """S223: prepare to stop the NEXT scoring call(s) made by THIS thread: returns a `stop()`
        (callable from any thread) that makes that call end early with `RerankTerminated`, until
        `disarm_terminate()`. The default provider cannot stop a call: it returns a no-op, and the
        running call simply finishes (S218: only its sample is discarded)."""
        return _noop

    def disarm_terminate(self) -> None:
        """S223: end this thread's armed scope (see `arm_terminate`). Default: nothing to do."""
        return None

    def rerank_timed(
        self, query: str, documents: list[str]
    ) -> tuple[list[float], float, float | None]:
        """`rerank()` plus the measurements the per-message width fit learns
        from (name-recall fix R1, plan P-3): `(scores, seconds, peak_growth_bytes)`.

        `seconds` is the wall time of the scoring call, read from the module
        clock seam `_clock`; the third value is the call's PEAK resident-memory growth
        in bytes (S220, S222: `peak_rss`; `None` when it cannot be read, e.g. off Linux).
        With the ONNX memory pool off the call returns its memory, so the RSS difference
        would read about 0. Only RECALL paths call
        this (passive recall and `search_memories`, via
        `normalize_against_anchors`); bootstrap-floor and calibration scoring
        call plain `rerank()` and so never feed the cost model (S24).

        `CrossEncoderProvider` overrides this to take both readings INSIDE its
        instance lock, so a concurrent recall waiting on the lock never
        inflates the measured time."""
        before = peak_rss.begin()
        start = _clock()
        scores = list(self.rerank(query, documents))
        seconds = _clock() - start
        return scores, seconds, peak_rss.growth_since(before)

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

    _session_checked = False  # S222: the live session's options are read back once (below)
    # S223 (class defaults: tests build providers without `__init__`): see `__init__`.
    _want_terminable = False
    _run_wrapped = False
    _wrapper_calls = 0
    _drift_checked = False
    _install_failed = False

    def __init__(self, model_id: str, cache_dir: str | Path) -> None:
        # Imported lazily (module-scoped, not top-level) so importing this
        # module never requires fastembed/onnxruntime to be installed unless
        # the real provider is actually constructed (tests exclusively use
        # FakeRerankerProvider). Confirmed path for fastembed 0.8.0: the
        # cross-encoder lives under `fastembed.rerank.cross_encoder`, NOT at
        # the top-level `fastembed` namespace.
        from fastembed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder

        self._model_id = model_id
        # S222: the ONNX CPU memory pool and the memory-pattern planner are OFF, so the
        # memory a call needed goes back after the call (the per-call growth is then what
        # `ram_fit` models). fastembed exposes only the pool option; the pattern option is
        # added through the options hook below.
        self._model = TextCrossEncoder(
            model_name=model_id,
            cache_dir=str(cache_dir),
            lazy_load=True,
            enable_cpu_mem_arena=False,
        )
        _install_memory_pattern_off(self._model)
        # Same lazy-load check-then-act race FastEmbedProvider.__init__
        # documents (fastembed's OnnxTextModel first-load path has no lock of
        # its own); a shared instance of this provider (the process-wide
        # cache below) can have .rerank() called concurrently from two
        # threads — the turn thread (passive/active recall) and, in
        # principle, a future background caller. Serialize the whole
        # rerank() call (construction + inference) behind one instance lock,
        # matching FastEmbedProvider's conservative choice.
        self._rerank_lock = threading.Lock()
        # S223: stoppable calls. `_local.scope` is the RunOptions armed by the CALLING thread (read
        # and written only by that thread); `_want_terminable` is set by the first arm and makes
        # the next scored call wrap the session's `run` (a provider nobody ever armed, such as the
        # bridge's, never touches the session); `_run_wrapped`, `_wrapper_calls` and
        # `_drift_checked` are read and written under `_rerank_lock`.
        self._local = threading.local()

    def arm_terminate(self) -> Callable[[], None]:
        """S223: see `RerankerProvider.arm_terminate`. Makes a fresh ONNX Runtime `RunOptions` for
        this thread; the returned `stop()` sets its `terminate` flag, which ORT polls while the
        call runs (a probe on this host: the production model's call stopped 118-216 ms after
        the flag, against 6-9 s for the unstopped call). The options belong to THIS thread's call
        only, so a stop can never reach another thread's call (the floor-bootstrap thread, a
        reply's own recall rerank). A stop before the call starts makes the call start terminated.
        `onnxruntime` unavailable: a no-op stop (the S218 fallback)."""
        if self._install_failed:
            return _noop  # the session cannot be wrapped (warned once): S218 fallback
        try:
            import onnxruntime as ort

            options = ort.RunOptions()
        except Exception:  # noqa: BLE001
            log.warning("reranker: cannot make runs stoppable (onnxruntime RunOptions)", exc_info=True)
            return _noop
        self._local.scope = options
        self._want_terminable = True

        def stop() -> None:
            options.terminate = True

        return stop

    def disarm_terminate(self) -> None:
        """S223: end this thread's armed scope."""
        self._local.scope = None

    def _install_terminable_run(self) -> None:
        """S223: once, before the first scored call after an arm, shadow the loaded session's
        `run` with a wrapper that hands the calling thread's armed `RunOptions` to ONNX Runtime
        (fastembed's own call passes none). Caller holds `_rerank_lock`. Without an armed scope on
        the calling thread (or with options supplied by the caller) the wrapper forwards its
        arguments untouched, so every other caller sees the original behaviour. A failure to
        install is warned once and leaves the S218 fallback (the call finishes, its sample is
        discarded)."""
        if self._run_wrapped or not self._want_terminable:
            return
        self._run_wrapped = True
        try:
            inner = self._model.model
            if getattr(inner, "model", None) is None:
                inner.load_onnx_model()
            session = inner.model
            original = session.run
            local = self._local

            def terminable_run(output_names: Any, input_feed: Any, run_options: Any = None) -> Any:
                self._wrapper_calls += 1
                options = getattr(local, "scope", None)
                if options is None or run_options is not None:
                    return original(output_names, input_feed, run_options)
                try:
                    return original(output_names, input_feed, options)
                except Exception as exc:
                    if options.terminate:
                        raise RerankTerminated("the scoring call was stopped on purpose") from exc
                    raise

            session.run = terminable_run
        except Exception:  # noqa: BLE001 - never break a recall because the stop cannot be installed
            log.warning(
                "reranker: could not make the scoring session stoppable; a reply that starts "
                "during the RAM measurement waits for its call (S223 fallback to S218)",
                exc_info=True,
            )
            self._run_wrapped = False
            self._want_terminable = False
            self._install_failed = True  # never retried, never warned again

    def _check_wrapper_reached_once(self, calls_before: int, documents: Sequence[str]) -> None:
        """S223 drift check: after the first scored call (at least one document) through the
        wrapper, warn once if the wrapper was never reached (fastembed no longer calls
        `session.run`, so a stop would do nothing). Caller holds `_rerank_lock`. Never raises."""
        if not self._run_wrapped or self._drift_checked or not documents:
            return
        self._drift_checked = True
        if self._wrapper_calls == calls_before:
            log.warning(
                "reranker: the scoring call did not go through the session's run; the RAM "
                "measurement cannot be stopped when a reply starts (S223 fallback to S218)"
            )

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        with self._rerank_lock:
            self._install_terminable_run()
            calls_before = self._wrapper_calls
            scores = list(self._model.rerank(query, documents))
            self._check_wrapper_reached_once(calls_before, documents)
            self._check_session_options_once()
            return scores

    def _check_session_options_once(self) -> None:
        """S222: after the first call (the session exists), read back the live session's
        options and warn if the memory pool or the memory patterns are still on (a
        fastembed release that stopped honouring the hook would otherwise go unnoticed).
        Caller holds `_rerank_lock`. Never raises."""
        if self._session_checked:
            return
        self._session_checked = True
        try:
            options = self._model.model.model.get_session_options()
            if options.enable_cpu_mem_arena or options.enable_mem_pattern:
                log.warning(
                    "reranker: the ONNX session still has the memory pool (%s) or memory patterns (%s) on; "
                    "the RAM model assumes both off (S222)",
                    options.enable_cpu_mem_arena,
                    options.enable_mem_pattern,
                )
        except Exception:  # noqa: BLE001 - a diagnostic must never break a recall
            log.debug("reranker: could not read back the ONNX session options", exc_info=True)

    def rerank_timed(
        self, query: str, documents: list[str]
    ) -> tuple[list[float], float, float | None]:
        """See `RerankerProvider.rerank_timed`. Both readings are taken while
        holding `_rerank_lock`, so time spent WAITING for another thread's
        rerank is never counted as this call's cost (plan P-3)."""
        with self._rerank_lock:
            self._install_terminable_run()  # before the clock: the install may load the session
            calls_before = self._wrapper_calls
            before = peak_rss.begin()
            start = _clock()
            scores = list(self._model.rerank(query, documents))
            seconds = _clock() - start
            growth = peak_rss.growth_since(before)
            self._check_wrapper_reached_once(calls_before, documents)
            self._check_session_options_once()
        return scores, seconds, growth

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


def _install_memory_pattern_off(text_cross_encoder: Any) -> None:
    """S222: make the fastembed cross-encoder's session options also turn the memory-pattern
    planner off. fastembed (0.8) exposes only `enable_cpu_mem_arena`; its `_load_onnx_model`
    calls `self.add_extra_session_options(options, extra)` on the inner model, so an instance
    attribute shadows the classmethod: it applies fastembed's own handling, then sets
    `enable_mem_pattern = False`. A failure to install is warned once and never breaks
    construction (the pool is still off; `_check_session_options_once` reports the live state)."""
    try:
        inner = text_cross_encoder.model
        original = inner.add_extra_session_options

        def _with_pattern_off(session_options: Any, extra_options: dict[str, Any]) -> None:
            original(session_options, extra_options)
            session_options.enable_mem_pattern = False

        inner.add_extra_session_options = _with_pattern_off
    except Exception:  # noqa: BLE001
        log.warning("reranker: could not turn the ONNX memory patterns off (S222)", exc_info=True)


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
#   - MEMORY growth by call shape (S32, S220): a call of `pairs` (B) documents padded to
#     the longest pair `L` needs about a + b*T + c*B*L^2 bytes of peak extra memory
#     (T = B*L), fitted per machine by `ram_fit` over every measured call of the window and
#     checked against the live headroom. With the ONNX memory pool and memory patterns off
#     (S222) the call gives its memory back, so each call's own PEAK growth is the sample
#     (`peak_rss`); there is no high water and nothing is "already resident".
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
# The time series and the memory samples are rolling 24-hour windows saved in
# memories.db (S146, S171; see `cost_series`); the warm-up state and the kept fit stay process
# memory. The first measured recall rerank of a process per model id waits for a usable memory
# model (S200: measured at a lull by the ladder below; a window with too few shapes or no
# full-length sample is no model), where the per-call peak is readable (Linux; elsewhere no
# RAM bound exists, S206 posture). A running process keeps the last usable fit it computed when
# the window stops giving one (a quiet day; H4b: it never reranks without a RAM bound) and has
# the next lull refresh it.
# Stuck-width recovery (S100, S113, S116): the last query that went to the cosine path for lack
# of budget is kept per model id; while the estimate says its first 5 candidates do not fit the
# full latency budget (time term), the `rerank_width_remeasure` gated job re-measures them once
# per chat lull (`stuck_width_due`, `remeasure_stuck_width`), behind the same memory-model gate
# as every rerank.

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


@dataclass(frozen=True)
class RerankCostEstimate:
    """The fitted cost model for one reranker model id (see the section
    header). `ram_fit` is the memory model over call shapes (S220): the window's usable fit, or
    this process's last usable one (`ram_fit_kept`, H4b: a running process never reranks without
    a RAM bound), or `None` (the RAM term of the width fit is then skipped, and so is the
    re-measure job's RAM-headroom check). `measured_batches` = the measurements behind the time
    fit in effect."""

    overhead_seconds: float
    seconds_per_token: float
    ram_fit: ram_fit.RamFit | None
    measured_batches: int
    ram_fit_kept: bool = False


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


# The memory samples are the saved series `_ram_shapes`; process memory holds the model ids
# whose warm-up has run, the last query per
# model id that went to the cosine path for lack of budget (S113), the chat
# activity marker of each model id's last re-measure (S100), and the "too long"
# verdicts (S127: remembered until the cost estimate next changes, decided by
# which samples the window holds, not by comparing fitted floats; saved with the
# host-keyed series and read back once per process, S179), and the kept memory fits
# (`_ram_fit_kept`, H4b). One lock
# guards them all (every read and every read-modify-write), so a concurrent
# passive recall, `search_memories` call (#297: the tool runs in a bridge
# worker thread) and the re-measure job never lose an update. The time series
# itself is `_rerank_cost` (rolling window, memories.db, `cost_series`).
_rerank_cost = cost_series.CostSeries("rerank")
# S220: the memory growth of every measured call by shape (pairs, padded length), saved like the
# time series (host-keyed, 24 h window; `cost_series.RAM_SHAPE_SERIES`). The fit over its window is
# `ram_fit.fit_ram`; memoised below per (database, model id) so one reply does not refit it at the
# gate, the width fit and the estimate.
_ram_shapes = cost_series.CostSeries(cost_series.RAM_SHAPE_SERIES)
# H4b (S180/S204 in their new form): the last usable fit this running process computed, per
# (database, model id). It serves the gate and the width fit when the window no longer gives one
# (the full-length sample aged out after a quiet day). Process memory, never saved; guarded by
# `_cost_lock` with the memo.
_ram_fit_kept: dict[tuple[str, str], ram_fit.RamFit] = {}
_ram_fit_memo: dict[
    tuple[str, str], tuple[tuple[ram_fit.ShapeGroup, ...], bool, ram_fit.RamFit | None]
] = {}
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

def _window_ram_fit(store: MemoryStore, model_id: str) -> ram_fit.RamFit | None:
    """The usable fit over the shape series' window (S220), or `None` (not enough distinct
    shapes, no full-length sample, no finished ladder pass, or the database read failed). A
    window whose ladder pass did not finish (no completion marker) gives no fit: its single
    pairs would extrapolate batch sizes (under-predicting them by 17 to 100 percent). One
    aggregate read per call (a row per distinct shape); the fit is memoised on the exact groups,
    so repeated calls within a reply cost one read and a tuple comparison. A usable fit is also
    remembered as the process's kept fit (H4b)."""
    rows = _ram_shapes.shape_groups(store, model_id)
    if rows is None:
        return None
    groups = tuple(ram_fit.groups_from_rows(rows))
    complete = ram_fit.ladder_complete(rows)
    key = (cost_series._store_key(store), model_id)
    with _cost_lock:
        memo = _ram_fit_memo.get(key)
    if memo is not None and memo[0] == groups and memo[1] == complete:
        return memo[2]
    fit = ram_fit.fit_ram(groups) if complete else None
    with _cost_lock:
        _ram_fit_memo[key] = (groups, complete, fit)
        if fit is not None:
            _ram_fit_kept[key] = fit
    return fit


def _ram_fit_for(store: MemoryStore, model_id: str) -> tuple[ram_fit.RamFit | None, bool]:
    """`(fit, kept)`: the window's usable fit, else this running process's last usable one
    (`kept` True), else `(None, False)`. The window is read BEFORE the kept fit, and a fit the
    window gives is stored as the kept one before it is returned, so a reader never sees the
    window answer and then no kept fit."""
    fit = _window_ram_fit(store, model_id)
    if fit is not None:
        return fit, False
    with _cost_lock:
        kept = _ram_fit_kept.get((cost_series._store_key(store), model_id))
    return (kept, True) if kept is not None else (None, False)


def rerank_cost_estimate(model_id: str, store: MemoryStore) -> RerankCostEstimate | None:
    """The current cost model for `model_id`: the time fit of the last 24
    hours (S146; saved, so available from the first message after a restart)
    and the memory model over call shapes (S220: the window's usable fit, else this process's
    last one). `None` when there is no time estimate (the first startup on this machine: the
    width is then the S5 minimum, S24). A process with neither a window fit nor a kept fit has
    `ram_fit = None` and takes the cosine path (S200: `search_paragraphs` and
    `rerank_for_recall` check `ram_figure_ready` first)."""
    fit = _rerank_cost.fit(store, model_id)
    if fit is None:
        return None
    ram, kept = _ram_fit_for(store, model_id)
    return RerankCostEstimate(
        overhead_seconds=fit.overhead,
        seconds_per_token=fit.rate,
        ram_fit=ram,
        measured_batches=fit.samples,
        ram_fit_kept=kept,
    )


def _record_rerank_cost(
    model_id: str,
    store: MemoryStore,
    padded_tokens: int,
    seconds: float,
    peak_growth_bytes: float | None,
    *,
    pairs: int | None = None,
    longest: int | None = None,
    full_length: bool = False,
) -> None:
    """Add one measured rerank: its time and, when its peak growth could be read, its memory
    growth by shape `(pairs, longest)` to the buffered series (S164: nothing is written to the
    database here). Every call is a sample (S220: no high water; with the memory pool off the
    call's growth is its own). `full_length` marks the ladder's full-input-length pair (H4)."""
    if peak_growth_bytes is not None and pairs is not None and longest is not None:
        x = ram_fit.encode_shape(pairs, longest, full=full_length)
        if x is None:
            _warn_ram_figure_once(f"a rerank of {pairs} pairs x {longest} tokens cannot be recorded in the RAM series")
        else:
            _ram_shapes.record(store, model_id, x, max(0.0, peak_growth_bytes))
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


# ---------------------------------------------------------------------------
# S200, S208, S220: no re-scoring until the machine's memory model exists (fixes the S193
# first-rerank regression). On a machine with no usable memory model a recall turn takes the
# plain meaning-match (cosine) path and only FLAGS the need; the model is then measured off the
# reply path (the `rerank_ram_figure` gated job at a chat lull in the bridge, a background thread
# after the reply under `nell chat --no-bridge`) by a LADDER of single measured calls, in
# S208's order:
#   phase 1, lengths: one pair at the model's full input length (the worst case per pair, marked
#     as such), then one pair at half the shortest length measured, and so on, until the window
#     holds enough distinct shapes (`ram_fit.enough_shapes`: more shapes than the formula has
#     parameters, full rank, and the full-length sample);
#   phase 2, batches of increasing size, never starting with a big one: 2, 4, 8, ... pairs up to
#     the most a recall sends, each at the longest length the time budget allows for that many
#     pairs (every real call fits the time budget, so each real call is dominated by one of these
#     measured shapes), each only when the fit says it fits the live headroom.
# The window is read to decide every rung, so a restart or an abandoned pass resumes where the
# window stands. Real reranks add their own shapes. The ONNX memory pool and memory patterns are
# off (S222), so a call's memory goes back after it and the sample is the call's PEAK growth
# (`peak_rss`); `fit_rerank_width` sizes a batch so its predicted growth fits the live headroom.
# S203/S212/S220 retired the width-5 seeds, the high water, the kept per-token figure and the
# S182 re-seed; a running process keeps its last usable fit instead (H4b).
# ---------------------------------------------------------------------------

# Test seam: tests/conftest.py sets this False so the suite's recall tests keep
# reranking without a seeded model; the S200 tests set it True. Production: True.
_require_ram_figure = True
# Reranker model ids a turn found without a RAM model (the job is due only for
# these) and the chat-activity marker of each id's last attempt (once per lull).
_ram_figure_needed: set[str] = set()
_ram_figure_attempt: dict[str, object] = {}

# The full-length pair is built from this sentence repeated. The text starts at
# `_PAIR_TEXT_START_CHARS` and doubles up to `_PAIR_TEXT_DOUBLINGS` times until
# the pair's surviving token count stops growing (the model's input limit
# truncates it). Any plain-English text of a few thousand characters passes a
# 1,024-token limit; the doubling is only the search for the limit, not a size.
_PAIR_SENTENCE = "The quick brown fox jumps over the lazy dog while the river runs past the old stone bridge. "
_PAIR_TEXT_START_CHARS = 4096
_PAIR_TEXT_DOUBLINGS = 8


_ram_figure_warned: set[str] = set()


class RamMeasurementAbandoned(Exception):  # noqa: N818 - a control signal, not an error
    """S218: a chat reply started while a ladder rung ran (`nell chat --no-bridge`), so that
    rung is abandoned and nothing from it is recorded (earlier rungs' samples stay). Raised only
    when the caller passed a `watch`; the bridge's lull job never does."""


def _warn_ram_figure_once(reason: str, *, exc_info: bool = False) -> None:
    """A RAM-model refusal (or an unreadable memory probe) can recur at every lull or
    every message: warn once per reason per process, with the traceback when `exc_info`
    is set (a probe failure), then debug."""
    first = reason not in _ram_figure_warned
    _ram_figure_warned.add(reason)
    (log.warning if first else log.debug)("reranker: %s", reason, exc_info=exc_info and first)


def _ram_bound_applies() -> bool:
    """Whether a RAM bound exists on this machine at all: the memory model is required (the
    test seam) and the per-call peak can be read (`peak_rss`: Linux). Where it cannot be read
    nothing holds a turn back and the width is time-bound only (S206 posture); an unreadable
    measurement is never read as unlimited memory."""
    return _require_ram_figure and peak_rss.available()


def _window_ready(store: MemoryStore, model_id: str) -> bool:
    """The WINDOW gives a usable memory model (or no RAM bound applies here): what the
    measuring job needs to know. Cheap: one aggregate read."""
    if not _ram_bound_applies():
        return True
    return _window_ram_fit(store, model_id) is not None


def ram_figure_ready(model_id: str, store: MemoryStore) -> bool:
    """S200, S220: whether a rerank may run for `model_id`. True when no RAM bound applies
    (the per-call peak is unreadable here: no model can be measured, so the turn is not held
    back and the width is time-bound only), when the 24-hour window gives a usable memory
    model, or when this running process still holds the last one it computed (H4b: it never
    reranks without a RAM bound; the need is then flagged so the next lull refreshes the
    window). A failed database read is no window fit (cosine path unless a fit is kept).
    Cheap: one aggregate read, no model."""
    if not _ram_bound_applies():
        return True
    fit, kept = _ram_fit_for(store, model_id)
    if fit is None:
        return False
    if kept:
        note_ram_figure_needed(model_id)
    return True


def ram_figure_refresh_wanted(model_id: str) -> bool:
    """Whether a refresh of the memory model is flagged for `model_id` (a turn found none, or
    served only a kept one): the direct chat starts its background ladder for it."""
    with _cost_lock:
        return model_id in _ram_figure_needed


def note_ram_figure_needed(model_id: str) -> None:
    """S200: a recall turn that would rerank found no usable RAM model (or only a kept one);
    flag it so the next lull's `rerank_ram_figure` job measures one. Never loads anything."""
    with _cost_lock:
        _ram_figure_needed.add(model_id)


def ram_figure_due(
    activity_marker: object, store_factory: Callable[[], AbstractContextManager[MemoryStore]]
) -> bool:
    """Whether the `rerank_ram_figure` job is due (S200): a turn flagged the need
    for the runtime reranker model id, the window gives no usable model yet, and no attempt has
    run at this chat activity marker (once per lull; no time constant). A flag whose model has
    since appeared is dropped here. In-memory checks first; the store is opened only past them."""
    model_id = resolve_reranker_model_id()
    with _cost_lock:
        if model_id not in _ram_figure_needed:
            return False
        stamp = _ram_figure_attempt.get(model_id, _NEVER)
    if stamp is not _NEVER and stamp == activity_marker:
        return False
    with store_factory() as store:
        if _window_ready(store, model_id):
            with _cost_lock:
                _ram_figure_needed.discard(model_id)
            return False
    return True


def _full_length_pair(provider: RerankerProvider) -> tuple[str, int] | None:
    """A text whose (text, text) pair has the model's full input length, and that
    length in tokens; `None` when the provider cannot count tokens or the count was
    still growing at the last doubling (the pair would not be full length)."""
    previous = 0
    chars = _PAIR_TEXT_START_CHARS
    for _ in range(_PAIR_TEXT_DOUBLINGS + 1):
        text = (_PAIR_SENTENCE * (chars // len(_PAIR_SENTENCE) + 1))[:chars]
        lengths = provider.pair_token_lengths(text, [text])
        if lengths is None or len(lengths) != 1:
            return None
        if lengths[0] <= previous:
            return text, lengths[0]
        previous = lengths[0]
        chars *= 2
    return None


def _text_of_length(provider: RerankerProvider, base: str, target: int) -> tuple[str, int] | None:
    """The longest prefix of `base` whose (text, text) pair measures at most `target` tokens,
    and that measured length (at least 1 token). Found by bisecting characters against the provider's own
    tokenizer: the tokenizer truncates at the model limit, so a length cut in proportion to
    characters is wrong near it. `None` when no prefix fits or the tokenizer cannot count."""
    best: tuple[int, int] | None = None
    low, high = 1, len(base)
    while low <= high:
        mid = (low + high) // 2
        text = base[:mid]
        lengths = provider.pair_token_lengths(text, [text])
        if lengths is None or len(lengths) != 1:
            return None
        if lengths[0] <= target:
            best = (mid, lengths[0])
            low = mid + 1
        else:
            high = mid - 1
    return None if best is None or best[1] < 1 else (base[: best[0]], best[1])


def _abandon_if_replying(watch: Callable[[], bool] | None) -> None:
    """S218: raise `RamMeasurementAbandoned` when the caller's `watch` says a reply has started
    (or was already in flight when the watch was taken). `None` = no watch: never raises."""
    if watch is not None and not watch():
        log.info("reranker: RAM measurement abandoned: a reply started while it ran (S218)")
        raise RamMeasurementAbandoned("a reply started while the RAM measurement ran")


@contextlib.contextmanager
def _stop_on_reply(provider: RerankerProvider, watch: Callable[[], bool] | None):
    """S223: while the block runs, a reply that starts stops the call this thread makes on
    `provider` (`arm_terminate`; the stop is registered with the watch's `on_reply_start`, so it
    runs on the replying thread the moment the reply starts). On every exit the hook is
    unregistered and the scope disarmed. No watch (the bridge's lull job), or a watch without
    `on_reply_start` (a plain callable): nothing is armed and the call simply finishes (S218)."""
    register = getattr(watch, "on_reply_start", None)
    if register is None:
        yield
        return
    stop = provider.arm_terminate()
    unregister = _noop
    try:
        unregister = register(stop)
        yield
    finally:
        unregister()
        provider.disarm_terminate()


@dataclass(frozen=True)
class _Rung:
    """One ladder step: `pairs` documents, each pair at most `target` tokens long; `full`
    marks the model's full-length pair."""

    pairs: int
    target: int
    full: bool = False


def _window_groups(store: MemoryStore, model_id: str) -> list[ram_fit.ShapeGroup] | None:
    rows = _ram_shapes.shape_groups(store, model_id)
    return None if rows is None else ram_fit.groups_from_rows(rows)


def _next_length_rung(groups: Sequence[ram_fit.ShapeGroup], cap: int) -> _Rung | None:
    """Phase 1: the next single-pair rung, or `None` when the lengths are done. The first is the
    full-length pair; each next one targets half the shortest B=1 length measured, so its shape
    is new and every rung is dominated by one already measured (until `enough_shapes`)."""
    if not any(g.full for g in groups):
        return _Rung(1, cap, True)
    if ram_fit.enough_shapes(groups):
        return None
    single = [g.length for g in groups if g.pairs == 1]
    target = min(single) // 2 if single else 0
    return _Rung(1, target) if target >= 1 else None


def _batch_sizes() -> list[int]:
    """Phase 2 batch sizes: doubling from 2 up to the most pairs a recall sends (every
    candidate plus every anchor)."""
    top = CANDIDATE_POOL + P
    sizes, size = [], 2
    while size < top:
        sizes.append(size)
        size *= 2
    sizes.append(top)
    return sizes


def _run_rung(
    provider: RerankerProvider,
    store: MemoryStore,
    model_id: str,
    base_text: str,
    rung: _Rung,
    cap: int,
    watch: Callable[[], bool] | None,
) -> bool:
    """Measure one rung and record its time and its peak growth. Returns False (the pass ends)
    when the rung cannot be built or its growth cannot be read. An abandon (a reply started)
    discards this rung's sample and propagates."""
    if rung.full:
        built = (base_text, cap)
    else:
        built = _text_of_length(provider, base_text, rung.target)
    if built is None:
        _warn_ram_figure_once("RAM measurement stopped: a rung could not be built at the wanted length")
        return False
    text, length = built
    _abandon_if_replying(watch)
    try:
        with _stop_on_reply(provider, watch):
            _scores, seconds, growth = provider.rerank_timed(text, [text] * rung.pairs)
    except RerankTerminated:
        # S223: a reply started and stopped the call; nothing was measured.
        log.info("reranker: RAM measurement stopped: a reply started during the call (S223)")
        raise RamMeasurementAbandoned("a reply started and stopped the RAM measurement's call") from None
    _abandon_if_replying(watch)  # a reply started after the last node: this sample is discarded
    if growth is None:
        _warn_ram_figure_once("RAM measurement stopped: the call's peak growth could not be read")
        return False
    _record_rerank_cost(
        model_id,
        store,
        rung.pairs * length,
        seconds,
        growth,
        pairs=rung.pairs,
        longest=length,
        full_length=rung.full,
    )
    log.info(
        "reranker: RAM ladder rung %d pairs x %d tokens: peak growth %.0f MB, %.2f s",
        rung.pairs,
        length,
        growth / 1e6,
        seconds,
    )
    return True


def measure_ram_figure(
    provider: RerankerProvider, store: MemoryStore, *, watch: Callable[[], bool] | None = None
) -> bool:
    """S200, S208, S220: run the ladder (see the section header) on `provider` until the window
    gives a usable memory model, one measured call per rung, each recorded like any recall
    rerank (`_record_rerank_cost`). Returns True iff the window gives a usable model afterwards.
    The next rung is always decided from the window, so a restart or an abandoned pass resumes
    where it stands, and no rung repeats a shape. A rung that cannot be built, whose peak growth
    cannot be read, or that recorded nothing new ends the pass (it is retried at a later lull).
    A pass that gets through phase 2 (every batch size run, dominated, or ruled out by the time
    budget or the headroom) records the completion marker; only then does the window give a model
    for sizing (`_window_ram_fit`), so the first batches are never sized from single pairs alone.
    The FIRST rung (the full-length pair, about 300 MB on the reference model) runs without a
    headroom check, like the first-ever seed always did (S184, S200). Provider exceptions
    propagate.

    S218: with a `watch` (the direct chat passes `cli_throttle.reply_watch()`), a reply that
    starts while this runs abandons the rung in progress: `RamMeasurementAbandoned` is raised,
    checked before the pair-length probe, before and after each call, and that rung's sample is
    not recorded (earlier rungs' samples stay). S223: the call already running is STOPPED when
    the reply starts (the watch's `on_reply_start` calls the stop `provider.arm_terminate()`
    returned: ONNX Runtime's `RunOptions.terminate`, so the reply does not wait for the call and
    the call's own tail does not overlap the reply); a provider that cannot be stopped (the base
    `arm_terminate`, or a session that could not be wrapped) lets the call finish and only its
    SAMPLE is guaranteed not to overlap a reply (S218). With the memory pool off an abandoned
    call leaves no residue, so the pass simply resumes after the reply. The model load, the
    one-token warm-up and the tokenizer-only length probe are not stoppable."""
    model_id = provider.model_id()
    _abandon_if_replying(watch)
    pair = _full_length_pair(provider)
    if pair is None:
        _warn_ram_figure_once("RAM measurement not done: no full-length pair could be built")
        return False
    base_text, cap = pair
    _abandon_if_replying(watch)
    _warm_up_once(provider, "warm up", ["warm up"])
    # Phase 1: lengths.
    while True:
        groups = _window_groups(store, model_id)
        if groups is None:
            _warn_ram_figure_once("RAM measurement stopped: the saved series could not be read")
            return False
        rung = _next_length_rung(groups, cap)
        if rung is None:
            break
        shortest = min((g.length for g in groups if g.pairs == 1), default=None)
        if not _run_rung(provider, store, model_id, base_text, rung, cap, watch):
            return _window_ready(store, model_id)
        after = _window_groups(store, model_id)
        if after is None or not _rung_made_progress(rung, shortest, after):
            _warn_ram_figure_once("RAM measurement stopped: a rung recorded nothing new")
            return _window_ready(store, model_id)
    # Phase 2: batches of increasing size, at the time-budget frontier. The pass is COMPLETE when
    # every size has run, been dominated by a measured shape, or been ruled out by the time
    # budget or the live headroom (a larger one fits no better); a rung that cannot be built or
    # read leaves it incomplete, to be retried at a later lull.
    completed = True
    for pairs in _batch_sizes():
        _abandon_if_replying(watch)
        groups = _window_groups(store, model_id)
        time_fit = _rerank_cost.fit(store, model_id)
        fit = None if groups is None else ram_fit.fit_ram(groups)
        if fit is None or time_fit is None or time_fit.rate <= 0.0:
            break
        budget = _full_rerank_budget()
        if budget <= time_fit.overhead:
            break
        length = min(cap, int((budget - time_fit.overhead) / time_fit.rate) // pairs)
        if length < 1 or ram_fit.dominated(pairs, length, ram_fit.distinct_shapes(groups)):
            continue
        headroom = _available_ram_headroom_bytes()
        if headroom is not None and fit.predict(pairs, length) > headroom:
            log.info("reranker: RAM ladder stopped: %d pairs x %d tokens would not fit the headroom", pairs, length)
            break
        if not _run_rung(provider, store, model_id, base_text, _Rung(pairs, length), cap, watch):
            completed = False
            break
    if completed:
        _mark_ladder_complete(store, model_id)
    return _window_ready(store, model_id)


def _rung_made_progress(rung: _Rung, shortest: int | None, after: Sequence[ram_fit.ShapeGroup]) -> bool:
    """Whether a phase-1 rung left a new shape in the window: the full-length pair its marker, a
    shorter rung a single-pair length below the shortest one measured before it. A rung that
    recorded nothing (an unpackable shape, an unreadable series) would otherwise repeat forever."""
    if rung.full:
        return any(g.full for g in after)
    singles = [g.length for g in after if g.pairs == 1]
    return bool(singles) and shortest is not None and min(singles) < shortest


def _mark_ladder_complete(store: MemoryStore, model_id: str) -> None:
    """Record, once per window, that a ladder pass finished (`ram_fit.COMPLETE_MARKER`)."""
    rows = _ram_shapes.shape_groups(store, model_id)
    if rows is None or ram_fit.ladder_complete(rows):
        return
    _ram_shapes.record(store, model_id, ram_fit.COMPLETE_MARKER, 0.0)


def run_ram_figure(
    activity_marker: object, store: MemoryStore, *, watch: Callable[[], bool] | None = None
) -> bool:
    """The `rerank_ram_figure` job's run (S200): stamp the attempt, then run the ladder on the
    runtime reranker (building the provider here, at a lull, is what keeps the model load off the
    reply path). Returns True iff the window then gives a usable memory model; when it already
    does, nothing is built or measured and the result is False. Provider exceptions propagate
    (the central pass logs them).

    S218: `watch` (the direct chat's `cli_throttle.reply_watch()`, taken by the caller before
    the due check and the store) is checked before the provider is built and passed to
    `measure_ram_figure`. `RamMeasurementAbandoned` removes this model's attempt stamp, so the
    re-run after the reply is not held back by the once-per-marker rule, keeps the need flag and
    propagates. The bridge's lull job passes no watch."""
    model_id = resolve_reranker_model_id()
    with _cost_lock:
        _ram_figure_attempt[model_id] = activity_marker
    if _window_ready(store, model_id):
        with _cost_lock:
            _ram_figure_needed.discard(model_id)
        return False
    try:
        _abandon_if_replying(watch)
        ready = measure_ram_figure(build_reranker_provider(store=store), store, watch=watch)
    except RamMeasurementAbandoned:
        with _cost_lock:
            _ram_figure_attempt.pop(model_id, None)
        raise
    if ready:
        with _cost_lock:
            _ram_figure_needed.discard(model_id)
    return ready


def _reset_rerank_cost_model() -> None:
    """Test-only: forget every model id's kept fits, memos, warm-up state, budget misses,
    re-measure markers and "too long" verdicts. The saved series live in the database and
    the process buffer (`cost_series._reset_pending`)."""
    with _cost_lock:
        _ram_figure_needed.clear()
        _ram_figure_attempt.clear()
        _ram_figure_warned.clear()
        _ram_fit_kept.clear()
        _ram_fit_memo.clear()
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
    cost), the re-measure job's one trigger (`_needs_remeasure`)."""
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
    """Whether the re-measure job should time the miss's 5 candidates (S100, S113): the time
    term says they are stuck (`_is_stuck`). (The S182 RAM re-seed, a kept per-token figure
    that blocked the batch, is retired with that figure: S212, S220.)"""
    return estimate is not None and _is_stuck(miss, estimate)


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
    current estimate (the time term says fewer than 5 candidates fit: `_needs_remeasure`), the
    memory model is ready (`ram_figure_ready`: the job never runs an unfitted batch, S220),
    its warm-up has run in this process and its provider is
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
        if not ram_figure_ready(model_id, store):
            return False
        if _too_long_remembered(model_id, store):
            return False
        if not _needs_remeasure(miss, rerank_cost_estimate(model_id, store)):
            _drop_miss(model_id, miss)
            return False
    return True


def remeasure_stuck_width(activity_marker: object, store: MemoryStore) -> bool:
    """The `rerank_width_remeasure` job's run (S100, S113, S116): one timed
    rerank of the stuck miss's 5 candidates (5 real + `anchor_count(5)`
    anchors) on the already-loaded provider, added to the cost sums (its shape is a free
    memory sample too). Then the miss is cleared when the measured seconds are within the
    full budget (S116 a: 5 fit) and dropped when they exceed it (S116 b: genuinely too
    long, not stuck), and that verdict is remembered until the estimate next
    changes (S127: a sample arrives or one it counted ages out). The marker
    is stamped after the verdict check and before the measurement, so a failing
    or RAM-skipped run still runs once per lull, while a run skipped by a held
    verdict stamps nothing. Skipped (miss kept) when the memory model is not ready (S220: the
    batch is never run unfitted, same gate as every rerank), the miss is no longer
    stuck (then dropped), a "too long" verdict is remembered, the memory model holds no
    batch of more than one pair and the batch is larger than any shape measured (S242,
    `RamFit.within_measured`), or the memory model predicts the batch does not fit the current
    headroom. Returns True iff the rerank ran. Provider
    exceptions propagate (the central pass logs them)."""
    model_id = resolve_reranker_model_id()
    miss, _, warmed = _job_state(model_id)
    provider = _provider_cache.get(model_id)
    if miss is None or not warmed or provider is None:
        return False
    if not ram_figure_ready(model_id, store):
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
    pairs = len(miss.documents) + k
    longest = max([*miss.candidate_tokens, *miss.anchor_tokens[:k]])
    padded = pairs * longest
    if estimate is not None and estimate.ram_fit is not None:
        headroom = _available_ram_headroom_bytes()
        if headroom is not None:
            if not estimate.ram_fit.within_measured(pairs, longest):
                log.info("rerank width re-measure: skipped, the batch is larger than any shape measured (S242)")
                return False
            if estimate.ram_fit.predict(pairs, longest) > headroom:
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
    pairs = len(documents) + k
    _record_rerank_cost(
        model_id,
        store,
        pairs * longest,
        normalization.seconds,
        normalization.peak_growth_bytes,
        pairs=pairs,
        longest=longest,
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

_CGROUP_V1_ROOT = "/sys/fs/cgroup/memory"

# The kernel's own "no limit" sentinel for cgroup v1's memory.limit_in_bytes
# (there is no explicit "unbounded" marker like v2's "max"): LONG_MAX rounded down
# to a page boundary, so 0x7FFFFFFFFFFFF000 on 4 KiB pages, 0x7FFFFFFFFFFFC000 on
# 16 KiB and 0x7FFFFFFFFFFF0000 on 64 KiB. The constant is the 64 KiB value, so a
# limit at or above it is unbounded on every page size (S237 review F6); no real
# cap comes near it. Never a real cap.
_CGROUP_V1_UNBOUNDED_SENTINEL = 0x7FFFFFFFFFFF0000


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


def _cgroup_ancestor_dirs(root: str, cgroup_path: str) -> list[str]:
    """Every cgroup directory from the process's own cgroup UP TO `root` (the v2 mount or the
    v1 memory-controller mount), deepest first. Used to find the EFFECTIVE memory limit: a
    cgroup's own limit may read unbounded while an ANCESTOR still caps it, so the effective
    limit is the tightest one along this chain, not just the process's own level."""
    segments = [seg for seg in cgroup_path.split("/") if seg]
    dirs = []
    for depth in range(len(segments), -1, -1):
        sub = "/".join(segments[:depth])
        dirs.append(f"{root}/{sub}" if sub else root)
    return dirs


# S233: the five `memory.stat` lines that say how much of a cgroup's usage is page cache the
# kernel can give back. v2 names them `file` (all cached file data, tmpfs and shmem included),
# `file_mapped`, `file_dirty`, `file_writeback`, `shmem`; v1 has the hierarchical `total_`
# forms. They are matched exactly, never by substring.
_RECLAIM_KEYS_V2 = ("file", "file_mapped", "file_dirty", "file_writeback", "shmem")
_RECLAIM_KEYS_V1 = ("total_cache", "total_mapped_file", "total_dirty", "total_writeback", "total_shmem")


def _cgroup_reclaimable_file_bytes(stat_path: str, keys: tuple[str, ...]) -> float:
    """The file cache a cgroup can give back under memory pressure, in bytes, from its
    `memory.stat`: all cached file data minus what stays counted (S233),
    `file - file_mapped - file_dirty - file_writeback - shmem` (v1: `total_cache`,
    `total_mapped_file`, `total_dirty`, `total_writeback`, `total_shmem`, the hierarchical forms
    that match v1's hierarchical `memory.usage_in_bytes`), floored at 0. `keys` is
    `_RECLAIM_KEYS_V2` or `_RECLAIM_KEYS_V1`.

    The cgroup's usage counter counts page cache, such as the model files the process read,
    and the kernel gives back clean cache that no process has mapped when memory is needed,
    whether the kernel has marked it active or inactive; the headroom readers subtract that
    part from the usage so it does not read as used. What stays counted: pages some process
    keeps mapped (`file_mapped`: the loaded model weights), pages not yet written to disk
    (`file_dirty`, `file_writeback`) and tmpfs/shmem pages (swap-backed: they cannot be given
    back without swap, and `file` includes them). Mapped shmem sits in both `file_mapped` and
    `shmem`, so it is taken off twice, which only lowers the figure. Slab (`slab_reclaimable`)
    is not counted as reclaimable, which is stricter than the host-wide `available` figure.

    Keys are matched exactly (the first whitespace-split token of a line; the first line wins),
    so `file` never matches `file_mapped` or `total_cache`. Returns 0.0, meaning "subtract
    nothing" (the figure before S230, the tighter and safer side), when the file is absent
    (silent: a cgroup without the memory files), unreadable, ANY of the five lines is missing
    (a missing term would over-read), or a value is not a plain non-negative integer (a sign, an
    underscore, a non-ASCII digit, nan, inf or a value too large to be finite are all
    rejected); all of these except the absent file are warned once per process. Never raises,
    never negative."""
    try:
        with open(stat_path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return 0.0
    except Exception:  # noqa: BLE001 - fail-soft: a headroom probe must never break a recall
        _warn_ram_figure_once(
            "a cgroup memory.stat could not be read: reclaimable file cache is not subtracted from the headroom",
            exc_info=True,
        )
        return 0.0
    values: dict[str, float] = {}
    for line in content.splitlines():
        parts = line.split()
        if not parts or parts[0] not in keys or parts[0] in values:
            continue
        value = _cgroup_plain_bytes(parts[1]) if len(parts) >= 2 else None
        if value is not None:
            values[parts[0]] = value
            continue
        _warn_ram_figure_once(
            "a cgroup memory.stat page-cache value is not a plain integer: "
            "reclaimable file cache is not subtracted from the headroom"
        )
        return 0.0
    if len(values) != len(keys):
        _warn_ram_figure_once(
            "a cgroup memory.stat lacks one of the page-cache lines: "
            "reclaimable file cache is not subtracted from the headroom"
        )
        return 0.0
    cache, mapped, dirty, writeback, shmem = (values[k] for k in keys)
    return max(0.0, cache - mapped - dirty - writeback - shmem)


def _cgroup_plain_bytes(raw: str) -> float | None:
    """The one strict parse of a cgroup number (S239): ASCII digits only, finite. A sign, a
    decimal point, an exponent, an underscore, a non-ASCII digit, nan, inf, an empty string, text
    inside the digits and a value so long it overflows to inf (309+ digits) are all rejected
    (`None`); only surrounding blanks and the trailing newline are ignored."""
    text = raw.strip(" \t\r\n")
    if text.isascii() and text.isdigit():
        value = float(text)  # a 309-digit value parses to inf: rejected below
        if math.isfinite(value):
            return value
    return None


# What a warning about an unusable cgroup file says happens next (S240): a level known to be capped
# counts as having no headroom, a level not known to be capped is not used.
_CGROUP_UNUSABLE_NOTE = (
    "a level known to be capped then counts as having no headroom; a level not known to be capped is not used"
)


def _cgroup_read(
    path: str, what: str, *, note: str = _CGROUP_UNUSABLE_NOTE
) -> tuple[str | None, str | None]:
    """One cgroup file as `(text, None)`, or `(None, "absent")` when it does not exist (silent: the
    level exposes no such file), or `(None, "unreadable")` for any other failure to read it
    (warned once per process and file name, with the traceback). `what` is the file's base name.
    The two failures are told apart because S240 treats them differently: an absent limit file is
    no limit, an unreadable one is an unknown."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read(), None
    except FileNotFoundError:
        return None, "absent"
    except Exception:  # noqa: BLE001 - fail-soft: a headroom probe must never break a recall
        _warn_ram_figure_once(f"a cgroup {what} could not be read: {note}", exc_info=True)
        return None, "unreadable"


def _cgroup_read_file(path: str, what: str, *, note: str = _CGROUP_UNUSABLE_NOTE) -> str | None:
    """The text of one cgroup file, or `None` when it is absent or unreadable (`_cgroup_read`); `note`
    is what the once-per-process warning says happens next."""
    return _cgroup_read(path, what, note=note)[0]


def _warn_cgroup_not_plain(what: str) -> None:
    _warn_ram_figure_once(f"a cgroup {what} is not a plain integer: {_CGROUP_UNUSABLE_NOTE}")


def _cgroup_counter_bytes(path: str, what: str, *, capped: bool = False) -> float | None:
    """A one-number cgroup counter (`memory.current`, `memory.usage_in_bytes`) in bytes, or
    `None` when it cannot be used: absent, unreadable or not a plain non-negative integer
    (`_cgroup_plain_bytes`; the last two warned once per process and kind). `capped` marks the
    usage of a level known to be capped (S240): the result is `None` for an absent file either way,
    but there it is also warned once, because the caller then counts the level as having no
    headroom."""
    raw, failure = _cgroup_read(path, what)
    if raw is None:
        if failure == "absent" and capped:
            _warn_ram_figure_once(f"a cgroup {what} is absent on a level that is capped: {_CGROUP_UNUSABLE_NOTE}")
        return None
    value = _cgroup_plain_bytes(raw)
    if value is None:
        _warn_cgroup_not_plain(what)
    return value


def _cgroup_v2_level_limit(level_dir: str) -> tuple[float | None, bool] | None:
    """What a cgroup v2 level imposes on its processes and descendants (S239, S240), as
    `(limit, degraded)`, or `None` when it gives no sign of a cap.

    The limit is the lower of the numeric `memory.max` and `memory.high` (the kernel throttles a
    cgroup that runs past `memory.high`, so running past it is as unwelcome as running past
    `memory.max`). "max" and an absent file impose nothing and are silent. A file that is present
    but unusable (unreadable, or not a plain integer; each warned once) makes the level DEGRADED
    when the level is known to be capped: its other limit file reads as a number, or reads "max"
    (a level with a usable limit file and an unusable one is a cgroup whose second limit we cannot
    see, so it is treated as capped, and its headroom is 0). A level none of whose limit files is
    usable gives no evidence of a cap and is skipped (`None`, with the warnings for its unusable
    files): a looser ancestor can then bind there."""
    numbers: list[float] = []
    saw_max = False
    unusable = False
    for name in ("memory.max", "memory.high"):
        raw, failure = _cgroup_read(f"{level_dir}/{name}", name)
        if failure == "unreadable":
            unusable = True
            continue
        if raw is None:
            continue  # absent: no such limit
        if raw.strip() == "max":
            saw_max = True
            continue
        value = _cgroup_plain_bytes(raw)
        if value is None:
            _warn_cgroup_not_plain(name)
            unusable = True
            continue
        numbers.append(value)
    if numbers:
        return min(numbers), unusable
    if saw_max and unusable:
        return None, True
    return None


def _cgroup_v1_level_limit(level_dir: str) -> tuple[float, bool] | None:
    """The limit a cgroup v1 level imposes (`memory.limit_in_bytes`) as `(limit, False)`, or
    `None` when it has none: the file is absent, unreadable or not a plain integer (the last two
    warned once), or the value is at or above the kernel's unbounded sentinel
    (`_CGROUP_V1_UNBOUNDED_SENTINEL`). A level whose only limit file is unusable gives no
    evidence of a cap and is skipped."""
    value = _cgroup_counter_bytes(f"{level_dir}/memory.limit_in_bytes", "memory.limit_in_bytes")
    if value is None or value >= _CGROUP_V1_UNBOUNDED_SENTINEL:
        return None
    return value, False


def _cgroup_chain_headroom(
    limited: list[tuple[str, float | None, bool]], usage_name: str, keys: tuple[str, ...]
) -> float | None:
    """The headroom of a chain of cgroup levels, the smallest over its levels that are known to be
    capped (S236, S240), for `limited` = those levels as (directory, limit, degraded), deepest
    first. A level's headroom is `max(0, limit - (usage - reclaimable))` with THAT level's own
    usage (`usage_name`, which counts all of its descendants, so the siblings of the process's
    cgroup too: S235) and its own `memory.stat` (`_cgroup_reclaimable_file_bytes`, capped at that
    usage); the level that would run out first binds, which is not always the one with the
    smallest limit.

    A level known to be capped whose figure cannot be measured has headroom 0 (S240), so an
    unmeasurable capped level can never let a looser ancestor bind: a DEGRADED level (a limit
    file we cannot read, `_cgroup_v2_level_limit`) and a level whose usage counter is absent,
    unreadable or not a plain integer both count 0, and neither has its `memory.stat` opened. The
    process's own usage is never borrowed for an ancestor. `None` when no level is capped (the
    caller falls back)."""
    headroom: float | None = None
    for level_dir, limit, degraded in limited:
        usage = None if degraded else _cgroup_counter_bytes(f"{level_dir}/{usage_name}", usage_name, capped=True)
        if degraded or usage is None or limit is None:
            level_headroom = 0.0
        else:
            reclaimable = min(_cgroup_reclaimable_file_bytes(f"{level_dir}/memory.stat", keys), usage)
            level_headroom = max(0.0, limit - (usage - reclaimable))
        headroom = level_headroom if headroom is None else min(headroom, level_headroom)
    return headroom


def _cgroup_v2_memory_headroom_bytes() -> float | None:
    """cgroup v2 memory headroom for the CALLING PROCESS, resolved via `/proc/self/cgroup`
    (see `_read_proc_self_cgroup_v2_path`) rather than the fixed cgroup2 root (the root has no
    `memory.max`/`memory.current` of its own, so a root-only read can never see a real
    per-process cap).

    The kernel enforces EVERY level's limit on the chain from the process's own cgroup up to the
    root, against that level's own usage, which counts all of its descendants (the siblings of
    the process's cgroup too). A level's limit is the lower of its numeric `memory.max` and
    `memory.high` (S239, `_cgroup_v2_level_limit`). So the headroom is computed per capped
    level, `max(0, limit - (memory.current - reclaimable))` with that level's own
    `memory.current` and `memory.stat` (S235), and the result is the MINIMUM over the levels
    (`_cgroup_chain_headroom`): the level that would run out first binds, which is not always
    the one with the smallest limit (an ancestor with a larger limit whose siblings have filled
    it binds first). Levels with no numeric limit impose nothing and are not read further.
    `reclaimable` is the file cache the kernel can give back (`_cgroup_reclaimable_file_bytes`,
    S233: `file - file_mapped - file_dirty - file_writeback - shmem`, capped at that level's
    usage); mapped model pages, dirty and writeback pages and tmpfs stay counted, and so does
    slab. A level whose `memory.stat` cannot be used subtracts nothing for that level.

    A level known to be capped whose `memory.current` cannot be used (absent, unreadable, not a
    plain integer), or one of whose two limit files is present but unusable, has headroom 0
    (S240), so an unmeasurable capped level can never let a looser ancestor bind; the process's
    own usage is never borrowed for an ancestor. A level none of whose limit files is usable
    gives no evidence of a cap and is skipped. Numbers are plain non-negative integers only
    (`_cgroup_plain_bytes`, S239: no sign, decimal point or exponent). In a container whose
    cgroup path is absent under the mounted tree (S243) the mount root is the last level of
    the chain and is read with its own files like any ancestor.

    Returns `None` (never a fabricated headroom) when: not Linux, `/proc/self/cgroup` is
    missing/unparsable, or no level on the chain has a usable numeric limit (none is known to be
    capped: no cap anywhere -> no need to even read usage; a different cause from a capped level
    that cannot be measured, which gives 0.0). `None` means "the caller falls back to the next source" (cgroup v1, then
    psutil's host-wide available figure), never "unlimited"."""
    if sys.platform != "linux":
        return None
    cgroup_path = _read_proc_self_cgroup_v2_path()
    if cgroup_path is None:
        return None
    dirs = _cgroup_ancestor_dirs(_CGROUP_V2_ROOT, cgroup_path)
    limited = [(d, *found) for d in dirs if (found := _cgroup_v2_level_limit(d)) is not None]
    if not limited:
        return None  # no level known to be capped on the chain -> caller falls back
    return _cgroup_chain_headroom(limited, "memory.current", _RECLAIM_KEYS_V2)


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


def _cgroup_v1_walk_dirs(dirs: list[str]) -> list[str]:
    """The levels of a cgroup v1 chain (`dirs`, the process's own level first) that the kernel
    charges this cgroup to (S237). v1 charges a cgroup into its parent's counters only when the
    PARENT has hierarchy accounting on, which the parent's own `memory.use_hierarchy` shows ("1";
    a cgroup keeps it fixed while it has children). So an ancestor is walked only while its own
    `memory.use_hierarchy` does not read "0": at a "0" the walk ends there, because that level's
    usage and limit do not involve this cgroup (nor do those of anything above it). An absent,
    unreadable or any other value keeps walking (the safe side: more bounds). One read per
    ancestor level visited, the root included. (From the kernel's memory-controller rules; this
    VM is cgroup v2 only, so it is tested with synthetic trees.)"""
    walked = [dirs[0]]
    for parent in dirs[1:]:
        raw = _cgroup_read_file(
            f"{parent}/memory.use_hierarchy",
            "memory.use_hierarchy",
            note="the v1 walk keeps going past this level (the safe side: more bounds)",
        )
        if raw is not None and raw.strip() == "0":
            break
        walked.append(parent)
    return walked


def _cgroup_file_absent(path: str) -> bool:
    """True only when `path` does not exist (FileNotFoundError on opening it); any other outcome,
    a permission error included, is not absence. Nothing is read."""
    try:
        with open(path, encoding="utf-8"):
            return False
    except FileNotFoundError:
        return True
    except Exception:  # noqa: BLE001 - not absence: the level is walked and read normally
        return False


def _cgroup_v1_chain_dirs(cgroup_path: str) -> list[str]:
    """The levels of the v1 chain to measure (S243, S237), the process's own level first. If the
    path `/proc/self/cgroup` names is absent under the mounted memory-controller tree (a
    container without a cgroup namespace view: the mount root IS the container's own cgroup),
    the chain starts at the first level that exists (the mount root in that layout), and that
    level is treated as the process's own level; it is not an ancestor, so it is not subject to
    the `memory.use_hierarchy` rule below. `[]` when no level exists. From there the chain is
    cut by `_cgroup_v1_walk_dirs`."""
    all_dirs = _cgroup_ancestor_dirs(_CGROUP_V1_ROOT, cgroup_path)
    for start, level in enumerate(all_dirs):
        if not _cgroup_file_absent(f"{level}/memory.limit_in_bytes"):
            return _cgroup_v1_walk_dirs(all_dirs[start:])
    return []


def _cgroup_v1_memory_headroom_bytes() -> float | None:
    """cgroup v1 fallback for `_cgroup_v2_memory_headroom_bytes` (older kernels/distros without
    the unified v2 hierarchy), computed the same way (S237, S240): for every level of the chain
    from the process's own cgroup up to the memory-controller root, `memory.limit_in_bytes -
    (memory.usage_in_bytes - reclaimable file cache)` with that level's own usage and
    `memory.stat`, and the result is the MINIMUM over the levels known to be capped
    (`_cgroup_chain_headroom`). The reclaimable cache is S233's `total_cache - total_mapped_file
    - total_dirty - total_writeback - total_shmem` (`_cgroup_reclaimable_file_bytes`, the
    hierarchical `total_` keys), so clean unmapped cache such as read model files does not read
    as used; mapped pages, dirty and writeback pages and tmpfs stay counted. The chain is read
    from the CALLING PROCESS's own cgroup path, resolved via `/proc/self/cgroup`'s `memory`
    controller line (`_read_proc_self_cgroup_v1_memory_path`) rather than the fixed root; when
    that path is absent under the mounted tree the first existing level is the process's own
    (`_cgroup_v1_chain_dirs`, S243).

    In cgroup v1 a cgroup's `memory.limit_in_bytes` is its own setting: the kernel enforces each
    ancestor's limit separately against that ancestor's usage, which includes this cgroup and its
    siblings, so each ancestor is read with its own counters, and the walk stops at an ancestor
    whose own `memory.use_hierarchy` reads "0" (`_cgroup_v1_walk_dirs`: no hierarchical charging,
    so its counters do not involve this cgroup).

    An unbounded v1 limit reads back as the kernel's own huge sentinel value (there is no
    explicit "unbounded" marker like v2's `"max"`); a level at or above
    `_CGROUP_V1_UNBOUNDED_SENTINEL` has no limit and is not read further, and a level whose limit
    file is unusable gives no evidence of a cap and is skipped. A capped level whose usage cannot
    be used has headroom 0, as in the v2 reader (S240). `None` on no v1 controller, no level
    known to be capped on the walked chain - the same "caller falls back" signal as every other
    case here, never a fabricated headroom."""
    if sys.platform != "linux":
        return None

    cgroup_path = _read_proc_self_cgroup_v1_memory_path()
    if cgroup_path is None:
        return None
    dirs = _cgroup_v1_chain_dirs(cgroup_path)
    limited = [(d, *found) for d in dirs if (found := _cgroup_v1_level_limit(d)) is not None]
    if not limited:
        return None
    return _cgroup_chain_headroom(limited, "memory.usage_in_bytes", _RECLAIM_KEYS_V1)


def _psutil_available_bytes() -> float | None:
    """Host-wide free memory: `psutil.virtual_memory().available` (S211: the same call on
    Linux, macOS and Windows; the kernel's "usable without swapping" estimate), read on every
    headroom call and taken as the smaller of it and the cgroup figure (S241, see
    `_available_ram_headroom_bytes`). `None` when psutil cannot read it (warned once per
    process), never a fabricated figure."""
    try:
        import psutil

        return float(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 — fail-soft: a headroom probe must never break a recall
        _warn_ram_figure_once(
            "the host free memory could not be read (psutil): the RAM bound's headroom term is skipped",
            exc_info=True,
        )
        return None


def _available_ram_headroom_bytes() -> float | None:
    """Cheap, per-call RAM-headroom read for the width fit's memory
    bound (`rerank_for_recall`) — a fresh read on every message, since
    this is inexpensive (a
    handful of small `/proc`-or-`/sys` reads) and lets width react to
    headroom changes mid-session (another process on the box growing), the
    same trade-off the spec's own Open Reconfirmation favors absent a
    concrete reason not to.

    The headroom is the SMALLER of two figures (S241): the cgroup figure and the host-wide
    available memory read through `psutil`. A cgroup cap above what the machine has free gives no
    extra memory, so the cap's room alone must never exceed what the host can supply.

    The cgroup figures are limit - (usage - reclaimable file cache) (S233): clean file cache
    that no process has mapped is given back by the kernel under pressure, so it does not read
    as used (`file - file_mapped - file_dirty - file_writeback - shmem` of `memory.stat`; v1
    `total_` forms); mapped model pages, dirty and writeback pages and tmpfs stay counted. The
    cgroup figure (v2 and v1 alike) is the minimum over the capped levels of the chain, each
    against its own usage (S235, S237); a level's limit is the lower of `memory.max` and
    `memory.high` on v2 (S239), and a capped level that cannot be measured counts 0 (S240).
    The psutil figure also credits mapped pages and slab, which the cgroup figure does not, so
    the minimum is usually the cgroup one under a cap.

    cgroup-limit-aware, per the pre-flip revision's Change-3 requirement:
    Testing's own OOM reproduction ran under an explicit cgroup cap that
    the host's plain free-memory reading does NOT reflect, so a host-only
    read would fail to prevent the exact OOM this change fixes. Tries cgroup
    v2 first, then v1 (both Linux only); psutil (S211: the same call on Linux, macOS and
    Windows) is read on every call and bounds the cgroup figure from above. With no capped
    level anywhere the cgroup figure is absent and psutil's reading stands alone. One reading
    absent leaves the other. The one remaining `None` — never a fabricated number — is when
    neither can be read; the width fit treats `None` as "skip the memory term" and the judge
    tier guard floors to the safe tier, never as "unlimited headroom". Also imported by
    `judge_selftune.py`."""
    cgroup = _cgroup_v2_memory_headroom_bytes()
    if cgroup is None:
        cgroup = _cgroup_v1_memory_headroom_bytes()
    host = _psutil_available_bytes()
    readings = [r for r in (cgroup, host) if r is not None]
    return min(readings) if readings else None


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
    factors) and, when a memory model and a headroom reading exist, the
    peak growth it is predicted to need (`ram_fit.RamFit.predict` of its pair
    count and longest pair, see the section header; the whole growth, no
    margin) fits the headroom (S32, S65: no chunking), and, while the memory model holds no
    measured batch of more than one pair, it has no more pairs and no longer a padded length
    than the largest shape measured (`RamFit.within_measured`, S242: such a batch would be
    extrapolated from single pairs; the cap is part of the RAM term, so a missing headroom
    reading leaves the width time-bound as before).
    Both factors of the padded size only grow with n, so the scan stops at
    the first prefix that does not fit.

    The result may be below the S5 minimum; the caller then does not rerank
    (S5/S23). Pure: no clock, no I/O."""
    limit = min(len(candidate_pair_tokens), max_real)
    if estimate is None:
        return min(RERANK_MIN_REAL_CANDIDATES, limit)
    memory = estimate.ram_fit
    ram_bound = memory is not None and headroom_bytes is not None
    width = 0
    longest_real = 0
    for n in range(1, limit + 1):
        longest_real = max(longest_real, candidate_pair_tokens[n - 1])
        k = anchor_count(n)
        longest = max([longest_real, *anchor_pair_tokens[:k]])
        pairs = n + k
        if prefix_cost(estimate, pairs, longest) > budget_seconds:
            break
        if ram_bound and (not memory.within_measured(pairs, longest) or memory.predict(pairs, longest) > headroom_bytes):
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
    if not ram_figure_ready(model_id, store):
        # S203 (a process about to rerank with no usable memory model): no width-5 seeding
        # batch any more. `search_paragraphs` holds such a turn back before it builds the
        # reranker (S200); this is the same rule for a model that aged out between that check
        # and here. Nothing is warmed up or ranked; the need is flagged and the ladder (S208,
        # S220) makes the model.
        note_ram_figure_needed(model_id)
        return RecallRerank(
            width=0, reranked=False, hand_off="ram_figure", normalization=None, measured=False
        )
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
    headroom = _available_ram_headroom_bytes() if estimate is not None else None
    width = fit_rerank_width(candidate_tokens, anchor_tokens, estimate, budget_seconds, headroom, cap)

    if width < RERANK_MIN_REAL_CANDIDATES:
        # The MEASURED estimate says fewer than the S5 minimum fit this
        # query's budget, so it is not reranked and adds no measurement. It is
        # kept as the model id's last budget miss (S113); while the estimate
        # says its first 5 do not fit the FULL budget, the re-measure job
        # re-times them at a chat lull (S100), so an inflated estimate cannot
        # absorb every later message. (The RAM term can land here too, for a
        # batch whose predicted growth does not fit the headroom, or, while the
        # memory model holds single pairs only, for a batch larger than any
        # shape measured (S242); the time-stuck job leaves that miss alone. The
        # headroom changes by itself, the single-pair model only when the
        # ladder next measures a batch.)
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
    `seconds` / `peak_growth_bytes` — the scoring call's measurements from
    `RerankerProvider.rerank_timed` (feed the width fit's cost model).
    """

    scores: list[float]
    real_width: int
    did_normalize: bool
    anchor_count: int = 0
    seconds: float | None = None
    peak_growth_bytes: float | None = None


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
        scores, seconds, growth = provider.rerank_timed(query, real)
        return AnchorNormalizationResult(
            scores=scores,
            real_width=real_width,
            did_normalize=False,
            seconds=seconds,
            peak_growth_bytes=growth,
        )

    raw_scores, seconds, growth = provider.rerank_timed(query, real + ANCHOR_POOL[:k])
    real_scores = raw_scores[:real_width]
    anchor_scores = raw_scores[real_width:]
    normalized = _median_normalize(real_scores, anchor_scores)
    return AnchorNormalizationResult(
        scores=normalized,
        real_width=real_width,
        did_normalize=True,
        anchor_count=k,
        seconds=seconds,
        peak_growth_bytes=growth,
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
