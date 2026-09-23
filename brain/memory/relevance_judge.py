"""Local relevance judge + Haiku tie-break for F2a's offline daily
calibration tick (#250 §6, inc6).

Runs ENTIRELY inside `brain.bridge.supervisor._run_calibration_tick` (the
once-daily idle-gated cadence) — never on the per-turn recall path. Two
stages, mirroring `brain/memory/reranker.py`'s provider-ABC + process-cache
shape:

  1. `TorchCrossEncoderJudge` (`BAAI/bge-reranker-v2-m3`, model_tier's
     `TIER_RELEVANCE_JUDGE`) scores every candidate in the day's SAMPLE of
     `calibration_log` rows. Clear-case pairs get their local-judge label
     directly; pairs landing in the self-contained sigmoid-margin AMBIGUOUS
     BAND around the judge's own decision boundary route to stage 2.
  2. `_make_haiku_tiebreak` (mirrors `brain.engines.consolidation.
     _make_haiku_classifier`'s shape) resolves the ambiguous band only —
     kept low-volume by construction (ambiguous-fraction × the daily sample),
     so API cost stays negligible per the spec.

LOAD MECHANISM (Roy 2026-09-18, "torch it is then"): `TorchCrossEncoderJudge`
loads the judge DIRECTLY via `torch`/`sentence-transformers`
(`sentence_transformers.CrossEncoder`), NOT fastembed/onnxruntime — the one
place in this codebase torch is used. `torch`/`sentence_transformers` are
imported LAZILY, inside `TorchCrossEncoderJudge.__init__` only — never at
this module's top level, and never on the per-turn hot path (the e5-large
embedder and jina reranker stay ONNX/onnxruntime, torch-free). Importing
this module, or any recall/reranker/embedder module, never pulls `torch`
into `sys.modules` — only actually constructing a `TorchCrossEncoderJudge`
does (i.e. only when the daily tick runs a real judge pass). See
`model_tier.py`'s `MODEL_RELEVANCE_JUDGE` comment for the full rationale.

Fault isolation (spec Section 5/6, "must not crash the tick or the bridge"):
`label_calibration_sample` is the ONE entry point `_run_calibration_tick`
calls, and it never raises — a judge-construction failure (missing torch,
failed download, ...), a per-row/per-candidate failure, or a Haiku failure
are all caught and logged individually, at the narrowest scope each can be
isolated to, so one bad candidate/row never loses an otherwise-good pass and
one bad pass never crashes the tick.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from brain import prompt_strings, tunables
from brain.bridge.provider import LLMProvider

# F2a inc7 (#250 §7): TYPE_CHECKING-gated, not a runtime top-level import.
# `MemoryStore` is used here ONLY as a type hint (`from __future__ import
# annotations` above already defers every annotation to a string, so the
# name is never evaluated at runtime); the runtime edge back to store.py
# this used to be (`from brain.memory.store import MemoryStore` at module
# top) created a genuine store.py <-> relevance_judge.py IMPORT CYCLE risk
# once store.py's own retention-window derivation (inc7,
# `floor_calibration.py`) needed to read `CALIBRATION_SAMPLE_ROWS` from
# THIS module — whichever module happened to be imported first would hit
# a half-initialized sibling depending on import order. Since this module
# never actually calls anything ON the `MemoryStore` class itself (only
# duck-typed attribute/method access on the `store` parameter), dropping
# the runtime import removes the cycle entirely rather than relying on a
# fragile "import order happens to work out" hope.
if TYPE_CHECKING:
    from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Text externalized to prompt_strings.toml [memory.relevance_judge] (#129 convention).
_HAIKU_TIEBREAK_PROMPT = prompt_strings.register("memory.relevance_judge.haiku_tiebreak_prompt")

# ---------------------------------------------------------------------------
# Tunables (I7/I3) — derived defaults, operator-overridable via tunables.json.
# ---------------------------------------------------------------------------

# How many calibration_log ROWS the daily tick judges (not every logged row —
# spec Section 7's "sampling IS the design"). Two-sided derivation, same
# shape as store.py's CALIBRATION_LOG_RETENTION_WINDOW_DAYS:
#   (i) LOWER bound — the spec cites the cutoff-fit's research basis as
#       "robust at hundreds of pairs" (Section 7). Each sampled row yields
#       one judged PAIR per candidate in that row's rerank pool (the
#       auto-scaled width `reranker.get_rerank_width` chose that turn,
#       bounded above by `relevance.CANDIDATE_POOL`=50 and below by its own
#       floor of 1 on a maximally latency-starved potato host). 100 rows is
#       chosen so that even a persistently narrow, single-candidate pool
#       (the auto-scaler's worst case) still clears "hundreds of pairs" by
#       inc7's floor-fit time (the fit itself draws from accumulated
#       labeled rows across the retention window, not one day in
#       isolation) — a genuinely narrow-pool host is rare in practice
#       (get_rerank_width only bottoms out at 1 under a very tight budget +
#       slow per-doc cost), so ordinary hosts clear it in a single day.
#   (ii) UPPER bound — bounding the local judge's daily compute on the
#       no-AVX2 potato baseline (spec Section 7): 100 forward passes/day
#       through a cross-encoder is a bounded, once-daily idle cost, not the
#       thousands of raw per-turn rows actually logged.
# Operator-tunable (`calibration.judge_sample_rows`) for a box where either
# side of this balance needs shifting.
CALIBRATION_SAMPLE_ROWS: int = tunables.register("calibration.judge_sample_rows", 100)

# Half-width of the self-contained AMBIGUOUS BAND around the judge's own
# sigmoid decision boundary (p=0.5 — where a sigmoid's classification
# confidence is at its minimum), spec Section 6: NOT derived from any
# external floor (that would be a later increment's job, and the spec
# explicitly requires this band to be self-contained/build-time). Chosen as
# a modest fraction (10% total width: [0.45, 0.55]) of the full [0,1]
# probability range — narrow enough to keep Haiku tie-break volume low (the
# spec's explicit "kept low-volume" requirement), wide enough to catch the
# region where the judge's own confidence is genuinely lowest.
# Operator-tunable (`calibration.judge_ambiguous_band_half_width`).
AMBIGUOUS_BAND_HALF_WIDTH: float = tunables.register(
    "calibration.judge_ambiguous_band_half_width", 0.05
)


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid — maps a raw cross-encoder logit
    into [0, 1] so the ambiguous band has a fixed, model-independent
    midpoint (0.5) regardless of the judge's raw logit scale."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def label_for_score(
    raw_score: float, *, band_half_width: float | None = None
) -> tuple[str, bool]:
    """Return `(provisional_label, is_ambiguous)` for one judge score.

    `provisional_label` is `"relevant"` or `"irrelevant"` — the local
    judge's own call, used as the FINAL label whenever `is_ambiguous` is
    False, and as the fallback if `is_ambiguous` is True but the Haiku
    tie-break itself fails (fail-soft toward the signal that already
    exists, mirroring `consolidation._make_haiku_classifier`'s
    fail-toward-keeping-content posture).

    `is_ambiguous` is True when the sigmoid-transformed score falls inside
    `AMBIGUOUS_BAND_HALF_WIDTH` of the 0.5 decision boundary — those, and
    only those, positions get a Haiku tie-break call (acceptance #7: "no
    Haiku call on clear cases").
    """
    if band_half_width is None:
        band_half_width = tunables.get_tunable(
            "calibration.judge_ambiguous_band_half_width", AMBIGUOUS_BAND_HALF_WIDTH
        )
    p = _sigmoid(raw_score)
    provisional = "relevant" if p >= 0.5 else "irrelevant"
    is_ambiguous = abs(p - 0.5) < band_half_width
    return provisional, is_ambiguous


# ---------------------------------------------------------------------------
# Judge provider ABC + concrete implementations — mirrors
# brain/memory/reranker.py's RerankerProvider / CrossEncoderProvider /
# FakeRerankerProvider shape.
# ---------------------------------------------------------------------------


class RelevanceJudgeProvider(ABC):
    """Abstract local relevance judge. Subclasses implement `score` and `model_id`."""

    @abstractmethod
    def score(self, query: str, document: str) -> float:
        """Raw relevance logit for (query, document) — NOT [0,1]-normalized
        (see `label_for_score`, which applies the sigmoid). Higher = more
        relevant, same convention as `RerankerProvider.rerank`."""

    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier for the model producing these scores."""


class TorchCrossEncoderJudge(RelevanceJudgeProvider):
    """Real local judge via `sentence_transformers.CrossEncoder` (torch
    backend, CPU-only install — see pyproject.toml). Production default.

    Model id comes from `model_tier.py` (`model_for_tier(TIER_RELEVANCE_
    JUDGE)`), never hardcoded here — same convention as `CrossEncoderProvider`
    in reranker.py. Runs OFFLINE, only inside the daily calibration tick, so
    torch's load cost is irrelevant to per-turn recall latency (I6).
    """

    def __init__(self, model_id: str, cache_dir: str | Path) -> None:
        # Imported lazily (function-scoped, not top-level) so importing this
        # MODULE never requires torch/sentence-transformers to be installed
        # unless the real judge is actually constructed — mirrors
        # CrossEncoderProvider's identical lazy fastembed import, and is the
        # load-bearing reason this module's own import never pulls torch
        # into sys.modules.
        from sentence_transformers import CrossEncoder

        self._model_id = model_id
        self._model = CrossEncoder(model_id, cache_folder=str(cache_dir))
        # Same rationale as CrossEncoderProvider._rerank_lock: a shared
        # instance of this provider (the process-wide cache below) could in
        # principle have .score() called concurrently; serialize inference
        # behind one instance lock rather than assume single-threaded use.
        self._score_lock = threading.Lock()

    def score(self, query: str, document: str) -> float:
        with self._score_lock:
            # activation_fn=None does NOT mean "no activation" — sentence-
            # transformers' own docs: "If None, the model.activation_fn will
            # be used, which defaults to torch.nn.Sigmoid if num_labels=1,
            # else Identity." bge-reranker-v2-m3 is a num_labels=1 cross-
            # encoder, so activation_fn=None would silently pre-sigmoid the
            # score here, and label_for_score below would then apply ITS
            # OWN sigmoid on top — a double-sigmoid that compresses every
            # score toward 0.5 and corrupts the ambiguous-band logic.
            # Passing an explicit identity callable forces the RAW logit
            # through regardless of the model's own configured default —
            # confirmed against the installed sentence-transformers 6.1.0.
            result = self._model.predict([(query, document)], activation_fn=lambda x: x)
        return float(result[0])

    def model_id(self) -> str:
        return self._model_id


class FakeRelevanceJudgeProvider(RelevanceJudgeProvider):
    """Deterministic, scriptable judge for tests — zero network, zero model
    load, zero torch import. Mirrors `reranker.FakeRerankerProvider`:
    SCRIPTED (not hash-based) so a test can place a given (query, document)
    pair exactly where it needs it (deep in "relevant", deep in
    "irrelevant", or dead-center in the ambiguous band) rather than getting
    a consistent-but-uncontrollable score.
    """

    _DEFAULT_UNSCORED = -1_000.0

    def __init__(
        self, scores: dict[tuple[str, str], float] | None = None, *, default: float = _DEFAULT_UNSCORED
    ) -> None:
        self._scores = scores or {}
        self._default = default

    def score(self, query: str, document: str) -> float:
        return self._scores.get((query, document), self._default)

    def model_id(self) -> str:
        return "fake-relevance-judge"


# Process-wide provider cache keyed by model_id (mirrors reranker.py's
# _provider_cache). Kept at module scope so `_reset_judge_provider_cache`
# (test-only) can reach it and so monkeypatching the *function* fully
# controls behavior.
_provider_cache: dict[str, RelevanceJudgeProvider] = {}
_provider_cache_lock = threading.Lock()


def build_judge_provider() -> RelevanceJudgeProvider:
    """The production judge provider: `TorchCrossEncoderJudge` pinned to
    `model_tier.TIER_RELEVANCE_JUDGE`'s model id, caching the model file in
    the shared `get_cache_dir()` (one download across every persona on the
    box, same reasoning as the embedder/reranker).

    PROCESS-WIDE CACHING, same rationale as `reranker.build_reranker_
    provider`: constructing a `TorchCrossEncoderJudge` loads a real torch
    model — expensive to redo every tick. Keyed by model_id (double-checked
    locking: unlocked fast-path read for the common already-cached case).

    TEST ISOLATION: tests must monkeypatch this function directly (mirrors
    `reranker.build_reranker_provider`'s test-fixture convention) rather
    than relying on the cache alone — `_reset_judge_provider_cache` is the
    test-only reset hook for the cache itself.
    """
    from brain.bridge.model_tier import TIER_RELEVANCE_JUDGE, model_for_tier
    from brain.paths import get_cache_dir

    model_id = model_for_tier(TIER_RELEVANCE_JUDGE)

    provider = _provider_cache.get(model_id)
    if provider is not None:
        return provider

    with _provider_cache_lock:
        provider = _provider_cache.get(model_id)
        if provider is not None:
            return provider
        provider = TorchCrossEncoderJudge(model_id=model_id, cache_dir=get_cache_dir())
        _provider_cache[model_id] = provider
        return provider


def _reset_judge_provider_cache() -> None:
    """Test-only: clear the process-wide judge provider cache."""
    with _provider_cache_lock:
        _provider_cache.clear()


# ---------------------------------------------------------------------------
# Haiku tie-break — mirrors consolidation._make_haiku_classifier's shape
# (construction + call + fail-soft-on-any-failure), so it is mockable the
# same way (inject the returned callable directly in tests; never hits the
# network in the unit suite).
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str:
    """Pull the first {...} JSON object out of a model reply. Local copy of
    consolidation._extract_json's exact logic — kept module-local rather
    than importing a private helper across modules."""
    import re

    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else "{}"


def _make_haiku_tiebreak(provider: LLMProvider) -> Callable[[str, str], str | None]:
    """Build a Haiku-backed tie-break callable from a generation provider.

    Called ONLY for candidates the local judge's ambiguous band routes here
    (acceptance #7: no Haiku call on clear cases). Returns `"relevant"` /
    `"irrelevant"` on a clean parse, or `None` on ANY provider/parse
    failure — `None` means "no override," so the caller's fallback is the
    local judge's own provisional label at that position (fail-soft toward
    the signal that already exists, mirroring `_make_haiku_classifier`'s
    fail-toward-keeping-content posture).
    """

    def _tiebreak(query: str, document: str) -> str | None:
        user = f"QUERY: {query[:400]}\nCANDIDATE: {document[:400]}"
        try:
            raw = provider.generate(user, system=_HAIKU_TIEBREAK_PROMPT)
            data = json.loads(_extract_json(raw))
            label = str(data.get("label", ""))
            if label not in ("relevant", "irrelevant"):
                return None
            return label
        except Exception:  # noqa: BLE001
            logger.warning("calibration judge: Haiku tie-break failed; keeping local judge's label")
            return None

    return _tiebreak


# ---------------------------------------------------------------------------
# Orchestration — the ONE entry point the daily calibration tick calls.
#
# F2c (durable note, spec §6): Haiku is the effective relevance ORACLE this
# judge is being converged toward. The tie-break below already treats a
# Haiku verdict as ground truth over the local judge's own provisional
# label at ambiguous positions, and F2c's later weekly self-tune (knob-refit
# / LoRA / full fine-tune, not built in this increment) trains the local
# judge's score-to-label mapping toward the accumulated Haiku decisions
# logged here. If a relevance-quality problem shows up downstream later,
# this is the place to look first: what the judge converges toward is
# Haiku's own labeling behavior, not some independently-verified ground
# truth, so a systematic Haiku bias would propagate into the judge rather
# than being caught by it.
# ---------------------------------------------------------------------------


def label_calibration_sample(
    store: MemoryStore,
    *,
    provider: LLMProvider | None = None,
    judge: RelevanceJudgeProvider | None = None,
    sample_rows: int | None = None,
) -> int:
    """Label a SAMPLE of unlabeled `calibration_log` rows: the local judge
    (`judge`, or the real `build_judge_provider()` if not injected) scores
    every candidate in each sampled row; ambiguous-band candidates get a
    Haiku tie-break via `provider` (skipped entirely if `provider` is None
    — degrades to local-judge-only labeling, mirroring `consolidation.
    run_consolidation`'s degrade-to-`_promote_all_classifier` when no
    provider is available).

    Returns the number of ROWS labeled this call (0 if there was nothing
    to label, or if judge construction itself failed).

    FAULT ISOLATION (spec: "must not crash the tick or the bridge"):
      - judge construction failure -> logged, returns 0, no rows touched.
      - one row's failure -> logged, that row is skipped (stays unlabeled,
        eligible for a later tick's sample) — does not stop the pass.
      - one candidate's failure within an otherwise-fine row -> logged,
        that position gets the "error" label sentinel (distinct from
        "relevant"/"irrelevant"/"unknown" — never silently mislabeled as
        either), the rest of the row's candidates still get labeled.
      - a Haiku failure -> handled inside `_make_haiku_tiebreak` itself
        (returns None, not an exception) — never reaches this function's
        try/except at all.
    """
    if sample_rows is None:
        sample_rows = tunables.get_tunable("calibration.judge_sample_rows", CALIBRATION_SAMPLE_ROWS)

    rows = store.sample_unlabeled_calibration_rows(limit=int(sample_rows))
    if not rows:
        return 0

    if judge is None:
        try:
            judge = build_judge_provider()
        except Exception:  # noqa: BLE001 — torch missing, download failed, etc.
            logger.exception(
                "calibration judge: failed to construct the local judge provider — skipping this pass"
            )
            return 0

    haiku_tiebreak = _make_haiku_tiebreak(provider) if provider is not None else None

    labeled = 0
    for row in rows:
        try:
            query = row["query"]
            candidate_ids: list[str] = row["candidate_ids"]
            local_labels: list[str] = []
            haiku_labels: list[str | None] = []
            # F2c inc1 (data foundation only, spec §3 Addition A): the raw
            # judge score/logit, accumulated alongside the derived labels in
            # this same loop and positionally aligned with `candidate_ids`
            # (same convention as `local_labels`/`haiku_labels`) — a `None`
            # entry marks a position this pass never scored (the
            # "unknown"/"error" sentinels below), distinct from a real 0.0
            # score. Persisted via `write_calibration_labels` so F2c's later
            # knob-refit (not built here) has the raw score to fit a
            # threshold/Platt mapping over, instead of only the label the
            # score was already collapsed into.
            raw_scores: list[float | None] = []
            for cid in candidate_ids:
                try:
                    mem = store.get(cid, bump=False)
                    if mem is None:
                        # Candidate no longer exists (deleted/forgotten since
                        # it was logged) — not judgeable; distinct sentinel,
                        # never silently coerced into relevant/irrelevant.
                        local_labels.append("unknown")
                        haiku_labels.append(None)
                        raw_scores.append(None)
                        continue
                    raw_score = judge.score(query, mem.content)
                    provisional, is_ambiguous = label_for_score(raw_score)
                    local_labels.append(provisional)
                    raw_scores.append(float(raw_score))
                    if is_ambiguous and haiku_tiebreak is not None:
                        haiku_labels.append(haiku_tiebreak(query, mem.content))
                    else:
                        haiku_labels.append(None)
                except Exception:  # noqa: BLE001 — one candidate must not sink the row
                    logger.exception(
                        "calibration judge: candidate %r in row id=%s failed; marking error",
                        cid, row.get("id"),
                    )
                    local_labels.append("error")
                    haiku_labels.append(None)
                    raw_scores.append(None)
            store.write_calibration_labels(
                row["id"], local_labels, haiku_labels, local_judge_raw_score=raw_scores
            )
            labeled += 1
        except Exception:  # noqa: BLE001 — one row must not sink the whole pass
            logger.exception(
                "calibration judge: row id=%s failed; leaving unlabeled for a later tick",
                row.get("id"),
            )
    return labeled
