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
from typing import TYPE_CHECKING, Any

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
#       one judged PAIR per candidate in that row's candidate list: a
#       reranked row carries every real candidate of its per-message rerank
#       width (at least `RERANK_MIN_REAL_CANDIDATES` = 5, S5, at most
#       `relevance.CANDIDATE_POOL`=50); a cosine-scale row (name-recall fix
#       R2, S60) carries up to 9. 100 rows therefore yield at least 500
#       pairs from reranked rows, comfortably past "hundreds of pairs" by
#       inc7's floor-fit time even at the minimum width. The two scales are
#       sampled separately (S25: `label_calibration_sample`), so each gets
#       up to this many rows and neither dilutes the other's fit.
#   (ii) UPPER bound — bounding the local judge's daily compute on the
#       no-AVX2 potato baseline (spec Section 7): 100 rows a day per scale
#       (at most 200 across the reranked and cosine scales, whose rows
#       carry at most 50 and 9 candidates) through a cross-encoder is a
#       bounded, once-daily idle cost, not the thousands of raw per-turn
#       rows actually logged.
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
    raw_score: float,
    *,
    band_half_width: float | None = None,
    slope: float | None = None,
    intercept: float | None = None,
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

    `slope`/`intercept` (F2c inc3, spec §5 "knob-refit"): an ABSENT-SAFE
    way to apply a persona's FITTED Platt calibration
    (`judge_selftune.fit_platt_knob`, persisted via
    `MemoryStore.write_judge_knob_calibration`) in place of the fixed
    sigmoid-0.5 default this function has always used. `p` becomes
    `sigmoid(slope * raw_score + intercept)` instead of `sigmoid(raw_score)`
    — passing `slope=1.0, intercept=0.0` (or leaving both `None`, the
    default) reproduces today's exact fixed behavior bit-for-bit. Either
    argument being `None` falls back to that default for just the missing
    one, rather than raising — a caller is never required to supply both.
    The band comparison (`abs(p - 0.5) < band_half_width`) is unchanged:
    the ambiguous band is defined on the (possibly recalibrated)
    probability, not on the raw score directly, so a fitted mapping that
    shifts the decision boundary also shifts where the ambiguous band
    sits, which is the intended effect (the band should track wherever
    the judge's OWN cutoff currently is).

    F2c INC4a wires the LOAD side: `label_calibration_sample` below now
    reads this persona's persisted knob via `store.get_judge_knob_
    calibration(judge.model_id())` once per call and passes it through to
    every `label_for_score` call in that pass — so the LIVE judge pass
    genuinely uses a persona's own fitted knob once inc3's weekly tick has
    written one, and stays byte-for-byte unchanged (both args `None`) for
    any persona that hasn't had a knob-refit complete yet.
    """
    if band_half_width is None:
        band_half_width = tunables.get_tunable(
            "calibration.judge_ambiguous_band_half_width", AMBIGUOUS_BAND_HALF_WIDTH
        )
    effective_slope = slope if slope is not None else 1.0
    effective_intercept = intercept if intercept is not None else 0.0
    p = _sigmoid(effective_slope * raw_score + effective_intercept)
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

    def knob_key(self) -> str:
        """The `judge_knob_calibration` key of the knob fit on THIS judge's
        scores (F2c inc9). The base judge (and every test fake) uses its
        `model_id()`; `FullModelJudge` adds its checkpoint, so a knob is only
        ever read for the one model object that produces the scores."""
        return self.model_id()


def judge_knob_key(model_id: str, checkpoint: str | Path | None) -> str:
    """The single definition of a knob key (F2c inc9, spec §5): the plain
    `model_id` when no tuned checkpoint serves, else
    `"<model_id>@<checkpoint dir name>"`. Checkpoint dir names are one-use
    uuids (`judge_lora.staged_adapter_path`) and `@` cannot occur in a Hugging
    Face repo id, so a key matches only the checkpoint its knob was fit on.
    Pure string work: no filesystem access, no torch."""
    if checkpoint is None:
        return model_id
    return f"{model_id}@{Path(checkpoint).name}"


def _hf_cached(model_id: str, cache_dir: str | Path | None) -> bool:
    """True iff `model_id` is fully present in the local Hugging Face cache
    at `cache_dir`, checked WITHOUT ever touching the network (S2/S19,
    inc5). Uses `huggingface_hub.snapshot_download(..., local_files_only=
    True)` — the same resolver `CrossEncoder` itself uses internally — so
    "cached" here means exactly what a real offline load would need, not an
    approximation of it (e.g. a bare directory-listing guess). A repo
    missing entirely -> `LocalEntryNotFoundError` -> False, never raises.
    `HFValidationError` (`model_id` isn't shaped like `namespace/repo_name`
    — a non-existent local path reaching here, since `offline_load_kwargs`
    routes any REAL local directory around this function entirely) is
    treated the same as "not cached": False, so the caller falls back to
    today's online-load attempt rather than crashing the tick on a
    malformed id.

    Round-3 CI follow-up: `snapshot_download(local_files_only=True)`
    succeeding only proves every needed file's symlink EXISTS locally, not
    that it finished downloading — a process killed mid-transfer (the CI
    diagnosis: a 300s-timeout-killed subprocess mid-download) can leave a
    genuinely truncated blob at the correct symlinked path, which this
    resolver call alone does NOT detect (confirmed empirically: a copy of
    this repo's real cache with `model.safetensors` truncated to 10MB still
    resolves as "cached" here without the check below). Per spec S19, a
    partially downloaded model is NOT "in the cache" -- so `_snapshot_
    complete` additionally verifies every file the load needs is present at
    its FULL recorded size, offline, before this returns True.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError

    try:
        snapshot_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, HFValidationError):
        return False
    return _snapshot_complete(Path(snapshot_path))


def _snapshot_complete(snapshot_dir: Path) -> bool:
    """Offline completeness check for an already-resolved HF cache snapshot
    directory (round-3 CI follow-up, S19): every file the resolved revision
    lists must exist (through the snapshot dir's symlinks into `blobs/`)
    at exactly its recorded byte size.

    Ground truth is `<repo_root>/trees/<revision>.json` -- huggingface_hub's
    OWN locally-cached git-tree manifest for this exact revision (present
    for every repo this project's pinned `huggingface-hub==1.30.0` downloads;
    confirmed by inspecting all three models this project currently caches).
    This is the same manifest `snapshot_download` itself consults to resolve
    the snapshot in the first place, so checking against it re-verifies
    EXACTLY the file set `_hf_cached`'s caller already expects, never a
    stricter or looser scope (no risk of flagging a file the real load never
    needed, since the manifest lists exactly what a full, unrestricted
    `snapshot_download` — what this module calls — resolves).

    Deliberately SIZE, not a full re-hash: a truncated/killed transfer is
    caught by size alone (this follow-up's diagnosed failure mode), and
    re-hashing a ~2.2GB weights file on every judge build (called once per
    calibration/self-tune tick, so not hot-path, but still non-trivial CPU)
    is a real cost for a failure mode size already catches. A file whose
    bytes are wrong but whose LENGTH happens to match (e.g. this project's
    documented VM block-corruption failure mode) is NOT caught here --
    `TorchCrossEncoderJudge.__init__`'s offline-load-raises retry is the
    second, independent layer of defense for that residual case (round-3
    S19 note there).

    If the manifest itself is missing or unreadable, this falls back to
    trusting `snapshot_download`'s own resolution (True) rather than
    forcing an online re-fetch. Round-3 cold code red-team (agentId
    a1aba2a010dee57f6, MAJOR): a `trees/<revision>.json` manifest is written
    only as a side effect of an online download by a tree-cache-aware
    huggingface_hub client -- a cache populated before this project pinned
    `huggingface-hub==1.30.0` (or via `local_dir` mode, which keeps its own
    tree cache at a different path) would never have one, and returning
    False there would silently reclassify an already-good, already-accepted
    cache as "not cached" on every persona that upgrades into this fix,
    forcing an unwanted ~2.2GB re-download (or a silent tick no-op if
    offline) with no user-visible signal. Falling back to True for a
    missing manifest is exactly today's PRE-round-3 behavior for that case
    (no new regression introduced for it) while still gaining the size
    check for every fresh download going forward, which always gets one
    (confirmed: every model this project currently caches has a `trees/`
    entry). Logged once per occurrence so a persona silently missing this
    verification is still observable.
    """
    revision = snapshot_dir.name
    repo_root = snapshot_dir.parent.parent
    tree_path = repo_root / "trees" / f"{revision}.json"
    try:
        manifest = json.loads(tree_path.read_text())
        files: dict[str, dict[str, Any]] = manifest["files"]
    except (OSError, ValueError, KeyError):
        logger.warning(
            "no readable trees/%s.json manifest under %s -- cannot size-verify this "
            "snapshot's completeness, falling back to trusting it (pre-round-3 behavior)",
            revision,
            repo_root,
        )
        return True

    for rel_path, meta in files.items():
        expected_size = meta.get("lfs_size", meta.get("size"))
        if expected_size is None:
            return False
        try:
            actual_size = (snapshot_dir / rel_path).stat().st_size
        except OSError:
            return False  # missing file, or a broken symlink (blob deleted)
        if actual_size != expected_size:
            return False  # truncated/corrupted-length blob
    return True


def offline_load_kwargs(model_id_or_path: str, cache_dir: str | Path | None) -> dict[str, Any]:
    """The kwargs to splat into a `CrossEncoder(...)` construction at a judge
    load site (S2/S19, inc5) so a cached model makes ZERO Hugging Face
    requests — not merely "no full download", but no HEAD/GET at all (C3a).

    A LOCAL checkpoint directory (a persona's own tuned judge,
    `judge_full_ft.load_full_scorer` / `judge_full_ft.build_full_ft_retrain_
    fn` / `judge_lora.build_lora_retrain_fn`'s `start_model_path` when it
    names a prior week's saved checkpoint) is already on disk — no Hub
    resolution is possible or needed, so it goes straight to offline mode
    with no network round-trip to decide. A Hugging Face repo id (the BASE
    judge, or a weight-retrain's `start_model_path` before any checkpoint
    exists yet) is checked via `_hf_cached`: present -> offline mode (no HF
    request at all); absent -> `{}` (today's online load, C3b) — never a
    process-global `HF_HUB_OFFLINE` mutation, which is read at import time
    and would leak to other threads' unrelated loads (2-plan §2).

    "Offline mode" is two kwargs, not one — empirically confirmed (this
    module, manual trace) against the installed transformers/sentence-
    transformers/huggingface_hub versions:
      - `local_files_only=True`: covers the base config/tokenizer/weights
        resolution (`AutoConfig`/`AutoModel.from_pretrained`'s own `hub_
        kwargs`).
      - `model_kwargs={"adapter_kwargs": {"local_files_only": True}}`:
        covers a SEPARATE, independent check `transformers.models.auto.
        auto_factory._BaseAutoModelClass.from_pretrained` runs before that —
        `find_adapter_config_file(...)` probing for a PEFT `adapter_config.
        json` — which reads its own `local_files_only` from `adapter_kwargs`
        ONLY, not from the top-level `local_files_only` param at all (a
        transformers library quirk, not a bge-reranker-v2-m3 specifics):
        without this second kwarg, a fully-cached, `local_files_only=True`
        load still issues a real HEAD request for `adapter_config.json` and
        falls back to the cache only after that request errors/times out —
        exactly the residual network touch C3a forbids. Confirmed by socket-
        level connect-call counting: 0 with both kwargs, 1+ (with retries)
        with `local_files_only=True` alone.
    """
    if Path(model_id_or_path).is_dir():
        offline = True
    else:
        offline = _hf_cached(model_id_or_path, cache_dir)
    if not offline:
        return {}
    return {"local_files_only": True, "model_kwargs": {"adapter_kwargs": {"local_files_only": True}}}


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
        # S81 owner ruling (Roy, RAM-spike-fix ledger, "CPU everywhere
        # (Recommended)"): every judge construction site runs on CPU on
        # every platform, never MPS. Without an explicit `device=`,
        # sentence_transformers.util.get_device_name() picks 'mps' whenever
        # torch.backends.mps.is_available() is true -- which a real macOS
        # install's torch build reports, and release_judge() has no
        # MPS-specific release call, so an MPS-resident judge risked a
        # low-memory Mac hitting the same OOM CI saw under MPS's smaller
        # memory budget.
        load_kwargs = offline_load_kwargs(model_id, cache_dir)
        try:
            self._model = CrossEncoder(
                model_id, cache_folder=str(cache_dir), device="cpu", **load_kwargs
            )
        except Exception:
            # Round-3 CI follow-up (S19): `_hf_cached`'s size check (see its
            # docstring) catches a truncated/killed-mid-download blob, but
            # NOT a same-size-wrong-bytes corruption (this project's
            # documented VM block-corruption failure mode) or any other way
            # an offline load can fail despite passing that check. `load_
            # kwargs` non-empty means we DID attempt the offline path (S2/
            # S19: only reached when `_hf_cached` said "cached"); retry
            # ONCE without it so huggingface_hub's real network downloader
            # (which DOES hash-verify on transfer, unlike a local_files_
            # only resolution) can detect and re-fetch whatever is actually
            # bad. If `load_kwargs` was already empty, we were already on
            # the online path — no second online attempt to make; let the
            # original exception propagate rather than silently retrying
            # the identical call.
            if not load_kwargs:
                raise
            logger.warning(
                "offline load of judge %r failed despite _hf_cached reporting it complete "
                "-- retrying online once (S19: an offline load that raises is treated as "
                "not really cached)",
                model_id,
                exc_info=True,
            )
            self._model = CrossEncoder(model_id, cache_folder=str(cache_dir), device="cpu")
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


class FullModelJudge(RelevanceJudgeProvider):
    """A per-persona TUNED judge backed by the persona's saved plain
    checkpoint (F2c inc6; since inc7 the ONLY tuned-judge form, spec §5 "One
    stored tuned model per persona": a full fine-tune or a LoRA week merged in
    memory — never an adapter). Wraps `judge_full_ft.load_full_scorer`'s
    `(query, doc) -> raw float` callable (a PLAIN
    `sentence_transformers.CrossEncoder` reload) as a
    `RelevanceJudgeProvider`, so the daily calibration tick serves a persona's
    tuned judge exactly where it would otherwise serve the base judge, and
    `label_for_score` applies the persona's knob on top exactly as for the
    base judge.

    `model_id()` returns the BASE model id (the model family). The knob,
    though, is bound to THIS checkpoint (F2c inc9, spec §5): `knob_key()` is
    `judge_knob_key(base id, checkpoint dir)`, the key the weekly tick writes
    this checkpoint's knob under before swapping it in, so
    `label_calibration_sample` can never apply a knob fit on a different
    checkpoint. Torch is imported lazily inside `load_full_scorer` (I6), never
    at import time.
    """

    def __init__(
        self, base_model_id: str, full_dir: str, *, cache_dir: str | None = None
    ) -> None:
        from brain.memory.judge_full_ft import load_full_scorer

        self._model_id = base_model_id
        self._full_dir = full_dir
        self._scorer = load_full_scorer(full_dir, cache_dir=cache_dir)

    def score(self, query: str, document: str) -> float:
        return float(self._scorer((query, document)))

    def model_id(self) -> str:
        return self._model_id

    def knob_key(self) -> str:
        return judge_knob_key(self._model_id, self._full_dir)


# Process-wide provider cache keyed by model_id (mirrors reranker.py's
# _provider_cache). Kept at module scope so `_reset_judge_provider_cache`
# (test-only) can reach it and so monkeypatching the *function* fully
# controls behavior. ONLY the shared BASE judge is cached here — a
# per-persona tuned checkpoint judge is built fresh per call and never cached
# (F2c inc5b-2/inc7: a newly-accepted checkpoint every week would otherwise
# grow the cache unbounded; it is built at most once per daily calibration
# tick, offline, so the once/day load cost is acceptable).
_provider_cache: dict[str, RelevanceJudgeProvider] = {}
_provider_cache_lock = threading.Lock()


def build_judge_provider(full_model_dir: str | None = None) -> RelevanceJudgeProvider:
    """The production judge provider. With NO `full_model_dir`: the shared
    base `TorchCrossEncoderJudge` pinned to `model_tier.TIER_RELEVANCE_JUDGE`'s
    model id, caching the model file in the shared `get_cache_dir()` (one
    download across every persona on the box) and PROCESS-WIDE cached by
    model_id (double-checked locking) — unchanged from f2a-inc6.

    With `full_model_dir` (F2c inc6/inc7): a per-persona `FullModelJudge`
    serving that persona's tuned plain checkpoint, built FRESH and NOT cached
    (see the `_provider_cache` note above). Since inc7 a persona has exactly one
    tuned judge and it is always a plain checkpoint, so there is no adapter
    argument and no precedence between stores.

    The caller (`supervisor._run_calibration_tick`) resolves the persona's
    current checkpoint (`judge_lora.resolve_current_checkpoint`) and passes it
    here; an absent / unresolvable pointer means `None` and the base judge
    (I9, absent → base).

    TEST ISOLATION: tests must monkeypatch this function directly (mirrors
    `reranker.build_reranker_provider`'s test-fixture convention) rather
    than relying on the cache alone — `_reset_judge_provider_cache` is the
    test-only reset hook for the cache itself.
    """
    from brain.bridge.model_tier import TIER_RELEVANCE_JUDGE, model_for_tier
    from brain.paths import get_cache_dir

    model_id = model_for_tier(TIER_RELEVANCE_JUDGE)

    if full_model_dir is not None:
        return FullModelJudge(model_id, full_model_dir, cache_dir=str(get_cache_dir()))

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


def release_judge() -> None:
    """Release the judge's RAM after a calibration or self-tune job FINISHES
    (S11/S27, inc5; the PAUSE arm is INC-10). Called in the `finally` of
    `supervisor._run_calibration_tick`'s judge-labeling step and of
    `judge_selftune._run_judge_selftune_tick` — unconditionally, whether or
    not that tick actually built a judge this time (a no-op is cheap; a
    missed release is a RAM leak, so every finish path calls this rather
    than only the ones known to have built something).

    Drops BOTH kinds of judge reference this module can hold at the end of a
    tick:
      - the cached shared BASE judge (`_provider_cache`, `TorchCrossEncoder
        Judge`) — popped under `_provider_cache_lock` so a concurrent
        `build_judge_provider()` call never observes a half-cleared cache;
      - any per-persona `FullModelJudge` / weight-retrain scratch model the
        caller built and returned from this call — those are NEVER cached
        (module docstring above `_provider_cache`), so ordinary CPython
        refcounting already drops them once the caller's own local
        variables go out of scope; `gc.collect()` below is what reclaims
        them if a reference cycle (torch's autograd graph, a bound closure)
        kept one alive past that point.

    Then, on every platform, `gc.collect()` (reclaims any of the above still
    alive only via a cycle); on Linux, `ctypes.CDLL("libc.so.6").
    malloc_trim(0)` — glibc's allocator does not always return freed pages
    to the OS on `free()` alone (O7: Linux RSS only drops after gc+trim),
    so this is the step that actually shows up in `psutil`'s RSS reading.
    macOS gets its analogue, `malloc_zone_pressure_relief(NULL, 0)` from
    libSystem (guarded by `sys.platform == "darwin"`, fail-soft the same
    way); Windows needs none (its heap returns freed pages on its own — the
    C2 RSS test passes there without one).
    The Linux call is guarded both by `sys.platform.startswith("linux")`
    (never attempted on macOS/Windows, which have no `libc.so.6` and no
    `malloc_trim` — I13) and by a try/except around the `CDLL`/symbol lookup itself (a musl-based
    Linux, or a hardened glibc build missing the symbol, would otherwise
    raise here; logged once, not re-raised, since a failed trim only means
    RSS drops less promptly, not that anything is wrong).
    """
    import gc
    import sys

    with _provider_cache_lock:
        _provider_cache.clear()

    gc.collect()

    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            logger.warning("release_judge: malloc_trim(0) unavailable on this libc — RSS may drop later")
    elif sys.platform == "darwin":
        # macOS analogue of the Linux malloc_trim(0) above: libSystem's
        # malloc zones keep freed pages resident after free(), so after the
        # judge's tensors are dropped RSS can sit well above the pre-load
        # level (CI measured 26.3% retained vs the 25% C2 bound on macos-14;
        # the calibration arm retained ~20%). `malloc_zone_pressure_relief(
        # NULL, 0)` asks every zone to return whatever it can to the OS.
        # Fail-soft like the Linux branch: any failure to find or call it is
        # logged and swallowed, since it only affects how promptly RSS drops.
        try:
            import ctypes

            relief = ctypes.CDLL("/usr/lib/libSystem.B.dylib").malloc_zone_pressure_relief
            relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            relief.restype = ctypes.c_size_t
            relief(None, 0)
        except Exception:  # ctypes can raise OSError / AttributeError / ArgumentError
            logger.warning("release_judge: malloc_zone_pressure_relief unavailable on this macOS — RSS may drop later")


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
    full_model_dir: str | None = None,
    should_pause: Callable[[], bool] | None = None,
    progress_out: dict[str, bool] | None = None,
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

    F2c INC4a (spec §5 "where the tuned judge loads from"): before scoring,
    reads THIS PERSONA's persisted knob-refit params for the judge's knob
    key (`judge.knob_key()`: its model id, plus its checkpoint for a tuned
    judge, F2c inc9) via `store.get_judge_knob_calibration` — `store` is always the
    persona-scoped `MemoryStore` the caller (`_run_calibration_tick`)
    constructed from that persona's OWN `memories.db` (I1: one store per
    persona is the isolation boundary, no persona-scoping column needed,
    same as the table itself), so this is a per-persona load by
    construction (AC11) — no cross-persona bleed is possible without
    passing another persona's store in. ABSENT-SAFE: no persisted row for
    this judge model id (fresh install, or a persona whose weekly tick has
    never completed a knob-refit) -> `slope`/`intercept` both stay `None`,
    and `label_for_score` below falls back to its fixed sigmoid-0.5/
    band-0.05 behavior, byte-identical to pre-inc4a labeling. Read ONCE per
    call (the mapping is constant for the whole sampled batch), not
    per-candidate.

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

    ``should_pause`` (ram-spike-fix INC-10, S14/S31/S41/S65): checked after
    each row is written, before the next one starts — the S32 table's item
    unit for daily calibration is "one calibration_log row". When it fires
    with rows still remaining, the loop stops there; `progress_out["paused"]`
    (if a dict was passed) is set True so the caller can skip floor
    derivation and report the tick as paused rather than completed. The
    "already-labeled rows not re-labeled" resume guarantee (C8) needs no new
    cursor: `store.sample_unlabeled_calibration_rows` only ever samples
    `local_judge_label IS NULL` rows, so a labeled row is never re-sampled.
    """
    if sample_rows is None:
        sample_rows = tunables.get_tunable("calibration.judge_sample_rows", CALIBRATION_SAMPLE_ROWS)

    # Name-recall fix R2 (S25): the two score scales are labeled separately,
    # each with its own `sample_rows` budget, so a busy reranked scale never
    # starves the cosine scale's daily sample (or the reverse): the cosine
    # floor is fit from cosine-scale rows only. (Labels themselves are scale-
    # independent: the judge scores query x document text, not the logged
    # score.) Rows of any non-cosine scale, legacy 'raw' included, stay in the
    # first sample exactly as before.
    rows = store.sample_unlabeled_calibration_rows(limit=int(sample_rows), cosine=False)
    rows += store.sample_unlabeled_calibration_rows(limit=int(sample_rows), cosine=True)
    if not rows:
        return 0

    if judge is None:
        try:
            # F2c inc7: the persona's current tuned checkpoint resolved by
            # `_run_calibration_tick` (`full_model_dir`, a plain CrossEncoder
            # checkpoint), or None → the base judge (I9). Built LAZILY here —
            # only when there are actually rows to label — so a tick with
            # nothing to do never loads a torch model.
            judge = build_judge_provider(full_model_dir=full_model_dir)
        except Exception:  # noqa: BLE001 — torch missing, download failed, etc.
            logger.exception(
                "calibration judge: failed to construct the local judge provider — skipping this pass"
            )
            return 0

    # F2c inc4a: this persona's fitted Platt knob for THIS judge, or
    # (None, None) if absent — see the docstring above. inc9: keyed by
    # `judge.knob_key()` (the base id, or base id @ the served checkpoint), so
    # the knob read is always the one fit on the model object scoring here.
    knob = store.get_judge_knob_calibration(judge.knob_key())
    knob_slope: float | None = knob["slope"] if knob is not None else None
    knob_intercept: float | None = knob["intercept"] if knob is not None else None

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
                    provisional, is_ambiguous = label_for_score(
                        raw_score, slope=knob_slope, intercept=knob_intercept
                    )
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
        if should_pause is not None and row is not rows[-1] and should_pause():
            logger.info("calibration judge: pausing between rows for chat (INC-10)")
            if progress_out is not None:
                progress_out["paused"] = True
            break
    return labeled
