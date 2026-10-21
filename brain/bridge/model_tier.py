"""Which model backs each LLM call: one tunables line per job type, and the ONE accessor that reads it.

Every model a Claude call runs on is chosen here and nowhere else. A call site declares the JOB it is doing (a "tier"
name below, which has always meant job type) and asks ``model_for_tier(tier)`` (or builds a provider with
``build_tier_provider``); it never holds a model id. There are no size tiers: each job type has its own key in the
user's tunables file (``$KINDLED_HOME/tunables.json``, section ``overrides``), registered once on one commented line
of ``_CLAUDE_JOBS`` below. To change the model for a job, change that one line (the shipped default) or add one line to
the user's overrides (this install only). Both are read on EVERY call, so a change reaches a running bridge with no
restart.

Pinning a default is one line: edit the id on the job's ``tunables.register`` line. The words the UI offers (Sonnet,
Opus, Haiku) are vocabulary for the picker, ``nell model set`` and the setup wizard (``resolve_choice``); the tunables
file written by those paths holds ids, never the words (a word typed into the file by hand is accepted and handed to
the CLI as the CLI's own alias, not as the pinned id), and a word that means the job's own default REMOVES the override instead of writing
a copy, so a user who picked the word follows every later change of the shipped default (``set_job_models``).

A configured id is checked against the model grammar (``is_valid_model``); a value that fails it is ignored with one
warning and the shipped default runs, and the bad text stays in the file untouched. A configured id that Claude
REJECTED (it may not exist, or the account may not have access) makes that job run its shipped default from the next
call on, with the rejection notice kept (``brain.bridge.model_error``); the record is trusted only while it is newer
than the last write to the tunables file, so editing the file ends the fallback without a restart. Limits: when the
rejected id IS the shipped default (an old ``claude`` that does not know it) there is nothing to fall back to, and
exactly one call per process fails before the fallback engages.

The embedder, reranker and relevance judge are not Claude models: their ids are plain constants below, resolved by
the same ``model_for_tier`` through ``LOCAL_MODEL_IDS``.

Generalizes the pattern already used by ``build_compaction_provider`` (``brain/chat/compaction.py``) and
``build_self_model_provider`` (``brain/self_model/articulate.py``): resolve the provider *kind* from
``persona_config.json``, then build the provider for the job.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from brain import tunables
from brain.utils.log_throttle import warn_throttled

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The model grammar: what a model setting may look like
# ---------------------------------------------------------------------------
# The three words the picker, `nell model set` and the wizard accept as a CHOICE (resolved by resolve_choice).
KNOWN_MODELS = frozenset({"sonnet", "opus", "haiku"})
# A full model id, stored and passed to `claude --model` as written. Matched with fullmatch, never `$`: `$` would
# accept a trailing newline. The picker in app/src/lib/modelId.ts uses the same pattern and length cap; keep the two
# in sync (a shared table of accept/reject strings is asserted on both sides).
MODEL_ID_PATTERN = re.compile(r"claude-[a-z0-9.-]+")
# Longest full id accepted on EVERY write path (tunables read, the picker endpoint, the CLI). Real ids are well under
# it; the cap keeps an absurd value off the CLI's argv, out of the usage log and out of the UI, and keeps the CLI's
# rejection sentence (cut at 200 characters by the provider) parseable.
MODEL_ID_MAX_LENGTH = 64
MODEL_ID_RULE = (
    "use sonnet, opus or haiku, or a full id that starts with claude- and has only lowercase letters, "
    f"digits, dots and dashes, at most {MODEL_ID_MAX_LENGTH} characters"
)


def is_valid_model(value: object) -> bool:
    """True for one of the three words (sonnet / opus / haiku) or a full ``claude-...`` model id."""
    if not isinstance(value, str):
        return False
    if value in KNOWN_MODELS:
        return True
    return len(value) <= MODEL_ID_MAX_LENGTH and MODEL_ID_PATTERN.fullmatch(value) is not None


def unsupported_display(raw: object) -> str:
    """A rejected setting's text made safe to show: control characters removed, cut at the id length cap."""
    clean = "".join(ch for ch in str(raw) if ch.isprintable())
    if not clean:  # nothing printable: say what it was rather than show an empty value
        return "(non-printing characters)" if str(raw) else "(empty)"
    return clean if len(clean) <= MODEL_ID_MAX_LENGTH else clean[:MODEL_ID_MAX_LENGTH] + "…"


# ---------------------------------------------------------------------------
# Job types (the "tier" names every model-selection call site declares)
# ---------------------------------------------------------------------------
TIER_INTERACTIVE_CHAT = "interactive-chat"
TIER_ATTUNEMENT_DETECTOR = "attunement-detector"
TIER_COMPACTION = "compaction"
TIER_SELF_MODEL_ARTICULATE = "self-model-articulate"
TIER_BACKGROUND_CLASSIFIER = "background-classifier"
TIER_BACKGROUND_GENERATIVE = "background-generative"
TIER_BACKGROUND_HOUSEKEEPING = "background-housekeeping"

# ---------------------------------------------------------------------------
# The Claude jobs: ONE tunables line each. (job type) -> (tunables key, shipped default model)
# ---------------------------------------------------------------------------
# This table is the DEFINITION of the shipped defaults, read by model_for_tier on every call. Nothing else in the code
# holds a model id. The comment on each line says what the job IS.
_CLAUDE_JOBS: dict[str, tuple[str, str]] = {
    TIER_INTERACTIVE_CHAT: ("model.interactive_chat", tunables.register("model.interactive_chat", "claude-sonnet-5")),  # the live chat turn: every reply the kindled gives in conversation
    TIER_BACKGROUND_GENERATIVE: ("model.background_generative", tunables.register("model.background_generative", "claude-sonnet-5")),  # background writing in the kindled's own voice: dreams, reflexes, research, soul review, voice reflection and the maker
    TIER_BACKGROUND_CLASSIFIER: ("model.background_classifier", tunables.register("model.background_classifier", "haiku")),  # cheap background judgments that tag or classify: emotion tagging, memory consolidation tie-breaks, vocabulary repair, creative-DNA crystallization, tool-loop monologue extraction
    TIER_BACKGROUND_HOUSEKEEPING: ("model.background_housekeeping", tunables.register("model.background_housekeeping", "haiku")),  # background upkeep that never speaks to the user: closing sessions, interest sweeps, finalize ticks, initiate reviews, daemon snapshots, draft fragments
    TIER_COMPACTION: ("model.compaction", tunables.register("model.compaction", "haiku")),  # summarising old chat history into a short digest when a session is rolled over
    TIER_SELF_MODEL_ARTICULATE: ("model.self_model_articulate", tunables.register("model.self_model_articulate", "haiku")),  # putting the kindled's measured self-model into plain sentences
    TIER_ATTUNEMENT_DETECTOR: ("model.attunement_detector", tunables.register("model.attunement_detector", "claude-haiku-4-5")),  # reading recent turns for the user's current state and recurring patterns (the attunement detector); stays on Haiku 4.5 (not the haiku alias, which moves to newer Haiku generations); the dateless id follows the newest 4.5 snapshot
}
CLAUDE_TIERS: tuple[str, ...] = tuple(_CLAUDE_JOBS)

# What the picker's, `nell model set`'s and the wizard's "Opus" choice means. No job runs Opus by default.
OPUS_CHOICE = "claude-opus-5"


# ---------------------------------------------------------------------------
# Non-Claude models (O1: not part of the one-line-per-job rule)
# ---------------------------------------------------------------------------
# Mini-model swaps (embedder, reranker, relevance-judge): model ids can be
# swapped by changing the constant here, but ONLY as a same-interface drop-in.
# A replacement with a different task-shape or I/O contract (for example, an
# NLI-style relevance judge instead of the cross-encoder judge, which needs a
# query-to-hypothesis reformulation) requires integration code changes, not just
# this config swap. Exception: the embedder's vector DIMENSION is handled
# dynamically (the one-touch embedding-dimension work), so a different-dimension
# embedder IS a clean swap. The caveat applies to task-SHAPE, not dimension.
# For the F2a relevance judge specifically: the retrieval floor auto-re-derives
# on the next daily calibration after a same-shape judge swap, so no manual
# recalibration is needed. UNLIKE the ONNX-loaded embedder/reranker above,
# the judge (MODEL_RELEVANCE_JUDGE below) loads via torch/sentence-
# transformers (#250 F2a inc6, Roy 2026-09-18 "torch it is then") — so it
# swaps to ANY cross-encoder id, PyTorch-only or ONNX alike, whereas the
# ONNX-loaded embedder/reranker can only swap to models that ship a usable
# ONNX export.

# MODEL_EMBEDDING is NOT a Claude model — it's a local ONNX embedding model id
# (fastembed/HuggingFace naming), the standing convention (per the local
# semantic-retrieval spec) for embeddings AND any future minimodel (#228):
# minimodel selection lives here alongside the Claude tiers, one constant to
# repoint every embedding call site. See TIER_EMBEDDING below for why it
# resolves via a dedicated accessor rather than build_tier_provider.
MODEL_EMBEDDING = "intfloat/multilingual-e5-large"  # 1024-dim, multilingual, via fastembed
# #259 F1 model-swap (2026-09-17, Roy): swapped from BAAI/bge-small-en-v1.5
# (384-dim, English-only) to this multilingual model per the F1 spec's F4
# forward-compat item — fastembed-native, mit-licensed per fastembed's own
# model registry (`TextEmbedding.list_supported_models()`), confirmed
# supported there directly (checked against the installed fastembed). Ships
# sharded "external data" ONNX weights (model.onnx + model.onnx_data), which
# need the onnxruntime symlink workaround in brain/memory/embeddings.py
# (FastEmbedProvider.__init__ / _materialize_symlinked_files) to load under
# our pinned onnxruntime==1.29.0 — see that module for the full story.
#
# DOCUMENTED SANITY VALUE ONLY (#259 inc7 red-team F1) — NOT the load-bearing
# source of the dimension actually used to embed/decode/cluster. That comes
# from the REAL model's own output (`FastEmbedProvider.embedding_dim()`,
# probed via a one-time embed call — see brain/memory/embeddings.py), so a
# MODEL_EMBEDDING swap to a different-dim model (this one included) is
# genuinely one-touch: this constant does NOT need to change together for the
# system to keep working. It still matters for one thing: a loud startup/
# first-use health-check log (in FastEmbedProvider.embed(), see that class)
# compares the real probed dim against this constant and logs an error on a
# mismatch, so a desync (this constant going stale after a model swap) is
# caught LOUDLY rather than silently. Update it to match MODEL_EMBEDDING when
# you change that constant, but nothing breaks if you forget.
MODEL_EMBEDDING_DIM = 1024

# MODEL_RERANKER is also not a Claude model — it's a local ONNX cross-encoder
# reranker id (fastembed's TextCrossEncoder), the #231 reranker re-architecture
# (same standing convention as MODEL_EMBEDDING above): a cross-encoder reads
# (query, memory) TOGETHER and scores true relevance, replacing the old
# cosine-floor/gap auto-calibration that didn't generalize across corpus
# shapes. See brain/memory/reranker.py.
# #250 F2a inc1 model-swap (2026-09-17, Roy): swapped from the English-only
# Xenova/ms-marco-MiniLM-L-6-v2 to this multilingual model per the F4
# forward-compat item (see semantic-retrieval-LEDGER.md, agent a2feb8f5) —
# fastembed-native (TextCrossEncoder registry), a SINGLE self-contained
# fp32 .onnx file (additional_files: []), ~1.11GB, so it needs NO
# materialize-files workaround (unlike F1's multilingual-e5-large, which
# hits the onnxruntime external-data-path bug because it ships sharded
# external weights). fp16 export (~2x faster) is a LATER increment — see
# MODEL_RERANKER_FP16 below, now the PINNED default per the pre-flip
# revision's Change 2 (this constant, MODEL_RERANKER, is the fp32 fallback
# an operator override can still select). Score type: raw, unbounded logit
# (same shape as the outgoing ms-marco score, but a DIFFERENT SCALE) — AS OF THIS commit (inc1)
# the abstention floor was still the old hardcoded MiniLM-scaled constant, a
# known/expected scale mismatch; F2a's later increments replace it with a
# floor derived daily against jina's own scale (§7) and cut every consumer
# over to read it live (§8, inc8) — see `brain/memory/semantic_recall.py`.
# License: CC-BY-NC-4.0 (non-commercial), accepted for this free/open-source,
# non-commercial project.
MODEL_RERANKER = "jinaai/jina-reranker-v2-base-multilingual"  # ONNX cross-encoder, ~1.11GB fp32, via fastembed

# fp16 export of the SAME model (F2a inc2, #250 §2 origin) — the HF repo
# above also ships onnx/model_fp16.onnx (~557MB, confirmed present on the
# repo), but it is NOT pre-registered in fastembed's built-in
# TextCrossEncoder registry (only the fp32 onnx/model.onnx export above
# is). brain/memory/reranker.py registers this id via TextCrossEncoder.
# add_custom_model() pointing at that file — never read directly via
# model_for_tier/LOCAL_MODEL_IDS, so it stays a plain companion constant here
# rather than its own tier (same non-tier treatment MODEL_EMBEDDING_DIM
# gets above).
#
# ORIGINALLY (F2a inc2) gated behind a cached first-use fp16-vs-fp32
# accuracy self-check in reranker.py. The pre-flip revision's Change 2
# REMOVED that self-check entirely (a confirmed ~1.16 GiB one-time
# dual-load startup memory spike) and made this id the PINNED default
# instead: Testing's matched-width A/B reproduced fp32's surface/abstain
# decisions on all 40 bundled queries, so fp16-vs-fp32 agreement is a
# property of the bundled model weights (identical on every box), not
# something a per-box runtime probe needs to establish. `reranker.
# RERANKER_PRECISION`'s tunable default resolves to this id;
# `reranker.build_reranker_provider()` resolves back to MODEL_RERANKER
# (fp32) only on an explicit "reranker.precision"="fp32" operator override.
# 8-bit quantization was explicitly ruled out (a 278M model has less
# redundancy to absorb an 8-bit accuracy hit than a larger model would);
# fp16 is the one quantization lever here.
MODEL_RERANKER_FP16 = "jinaai/jina-reranker-v2-base-multilingual-fp16"

# MODEL_RELEVANCE_JUDGE is also not a Claude model — it's the offline local
# relevance judge for F2a's daily calibration tick (#250 §6, inc6). Runs
# ENTIRELY inside the once-daily idle-gated calibration tick
# (brain/bridge/supervisor.py's _run_calibration_tick), never on the
# per-turn recall path, so its own latency is irrelevant to recall — the
# reason it can be a bigger/slower model than the per-turn reranker.
# LOAD MECHANISM (Roy 2026-09-18, "torch it is then"): DIRECTLY via torch/
# sentence-transformers (brain/memory/relevance_judge.py), NOT fastembed/
# onnxruntime — deliberately different from MODEL_EMBEDDING/MODEL_RERANKER
# above. torch is a project dependency SCOPED to this offline judge only
# (see pyproject.toml's dependency comment); it is imported lazily, only
# when the daily tick actually runs a judge pass, so importing this module
# or any per-turn recall module never pulls torch in. Rationale for going
# through torch instead of ONNX here specifically: the judge is offline so
# torch's extra weight costs no per-turn latency, and direct-torch lets a
# user swap the judge to ANY cross-encoder id (PyTorch-only or ONNX-only
# alike) via this one constant — an ONNX-only runtime would restrict swaps
# to models that happen to ship a usable ONNX export.
# Independence rationale (settled on the ledger, not re-litigated here):
# bge-reranker-v2-m3 and jina (MODEL_RERANKER above) share an XLM-RoBERTa
# BASE model but are fine-tuned on different data — a soft correlated-blind-
# spot concern, not the hard self-judging circularity that sharing weights
# would create; Haiku's independent tie-break (relevance_judge.py) covers
# the ambiguous cases where that soft concern would matter most.
# License: apache-2.0 (permissively licensed; this model was already
# weighed and REJECTED as the per-turn reranker on latency grounds — too
# heavy per-doc for the no-AVX2 potato baseline's rerank-width auto-scaling
# — but that latency is irrelevant here, offline and once-daily).
MODEL_RELEVANCE_JUDGE = "BAAI/bge-reranker-v2-m3"

# Embedding is not a Claude-LLM tier — it never goes through get_provider/
# build_tier_provider (those are Claude-scoped: claude-cli/ollama/fake
# provider *kinds*, Claude model aliases). It's registered here anyway so the
# model id lives in ONE place with every other model selection; construct the
# actual provider via build_embedding_provider() in brain/memory/embeddings.py,
# which reads model_for_tier(TIER_EMBEDDING) rather than hardcoding the id.
TIER_EMBEDDING = "embedding"
# Same non-Claude-tier treatment as TIER_EMBEDDING above — registered here so
# the model id lives in ONE place; construct the actual provider via
# build_reranker_provider() in brain/memory/reranker.py, which reads
# model_for_tier(TIER_RERANKER) rather than hardcoding the id.
TIER_RERANKER = "reranker"
# Same non-Claude-tier treatment as TIER_EMBEDDING/TIER_RERANKER above —
# registered here so the model id lives in ONE place; construct the actual
# judge provider via build_judge_provider() in brain/memory/relevance_judge.py
# (#250 F2a inc6), which reads model_for_tier(TIER_RELEVANCE_JUDGE) rather
# than hardcoding the id. See MODEL_RELEVANCE_JUDGE's own comment above for
# why this tier loads via torch instead of the ONNX path the other two use.
TIER_RELEVANCE_JUDGE = "relevance-judge"

# The non-Claude tiers' model ids (embedder, reranker, relevance judge): plain constants, no tunables line.
LOCAL_MODEL_IDS: dict[str, str] = {
    TIER_EMBEDDING: MODEL_EMBEDDING,
    TIER_RERANKER: MODEL_RERANKER,
    TIER_RELEVANCE_JUDGE: MODEL_RELEVANCE_JUDGE,
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _configured(key: str, default: str) -> str:
    """The job's configured id: the user's override when it is a valid model setting, else the shipped default.

    ``tunables.get_tunable`` already ignores a value of the wrong type (a number, a list) with its own one-time
    warning. A string that fails the model grammar is ignored here with ONE throttled warning per distinct bad value;
    the text stays in ``tunables.json`` untouched (#446).
    """
    value = tunables.get_tunable(key, default)
    if is_valid_model(value):
        return value
    warn_throttled(
        logger,
        f"model_tier:invalid:{key}:{unsupported_display(value)}",
        "model_tier: the setting %s=%r is not a usable model (%s); the shipped default %s runs instead. "
        "The setting in tunables.json was left as it is.",
        key,
        unsupported_display(value),
        MODEL_ID_RULE,
        default,
        exc_info=False,
    )
    return default


def model_for_tier(tier: str) -> str:
    """The model id a job runs RIGHT NOW. Read on every call, so a tunables edit reaches a running bridge.

    Claude jobs: the configured id (``_configured``), unless Claude rejected that id since the settings file was last
    written, in which case the job's shipped default (the rejection notice stays). Non-Claude tiers return their
    constant. Raises ``KeyError`` on an unknown tier: fail loud, a typo'd tier name is a bug, not a fallback case.
    """
    job = _CLAUDE_JOBS.get(tier)
    if job is None:
        return LOCAL_MODEL_IDS[tier]
    from brain.bridge import (
        model_error,  # function-scoped: model_error imports this module's grammar at top
    )

    key, default = job
    configured = _configured(key, default)
    # An id in use again must never stay ignored (`switch` ignores the id a user moved away from; a CLI write or a hand
    # edit can configure it again without this process seeing the change), or a real rejection would be dropped silently.
    model_error.activate([configured])
    if configured != default and model_error.is_rejected(configured, since_epoch=tunables.overrides_mtime()):
        warn_throttled(
            logger,
            f"model_tier:fallback:{key}:{configured}",
            "model_tier: Claude rejected %s for the %s job, so it runs its shipped default %s until the settings file "
            "is changed or the app is restarted",
            configured,
            tier,
            default,
            exc_info=False,
        )
        return default
    return configured


def configured_job_models() -> dict[str, str]:
    """Each Claude job's CONFIGURED id (a valid override, else the shipped default), whatever Claude accepted.

    The allow-list for the rejection notice and the ``keep`` set of ``model_error.switch``: a rejected id must not
    drop out of the notice the moment the fallback starts, so these are not the effective ids.
    """
    return {tier: _configured(key, default) for tier, (key, default) in _CLAUDE_JOBS.items()}


def override_text(tier: str) -> str | None:
    """The user's override for a Claude job when it is NOT a usable model setting (display-safe), else None."""
    key, _default = _CLAUDE_JOBS[tier]
    raw = tunables.get_override(key)
    if raw is None or is_valid_model(raw):
        return None
    return unsupported_display(raw)


def has_override(tier: str) -> bool:
    """True when the user's tunables file holds a setting (valid or not) for this Claude job."""
    return tunables.get_override(_CLAUDE_JOBS[tier][0]) is not None


def shipped_default(tier: str) -> str:
    """The job's shipped default id (the registered default, ignoring any override)."""
    return _CLAUDE_JOBS[tier][1]


def is_shipped_default(model: str) -> bool:
    """True when ``model`` is the shipped default of at least one Claude job."""
    return any(default == model for _key, default in _CLAUDE_JOBS.values())


# ---------------------------------------------------------------------------
# The words the UI offers, and writing the settings file
# ---------------------------------------------------------------------------


def resolve_choice(value: str) -> str | None:
    """The id a picker word or a typed value means, else None when it is not a usable model setting.

    ``sonnet`` is the chat job's shipped default (read from the job table, no second literal), ``opus`` is
    ``OPUS_CHOICE``, ``haiku`` stays the alias ``haiku``, any other valid value is itself. Pure: reads no file.
    """
    if value == "sonnet":
        return _CLAUDE_JOBS[TIER_INTERACTIVE_CHAT][1]
    if value == "opus":
        return OPUS_CHOICE
    if is_valid_model(value):
        return value
    return None


def choice_for(model: str | None) -> str | None:
    """The picker word whose meaning is exactly ``model``, else None (a custom id highlights no radio)."""
    if model is None:
        return None
    for word in ("sonnet", "opus", "haiku"):
        if resolve_choice(word) == model:
            return word
    return None


def set_job_models(updates: dict[str, str | None], *, only_missing: bool = False) -> dict[str, str | None]:
    """Write the model setting of one or more jobs; the ONLY caller of ``tunables.set_overrides`` for model keys.

    ``updates`` maps a job type to the RAW value the user gave (a word or a full id) or ``None`` to clear the override.
    The write rule, per job: a WORD that means that job's shipped default REMOVES the override (so the user keeps
    following the product's pin); ``opus`` writes ``OPUS_CHOICE``; ``haiku`` writes ``haiku`` unless that is the job's
    default; a FULL ID is always written exactly as given, even when it equals the default (a deliberate pin).
    Raises ``ValueError`` for a value that is not a usable model setting and ``KeyError`` for an unknown job;
    ``tunables.TunablesWriteError`` when the file cannot be changed (nothing is written).

    After the write the rejection record of every id written AND every id removed is dropped, so a stale record
    cannot outlive the pick. Returns ``{job: written id, or None for a removal}`` for the settings actually changed.
    """
    from brain.bridge import (
        model_error,  # function-scoped: model_error imports this module's grammar at top
    )

    keys: dict[str, str] = {}
    mapped: dict[str, str | None] = {}
    previous: dict[str, Any] = {}
    for tier, raw in updates.items():
        key, default = _CLAUDE_JOBS[tier]
        keys[key] = tier
        previous[key] = tunables.get_override(key)
        if raw is None:
            mapped[key] = None
            continue
        if not is_valid_model(raw):
            raise ValueError(f"{raw!r} is not a usable model setting ({MODEL_ID_RULE})")
        if raw in KNOWN_MODELS:
            resolved = resolve_choice(raw)
            mapped[key] = None if resolved == default else resolved
        else:
            mapped[key] = raw
    written = tunables.set_overrides(mapped, only_missing=only_missing)
    # Picking an id again clears what Claude said about it, even when the file already held that exact value; an id
    # the pick replaces or removes is forgotten too. (Not the ids of keys `only_missing` left alone.)
    for key, value in mapped.items():
        if only_missing and key not in written:
            continue
        for forgotten in (value, previous[key]):
            if isinstance(forgotten, str):
                model_error.note_success(forgotten)
    return {keys[key]: value for key, value in written.items()}


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _provider_kind(persona_dir: Path) -> str:
    """The persona's provider *kind* (claude-cli / ollama / fake), read STRICTLY from ``persona_config.json``.

    The kind is never guessed from an unreadable file (a local-model persona's prompts must not go to Claude on a
    transient error), so a read error propagates. No file means the default kind. The persona file no longer holds a
    model: the model comes from the job's tunables line.
    """
    from brain.persona_config import DEFAULT_PROVIDER, PersonaConfig

    cfg = Path(persona_dir) / "persona_config.json"
    return PersonaConfig.load(cfg).provider if cfg.exists() else DEFAULT_PROVIDER


def build_tier_provider(persona_dir: Path, tier: str) -> Any:
    """The provider a given job should use: the persona's provider *kind* (claude-cli / ollama / fake), running the
    model the job's tunables line names (resolved at every call, not at construction).

    Generalizes ``build_compaction_provider``/``build_self_model_provider``: same shape, parametrized by job type.
    The function-scoped import of ``get_provider`` is the #154 test seam (a test that patches
    ``brain.bridge.provider.get_provider`` is not bypassed by a module-level binding here).
    """
    from brain.bridge.provider import get_provider

    return get_provider(_provider_kind(persona_dir), tier=tier)


def model_label_for_provider(provider: Any, tier: str) -> str:
    """Best-effort ACTUAL model label for usage logs: what the provider would run now, else the job's own model.

    A Claude provider answers with its live id (``_cli_model``); another provider type reports its own ``_model``;
    a provider with neither (``FakeProvider`` in tests) falls back to ``model_for_tier(tier)``. Centralized so no
    site hardcodes a label string.
    """
    live = getattr(provider, "_cli_model", None)
    if callable(live):
        return live()
    return getattr(provider, "_model", None) or model_for_tier(tier)
