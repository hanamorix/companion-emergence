"""Model-tier accessor — the ONE indirection point for which model backs a given
LLM call (#129 spirit; fixes #154's Sonnet/Haiku mislabeling).

Every model-selection site whose EFFECTIVE MODEL genuinely differs from a
shared/ambient provider it would otherwise inherit resolves that model through
``model_for_tier(tier)`` (or the convenience builder ``build_tier_provider``),
rather than hand-rolling its own ``model_override=`` literal.

``dev-cli`` (``brain/cli.py``) is the one tier that intentionally does NOT call
through this module: its handlers support a developer ``--provider`` CLI-flag
override (``_resolve_routing``) this accessor cannot replicate (it always
reads ``persona_config.json`` directly, with no flag-override channel) —
routing it would silently drop that override, a real capability regression.
Left as ``get_provider(provider_name)``, unchanged; see
``changes/fix-154-model-tier-accessor/decisions.md`` for the full history.

Every other tier — including ``interactive-chat`` and ``background-generative``
— DOES route through this module (``build_tier_provider`` or, for
``interactive-chat`` specifically, ``build_interactive_chat_provider`` below).
An earlier attempt at this (#154's first pass) tried routing them through
plain ``build_tier_provider`` and reverted after it broke a wide swath of
pre-existing tests — not because routing itself was wrong, but because two
FIXABLE test-seam gaps hadn't been closed yet (real ``claude`` CLI subprocess
spawns from unit tests using bare ``tmp_path`` personas with no
``persona_config.json``, and a ``get_provider``-patching test seam bypassed by
this module's function-scoped import). Both are closed as of the pass that
finished this migration; see ``changes/154-complete-model-tier-routing/`` for
that work's spec/plan/decisions log. Two levels of indirection for every site
that routes through here:

  MODEL_LITTLE / MODEL_MEDIUM (MODEL_BIGGEST reserved) — the concrete model
    strings. Change ONE of these constants to re-point every tier that uses it
    (e.g. swap MODEL_LITTLE to a newer Haiku snapshot).
  TIER_MODEL — which named tier uses which model. Adding a new size (e.g. Opus
    for a future "biggest" tier) is one new MODEL_* constant plus reassigning
    the tiers that should use it in TIER_MODEL — no call site is touched, FOR
    EVERY TIER EXCEPT ``TIER_INTERACTIVE_CHAT`` (see its own comment in
    TIER_MODEL below, and ``build_interactive_chat_provider``'s docstring):
    that one tier's real model comes from the persona's own
    ``persona_config.json`` field, not from this dict, so reassigning its
    TIER_MODEL entry is deliberately decorative there — a documented, tested
    exception (see ``test_model_tier.py``), not an oversight.

Config-ready (#129): TIER_MODEL is the single mapping a future external config
surface points at instead of this hardcoded dict; call sites only ever ask
``model_for_tier(TIER_X)``, never read TIER_MODEL directly, so externalizing
it later (e.g. via ``brain.tunables``) is a change internal to this module
only — again, for every tier except ``TIER_INTERACTIVE_CHAT``, whose real
model is sourced from persona config, not this module (see above).

Generalizes the pattern already used by ``build_compaction_provider``
(``brain/chat/compaction.py``) and ``build_self_model_provider``
(``brain/self_model/articulate.py``): resolve the provider *kind* from
``persona_config.json``, then ``get_provider(kind, persona_dir=...,
model_override=<pinned model>)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Model-size constants — the concrete models. One constant, many tiers.
# ---------------------------------------------------------------------------
MODEL_LITTLE = "haiku"  # cheap/fast: classifiers, housekeeping, cheap ticks
MODEL_MEDIUM = "sonnet"  # persona-quality generation: chat, background-generative
# MODEL_BIGGEST is reserved for a future top tier (e.g. Opus). Introducing it:
#   MODEL_BIGGEST = "opus"
#   then reassign e.g. TIER_MODEL[TIER_BACKGROUND_GENERATIVE] = MODEL_BIGGEST
# No call-site changes are required for that tier (or any tier EXCEPT
# TIER_INTERACTIVE_CHAT — see that constant's own TIER_MODEL comment below).

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
# external weights). fp16 export (~2x faster) is a LATER increment, gated
# on a build-time fp16-vs-fp32 accuracy check — this fp32 default is
# increment 1 only. Score type: raw, unbounded logit (same shape as the
# outgoing ms-marco score, but a DIFFERENT SCALE) — AS OF THIS commit (inc1)
# the abstention floor was still the old hardcoded MiniLM-scaled constant, a
# known/expected scale mismatch; F2a's later increments replace it with a
# floor derived daily against jina's own scale (§7) and cut every consumer
# over to read it live (§8, inc8) — see `brain/memory/semantic_recall.py`.
# License: CC-BY-NC-4.0 (non-commercial), accepted for this free/open-source,
# non-commercial project.
MODEL_RERANKER = "jinaai/jina-reranker-v2-base-multilingual"  # ONNX cross-encoder, ~1.11GB fp32, via fastembed

# fp16 export of the SAME model (F2a inc2, #250 §2) — the HF repo above also
# ships onnx/model_fp16.onnx (~557MB, confirmed present on the repo), but it
# is NOT pre-registered in fastembed's built-in TextCrossEncoder registry
# (only the fp32 onnx/model.onnx export above is). brain/memory/reranker.py
# registers this id via TextCrossEncoder.add_custom_model() pointing at that
# file, then uses it ONLY as the fp16 candidate in a cached first-use
# fp16-vs-fp32 accuracy self-check (mirrors reranker.py's existing warm
# per-doc latency auto-calibration: measured once, cached, not
# per-recall) — never read directly via model_for_tier/TIER_MODEL, so it
# stays a plain companion constant here rather than its own tier (same
# non-tier treatment MODEL_EMBEDDING_DIM gets above). The self-check
# decides, per box, whether production actually serves this fp16 export or
# falls back to MODEL_RERANKER's fp32 export — either the fp16 export fails
# to preserve fp32's surface/abstain decisions on the bundled representative
# pairs, or it does but isn't measurably faster on this host (a no-AVX2
# potato CPU may not accelerate fp16), and fp32 stays the safe default in
# both cases. 8-bit quantization was explicitly ruled out (a 278M model has
# less redundancy to absorb an 8-bit accuracy hit than a larger model
# would); fp16 is the one quantization lever here.
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

# attunement-detector keeps its own pre-existing PINNED snapshot id verbatim
# (not the bare "haiku" alias) — this predates #154 and substituting the alias
# could silently repoint it to a different snapshot over time. Moved here from
# brain/attunement/detector.py's former _DETECTOR_MODEL constant.
_ATTUNEMENT_DETECTOR_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Tier names — the categories every model-selection call site declares.
# ---------------------------------------------------------------------------
TIER_INTERACTIVE_CHAT = "interactive-chat"
TIER_ATTUNEMENT_DETECTOR = "attunement-detector"
TIER_COMPACTION = "compaction"
TIER_SELF_MODEL_ARTICULATE = "self-model-articulate"
TIER_BACKGROUND_CLASSIFIER = "background-classifier"
TIER_BACKGROUND_GENERATIVE = "background-generative"
TIER_BACKGROUND_HOUSEKEEPING = "background-housekeeping"
TIER_DEV_CLI = "dev-cli"
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

TIER_MODEL: dict[str, str] = {
    # NOMINAL/DEFAULT ONLY — decorative for this one tier (future "biggest"
    # slot when reassigned). build_interactive_chat_provider() below never
    # reads this entry; the live /chat route's real model always comes from
    # the persona's own persona_config.json `.model` field (falling back to
    # PersonaConfig.DEFAULT_MODEL, which happens to equal MODEL_MEDIUM today).
    # Reassigning this line does NOT change what interactive-chat runs on —
    # see build_interactive_chat_provider's docstring + test_model_tier.py.
    TIER_INTERACTIVE_CHAT: MODEL_MEDIUM,
    TIER_ATTUNEMENT_DETECTOR: _ATTUNEMENT_DETECTOR_MODEL,  # pinned snapshot, preserved
    TIER_COMPACTION: MODEL_LITTLE,
    TIER_SELF_MODEL_ARTICULATE: MODEL_LITTLE,
    TIER_BACKGROUND_CLASSIFIER: MODEL_LITTLE,
    TIER_BACKGROUND_GENERATIVE: MODEL_MEDIUM,
    TIER_BACKGROUND_HOUSEKEEPING: MODEL_LITTLE,
    TIER_DEV_CLI: MODEL_MEDIUM,
    TIER_EMBEDDING: MODEL_EMBEDDING,
    TIER_RERANKER: MODEL_RERANKER,
    TIER_RELEVANCE_JUDGE: MODEL_RELEVANCE_JUDGE,
}


def model_for_tier(tier: str) -> str:
    """The model string a given tier should run.

    Raises ``KeyError`` on an unknown tier — fail loud, not silently default;
    a typo'd tier name is a bug, not a fallback case.
    """
    return TIER_MODEL[tier]


def _resolve_provider_kind(persona_dir: Path) -> str:
    """The persona's provider *kind* (claude-cli / ollama / fake), read from
    ``persona_config.json`` if present, else ``DEFAULT_PROVIDER``.

    Shared by ``build_tier_provider`` and ``build_interactive_chat_provider`` so
    the two builders can never silently drift in how they resolve provider
    *kind* (as distinct from *model*, which they resolve differently — see
    ``build_interactive_chat_provider``'s docstring).
    """
    from brain.persona_config import DEFAULT_PROVIDER, PersonaConfig

    cfg = Path(persona_dir) / "persona_config.json"
    if cfg.exists():
        return PersonaConfig.load(cfg).provider
    return DEFAULT_PROVIDER


def build_tier_provider(persona_dir: Path, tier: str) -> Any:
    """The provider a given tier should use: the persona's provider *kind*
    (claude-cli / ollama / fake) but forced to ``model_for_tier(tier)``.

    Generalizes ``build_compaction_provider``/``build_self_model_provider`` —
    same shape, parametrized by tier instead of duplicated per call site.
    """
    from brain.bridge.provider import get_provider

    name = _resolve_provider_kind(persona_dir)
    return get_provider(name, persona_dir=Path(persona_dir), model_override=model_for_tier(tier))


def build_interactive_chat_provider(persona_dir: Path) -> Any:
    """The provider ``TIER_INTERACTIVE_CHAT`` (the live ``/chat`` route) should
    use: the persona's provider *kind*, exactly like ``build_tier_provider``,
    but WITHOUT forcing ``model_override``.

    This is the one deliberate exception in this module: every other tier's
    model is fully owned by ``TIER_MODEL`` (change the constant, every call
    site follows). Interactive-chat is different because ``persona_config.json``
    has its own genuinely user-facing ``.model`` field (a validated 3-value
    enum today: sonnet/opus/haiku — see ``PersonaConfig.KNOWN_MODELS``) that a
    persona owner can already set independently of any tier default, and this
    is the ONE real call site that honors it. Forcing ``model_override=
    model_for_tier(TIER_INTERACTIVE_CHAT)`` here would silently strip that
    per-persona choice for anyone not on the tier's nominal model — an
    unintended behavior change this function exists specifically to avoid.
    ``TIER_MODEL[TIER_INTERACTIVE_CHAT]`` therefore stays nominal/default-only
    for this tier (see its own comment there); reassigning it does not change
    what this function returns — ``test_model_tier.py`` asserts exactly that.
    """
    from brain.bridge.provider import get_provider

    name = _resolve_provider_kind(persona_dir)
    return get_provider(name, persona_dir=Path(persona_dir))


def model_label_for_provider(provider: Any, tier: str) -> str:
    """Best-effort ACTUAL model label for usage logs.

    Prefers the provider's own recorded ``._model`` (set from the
    ``model_override`` passed to ``get_provider``), so the label reflects what
    really ran. Falls back to the tier's nominal model only for a provider type
    with no such attribute (e.g. ``FakeProvider`` in tests). Mirrors
    ``self_model/articulate.py``'s ``_provider_model_label``; centralized here
    so no site hardcodes a label string.
    """
    return getattr(provider, "_model", None) or model_for_tier(tier)
