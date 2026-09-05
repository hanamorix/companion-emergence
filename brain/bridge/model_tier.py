"""Model-tier accessor — the ONE indirection point for which model backs a given
LLM call (#129 spirit; fixes #154's Sonnet/Haiku mislabeling).

Every model-selection site whose EFFECTIVE MODEL genuinely differs from a
shared/ambient provider it would otherwise inherit resolves that model through
``model_for_tier(tier)`` (or the convenience builder ``build_tier_provider``),
rather than hand-rolling its own ``model_override=`` literal.

Deliberate exception, do not "complete" this migration: sites whose tier
resolves to the SAME model an existing ambient/shared provider already uses
(``interactive-chat``, ``background-generative``, ``dev-cli`` — all
``MODEL_MEDIUM`` today) intentionally do NOT call through this module and keep
reusing that shared provider object instead (see ``brain/bridge/server.py``'s
lifespan, ``brain/engines/heartbeat.py``'s ``_try_fire_dream``/
``_try_fire_reflex``/``_try_fire_research``, and
``brain/bridge/supervisor.py``'s soul-review/voice-reflection/maker/notes
ticks). Migrating those sites to this accessor was tried once (#154's first
pass) and reverted after it broke a wide swath of pre-existing tests for zero
behavior change — see that project's guarded-change decisions log for the
full diagnosis. Two levels of indirection for the sites that DO route through
here:

  MODEL_LITTLE / MODEL_MEDIUM (MODEL_BIGGEST reserved) — the concrete model
    strings. Change ONE of these constants to re-point every tier that uses it
    (e.g. swap MODEL_LITTLE to a newer Haiku snapshot).
  TIER_MODEL — which named tier uses which model. Adding a new size (e.g. Opus
    for a future "biggest" tier) is one new MODEL_* constant plus reassigning the
    tiers that should use it in TIER_MODEL — no call site is touched.

Config-ready (#129): TIER_MODEL is the single mapping a future external config
surface points at instead of this hardcoded dict; call sites only ever ask
``model_for_tier(TIER_X)``, never read TIER_MODEL directly, so externalizing it
later (e.g. via ``brain.tunables``) is a change internal to this module only.

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
#   then reassign e.g. TIER_MODEL[TIER_INTERACTIVE_CHAT] = MODEL_BIGGEST
# No call-site changes are required either way.

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

TIER_MODEL: dict[str, str] = {
    TIER_INTERACTIVE_CHAT: MODEL_MEDIUM,  # future "biggest" slot
    TIER_ATTUNEMENT_DETECTOR: _ATTUNEMENT_DETECTOR_MODEL,  # pinned snapshot, preserved
    TIER_COMPACTION: MODEL_LITTLE,
    TIER_SELF_MODEL_ARTICULATE: MODEL_LITTLE,
    TIER_BACKGROUND_CLASSIFIER: MODEL_LITTLE,
    TIER_BACKGROUND_GENERATIVE: MODEL_MEDIUM,
    TIER_BACKGROUND_HOUSEKEEPING: MODEL_LITTLE,
    TIER_DEV_CLI: MODEL_MEDIUM,
}


def model_for_tier(tier: str) -> str:
    """The model string a given tier should run.

    Raises ``KeyError`` on an unknown tier — fail loud, not silently default;
    a typo'd tier name is a bug, not a fallback case.
    """
    return TIER_MODEL[tier]


def build_tier_provider(persona_dir: Path, tier: str) -> Any:
    """The provider a given tier should use: the persona's provider *kind*
    (claude-cli / ollama / fake) but forced to ``model_for_tier(tier)``.

    Generalizes ``build_compaction_provider``/``build_self_model_provider`` —
    same shape, parametrized by tier instead of duplicated per call site.
    """
    from brain.bridge.provider import get_provider
    from brain.persona_config import DEFAULT_PROVIDER, PersonaConfig

    name = DEFAULT_PROVIDER
    cfg = Path(persona_dir) / "persona_config.json"
    if cfg.exists():
        name = PersonaConfig.load(cfg).provider
    return get_provider(name, persona_dir=Path(persona_dir), model_override=model_for_tier(tier))


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
