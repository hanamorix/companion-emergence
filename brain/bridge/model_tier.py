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
