"""Tests for brain/bridge/model_tier.py — the centralized model-tier accessor (#154).

Covers the guarded-change criteria (see changes/fix-154-model-tier-accessor/
1.5-criteria.md in the fix-80-167-77 worktree this landed from):

C1  — every tier resolves via the accessor (static + constructive checks here).
C4  — the accessor is N-model extensible (adding a size = one constant +
      one reassignment, zero call-site changes).
RTS — construction-only assertions: never calls .generate(), so no live
      `claude` CLI subprocess or network access is needed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from brain.bridge.model_tier import (
    MODEL_LITTLE,
    MODEL_MEDIUM,
    TIER_ATTUNEMENT_DETECTOR,
    TIER_BACKGROUND_CLASSIFIER,
    TIER_BACKGROUND_GENERATIVE,
    TIER_BACKGROUND_HOUSEKEEPING,
    TIER_COMPACTION,
    TIER_DEV_CLI,
    TIER_INTERACTIVE_CHAT,
    TIER_MODEL,
    TIER_SELF_MODEL_ARTICULATE,
    build_interactive_chat_provider,
    build_tier_provider,
    model_for_tier,
    model_label_for_provider,
)
from brain.bridge.provider import ClaudeCliProvider


def _claude_cli_persona(tmp_path: Path) -> Path:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "claude-cli"}')
    return persona_dir


def _claude_cli_persona_with_model(tmp_path: Path, model: str) -> Path:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "persona_config.json").write_text(
        f'{{"provider": "claude-cli", "model": "{model}"}}'
    )
    return persona_dir


# ---------------------------------------------------------------------------
# model_for_tier
# ---------------------------------------------------------------------------


def test_model_for_tier_resolves_every_declared_tier():
    for tier in (
        TIER_INTERACTIVE_CHAT,
        TIER_ATTUNEMENT_DETECTOR,
        TIER_COMPACTION,
        TIER_SELF_MODEL_ARTICULATE,
        TIER_BACKGROUND_CLASSIFIER,
        TIER_BACKGROUND_GENERATIVE,
        TIER_BACKGROUND_HOUSEKEEPING,
        TIER_DEV_CLI,
    ):
        assert isinstance(model_for_tier(tier), str)


def test_model_for_tier_unknown_tier_raises_keyerror():
    """Fail loud on a typo'd tier name — never silently default (ST design)."""
    with pytest.raises(KeyError):
        model_for_tier("not-a-real-tier")


def test_classifier_and_housekeeping_resolve_to_haiku():
    assert model_for_tier(TIER_BACKGROUND_CLASSIFIER) == MODEL_LITTLE == "haiku"
    assert model_for_tier(TIER_BACKGROUND_HOUSEKEEPING) == MODEL_LITTLE == "haiku"


def test_generative_interactive_chat_and_dev_cli_resolve_to_sonnet():
    assert model_for_tier(TIER_BACKGROUND_GENERATIVE) == MODEL_MEDIUM == "sonnet"
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == MODEL_MEDIUM == "sonnet"
    assert model_for_tier(TIER_DEV_CLI) == MODEL_MEDIUM == "sonnet"


def test_already_correct_tiers_preserve_their_exact_prior_values():
    """No unintended model change (C2): compaction/self-model-articulate keep
    the literal "haiku" they already used; attunement-detector keeps its own
    PINNED snapshot id verbatim, not the bare "haiku" alias (a substituted
    alias could silently repoint to a different snapshot over time)."""
    assert model_for_tier(TIER_COMPACTION) == "haiku"
    assert model_for_tier(TIER_SELF_MODEL_ARTICULATE) == "haiku"
    assert model_for_tier(TIER_ATTUNEMENT_DETECTOR) == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# build_tier_provider — construction-only, no subprocess spawned
# ---------------------------------------------------------------------------


def test_build_tier_provider_constructs_claude_cli_provider_with_the_tier_model(tmp_path):
    persona_dir = _claude_cli_persona(tmp_path)
    provider = build_tier_provider(persona_dir, TIER_BACKGROUND_CLASSIFIER)
    assert isinstance(provider, ClaudeCliProvider)
    assert provider._model == "haiku"  # noqa: SLF001


def test_build_tier_provider_generative_tier_resolves_to_sonnet(tmp_path):
    persona_dir = _claude_cli_persona(tmp_path)
    provider = build_tier_provider(persona_dir, TIER_BACKGROUND_GENERATIVE)
    assert isinstance(provider, ClaudeCliProvider)
    assert provider._model == "sonnet"  # noqa: SLF001


def test_build_tier_provider_defaults_provider_kind_when_no_persona_config(tmp_path):
    """No persona_config.json at all → falls back to DEFAULT_PROVIDER
    ("claude-cli"), mirroring build_compaction_provider/build_self_model_provider."""
    persona_dir = tmp_path / "no_config_persona"
    persona_dir.mkdir()
    provider = build_tier_provider(persona_dir, TIER_BACKGROUND_HOUSEKEEPING)
    assert isinstance(provider, ClaudeCliProvider)
    assert provider._model == "haiku"  # noqa: SLF001


# ---------------------------------------------------------------------------
# build_interactive_chat_provider — D1: the ONE tier that does NOT force
# model_override, so a persona's own persona_config.json `.model` field keeps
# winning. Completes #154 (dream/reflex/research/soul-review/voice-reflection/
# maker/notes + interactive-chat routed through the accessor; see
# changes/154-complete-model-tier-routing/).
# ---------------------------------------------------------------------------


def test_build_interactive_chat_provider_honors_personas_own_model(tmp_path):
    """C2 (no unintended model change): a persona configured for a NON-default
    model ("opus") must keep running on "opus" for interactive-chat — the one
    real call site that honors persona_config.json's own `.model` field.
    Oracle (C6): this would fail against a naive build_tier_provider-based
    implementation, which unconditionally forces model_override=MODEL_MEDIUM
    ("sonnet"), silently stripping the persona's own choice."""
    persona_dir = _claude_cli_persona_with_model(tmp_path, "opus")
    provider = build_interactive_chat_provider(persona_dir)
    assert isinstance(provider, ClaudeCliProvider)
    assert provider._model == "opus"  # noqa: SLF001


def test_build_interactive_chat_provider_defaults_to_default_model_when_unset(tmp_path):
    """No persona_config.json at all → DEFAULT_PROVIDER + PersonaConfig's own
    DEFAULT_MODEL fallback (get_provider's existing chain), NOT a tier-forced
    override — mirrors build_tier_provider's provider-*kind* fallback but
    without the model_override this function deliberately omits."""
    from brain.persona_config import DEFAULT_MODEL

    persona_dir = tmp_path / "no_config_persona"
    persona_dir.mkdir()
    provider = build_interactive_chat_provider(persona_dir)
    assert isinstance(provider, ClaudeCliProvider)
    assert provider._model == DEFAULT_MODEL  # noqa: SLF001


def test_interactive_chat_tier_model_entry_is_decorative(monkeypatch, tmp_path):
    """C9 — reassigning TIER_MODEL[TIER_INTERACTIVE_CHAT] must NOT change what
    build_interactive_chat_provider actually resolves (unlike every other
    tier, proven by test_n_model_extensible_new_size_needs_no_call_site_
    changes above for TIER_BACKGROUND_GENERATIVE). This is the guard against a
    future maintainer "completing" model_tier.py's own extensibility
    docstring for this one tier and silently reintroducing the persona-model-
    override regression D1 exists to prevent."""
    persona_dir = _claude_cli_persona_with_model(tmp_path, "opus")
    monkeypatch.setitem(TIER_MODEL, TIER_INTERACTIVE_CHAT, "haiku")
    provider = build_interactive_chat_provider(persona_dir)
    # Still "opus" (the persona's own model) — NOT "haiku" (the monkeypatched
    # TIER_MODEL value), proving model_for_tier(TIER_INTERACTIVE_CHAT) is
    # never consulted by this function.
    assert provider._model == "opus"  # noqa: SLF001


def test_model_label_for_interactive_chat_reflects_the_real_resolved_model(tmp_path):
    """C7 — the usage-log label for interactive-chat must reflect whatever
    model persona_config.json actually specified, not a hardcoded tier value."""
    persona_dir = _claude_cli_persona_with_model(tmp_path, "opus")
    provider = build_interactive_chat_provider(persona_dir)
    assert model_label_for_provider(provider, TIER_INTERACTIVE_CHAT) == "opus"


# ---------------------------------------------------------------------------
# model_label_for_provider
# ---------------------------------------------------------------------------


def test_model_label_for_provider_prefers_the_providers_real_model(tmp_path):
    persona_dir = _claude_cli_persona(tmp_path)
    provider = build_tier_provider(persona_dir, TIER_BACKGROUND_CLASSIFIER)
    assert model_label_for_provider(provider, TIER_BACKGROUND_CLASSIFIER) == "haiku"


def test_model_label_for_provider_falls_back_for_a_provider_with_no_model_attr():
    class _NoModelAttrProvider:
        pass

    label = model_label_for_provider(_NoModelAttrProvider(), TIER_BACKGROUND_GENERATIVE)
    assert label == "sonnet"


# ---------------------------------------------------------------------------
# C4 — N-model extensible: adding a size is one constant + one reassignment,
# zero call-site edits. Proven by mutating TIER_MODEL from inside the test
# itself and confirming model_for_tier reflects it immediately — no code
# outside this module needed to change to observe the new value.
# ---------------------------------------------------------------------------


def test_n_model_extensible_new_size_needs_no_call_site_changes(monkeypatch):
    """Uses TIER_BACKGROUND_GENERATIVE, NOT TIER_INTERACTIVE_CHAT — the latter
    is the one documented exception (see build_interactive_chat_provider's
    docstring + test_interactive_chat_tier_model_entry_is_decorative below):
    reassigning ITS TIER_MODEL entry does NOT change what actually runs, so it
    would be a misleading "no call site changes" demonstration subject.
    TIER_BACKGROUND_GENERATIVE genuinely has the property this test asserts."""
    monkeypatch.setitem(TIER_MODEL, TIER_BACKGROUND_GENERATIVE, "opus")
    assert model_for_tier(TIER_BACKGROUND_GENERATIVE) == "opus"
    # Every OTHER tier is untouched by the single reassignment.
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == "sonnet"
    assert model_for_tier(TIER_BACKGROUND_CLASSIFIER) == "haiku"


# ---------------------------------------------------------------------------
# C1 — static check: no site outside this module hand-rolls model_override=
# or constructs a bare ClaudeCliProvider(model=...) (except the two
# owner-ruled-out-of-scope exceptions and this module itself).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BRAIN_ROOT = _REPO_ROOT / "brain"


def _iter_brain_py_files():
    yield from _BRAIN_ROOT.rglob("*.py")


def test_no_site_outside_model_tier_hand_rolls_model_override():
    offenders = []
    for path in _iter_brain_py_files():
        if path.name == "model_tier.py":
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"model_override\s*=", text):
            offenders.append(str(path))
    assert offenders == [], f"model_override= used outside model_tier.py: {offenders}"


def test_no_site_hardcodes_a_literal_model_string_into_claude_cli_provider():
    """A site MAY construct ClaudeCliProvider(model=...) directly (detector.py and
    emotion_backfill.py's dead fallback both do — provider-*kind* resolution is
    deliberately out of scope for the detector, and the fallback branch is
    confirmed dead in production) as long as the model VALUE is sourced from
    model_for_tier(...)/the accessor, never a hardcoded literal string — that is
    the actual C1 property (no site hand-rolls the MODEL VALUE), not a ban on
    calling ClaudeCliProvider's constructor outside this module."""
    offenders = []
    for path in _iter_brain_py_files():
        if path.name in ("model_tier.py", "provider.py"):
            continue
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"ClaudeCliProvider\(\s*\n?\s*model\s*=\s*(\S+)", text):
            value_start = m.group(1)
            if value_start.startswith(("'", '"')):
                offenders.append(f"{path}: literal model= {value_start!r}")
    assert offenders == [], f"ClaudeCliProvider(model=<literal>) outside the accessor: {offenders}"
