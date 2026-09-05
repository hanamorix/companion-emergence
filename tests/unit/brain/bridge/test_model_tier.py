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
    monkeypatch.setitem(TIER_MODEL, TIER_INTERACTIVE_CHAT, "opus")
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == "opus"
    # Every OTHER tier is untouched by the single reassignment.
    assert model_for_tier(TIER_BACKGROUND_GENERATIVE) == "sonnet"
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
