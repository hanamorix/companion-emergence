"""#441 scope addition: background (tier) models follow the chat model's version.

Rule (owner, 2026-10-17): when the chat model is a full id, each routed Claude tier uses the SAME version in
its own family (chat claude-sonnet-5 -> haiku-family tiers claude-haiku-5); a derived id Claude rejects falls
back to what that tier uses today; an explicit per-tier override wins; an alias chat model changes nothing.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from brain.bridge import model_error
from brain.bridge.model_tier import (
    ROUTED_CLAUDE_TIERS,
    TIER_ATTUNEMENT_DETECTOR,
    TIER_BACKGROUND_CLASSIFIER,
    TIER_BACKGROUND_GENERATIVE,
    TIER_BACKGROUND_HOUSEKEEPING,
    TIER_COMPACTION,
    TIER_MODEL,
    TIER_SELF_MODEL_ARTICULATE,
    build_tier_provider,
    candidate_tier_models,
    derive_model,
    resolve_tier_model,
    tier_fallbacks,
    tier_family,
)
from brain.persona_config import PersonaConfig, is_valid_model

PINNED = "claude-haiku-4-5-20251001"
SENTENCE = "There's an issue with the selected model ({model}). It may not exist or you may not have access to it."


def _persona(tmp_path: Path, model: str = "sonnet", **extra) -> Path:
    d = tmp_path / "persona"
    d.mkdir(exist_ok=True)
    (d / "persona_config.json").write_text(
        json.dumps({"provider": "claude-cli", "searcher": "noop", "model": model, **extra})
    )
    return d


def _proc(rc: int, stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


def _reject_only(*bad: str):
    """A subprocess.run stand-in that rejects the listed models, as the real CLI does."""

    def _run(cmd, **kw):
        model = cmd[cmd.index("--model") + 1]
        if model in bad:
            return _proc(1, json.dumps({"is_error": True, "api_error_status": 404, "result": SENTENCE.format(model=model)}))
        return _proc(0, json.dumps({"is_error": False, "result": "ok"}))

    return _run


# --- C19: derivation ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chat", "family", "expected"),
    [
        ("claude-sonnet-5", "haiku", "claude-haiku-5"),
        ("claude-sonnet-5", "opus", "claude-opus-5"),
        ("claude-sonnet-5", "sonnet", "claude-sonnet-5"),
        ("claude-opus-4-8", "haiku", "claude-haiku-4-8"),
        ("claude-sonnet-4-5-20250929", "haiku", "claude-haiku-4-5"),  # date dropped across families
        ("claude-3-5-sonnet-20241022", "haiku", "claude-3-5-haiku"),
        ("claude-sonnet-4-5-20250929", "sonnet", "claude-sonnet-4-5-20250929"),  # same family: verbatim, date kept
        ("claude-sonnet-99", "haiku", "claude-haiku-99"),  # no version list anywhere
        ("claude-fable-5", "haiku", None),  # no family component
        ("sonnet", "haiku", None),  # aliases never derive
        ("opus", "opus", None),
        ("claude-sonnet-" + "9" * 55, "opus", None),  # derived id would exceed 64 characters
    ],
)
def test_derive_model(chat, family, expected):
    assert derive_model(chat, family) == expected
    if expected is not None:
        assert is_valid_model(expected)


@pytest.mark.parametrize(
    ("model", "family"),
    [("sonnet", "sonnet"), ("opus", "opus"), ("haiku", "haiku"), (PINNED, "haiku"), ("claude-sonnet-5", "sonnet"),
     ("claude-3-5-haiku-20241022", "haiku"), ("claude-fable-5", None), ("gpt4", None)],
)
def test_tier_family(model, family):
    assert tier_family(model) == family


def test_derivation_code_has_no_version_table():
    funcs = (tier_family, derive_model, model_tier._resolve, resolve_tier_model, candidate_tier_models, tier_fallbacks)
    src = "\n".join(inspect.getsource(f) for f in funcs)
    literals = set(re.findall(r"claude-[a-z0-9.-]+", src))
    assert literals <= set(), literals  # no model-id literal in the derivation/resolution code


def test_routed_tiers_match_the_config_whitelist():
    from brain.persona_config import ROUTED_TIER_NAMES

    assert tuple(ROUTED_CLAUDE_TIERS) == tuple(ROUTED_TIER_NAMES)
    assert set(ROUTED_CLAUDE_TIERS) <= set(TIER_MODEL)


# --- C20 / C26: resolution and alias preservation -----------------------------------------------


@pytest.mark.parametrize("tier", ROUTED_CLAUDE_TIERS)
def test_alias_chat_model_keeps_every_tier_exactly_as_today(tmp_path, tier):
    for alias in ("sonnet", "opus", "haiku"):
        d = _persona(tmp_path, alias)
        assert resolve_tier_model(d, tier) == (TIER_MODEL[tier], "default")
        assert build_tier_provider(d, tier)._model == TIER_MODEL[tier]


def test_pinned_attunement_snapshot_is_the_default():
    assert TIER_MODEL[TIER_ATTUNEMENT_DETECTOR] == PINNED


def test_full_id_chat_model_moves_each_family_to_the_same_version(tmp_path):
    d = _persona(tmp_path, "claude-sonnet-5")
    for tier in (TIER_COMPACTION, TIER_SELF_MODEL_ARTICULATE, TIER_BACKGROUND_CLASSIFIER, TIER_BACKGROUND_HOUSEKEEPING):
        assert resolve_tier_model(d, tier) == ("claude-haiku-5", "derived")
    assert resolve_tier_model(d, TIER_BACKGROUND_GENERATIVE) == ("claude-sonnet-5", "derived")
    assert resolve_tier_model(d, TIER_ATTUNEMENT_DETECTOR) == ("claude-haiku-5", "derived")  # replaces the pin (owner rule)
    assert build_tier_provider(d, TIER_COMPACTION)._model == "claude-haiku-5"


def test_chat_model_with_no_family_leaves_tiers_on_today(tmp_path):
    d = _persona(tmp_path, "claude-fable-5")
    for tier in ROUTED_CLAUDE_TIERS:
        assert resolve_tier_model(d, tier) == (TIER_MODEL[tier], "default")


def test_non_claude_provider_never_gets_derived_ids(tmp_path):
    d = _persona(tmp_path, "claude-sonnet-5", provider="ollama")
    for tier in ROUTED_CLAUDE_TIERS:
        assert resolve_tier_model(d, tier) == (TIER_MODEL[tier], "default")


def test_override_beats_derivation_and_default(tmp_path):
    d = _persona(tmp_path, "claude-sonnet-5", tier_models={TIER_COMPACTION: "claude-haiku-4-5", TIER_BACKGROUND_GENERATIVE: "opus"})
    assert resolve_tier_model(d, TIER_COMPACTION) == ("claude-haiku-4-5", "override")
    assert resolve_tier_model(d, TIER_BACKGROUND_GENERATIVE) == ("opus", "override")
    assert resolve_tier_model(d, TIER_BACKGROUND_CLASSIFIER) == ("claude-haiku-5", "derived")
    d2 = _persona(tmp_path, "sonnet", tier_models={TIER_COMPACTION: "claude-haiku-4-5"})  # override with an alias chat model
    assert resolve_tier_model(d2, TIER_COMPACTION) == ("claude-haiku-4-5", "override")


def test_missing_or_corrupt_config_and_no_dir_fall_back_to_default(tmp_path):
    for tier in ROUTED_CLAUDE_TIERS:
        assert resolve_tier_model(None, tier) == (TIER_MODEL[tier], "default")
        assert resolve_tier_model(tmp_path / "nope", tier) == (TIER_MODEL[tier], "default")
    d = tmp_path / "bad"
    d.mkdir()
    (d / "persona_config.json").write_text("{not json")
    assert resolve_tier_model(d, TIER_COMPACTION) == (TIER_MODEL[TIER_COMPACTION], "default")


def test_resolution_follows_config_edits_without_any_reset(tmp_path):
    d = _persona(tmp_path, "claude-sonnet-5")
    assert resolve_tier_model(d, TIER_COMPACTION)[0] == "claude-haiku-5"
    _persona(tmp_path, "claude-opus-4-8")
    assert resolve_tier_model(d, TIER_COMPACTION)[0] == "claude-haiku-4-8"  # no cache to invalidate


def test_attunement_detector_and_default_tagger_use_the_resolved_model(tmp_path, monkeypatch):
    from brain.attunement import detector
    from brain.bridge import provider as provider_mod
    from brain.ingest import emotion_backfill

    built: list[str] = []

    class _Rec:
        def __init__(self, model="sonnet", timeout_seconds=300):
            built.append(model)

        def generate(self, *a, **k):
            return "{}"

    monkeypatch.setattr(provider_mod, "ClaudeCliProvider", _Rec)
    d = _persona(tmp_path, "claude-sonnet-5")
    detector._call_haiku("sys", "user", persona_dir=d)
    emotion_backfill._make_default_tagger(None, persona_dir=d)
    assert built == ["claude-haiku-5", "claude-haiku-5"]
    built.clear()
    d2 = _persona(tmp_path, "sonnet")
    detector._call_haiku("sys", "user", persona_dir=d2)
    emotion_backfill._make_default_tagger(None, persona_dir=d2)
    assert built == [PINNED, "haiku"]  # alias chat model: exactly today's models


# --- C21: fallback ---------------------------------------------------------------------------------


def test_runtime_rejection_of_a_derived_id_falls_back_to_each_tiers_default(tmp_path):
    d = _persona(tmp_path, "claude-sonnet-5")
    calls: list[str] = []
    run = _reject_only("claude-haiku-5")

    def _spy(cmd, **kw):
        calls.append(cmd[cmd.index("--model") + 1])
        return run(cmd, **kw)

    with patch("brain.bridge.provider.subprocess.run", side_effect=_spy):
        with pytest.raises(RuntimeError, match="selected model"):
            build_tier_provider(d, TIER_BACKGROUND_CLASSIFIER).generate("tick", persona_dir=None)
    assert calls == ["claude-haiku-5"]  # the failing call is not retried
    assert "claude-haiku-5" in {r["model"] for r in model_error.state()}
    assert build_tier_provider(d, TIER_BACKGROUND_CLASSIFIER)._model == "haiku"
    assert build_tier_provider(d, TIER_ATTUNEMENT_DETECTOR)._model == PINNED  # the pin returns, not a moving alias
    assert resolve_tier_model(d, TIER_BACKGROUND_CLASSIFIER) == ("haiku", "fallback")
    assert resolve_tier_model(d, TIER_BACKGROUND_GENERATIVE) == ("claude-sonnet-5", "derived")  # other family untouched
    # visible through the existing path: the rejected derived id is reported while it is being replaced
    cfg = PersonaConfig.load(d / "persona_config.json")
    assert "claude-haiku-5" in candidate_tier_models(cfg)


def test_persisted_unavailable_derived_id_resolves_to_the_default(tmp_path):
    d = _persona(tmp_path, "claude-sonnet-5", derived_unavailable=["claude-haiku-5"])
    assert resolve_tier_model(d, TIER_COMPACTION) == ("haiku", "fallback")
    assert resolve_tier_model(d, TIER_ATTUNEMENT_DETECTOR) == (PINNED, "fallback")
    assert resolve_tier_model(d, TIER_BACKGROUND_GENERATIVE) == ("claude-sonnet-5", "derived")


def test_explicit_override_is_never_replaced_but_is_reported(tmp_path):
    d = _persona(tmp_path, "claude-sonnet-5", tier_models={TIER_COMPACTION: "claude-haiku-9"})
    model_error.note_failure(SENTENCE.format(model="claude-haiku-9"))
    assert resolve_tier_model(d, TIER_COMPACTION) == ("claude-haiku-9", "override")
    cfg = PersonaConfig.load(d / "persona_config.json")
    assert "claude-haiku-9" in candidate_tier_models(cfg)
    assert model_error.state(only=candidate_tier_models(cfg))[0]["model"] == "claude-haiku-9"


def test_tier_fallbacks_report_per_distinct_pair_and_never_for_the_chat_id(tmp_path):
    d = _persona(tmp_path, "claude-sonnet-5", derived_unavailable=["claude-haiku-5"])
    cfg = PersonaConfig.load(d / "persona_config.json")
    assert tier_fallbacks(cfg) == [
        {"family": "haiku", "wanted": "claude-haiku-5", "using": "haiku"},
        {"family": "haiku", "wanted": "claude-haiku-5", "using": PINNED},
    ]
    # M4: derived == chat id (the sonnet-family tier) never produces a fallback entry; the chat-model line covers it
    d2 = _persona(tmp_path, "claude-sonnet-5", derived_unavailable=["claude-sonnet-5"])
    assert tier_fallbacks(PersonaConfig.load(d2 / "persona_config.json")) == []
    model_error.note_failure(SENTENCE.format(model="claude-sonnet-5"))
    assert tier_fallbacks(PersonaConfig.load(d2 / "persona_config.json")) == []
    assert [r["model"] for r in model_error.state(only={"claude-sonnet-5"})] == ["claude-sonnet-5"]
    # alias chat model / nothing unavailable
    assert tier_fallbacks(PersonaConfig.load(_persona(tmp_path, "sonnet", derived_unavailable=["claude-haiku-5"]) / "persona_config.json")) == []
    assert tier_fallbacks(PersonaConfig.load(_persona(tmp_path, "claude-sonnet-5") / "persona_config.json")) == []


def test_fallbacks_disappear_when_the_chat_model_changes(tmp_path):
    d = _persona(tmp_path, "claude-sonnet-5", derived_unavailable=["claude-haiku-5"])
    _persona(tmp_path, "claude-opus-4-8")  # rewritten without the list, as set_persona_model does
    assert tier_fallbacks(PersonaConfig.load(d / "persona_config.json")) == []
    assert resolve_tier_model(d, TIER_COMPACTION) == ("claude-haiku-4-8", "derived")


def test_ignored_candidate_cannot_suppress_a_rejection(tmp_path):
    # M1: an id ignored after a switch-away that becomes active again in a tier role (here through a config
    # edit the bridge never saw, like `nell model set-tier`) must record a real rejection.
    model_error.switch("claude-haiku-9", "sonnet", keep=frozenset())
    model_error.note_failure(SENTENCE.format(model="claude-haiku-9"))
    assert model_error.state() == []  # ignored while unused
    d = _persona(tmp_path, "sonnet", tier_models={TIER_COMPACTION: "claude-haiku-9"})
    resolve_tier_model(d, TIER_COMPACTION)  # activates the id
    model_error.note_failure(SENTENCE.format(model="claude-haiku-9"))
    assert [r["model"] for r in model_error.state()] == ["claude-haiku-9"]


def test_derived_id_ignored_after_a_switch_still_falls_back_when_rejected(tmp_path):
    model_error.switch("claude-sonnet-5", "sonnet", keep=frozenset())  # user moved to an alias
    d = _persona(tmp_path, "claude-opus-5")  # later: chat opus-5 derives claude-sonnet-5 for the generative tier
    with patch("brain.bridge.provider.subprocess.run", side_effect=_reject_only("claude-sonnet-5")):
        with pytest.raises(RuntimeError):
            build_tier_provider(d, TIER_BACKGROUND_GENERATIVE).generate("tick")
    assert resolve_tier_model(d, TIER_BACKGROUND_GENERATIVE) == ("sonnet", "fallback")


def test_a_config_read_error_never_changes_which_provider_a_tier_uses(tmp_path, monkeypatch):
    # An ollama persona must never have a background prompt sent to Claude because of a transient read error.
    d = _persona(tmp_path, "claude-sonnet-5", provider="ollama")
    assert type(build_tier_provider(d, TIER_COMPACTION)).__name__ == "OllamaProvider"

    def _boom(cls, path):
        raise PermissionError("file is being replaced")

    monkeypatch.setattr(PersonaConfig, "load", classmethod(_boom))
    with pytest.raises(PermissionError):  # as before this feature: the job fails and retries on its next tick
        build_tier_provider(d, TIER_COMPACTION)
    # the model-only lookups stay fail-soft: they can only choose a model, never a provider
    assert resolve_tier_model(d, TIER_COMPACTION) == (TIER_MODEL[TIER_COMPACTION], "default")


def test_rejection_record_is_cleared_when_the_id_is_confirmed_again():
    model_error.note_failure(SENTENCE.format(model="claude-haiku-5"))
    assert "claude-haiku-5" in model_error.rejected_models()
    model_error.note_success("claude-haiku-5")
    assert "claude-haiku-5" not in model_error.rejected_models()
