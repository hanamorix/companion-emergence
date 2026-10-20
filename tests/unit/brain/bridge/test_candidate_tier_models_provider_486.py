"""#486: `candidate_tier_models` applies the same provider guard as `_resolve` and `tier_fallbacks`.

A persona on another provider whose model string looks like a full Claude id got derived tier ids in the "ids whose rejections are
reported and that must not be ignored" set, although no tier would ever run them. One helper, `derived_tier_model`, now answers
"the id derived for this tier, or None" for all three functions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from brain.bridge import model_tier
from brain.bridge.model_tier import (
    ROUTED_CLAUDE_TIERS,
    candidate_tier_models,
    model_for_tier,
    tier_family,
)


def _cfg(provider: str, model: str = "claude-sonnet-5", **extra):
    return SimpleNamespace(provider=provider, model=model, tier_models={}, derived_unavailable=[], **extra)


def _derived_ids(cfg) -> set[str]:
    return {d for t in ROUTED_CLAUDE_TIERS if (d := model_tier.derive_model(cfg.model, tier_family(model_for_tier(t))))}


def test_another_provider_gets_no_derived_ids_in_the_candidate_set():
    cfg = _cfg("ollama")
    defaults = {model_for_tier(t) for t in ROUTED_CLAUDE_TIERS}
    assert _derived_ids(cfg) - defaults  # the model string DOES look like a full Claude id, so ids could be derived

    assert candidate_tier_models(cfg) == defaults  # only the defaults: nothing derived that no tier will ever run


def test_the_default_provider_still_lists_the_derived_ids():
    # PIN
    cfg = _cfg("claude-cli")

    out = candidate_tier_models(cfg)

    assert _derived_ids(cfg) <= out


def test_an_explicit_override_is_still_listed_for_any_provider():
    # PIN
    cfg = _cfg("ollama")
    cfg.tier_models = {ROUTED_CLAUDE_TIERS[0]: "claude-haiku-9"}

    assert "claude-haiku-9" in candidate_tier_models(cfg)


def test_the_shared_helper_answers_none_for_any_provider_but_the_default():
    for provider in ("ollama", "fake"):
        cfg = _cfg(provider)
        assert all(model_tier.derived_tier_model(cfg, tier) is None for tier in ROUTED_CLAUDE_TIERS)
    cfg = _cfg("claude-cli")
    assert {model_tier.derived_tier_model(cfg, tier) for tier in ROUTED_CLAUDE_TIERS} - {None}


def test_the_helper_is_the_only_place_that_derives_a_tier_id():
    import inspect

    src = {f.__name__: inspect.getsource(f) for f in (model_tier._resolve, candidate_tier_models, model_tier.tier_fallbacks)}
    for name, text in src.items():
        assert "derive_model(" not in text, f"{name} must use derived_tier_model, not derive_model directly"


@pytest.mark.parametrize("provider", ["ollama", "fake"])
def test_resolve_never_derives_for_another_provider(provider):
    # PIN: `_resolve` already had the guard
    cfg = _cfg(provider)
    for tier in ROUTED_CLAUDE_TIERS:
        assert model_tier._resolve(cfg, tier)[1] in {"default", "override"}
