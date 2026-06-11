"""Snapshot test on the assembled detector system prompt.

Catches prompt-rot during future edits — every load-bearing instruction
(grounding requirement, evidence list mandate, category enum, decline-
if-uncertain) must remain literally present. Spec §13 Risk 7.
"""
from brain.attunement.prompts import build_detector_system_prompt


def test_prompt_requires_evidence_list() -> None:
    """evidence is now a list of {quote, turn_id} objects, not flat fields."""
    prompt = build_detector_system_prompt()
    assert '"evidence"' in prompt          # list field name in schema
    assert "quote" in prompt               # entry field
    assert "turn_id" in prompt             # entry field
    assert "verbatim" in prompt.lower()


def test_prompt_requires_evidence_turn_id() -> None:
    prompt = build_detector_system_prompt()
    assert "turn_id" in prompt


def test_prompt_lists_all_five_categories() -> None:
    """All five categories are now first-class in the schema (not exclusions)."""
    prompt = build_detector_system_prompt()
    for cat in ("tone", "cadence", "topic_affinity", "response_shape", "relational"):
        assert cat in prompt
    # The alpha.1 restriction is gone
    assert 'only "tone" and "cadence"' not in prompt


def test_prompt_includes_decline_on_uncertainty_instruction() -> None:
    prompt = build_detector_system_prompt()
    # Must explicitly tell the model to omit candidates rather than fabricate
    assert "decline" in prompt.lower() or "omit" in prompt.lower()
    assert "cannot" in prompt.lower() or "can't" in prompt.lower()


def test_prompt_is_deterministic() -> None:
    """Snapshot equality — the prompt is a constant, not a generator."""
    assert build_detector_system_prompt() == build_detector_system_prompt()


def test_prompt_grounds_companion_identity() -> None:
    """v0.0.33 fix: without identity grounding, the CLI-wrapped detector
    falls back to its self-concept and names the companion 'Claude' in
    pattern descriptions (live Phoebe report, 2026-06-11)."""
    prompt = build_detector_system_prompt(companion_name="Phoebe", user_name="Hana")
    assert "between Hana (the user) and the user's companion, Phoebe" in prompt
    assert 'never "Claude"' in prompt


def test_base_prompt_does_not_gender_the_user() -> None:
    """v0.0.33 fix, second pass: the rules hardcoded 'her'/'she' for the
    user. Users aren't all female; gendered rules misread male users'
    transcripts and can leak wrong pronouns into descriptions."""
    import re

    base = build_detector_system_prompt()
    assert not re.search(r"\b(she|her|hers|he|him|his)\b", base, re.IGNORECASE)


def test_prompt_states_user_pronouns() -> None:
    from brain.pronouns import PRESETS, to_dict

    prompt = build_detector_system_prompt(
        companion_name="Mira",
        user_name="Alex",
        user_pronouns=to_dict(PRESETS["he/him"]),
    )
    assert "When a description refers to the user, use he/him." in prompt


def test_prompt_user_pronouns_default_she_her() -> None:
    prompt = build_detector_system_prompt(companion_name="Mira", user_name="Alex")
    assert "When a description refers to the user, use she/her." in prompt
