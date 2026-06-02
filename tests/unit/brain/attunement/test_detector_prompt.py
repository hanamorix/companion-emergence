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
