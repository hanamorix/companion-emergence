"""Snapshot test on the assembled detector system prompt.

Catches prompt-rot during future edits — every load-bearing instruction
(grounding requirement, evidence_quote mandate, category enum, decline-
if-uncertain) must remain literally present. Spec §13 Risk 7.
"""
from brain.attunement.prompts import build_detector_system_prompt


def test_prompt_requires_evidence_quote() -> None:
    prompt = build_detector_system_prompt()
    assert "evidence_quote" in prompt
    assert "verbatim" in prompt.lower()


def test_prompt_requires_evidence_turn_id() -> None:
    prompt = build_detector_system_prompt()
    assert "evidence_turn_id" in prompt


def test_prompt_lists_category_enum_for_alpha_1() -> None:
    prompt = build_detector_system_prompt()
    # alpha.1 positive enum: tone + cadence
    assert "tone" in prompt
    assert "cadence" in prompt
    # The prompt may reference future categories ONLY as exclusions ("do not emit").
    # Verify the schema's allowed list does NOT include future ones — they only
    # appear in negative-instruction contexts.
    schema_block_end = prompt.find("CRITICAL RULES")
    schema_block = prompt[:schema_block_end]
    assert "topic_affinity" not in schema_block
    assert "relational" not in schema_block
    assert "response_shape" not in schema_block


def test_prompt_includes_decline_on_uncertainty_instruction() -> None:
    prompt = build_detector_system_prompt()
    # Must explicitly tell the model to omit candidates rather than fabricate
    assert "decline" in prompt.lower() or "omit" in prompt.lower()
    assert "cannot" in prompt.lower() or "can't" in prompt.lower()


def test_prompt_is_deterministic() -> None:
    """Snapshot equality — the prompt is a constant, not a generator."""
    assert build_detector_system_prompt() == build_detector_system_prompt()
