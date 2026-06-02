from brain.attunement.prompts import build_detector_system_prompt


def test_prompt_unlocks_new_categories_and_relational_rule():
    p = build_detector_system_prompt()
    for cat in ("topic_affinity", "response_shape", "relational"):
        assert cat in p
    assert 'only "tone" and "cadence"' not in p
    assert "relational" in p.lower() and "evidence" in p.lower()
    assert '"evidence"' in p          # evidence is now a list field in the schema
    assert "addressed_pattern_ids" in p
