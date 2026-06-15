import json

from brain.maker.maker import Making, build_making_prompt, parse_making


def test_build_prompt_narrates_charge_sources():
    p = build_making_prompt(
        charge_sources=["grief from the lost dog", "last night's dream"],
        emotion_summary="grief 6, tenderness 3",
    )
    assert "grief from the lost dog" in p
    assert "make what you need" in p.lower()


def test_parse_valid_free_agency_output():
    raw = json.dumps(
        {
            "type": "elegy",
            "title": "For the dog",
            "content": "Soft paws, gone.",
            "disposition": "private",
            "private_reason": "still raw",
        }
    )
    m = parse_making(raw)
    assert isinstance(m, Making)
    assert m.type == "elegy" and m.disposition == "private"
    assert m.private_reason == "still raw"


def test_parse_invalid_disposition_falls_back_to_private():
    raw = json.dumps({"type": "x", "title": "t", "content": "c", "disposition": "PUBLIC!!"})
    m = parse_making(raw)
    assert m.disposition == "private"  # safest default


def test_parse_malformed_raises():
    import pytest

    with pytest.raises(ValueError):
        parse_making("not json at all")
