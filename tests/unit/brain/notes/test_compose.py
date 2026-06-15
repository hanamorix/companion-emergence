import json

from brain.notes.compose import Note, build_note_prompt, parse_note


def test_prompt_includes_interior_and_user():
    p = build_note_prompt(user_name="Hana", dreams_summary="a dream of the sea",
                          emotion_summary="tenderness 4", last_session_summary="we talked about her book")
    assert "Hana" in p and "the sea" in p and "write" in p.lower()


def test_parse_valid_note():
    raw = json.dumps({"subject": "the sea", "body": "I dreamt of the sea and thought of you."})
    n = parse_note(raw)
    assert isinstance(n, Note)
    assert n.subject == "the sea"
    assert "sea" in n.body


def test_parse_malformed_raises():
    import pytest
    with pytest.raises(ValueError):
        parse_note("not json")
