import json

from brain.notes.compose import Note, build_note_prompt, parse_note


def test_prompt_includes_interior_and_user():
    p = build_note_prompt(persona_name="Nell", user_name="Hana", dreams_summary="a dream of the sea",
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


def test_make_note_records_an_unparseable_note(tmp_path):
    """#364: make_note still raises; the reply is kept."""
    import pytest

    from brain.health.parse_failures import PARSE_FAILURE_LOG
    from brain.notes.compose import make_note

    class P:
        def complete(self, prompt):
            return "no note today"

    with pytest.raises(ValueError):
        make_note(P(), persona_name="n", user_name="u", dreams_summary="", emotion_summary="",
                  last_session_summary=None, persona_dir=tmp_path)
    (row,) = [json.loads(line) for line in (tmp_path / PARSE_FAILURE_LOG).read_text(encoding="utf-8").splitlines()]
    assert row["site"] == "notes" and row["raw"] == "no note today"
