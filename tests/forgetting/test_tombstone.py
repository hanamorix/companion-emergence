from brain.forgetting.tombstone import summarise


def test_short_body_returned_unchanged():
    body = "a brief memory."
    assert summarise(body) == body


def test_long_body_truncates_at_first_sentence():
    body = "First sentence here. Second sentence with more detail. Third."
    s = summarise(body, max_chars=50)
    assert s == "First sentence here."


def test_long_body_truncates_at_word_boundary_when_first_sentence_too_long():
    body = "A very very very long sentence with no early punctuation that must be word-truncated somehow"
    s = summarise(body, max_chars=30)
    assert s.endswith("…")
    assert len(s) <= 30
    # Word boundary preserved
    assert not s[:-1].rstrip().endswith(" ")
    # Reasonable prefix kept
    assert s.startswith("A very very")


def test_whitespace_runs_collapsed():
    body = "this  has   multiple\t\twhitespace runs."
    s = summarise(body)
    assert "  " not in s
    assert "\t" not in s
