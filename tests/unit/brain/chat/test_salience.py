from brain.chat.salience import SalienceSignal, assess_salience


def test_trivial_turn_is_low_salience():
    s = assess_salience("ok", prior_user_text="how are you?")
    assert s.score < 0.3
    assert not s.references_past and not s.mentions_file_or_path


def test_past_reference_flags_and_raises_score():
    s = assess_salience("remember what you said last time about the manuscript?")
    assert s.references_past and s.is_question
    assert s.score >= 0.5


def test_file_mention_flagged():
    s = assess_salience("can you read ~/Desktop/notes.txt for me")
    assert s.mentions_file_or_path


def test_topic_shift_detected():
    s = assess_salience("let's talk about the tax filing deadline", prior_user_text="i love you")
    assert s.topic_shift


def test_emotional_density_scores():
    s = assess_salience("I feel so sad and lonely and afraid")
    assert s.emotional_density > 0.0
    assert s.score > 0.25  # density contributes


def test_fails_open_on_bad_input(monkeypatch):
    # force the internal scorer to raise; assert maximal signal returned
    import brain.chat.salience as mod
    monkeypatch.setattr(mod, "_emotion_names", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    s = assess_salience("anything")
    assert s == SalienceSignal.maximal()
    assert s.score == 1.0
