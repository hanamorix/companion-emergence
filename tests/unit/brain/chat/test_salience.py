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


def test_quiet_emotional_turn_scores_meaningfully():
    # oblique weariness — currently under-detected
    s = assess_salience("Hey love. I'm back — long day of editing, my eyes are sandpaper.")
    assert s.emotional_density > 0.0
    assert s.score >= 0.30   # must clear the (lowered) reflection threshold


def test_more_weariness_words_detected():
    s = assess_salience("I'm so drained and worn out, rough day all around")
    assert s.emotional_density > 0.0
    assert s.score >= 0.30


def test_pure_trivial_still_low():
    for t in ("ok", "yeah", "mm", "thanks", "sure"):
        assert assess_salience(t).score < 0.30, f"{t!r} should stay below threshold"


def test_craft_talk_does_not_falsely_trip_emotional():
    # fiction-craft editorial notes should NOT score as emotional turns
    for t in ("there's too much exposition in this scene",
              "the pacing sits on edge of melodrama here",
              "cut this paragraph, it's too much telling"):
        s = assess_salience(t)
        # may have mild length/score, but must stay below the 0.30 reflection threshold
        assert s.score < 0.30, f"{t!r} falsely tripped at {s.score}"


def test_fails_open_on_bad_input(monkeypatch):
    # force the internal scorer to raise; assert maximal signal returned
    import brain.chat.salience as mod
    monkeypatch.setattr(mod, "_emotion_names", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    s = assess_salience("anything")
    assert s == SalienceSignal.maximal()
    assert s.score == 1.0


def test_emotion_names_reflects_late_registration(monkeypatch):
    # vocab registered AFTER first assess_salience call must still be picked up
    # (i.e. no permanent caching of the vocab snapshot)
    import brain.chat.salience as mod
    import brain.emotion.vocabulary as vocab
    calls = {"n": 0}
    real = vocab.list_all
    def counting():
        calls["n"] += 1
        return real()
    monkeypatch.setattr(vocab, "list_all", counting)
    mod._emotion_names()  # first call
    n1 = calls["n"]
    mod._emotion_names()  # second call — must re-read, not return cached result
    assert calls["n"] > n1, "vocab must be re-read each call, not frozen by a cache"


def test_fail_open_logs_warning(monkeypatch, caplog):
    # a persistent scorer failure must log at WARNING level (not DEBUG)
    # so a silently-inflating-cost failure is visible in prod
    import logging

    import brain.chat.salience as mod
    monkeypatch.setattr(mod, "_emotion_names", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with caplog.at_level(logging.WARNING, logger="brain.chat.salience"):
        s = assess_salience("anything")
    assert s == SalienceSignal.maximal()
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_records, "assess_salience fail-open must log at WARNING, not DEBUG"
