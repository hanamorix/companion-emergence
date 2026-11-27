"""The emotion tagger (#747): the model call that tags a memory without feelings, and the helper that writes the tags.

Synthetic content only (synthetic user Bob, persona Canary); a scripted provider, never a real model.
"""

from __future__ import annotations

import logging

import pytest

from brain.bridge.cli_throttle import ThrottleDeferred
from brain.emotion import vocabulary
from brain.ingest import emotion_tagger
from brain.ingest.emotion_tagger import _normalize, make_emotion_tagger, tag_if_feelingless
from brain.memory.store import Memory, MemoryStore
from brain.utils import log_throttle


class _Provider:
    """Records each ``generate`` call and answers with a scripted reply (or raises it)."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt, *, system=None):
        self.calls.append((prompt, system))
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


def _memory(content="Bob told Canary about his week", **kwargs):
    return Memory.create_new(content=content, memory_type="conversation", domain="us", **kwargs)


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


# --- the tagger: the two persona-facing strings move unchanged, and the reply is read tolerantly -------------------------


def test_the_tagger_sends_the_backfills_exact_system_prompt_and_prompt(monkeypatch):
    # A literal copy of the strings as they stood in prompt_strings.toml [ingest.emotion_backfill] before the move (the section is now [ingest.emotion_tagger]).
    system = (
        "You are an emotion-tagging assistant. Return a JSON object mapping emotion names to intensities (0–10). "
        "Omit emotions with intensity 0. Return ONLY the JSON object, no prose, no markdown fences."
    )
    prompt = (
        "Return a JSON object mapping emotion names to intensities (0-10) for the following memory. "
        "Use ONLY names from this list: joy, trust. Omit emotions with intensity 0. "
        "Return ONLY the JSON object, no other text.\n\nMemory: Bob told Canary about his week"
    )
    monkeypatch.setattr(emotion_tagger, "_load_vocab", lambda: frozenset({"joy", "trust"}))
    provider = _Provider('{"joy": 4}')
    assert make_emotion_tagger(provider)(_memory()) == {"joy": 4.0}
    assert provider.calls == [(prompt, system)]


@pytest.mark.parametrize(
    "reply",
    [
        '{"joy": 6, "awe": 2.5}',
        '```json\n{"joy": 6, "awe": 2.5}\n```',
        'Here are the feelings: {"joy": 6, "awe": 2.5} Hope that helps.',
    ],
    ids=["bare", "fenced", "prose-around"],
)
def test_a_fenced_or_wrapped_reply_is_parsed(reply):
    assert make_emotion_tagger(_Provider(reply))(_memory()) == {"joy": 6.0, "awe": 2.5}


def test_names_are_not_filtered_but_values_must_be_usable():
    """The registered-name filter is gone (#801 section 6): the store's write checks a name against her vocabulary."""
    reply = '{"joy": 12, "trust": "high", "not_a_feeling": 5, "awe": 0, "grief": -2}'
    assert make_emotion_tagger(_Provider(reply))(_memory()) == {"joy": 10.0, "not_a_feeling": 5.0}


@pytest.mark.parametrize("reply", ["no object here", "{not json", "[1, 2]", ""], ids=["prose", "broken", "array", "empty"])
def test_a_reply_without_a_usable_object_tags_nothing(reply):
    assert make_emotion_tagger(_Provider(reply))(_memory()) == {}


def test_a_provider_error_tags_nothing_and_a_deferred_call_is_not_swallowed():
    assert make_emotion_tagger(_Provider(RuntimeError("cli failed")))(_memory()) == {}
    with pytest.raises(ThrottleDeferred):
        make_emotion_tagger(_Provider(ThrottleDeferred("not now")))(_memory())


def test_a_provider_that_fails_on_every_call_warns_once_not_on_every_write(caplog):
    log_throttle._last.clear()
    tagger = make_emotion_tagger(_Provider(RuntimeError("cli failed")))
    with caplog.at_level(logging.DEBUG, logger="brain.ingest.emotion_tagger"):
        for _ in range(3):
            assert tagger(_memory()) == {}
    levels = [r.levelno for r in caplog.records if "tagger error" in r.getMessage()]
    assert levels == [logging.WARNING, logging.DEBUG, logging.DEBUG]


# --- HC3: a reply with no usable object says which kind of failure it was ------------------------------------------------


def _reply_records(caplog, reply, memory=None):
    """Run one tagging call on ``reply`` and return the records of the two no-object lines."""
    log_throttle._last.clear()
    memory = memory or _memory()
    with caplog.at_level(logging.DEBUG, logger="brain.ingest.emotion_tagger"):
        assert make_emotion_tagger(_Provider(reply))(memory) == {}
    return memory, [r for r in caplog.records if "empty reply" in r.getMessage() or "no JSON object" in r.getMessage()]


@pytest.mark.parametrize("reply", ["", "  \n\t "], ids=["empty", "whitespace-only"])
def test_an_empty_reply_logs_the_empty_reply_line(caplog, reply):
    memory, records = _reply_records(caplog, reply)
    assert [(r.levelno, r.getMessage(), bool(r.exc_info)) for r in records] == [
        (logging.WARNING, f"emotion_tagger: empty reply for memory {memory.id}", False)
    ]


@pytest.mark.parametrize("reply", ["I cannot tag this.", "{not json", "[1, 2]", '{"joy": 4'], ids=["prose", "unclosed", "array", "truncated"])
def test_a_reply_with_no_json_object_logs_the_no_object_line_with_its_length_and_none_of_its_text(caplog, reply):
    memory, records = _reply_records(caplog, reply)
    assert [(r.levelno, r.getMessage(), bool(r.exc_info)) for r in records] == [
        (logging.WARNING, f"emotion_tagger: reply held no JSON object for memory {memory.id} ({len(reply)} characters)", False)
    ]
    assert reply not in records[0].getMessage()


def test_the_two_kinds_have_their_own_throttle_keys(caplog):
    log_throttle._last.clear()
    with caplog.at_level(logging.DEBUG, logger="brain.ingest.emotion_tagger"):
        for reply in ("", "prose", "", "prose"):
            assert make_emotion_tagger(_Provider(reply))(_memory()) == {}
    levels = [(r.getMessage().split(" for memory")[0], r.levelno) for r in caplog.records]
    assert levels == [
        ("emotion_tagger: empty reply", logging.WARNING),
        ("emotion_tagger: reply held no JSON object", logging.WARNING),
        ("emotion_tagger: empty reply", logging.DEBUG),
        ("emotion_tagger: reply held no JSON object", logging.DEBUG),
    ]


@pytest.mark.no_base_failure(reason="pins that a reply with an object logs no failure line; base logs nothing there either. Bites a mutant that logs on every reply (checked by hand)")
@pytest.mark.parametrize(
    "reply",
    ["{}", '{"joy": 4}', '```json\n{"joy": 4}\n```', 'Feelings: {"joy": 4} done', '{"joy": 0}'],
    ids=["empty-object", "bare", "fenced", "prose-around", "zero-intensity"],
)
def test_a_reply_that_holds_an_object_logs_no_failure_line(caplog, reply):
    log_throttle._last.clear()
    with caplog.at_level(logging.DEBUG, logger="brain.ingest.emotion_tagger"):
        make_emotion_tagger(_Provider(reply))(_memory())
    assert [r for r in caplog.records if "empty reply" in r.getMessage() or "no JSON object" in r.getMessage()] == []


@pytest.mark.no_base_failure(reason="pins that a failed call or an invalid object keeps the one tagger error line and gets no new line; base does the same. Bites a mutant that logs on every reply (checked by hand)")
def test_a_balanced_invalid_object_and_a_provider_failure_log_only_the_tagger_error_line(caplog):
    for reply in ("{not json}", RuntimeError("connection reset"), None):
        log_throttle._last.clear()
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="brain.ingest.emotion_tagger"):
            assert make_emotion_tagger(_Provider(reply))(_memory()) == {}
        assert [r.getMessage().split(":")[1].strip() for r in caplog.records] == ["tagger error"]


def test_normalize_drops_values_that_are_not_finite_positive_numbers_and_keeps_every_name():
    raw = {"joy": float("nan"), "awe": float("inf"), "x": 3, "ok": "5"}
    assert _normalize(raw) == {"x": 3.0, "ok": 5.0}
    assert _normalize({"joy": "5", "awe": 11.5}) == {"joy": 5.0, "awe": 10.0}
    assert _normalize({"anything": 3}) == {"anything": 3.0}


# --- the name matching (HC2): case, separators between a blend's parts, part order, pooling ----------------------------------

def _extension(name):
    return vocabulary.Emotion(name=name, description="x", category="persona_extension", decay_half_life_days=14.0, intensity_clamp=10.0)


@pytest.fixture
def extensions():
    """A registered blend and two extensions registered with a space or a double underscore in their names (stored tidied)."""
    names = ["love_grief_blend", "odd name", "odd__pair"]
    for name in names:
        vocabulary.register(_extension(name))
    try:
        yield
    finally:
        for name in names:
            vocabulary._unregister(name)


def _loaded():
    return frozenset(e.name for e in vocabulary.list_all())


@pytest.mark.parametrize("spelling", ["Joy", "JOY", "jOy"])
def test_a_feeling_name_matches_whatever_its_case(spelling):
    assert _normalize({spelling: 4}) == {"joy": 4.0}


@pytest.mark.parametrize(
    "spelling",
    [
        "grief-love_blend",
        "grief love blend",
        "grief/love/blend",
        "grief+love+blend",
        "grief --/ love blend",
        "grief_love__blend",
        "grief love - blend",
        "grief-love-blend",
        "-grief_love_blend-",
        "  _grief love blend_ ",
        "Grief-Love Blend",
    ],
)
def test_any_run_of_separators_between_a_blends_parts_and_before_blend_matches(extensions, spelling):
    assert _normalize({spelling: 6}) == {"grief_love_blend": 6.0}


@pytest.mark.parametrize("spelling", ["love_grief_blend", "love-grief blend", "LOVE / GRIEF / BLEND", "love__grief--blend"])
def test_reordered_blend_parts_match_the_registered_blend(extensions, spelling):
    assert _normalize({spelling: 3}) == {"grief_love_blend": 3.0}


def test_the_stored_key_is_the_tidied_form_of_the_name(extensions):
    """The tagger keys by the tidied form; the store's write moves the stored spelling onto the entry's (#801 section 6)."""
    kept = _normalize({"Love Grief Blend": 2, "Comfort-Seeking": 3})
    assert set(kept) == {"grief_love_blend", "comfort_seeking"}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"joy": 2, "Joy": 7}, {"joy": 4.5}),
        ({"Joy": 7, "joy": 2}, {"joy": 4.5}),
        ({"grief love blend": 2, "love_grief_blend": 5}, {"grief_love_blend": 3.5}),
        ({"grief love blend": 5, "grief_love_blend": 2}, {"grief_love_blend": 3.5}),
        ({"joy-": 2, "joy!": 8}, {"joy": 5.0}),
        ({"joy-": 8, "joy": 3, "-joy": 4}, {"joy": 5.0}),
        ({"Joy": 15, "JOY": 11}, {"joy": 10.0}),
    ],
    ids=["case-first-smaller", "case-first-larger", "order", "separator-first-larger", "same-stepped-name-later-larger", "same-stepped-name-three", "clamped"],
)
def test_two_spellings_of_one_emotion_are_averaged(extensions, raw, expected):
    """Two spellings of one feeling in one reply carry their mean (#801 section 6, S8), not the larger value."""
    assert _normalize(raw) == expected


def test_a_spelling_with_an_unusable_value_does_not_hide_a_valid_one(extensions):
    assert _normalize({"Joy": "high", "JOY": 4, "joy-": float("nan"), "joy ": 0}) == {"joy": 4.0}


@pytest.mark.parametrize("name", ["---", "", "_", "  "])
def test_a_name_that_tidies_to_nothing_is_dropped(name):
    assert _normalize({name: 5, "Joy": 1}) == {"joy": 1.0}


def test_the_docstring_no_longer_says_the_larger_value_is_kept():
    assert "larger" not in (_normalize.__doc__ or "")


def test_the_tagger_returns_variant_spellings_pooled_by_mean(monkeypatch):
    monkeypatch.setattr(emotion_tagger, "_load_vocab", _loaded)
    reply = '{"Joy": 3, "joy": 5, "Comfort-Seeking": 2}'
    assert make_emotion_tagger(_Provider(reply))(_memory()) == {"joy": 4.0, "comfort_seeking": 2.0}


def test_a_tagger_whose_vocabulary_cannot_load_still_asks_for_any_name(monkeypatch):
    def broken():
        raise RuntimeError("no vocabulary")

    monkeypatch.setattr("brain.emotion.vocabulary.list_all", broken)
    assert emotion_tagger._load_vocab() == frozenset()
    provider = _Provider('{"Anything Goes": 3}')
    assert make_emotion_tagger(provider)(_memory()) == {"anything_goes": 3.0}
    assert "(any emotion name)" in provider.calls[0][0]


# --- tag_if_feelingless --------------------------------------------------------------------------------------------------


def _tagger(tags):
    calls: list[str] = []

    def tagger(memory):
        calls.append(memory.id)
        return dict(tags)

    tagger.calls = calls
    return tagger


def test_a_row_without_feelings_is_tagged_with_a_store_update_that_folds(store, emotion_engine_stub):
    row = _memory()
    store.create(row, fold=False)
    tagger = _tagger({"joy": 6.0})
    assert tag_if_feelingless(store, tagger, row) is True
    got = store.get(row.id, bump=False)
    assert got._stored_emotions == {"joy": 6.0}
    assert got.recall_count == 0.0  # the row was read without a recall bump
    assert [c[0] for c in emotion_engine_stub.recorded_calls] == ["on_emotions_assigned"]  # `store.update` fired the hook once


def test_the_tags_mood_event_carries_the_rows_rowid(store, emotion_engine_stub):
    row = _memory()
    store.create(row, fold=False)
    rowid = store._conn.execute("SELECT rowid FROM memories WHERE id = ?", (row.id,)).fetchone()[0]
    assert tag_if_feelingless(store, _tagger({"joy": 6.0}), row) is True
    assigned = [c for c in emotion_engine_stub.recorded_calls if c[0] == "on_emotions_assigned"]
    assert len(assigned) == 1
    assert assigned[0][1].id == row.id
    assert assigned[0][1].rowid == rowid  # the engine's `assigned` event takes this rowid (`engine.py` `on_emotions_assigned`)


def test_a_row_with_feelings_is_not_sent_to_the_tagger(store):
    row = _memory(emotions={"awe": 3.0})
    store.create(row, fold=False)
    tagger = _tagger({"joy": 6.0})
    assert tag_if_feelingless(store, tagger, row) is False
    assert tagger.calls == []
    assert store.get(row.id, bump=False)._stored_emotions == {"awe": 3.0}


def test_no_tagger_means_nothing_happens(store):
    row = _memory()
    store.create(row, fold=False)
    assert tag_if_feelingless(store, None, row) is False
    assert store.get(row.id, bump=False)._stored_emotions == {}


def test_a_tagger_that_returns_nothing_writes_nothing(store, monkeypatch):
    row = _memory()
    store.create(row, fold=False)
    written: list[dict] = []
    real_update = MemoryStore.update
    monkeypatch.setattr(MemoryStore, "update", lambda self, mid, **f: (written.append(f), real_update(self, mid, **f))[1])
    assert tag_if_feelingless(store, _tagger({}), row) is False
    assert written == []
    assert store.get(row.id, bump=False)._stored_emotions == {}


def test_feelings_written_while_the_tagger_was_working_are_kept(store):
    row = _memory()
    store.create(row, fold=False)

    def tagger(memory):
        store.update(memory.id, emotions={"awe": 2.0}, fold=False)  # another writer lands during the slow call
        return {"joy": 9.0}

    assert tag_if_feelingless(store, tagger, row) is False
    assert store.get(row.id, bump=False)._stored_emotions == {"awe": 2.0}


def test_a_row_deleted_while_the_tagger_was_working_is_not_written(store):
    row = _memory()
    store.create(row, fold=False)

    def tagger(memory):
        store._conn.execute("DELETE FROM memories WHERE id = ?", (memory.id,))
        store._conn.commit()
        return {"joy": 9.0}

    assert tag_if_feelingless(store, tagger, row) is False
    assert store.get(row.id, bump=False) is None


def test_the_helper_raises_what_the_tagger_raises(store):
    row = _memory()
    store.create(row, fold=False)

    def tagger(_memory_row):
        raise ThrottleDeferred("not now")

    with pytest.raises(ThrottleDeferred):
        tag_if_feelingless(store, tagger, row)
    assert store.get(row.id, bump=False)._stored_emotions == {}
