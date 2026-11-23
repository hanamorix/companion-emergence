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


def test_only_registered_names_with_a_usable_intensity_are_kept():
    reply = '{"joy": 12, "trust": "high", "not_a_feeling": 5, "awe": 0, "grief": -2}'
    assert make_emotion_tagger(_Provider(reply))(_memory()) == {"joy": 10.0}


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


def test_normalize_drops_names_that_are_not_registered_and_values_that_are_not_finite_positive_numbers():
    vocab = _loaded()  # the registry decides what is registered; `vocab` only says it loaded
    raw = {"joy": float("nan"), "awe": float("inf"), "x": 3, "ok": "5"}
    assert _normalize(raw, vocab) == {}
    assert _normalize({"joy": "5", "awe": 11.5}, vocab) == {"joy": 5.0, "awe": 10.0}
    assert _normalize({"anything": 3}, frozenset()) == {"anything": 3.0}  # no vocabulary loaded: every name allowed


# --- the name matching (HC2): case, separators between a blend's parts, part order, pooling ----------------------------------

_KEEPS_TODAYS_BEHAVIOUR = "pins behaviour HC2 keeps unchanged (the exact-match test it replaces already gave this result), so it passes on the old code by design"


def _extension(name):
    return vocabulary.Emotion(name=name, description="x", category="persona_extension", decay_half_life_days=14.0, intensity_clamp=10.0)


@pytest.fixture
def extensions():
    """A registered blend and two extensions whose own names hold a space or a double underscore (the registry validates nothing)."""
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
    assert _normalize({spelling: 4}, _loaded()) == {"joy": 4.0}


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
    assert _normalize({spelling: 6}, _loaded()) == {"grief_love_blend": 6.0}


@pytest.mark.parametrize("spelling", ["love_grief_blend", "love-grief blend", "LOVE / GRIEF / BLEND", "love__grief--blend"])
def test_reordered_blend_parts_match_the_registered_blend(extensions, spelling):
    assert _normalize({spelling: 3}, _loaded()) == {"grief_love_blend": 3.0}


def test_the_stored_key_is_the_registered_name(extensions):
    kept = _normalize({"Love Grief Blend": 2, "Comfort-Seeking": 3}, _loaded())
    assert set(kept) == {"grief_love_blend", "comfort_seeking"}
    for key in kept:
        found = vocabulary.get(key)
        assert found is not None and found.name == key


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"joy": 2, "Joy": 7}, {"joy": 7.0}),
        ({"Joy": 7, "joy": 2}, {"joy": 7.0}),
        ({"grief love blend": 2, "love_grief_blend": 5}, {"grief_love_blend": 5.0}),
        ({"grief love blend": 5, "grief_love_blend": 2}, {"grief_love_blend": 5.0}),
        ({"joy-": 2, "joy!": 9}, {"joy": 9.0}),
        ({"joy-": 8, "joy": 3, "-joy": 4}, {"joy": 8.0}),
        ({"Joy": 15, "JOY": 11}, {"joy": 10.0}),
    ],
    ids=["case-first-smaller", "case-first-larger", "order", "separator-first-larger", "same-stepped-name-later-larger", "same-stepped-name-three", "clamped"],
)
def test_two_spellings_of_one_emotion_pool_and_the_larger_value_is_kept(extensions, raw, expected):
    assert _normalize(raw, _loaded()) == expected


def test_a_spelling_with_an_unusable_value_does_not_hide_a_valid_one(extensions):
    assert _normalize({"Joy": "high", "JOY": 4, "joy-": float("nan"), "joy ": 0}, _loaded()) == {"joy": 4.0}


@pytest.mark.parametrize("name", ["not_a_feeling", "---", "", "blend", "_blend_", "joy2", "joy joy", "love_grief_grief_blend", "gr1ef love blend"])
def test_a_name_that_is_not_registered_whatever_its_spelling_is_dropped(extensions, name):
    assert _normalize({name: 5, "Joy": 1}, _loaded()) == {"joy": 1.0}


@pytest.mark.no_base_failure(reason=_KEEPS_TODAYS_BEHAVIOUR)
def test_an_extension_whose_own_name_holds_a_space_or_double_underscore_still_matches_when_echoed(extensions):
    assert _normalize({"odd name": 2, "odd__pair": 3}, _loaded()) == {"odd name": 2.0, "odd__pair": 3.0}


@pytest.mark.no_base_failure(reason=_KEEPS_TODAYS_BEHAVIOUR)
def test_with_no_vocabulary_loaded_names_are_kept_as_given_and_the_lookup_is_not_called(monkeypatch):
    def boom(name):
        raise AssertionError("the vocabulary lookup must not run when the vocabulary could not be loaded")

    monkeypatch.setattr(vocabulary, "get", boom)
    raw = {"Grief-Love Blend": 2, "joy": 20, "x": "no", "nan": float("nan"), "---": 1}
    assert _normalize(raw, frozenset()) == {"Grief-Love Blend": 2.0, "joy": 10.0, "---": 1.0}


def test_the_tagger_stores_registered_names_for_a_reply_with_variant_spellings(monkeypatch):
    monkeypatch.setattr(emotion_tagger, "_load_vocab", _loaded)
    reply = '{"Joy": 3, "joy": 5, "Comfort-Seeking": 2}'
    assert make_emotion_tagger(_Provider(reply))(_memory()) == {"joy": 5.0, "comfort_seeking": 2.0}


@pytest.mark.no_base_failure(reason=_KEEPS_TODAYS_BEHAVIOUR)
def test_a_tagger_whose_vocabulary_cannot_load_keeps_every_name(monkeypatch):
    def broken():
        raise RuntimeError("no vocabulary")

    monkeypatch.setattr("brain.emotion.vocabulary.list_all", broken)
    assert emotion_tagger._load_vocab() == frozenset()
    assert make_emotion_tagger(_Provider('{"Anything Goes": 3}'))(_memory()) == {"Anything Goes": 3.0}


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
    row = _memory(emotions={"trust": 3.0})
    store.create(row, fold=False)
    tagger = _tagger({"joy": 6.0})
    assert tag_if_feelingless(store, tagger, row) is False
    assert tagger.calls == []
    assert store.get(row.id, bump=False)._stored_emotions == {"trust": 3.0}


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
        store.update(memory.id, emotions={"trust": 2.0}, fold=False)  # another writer lands during the slow call
        return {"joy": 9.0}

    assert tag_if_feelingless(store, tagger, row) is False
    assert store.get(row.id, bump=False)._stored_emotions == {"trust": 2.0}


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
