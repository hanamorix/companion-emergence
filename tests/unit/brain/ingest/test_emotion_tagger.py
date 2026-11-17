"""The emotion tagger (#747): the model call that tags a memory without feelings, and the helper that writes the tags.

Synthetic content only (synthetic user Bob, persona Canary); a scripted provider, never a real model.
"""

from __future__ import annotations

import logging

import pytest

from brain.bridge.cli_throttle import ThrottleDeferred
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
    # A literal copy of the strings as they stood in prompt_strings.toml [ingest.emotion_backfill] before the move.
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
    vocab = frozenset({"joy", "trust"})
    raw = {"joy": float("nan"), "trust": float("inf"), "x": 3, "ok": "5"}
    assert _normalize(raw, vocab) == {}
    assert _normalize({"joy": "5", "trust": 11.5}, vocab) == {"joy": 5.0, "trust": 10.0}
    assert _normalize({"anything": 3}, frozenset()) == {"anything": 3.0}  # no vocabulary loaded: every name allowed


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
