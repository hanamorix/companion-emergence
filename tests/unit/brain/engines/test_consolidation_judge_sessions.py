"""S71: the memory-sorting step judges its items in one live Claude session per instruction kind per run, cleared between items.

Real `run_consolidation`, real `ClaudeCliProvider`, a fake `claude` that is a real child process keeping a conversation history
like the real CLI (`tests/fake_claude_session.py`). The proof that no earlier item is in context is what each PROCESS logged
it held when it answered: for every item of both kinds, exactly that item's own message and nothing before it.
"""

from __future__ import annotations

import os
import sys

import pytest

import brain.engines.consolidation as cons
from brain.bridge import cli_throttle
from brain.bridge.provider import ClaudeCliProvider
from brain.engines.consolidation import run_consolidation
from brain.memory.hebbian import HebbianMatrix
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore
from tests.fake_claude_session import install

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the fake claude is a #! script")

A, B, C = "alpha candidate text", "bravo candidate text", "charlie candidate text"


@pytest.fixture
def persona(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    heb = HebbianMatrix(tmp_path / "hebbian.db")
    yield tmp_path, store, heb, PendingQueue(tmp_path)
    store.close()
    heb.close()


def _mem(content: str) -> Memory:
    return Memory.create_new(content=content, memory_type="dream", domain="us", emotions={})


def _stored(store, content: str) -> Memory:
    memory = _mem(content)
    store.create(memory)
    return memory


def _run(tmp, store, heb, **kw):
    return run_consolidation(
        store,
        persona_dir=tmp,
        provider=ClaudeCliProvider(timeout_seconds=10),
        hebbian=heb,
        emotion_tagger=lambda _m: {},  # the tagger is a third instruction kind and not under test
        **kw,
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_run_judges_every_item_of_both_kinds_alone_in_one_process_per_kind(persona, tmp_path, monkeypatch):
    tmp, store, heb, queue = persona
    fake = install(tmp_path, monkeypatch)
    first, second = _stored(store, "stored memory one"), _stored(store, "stored memory two")
    for text in (A, B, C):
        queue.enqueue(_mem(text), source="dream")
    queue.enqueue_reappraisal(first.id, source="recall")
    queue.enqueue_reappraisal(second.id, source="recall")

    result = _run(tmp, store, heb)

    assert result.batch == 3
    spawns = fake.events("spawn")
    assert len(spawns) == 2  # one live process per instruction kind, not one per item (5)
    assert {s["system"] for s in spawns} == {cons._CLASSIFIER_PROMPT, cons._REAPPRAISER_PROMPT}
    answers = fake.answers()
    assert len(answers) == 5
    # No earlier item in context: every answer was given by a process that held exactly one message.
    assert all(len(a["held"]) == 1 for a in answers)
    # ... and it was that item's own message.
    # (The re-checks run first, then the new candidates, in queue order.)
    own = ["stored memory one", "stored memory two", A, B, C]
    assert [text in answer["held"][0] for text, answer in zip(own, answers, strict=True)] == [True] * 5
    # The step is over: both processes are gone.
    assert not any(_alive(s["pid"]) for s in spawns)
    assert len(fake.events("exit")) == 2


def test_a_run_with_one_kind_opens_one_process_and_a_run_with_nothing_opens_none(persona, tmp_path, monkeypatch):
    tmp, store, heb, queue = persona
    fake = install(tmp_path, monkeypatch)
    _run(tmp, store, heb)  # nothing queued
    assert fake.events("spawn") == []
    for text in (A, B):
        queue.enqueue(_mem(text), source="dream")
    _run(tmp, store, heb)
    assert [s["system"] for s in fake.events("spawn")] == [cons._CLASSIFIER_PROMPT]
    assert len(fake.answers()) == 2 and all(len(a["held"]) == 1 for a in fake.answers())


def test_the_sessions_are_closed_when_the_run_is_deferred_part_way(persona, tmp_path, monkeypatch):
    tmp, store, heb, queue = persona
    fake = install(tmp_path, monkeypatch)
    for text in (A, B, C):
        queue.enqueue(_mem(text), source="dream")
    # The bridge starts stopping once the first item has been answered: the second call is deferred ("not now").
    monkeypatch.setattr(cli_throttle, "shutdown_requested", lambda: len(fake.answers()) >= 1)

    _run(tmp, store, heb)

    assert len(fake.answers()) == 1
    assert not any(_alive(s["pid"]) for s in fake.events("spawn"))
    assert len(fake.events("exit")) == 1


def test_the_sessions_are_closed_when_the_run_is_killed_by_a_base_exception(persona, tmp_path, monkeypatch):
    tmp, store, heb, queue = persona
    fake = install(tmp_path, monkeypatch)
    for text in (A, B):
        queue.enqueue(_mem(text), source="dream")
    real_create = store.create

    def _kill_on_b(memory, *a, **k):
        if memory.content == B:
            raise KeyboardInterrupt
        return real_create(memory, *a, **k)

    monkeypatch.setattr(store, "create", _kill_on_b)

    with pytest.raises(KeyboardInterrupt):
        _run(tmp, store, heb)

    assert len(fake.answers()) == 2
    assert not any(_alive(s["pid"]) for s in fake.events("spawn"))
    assert len(fake.events("exit")) == 1


def test_a_provider_that_is_not_the_claude_cli_judges_each_item_with_one_generate_call(persona):
    tmp, store, heb, queue = persona
    for text in (A, B):
        queue.enqueue(_mem(text), source="dream")

    class _Stub:
        def __init__(self):
            self.calls = []

        def generate(self, prompt, *, system=None):
            self.calls.append((system, prompt))
            return '{"verdict": "new"}'

    stub = _Stub()
    run_consolidation(store, persona_dir=tmp, provider=stub, hebbian=heb, emotion_tagger=lambda _m: {})

    assert [system for system, _ in stub.calls] == [cons._CLASSIFIER_PROMPT] * 2
