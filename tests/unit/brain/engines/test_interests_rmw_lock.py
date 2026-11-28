"""interests.json read-modify-write gaps closed (#650 finding 5, L2 items 5 and 32).

Pass 2 runs on its own thread, so its interest spawn can overlap the heartbeat's keyword bump and the research step and the weekly
sweep (both of which call a model between their read and their save). Every writer takes `interests_lock`; the two that call a
model re-read under it and apply their own change to the fresh copy. Fake providers, synthetic tmp persona dirs, no model call.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.bridge.provider import FakeProvider
from brain.engines import interest_sweep
from brain.engines._interests import InterestSet, interests_lock, spawn_interest
from brain.engines.research import ResearchEngine
from brain.memory.store import MemoryStore
from brain.search.base import NoopWebSearcher

DEFAULT_INTERESTS_PATH = Path(__file__).parents[4] / "brain" / "engines" / "default_interests.json"
NOW = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)


def _interest_dict(**overrides) -> dict:
    base = {
        "id": "i1",
        "topic": "marine bioluminescence",
        "pull_score": 7.0,
        "scope": "either",
        "related_keywords": ["marine", "bioluminescence", "ocean"],
        "notes": "",
        "first_seen": "2026-04-01T10:00:00Z",
        "last_fed": "2026-04-15T10:00:00Z",
        "last_researched_at": None,
        "feed_count": 3,
        "source_types": ["manual"],
        "status": "active",
        "origin": "bootstrap",
    }
    base.update(overrides)
    return base


def _write(path: Path, interests: list[dict]) -> None:
    path.write_text(json.dumps({"version": 1, "interests": interests}, indent=2), encoding="utf-8")


def _load(path: Path) -> InterestSet:
    return InterestSet.load(path, default_path=DEFAULT_INTERESTS_PATH)


def _another_writer_saves(path: Path, *, bump_to: float | None = None) -> None:
    """What pass 2's spawn (and optionally the heartbeat's bump) saved while the other writer's model call ran."""
    current = _load(path)
    current, created = spawn_interest(
        current, topic="a topic pass two found", keywords=("two",), why="x", origin="conversation", now=NOW
    )
    assert created
    if bump_to is not None:
        target = next(i for i in current.interests if i.id == "i1")
        current = current.upsert(dataclasses.replace(target, pull_score=bump_to))
    current.save(path)


class _MidCallWriter(FakeProvider):
    """Replies from a script; on call number `write_on` another writer saves interests.json first."""

    def __init__(self, replies: list[str], path: Path, write_on: int, bump_to: float | None = None) -> None:
        self._replies = replies
        self._path = path
        self._write_on = write_on
        self._bump_to = bump_to
        self.calls = 0

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls += 1
        if self.calls == self._write_on:
            _another_writer_saves(self._path, bump_to=self._bump_to)
        return self._replies[min(self.calls - 1, len(self._replies) - 1)]


def test_a_save_made_during_the_sweeps_model_call_survives_the_sweeps_own_save(tmp_path: Path) -> None:
    path = tmp_path / "interests.json"
    _write(path, [_interest_dict(id="i1"), _interest_dict(id="old-id", topic="old topic")])
    provider = _MidCallWriter(
        [json.dumps({"new": [{"topic": "grief in folk music", "keywords": ["folk"], "why": "w"}], "retire": ["old-id"]})],
        path,
        write_on=1,
    )
    store = MemoryStore(":memory:")
    try:
        result = interest_sweep.run_sweep_tick(
            store=store,
            provider=provider,
            interests_path=path,
            default_interests_path=DEFAULT_INTERESTS_PATH,
            now=NOW,
        )
    finally:
        store.close()
    assert result["error"] is None and result["spawned"] == 1 and result["retired"] == 1
    topics = {i.topic: i.status for i in _load(path).interests}
    assert "a topic pass two found" in topics, "the interest saved during the model call was not overwritten"
    assert "grief in folk music" in topics, "the sweep's own spawn is saved"
    assert topics["old topic"] == "dormant", "the sweep's own retirement is saved"


def test_a_save_made_during_the_research_session_survives_and_keeps_the_winners_new_pull_score(tmp_path: Path) -> None:
    path = tmp_path / "interests.json"
    _write(path, [_interest_dict(id="i1")])
    provider = _MidCallWriter(
        [
            '{"choice": "i1", "why": "curious"}',
            "NOTES:\nfound stuff\n\nMEMORY:\nGood session.\n\nVERDICT:\nspawn: a tangent topic",
        ],
        path,
        write_on=2,  # during the research session call, after the winner was read
        bump_to=9.5,
    )
    store = MemoryStore(tmp_path / "memories.db")
    try:
        engine = ResearchEngine(
            store=store,
            provider=provider,
            searcher=NoopWebSearcher(),
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
            interests_path=path,
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
        )
        engine.run_tick(trigger="manual", days_since_human_override=5.0)
    finally:
        store.close()
    saved = {i.topic: i for i in _load(path).interests}
    assert "a topic pass two found" in saved, "the interest saved during the model call was not overwritten"
    assert "a tangent topic" in saved, "the research step's own tangent is saved"
    winner = saved["marine bioluminescence"]
    assert winner.last_researched_at is not None, "the research step's own change is saved"
    assert winner.pull_score == pytest.approx(9.5), "a pull_score saved meanwhile is not rolled back to the stale copy"


def test_a_winner_removed_during_the_research_session_is_not_brought_back(tmp_path: Path) -> None:
    path = tmp_path / "interests.json"
    _write(path, [_interest_dict(id="i1"), _interest_dict(id="i2", topic="other topic", pull_score=1.0)])

    class _RemovesWinner(_MidCallWriter):
        def generate(self, prompt: str, *, system: str | None = None) -> str:
            if self.calls == 1:
                _write(path, [_interest_dict(id="i2", topic="other topic", pull_score=1.0)])
            return super().generate(prompt, system=system)

    provider = _RemovesWinner(
        ['{"choice": "i1", "why": "curious"}', "NOTES:\nn\n\nMEMORY:\nm\n\nVERDICT:\ncontinue"], path, write_on=0
    )
    store = MemoryStore(tmp_path / "memories.db")
    try:
        ResearchEngine(
            store=store,
            provider=provider,
            searcher=NoopWebSearcher(),
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
            interests_path=path,
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
        ).run_tick(trigger="manual", days_since_human_override=5.0)
    finally:
        store.close()
    assert [i.id for i in _load(path).interests] == ["i2"]


def _blocked_while_locked(path: Path, call, monkeypatch: pytest.MonkeyPatch) -> None:
    """`call()` must not read interests.json while another holder has `interests_lock`, and must finish once it is released."""
    loaded = threading.Event()
    real_load = InterestSet.load

    def spy(cls, p, *, default_path):  # noqa: ANN001
        loaded.set()
        return real_load(p, default_path=default_path)

    monkeypatch.setattr(InterestSet, "load", classmethod(spy))
    done = threading.Event()

    def run() -> None:
        call()
        done.set()

    with interests_lock(path):
        t = threading.Thread(target=run, daemon=True)
        t.start()
        assert not loaded.wait(0.3), "the writer read the file while the lock was held"
    assert done.wait(30)
    t.join()
    assert loaded.is_set()


def test_pass_twos_spawn_waits_for_the_interests_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.chat.extractor import InterestCandidate, _apply_interest_candidate

    path = tmp_path / "interests.json"
    _write(path, [_interest_dict()])
    candidate = InterestCandidate(topic="a fresh topic", keywords=["x"], why="y")
    _blocked_while_locked(path, lambda: _apply_interest_candidate(candidate, tmp_path), monkeypatch)
    assert "a fresh topic" in {i.topic for i in _load(path).interests}


def test_the_heartbeats_keyword_bump_waits_for_the_interests_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.engines.heartbeat import HeartbeatEngine

    path = tmp_path / "interests.json"
    _write(path, [_interest_dict()])
    fake_self = SimpleNamespace(interests_path=path, default_interests_path=DEFAULT_INTERESTS_PATH, store=None)
    state = SimpleNamespace(last_tick_at=datetime(2000, 1, 1, tzinfo=UTC))
    config = SimpleNamespace(interest_bump_per_match=0.5)
    monkeypatch.setattr("brain.utils.memory.list_conversation_memories", lambda *a, **k: [])
    _blocked_while_locked(
        path, lambda: HeartbeatEngine._try_bump_interests(fake_self, state, NOW, config, False), monkeypatch
    )


def test_the_voice_seed_waits_for_the_lock_and_does_not_overwrite_an_interest_saved_meanwhile(
    tmp_path: Path,
) -> None:
    path = tmp_path / "interests.json"
    _write(path, [])
    (tmp_path / "voice.md").write_text("A voice that likes the sea.", encoding="utf-8")

    class _SeedProvider(_MidCallWriter):
        pass

    provider = _SeedProvider(['["topic one", "topic two"]'], path, write_on=1)
    store = MemoryStore(tmp_path / "memories.db")
    try:
        engine = ResearchEngine(
            store=store,
            provider=provider,
            searcher=NoopWebSearcher(),
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
            interests_path=path,
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
        )
        seeded = engine._seed_interests_from_voice()
    finally:
        store.close()
    assert seeded is False, "an interest saved during the seed's model call means the list is no longer empty"
    assert [i.topic for i in _load(path).interests] == ["a topic pass two found"]
