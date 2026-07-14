"""Tests for brain.engines.interest_sweep — weekly sweep spawning/retiring interests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.bridge.provider import FakeProvider
from brain.engines._interests import InterestSet
from brain.memory.store import Memory, MemoryStore


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not find pyproject.toml above {here}")


DEFAULT_INTERESTS_PATH = _find_repo_root() / "brain" / "engines" / "default_interests.json"

NOW = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)


class ScriptedProvider(FakeProvider):
    """Returns replies from a fixed script, in call order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        idx = min(len(self.calls) - 1, len(self._replies) - 1)
        return self._replies[idx]


def _write_interests(path: Path, interests: list[dict]) -> None:
    path.write_text(json.dumps({"version": 1, "interests": interests}, indent=2), encoding="utf-8")


def _seed_conversation_memory(
    store: MemoryStore, content: str, emotions: dict[str, float] | None = None
) -> str:
    mem = Memory.create_new(
        content=content,
        memory_type="conversation",
        domain="us",
        emotions=emotions or {},
    )
    store.create(mem)
    return mem.id


def _seed_dream_memory(
    store: MemoryStore, content: str, emotions: dict[str, float] | None = None
) -> str:
    mem = Memory.create_new(
        content=content,
        memory_type="dream",
        domain="us",
        emotions=emotions or {},
    )
    store.create(mem)
    return mem.id


def _seed_monologue_memory(
    store: MemoryStore, content: str, emotions: dict[str, float] | None = None
) -> str:
    mem = Memory.create_new(
        content=content,
        memory_type="monologue_trace",
        domain="us",
        emotions=emotions or {},
    )
    store.create(mem)
    return mem.id


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


@pytest.fixture
def sweep_env(tmp_path: Path) -> dict:
    """Basic test environment with one existing interest and scripted provider."""
    store = MemoryStore(":memory:")
    interests_path = tmp_path / "interests.json"

    # Seed a couple of memories
    _seed_conversation_memory(store, "I've been thinking about grief in folk music lately")
    _seed_dream_memory(store, "I was walking through a concert hall made of water")

    # Seed one existing interest to keep + one to retire
    _write_interests(
        interests_path,
        [
            _interest_dict(
                id="i1",
                topic="marine bioluminescence",
                status="active",
                origin="bootstrap",
            ),
            _interest_dict(
                id="old-id",
                topic="old topic",
                status="active",
                origin="bootstrap",
            ),
        ],
    )

    # Provider returns valid JSON with 1 new + 1 retire
    provider = ScriptedProvider([
        json.dumps({
            "new": [{"topic": "grief in folk music", "keywords": ["folk", "grief"], "why": "recurs in inner life"}],
            "retire": ["old-id"],
        })
    ])

    return {
        "store": store,
        "provider": provider,
        "interests_path": interests_path,
        "default_interests_path": DEFAULT_INTERESTS_PATH,
    }


@pytest.fixture
def sweep_env_with_five_proposals(tmp_path: Path) -> dict:
    """Test environment where provider proposes 5 new + 5 retire (to test capping at 3)."""
    store = MemoryStore(":memory:")
    interests_path = tmp_path / "interests.json"

    # Seed memories
    for i in range(3):
        _seed_conversation_memory(store, f"Conversation topic {i}")
        _seed_dream_memory(store, f"Dream {i}")

    # Seed 5 existing interests to retire
    existing = [
        _interest_dict(id=f"old{i}", topic=f"old topic {i}", status="active", origin="bootstrap")
        for i in range(5)
    ]
    _write_interests(interests_path, existing)

    # Provider returns 5 new + 5 retire IDs
    provider = ScriptedProvider([
        json.dumps({
            "new": [
                {"topic": f"new topic {i}", "keywords": ["new"], "why": "proposed"}
                for i in range(5)
            ],
            "retire": [f"old{i}" for i in range(5)],
        })
    ])

    return {
        "store": store,
        "provider": provider,
        "interests_path": interests_path,
        "default_interests_path": DEFAULT_INTERESTS_PATH,
    }


@pytest.fixture
def sweep_env_garbage_provider(tmp_path: Path) -> dict:
    """Test environment with provider that returns unparseable garbage."""
    store = MemoryStore(":memory:")
    interests_path = tmp_path / "interests.json"

    _seed_conversation_memory(store, "Some conversation")
    _write_interests(interests_path, [])

    # Provider returns garbage that can't be parsed as JSON
    provider = ScriptedProvider(["this is not json at all!"])

    return {
        "store": store,
        "provider": provider,
        "interests_path": interests_path,
        "default_interests_path": DEFAULT_INTERESTS_PATH,
    }


@pytest.fixture
def sweep_env_unknown_retire(tmp_path: Path) -> dict:
    """Test environment where provider tries to retire an ID that doesn't exist."""
    store = MemoryStore(":memory:")
    interests_path = tmp_path / "interests.json"

    _seed_conversation_memory(store, "Some conversation")

    # Seed one existing interest
    _write_interests(
        interests_path,
        [_interest_dict(id="i1", topic="topic 1", status="active", origin="bootstrap")],
    )

    # Provider tries to retire a non-existent ID (should be silently ignored)
    provider = ScriptedProvider([
        json.dumps({
            "new": [],
            "retire": ["nonexistent-id"],
        })
    ])

    return {
        "store": store,
        "provider": provider,
        "interests_path": interests_path,
        "default_interests_path": DEFAULT_INTERESTS_PATH,
    }


def test_sweep_spawns_and_retires(sweep_env):
    """Sweep should spawn 1 new interest and retire 1 existing."""
    from brain.engines.interest_sweep import run_sweep_tick

    result = run_sweep_tick(**sweep_env, now=NOW)
    assert result == {"spawned": 1, "retired": 1, "error": None}

    s = InterestSet.load(sweep_env["interests_path"], default_path=sweep_env["default_interests_path"])
    assert any(i.origin == "sweep" and i.status == "active" for i in s.interests)
    assert any(i.status == "dormant" for i in s.interests)


def test_sweep_caps_at_three_each(sweep_env_with_five_proposals):
    """Sweep should cap spawns and retires at 3 each."""
    from brain.engines.interest_sweep import run_sweep_tick

    result = run_sweep_tick(**sweep_env_with_five_proposals, now=NOW)
    assert result["spawned"] <= 3
    assert result["retired"] <= 3


def test_sweep_never_raises_on_garbage(sweep_env_garbage_provider):
    """Sweep should never raise, even on garbage provider output."""
    from brain.engines.interest_sweep import run_sweep_tick

    result = run_sweep_tick(**sweep_env_garbage_provider, now=NOW)
    assert result["spawned"] == 0
    assert result["error"] is not None


def test_retire_unknown_id_ignored(sweep_env_unknown_retire):
    """Sweep should ignore retirement of non-existent IDs without error."""
    from brain.engines.interest_sweep import run_sweep_tick

    result = run_sweep_tick(**sweep_env_unknown_retire, now=NOW)
    assert result["retired"] == 0
    assert result["error"] is None
