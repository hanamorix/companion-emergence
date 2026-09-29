"""ram-spike-fix INC-11 — the heartbeat publishes "heartbeat" to
`background_jobs` for the duration of a real tick (spec §6, S15/S23/S38;
criterion C11(a)).

Kept in its own file (not test_heartbeat_decay_batching.py) so this
increment's build doesn't touch a file another concurrent builder in this
worktree may have in-flight changes to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.bridge import background_jobs, cli_throttle
from brain.bridge.provider import FakeProvider
from brain.engines.heartbeat import HeartbeatEngine
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


@pytest.fixture(autouse=True)
def _reset_cli_throttle():
    cli_throttle.reset()
    yield
    cli_throttle.reset()


@pytest.fixture(autouse=True)
def _reset_background_jobs():
    background_jobs._reset_for_tests()
    yield
    background_jobs._reset_for_tests()


@pytest.fixture(autouse=True)
def _safe_tier_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "brain.engines.heartbeat.build_tier_provider",
        lambda *a, **k: FakeProvider(),
    )


def _engine(tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix) -> HeartbeatEngine:
    return HeartbeatEngine(
        store=store,
        hebbian=hebbian,
        provider=FakeProvider(),
        state_path=tmp_path / "hb_state.json",
        config_path=tmp_path / "hb_config.json",
        dream_log_path=tmp_path / "dreams.log.jsonl",
        heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
        persona_name="Nell",
        persona_system_prompt="You are Nell.",
    )


def test_heartbeat_is_published_during_a_real_tick_and_cleared_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memories.db"
    store = MemoryStore(db)
    hebbian = HebbianMatrix(tmp_path / "hebbian.db")
    try:
        m = Memory.create_new(
            content="seed", memory_type="conversation", domain="us",
            emotions={"love": 9.0, "tenderness": 5.0},
        )
        store.create(m)
        engine = _engine(tmp_path, store, hebbian)

        # First-ever tick just initializes state and defers work (no decay
        # call) — not the case under test, run it out of the way first.
        assert background_jobs.snapshot() == []
        engine.run_tick(trigger="background")
        assert background_jobs.snapshot() == [], "first-ever (deferred) tick must not publish"

        seen_during: list[list[str]] = []
        real_apply = HeartbeatEngine._apply_emotion_decay

        def _spy(self, *args, **kwargs):
            seen_during.append(background_jobs.snapshot())
            return real_apply(self, *args, **kwargs)

        monkeypatch.setattr(HeartbeatEngine, "_apply_emotion_decay", _spy)

        result = engine.run_tick(trigger="background")

        assert result.initialized is False, "this must be the real (non-deferred) tick"
        assert seen_during == [["heartbeat"]], "must be published for the decay call itself"
        assert background_jobs.snapshot() == [], "must clear once run_tick returns"
    finally:
        store.close()
        hebbian.close()


def test_heartbeat_locked_out_by_another_pass_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C30's cross-process guard: a pass that loses the lock returns
    immediately (skipped_reason="heartbeat_locked") without ever reaching
    the decay call — must never appear in the registry."""
    from brain.utils.file_lock import file_lock

    db = tmp_path / "memories.db"
    store = MemoryStore(db)
    hebbian = HebbianMatrix(tmp_path / "hebbian.db")
    try:
        engine = _engine(tmp_path, store, hebbian)
        engine.run_tick(trigger="background")  # first-ever tick, initializes state

        with file_lock(engine.state_path, blocking=False):
            result = engine.run_tick(trigger="background")

        assert result.skipped_reason == "heartbeat_locked"
        assert background_jobs.snapshot() == []
    finally:
        store.close()
        hebbian.close()
