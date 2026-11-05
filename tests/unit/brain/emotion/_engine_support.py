"""Shared builders for the Engine 4 tests (drain, direct apply, catch-up, simulation, wiring).

Every time is built from a LOCAL civil date (local noon converted to UTC), because the engine's day is the machine's local
calendar day and nothing here sets the process time zone. Personas are tmp directories; a database is built with
`MemoryStore.create(fold=False)` so the engine is not told. Nothing here touches a real persona.
"""

from __future__ import annotations

import contextlib
import json
import logging
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from brain.emotion import engine, engine_files
from brain.emotion.engine import EmotionEngine
from brain.emotion.engine_utils import Average, EmotionState, EngineState, Event
from brain.memory.store import Memory, MemoryStore
from brain.utils import file_lock, file_ops

NOW = datetime(2026, 6, 10, 15, 30, tzinfo=UTC)
DAY0 = date(2026, 5, 1)


def local_noon(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time().replace(hour=12)).astimezone(UTC)


def local_day(offset: int) -> date:
    return DAY0 + timedelta(days=offset)


def row(
    created: datetime,
    *,
    accessed: datetime | None = None,
    emotions: dict | None = None,
    memory_type: str = "conversation",
    mid: str | None = None,
    metadata: dict | None = None,
) -> Memory:
    emo = {"joy": 6.0} if emotions is None else dict(emotions)
    numbers = [float(v) for v in emo.values()]
    return Memory(
        id=mid or str(uuid.uuid4()),
        content="an engine test memory",
        memory_type=memory_type,
        domain="us",
        created_at=created,
        emotions=emo,
        tags=[],
        importance=1.0,
        score=sum(numbers),
        metadata=dict(metadata or {}),
        peak_emotion_intensity=max(numbers, default=0.0),
        last_accessed_at=accessed,
    )


def make_persona(root: Path, rows: tuple[Memory, ...] | list[Memory] = ()) -> Path:
    """A persona directory with a `memories.db` holding `rows` (written without telling the engine)."""
    root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(root / "memories.db")
    try:
        for memory in rows:
            store.create(memory, fold=False)
    finally:
        store.close()
    return root


def install(persona_dir: Path, now: datetime = NOW) -> EmotionEngine:
    """Register an engine with a fixed clock as THE engine of `persona_dir` (what `engine_for` and the store hooks return)."""
    instance = EmotionEngine(persona_dir.resolve(), now=lambda: now)
    with engine._engines_lock:
        engine._engines[persona_dir.resolve()] = instance
    return instance


def simple_state(*, saved_at: datetime, joy: float = 4.0, day: str | None = None, queue_line: int = 0) -> EngineState:
    """A small state: one emotion (`joy`) with one entry of weight 1, as a saved copy would hold."""
    return EngineState(
        saved_at=saved_at,
        last_update=saved_at,
        active_day=day,
        oldest_entry_at=saved_at,
        persona_start_at=saved_at - timedelta(days=60),
        emotions={
            "joy": EmotionState(short=Average(joy, 1.0), long=Average(joy, 1.0), last_entry_day=day),
        },
        queue_line=queue_line,
    )


def save(persona_dir: Path, state: EngineState) -> None:
    engine_files.save_state(persona_dir, state)


def save_as_backup(persona_dir: Path, state: EngineState) -> None:
    """Leave `state` as the ONLY usable copy of the state file: it is `.bak1`, and the main file is garbage."""
    save(persona_dir, state)
    save(persona_dir, state)  # the second save rotates the first into .bak1
    engine_files.state_path(persona_dir).write_bytes(b"\x00 not json")


def load(persona_dir: Path) -> engine_files.LoadedState:
    return engine_files.read_state(persona_dir)


def state_of(persona_dir: Path) -> EngineState:
    loaded = engine_files.read_state(persona_dir)
    assert loaded.state is not None, loaded.problem
    return loaded.state


def queue_events(persona_dir: Path) -> list[Event]:
    return engine_files.read_queue(persona_dir, from_line=0).events


def queue_text_lines(persona_dir: Path) -> list[dict]:
    path = engine_files.queue_path(persona_dir)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def showing(memory_id: str, at: datetime, emotions: dict | None = None, modifier: float = 1.0) -> Event:
    return Event(kind="showing", at=at, memory_id=memory_id, emotions=emotions or {"joy": 5.0}, modifier=modifier)


def change(at: datetime, emotions: dict | None = None, source: str = "pass2") -> Event:
    return Event(kind="change", at=at, emotions=emotions or {"joy": 1.0}, source=source)


def commit_event(memory_id: str, at: datetime, emotions: dict | None = None, climax: float = 0.0) -> Event:
    return Event(kind="commit", at=at, memory_id=memory_id, emotions=emotions or {"joy": 6.0}, modifier=1.0, climax=climax)


def queue(persona_dir: Path, *events: Event) -> None:
    for event in events:
        assert engine_files.append_event(persona_dir, event)


def refuse_the_switch_stat(monkeypatch) -> None:  # noqa: ANN001 - a pytest MonkeyPatch
    """Make the stat of the switch file fail with a permission error (any error other than "no such file"): the switch is unknown."""
    real_stat = file_ops.stat_path

    def refuse(path: str | Path):
        if Path(path).name == "emotion_switch.json":
            raise PermissionError("denied")
        return real_stat(path)

    monkeypatch.setattr(file_ops, "stat_path", refuse)


def ruin_the_state(persona_dir: Path) -> None:
    """Leave every copy of the state file unreadable (main and the three backups)."""
    base = engine_files.state_path(persona_dir)
    for suffix in ("", ".bak1", ".bak2", ".bak3"):
        target = base.with_name(base.name + suffix)
        if target.exists():
            target.write_bytes(b"\x00")


SKIPPED = "this pass is skipped"  # the log line of a pass skipped because the switch cannot be inspected


def skipped_warnings(caplog) -> list[str]:  # noqa: ANN001 - a pytest LogCaptureFixture
    """WARNING lines saying the pass was skipped for an unknown switch."""
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and SKIPPED in r.getMessage()]


def engine_errors(caplog) -> list[str]:  # noqa: ANN001 - a pytest LogCaptureFixture
    """ERROR lines of the engine itself (the file reader logs its own error for an unreadable state, a different matter)."""
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR and r.name == "brain.emotion.engine"]


def tree(root: Path) -> dict[str, bytes | None]:
    """Every file under `root` with its bytes; the empty `.lock` sidecars a lock leaves behind are not state and are left out."""
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
        if p.suffix != ".lock"
    }


class LockSpy:
    """Calls through to `file_lock.locked_within` and `file_lock.file_lock` and records every call.

    `within` holds each `locked_within(path, wait)` call; `acquisitions` every `file_lock` entry (`locked_within` polls the
    non-blocking form, so one uncontended `locked_within` is one acquisition); `blocking` the calls of the blocking form,
    which engine code must never make.
    """

    def __init__(self, monkeypatch) -> None:  # noqa: ANN001 - a pytest MonkeyPatch
        self.within: list[tuple[Path, float]] = []
        self.acquisitions: list[Path] = []
        self.blocking: list[Path] = []
        real_within = file_lock.locked_within
        real_blocking = file_lock.file_lock
        spy = self

        @contextlib.contextmanager
        def within(path, wait, **kwargs):  # noqa: ANN001
            spy.within.append((Path(path), wait))
            with real_within(path, wait, **kwargs):
                yield

        @contextlib.contextmanager
        def blocking(path, *args, **kwargs):  # noqa: ANN001
            spy.acquisitions.append(Path(path))
            if kwargs.get("blocking", True):
                spy.blocking.append(Path(path))
            with real_blocking(path, *args, **kwargs) as acquired:
                yield acquired

        monkeypatch.setattr(file_lock, "locked_within", within)
        monkeypatch.setattr(file_lock, "file_lock", blocking)

    @property
    def calls(self) -> int:
        """Lock acquisitions of any kind."""
        return len(self.acquisitions) + len(self.within)


def joy_of(state: EngineState) -> EmotionState:
    return state.emotions["joy"]


