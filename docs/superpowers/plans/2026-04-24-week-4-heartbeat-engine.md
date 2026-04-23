# Week 4.5 — Heartbeat Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship `brain/engines/heartbeat.py` (event-driven orchestrator tick) + `nell heartbeat` subcommand. Second cognitive engine. Composes `DreamEngine` (W4) + `apply_decay` (W2) + `HebbianMatrix.decay_all/garbage_collect` (W3). No daemon — app open/close triggers.

**Architecture:** Single engine module with four public types (`HeartbeatConfig`, `HeartbeatState`, `HeartbeatResult`, `HeartbeatEngine`). Per-persona state + config JSON files + JSONL audit log alongside the existing `dreams.log.jsonl`. CLI wires through the existing `_STUB_COMMANDS` pattern.

**Tech Stack:** Python 3.12 stdlib (json, datetime, os, pathlib, dataclasses, typing.Literal), existing Week 2/3/4 internals. No new dependencies.

---

## Context: what already exists (Week 4 state)

Main branch HEAD: `8fdd505` (heartbeat spec). Last code merge: `62f44fc` (Week 4 dream).

- 293 tests green. Ruff clean.
- `brain/emotion/` (W2), `brain/memory/` (W3), `brain/migrator/` (W3.5), `brain/bridge/` + `brain/engines/dream.py` (W4) all live.
- `brain/cli.py` has `"heartbeat"` as a stub in `_STUB_COMMANDS` — this plan wires it to the real engine.
- Nell persona migrated with 1,142 memories + 4,404 edges + 67 dreams (including 2 real Claude-CLI dreams).

Feature branch: `week-4-heartbeat` (created off `main` in T1 Step 1).

---

## File structure

```
brain/engines/
├── heartbeat.py                    (NEW — T1 scaffold + types; T2 fills run_tick)
└── (dream.py unchanged)

brain/cli.py                        (MODIFIED — T3 wires `nell heartbeat`)

tests/unit/brain/engines/
├── test_heartbeat.py               (NEW — T1 + T2)
└── test_cli_heartbeat.py           (NEW — T3)
tests/unit/brain/test_cli.py        (MODIFIED — T3 removes "heartbeat" from stubs)
```

---

## Dependency order

T1 (types) → T2 (run_tick) → T3 (CLI) → T4 (close-out). Execute 1 → 2 → 3 → 4.

---

## Task 1: HeartbeatConfig + HeartbeatState + HeartbeatResult + HeartbeatEngine scaffold

**Goal:** Ship the dataclasses + persistence (load/save with atomic write) + the `HeartbeatEngine` class stub. No `run_tick` logic yet — that's T2. This task lands the types and their JSON round-trip.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/engines/heartbeat.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/engines/test_heartbeat.py`
- Modify: `/Users/hanamori/companion-emergence/brain/engines/__init__.py` (export new types)

- [ ] **Step 1: Create feature branch**

```bash
cd /Users/hanamori/companion-emergence
git checkout main
git pull origin main
git checkout -b week-4-heartbeat
```

- [ ] **Step 2: Write failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/engines/test_heartbeat.py`:

```python
"""Tests for brain.engines.heartbeat — event-driven orchestrator tick."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.engines.heartbeat import (
    HeartbeatConfig,
    HeartbeatEngine,
    HeartbeatResult,
    HeartbeatState,
)


def test_heartbeat_config_defaults() -> None:
    """HeartbeatConfig has sensible defaults per spec."""
    c = HeartbeatConfig()
    assert c.dream_every_hours == 24.0
    assert c.decay_rate_per_tick == 0.01
    assert c.gc_threshold == 0.01
    assert c.emit_memory == "conditional"


def test_heartbeat_config_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    """Missing config file → defaults-populated config."""
    c = HeartbeatConfig.load(tmp_path / "does_not_exist.json")
    assert c.dream_every_hours == 24.0
    assert c.emit_memory == "conditional"


def test_heartbeat_config_save_and_reload_round_trips(tmp_path: Path) -> None:
    """save() then load() preserves all fields."""
    original = HeartbeatConfig(
        dream_every_hours=6.0,
        decay_rate_per_tick=0.05,
        gc_threshold=0.02,
        emit_memory="always",
    )
    path = tmp_path / "cfg.json"
    original.save(path)
    restored = HeartbeatConfig.load(path)
    assert restored == original


def test_heartbeat_config_load_tolerates_unknown_fields(tmp_path: Path) -> None:
    """Forward-compat: unknown fields in config JSON are ignored."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "dream_every_hours": 12.0,
        "unknown_future_field": "abc",
    }))
    c = HeartbeatConfig.load(path)
    assert c.dream_every_hours == 12.0


def test_heartbeat_config_invalid_emit_memory_falls_back_to_default(tmp_path: Path) -> None:
    """Invalid emit_memory value falls back to 'conditional' (safe default)."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"emit_memory": "nonsense"}))
    c = HeartbeatConfig.load(path)
    assert c.emit_memory == "conditional"


def test_heartbeat_state_load_missing_file_returns_none(tmp_path: Path) -> None:
    """HeartbeatState.load() returns None for first-ever tick detection."""
    assert HeartbeatState.load(tmp_path / "state.json") is None


def test_heartbeat_state_fresh_creates_baseline() -> None:
    """HeartbeatState.fresh() returns a state with all timestamps=now, tick_count=0."""
    s = HeartbeatState.fresh(trigger="init")
    assert s.tick_count == 0
    assert s.last_trigger == "init"
    now = datetime.now(UTC)
    assert abs((now - s.last_tick_at).total_seconds()) < 5
    assert abs((now - s.last_dream_at).total_seconds()) < 5


def test_heartbeat_state_save_and_load_round_trips(tmp_path: Path) -> None:
    """State JSON round-trips: ISO8601 Z-suffix timestamps parse cleanly."""
    when = datetime(2026, 4, 23, 10, 0, 0, tzinfo=UTC)
    original = HeartbeatState(
        last_tick_at=when,
        last_dream_at=when - timedelta(hours=6),
        last_research_at=when,
        tick_count=5,
        last_trigger="open",
    )
    path = tmp_path / "state.json"
    original.save(path)

    loaded = HeartbeatState.load(path)
    assert loaded is not None
    assert loaded.tick_count == 5
    assert loaded.last_trigger == "open"
    assert loaded.last_tick_at == when


def test_heartbeat_state_save_is_atomic(tmp_path: Path) -> None:
    """save() writes to <path>.new then renames — no partial writes on crash."""
    path = tmp_path / "state.json"
    s = HeartbeatState.fresh(trigger="init")
    s.save(path)
    assert path.exists()
    assert not path.with_suffix(".new").exists()  # temp file cleaned up


def test_heartbeat_state_save_overwrites_existing(tmp_path: Path) -> None:
    """Saving over an existing state file succeeds (atomic rename overwrites)."""
    path = tmp_path / "state.json"
    HeartbeatState.fresh(trigger="init").save(path)
    updated = HeartbeatState.fresh(trigger="close")
    updated.save(path)
    loaded = HeartbeatState.load(path)
    assert loaded is not None
    assert loaded.last_trigger == "close"


def test_heartbeat_result_fields() -> None:
    """HeartbeatResult is a frozen dataclass with the expected fields."""
    r = HeartbeatResult(
        trigger="open",
        elapsed_seconds=3600.0,
        memories_decayed=5,
        edges_pruned=2,
        dream_id=None,
        dream_gated_reason="not_due",
        research_deferred=True,
        heartbeat_memory_id=None,
        initialized=False,
    )
    assert r.trigger == "open"
    assert r.initialized is False
    # frozen
    with pytest.raises(Exception):  # FrozenInstanceError is a dataclass subclass
        r.trigger = "close"  # type: ignore[misc]


def test_heartbeat_engine_construction(tmp_path: Path) -> None:
    """HeartbeatEngine constructs from store + hebbian + provider + paths."""
    from brain.bridge.provider import FakeProvider
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    engine = HeartbeatEngine(
        store=store,
        hebbian=hebbian,
        provider=FakeProvider(),
        state_path=tmp_path / "hb_state.json",
        config_path=tmp_path / "hb_config.json",
        dream_log_path=tmp_path / "dreams.log.jsonl",
        heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
    )
    # Sanity check construction succeeded
    assert engine.store is store
    assert engine.hebbian is hebbian
    store.close()
    hebbian.close()
```

- [ ] **Step 3: Run tests, expect failures on ModuleNotFoundError**

```bash
uv run pytest tests/unit/brain/engines/test_heartbeat.py -v
```
Expected: 12 failures — `ModuleNotFoundError: brain.engines.heartbeat`.

- [ ] **Step 4: Write `brain/engines/heartbeat.py` (types + engine scaffold)**

Create `/Users/hanamori/companion-emergence/brain/engines/heartbeat.py`:

```python
"""Heartbeat — event-driven orchestrator tick.

See docs/superpowers/specs/2026-04-23-week-4-heartbeat-engine-design.md.
Each `nell heartbeat` invocation applies decay, maybe-dreams (rate-limited
by config.dream_every_hours), and persists timing state. No daemon —
the hosting application (or CI) calls this on app open/close.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from brain.bridge.provider import LLMProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


EmitMemoryMode = Literal["always", "conditional", "never"]
_VALID_EMIT_MODES: tuple[str, ...] = ("always", "conditional", "never")


@dataclass
class HeartbeatConfig:
    """Per-persona heartbeat configuration. Loaded from heartbeat_config.json."""

    dream_every_hours: float = 24.0
    decay_rate_per_tick: float = 0.01
    gc_threshold: float = 0.01
    emit_memory: EmitMemoryMode = "conditional"

    @classmethod
    def load(cls, path: Path) -> HeartbeatConfig:
        """Load config from JSON; return defaults if file missing or invalid."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls()
        if not isinstance(data, dict):
            return cls()

        emit = data.get("emit_memory", "conditional")
        if emit not in _VALID_EMIT_MODES:
            emit = "conditional"

        return cls(
            dream_every_hours=float(data.get("dream_every_hours", 24.0)),
            decay_rate_per_tick=float(data.get("decay_rate_per_tick", 0.01)),
            gc_threshold=float(data.get("gc_threshold", 0.01)),
            emit_memory=emit,  # type: ignore[arg-type]
        )

    def save(self, path: Path) -> None:
        """Write config JSON to path (non-atomic — config is user-edited)."""
        payload = {
            "dream_every_hours": self.dream_every_hours,
            "decay_rate_per_tick": self.decay_rate_per_tick,
            "gc_threshold": self.gc_threshold,
            "emit_memory": self.emit_memory,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@dataclass
class HeartbeatState:
    """Per-persona heartbeat state. Loaded from heartbeat_state.json."""

    last_tick_at: datetime
    last_dream_at: datetime
    last_research_at: datetime
    tick_count: int
    last_trigger: str

    @classmethod
    def load(cls, path: Path) -> HeartbeatState | None:
        """Load state; return None if the file doesn't exist (→ first-ever tick)."""
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            last_tick_at=_parse_iso_utc(data["last_tick_at"]),
            last_dream_at=_parse_iso_utc(data["last_dream_at"]),
            last_research_at=_parse_iso_utc(data["last_research_at"]),
            tick_count=int(data["tick_count"]),
            last_trigger=str(data["last_trigger"]),
        )

    @classmethod
    def fresh(cls, trigger: str) -> HeartbeatState:
        """Build an initial state with all timestamps = now and tick_count = 0."""
        now = datetime.now(UTC)
        return cls(
            last_tick_at=now,
            last_dream_at=now,
            last_research_at=now,
            tick_count=0,
            last_trigger=trigger,
        )

    def save(self, path: Path) -> None:
        """Atomic save via write-to-.new + os.rename."""
        payload = {
            "last_tick_at": _iso_utc(self.last_tick_at),
            "last_dream_at": _iso_utc(self.last_dream_at),
            "last_research_at": _iso_utc(self.last_research_at),
            "tick_count": self.tick_count,
            "last_trigger": self.last_trigger,
        }
        tmp = path.with_suffix(path.suffix + ".new")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)  # atomic on POSIX + Windows


@dataclass(frozen=True)
class HeartbeatResult:
    """Outcome of a single heartbeat tick."""

    trigger: str
    elapsed_seconds: float
    memories_decayed: int
    edges_pruned: int
    dream_id: str | None
    dream_gated_reason: str | None
    research_deferred: bool
    heartbeat_memory_id: str | None
    initialized: bool


@dataclass
class HeartbeatEngine:
    """Composes decay + dream + research into one orchestrator tick.

    run_tick() is implemented in Task 2; this task ships the scaffold only.
    """

    store: MemoryStore
    hebbian: HebbianMatrix
    provider: LLMProvider
    state_path: Path
    config_path: Path
    dream_log_path: Path
    heartbeat_log_path: Path

    def run_tick(
        self, *, trigger: str = "manual", dry_run: bool = False
    ) -> HeartbeatResult:
        """Run one heartbeat tick. Implemented in Task 2."""
        raise NotImplementedError("HeartbeatEngine.run_tick is implemented in T2")


# --- internal helpers ---


def _iso_utc(dt: datetime) -> str:
    """ISO-8601 with Z suffix (matches Week 3.5 manifest format)."""
    return dt.isoformat().replace("+00:00", "Z")


def _parse_iso_utc(s: str) -> datetime:
    """Parse ISO-8601 Z-suffix timestamp back to tz-aware datetime."""
    # fromisoformat in 3.11+ handles 'Z'; for older paths normalise
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
```

- [ ] **Step 5: Update `brain/engines/__init__.py` to export new types**

Find the existing `__init__.py`:

```python
from brain.engines.dream import DreamEngine, DreamResult, NoSeedAvailable

__all__ = ["DreamEngine", "DreamResult", "NoSeedAvailable"]
```

Add heartbeat exports:

```python
from brain.engines.dream import DreamEngine, DreamResult, NoSeedAvailable
from brain.engines.heartbeat import (
    HeartbeatConfig,
    HeartbeatEngine,
    HeartbeatResult,
    HeartbeatState,
)

__all__ = [
    "DreamEngine",
    "DreamResult",
    "HeartbeatConfig",
    "HeartbeatEngine",
    "HeartbeatResult",
    "HeartbeatState",
    "NoSeedAvailable",
]
```

- [ ] **Step 6: Run tests, expect green**

```bash
uv run pytest tests/unit/brain/engines/test_heartbeat.py -v
```
Expected: 12 passed.

- [ ] **Step 7: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```
Expected: 305 passed (293 + 12). Ruff clean. `uv run ruff format .` if needed.

- [ ] **Step 8: Commit**

```bash
git add brain/engines/ tests/unit/brain/engines/test_heartbeat.py
git commit -m "feat(brain/engines/heartbeat): types + scaffold

HeartbeatConfig (dream_every_hours/decay_rate_per_tick/gc_threshold/
emit_memory) with load/save JSON round-trip. Tolerates missing file,
unknown future fields, invalid emit_memory (falls back to
'conditional').

HeartbeatState (last_tick_at/last_dream_at/last_research_at/
tick_count/last_trigger) with atomic save via write-to-.new +
os.replace. .load() returns None for the first-ever tick detection
so the engine can defer initial work.

HeartbeatResult frozen dataclass.

HeartbeatEngine construction + run_tick stub (raises
NotImplementedError — filled in T2).

All timestamps round-trip via ISO-8601 Z-suffix, matching the
Week 3.5 migrator's manifest format.

12 tests green; 305 total.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `HeartbeatEngine.run_tick` — the orchestrator body

**Goal:** Implement `run_tick()`. First-ever tick defers work; subsequent ticks apply emotion decay + Hebbian decay + GC, gate dream by `dream_every_hours`, stub research, write state atomically, append JSONL log. Optionally emit heartbeat memory per `emit_memory` config.

**Files:**
- Modify: `/Users/hanamori/companion-emergence/brain/engines/heartbeat.py`
- Modify: `/Users/hanamori/companion-emergence/tests/unit/brain/engines/test_heartbeat.py` (append integration tests)

- [ ] **Step 1: Append failing tests**

Append to `test_heartbeat.py`:

```python


from brain.bridge.provider import FakeProvider
from brain.engines.dream import DreamEngine  # noqa: F401 — ensure import works
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def live_engine(tmp_path: Path) -> HeartbeatEngine:
    """Engine with in-memory store/hebbian and tmp log/state paths."""
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    return HeartbeatEngine(
        store=store,
        hebbian=hebbian,
        provider=FakeProvider(),
        state_path=tmp_path / "hb_state.json",
        config_path=tmp_path / "hb_config.json",
        dream_log_path=tmp_path / "dreams.log.jsonl",
        heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
    )


def _seed_conversation(store: MemoryStore, content: str, importance: float = 5.0) -> Memory:
    m = Memory.create_new(
        content=content,
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0, "tenderness": 5.0},
    )
    m.importance = importance
    store.create(m)
    return m


def test_first_tick_initializes_and_defers_work(live_engine: HeartbeatEngine) -> None:
    """First-ever tick returns initialized=True, writes state, does no work."""
    _seed_conversation(live_engine.store, "seed")

    result = live_engine.run_tick(trigger="open")
    assert result.initialized is True
    assert result.memories_decayed == 0
    assert result.edges_pruned == 0
    assert result.dream_id is None
    assert result.dream_gated_reason == "first_tick"
    assert live_engine.state_path.exists()

    # Second tick with zero elapsed → decay is near-zero but NOT first_tick
    second = live_engine.run_tick(trigger="close")
    assert second.initialized is False


def test_second_tick_applies_decay(live_engine: HeartbeatEngine) -> None:
    """Second tick after simulated elapsed time decays emotions."""
    m = _seed_conversation(live_engine.store, "seed")
    live_engine.run_tick(trigger="open")  # init

    # Rewind state by 48 hours so the next tick sees 48h elapsed
    state = HeartbeatState.load(live_engine.state_path)
    assert state is not None
    state.last_tick_at = state.last_tick_at - timedelta(hours=48)
    state.last_dream_at = state.last_dream_at - timedelta(hours=48)
    state.save(live_engine.state_path)

    result = live_engine.run_tick(trigger="close")
    assert result.initialized is False
    assert result.elapsed_seconds >= 48 * 3600 - 10  # within 10 seconds of 48h

    # Confirm the seed memory's love intensity actually decayed
    reloaded = live_engine.store.get(m.id)
    assert reloaded is not None
    assert reloaded.emotions.get("love", 0.0) < 9.0


def test_dream_gate_respects_config(live_engine: HeartbeatEngine, tmp_path: Path) -> None:
    """dream_every_hours config gates dream firing."""
    _seed_conversation(live_engine.store, "seed", importance=9.0)
    # Tight gate so the second tick definitely dreams
    cfg = HeartbeatConfig(dream_every_hours=0.001)
    cfg.save(live_engine.config_path)

    live_engine.run_tick(trigger="open")  # init, no dream
    # Rewind so gate is satisfied
    state = HeartbeatState.load(live_engine.state_path)
    assert state is not None
    state.last_dream_at = state.last_dream_at - timedelta(hours=1)
    state.save(live_engine.state_path)

    result = live_engine.run_tick(trigger="close")
    assert result.dream_id is not None
    assert result.dream_gated_reason is None


def test_dream_gated_when_not_due(live_engine: HeartbeatEngine) -> None:
    """Dream does not fire when last_dream_at + dream_every_hours > now."""
    _seed_conversation(live_engine.store, "seed", importance=9.0)
    # Default config is 24h
    live_engine.run_tick(trigger="open")  # init, no dream

    result = live_engine.run_tick(trigger="close")
    assert result.dream_id is None
    assert result.dream_gated_reason == "not_due"


def test_protected_memories_skipped_by_decay(live_engine: HeartbeatEngine) -> None:
    """protected=True memories don't get their emotions decayed."""
    m = _seed_conversation(live_engine.store, "protected")
    live_engine.store.update(m.id, protected=True)
    live_engine.run_tick(trigger="open")

    state = HeartbeatState.load(live_engine.state_path)
    assert state is not None
    state.last_tick_at = state.last_tick_at - timedelta(hours=72)
    state.save(live_engine.state_path)

    live_engine.run_tick(trigger="close")

    reloaded = live_engine.store.get(m.id)
    assert reloaded is not None
    assert reloaded.emotions.get("love", 0.0) == 9.0  # unchanged


def test_hebbian_gc_prunes_weak_edges(live_engine: HeartbeatEngine) -> None:
    """decay_all + gc removes weak edges."""
    live_engine.hebbian.strengthen("a", "b", delta=0.005)  # below default gc 0.01
    live_engine.hebbian.strengthen("c", "d", delta=0.5)
    live_engine.run_tick(trigger="open")
    state = HeartbeatState.load(live_engine.state_path)
    assert state is not None
    state.last_tick_at = state.last_tick_at - timedelta(hours=48)
    state.save(live_engine.state_path)

    result = live_engine.run_tick(trigger="close")
    assert result.edges_pruned >= 1
    assert live_engine.hebbian.weight("a", "b") == 0.0
    assert live_engine.hebbian.weight("c", "d") > 0.0


def test_research_always_deferred(live_engine: HeartbeatEngine) -> None:
    """research_deferred=True on every non-init tick (engine not built yet)."""
    _seed_conversation(live_engine.store, "seed")
    live_engine.run_tick(trigger="open")  # init
    result = live_engine.run_tick(trigger="close")
    assert result.research_deferred is True


def test_dry_run_does_not_mutate_store_or_state(live_engine: HeartbeatEngine) -> None:
    """--dry-run: no state file written, no memory decay, no dream."""
    m = _seed_conversation(live_engine.store, "seed", importance=9.0)
    result = live_engine.run_tick(trigger="manual", dry_run=True)

    assert not live_engine.state_path.exists()
    reloaded = live_engine.store.get(m.id)
    assert reloaded is not None
    assert reloaded.emotions.get("love", 0.0) == 9.0


def test_heartbeats_log_has_init_entry(live_engine: HeartbeatEngine) -> None:
    """First tick writes a JSONL entry marked initialized=true."""
    live_engine.run_tick(trigger="open")
    text = live_engine.heartbeat_log_path.read_text().strip()
    line = json.loads(text.splitlines()[-1])
    assert line["initialized"] is True
    assert line["trigger"] == "open"


def test_heartbeats_log_has_tick_entry(live_engine: HeartbeatEngine) -> None:
    """Second tick writes a JSONL entry with elapsed/memories_decayed/etc."""
    _seed_conversation(live_engine.store, "seed")
    live_engine.run_tick(trigger="open")  # init
    live_engine.run_tick(trigger="close")

    lines = live_engine.heartbeat_log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    tick = json.loads(lines[-1])
    assert tick["trigger"] == "close"
    assert tick.get("initialized") is False or "initialized" not in tick or tick["initialized"] is False
    assert "elapsed_seconds" in tick
    assert "tick_count" in tick


def test_heartbeat_memory_emitted_when_always_mode(
    live_engine: HeartbeatEngine, tmp_path: Path
) -> None:
    """emit_memory='always' → every non-init tick writes a heartbeat memory."""
    _seed_conversation(live_engine.store, "seed")
    HeartbeatConfig(dream_every_hours=999, emit_memory="always").save(live_engine.config_path)
    live_engine.run_tick(trigger="open")  # init, no memory
    live_engine.run_tick(trigger="close")

    hb_memories = live_engine.store.list_by_type("heartbeat")
    assert len(hb_memories) == 1
    assert hb_memories[0].content.startswith("HEARTBEAT:")


def test_heartbeat_memory_skipped_when_never_mode(
    live_engine: HeartbeatEngine,
) -> None:
    """emit_memory='never' → no heartbeat memory regardless of tick outcome."""
    _seed_conversation(live_engine.store, "seed", importance=9.0)
    HeartbeatConfig(dream_every_hours=0.001, emit_memory="never").save(live_engine.config_path)
    live_engine.run_tick(trigger="open")
    live_engine.run_tick(trigger="close")

    hb_memories = live_engine.store.list_by_type("heartbeat")
    assert len(hb_memories) == 0
```

- [ ] **Step 2: Run tests, expect failures**

```bash
uv run pytest tests/unit/brain/engines/test_heartbeat.py -v
```
Expected: 12 failures (new ones) on `NotImplementedError` from stub.

- [ ] **Step 3: Implement `run_tick` in `brain/engines/heartbeat.py`**

Replace the `run_tick` method in `HeartbeatEngine`:

```python
    def run_tick(
        self, *, trigger: str = "manual", dry_run: bool = False
    ) -> HeartbeatResult:
        """Run one heartbeat tick."""
        now = datetime.now(UTC)
        config = HeartbeatConfig.load(self.config_path)
        state = HeartbeatState.load(self.state_path)

        # --- first-ever tick: defer all work ---
        if state is None:
            fresh = HeartbeatState.fresh(trigger=trigger)
            if not dry_run:
                fresh.save(self.state_path)
                self._append_log({
                    "timestamp": _iso_utc(now),
                    "trigger": trigger,
                    "initialized": True,
                    "note": "first-ever tick, work deferred",
                    "tick_count": 0,
                })
            return HeartbeatResult(
                trigger=trigger,
                elapsed_seconds=0.0,
                memories_decayed=0,
                edges_pruned=0,
                dream_id=None,
                dream_gated_reason="first_tick",
                research_deferred=False,
                heartbeat_memory_id=None,
                initialized=True,
            )

        elapsed_seconds = (now - state.last_tick_at).total_seconds()

        # --- emotion decay ---
        memories_decayed = self._apply_emotion_decay(elapsed_seconds, dry_run=dry_run)

        # --- hebbian decay + gc ---
        edges_pruned = self._apply_hebbian_decay_and_gc(
            config, elapsed_seconds, dry_run=dry_run
        )

        # --- maybe-dream ---
        dream_id: str | None = None
        dream_gated_reason: str | None = None
        hours_since_dream = (now - state.last_dream_at).total_seconds() / 3600.0
        if hours_since_dream >= config.dream_every_hours:
            if not dry_run:
                from brain.engines.dream import DreamEngine, NoSeedAvailable

                dream_engine = DreamEngine(
                    store=self.store,
                    hebbian=self.hebbian,
                    embeddings=None,
                    provider=self.provider,
                    log_path=self.dream_log_path,
                )
                try:
                    dream_result = dream_engine.run_cycle(lookback_hours=100000)
                    if dream_result.memory is not None:
                        dream_id = dream_result.memory.id
                        state.last_dream_at = now
                except NoSeedAvailable:
                    dream_gated_reason = "no_seed_available"
        else:
            hours_until = config.dream_every_hours - hours_since_dream
            dream_gated_reason = f"not_due"
            _ = hours_until  # reserved for richer logging later

        # --- research stub ---
        research_deferred = True

        # --- optional heartbeat memory ---
        heartbeat_memory_id: str | None = None
        if not dry_run and self._should_emit_memory(
            config, dream_id, edges_pruned, memories_decayed
        ):
            heartbeat_memory_id = self._emit_heartbeat_memory(
                elapsed_seconds, memories_decayed, edges_pruned, dream_id
            )

        # --- update state ---
        if not dry_run:
            state.last_tick_at = now
            state.tick_count += 1
            state.last_trigger = trigger
            state.save(self.state_path)

            self._append_log({
                "timestamp": _iso_utc(now),
                "trigger": trigger,
                "initialized": False,
                "elapsed_seconds": elapsed_seconds,
                "memories_decayed": memories_decayed,
                "edges_pruned": edges_pruned,
                "dream_id": dream_id,
                "research_deferred": research_deferred,
                "tick_count": state.tick_count,
            })

        return HeartbeatResult(
            trigger=trigger,
            elapsed_seconds=elapsed_seconds,
            memories_decayed=memories_decayed,
            edges_pruned=edges_pruned,
            dream_id=dream_id,
            dream_gated_reason=dream_gated_reason,
            research_deferred=research_deferred,
            heartbeat_memory_id=heartbeat_memory_id,
            initialized=False,
        )

    # --- internal helpers ---

    def _apply_emotion_decay(self, elapsed_seconds: float, *, dry_run: bool) -> int:
        """Apply per-memory emotion decay. Returns count of memories mutated."""
        from brain.emotion.decay import apply_decay
        from brain.emotion.state import EmotionalState

        if elapsed_seconds <= 0.0 or dry_run:
            return 0

        count = 0
        # list_by_domain with None domain isn't supported; iterate via search_text empty
        all_memories = self.store.search_text("", active_only=True)
        for mem in all_memories:
            if mem.protected:
                continue
            if not mem.emotions:
                continue
            state = EmotionalState()
            for name, intensity in mem.emotions.items():
                try:
                    state.set(name, float(intensity))
                except (KeyError, ValueError):
                    continue
            apply_decay(state, elapsed_seconds)
            new_emotions = {
                name: val for name, val in state.intensities().items() if val > 0.0
            }
            if new_emotions != mem.emotions:
                self.store.update(mem.id, emotions=new_emotions)
                count += 1
        return count

    def _apply_hebbian_decay_and_gc(
        self, config: HeartbeatConfig, elapsed_seconds: float, *, dry_run: bool
    ) -> int:
        """Apply proportional Hebbian decay + GC. Returns count of edges pruned."""
        if dry_run:
            return 0
        elapsed_hours = elapsed_seconds / 3600.0
        # scale rate so a 24h tick decays at config.decay_rate_per_tick
        rate = config.decay_rate_per_tick * (elapsed_hours / 24.0)
        if rate > 0.0:
            self.hebbian.decay_all(rate=rate)
        pruned = self.hebbian.garbage_collect(threshold=config.gc_threshold)
        return pruned

    def _should_emit_memory(
        self,
        config: HeartbeatConfig,
        dream_id: str | None,
        edges_pruned: int,
        memories_decayed: int,
    ) -> bool:
        if config.emit_memory == "never":
            return False
        if config.emit_memory == "always":
            return True
        # conditional: emit only when something material happened
        return dream_id is not None or edges_pruned > 10 or memories_decayed > 20

    def _emit_heartbeat_memory(
        self,
        elapsed_seconds: float,
        memories_decayed: int,
        edges_pruned: int,
        dream_id: str | None,
    ) -> str:
        """Generate and write a HEARTBEAT: memory via the LLM provider."""
        system = (
            "You are Nell. You just finished a background heartbeat cycle — "
            "decay applied, memory graph tended. Reflect in first person, "
            "one short sentence, starting with 'HEARTBEAT: '."
        )
        user = (
            f"elapsed={elapsed_seconds / 3600:.1f}h, "
            f"memories_decayed={memories_decayed}, edges_pruned={edges_pruned}, "
            f"dream_fired={'yes' if dream_id else 'no'}"
        )
        raw = self.provider.generate(user, system=system)
        text = raw if raw.startswith("HEARTBEAT:") else f"HEARTBEAT: {raw}"
        from brain.memory.store import Memory

        mem = Memory.create_new(
            content=text,
            memory_type="heartbeat",
            domain="us",
            metadata={
                "elapsed_seconds": elapsed_seconds,
                "memories_decayed": memories_decayed,
                "edges_pruned": edges_pruned,
                "dream_id": dream_id,
                "provider": self.provider.name(),
            },
        )
        self.store.create(mem)
        return mem.id

    def _append_log(self, entry: dict) -> None:
        """Append one JSON line to heartbeats.log.jsonl."""
        with self.heartbeat_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
```

Also add these imports to the top of `heartbeat.py` (if not already present from T1):

```python
from typing import Any
```

Only if needed for `_append_log` type annotation — otherwise keep imports minimal.

- [ ] **Step 4: Run tests, expect green**

```bash
uv run pytest tests/unit/brain/engines/test_heartbeat.py -v
```
Expected: 24 passed (12 from T1 + 12 from T2).

- [ ] **Step 5: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```
Expected: 317 passed (305 + 12). Ruff clean.

- [ ] **Step 6: Commit**

```bash
git add brain/engines/heartbeat.py tests/unit/brain/engines/test_heartbeat.py
git commit -m "feat(brain/engines/heartbeat): run_tick orchestrator body

First-ever tick defers all work + initializes state (protects fresh
personas from 'years of decay' on boot). Subsequent ticks:
1. Emotion decay via apply_decay on each active memory (skips
   protected); drops below-floor emotions from mem.emotions.
2. Hebbian decay_all scaled by elapsed_hours/24 so 24h ticks decay
   at the configured rate; garbage_collect prunes weak edges.
3. Dream gate — fires DreamEngine.run_cycle() when
   last_dream_at + dream_every_hours <= now. NoSeedAvailable handled.
4. Research stub — always deferred.
5. Optional HEARTBEAT: memory per emit_memory config (always /
   conditional / never).
6. Atomic state save; append JSONL audit entry.

Dry-run short-circuits before any writes.

12 new tests; 24 heartbeat engine tests total; 317 suite.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `nell heartbeat` CLI wiring

**Goal:** Replace `"heartbeat"` stub in `brain/cli.py` with real subparser + handler. Mirror the pattern from `_dream_handler` (nested try/finally for DB close, persona-dir validation, etc.).

**Files:**
- Modify: `/Users/hanamori/companion-emergence/brain/cli.py`
- Modify: `/Users/hanamori/companion-emergence/tests/unit/brain/test_cli.py` (drop `"heartbeat"` from stubs list)
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/engines/test_cli_heartbeat.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/engines/test_cli_heartbeat.py`:

```python
"""Tests for the `nell heartbeat` CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def nell_persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "persona_root"
    root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(root))

    from brain.paths import get_persona_dir

    persona = get_persona_dir("nell")
    persona.mkdir(parents=True)

    store = MemoryStore(db_path=persona / "memories.db")
    seed = Memory.create_new(
        content="seed for heartbeat",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0},
    )
    seed.importance = 8.0
    store.create(seed)
    store.close()

    h = HebbianMatrix(db_path=persona / "hebbian.db")
    h.close()
    return persona


def test_nell_heartbeat_first_tick_initializes(nell_persona: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """First `nell heartbeat --trigger open` creates state + config + log entry; defers work."""
    from brain.cli import main

    rc = main(["heartbeat", "--trigger", "open", "--provider", "fake"])
    assert rc == 0
    assert (nell_persona / "heartbeat_state.json").exists()
    # config is not written on init by default; only loaded-with-defaults
    assert (nell_persona / "heartbeats.log.jsonl").exists()
    log_line = (nell_persona / "heartbeats.log.jsonl").read_text().strip()
    assert json.loads(log_line)["initialized"] is True


def test_nell_heartbeat_second_tick_does_work(nell_persona: Path) -> None:
    """Second invocation does real decay and writes updated state."""
    from brain.cli import main

    main(["heartbeat", "--trigger", "open", "--provider", "fake"])  # init
    rc = main(["heartbeat", "--trigger", "close", "--provider", "fake"])
    assert rc == 0

    state = json.loads((nell_persona / "heartbeat_state.json").read_text())
    assert state["tick_count"] == 1
    assert state["last_trigger"] == "close"


def test_nell_heartbeat_dry_run_no_writes(nell_persona: Path) -> None:
    """--dry-run doesn't create state file or log entry."""
    from brain.cli import main

    rc = main(["heartbeat", "--trigger", "manual", "--provider", "fake", "--dry-run"])
    assert rc == 0
    assert not (nell_persona / "heartbeat_state.json").exists()
    assert not (nell_persona / "heartbeats.log.jsonl").exists()


def test_nell_heartbeat_unknown_persona_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing persona dir → FileNotFoundError."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path / "empty"))
    from brain.cli import main

    with pytest.raises(FileNotFoundError, match="persona"):
        main(["heartbeat", "--trigger", "manual", "--persona", "ghost", "--provider", "fake"])


def test_nell_heartbeat_unknown_trigger_rejected(nell_persona: Path) -> None:
    """Argparse rejects --trigger values outside the enum."""
    from brain.cli import main

    with pytest.raises(SystemExit):
        main(["heartbeat", "--trigger", "frobnicate", "--provider", "fake"])
```

- [ ] **Step 2: Update `tests/unit/brain/test_cli.py`**

Find the stubs parametrize list. Remove `"heartbeat"` (same pattern as W3.5 removed `"migrate"` and W4 removed `"dream"`).

```bash
grep -n '"heartbeat"' /Users/hanamori/companion-emergence/tests/unit/brain/test_cli.py
```

- [ ] **Step 3: Wire `nell heartbeat` in `brain/cli.py`**

Read `brain/cli.py`. Apply changes:

1. Remove `"heartbeat"` from `_STUB_COMMANDS`.

2. Add import alongside existing engine imports:
```python
from brain.engines.heartbeat import HeartbeatEngine
```

3. Add `_heartbeat_handler` after `_dream_handler`:

```python
def _heartbeat_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell heartbeat` to the HeartbeatEngine."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir} — "
            f"run `nell migrate --install-as {args.persona}` first."
        )
    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        hebbian = HebbianMatrix(db_path=persona_dir / "hebbian.db")
        try:
            provider = get_provider(args.provider)
            engine = HeartbeatEngine(
                store=store,
                hebbian=hebbian,
                provider=provider,
                state_path=persona_dir / "heartbeat_state.json",
                config_path=persona_dir / "heartbeat_config.json",
                dream_log_path=persona_dir / "dreams.log.jsonl",
                heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            )
            result = engine.run_tick(trigger=args.trigger, dry_run=args.dry_run)
        finally:
            hebbian.close()
    finally:
        store.close()

    if result.initialized:
        print("Heartbeat initialized — work deferred until next tick.")
    elif args.dry_run:
        print("Heartbeat dry-run — no writes.")
        print(f"  elapsed: {result.elapsed_seconds / 3600:.2f}h")
        print(f"  would decay: {result.memories_decayed} memories")
        print(f"  would prune: {result.edges_pruned} edges")
        print(f"  dream: {'would fire' if result.dream_id else result.dream_gated_reason}")
    else:
        print(f"Heartbeat tick complete ({args.trigger}).")
        print(f"  elapsed: {result.elapsed_seconds / 3600:.2f}h")
        print(f"  decayed: {result.memories_decayed} memories, pruned {result.edges_pruned} edges")
        if result.dream_id:
            print(f"  dream fired: {result.dream_id}")
        else:
            print(f"  dream gated: {result.dream_gated_reason}")
    return 0
```

4. In `_build_parser()`, AFTER the migrate + dream subparser blocks, add:

```python
    hb_sub = subparsers.add_parser(
        "heartbeat",
        help="Run one heartbeat orchestrator tick against a persona.",
    )
    hb_sub.add_argument("--persona", default="nell")
    hb_sub.add_argument(
        "--trigger",
        choices=["open", "close", "manual"],
        default="manual",
    )
    hb_sub.add_argument(
        "--provider", default="claude-cli",
        help="LLM provider: claude-cli (default), fake, ollama.",
    )
    hb_sub.add_argument("--dry-run", action="store_true")
    hb_sub.set_defaults(func=_heartbeat_handler)
```

- [ ] **Step 4: Run tests, expect green**

```bash
uv run pytest tests/unit/brain/engines/test_cli_heartbeat.py -v
uv run pytest tests/unit/brain/test_cli.py -v
```
Expected: 5 new tests pass; existing test_cli.py green with heartbeat removed from stubs.

- [ ] **Step 5: Manual smoke**

```bash
uv run nell heartbeat --help
```
Expected: real help text with --persona, --trigger, --provider, --dry-run.

- [ ] **Step 6: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```
Expected: 322 passed (317 + 5). Ruff clean.

- [ ] **Step 7: Commit**

```bash
git add brain/cli.py tests/unit/brain/engines/test_cli_heartbeat.py tests/unit/brain/test_cli.py
git commit -m "feat(brain/cli): wire nell heartbeat subcommand to HeartbeatEngine

_heartbeat_handler opens the persona's memories.db + hebbian.db via
get_persona_dir(), constructs HeartbeatEngine with state + config +
log paths inside the persona dir, runs one tick, prints the result
summary (initialized / dry-run / real tick).

Nested try/finally on DB opens (same pattern as dream handler) so
HebbianMatrix open failure still closes MemoryStore cleanly.

heartbeat moves out of _STUB_COMMANDS; stub test expectation dropped.

5 new CLI tests; 322 total.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Close-out — CI + merge (no tag)

**Goal:** Verify CI green on 3 OSes, merge PR. No tag (reflex + research still pending; Week 4 tag waits for all 4 engines).

- [ ] **Step 1: Fresh install + verify**

```bash
cd /Users/hanamori/companion-emergence
rm -rf .venv
uv sync --all-extras
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
uv run nell heartbeat --help
```
Expected: 322 passed, ruff clean, help text prints.

- [ ] **Step 2: Dry-run smoke against real nell persona**

```bash
uv run nell heartbeat --persona nell --trigger manual --provider fake --dry-run
```
Expected: prints "Heartbeat dry-run — no writes" + summary. No changes to `~/Library/Application Support/companion-emergence/personas/nell/`.

- [ ] **Step 3: Push + PR**

```bash
git push -u origin week-4-heartbeat
gh pr create --title "feat: Week 4.5 — heartbeat engine (event-driven orchestrator tick)" --body "$(cat <<'EOF'
## Summary
- Adds `brain/engines/heartbeat.py` — event-driven orchestrator (HeartbeatEngine.run_tick + HeartbeatState + HeartbeatConfig + HeartbeatResult)
- Wires `nell heartbeat --trigger open|close|manual` subcommand
- Composes DreamEngine (W4) + apply_decay (W2) + HebbianMatrix.decay_all/gc (W3) — pure composition, no new dependencies
- 29 new tests; suite reaches 322 across macOS + Windows + Linux

## Per-task
| Task | Purpose | Tests |
|---|---|---|
| 1. Types + scaffold | HeartbeatConfig + HeartbeatState + HeartbeatResult + engine stub | 12 |
| 2. run_tick body | Decay + dream-gate + research stub + memory emit | 12 |
| 3. CLI wiring | `nell heartbeat` subparser + handler | 5 (−1 stub) |

## Design
- **Not a daemon.** Application lifecycle hooks (open/close) trigger `nell heartbeat`. Two ticks per session minimum.
- **First-ever tick defers** — fresh persona doesn't eat "years of decay" on boot.
- **Dream-gated** by `heartbeat_config.json:dream_every_hours` (default 24h).
- **Research stubbed** — deferred with log entry.
- **Optional HEARTBEAT: memory** per `emit_memory` config (always / conditional / never).

## Test plan
- [x] Fresh `uv sync --all-extras` succeeds
- [x] 322 tests pass locally
- [x] ruff check + format clean
- [x] `nell heartbeat --help` prints real usage
- [x] `nell heartbeat --dry-run` smoke against real nell persona
- [ ] CI matrix green across macOS + Ubuntu + Windows
- [ ] Hana runs `nell heartbeat --persona nell.sandbox --trigger manual` twice (init + real) and inspects
EOF
)"
```

- [ ] **Step 4: Watch CI**

```bash
sleep 15
gh run list --branch week-4-heartbeat --limit 1
gh run watch
```

- [ ] **Step 5: Merge + sync**

```bash
gh pr merge --merge --delete-branch
git checkout main
git pull origin main
```

**No tag.** Week 4 tag waits until reflex + research also ship.

---

## Week 4.5 green-light criterion

1. `uv sync --all-extras` fresh install succeeds.
2. `uv run pytest` → 322 passed.
3. Ruff clean.
4. `uv run nell heartbeat --help` prints real usage.
5. `uv run nell heartbeat --persona nell --trigger manual --dry-run` runs without error.
6. CI green on 3 OSes.
7. PR merged to main.

**User-side (not part of automated criterion):**
- Hana runs `nell heartbeat --persona nell.sandbox --trigger manual` twice and verifies the first initializes + the second does real work.
- If satisfied, runs against canonical nell.

---

## Notes for the engineer executing this plan

- **First-tick defer rule is load-bearing.** Without it, Nell's 1,142 migrated memories would eat "years of decay" on the first invocation. Task 2 Step 3 implements this; Task 1 Step 3 tests it indirectly via the load-returns-None contract.
- **Dream gating applies per-persona.** The `last_dream_at` field starts at init-time, so the first real tick (second invocation) won't dream unless at least `dream_every_hours` has elapsed since init. For fast testing, set `dream_every_hours: 0.001` in config.
- **Hebbian decay is time-scaled.** `decay_rate_per_tick` is the rate for a 24h tick. Shorter elapsed times decay proportionally less. A 10-minute tick barely touches weights; a 48h tick doubles the base rate.
- **Atomic state writes.** `HeartbeatState.save` writes to `<path>.new` then `os.replace`. Never `open(path, 'w')` directly on state.
- **Research engine is intentionally stubbed.** Every tick sets `research_deferred=True` and logs it. Don't implement research in this plan — it's a separate future engine.
- **Heartbeat memory prompt.** Short and present-tense, first-person, prefixed `HEARTBEAT:`. System prompt in `_emit_heartbeat_memory` is the single source of truth for the persona voice on heartbeat memories; no external template file needed.

---

*End of Week 4.5 plan.*
