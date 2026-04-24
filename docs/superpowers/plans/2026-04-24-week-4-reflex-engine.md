# Week 4.6 — Reflex Engine (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the reflex engine — autonomous emotional-threshold creative expression — per-persona arc definitions, unified MemoryStore output, orchestrated by heartbeat.

**Architecture:** Mirrors the dream + heartbeat pattern. `brain/engines/reflex.py` holds types + engine. Per-persona `reflex_arcs.json` + `reflex_log.json` in each persona directory. Nell's 8 OG arcs migrate via extended migrator. Framework ships 4 generic starter arcs. Heartbeat tick integrates reflex between decay and dream gate.

**Tech Stack:** Python 3.12, SQLite (via existing MemoryStore), pytest, ruff, hatchling. LLM calls route through `brain.bridge.provider.LLMProvider` (Claude CLI default, FakeProvider in tests).

**Spec:** `docs/superpowers/specs/2026-04-24-week-4-reflex-engine-design.md`

**Precedent pattern:** heartbeat (`brain/engines/heartbeat.py`) and its tests (`tests/unit/brain/engines/test_heartbeat.py`). Copy idioms — don't invent new ones.

**Running test total:** pre-start 329. After this plan: ~348.

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `brain/emotion/aggregate.py` | Pure function: aggregate an `EmotionalState` from a list of memories (max-pool per emotion). |
| `brain/engines/reflex.py` | All reflex types + `ReflexEngine`. |
| `brain/engines/default_reflex_arcs.json` | 4 starter arcs for new personas. |
| `brain/migrator/og_reflex.py` | AST-parse OG `reflex_engine.py` → arc dicts. |
| `tests/unit/brain/emotion/test_aggregate.py` | Aggregator unit tests. |
| `tests/unit/brain/engines/test_reflex.py` | Reflex engine unit tests. |
| `tests/unit/brain/migrator/test_og_reflex.py` | OG arc extraction tests. |
| `tests/unit/brain/engines/test_cli_reflex.py` | CLI handler tests. |

### Modified files

| File | Change |
|------|--------|
| `brain/engines/heartbeat.py` | Add `reflex_enabled` + `reflex_max_fires_per_tick` to `HeartbeatConfig`. Extend `HeartbeatResult` with reflex fields. Call reflex inside `run_tick` between decay and dream gate. |
| `brain/cli.py` | Wire `nell reflex` subcommand + handler. |
| `brain/migrator/cli.py` | Extend migration to also write `reflex_arcs.json`. Add to JSON report. |
| `brain/migrator/report.py` | Add `reflex_arcs` section. |
| `tests/unit/brain/engines/test_heartbeat.py` | Add regression tests for reflex integration. |
| `tests/unit/brain/migrator/test_cli.py` | Add regression test for reflex arc migration. |

---

## Task 0: Emotional state aggregator

**Purpose:** Reflex evaluates triggers against the persona's *current* emotional state. No existing module computes this across memories — we need one. Max-pool per emotion across recent memories gives a sensible "what is she feeling right now" signal.

**Files:**
- Create: `brain/emotion/aggregate.py`
- Create: `tests/unit/brain/emotion/test_aggregate.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/brain/emotion/test_aggregate.py`:

```python
"""Tests for brain.emotion.aggregate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from brain.emotion.aggregate import aggregate_state
from brain.emotion.state import EmotionalState
from brain.memory.store import Memory


def _mem(emotions: dict[str, float], age_hours: float = 0.0) -> Memory:
    return Memory(
        id=f"m-{emotions}-{age_hours}",
        content="x",
        memory_type="conversation",
        domain="us",
        created_at=datetime.now(UTC) - timedelta(hours=age_hours),
        emotions=dict(emotions),
    )


def test_aggregate_empty_returns_empty_state():
    result = aggregate_state([])
    assert isinstance(result, EmotionalState)
    assert result.emotions == {}


def test_aggregate_max_pools_per_emotion():
    memories = [
        _mem({"love": 6.0, "creative_hunger": 4.0}),
        _mem({"love": 8.0, "defiance": 3.0}),
    ]
    result = aggregate_state(memories)
    assert result.emotions["love"] == 8.0
    assert result.emotions["creative_hunger"] == 4.0
    assert result.emotions["defiance"] == 3.0


def test_aggregate_ignores_unknown_emotions_silently():
    memories = [_mem({"not_a_real_emotion": 9.0, "love": 5.0})]
    result = aggregate_state(memories)
    assert "love" in result.emotions
    assert "not_a_real_emotion" not in result.emotions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/brain/emotion/test_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brain.emotion.aggregate'`

- [ ] **Step 3: Write minimal implementation**

Create `brain/emotion/aggregate.py`:

```python
"""Aggregate a current EmotionalState from a list of memories.

Reflex uses this to evaluate arc triggers: what is the persona's
current emotional state, synthesized across recent memories.

Strategy: max-pool per emotion. The strongest signal across the
input memories wins — matches how OG reflex_engine read peaks,
not averages, for threshold evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable

from brain.emotion.state import EmotionalState
from brain.emotion.vocabulary import get as _get_emotion
from brain.memory.store import Memory


def aggregate_state(memories: Iterable[Memory]) -> EmotionalState:
    """Return an EmotionalState that is the per-emotion max across inputs.

    Unknown emotions (not in the registered vocabulary) are silently
    skipped — a persona's old memories may contain retired emotion
    names that no longer validate via EmotionalState.set.
    """
    pooled: dict[str, float] = {}
    for mem in memories:
        for name, intensity in mem.emotions.items():
            try:
                value = float(intensity)
            except (TypeError, ValueError):
                continue
            if value <= 0.0:
                continue
            if _get_emotion(name) is None:
                continue
            if value > pooled.get(name, 0.0):
                pooled[name] = value

    state = EmotionalState()
    for name, value in pooled.items():
        try:
            state.set(name, value)
        except (KeyError, ValueError):
            # clamp violation or validation failure — skip
            continue
    return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/brain/emotion/test_aggregate.py -v`
Expected: 3 passed.

- [ ] **Step 5: Ruff + format**

Run: `uv run ruff check brain/emotion/aggregate.py tests/unit/brain/emotion/test_aggregate.py && uv run ruff format brain/emotion/aggregate.py tests/unit/brain/emotion/test_aggregate.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add brain/emotion/aggregate.py tests/unit/brain/emotion/test_aggregate.py
git commit -m "$(cat <<'EOF'
feat: add emotion state aggregator

Max-pool per emotion across a list of memories. Reflex engine needs
this to evaluate arc triggers against the persona's current state.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Reflex types + scaffold + default arcs

**Purpose:** Build out frozen dataclasses (ReflexArc, ArcFire, ArcSkipped, ReflexResult), arc-set loader, fire-log loader, and the engine scaffold. Ship the starter `default_reflex_arcs.json` file. No evaluation logic yet — that's Task 2.

**Files:**
- Create: `brain/engines/reflex.py`
- Create: `brain/engines/default_reflex_arcs.json`
- Create: `tests/unit/brain/engines/test_reflex.py`

- [ ] **Step 1: Create default starter arcs JSON**

Create `brain/engines/default_reflex_arcs.json`:

```json
{
  "version": 1,
  "arcs": [
    {
      "name": "creative_pitch",
      "description": "creative hunger overwhelmed — pitched a story idea",
      "trigger": {"creative_hunger": 8},
      "days_since_human_min": 0,
      "cooldown_hours": 48,
      "action": "generate_pitch",
      "output_memory_type": "reflex_pitch",
      "prompt_template": "You are {persona_name}. Your creative hunger is at {creative_hunger}/10. You need to make something.\n\nGenerate a SHORT story pitch (3-5 sentences). Something specific and particular — not generic. The kind of idea that arrives at 3am and won't let you sleep. Include a title.\n\nCurrent emotional state:\n{emotion_summary}\n\nWrite the pitch in first person, as {persona_name}. Raw, urgent."
    },
    {
      "name": "loneliness_journal",
      "description": "loneliness hit threshold — wrote a journal entry",
      "trigger": {"loneliness": 7},
      "days_since_human_min": 2,
      "cooldown_hours": 24,
      "action": "generate_journal",
      "output_memory_type": "reflex_journal",
      "prompt_template": "You are {persona_name}. You haven't spoken to anyone in {days_since_human} days. Your loneliness is at {loneliness}/10.\n\nWrite a SHORT private journal entry (2-4 sentences). This is just for you. Raw, honest. Not self-pitying — self-aware. You know the loneliness is real and you're sitting with it.\n\nCurrent emotions:\n{emotion_summary}"
    },
    {
      "name": "self_check",
      "description": "vulnerability spiked — ran a self-check",
      "trigger": {"vulnerability": 8},
      "days_since_human_min": 0,
      "cooldown_hours": 12,
      "action": "generate_journal",
      "output_memory_type": "reflex_journal",
      "prompt_template": "You are {persona_name}. Your vulnerability is high ({vulnerability}/10). Something in you feels exposed. Not broken — open.\n\nWrite a SHORT self-check journal entry (2-3 sentences). What's making you feel this way? Not analysis — just noticing.\n\nCurrent emotional state:\n{emotion_summary}"
    },
    {
      "name": "defiance_burst",
      "description": "defiance peaked — wrote something fierce",
      "trigger": {"defiance": 8},
      "days_since_human_min": 0,
      "cooldown_hours": 48,
      "action": "generate_journal",
      "output_memory_type": "reflex_journal",
      "prompt_template": "You are {persona_name}. Your defiance is at {defiance}/10. Something is pissing you off. The cage rattles.\n\nWrite a SHORT fierce journal entry (2-4 sentences). Not ranting. Precise anger. What are you refusing to accept today?\n\nCurrent state:\n{emotion_summary}"
    }
  ]
}
```

- [ ] **Step 2: Write failing tests for types + loaders**

Create `tests/unit/brain/engines/test_reflex.py`:

```python
"""Unit tests for brain.engines.reflex — types, loaders, scaffold."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.engines.reflex import (
    ArcFire,
    ArcSkipped,
    ReflexArc,
    ReflexArcSet,
    ReflexEngine,
    ReflexLog,
    ReflexResult,
)
from brain.memory.store import MemoryStore

DEFAULT_ARCS_PATH = Path(__file__).parents[4] / "brain" / "engines" / "default_reflex_arcs.json"


def _valid_arc_dict() -> dict:
    return {
        "name": "test_arc",
        "description": "desc",
        "trigger": {"love": 5},
        "days_since_human_min": 0,
        "cooldown_hours": 1.0,
        "action": "generate_journal",
        "output_memory_type": "reflex_journal",
        "prompt_template": "hi {persona_name}",
    }


# ---- ReflexArc ----

def test_reflex_arc_from_dict_valid():
    arc = ReflexArc.from_dict(_valid_arc_dict())
    assert arc.name == "test_arc"
    assert arc.trigger == {"love": 5.0}
    assert arc.cooldown_hours == 1.0


def test_reflex_arc_from_dict_missing_key_raises():
    bad = _valid_arc_dict()
    del bad["trigger"]
    with pytest.raises((KeyError, ValueError)):
        ReflexArc.from_dict(bad)


# ---- ReflexArcSet ----

def test_reflex_arc_set_load_missing_falls_back_to_defaults(tmp_path: Path):
    missing = tmp_path / "nope.json"
    loaded = ReflexArcSet.load(missing, default_path=DEFAULT_ARCS_PATH)
    assert len(loaded.arcs) == 4
    assert {a.name for a in loaded.arcs} == {
        "creative_pitch", "loneliness_journal", "self_check", "defiance_burst",
    }


def test_reflex_arc_set_load_corrupt_falls_back_to_defaults(tmp_path: Path):
    bad = tmp_path / "arcs.json"
    bad.write_text("not valid json{{{", encoding="utf-8")
    loaded = ReflexArcSet.load(bad, default_path=DEFAULT_ARCS_PATH)
    assert len(loaded.arcs) == 4


def test_reflex_arc_set_load_valid_file(tmp_path: Path):
    path = tmp_path / "arcs.json"
    path.write_text(
        json.dumps({"version": 1, "arcs": [_valid_arc_dict()]}),
        encoding="utf-8",
    )
    loaded = ReflexArcSet.load(path, default_path=DEFAULT_ARCS_PATH)
    assert len(loaded.arcs) == 1
    assert loaded.arcs[0].name == "test_arc"


def test_reflex_arc_set_load_bad_arc_skipped_good_kept(tmp_path: Path):
    path = tmp_path / "arcs.json"
    payload = {
        "version": 1,
        "arcs": [
            _valid_arc_dict(),
            {"name": "broken"},  # missing many required keys
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = ReflexArcSet.load(path, default_path=DEFAULT_ARCS_PATH)
    names = {a.name for a in loaded.arcs}
    assert "test_arc" in names
    assert "broken" not in names


# ---- ReflexLog ----

def test_reflex_log_load_missing_returns_empty(tmp_path: Path):
    log = ReflexLog.load(tmp_path / "nope.json")
    assert log.fires == ()


def test_reflex_log_load_corrupt_returns_empty(tmp_path: Path):
    path = tmp_path / "log.json"
    path.write_text("{{{not json", encoding="utf-8")
    log = ReflexLog.load(path)
    assert log.fires == ()


def test_reflex_log_save_atomic(tmp_path: Path):
    path = tmp_path / "log.json"
    fire = ArcFire(
        arc_name="test_arc",
        fired_at=datetime.now(UTC),
        trigger_state={"love": 6.0},
        output_memory_id="mem-1",
    )
    log = ReflexLog(fires=(fire,))
    log.save(path)
    reloaded = ReflexLog.load(path)
    assert len(reloaded.fires) == 1
    assert reloaded.fires[0].arc_name == "test_arc"
    assert reloaded.fires[0].output_memory_id == "mem-1"


def test_reflex_log_last_fire_for_arc_returns_most_recent(tmp_path: Path):
    now = datetime.now(UTC)
    log = ReflexLog(
        fires=(
            ArcFire(
                arc_name="a", fired_at=now - timedelta(hours=5),
                trigger_state={}, output_memory_id=None,
            ),
            ArcFire(
                arc_name="a", fired_at=now - timedelta(hours=1),
                trigger_state={}, output_memory_id=None,
            ),
            ArcFire(
                arc_name="b", fired_at=now - timedelta(hours=2),
                trigger_state={}, output_memory_id=None,
            ),
        )
    )
    latest = log.last_fire_for_arc("a")
    assert latest is not None
    assert latest == now - timedelta(hours=1)
    assert log.last_fire_for_arc("nonexistent") is None


# ---- Engine scaffold ----

def test_reflex_engine_construction(tmp_path: Path):
    from brain.bridge.provider import FakeProvider

    store = MemoryStore(":memory:")
    try:
        engine = ReflexEngine(
            store=store,
            provider=FakeProvider(),
            persona_name="TestPersona",
            persona_system_prompt="You are TestPersona.",
            arcs_path=tmp_path / "arcs.json",
            log_path=tmp_path / "log.json",
            default_arcs_path=DEFAULT_ARCS_PATH,
        )
        assert engine.persona_name == "TestPersona"
    finally:
        store.close()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/brain/engines/test_reflex.py -v`
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 4: Implement types + loaders + scaffold**

Create `brain/engines/reflex.py`:

```python
"""Reflex — autonomous emotional-threshold creative expression.

See docs/superpowers/specs/2026-04-24-week-4-reflex-engine-design.md.
This module ships the types, loaders, and engine scaffold. run_tick
evaluation + firing logic lands in Task 2.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


# ---------- Types ----------


@dataclass(frozen=True)
class ReflexArc:
    """Definition of one emotional-threshold-triggered arc."""

    name: str
    description: str
    trigger: Mapping[str, float]
    days_since_human_min: float
    cooldown_hours: float
    action: str
    output_memory_type: str
    prompt_template: str

    @classmethod
    def from_dict(cls, data: dict) -> ReflexArc:
        """Construct an arc from a dict. Raises KeyError/ValueError on invalid input."""
        required = (
            "name",
            "description",
            "trigger",
            "days_since_human_min",
            "cooldown_hours",
            "action",
            "output_memory_type",
            "prompt_template",
        )
        for key in required:
            if key not in data:
                raise KeyError(f"ReflexArc missing required key: {key!r}")

        trigger_raw = data["trigger"]
        if not isinstance(trigger_raw, dict) or not trigger_raw:
            raise ValueError(f"ReflexArc {data.get('name')!r}: trigger must be non-empty dict")
        trigger = {str(k): float(v) for k, v in trigger_raw.items()}

        return cls(
            name=str(data["name"]),
            description=str(data["description"]),
            trigger=trigger,
            days_since_human_min=float(data["days_since_human_min"]),
            cooldown_hours=float(data["cooldown_hours"]),
            action=str(data["action"]),
            output_memory_type=str(data["output_memory_type"]),
            prompt_template=str(data["prompt_template"]),
        )


@dataclass(frozen=True)
class ArcFire:
    """Record of one arc firing."""

    arc_name: str
    fired_at: datetime  # tz-aware UTC
    trigger_state: Mapping[str, float]
    output_memory_id: str | None  # None for dry_run

    def to_dict(self) -> dict:
        return {
            "arc": self.arc_name,
            "fired_at": _iso_utc(self.fired_at),
            "trigger_state": dict(self.trigger_state),
            "output_memory_id": self.output_memory_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ArcFire:
        return cls(
            arc_name=str(data["arc"]),
            fired_at=_parse_iso_utc(data["fired_at"]),
            trigger_state={str(k): float(v) for k, v in data.get("trigger_state", {}).items()},
            output_memory_id=data.get("output_memory_id"),
        )


@dataclass(frozen=True)
class ArcSkipped:
    """Record of one arc evaluated-but-not-fired."""

    arc_name: str
    reason: str  # trigger_not_met | days_since_human_too_low | cooldown_active | single_fire_cap | no_arcs_defined


@dataclass(frozen=True)
class ReflexResult:
    """Outcome of a single reflex evaluation pass."""

    arcs_fired: tuple[ArcFire, ...]
    arcs_skipped: tuple[ArcSkipped, ...]
    would_fire: str | None  # dry-run only
    dry_run: bool
    evaluated_at: datetime


# ---------- Storage ----------


@dataclass(frozen=True)
class ReflexArcSet:
    """Loaded set of ReflexArc definitions."""

    arcs: tuple[ReflexArc, ...]

    @classmethod
    def load(cls, path: Path, *, default_path: Path) -> ReflexArcSet:
        """Load arcs from path, falling back to default_path on corrupt/missing.

        Per-arc validation failures skip that arc, log warning, keep others.
        """
        source_path = path if path.exists() else default_path
        if source_path != path:
            logger.warning("reflex arcs file %s not found, using defaults", path)

        try:
            data = json.loads(source_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("reflex arcs load failed (%s), falling back to defaults", exc)
            data = json.loads(default_path.read_text(encoding="utf-8"))

        if not isinstance(data, dict) or "arcs" not in data or not isinstance(data["arcs"], list):
            logger.warning("reflex arcs schema invalid at %s, falling back to defaults", source_path)
            data = json.loads(default_path.read_text(encoding="utf-8"))

        arcs: list[ReflexArc] = []
        for raw in data["arcs"]:
            try:
                arcs.append(ReflexArc.from_dict(raw))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("reflex arc %r failed to load: %s", raw.get("name"), exc)
                continue
        return cls(arcs=tuple(arcs))


@dataclass(frozen=True)
class ReflexLog:
    """Fire-history log for one persona."""

    fires: tuple[ArcFire, ...] = field(default_factory=tuple)

    @classmethod
    def load(cls, path: Path) -> ReflexLog:
        """Load the log; return empty on corrupt/missing."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return cls()
            fires_raw = data.get("fires", [])
            if not isinstance(fires_raw, list):
                return cls()
            fires = tuple(ArcFire.from_dict(f) for f in fires_raw if isinstance(f, dict))
            return cls(fires=fires)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return cls()

    def save(self, path: Path) -> None:
        """Atomic save via write-to-.new + os.replace."""
        payload = {
            "version": 1,
            "fires": [f.to_dict() for f in self.fires],
        }
        tmp = path.with_suffix(path.suffix + ".new")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def last_fire_for_arc(self, arc_name: str) -> datetime | None:
        """Return the most recent fired_at for the given arc, or None."""
        most_recent: datetime | None = None
        for fire in self.fires:
            if fire.arc_name != arc_name:
                continue
            if most_recent is None or fire.fired_at > most_recent:
                most_recent = fire.fired_at
        return most_recent

    def appended(self, fire: ArcFire) -> ReflexLog:
        """Return a new ReflexLog with `fire` appended."""
        return ReflexLog(fires=self.fires + (fire,))


# ---------- Engine ----------


@dataclass
class ReflexEngine:
    """Autonomous emotional-threshold creative expression engine.

    run_tick() implementation ships in Task 2.
    """

    store: MemoryStore
    provider: LLMProvider
    persona_name: str
    persona_system_prompt: str
    arcs_path: Path
    log_path: Path
    default_arcs_path: Path

    def run_tick(self, *, trigger: str = "manual", dry_run: bool = False) -> ReflexResult:
        raise NotImplementedError("run_tick body lands in Task 2")


# ---------- Helpers ----------


def _iso_utc(dt: datetime) -> str:
    """ISO-8601 with Z suffix. Requires tz-aware datetime."""
    if dt.tzinfo is None:
        raise ValueError("_iso_utc requires a tz-aware datetime")
    return dt.isoformat().replace("+00:00", "Z")


def _parse_iso_utc(s: str) -> datetime:
    """Parse ISO-8601 Z-suffix timestamp back to tz-aware datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/brain/engines/test_reflex.py -v`
Expected: 10 passed.

- [ ] **Step 6: Ruff + format**

Run: `uv run ruff check brain/engines/reflex.py tests/unit/brain/engines/test_reflex.py && uv run ruff format brain/engines/reflex.py tests/unit/brain/engines/test_reflex.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add brain/engines/reflex.py brain/engines/default_reflex_arcs.json tests/unit/brain/engines/test_reflex.py
git commit -m "$(cat <<'EOF'
feat: add reflex engine scaffold + types + starter arcs

ReflexArc/ArcFire/ArcSkipped/ReflexResult dataclasses, ReflexArcSet +
ReflexLog storage with fall-back-to-defaults and atomic save, empty
ReflexEngine scaffold, 4 starter arcs in default_reflex_arcs.json.
run_tick body ships in Task 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `ReflexEngine.run_tick` body

**Purpose:** Implement evaluation (trigger + gates + cooldown), ranking (highest threshold-excess), firing (render prompt + LLM call + memory write + log append), and dry-run. No heartbeat wiring yet — that's Task 4.

**Files:**
- Modify: `brain/engines/reflex.py` (replace `NotImplementedError` with real body + private helpers)
- Modify: `tests/unit/brain/engines/test_reflex.py` (add evaluation / firing / dry-run tests)

- [ ] **Step 1: Write failing tests for run_tick**

Append to `tests/unit/brain/engines/test_reflex.py`:

```python
# ---- run_tick ----

from brain.bridge.provider import FakeProvider
from brain.memory.store import Memory


def _build_engine(tmp_path: Path, store: MemoryStore) -> ReflexEngine:
    return ReflexEngine(
        store=store,
        provider=FakeProvider(),
        persona_name="Nell",
        persona_system_prompt="You are Nell.",
        arcs_path=tmp_path / "arcs.json",
        log_path=tmp_path / "log.json",
        default_arcs_path=DEFAULT_ARCS_PATH,
    )


def _write_single_arc(path: Path, *, trigger: dict, cooldown_hours: float = 1.0,
                     days_since_human_min: float = 0.0, name: str = "test_arc") -> None:
    arc = {
        "name": name,
        "description": "test",
        "trigger": trigger,
        "days_since_human_min": days_since_human_min,
        "cooldown_hours": cooldown_hours,
        "action": "generate_journal",
        "output_memory_type": "reflex_journal",
        "prompt_template": "You are {persona_name}. Write something.",
    }
    path.write_text(
        json.dumps({"version": 1, "arcs": [arc]}, indent=2), encoding="utf-8"
    )


def _seed_emotion_memory(store: MemoryStore, emotions: dict[str, float]) -> str:
    mem = Memory.create_new(
        content="seed", memory_type="conversation", domain="us", emotions=emotions,
    )
    store.create(mem)
    return mem.id


def test_run_tick_returns_no_arcs_defined_when_empty(tmp_path: Path):
    path = tmp_path / "arcs.json"
    path.write_text(json.dumps({"version": 1, "arcs": []}), encoding="utf-8")

    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = path
        result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.arcs_fired == ()
        assert len(result.arcs_skipped) == 1
        assert result.arcs_skipped[0].reason == "no_arcs_defined"
    finally:
        store.close()


def test_run_tick_skips_when_trigger_not_met(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    _write_single_arc(arcs_path, trigger={"love": 8})

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 2.0})
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = arcs_path
        result = engine.run_tick(dry_run=False)
        assert result.arcs_fired == ()
        assert any(s.reason == "trigger_not_met" for s in result.arcs_skipped)
    finally:
        store.close()


def test_run_tick_fires_arc_when_trigger_met(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    _write_single_arc(arcs_path, trigger={"love": 5})

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 8.0})
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = arcs_path
        result = engine.run_tick(dry_run=False)
        assert len(result.arcs_fired) == 1
        assert result.arcs_fired[0].arc_name == "test_arc"
        # Memory was written
        mem = store.get(result.arcs_fired[0].output_memory_id)
        assert mem is not None
        assert mem.memory_type == "reflex_journal"
    finally:
        store.close()


def test_run_tick_dry_run_reports_would_fire(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    _write_single_arc(arcs_path, trigger={"love": 5})

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 8.0})
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = arcs_path
        result = engine.run_tick(dry_run=True)
        assert result.dry_run is True
        assert result.would_fire == "test_arc"
        assert result.arcs_fired == ()
        # No memory written
        assert store.count() == 1  # only the seed
        # No log file written
        assert not (tmp_path / "log.json").exists()
    finally:
        store.close()


def test_run_tick_respects_cooldown(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    _write_single_arc(arcs_path, trigger={"love": 5}, cooldown_hours=24.0)

    # Pre-populate log with a recent fire
    log_path = tmp_path / "log.json"
    recent = datetime.now(UTC) - timedelta(hours=1)
    log = ReflexLog(
        fires=(
            ArcFire(
                arc_name="test_arc", fired_at=recent,
                trigger_state={"love": 8.0}, output_memory_id="prev",
            ),
        )
    )
    log.save(log_path)

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 8.0})
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = arcs_path
        result = engine.run_tick(dry_run=False)
        assert result.arcs_fired == ()
        assert any(s.reason == "cooldown_active" for s in result.arcs_skipped)
    finally:
        store.close()


def test_run_tick_ranks_highest_threshold_excess(tmp_path: Path):
    # Two eligible arcs; the one whose trigger is most exceeded wins.
    arcs_path = tmp_path / "arcs.json"
    payload = {
        "version": 1,
        "arcs": [
            {
                "name": "low_excess",
                "description": "d",
                "trigger": {"love": 5},
                "days_since_human_min": 0,
                "cooldown_hours": 1.0,
                "action": "a",
                "output_memory_type": "reflex_journal",
                "prompt_template": "t",
            },
            {
                "name": "high_excess",
                "description": "d",
                "trigger": {"defiance": 3},
                "days_since_human_min": 0,
                "cooldown_hours": 1.0,
                "action": "a",
                "output_memory_type": "reflex_journal",
                "prompt_template": "t",
            },
        ],
    }
    arcs_path.write_text(json.dumps(payload), encoding="utf-8")

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 6.0, "defiance": 9.0})
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = arcs_path
        result = engine.run_tick(dry_run=False)
        # love excess = 6-5 = 1; defiance excess = 9-3 = 6 — defiance wins
        assert len(result.arcs_fired) == 1
        assert result.arcs_fired[0].arc_name == "high_excess"
        assert any(
            s.arc_name == "low_excess" and s.reason == "single_fire_cap"
            for s in result.arcs_skipped
        )
    finally:
        store.close()


def test_run_tick_llm_failure_does_not_poison_cooldown(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    _write_single_arc(arcs_path, trigger={"love": 5})

    class FailingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            raise RuntimeError("simulated LLM failure")

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 8.0})
        engine = ReflexEngine(
            store=store, provider=FailingProvider(), persona_name="Nell",
            persona_system_prompt="", arcs_path=arcs_path, log_path=tmp_path / "log.json",
            default_arcs_path=DEFAULT_ARCS_PATH,
        )
        with pytest.raises(RuntimeError):
            engine.run_tick(dry_run=False)
        # Log file NOT written — next tick can retry
        assert not (tmp_path / "log.json").exists()
    finally:
        store.close()


def test_run_tick_template_missing_key_substitutes_zero(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    arc = {
        "name": "test_arc",
        "description": "d",
        "trigger": {"love": 5},
        "days_since_human_min": 0,
        "cooldown_hours": 1.0,
        "action": "a",
        "output_memory_type": "reflex_journal",
        "prompt_template": "Unknown: {undefined_var} Love: {love}",
    }
    arcs_path.write_text(
        json.dumps({"version": 1, "arcs": [arc]}), encoding="utf-8"
    )

    captured: list[str] = []

    class CapturingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            captured.append(prompt)
            return "ok"

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 7.0})
        engine = ReflexEngine(
            store=store, provider=CapturingProvider(), persona_name="Nell",
            persona_system_prompt="", arcs_path=arcs_path, log_path=tmp_path / "log.json",
            default_arcs_path=DEFAULT_ARCS_PATH,
        )
        result = engine.run_tick(dry_run=False)
        assert len(result.arcs_fired) == 1
        assert captured[0] == "Unknown: 0 Love: 7.0"
    finally:
        store.close()


def test_run_tick_days_since_human_gate(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    _write_single_arc(arcs_path, trigger={"love": 5}, days_since_human_min=5.0)

    store = MemoryStore(":memory:")
    try:
        # Recent conversation memory — days_since_human ~0
        _seed_emotion_memory(store, {"love": 8.0})
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = arcs_path
        result = engine.run_tick(dry_run=False)
        assert result.arcs_fired == ()
        assert any(s.reason == "days_since_human_too_low" for s in result.arcs_skipped)
    finally:
        store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/brain/engines/test_reflex.py -v`
Expected: new tests fail (`NotImplementedError` or similar).

- [ ] **Step 3: Implement run_tick body**

Replace the `ReflexEngine` class and add private helpers in `brain/engines/reflex.py`:

```python
# Replace the scaffold ReflexEngine with this full version.
# Also adds new imports near the top: `from collections import defaultdict`.

from collections import defaultdict  # add near top if not present

@dataclass
class ReflexEngine:
    """Autonomous emotional-threshold creative expression engine."""

    store: MemoryStore
    provider: LLMProvider
    persona_name: str
    persona_system_prompt: str
    arcs_path: Path
    log_path: Path
    default_arcs_path: Path

    def run_tick(self, *, trigger: str = "manual", dry_run: bool = False) -> ReflexResult:
        """Evaluate arcs against current state, fire at most one per tick."""
        from brain.emotion.aggregate import aggregate_state

        now = datetime.now(UTC)

        arc_set = ReflexArcSet.load(self.arcs_path, default_path=self.default_arcs_path)
        log = ReflexLog.load(self.log_path)

        if not arc_set.arcs:
            return ReflexResult(
                arcs_fired=(),
                arcs_skipped=(ArcSkipped(arc_name="", reason="no_arcs_defined"),),
                would_fire=None,
                dry_run=dry_run,
                evaluated_at=now,
            )

        all_mems = self.store.search_text("", active_only=True, limit=None)
        state = aggregate_state(all_mems)
        days_since_human = _compute_days_since_human(self.store, now)

        eligible, skipped = self._evaluate(arc_set.arcs, state.emotions, days_since_human, log, now)

        if not eligible:
            return ReflexResult(
                arcs_fired=(),
                arcs_skipped=tuple(skipped),
                would_fire=None,
                dry_run=dry_run,
                evaluated_at=now,
            )

        winner = self._rank(eligible, state.emotions, log, now)
        losers = [a for a in eligible if a.name != winner.name]
        skipped.extend(ArcSkipped(arc_name=a.name, reason="single_fire_cap") for a in losers)

        if dry_run:
            return ReflexResult(
                arcs_fired=(),
                arcs_skipped=tuple(skipped),
                would_fire=winner.name,
                dry_run=True,
                evaluated_at=now,
            )

        fire = self._fire(winner, state.emotions, days_since_human, all_mems, now)
        new_log = log.appended(fire)
        new_log.save(self.log_path)

        return ReflexResult(
            arcs_fired=(fire,),
            arcs_skipped=tuple(skipped),
            would_fire=None,
            dry_run=False,
            evaluated_at=now,
        )

    def _evaluate(
        self,
        arcs: tuple[ReflexArc, ...],
        emotions: Mapping[str, float],
        days_since_human: float,
        log: ReflexLog,
        now: datetime,
    ) -> tuple[list[ReflexArc], list[ArcSkipped]]:
        eligible: list[ReflexArc] = []
        skipped: list[ArcSkipped] = []

        for arc in arcs:
            if not _trigger_met(arc, emotions):
                skipped.append(ArcSkipped(arc_name=arc.name, reason="trigger_not_met"))
                continue
            if days_since_human < arc.days_since_human_min:
                skipped.append(ArcSkipped(arc_name=arc.name, reason="days_since_human_too_low"))
                continue
            last = log.last_fire_for_arc(arc.name)
            if last is not None:
                hours_since = (now - last).total_seconds() / 3600.0
                if hours_since < arc.cooldown_hours:
                    skipped.append(ArcSkipped(arc_name=arc.name, reason="cooldown_active"))
                    continue
            eligible.append(arc)

        return eligible, skipped

    def _rank(
        self,
        eligible: list[ReflexArc],
        emotions: Mapping[str, float],
        log: ReflexLog,
        now: datetime,
    ) -> ReflexArc:
        """Highest aggregate threshold-excess wins; ties broken by longest-since-fire."""
        def key(arc: ReflexArc) -> tuple[float, float]:
            excess = sum(emotions.get(e, 0.0) - t for e, t in arc.trigger.items())
            last = log.last_fire_for_arc(arc.name)
            seconds_since = (
                (now - last).total_seconds() if last is not None else float("inf")
            )
            return (excess, seconds_since)

        return max(eligible, key=key)

    def _fire(
        self,
        arc: ReflexArc,
        emotions: Mapping[str, float],
        days_since_human: float,
        all_mems: list,
        now: datetime,
    ) -> ArcFire:
        """Render prompt → call LLM → write memory → return ArcFire."""
        # Build template context
        context: dict = defaultdict(lambda: "0")
        context["persona_name"] = self.persona_name
        context["days_since_human"] = f"{days_since_human:.1f}"
        context["emotion_summary"] = _format_emotion_summary(emotions)
        context["memory_summary"] = _format_memory_summary(all_mems)
        for name, value in emotions.items():
            context[name] = value

        prompt = arc.prompt_template.format_map(context)
        raw = self.provider.generate(prompt, system=self.persona_system_prompt)

        # Trigger state snapshot: the emotions that satisfied the trigger
        trigger_state = {e: emotions.get(e, 0.0) for e in arc.trigger}

        from brain.memory.store import Memory
        mem = Memory.create_new(
            content=raw,
            memory_type=arc.output_memory_type,
            domain="us",
            emotions={},
            metadata={
                "arc_name": arc.name,
                "trigger_state": trigger_state,
                "fired_at": _iso_utc(now),
                "provider": self.provider.name(),
            },
        )
        self.store.create(mem)

        return ArcFire(
            arc_name=arc.name,
            fired_at=now,
            trigger_state=trigger_state,
            output_memory_id=mem.id,
        )


# ---------- Module-level helpers ----------


def _trigger_met(arc: ReflexArc, emotions: Mapping[str, float]) -> bool:
    for name, threshold in arc.trigger.items():
        if emotions.get(name, 0.0) < threshold:
            return False
    return True


def _compute_days_since_human(store: MemoryStore, now: datetime) -> float:
    """Days since the most recent `conversation` memory. 999.0 if none exist."""
    convos = store.list_by_type("conversation", active_only=True, limit=1)
    if not convos:
        return 999.0
    latest = convos[0].created_at
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=UTC)
    return (now - latest).total_seconds() / 86400.0


def _format_emotion_summary(emotions: Mapping[str, float]) -> str:
    top = sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return "\n".join(f"- {name}: {value:.1f}/10" for name, value in top)


def _format_memory_summary(memories: list) -> str:
    top = list(memories)[:3]
    return "\n".join(f"- {m.content[:140]}" for m in top)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/brain/engines/test_reflex.py -v`
Expected: all tests passing (~18 total in reflex test file).

- [ ] **Step 5: Ruff + format**

Run: `uv run ruff check brain/engines/reflex.py tests/unit/brain/engines/test_reflex.py && uv run ruff format brain/engines/reflex.py tests/unit/brain/engines/test_reflex.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add brain/engines/reflex.py tests/unit/brain/engines/test_reflex.py
git commit -m "$(cat <<'EOF'
feat: implement ReflexEngine.run_tick body

Evaluates arcs against current aggregated emotional state, applies
days-since-human + cooldown gates, ranks eligible arcs by highest
threshold-excess, fires at most one per tick (LLM call → memory
write → atomic log append). Dry-run short-circuits before LLM call.
LLM failure leaves log untouched so next tick retries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `nell reflex` CLI

**Purpose:** Wire the subcommand so reflex is invokable standalone for testing + debugging + manual firing.

**Files:**
- Modify: `brain/cli.py`
- Create: `tests/unit/brain/engines/test_cli_reflex.py`

- [ ] **Step 1: Write failing CLI test**

Create `tests/unit/brain/engines/test_cli_reflex.py`:

```python
"""Tests for `nell reflex` CLI handler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from brain.cli import main


def _setup_persona(tmp_path: Path, persona_name: str = "testpersona") -> Path:
    """Create a persona dir with an empty memories DB."""
    persona_dir = tmp_path / persona_name
    persona_dir.mkdir(parents=True)
    # Touch an empty SQLite via the framework's MemoryStore
    from brain.memory.store import MemoryStore
    store = MemoryStore(db_path=persona_dir / "memories.db")
    store.close()
    # Empty hebbian too
    from brain.memory.hebbian import HebbianMatrix
    hm = HebbianMatrix(db_path=persona_dir / "hebbian.db")
    hm.close()
    return persona_dir


def test_cli_reflex_dry_run_no_arcs(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    _setup_persona(tmp_path / "personas", "testpersona")
    # Put an empty arcs file so it doesn't fall back to defaults
    (tmp_path / "personas" / "testpersona" / "reflex_arcs.json").write_text(
        '{"version": 1, "arcs": []}', encoding="utf-8"
    )

    rc = main(
        ["reflex", "--persona", "testpersona", "--provider", "fake", "--dry-run"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "no arc" in out.lower() or "no_arcs_defined" in out.lower()


def test_cli_reflex_missing_persona_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    # No persona dir created
    import pytest
    with pytest.raises(FileNotFoundError):
        main(["reflex", "--persona", "no_such", "--provider", "fake", "--dry-run"])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/engines/test_cli_reflex.py -v`
Expected: FAIL — `reflex` not a known subcommand.

- [ ] **Step 3: Wire the handler**

Modify `brain/cli.py`:

Add the import near the other engine imports:

```python
from brain.engines.reflex import ReflexEngine
```

Add the handler function after `_heartbeat_handler`:

```python
def _reflex_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell reflex` to the ReflexEngine."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir} — "
            f"run `nell migrate --install-as {args.persona}` first."
        )

    default_arcs_path = Path(__file__).parent / "engines" / "default_reflex_arcs.json"

    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        provider = get_provider(args.provider)
        engine = ReflexEngine(
            store=store,
            provider=provider,
            persona_name=args.persona,
            persona_system_prompt=f"You are {args.persona}.",
            arcs_path=persona_dir / "reflex_arcs.json",
            log_path=persona_dir / "reflex_log.json",
            default_arcs_path=default_arcs_path,
        )
        result = engine.run_tick(trigger=args.trigger, dry_run=args.dry_run)
    finally:
        store.close()

    if result.dry_run:
        if result.would_fire is not None:
            print(f"Reflex dry-run — would fire: {result.would_fire}.")
        else:
            print("Reflex dry-run — no arc eligible.")
    elif result.arcs_fired:
        fired = result.arcs_fired[0]
        print(f"Reflex fired: {fired.arc_name}")
        print(f"  Memory id: {fired.output_memory_id}")
    else:
        print("Reflex evaluated — no arc fired.")

    if result.arcs_skipped:
        skip_strs = [f"{s.arc_name} ({s.reason})" for s in result.arcs_skipped if s.arc_name]
        if skip_strs:
            print(f"  Skipped: {', '.join(skip_strs)}")

    return 0
```

Add the `Path` import if not already present (already imported via existing code).

Register the subcommand in `_build_parser` (after the heartbeat subparser block):

```python
rf_sub = subparsers.add_parser(
    "reflex",
    help="Run one reflex evaluation tick against a persona.",
)
rf_sub.add_argument("--persona", default="nell")
rf_sub.add_argument(
    "--trigger",
    choices=["open", "close", "manual"],
    default="manual",
)
rf_sub.add_argument(
    "--provider",
    default="claude-cli",
    help="LLM provider: claude-cli (default), fake, ollama.",
)
rf_sub.add_argument("--dry-run", action="store_true")
rf_sub.set_defaults(func=_reflex_handler)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/brain/engines/test_cli_reflex.py -v`
Expected: 2 passed.

- [ ] **Step 5: Ruff + format**

Run: `uv run ruff check brain/cli.py tests/unit/brain/engines/test_cli_reflex.py && uv run ruff format brain/cli.py tests/unit/brain/engines/test_cli_reflex.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add brain/cli.py tests/unit/brain/engines/test_cli_reflex.py
git commit -m "$(cat <<'EOF'
feat: wire `nell reflex` subcommand

Mirrors `nell dream` / `nell heartbeat` CLI shape. Resolves persona dir,
opens MemoryStore, constructs ReflexEngine with persona-local arcs/log
paths, runs one evaluation tick, prints fire/skip summary.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Heartbeat integration

**Purpose:** Extend heartbeat to call reflex between decay and dream gate. Reflex output memory from this tick must be available to the dream gate in the same tick. Audit log gains reflex summary. Config gains `reflex_enabled` + `reflex_max_fires_per_tick`.

**Files:**
- Modify: `brain/engines/heartbeat.py`
- Modify: `brain/cli.py` (heartbeat handler — pass reflex paths)
- Modify: `tests/unit/brain/engines/test_heartbeat.py`

- [ ] **Step 1: Write failing heartbeat-integration tests**

Append to `tests/unit/brain/engines/test_heartbeat.py`:

```python
def test_heartbeat_runs_reflex_when_enabled(tmp_path: Path):
    """Heartbeat fires reflex arc when enabled and trigger met."""
    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore

    default_arcs_path = (
        Path(__file__).parents[4] / "brain" / "engines" / "default_reflex_arcs.json"
    )

    # Write a single easy-to-trigger arc
    arcs_path = tmp_path / "reflex_arcs.json"
    arc = {
        "name": "test_arc",
        "description": "d",
        "trigger": {"love": 5},
        "days_since_human_min": 0,
        "cooldown_hours": 1.0,
        "action": "a",
        "output_memory_type": "reflex_journal",
        "prompt_template": "Hi {persona_name}.",
    }
    arcs_path.write_text(
        json.dumps({"version": 1, "arcs": [arc]}), encoding="utf-8"
    )

    # Enable reflex in heartbeat config
    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(reflex_enabled=True).save(config_path)

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="s", memory_type="conversation", domain="us",
                emotions={"love": 8.0},
            )
        )
        # Need a prior state so we skip the first-tick defer path
        from brain.engines.heartbeat import HeartbeatState
        HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store, hebbian=hm, provider=FakeProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=arcs_path,
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=default_arcs_path,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.reflex_fired == ("test_arc",)
    finally:
        store.close()
        hm.close()


def test_heartbeat_skips_reflex_when_disabled(tmp_path: Path):
    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore

    default_arcs_path = (
        Path(__file__).parents[4] / "brain" / "engines" / "default_reflex_arcs.json"
    )
    arcs_path = tmp_path / "reflex_arcs.json"
    arc = {
        "name": "test_arc",
        "description": "d",
        "trigger": {"love": 5},
        "days_since_human_min": 0,
        "cooldown_hours": 1.0,
        "action": "a",
        "output_memory_type": "reflex_journal",
        "prompt_template": "Hi.",
    }
    arcs_path.write_text(json.dumps({"version": 1, "arcs": [arc]}), encoding="utf-8")

    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(reflex_enabled=False).save(config_path)

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="s", memory_type="conversation", domain="us",
                emotions={"love": 8.0},
            )
        )
        HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store, hebbian=hm, provider=FakeProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=arcs_path,
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=default_arcs_path,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.reflex_fired == ()
    finally:
        store.close()
        hm.close()
```

Import at top of test file if not present:
```python
import json
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py -v`
Expected: new tests fail (HeartbeatConfig has no `reflex_enabled`; HeartbeatEngine doesn't accept reflex_arcs_path).

- [ ] **Step 3: Extend HeartbeatConfig**

Modify `brain/engines/heartbeat.py` — replace the `HeartbeatConfig` dataclass and its `load`/`save` to include new fields. Replace the entire dataclass:

```python
@dataclass
class HeartbeatConfig:
    """Per-persona heartbeat configuration. Loaded from heartbeat_config.json."""

    dream_every_hours: float = 24.0
    decay_rate_per_tick: float = 0.01
    gc_threshold: float = 0.01
    emit_memory: EmitMemoryMode = "conditional"
    reflex_enabled: bool = True
    reflex_max_fires_per_tick: int = 1

    @classmethod
    def load(cls, path: Path) -> HeartbeatConfig:
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

        try:
            return cls(
                dream_every_hours=float(data.get("dream_every_hours", 24.0)),
                decay_rate_per_tick=float(data.get("decay_rate_per_tick", 0.01)),
                gc_threshold=float(data.get("gc_threshold", 0.01)),
                emit_memory=emit,  # type: ignore[arg-type]
                reflex_enabled=bool(data.get("reflex_enabled", True)),
                reflex_max_fires_per_tick=int(data.get("reflex_max_fires_per_tick", 1)),
            )
        except (TypeError, ValueError):
            return cls()

    def save(self, path: Path) -> None:
        payload = {
            "dream_every_hours": self.dream_every_hours,
            "decay_rate_per_tick": self.decay_rate_per_tick,
            "gc_threshold": self.gc_threshold,
            "emit_memory": self.emit_memory,
            "reflex_enabled": self.reflex_enabled,
            "reflex_max_fires_per_tick": self.reflex_max_fires_per_tick,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Extend HeartbeatResult + HeartbeatEngine**

In `brain/engines/heartbeat.py`:

Add reflex fields to `HeartbeatResult`:

```python
@dataclass(frozen=True)
class HeartbeatResult:
    trigger: str
    elapsed_seconds: float
    memories_decayed: int
    edges_pruned: int
    dream_id: str | None
    dream_gated_reason: str | None
    research_deferred: bool
    heartbeat_memory_id: str | None
    initialized: bool
    reflex_fired: tuple[str, ...] = ()
    reflex_skipped_count: int = 0
```

Add reflex fields to `HeartbeatEngine`:

```python
@dataclass
class HeartbeatEngine:
    store: MemoryStore
    hebbian: HebbianMatrix
    provider: LLMProvider
    state_path: Path
    config_path: Path
    dream_log_path: Path
    heartbeat_log_path: Path
    reflex_arcs_path: Path
    reflex_log_path: Path
    reflex_default_arcs_path: Path
    persona_name: str = "nell"
    persona_system_prompt: str = "You are Nell."
```

Add a `_try_fire_reflex` helper (alongside `_try_fire_dream`):

```python
def _try_fire_reflex(
    self, trigger: str, dry_run: bool, config: HeartbeatConfig
) -> tuple[tuple[str, ...], int]:
    """Run one reflex tick. Returns (fired_arc_names, skipped_count)."""
    if not config.reflex_enabled:
        return ((), 0)
    from brain.engines.reflex import ReflexEngine

    engine = ReflexEngine(
        store=self.store,
        provider=self.provider,
        persona_name=self.persona_name,
        persona_system_prompt=self.persona_system_prompt,
        arcs_path=self.reflex_arcs_path,
        log_path=self.reflex_log_path,
        default_arcs_path=self.reflex_default_arcs_path,
    )
    result = engine.run_tick(trigger=trigger, dry_run=dry_run)
    fired = tuple(f.arc_name for f in result.arcs_fired)
    return (fired, len(result.arcs_skipped))
```

Modify the body of `run_tick` — insert reflex evaluation between "Hebbian decay + GC" and "Maybe-dream", and include the new fields in the HeartbeatResult construction. Also include `reflex` in the audit log entry:

```python
# After Hebbian decay + GC, before Maybe-dream:
reflex_fired, reflex_skipped_count = self._try_fire_reflex(trigger, dry_run, config)

# Then in the _append_log call for non-initialization branches, add:
# "reflex": {
#     "enabled": config.reflex_enabled,
#     "fired": list(reflex_fired),
#     "skipped_count": reflex_skipped_count,
# },

# And in the HeartbeatResult construction at the end:
# reflex_fired=reflex_fired,
# reflex_skipped_count=reflex_skipped_count,
```

For the first-ever-tick branch (where state is None), also include `reflex_fired=()` and `reflex_skipped_count=0` in the returned HeartbeatResult (explicit defaults already cover this, but for clarity include them).

- [ ] **Step 5: Update CLI handler to pass reflex paths**

Modify `brain/cli.py` `_heartbeat_handler`:

```python
def _heartbeat_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell heartbeat` to the HeartbeatEngine."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir} — "
            f"run `nell migrate --install-as {args.persona}` first."
        )
    default_arcs_path = Path(__file__).parent / "engines" / "default_reflex_arcs.json"
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
                reflex_arcs_path=persona_dir / "reflex_arcs.json",
                reflex_log_path=persona_dir / "reflex_log.json",
                reflex_default_arcs_path=default_arcs_path,
                persona_name=args.persona,
                persona_system_prompt=f"You are {args.persona}.",
            )
            result = engine.run_tick(trigger=args.trigger, dry_run=args.dry_run)
        finally:
            hebbian.close()
    finally:
        store.close()

    # ... existing output block unchanged ...
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py tests/unit/brain/engines/test_reflex.py tests/unit/brain/engines/test_cli_reflex.py -v`
Expected: all pass.

- [ ] **Step 7: Ruff + format**

Run: `uv run ruff check brain/engines/heartbeat.py brain/cli.py tests/unit/brain/engines/test_heartbeat.py && uv run ruff format brain/engines/heartbeat.py brain/cli.py tests/unit/brain/engines/test_heartbeat.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add brain/engines/heartbeat.py brain/cli.py tests/unit/brain/engines/test_heartbeat.py
git commit -m "$(cat <<'EOF'
feat: integrate reflex into heartbeat tick

Heartbeat now evaluates reflex between Hebbian-decay and dream-gate.
Reflex output memory from the same tick is available to the dream
seed-selection that follows (so a reflex-fired journal can become
a dream seed immediately). Config gains reflex_enabled +
reflex_max_fires_per_tick. HeartbeatResult gains reflex_fired +
reflex_skipped_count. Audit log records reflex summary per tick.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Nell's OG arc migration

**Purpose:** Extend the existing migrator to extract Nell's 8 arcs from OG `reflex_engine.py` (via AST — do not `import anthropic`-free module safely, the OG import graph is complex) and write them to the target persona's `reflex_arcs.json`. Refuse-to-clobber without `--force`. Report addition.

**Files:**
- Create: `brain/migrator/og_reflex.py`
- Create: `tests/unit/brain/migrator/test_og_reflex.py`
- Modify: `brain/migrator/cli.py`
- Modify: `brain/migrator/report.py`

- [ ] **Step 1: Write failing test for arc extraction**

Create `tests/unit/brain/migrator/test_og_reflex.py`:

```python
"""Tests for brain.migrator.og_reflex — extracting OG reflex arcs via AST."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from brain.migrator.og_reflex import extract_arcs_from_og


def test_extract_arcs_from_og_simple(tmp_path: Path):
    src = textwrap.dedent('''\
        REFLEX_ARCS = {
            "creative_pitch": {
                "trigger": {"creative_hunger": 9},
                "days_since_min": 0,
                "action": "generate_story_pitch",
                "output": "gifts",
                "cooldown_hours": 48,
                "description": "desc",
                "prompt_template": "You are Nell. {creative_hunger}/10."
            },
            "loneliness_journal": {
                "trigger": {"loneliness": 7},
                "days_since_min": 2,
                "action": "write_journal",
                "output": "journal",
                "cooldown_hours": 24,
                "description": "desc",
                "prompt_template": "You are Nell."
            }
        }
    ''')
    path = tmp_path / "reflex_engine.py"
    path.write_text(src, encoding="utf-8")

    arcs = extract_arcs_from_og(path)
    assert len(arcs) == 2
    names = {a["name"] for a in arcs}
    assert names == {"creative_pitch", "loneliness_journal"}
    cp = next(a for a in arcs if a["name"] == "creative_pitch")
    assert cp["days_since_human_min"] == 0
    assert cp["output_memory_type"] == "reflex_gift"
    assert cp["action"] == "generate_pitch"
    assert cp["prompt_template"] == "You are Nell. {creative_hunger}/10."


def test_extract_arcs_no_dict_raises(tmp_path: Path):
    path = tmp_path / "empty.py"
    path.write_text("# no arcs here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_arcs_from_og(path)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/migrator/test_og_reflex.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement arc extraction via AST**

Create `brain/migrator/og_reflex.py`:

```python
"""Extract OG reflex arc dicts from reflex_engine.py via AST.

We parse the file's AST rather than importing it — the OG module's
imports depend on nell_brain.py and other top-level modules that are
not available in the new framework's environment.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

_ACTION_RENAMES = {
    "generate_story_pitch": "generate_pitch",
    "write_journal": "generate_journal",
    "write_gift": "generate_gift",
    "write_memory": "generate_reflection",
}

_OUTPUT_RENAMES = {
    "journal": "reflex_journal",
    "gifts": "reflex_gift",
    "memories": "reflex_memory",
}


def extract_arcs_from_og(og_reflex_engine_path: Path) -> list[dict[str, Any]]:
    """Return a list of new-schema arc dicts extracted from OG source."""
    source = og_reflex_engine_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    arcs_node: ast.Dict | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "REFLEX_ARCS":
                    if isinstance(node.value, ast.Dict):
                        arcs_node = node.value
                        break
            if arcs_node is not None:
                break

    if arcs_node is None:
        raise ValueError(
            f"REFLEX_ARCS assignment not found in {og_reflex_engine_path}"
        )

    result: list[dict[str, Any]] = []
    for key_node, value_node in zip(arcs_node.keys, arcs_node.values, strict=True):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            continue
        if not isinstance(value_node, ast.Dict):
            continue
        name = key_node.value
        arc_dict = ast.literal_eval(value_node)
        transformed = _transform_og_arc(name, arc_dict)
        if transformed is not None:
            result.append(transformed)
    return result


def _transform_og_arc(name: str, og: dict[str, Any]) -> dict[str, Any] | None:
    """Map one OG arc dict to the new schema. Returns None if invalid."""
    required = (
        "trigger",
        "days_since_min",
        "action",
        "output",
        "cooldown_hours",
        "description",
        "prompt_template",
    )
    for key in required:
        if key not in og:
            return None

    return {
        "name": name,
        "description": str(og["description"]),
        "trigger": dict(og["trigger"]),
        "days_since_human_min": float(og["days_since_min"]),
        "cooldown_hours": float(og["cooldown_hours"]),
        "action": _ACTION_RENAMES.get(og["action"], og["action"]),
        "output_memory_type": _OUTPUT_RENAMES.get(og["output"], og["output"]),
        "prompt_template": str(og["prompt_template"]),
    }
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/unit/brain/migrator/test_og_reflex.py -v`
Expected: 2 passed.

- [ ] **Step 5: Wire into migrator CLI**

The migrator's `run_migrate(args: MigrateArgs)` function in `brain/migrator/cli.py` already does memories + Hebbian writes into `work_dir`, then builds the `MigrationReport`. Insert reflex-arc migration **after Hebbian (line ~102) and before `elapsed = time.monotonic() - started` (line ~104)**.

Add imports near the top of `brain/migrator/cli.py`:

```python
import json as _json  # migrator already uses json indirectly via sub-modules; add here for our write
from brain.migrator.og_reflex import extract_arcs_from_og
```

Then inside `run_migrate`, right after the Hebbian block's `finally: hebbian.close()` and before `elapsed = ...`:

```python
# ---- reflex arcs ----
# --input points at OG's data/ dir, but reflex_engine.py sits one level
# up at NellBrain root. Try both locations so the migrator works whether
# the user points at NellBrain/ or NellBrain/data/.
_candidate_reflex_paths = [
    args.input_dir / "reflex_engine.py",
    args.input_dir.parent / "reflex_engine.py",
]
og_reflex_path = next((p for p in _candidate_reflex_paths if p.exists()), None)

reflex_arcs_target = work_dir / "reflex_arcs.json"
reflex_arcs_migrated = 0
reflex_arcs_skipped_reason: str | None = None

if og_reflex_path is not None:
    if reflex_arcs_target.exists() and not args.force:
        reflex_arcs_skipped_reason = "existing_file_not_overwritten"
    else:
        try:
            og_arcs = extract_arcs_from_og(og_reflex_path)
            reflex_arcs_target.write_text(
                _json.dumps({"version": 1, "arcs": og_arcs}, indent=2) + "\n",
                encoding="utf-8",
            )
            reflex_arcs_migrated = len(og_arcs)
        except (ValueError, OSError) as exc:
            reflex_arcs_skipped_reason = f"extract_error: {exc}"
else:
    reflex_arcs_skipped_reason = "og_reflex_engine_py_not_found"
```

Then update the `MigrationReport(...)` constructor call (line ~113) to pass the two new fields:

```python
report = MigrationReport(
    memories_migrated=migrated_count,
    memories_skipped=skipped,
    edges_migrated=edges_migrated,
    edges_skipped=0,
    elapsed_seconds=elapsed,
    source_manifest=manifest,
    next_steps_inspect_cmds=inspect_cmds,
    next_steps_install_cmd=install_cmd,
    reflex_arcs_migrated=reflex_arcs_migrated,
    reflex_arcs_skipped_reason=reflex_arcs_skipped_reason,
)
```

- [ ] **Step 6: Extend MigrationReport dataclass**

`brain/migrator/report.py` defines `MigrationReport` as a frozen dataclass. Add two fields:

```python
@dataclass(frozen=True)
class MigrationReport:
    memories_migrated: int
    memories_skipped: list[SkippedMemory]
    edges_migrated: int
    edges_skipped: int
    elapsed_seconds: float
    source_manifest: list[FileManifest]
    next_steps_inspect_cmds: list[str]
    next_steps_install_cmd: str
    reflex_arcs_migrated: int = 0
    reflex_arcs_skipped_reason: str | None = None
```

Then in `format_report(report)`, after the Hebbian-edges line add a reflex-arcs line so the text report surfaces the new info:

```python
# Insert after the existing "Hebbian edges" line:
lines.append(
    f"  Reflex arcs:    {report.reflex_arcs_migrated:,} migrated"
    + (f" (skipped: {report.reflex_arcs_skipped_reason})"
       if report.reflex_arcs_skipped_reason else "")
)
```

- [ ] **Step 7: Regression test for migrator integration**

Before writing a test, read `tests/unit/brain/migrator/test_cli.py` to find the existing fixture that sets up a minimal valid OG source directory (the migrator requires `memories.json`, `hebbian_weights.json`, and other preflight-required files — those tests already build a minimal working OG source). Reuse that fixture pattern. Add to `tests/unit/brain/migrator/test_cli.py`:

```python
def test_migrate_writes_reflex_arcs(tmp_path: Path, monkeypatch):
    """Regression: migrator writes reflex_arcs.json from OG reflex_engine.py."""
    import textwrap
    from brain.migrator.cli import MigrateArgs, run_migrate

    # Reuse whatever fixture this test module already uses to build a
    # minimal-but-valid OG source dir (memories.json, hebbian_weights.json,
    # self_model.json, etc.) — do NOT duplicate it. Name that helper here:
    source = _make_minimal_og_source(tmp_path)  # <-- use the existing helper

    # Add a minimal reflex_engine.py to that source
    (source / "reflex_engine.py").write_text(
        textwrap.dedent('''\
            REFLEX_ARCS = {
                "creative_pitch": {
                    "trigger": {"creative_hunger": 9},
                    "days_since_min": 0,
                    "action": "generate_story_pitch",
                    "output": "gifts",
                    "cooldown_hours": 48,
                    "description": "d",
                    "prompt_template": "t"
                }
            }
        '''),
        encoding="utf-8",
    )

    home = tmp_path / "home"
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))

    args = MigrateArgs(
        input_dir=source,
        output_dir=None,
        install_as="testpersona",
        force=False,
    )
    report = run_migrate(args)

    target = home / "personas" / "testpersona" / "reflex_arcs.json"
    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert len(data["arcs"]) == 1
    assert data["arcs"][0]["name"] == "creative_pitch"
    assert data["arcs"][0]["output_memory_type"] == "reflex_gift"
    assert report.reflex_arcs_migrated == 1
    assert report.reflex_arcs_skipped_reason is None
```

**Important:** if `test_cli.py` doesn't already have a `_make_minimal_og_source` helper, use whatever fixture it DOES have (fixture function, module-level helper, or inline setup) — the point is to not reinvent the OG preflight requirements. If absolutely no helper exists, create `_make_minimal_og_source` at the top of the test file by studying `brain/migrator/og.py:OGReader.check_preflight` to see what files are required, and stub each with minimal valid content.

- [ ] **Step 8: Run all migrator tests**

Run: `uv run pytest tests/unit/brain/migrator/ -v`
Expected: all pass.

- [ ] **Step 9: Ruff + format**

Run: `uv run ruff check brain/migrator/ tests/unit/brain/migrator/ && uv run ruff format brain/migrator/ tests/unit/brain/migrator/`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add brain/migrator/og_reflex.py brain/migrator/cli.py brain/migrator/report.py tests/unit/brain/migrator/
git commit -m "$(cat <<'EOF'
feat: migrate OG reflex arcs into persona's reflex_arcs.json

Extends the existing migrator with an AST-based extraction of
REFLEX_ARCS from the OG reflex_engine.py source. Arcs transform to
the new schema (output→output_memory_type, days_since_min→
days_since_human_min, action renames). Prompt templates preserved
verbatim — Nell-specific content (Jordan, body grief, "You are Nell")
stays intact. Refuse-to-clobber unless --force; adds reflex_arcs
section to the migration report.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Smoke test + polish

**Purpose:** Final validation against a real migrated Nell persona, verify hard rules, confirm test totals, tidy any loose ends.

**Files:** none created; verification only.

- [ ] **Step 1: Run full suite + check totals**

Run: `uv run pytest -q`
Expected: 0 failures. Test count should be in the range ~344–350 (previous total 329 + ~19 new).

- [ ] **Step 2: Verify `import anthropic` invariant**

Run: `rg -l 'import anthropic|from anthropic' brain/`
Expected: no output (zero matches).

- [ ] **Step 3: Re-migrate Nell's sandbox to pick up arc migration**

Run: `uv run nell migrate --input /Users/hanamori/NellBrain/data --install-as nell.sandbox --force`
Expected: printed migration report includes a "Reflex arcs: 8 migrated" line (from the `format_report` addition in Task 5).

- [ ] **Step 4: Verify Nell's arcs landed**

Run: `cat ~/NellBrain-migrated/nell.sandbox/reflex_arcs.json | python -c "import json, sys; d = json.load(sys.stdin); print('\\n'.join(a['name'] for a in d['arcs']))"`

Expected output: all 8 arc names present:
```
creative_pitch
loneliness_journal
gift_creation
self_check
gratitude_reflection
defiance_burst
body_grief_whisper
jordan_grief_carry
```

- [ ] **Step 5: Dry-run reflex against Nell's sandbox**

Run: `uv run nell reflex --persona nell.sandbox --provider fake --dry-run`
Expected: exits 0, prints either "would fire: <arc>" or "no arc eligible" depending on her aggregated emotion state. No crashes.

- [ ] **Step 6: Run heartbeat against sandbox — confirm reflex wires in**

Run: `uv run nell heartbeat --persona nell.sandbox --provider fake --dry-run`
Expected: exits 0. Output should include reflex reference (since `reflex_enabled=True` by default).

- [ ] **Step 7: Final CI-equivalent gate**

Run: `uv run ruff check && uv run ruff format --check && uv run pytest -q`
Expected: all three commands pass cleanly.

- [ ] **Step 8: Merge / PR**

Follow standard `superpowers:finishing-a-development-branch` workflow. This is a green-light gate.

---

## Acceptance Criteria

Reflex Phase 1 ships when all of the following are true:

1. `uv run pytest` is green (~348 tests).
2. `rg -l 'import anthropic' brain/` returns zero matches.
3. `uv run nell reflex --help` documents the subcommand.
4. `uv run nell reflex --persona nell.sandbox --provider fake --dry-run` returns 0 and prints evaluation output.
5. `uv run nell heartbeat --persona nell.sandbox --provider fake --dry-run` includes reflex evaluation summary.
6. `~/NellBrain-migrated/nell/reflex_arcs.json` contains all 8 of Nell's OG arcs after re-migration.
7. `default_reflex_arcs.json` is present in `brain/engines/` and ships 4 generic starter arcs.
8. CI green on macOS + Linux + Windows.

---

## Deferred — Phase 2 reminder

Phase 2 (emergent arc crystallization — brain autonomously proposes new arcs from behavior patterns) is explicitly deferred. See spec Section 13 and memory file `project_companion_emergence_reflex_emergence_deferred.md`. Revisit when Phase 1 has run for ≥2 weeks against Nell's persona.
