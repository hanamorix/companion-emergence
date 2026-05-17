# Brain Health Module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `brain/health/` self-healing module per the design spec, then wire it into every load/save path in the framework so the brain auto-detects and auto-heals corruption without user intervention. Identity-critical files can reconstruct from memories; SQLite databases run integrity checks; only true unrecoverable corruption alarms the user.

**Architecture:** New `brain/health/` package with 8 modules. Atomic-rewrite files use `attempt_heal` + `save_with_backup` (3-backup rotation, bumps to 6 on recurring corruption + activates verify-after-write). Append-only logs use `read_jsonl_skipping_corrupt` (per-line skip, generalised from the Phase 2a hardening pattern). SQLite stores run `PRAGMA integrity_check`. Heartbeat tick aggregates anomalies into the audit log; compact CLI surfaces `🩹` for self-treatment, persistent banner for true alarms. New `nell health show / check / acknowledge` CLI for inspection + alarm management.

**Tech Stack:** Python 3.12, dataclasses, JSON file I/O with atomic `os.replace` pattern, SQLite `PRAGMA integrity_check`, pytest. No new deps.

**Spec:** [docs/superpowers/specs/2026-04-25-brain-health-module-design.md](../specs/2026-04-25-brain-health-module-design.md)

**Recommended PR shape:** This plan is sized for two PRs at execution time:
- **PR-1 — Core helpers (T1-T8):** All `brain/health/` internals + SQLite integrity check. No wiring. Self-contained and reviewable independently.
- **PR-2 — Wiring + integration (T9-T14):** All the touch-points into existing modules + heartbeat audit + CLI + smoke. Builds on PR-1's helpers.

Either lands them sequentially (recommended) or merges as one PR if smaller-PR pressure isn't a concern.

---

## File Structure

**Created:**

| File | Responsibility |
|------|---------------|
| `brain/health/__init__.py` | Package marker. |
| `brain/health/anomaly.py` | `BrainAnomaly` + `AlarmEntry` frozen dataclasses + serialize/deserialize. |
| `brain/health/jsonl_reader.py` | `read_jsonl_skipping_corrupt(path)` — generalised from Phase 2a hardening. |
| `brain/health/attempt_heal.py` | `attempt_heal(path, default_factory, schema_validator)` + `save_with_backup(path, data, backup_count)` — core helpers. |
| `brain/health/adaptive.py` | `compute_treatment(persona_dir, file)` reads audit log, returns `FileTreatment(backup_count, verify_after_write)`. |
| `brain/health/reconstruct.py` | `reconstruct_vocabulary_from_memories(store) -> dict` for identity reconstruction. |
| `brain/health/walker.py` | `walk_persona(persona_dir) -> list[BrainAnomaly]`. |
| `brain/health/alarm.py` | `compute_pending_alarms(persona_dir) -> list[AlarmEntry]`. |
| `tests/unit/brain/health/__init__.py` | Test package marker. |
| `tests/unit/brain/health/test_anomaly.py` | 4 tests. |
| `tests/unit/brain/health/test_jsonl_reader.py` | 5 tests. |
| `tests/unit/brain/health/test_attempt_heal.py` | 12 tests (heal flow + cascading + reset + verify-after-write). |
| `tests/unit/brain/health/test_adaptive.py` | 5 tests. |
| `tests/unit/brain/health/test_reconstruct.py` | 4 tests. |
| `tests/unit/brain/health/test_walker.py` | 4 tests. |
| `tests/unit/brain/health/test_alarm.py` | 4 tests. |
| `tests/unit/brain/health/test_cli_health.py` | 6 tests for `nell health show / check / acknowledge`. |
| `tests/unit/brain/memory/test_store_integrity.py` | 2 tests for SQLite integrity check (one for store, one for hebbian). |

**Modified:**

| File | Change |
|------|--------|
| `brain/persona_config.py` | `load` calls `attempt_heal`; `save` calls `save_with_backup`. |
| `brain/user_preferences.py` | Same. |
| `brain/engines/heartbeat.py` | `HeartbeatConfig.load/save`, `HeartbeatState.load/save` use `attempt_heal`/`save_with_backup`. New `_collect_anomalies` mechanism + cross-file walk on multi-anomaly tick. `HeartbeatResult.anomalies: tuple[BrainAnomaly, ...]` + `pending_alarms_count: int`. Audit log payload gains `anomalies` + `pending_alarms_count`. |
| `brain/engines/_interests.py` | `InterestSet.load/save` use the helpers. |
| `brain/engines/reflex.py` | Reflex arc reads/writes use the helpers (find the `reflex_arcs.json` callsites and route them through `attempt_heal`/`save_with_backup`). |
| `brain/emotion/persona_loader.py` | `load_persona_vocabulary` uses `attempt_heal`; vocabulary writes (which today are inside the migrator + the growth scheduler) use `save_with_backup`. |
| `brain/growth/log.py` | `read_growth_log` becomes a thin wrapper around `read_jsonl_skipping_corrupt`. |
| `brain/growth/scheduler.py` | `_append_to_vocabulary` uses `save_with_backup`; `_read_current_vocabulary_names` uses `attempt_heal` with the existing schema check as validator. |
| `brain/memory/store.py` | `MemoryStore.__init__` runs `PRAGMA integrity_check`. Raises a typed `BrainIntegrityError` on failure. |
| `brain/memory/hebbian.py` | Same for `HebbianMatrix`. |
| `brain/cli.py` | New `nell health show / check / acknowledge` subcommands + compact heartbeat banner additions. |

---

## Task Decomposition

14 tasks. Sequential per subagent-driven-development. Tests-first throughout.

---

### Task 1: `BrainAnomaly` + `AlarmEntry` dataclasses

**Files:**
- Create: `brain/health/__init__.py`
- Create: `brain/health/anomaly.py`
- Create: `tests/unit/brain/health/__init__.py`
- Create: `tests/unit/brain/health/test_anomaly.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/brain/health/test_anomaly.py
"""Tests for brain.health.anomaly — BrainAnomaly + AlarmEntry frozen dataclasses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from brain.health.anomaly import AlarmEntry, BrainAnomaly


def test_brain_anomaly_construction() -> None:
    a = BrainAnomaly(
        timestamp=datetime(2026, 4, 25, 18, 30, tzinfo=UTC),
        file="emotion_vocabulary.json",
        kind="json_parse_error",
        action="restored_from_bak1",
        quarantine_path="emotion_vocabulary.json.corrupt-2026-04-25T18:30:00Z",
        likely_cause="user_edit",
        detail="Expecting ',' delimiter: line 12 column 5",
    )
    assert a.file == "emotion_vocabulary.json"
    assert a.kind == "json_parse_error"
    assert a.likely_cause == "user_edit"


def test_brain_anomaly_is_frozen() -> None:
    a = BrainAnomaly(
        timestamp=datetime.now(UTC),
        file="x.json",
        kind="json_parse_error",
        action="reset_to_default",
        quarantine_path=None,
        likely_cause="unknown",
        detail="",
    )
    with pytest.raises(FrozenInstanceError):
        a.file = "mutated"  # type: ignore[misc]


def test_brain_anomaly_to_dict_serialises_iso_utc() -> None:
    a = BrainAnomaly(
        timestamp=datetime(2026, 4, 25, 18, 30, tzinfo=UTC),
        file="x.json",
        kind="schema_mismatch",
        action="reset_to_default",
        quarantine_path=None,
        likely_cause="unknown",
        detail="missing 'emotions' key",
    )
    d = a.to_dict()
    assert d["timestamp"].endswith("Z")
    assert d["file"] == "x.json"
    assert d["quarantine_path"] is None


def test_alarm_entry_is_frozen() -> None:
    e = AlarmEntry(
        file="memories.db",
        kind="sqlite_integrity_fail",
        first_seen_at=datetime.now(UTC),
        occurrences_in_window=1,
    )
    with pytest.raises(FrozenInstanceError):
        e.file = "x"  # type: ignore[misc]
```

- [ ] **Step 2: Run tests — verify FAIL** with `ModuleNotFoundError: No module named 'brain.health'`.

- [ ] **Step 3: Implementation**

```python
# brain/health/__init__.py
"""brain.health — self-healing architecture.

Per docs/superpowers/specs/2026-04-25-brain-health-module-design.md.
"""
```

```python
# brain/health/anomaly.py
"""BrainAnomaly + AlarmEntry — structured records of detection events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from brain.utils.time import iso_utc, parse_iso_utc

AnomalyKind = Literal["json_parse_error", "schema_mismatch", "sqlite_integrity_fail"]
AnomalyAction = Literal[
    "restored_from_bak1",
    "restored_from_bak2",
    "restored_from_bak3",
    "reset_to_default",
    "reconstructed_from_memories",
    "alarmed_unrecoverable",
    "verify_after_write_failed",
]
LikelyCause = Literal["user_edit", "disk", "unknown"]


@dataclass(frozen=True)
class BrainAnomaly:
    timestamp: datetime
    file: str
    kind: AnomalyKind
    action: AnomalyAction
    quarantine_path: str | None
    likely_cause: LikelyCause
    detail: str

    def to_dict(self) -> dict:
        return {
            "timestamp": iso_utc(self.timestamp),
            "file": self.file,
            "kind": self.kind,
            "action": self.action,
            "quarantine_path": self.quarantine_path,
            "likely_cause": self.likely_cause,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BrainAnomaly:
        return cls(
            timestamp=parse_iso_utc(data["timestamp"]),
            file=str(data["file"]),
            kind=data["kind"],
            action=data["action"],
            quarantine_path=data.get("quarantine_path"),
            likely_cause=data.get("likely_cause", "unknown"),
            detail=str(data.get("detail", "")),
        )


@dataclass(frozen=True)
class AlarmEntry:
    file: str
    kind: str
    first_seen_at: datetime
    occurrences_in_window: int
```

- [ ] **Step 4: Tests pass.** Run: `uv run pytest tests/unit/brain/health/test_anomaly.py -v`

- [ ] **Step 5: Commit**

```bash
git add brain/health/__init__.py brain/health/anomaly.py tests/unit/brain/health/
git commit -m "feat(health): add BrainAnomaly + AlarmEntry dataclasses — Health T1"
```

---

### Task 2: `read_jsonl_skipping_corrupt` helper

**Files:**
- Create: `brain/health/jsonl_reader.py`
- Create: `tests/unit/brain/health/test_jsonl_reader.py`

Generalises the Phase 2a hardening pattern: per-line skip with line-index + 200-char content preview in the warning. Returns a list (not iterator) for simplicity in callers.

- [ ] **Step 1: Failing tests**

```python
# tests/unit/brain/health/test_jsonl_reader.py
"""Tests for brain.health.jsonl_reader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from brain.health.jsonl_reader import read_jsonl_skipping_corrupt


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_jsonl_skipping_corrupt(tmp_path / "missing.jsonl") == []


def test_well_formed_lines_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_text(
        json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n", encoding="utf-8"
    )
    out = read_jsonl_skipping_corrupt(p)
    assert out == [{"a": 1}, {"a": 2}]


def test_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n\n\n" + json.dumps({"a": 2}) + "\n", encoding="utf-8")
    assert read_jsonl_skipping_corrupt(p) == [{"a": 1}, {"a": 2}]


def test_skips_corrupt_lines_and_warns(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.WARNING)
    p = tmp_path / "log.jsonl"
    p.write_text(
        json.dumps({"good": 1}) + "\n{not valid\n" + json.dumps({"good": 2}) + "\n",
        encoding="utf-8",
    )
    out = read_jsonl_skipping_corrupt(p)
    assert out == [{"good": 1}, {"good": 2}]

    bad = [r for r in caplog.records if "malformed jsonl line" in r.getMessage()]
    assert len(bad) == 1
    msg = bad[0].getMessage()
    assert "line 2" in msg
    assert "{not valid" in msg


def test_warning_includes_path_and_truncates_long_content(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.WARNING)
    p = tmp_path / "log.jsonl"
    long_corrupt = "{" + ("x" * 500)
    p.write_text(long_corrupt + "\n", encoding="utf-8")
    read_jsonl_skipping_corrupt(p)
    msg = next(r.getMessage() for r in caplog.records if "malformed jsonl line" in r.getMessage())
    assert str(p) in msg
    assert "x" * 200 in msg  # 200-char preview
    assert "x" * 500 not in msg  # truncated
```

- [ ] **Step 2: Run — verify FAIL**

- [ ] **Step 3: Implementation**

```python
# brain/health/jsonl_reader.py
"""Append-only JSONL log reader with per-line corruption skip.

Generalises the pattern shipped in the Phase 2a hardening PR for the growth log.
Used by every *.log.jsonl reader in the brain — heartbeats, dreams, reflex,
research, growth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_jsonl_skipping_corrupt(path: Path) -> list[dict]:
    """Return parsed lines from `path`, skipping malformed lines with a WARNING.

    Per-line resilience: a single corrupt line never invalidates the lines
    around it. Each skipped line emits a warning that includes the line
    number, the file path, the parse exception, and a 200-char preview of
    the bad content — enough for a human to find and quarantine the line.
    """
    if not path.exists():
        return []
    out: list[dict] = []
    for line_index, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "skipping malformed jsonl line %d in %s: %.200s | content: %r",
                line_index,
                path,
                exc,
                raw[:200],
            )
            continue
        if isinstance(data, dict):
            out.append(data)
    return out
```

- [ ] **Step 4: Tests pass.**

- [ ] **Step 5: Commit**

```bash
git add brain/health/jsonl_reader.py tests/unit/brain/health/test_jsonl_reader.py
git commit -m "feat(health): add read_jsonl_skipping_corrupt — Health T2"
```

---

### Task 3: `attempt_heal` + `save_with_backup` core helpers

**Files:**
- Create: `brain/health/attempt_heal.py`
- Create: `tests/unit/brain/health/test_attempt_heal.py`

The meatiest task. 12 tests cover the heal flow exhaustively: healthy load, JSONDecodeError → restore from .bak1, schema validator failure → restore from .bak1, .bak1 also corrupt → walks to .bak2, all backups corrupt → reset to default factory output, save rotation correctness, stale .new cleanup, user_edit heuristic, verify-after-write success + failure.

- [ ] **Step 1: Failing tests**

```python
# tests/unit/brain/health/test_attempt_heal.py
"""Tests for brain.health.attempt_heal — core heal + save helpers."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.health.attempt_heal import attempt_heal, save_with_backup


def _default() -> dict:
    return {"version": 1, "value": "default"}


def _vocab_validator(data: object) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("emotions"), list):
        raise ValueError("missing 'emotions' list")


# ---- Healthy paths ----


def test_attempt_heal_missing_file_returns_default(tmp_path: Path) -> None:
    data, anomaly = attempt_heal(tmp_path / "x.json", _default)
    assert data == {"version": 1, "value": "default"}
    assert anomaly is None


def test_attempt_heal_well_formed_returns_data(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"version": 1, "value": "stored"}), encoding="utf-8")
    data, anomaly = attempt_heal(p, _default)
    assert data == {"version": 1, "value": "stored"}
    assert anomaly is None


def test_attempt_heal_validator_passes_well_formed(tmp_path: Path) -> None:
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
    data, anomaly = attempt_heal(p, _default, schema_validator=_vocab_validator)
    assert anomaly is None


# ---- Corrupt paths ----


def test_attempt_heal_corrupt_json_no_baks_resets_to_default(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text("{not json", encoding="utf-8")
    data, anomaly = attempt_heal(p, _default)
    assert data == {"version": 1, "value": "default"}
    assert anomaly is not None
    assert anomaly.action == "reset_to_default"
    assert anomaly.kind == "json_parse_error"
    # Quarantine present
    quarantines = list(tmp_path.glob("x.json.corrupt-*"))
    assert len(quarantines) == 1
    # Live file rewritten with default
    assert json.loads(p.read_text(encoding="utf-8")) == {"version": 1, "value": "default"}


def test_attempt_heal_restores_from_bak1(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    bak1 = tmp_path / "x.json.bak1"
    bak1.write_text(json.dumps({"version": 1, "value": "good"}), encoding="utf-8")
    p.write_text("{not json", encoding="utf-8")
    data, anomaly = attempt_heal(p, _default)
    assert data == {"version": 1, "value": "good"}
    assert anomaly.action == "restored_from_bak1"
    # bak1 is now the live file (was renamed in)
    assert json.loads(p.read_text(encoding="utf-8"))["value"] == "good"
    assert not bak1.exists()  # consumed


def test_attempt_heal_walks_to_bak2_when_bak1_corrupt(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text("{not json", encoding="utf-8")
    (tmp_path / "x.json.bak1").write_text("{also not json", encoding="utf-8")
    (tmp_path / "x.json.bak2").write_text(
        json.dumps({"version": 1, "value": "older_good"}), encoding="utf-8"
    )
    data, anomaly = attempt_heal(p, _default)
    assert data == {"version": 1, "value": "older_good"}
    assert anomaly.action == "restored_from_bak2"


def test_attempt_heal_schema_validator_failure_treated_as_corrupt(tmp_path: Path) -> None:
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps({"version": 1, "wrong_field": []}), encoding="utf-8")  # parses but invalid
    data, anomaly = attempt_heal(p, lambda: {"version": 1, "emotions": []}, schema_validator=_vocab_validator)
    assert anomaly is not None
    assert anomaly.kind == "schema_mismatch"
    assert anomaly.action == "reset_to_default"


def test_attempt_heal_user_edit_heuristic(tmp_path: Path) -> None:
    p = tmp_path / "user_preferences.json"
    p.write_text("{not json", encoding="utf-8")
    # mtime is now (recently edited), small file, content starts with {
    data, anomaly = attempt_heal(p, _default)
    assert anomaly is not None
    assert anomaly.likely_cause == "user_edit"


# ---- Save flow ----


def test_save_with_backup_first_save_no_bak(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    save_with_backup(p, {"a": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}
    assert not (tmp_path / "x.json.bak1").exists()
    assert not (tmp_path / "x.json.new").exists()


def test_save_with_backup_rotates_3_levels(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    save_with_backup(p, {"a": 1})  # live=1
    save_with_backup(p, {"a": 2})  # live=2, bak1=1
    save_with_backup(p, {"a": 3})  # live=3, bak1=2, bak2=1
    save_with_backup(p, {"a": 4})  # live=4, bak1=3, bak2=2, bak3=1
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 4}
    assert json.loads((tmp_path / "x.json.bak1").read_text(encoding="utf-8")) == {"a": 3}
    assert json.loads((tmp_path / "x.json.bak2").read_text(encoding="utf-8")) == {"a": 2}
    assert json.loads((tmp_path / "x.json.bak3").read_text(encoding="utf-8")) == {"a": 1}


def test_save_with_backup_caps_at_3_drops_oldest(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    for i in range(1, 6):
        save_with_backup(p, {"a": i})
    # live=5, bak1=4, bak2=3, bak3=2, oldest dropped
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 5}
    assert json.loads((tmp_path / "x.json.bak3").read_text(encoding="utf-8")) == {"a": 2}
    assert not (tmp_path / "x.json.bak4").exists()


def test_save_with_backup_unlinks_stale_new(tmp_path: Path) -> None:
    """If .new exists from a prior crash, save unlinks it before writing."""
    p = tmp_path / "x.json"
    (tmp_path / "x.json.new").write_text("stale partial content", encoding="utf-8")
    save_with_backup(p, {"a": 1})
    # Stale .new is gone; live file is correct
    assert not (tmp_path / "x.json.new").exists()
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}


def test_save_with_backup_higher_count_keeps_more(tmp_path: Path) -> None:
    """When backup_count=6, six backups are retained."""
    p = tmp_path / "x.json"
    for i in range(1, 8):
        save_with_backup(p, {"a": i}, backup_count=6)
    # live=7; bak1..bak6 = 6,5,4,3,2,1
    for k, expected in zip(range(1, 7), [6, 5, 4, 3, 2, 1], strict=True):
        bak = tmp_path / f"x.json.bak{k}"
        assert json.loads(bak.read_text(encoding="utf-8")) == {"a": expected}
```

- [ ] **Step 2: Run — verify FAIL**

- [ ] **Step 3: Implementation**

```python
# brain/health/attempt_heal.py
"""Core heal + save helpers — atomic .bak rotation + corruption recovery."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.health.anomaly import BrainAnomaly
from brain.utils.time import iso_utc

logger = logging.getLogger(__name__)


def attempt_heal(
    path: Path,
    default_factory: Callable[[], Any],
    schema_validator: Callable[[Any], None] | None = None,
) -> tuple[Any, BrainAnomaly | None]:
    """Load `path`, healing from .bak rotation if corrupt.

    Returns (data, anomaly_or_None).
      - Missing file → (default_factory(), None)
      - Healthy file → (parsed_data, None)
      - Corrupt file → quarantine, walk .bak1/.bak2/.bak3, restore freshest
        valid backup; if all corrupt, write default_factory() output.
        Returns (data, BrainAnomaly).
    """
    if not path.exists():
        return default_factory(), None

    try:
        data = _load_and_validate(path, schema_validator)
        return data, None
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return _heal_from_baks(path, default_factory, schema_validator, exc)


def _load_and_validate(path: Path, validator: Callable[[Any], None] | None) -> Any:
    data = json.loads(path.read_text(encoding="utf-8"))
    if validator is not None:
        validator(data)
    return data


def _heal_from_baks(
    path: Path,
    default_factory: Callable[[], Any],
    schema_validator: Callable[[Any], None] | None,
    original_exc: Exception,
) -> tuple[Any, BrainAnomaly]:
    now = datetime.now(UTC)
    likely_cause = _classify_cause(path)
    quarantine = path.with_name(f"{path.name}.corrupt-{iso_utc(now)}")
    os.replace(path, quarantine)

    kind = "json_parse_error" if isinstance(original_exc, json.JSONDecodeError) else "schema_mismatch"

    for bak_index in (1, 2, 3):
        bak = path.with_name(f"{path.name}.bak{bak_index}")
        if not bak.exists():
            continue
        try:
            data = _load_and_validate(bak, schema_validator)
        except (json.JSONDecodeError, ValueError, TypeError):
            # This bak is also corrupt — quarantine it too.
            bak_quarantine = path.with_name(f"{path.name}.bak{bak_index}.corrupt-{iso_utc(now)}")
            os.replace(bak, bak_quarantine)
            continue

        # Found a valid bak — restore.
        os.replace(bak, path)
        return data, BrainAnomaly(
            timestamp=now,
            file=path.name,
            kind=kind,  # type: ignore[arg-type]
            action=f"restored_from_bak{bak_index}",  # type: ignore[arg-type]
            quarantine_path=quarantine.name,
            likely_cause=likely_cause,
            detail=str(original_exc)[:500],
        )

    # All baks corrupt or missing — reset to default.
    default_data = default_factory()
    path.write_text(json.dumps(default_data, indent=2) + "\n", encoding="utf-8")
    return default_data, BrainAnomaly(
        timestamp=now,
        file=path.name,
        kind=kind,  # type: ignore[arg-type]
        action="reset_to_default",
        quarantine_path=quarantine.name,
        likely_cause=likely_cause,
        detail=str(original_exc)[:500],
    )


def _classify_cause(path: Path) -> str:
    """Heuristic: hand-edit vs disk vs unknown.

    user_edit: mtime within 60s, size < 100KB, content starts with { or [.
    Otherwise unknown. (No specific disk-error heuristic in v1.)
    """
    try:
        st = path.stat()
        if time.time() - st.st_mtime < 60 and st.st_size < 100_000:
            head = path.read_bytes()[:1].strip()
            if head and head[:1] in (b"{", b"["):
                return "user_edit"
    except OSError:
        pass
    return "unknown"


def save_with_backup(path: Path, data: Any, backup_count: int = 3) -> None:
    """Atomic save with .bak rotation.

    Writes <path>.new, rotates existing <path> → <path>.bak1 → ... → <path>.bak{N},
    drops oldest, then atomically replaces <path>. Stale <path>.new from a prior
    crash is unlinked before the new write begins.
    """
    new_path = path.with_name(path.name + ".new")
    if new_path.exists():
        new_path.unlink()  # stale from prior crash; always incomplete

    new_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    # Rotate: drop oldest first, then walk down toward live.
    oldest = path.with_name(f"{path.name}.bak{backup_count}")
    if oldest.exists():
        oldest.unlink()
    for i in range(backup_count - 1, 0, -1):
        src = path.with_name(f"{path.name}.bak{i}")
        dst = path.with_name(f"{path.name}.bak{i + 1}")
        if src.exists():
            os.replace(src, dst)
    if path.exists():
        os.replace(path, path.with_name(f"{path.name}.bak1"))

    os.replace(new_path, path)
```

- [ ] **Step 4: Tests pass.** Run: `uv run pytest tests/unit/brain/health/test_attempt_heal.py -v`

- [ ] **Step 5: Commit**

```bash
git add brain/health/attempt_heal.py tests/unit/brain/health/test_attempt_heal.py
git commit -m "feat(health): attempt_heal + save_with_backup core helpers — Health T3"
```

---

### Task 4: `adaptive.compute_treatment`

**Files:**
- Create: `brain/health/adaptive.py`
- Create: `tests/unit/brain/health/test_adaptive.py`

Reads the persona's `heartbeats.log.jsonl` (using `read_jsonl_skipping_corrupt` from T2), counts anomalies for the given file in the last 7 days, returns `FileTreatment(backup_count, verify_after_write)`. Bump rule: `≥3 corruptions in 7 days` → `(6, True)`. Revert: `30 days clean since most recent anomaly` → `(3, False)`.

- [ ] **Step 1: Failing tests**

```python
# tests/unit/brain/health/test_adaptive.py
"""Tests for brain.health.adaptive — backup-depth + verify-after-write driven by audit log."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.adaptive import FileTreatment, compute_treatment


def _audit_entry(file: str, when: datetime, kind: str = "json_parse_error") -> dict:
    return {
        "timestamp": when.isoformat().replace("+00:00", "Z"),
        "anomalies": [
            {
                "timestamp": when.isoformat().replace("+00:00", "Z"),
                "file": file,
                "kind": kind,
                "action": "restored_from_bak1",
                "quarantine_path": None,
                "likely_cause": "unknown",
                "detail": "",
            }
        ],
    }


def _seed_audit(persona_dir: Path, entries: list[dict]) -> None:
    persona_dir.mkdir(parents=True, exist_ok=True)
    p = persona_dir / "heartbeats.log.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def test_no_audit_log_returns_default(tmp_path: Path) -> None:
    t = compute_treatment(tmp_path, "x.json")
    assert t == FileTreatment(backup_count=3, verify_after_write=False)


def test_one_anomaly_within_window_returns_default(tmp_path: Path) -> None:
    _seed_audit(tmp_path, [_audit_entry("x.json", datetime.now(UTC) - timedelta(days=2))])
    t = compute_treatment(tmp_path, "x.json")
    assert t == FileTreatment(backup_count=3, verify_after_write=False)


def test_three_anomalies_within_window_bumps_treatment(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    _seed_audit(
        tmp_path,
        [
            _audit_entry("x.json", now - timedelta(days=1)),
            _audit_entry("x.json", now - timedelta(days=3)),
            _audit_entry("x.json", now - timedelta(days=5)),
        ],
    )
    t = compute_treatment(tmp_path, "x.json")
    assert t == FileTreatment(backup_count=6, verify_after_write=True)


def test_anomalies_for_other_file_not_counted(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    _seed_audit(
        tmp_path,
        [
            _audit_entry("y.json", now - timedelta(days=1)),
            _audit_entry("y.json", now - timedelta(days=2)),
            _audit_entry("y.json", now - timedelta(days=3)),
        ],
    )
    t = compute_treatment(tmp_path, "x.json")
    assert t == FileTreatment(backup_count=3, verify_after_write=False)


def test_anomalies_outside_window_not_counted(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    _seed_audit(
        tmp_path,
        [
            _audit_entry("x.json", now - timedelta(days=10)),
            _audit_entry("x.json", now - timedelta(days=12)),
            _audit_entry("x.json", now - timedelta(days=15)),
        ],
    )
    t = compute_treatment(tmp_path, "x.json")
    assert t == FileTreatment(backup_count=3, verify_after_write=False)
```

- [ ] **Step 2: Run — verify FAIL**

- [ ] **Step 3: Implementation**

```python
# brain/health/adaptive.py
"""Adaptive treatment — backup depth + verify-after-write computed from audit log."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.utils.time import parse_iso_utc

WINDOW_DAYS = 7
BUMP_THRESHOLD = 3
ELEVATED_BACKUP_COUNT = 6
DEFAULT_BACKUP_COUNT = 3


@dataclass(frozen=True)
class FileTreatment:
    backup_count: int
    verify_after_write: bool


def compute_treatment(persona_dir: Path, file: str) -> FileTreatment:
    """Read audit log; if `file` saw ≥3 anomalies in last 7 days, return elevated treatment."""
    audit_path = persona_dir / "heartbeats.log.jsonl"
    cutoff = datetime.now(UTC) - timedelta(days=WINDOW_DAYS)

    count = 0
    for entry in read_jsonl_skipping_corrupt(audit_path):
        for a in entry.get("anomalies") or []:
            if not isinstance(a, dict):
                continue
            if a.get("file") != file:
                continue
            try:
                ts = parse_iso_utc(a["timestamp"])
            except (KeyError, ValueError, TypeError):
                continue
            if ts >= cutoff:
                count += 1

    if count >= BUMP_THRESHOLD:
        return FileTreatment(backup_count=ELEVATED_BACKUP_COUNT, verify_after_write=True)
    return FileTreatment(backup_count=DEFAULT_BACKUP_COUNT, verify_after_write=False)
```

- [ ] **Step 4: Tests pass.**

- [ ] **Step 5: Commit**

```bash
git add brain/health/adaptive.py tests/unit/brain/health/test_adaptive.py
git commit -m "feat(health): adaptive backup-depth from audit log — Health T4"
```

---

### Task 5: `reconstruct.reconstruct_vocabulary_from_memories`

**Files:**
- Create: `brain/health/reconstruct.py`
- Create: `tests/unit/brain/health/test_reconstruct.py`

Scans `MemoryStore` for distinct emotion names referenced in any memory's `emotions` field. Returns a dict ready to write to `emotion_vocabulary.json`: framework baseline + persona-extension entries reconstructed from memories, with placeholder descriptions and conservative decay (1.0 days).

- [ ] **Step 1: Failing tests**

```python
# tests/unit/brain/health/test_reconstruct.py
"""Tests for brain.health.reconstruct — vocabulary reconstruction from memories."""

from __future__ import annotations

from brain.health.reconstruct import reconstruct_vocabulary_from_memories
from brain.memory.store import Memory, MemoryStore


def test_empty_store_returns_baseline_only() -> None:
    store = MemoryStore(":memory:")
    try:
        result = reconstruct_vocabulary_from_memories(store)
        names = {e["name"] for e in result["emotions"]}
        # 21 framework baseline emotions
        assert "love" in names
        assert "joy" in names
        assert "belonging" in names
        # No persona extensions
        for e in result["emotions"]:
            assert e["category"] == "core" or e["category"] == "complex"
    finally:
        store.close()


def test_reconstructs_persona_extensions_from_memories() -> None:
    store = MemoryStore(":memory:")
    try:
        # Seed memories that reference custom emotions not in baseline.
        store.create(
            Memory.create_new(
                content="x", memory_type="conversation", domain="us",
                emotions={"body_grief": 8.0, "love": 9.0},
            )
        )
        store.create(
            Memory.create_new(
                content="y", memory_type="conversation", domain="us",
                emotions={"creative_hunger": 7.0},
            )
        )

        result = reconstruct_vocabulary_from_memories(store)
        names = {e["name"] for e in result["emotions"]}
        assert "body_grief" in names
        assert "creative_hunger" in names
        # love is baseline, should also be present
        assert "love" in names

        # Extensions have placeholder description + conservative decay
        body_grief = next(e for e in result["emotions"] if e["name"] == "body_grief")
        assert "reconstructed from memory" in body_grief["description"]
        assert body_grief["category"] == "persona_extension"
        assert body_grief["decay_half_life_days"] == 1.0
    finally:
        store.close()


def test_baseline_names_in_memories_not_duplicated() -> None:
    """If a baseline emotion name appears in memories, it doesn't get duplicated as extension."""
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="x", memory_type="conversation", domain="us", emotions={"love": 9.0},
            )
        )
        result = reconstruct_vocabulary_from_memories(store)
        love_entries = [e for e in result["emotions"] if e["name"] == "love"]
        assert len(love_entries) == 1
        assert love_entries[0]["category"] == "core"  # baseline, not extension
    finally:
        store.close()


def test_returned_shape_matches_persona_loader_expectation() -> None:
    """Output is loadable by load_persona_vocabulary."""
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="x", memory_type="conversation", domain="us", emotions={"x_emotion": 5.0},
            )
        )
        result = reconstruct_vocabulary_from_memories(store)
        assert "version" in result
        assert isinstance(result["emotions"], list)
        for e in result["emotions"]:
            assert "name" in e
            assert "description" in e
            assert "category" in e
            assert "decay_half_life_days" in e
    finally:
        store.close()
```

- [ ] **Step 2: Run — verify FAIL**

- [ ] **Step 3: Implementation**

```python
# brain/health/reconstruct.py
"""Identity reconstruction — rebuild vocabulary from memory references.

When all backups of emotion_vocabulary.json are corrupt and reset would
otherwise fire, the brain re-learns its own vocabulary from how it has
been operating: scan memories.db for distinct emotion names, register
framework baseline + persona-extensions for any unknown names found.
"""

from __future__ import annotations

from brain.emotion import vocabulary as _vocabulary
from brain.memory.store import MemoryStore

PLACEHOLDER_DESCRIPTION = "(reconstructed from memory)"
PLACEHOLDER_DECAY_DAYS = 1.0  # conservative — fast decay until user re-tunes


def reconstruct_vocabulary_from_memories(store: MemoryStore) -> dict:
    """Build emotion_vocabulary.json content: baseline + extensions found in memories."""
    baseline_names: set[str] = {e.name for e in _vocabulary._BASELINE}
    seen_names: set[str] = set()
    for mem in store.search_text("", active_only=True, limit=None):
        for name in mem.emotions:
            seen_names.add(name)

    entries: list[dict] = []
    # Framework baseline always — these are immutable identity.
    for e in _vocabulary._BASELINE:
        entries.append(
            {
                "name": e.name,
                "description": e.description,
                "category": e.category,
                "decay_half_life_days": e.decay_half_life_days,
                "intensity_clamp": e.intensity_clamp,
            }
        )

    # Persona extensions: any emotion name in memories not in baseline.
    for name in sorted(seen_names - baseline_names):
        entries.append(
            {
                "name": name,
                "description": PLACEHOLDER_DESCRIPTION,
                "category": "persona_extension",
                "decay_half_life_days": PLACEHOLDER_DECAY_DAYS,
                "intensity_clamp": 10.0,
            }
        )

    return {"version": 1, "emotions": entries}
```

- [ ] **Step 4: Tests pass.**

- [ ] **Step 5: Commit**

```bash
git add brain/health/reconstruct.py tests/unit/brain/health/test_reconstruct.py
git commit -m "feat(health): reconstruct vocabulary from memories — Health T5"
```

---

### Task 6: `PRAGMA integrity_check` on SQLite stores

**Files:**
- Modify: `brain/memory/store.py` (add integrity check in `__init__`)
- Modify: `brain/memory/hebbian.py` (same)
- Create: `tests/unit/brain/memory/test_store_integrity.py`

Adds `BrainIntegrityError` to `brain/health/anomaly.py` (or a new module — judgment call). Stores raise this on integrity check failure; callers can catch and convert to a `BrainAnomaly` of kind `sqlite_integrity_fail` + action `alarmed_unrecoverable`.

- [ ] **Step 1: Failing tests**

```python
# tests/unit/brain/memory/test_store_integrity.py
"""Tests for SQLite integrity check on store + hebbian open."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.health.anomaly import BrainIntegrityError
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


def test_memory_store_clean_db_passes_integrity_check(tmp_path: Path) -> None:
    """A clean store opens without raising."""
    store = MemoryStore(db_path=tmp_path / "memories.db")
    store.close()
    # Re-open — integrity check runs again on fresh open
    store2 = MemoryStore(db_path=tmp_path / "memories.db")
    store2.close()


def test_memory_store_corrupt_db_raises_integrity_error(tmp_path: Path) -> None:
    """A file with bad SQLite header → BrainIntegrityError on open."""
    db = tmp_path / "memories.db"
    db.write_bytes(b"this is not a SQLite database")
    with pytest.raises(BrainIntegrityError):
        MemoryStore(db_path=db)


def test_hebbian_matrix_clean_db_passes(tmp_path: Path) -> None:
    h = HebbianMatrix(db_path=tmp_path / "hebbian.db")
    h.close()
    h2 = HebbianMatrix(db_path=tmp_path / "hebbian.db")
    h2.close()


def test_hebbian_matrix_corrupt_db_raises(tmp_path: Path) -> None:
    db = tmp_path / "hebbian.db"
    db.write_bytes(b"not sqlite")
    with pytest.raises(BrainIntegrityError):
        HebbianMatrix(db_path=db)
```

- [ ] **Step 2: Run — verify FAIL** (`BrainIntegrityError` doesn't exist; integrity check not wired).

- [ ] **Step 3: Implementation**

a) Append to `brain/health/anomaly.py`:

```python
class BrainIntegrityError(Exception):
    """Raised when a SQLite database fails PRAGMA integrity_check.

    The brain's memory or hebbian graph is unrecoverable from this state
    in v1 — surfaces as a Layer 3 alarm in the audit log.
    """

    def __init__(self, db_path: str, detail: str) -> None:
        super().__init__(f"integrity check failed for {db_path}: {detail}")
        self.db_path = db_path
        self.detail = detail
```

b) In `brain/memory/store.py`, find `MemoryStore.__init__` and after the connection is established, run integrity check before any other init logic:

```python
# After self._conn = sqlite3.connect(...) (or however the connection is opened)
result = self._conn.execute("PRAGMA integrity_check").fetchall()
if result != [("ok",)]:
    detail = "; ".join(str(row[0]) for row in result)
    self._conn.close()
    from brain.health.anomaly import BrainIntegrityError
    raise BrainIntegrityError(str(db_path), detail)
```

c) Same change in `brain/memory/hebbian.py:HebbianMatrix.__init__`.

Note: when the SQLite header is invalid, `sqlite3.connect` itself may not raise (since SQLite creates DBs lazily), but the first query will. The integrity check is the trigger. If `execute` raises `sqlite3.DatabaseError`, catch it and raise `BrainIntegrityError`.

Adjust:

```python
try:
    result = self._conn.execute("PRAGMA integrity_check").fetchall()
except sqlite3.DatabaseError as exc:
    self._conn.close()
    from brain.health.anomaly import BrainIntegrityError
    raise BrainIntegrityError(str(db_path), str(exc)) from exc
if result != [("ok",)]:
    ...
```

- [ ] **Step 4: Tests pass.** Run: `uv run pytest tests/unit/brain/memory/test_store_integrity.py -v`

Then full suite to verify no regression: `uv run pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add brain/health/anomaly.py brain/memory/store.py brain/memory/hebbian.py tests/unit/brain/memory/test_store_integrity.py
git commit -m "feat(health): SQLite integrity check on store + hebbian open — Health T6"
```

---

### Task 7: `walk_persona`

**Files:**
- Create: `brain/health/walker.py`
- Create: `tests/unit/brain/health/test_walker.py`

Walks every persona file using `attempt_heal` (for atomic-rewrite) or noop (for append-only — those self-heal at read time). Includes SQLite integrity check via opening the stores. Returns a list of `BrainAnomaly` (empty if everything healthy).

- [ ] **Step 1: Failing tests**

```python
# tests/unit/brain/health/test_walker.py
"""Tests for brain.health.walker."""

from __future__ import annotations

import json
from pathlib import Path

from brain.health.walker import walk_persona
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


def _setup_persona(tmp_path: Path) -> Path:
    persona = tmp_path / "persona"
    persona.mkdir()
    MemoryStore(db_path=persona / "memories.db").close()
    HebbianMatrix(db_path=persona / "hebbian.db").close()
    return persona


def test_walk_clean_persona_no_anomalies(tmp_path: Path) -> None:
    persona = _setup_persona(tmp_path)
    anomalies = walk_persona(persona)
    assert anomalies == []


def test_walk_detects_corrupt_atomic_rewrite_file(tmp_path: Path) -> None:
    persona = _setup_persona(tmp_path)
    (persona / "user_preferences.json").write_text("{not json", encoding="utf-8")
    anomalies = walk_persona(persona)
    assert len(anomalies) == 1
    assert anomalies[0].file == "user_preferences.json"
    # File is healed (reset to default since no .bak)
    assert (persona / "user_preferences.json").exists()


def test_walk_detects_sqlite_corruption(tmp_path: Path) -> None:
    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / "memories.db").write_bytes(b"not sqlite")
    HebbianMatrix(db_path=persona / "hebbian.db").close()  # clean
    anomalies = walk_persona(persona)
    assert any(a.kind == "sqlite_integrity_fail" for a in anomalies)


def test_walk_returns_multiple_anomalies(tmp_path: Path) -> None:
    persona = _setup_persona(tmp_path)
    (persona / "user_preferences.json").write_text("{not json", encoding="utf-8")
    (persona / "persona_config.json").write_text("{also not json", encoding="utf-8")
    anomalies = walk_persona(persona)
    assert len(anomalies) >= 2
    files = {a.file for a in anomalies}
    assert "user_preferences.json" in files
    assert "persona_config.json" in files
```

- [ ] **Step 2: Run — verify FAIL**

- [ ] **Step 3: Implementation**

```python
# brain/health/walker.py
"""Proactive walk over every persona file — used by `nell health check`
and triggered automatically when a heartbeat tick produces ≥2 anomalies."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.health.anomaly import BrainAnomaly, BrainIntegrityError
from brain.health.attempt_heal import attempt_heal

# Atomic-rewrite files this walker checks. Each entry: (filename, default_factory).
_DEFAULTS: dict[str, dict] = {
    "user_preferences.json": {"dream_every_hours": 24.0},
    "persona_config.json": {"provider": "claude-cli", "searcher": "ddgs"},
    "heartbeat_config.json": {},  # all-default HeartbeatConfig
    "heartbeat_state.json": {},  # triggers fresh-init via load fallback
    "interests.json": {"version": 1, "interests": []},
    "reflex_arcs.json": {"version": 1, "arcs": []},
    "emotion_vocabulary.json": {"version": 1, "emotions": []},
}


def walk_persona(persona_dir: Path) -> list[BrainAnomaly]:
    """Check every persona file. Heal what's healable; report anomalies."""
    anomalies: list[BrainAnomaly] = []

    # Atomic-rewrite files
    for name, default in _DEFAULTS.items():
        path = persona_dir / name
        _, anomaly = attempt_heal(path, default_factory=lambda d=default: d)
        if anomaly is not None:
            anomalies.append(anomaly)

    # SQLite integrity
    for db_name in ("memories.db", "hebbian.db"):
        db_path = persona_dir / db_name
        if not db_path.exists():
            continue
        try:
            if db_name == "memories.db":
                from brain.memory.store import MemoryStore

                MemoryStore(db_path=db_path).close()
            else:
                from brain.memory.hebbian import HebbianMatrix

                HebbianMatrix(db_path=db_path).close()
        except BrainIntegrityError as exc:
            anomalies.append(
                BrainAnomaly(
                    timestamp=datetime.now(UTC),
                    file=db_name,
                    kind="sqlite_integrity_fail",
                    action="alarmed_unrecoverable",
                    quarantine_path=None,
                    likely_cause="disk",
                    detail=exc.detail[:500],
                )
            )

    return anomalies
```

- [ ] **Step 4: Tests pass.**

- [ ] **Step 5: Commit**

```bash
git add brain/health/walker.py tests/unit/brain/health/test_walker.py
git commit -m "feat(health): walk_persona for proactive scan — Health T7"
```

---

### Task 8: `compute_pending_alarms`

**Files:**
- Create: `brain/health/alarm.py`
- Create: `tests/unit/brain/health/test_alarm.py`

Computes alarms from the audit log + adaptive state. An alarm fires for: ≥3 anomalies on the same file in 7 days AFTER adaptive treatment was already in effect (i.e., 6 anomalies total — 3 to bump, 3 more after); reset_to_default on identity-critical file; sqlite_integrity_fail. Acknowledged events suppressed.

- [ ] **Step 1: Failing tests**

```python
# tests/unit/brain/health/test_alarm.py
"""Tests for brain.health.alarm."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.alarm import compute_pending_alarms


def _audit_entry(file: str, when: datetime, action: str = "restored_from_bak1", kind: str = "json_parse_error") -> dict:
    return {
        "timestamp": when.isoformat().replace("+00:00", "Z"),
        "anomalies": [
            {
                "timestamp": when.isoformat().replace("+00:00", "Z"),
                "file": file,
                "kind": kind,
                "action": action,
                "quarantine_path": None,
                "likely_cause": "unknown",
                "detail": "",
            }
        ],
    }


def _seed(persona_dir: Path, entries: list[dict]) -> None:
    persona_dir.mkdir(parents=True, exist_ok=True)
    p = persona_dir / "heartbeats.log.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def test_no_anomalies_no_alarms(tmp_path: Path) -> None:
    assert compute_pending_alarms(tmp_path) == []


def test_reset_to_default_on_identity_file_alarms(tmp_path: Path) -> None:
    _seed(tmp_path, [_audit_entry("emotion_vocabulary.json", datetime.now(UTC), action="reset_to_default")])
    alarms = compute_pending_alarms(tmp_path)
    assert len(alarms) == 1
    assert alarms[0].file == "emotion_vocabulary.json"


def test_sqlite_integrity_fail_alarms(tmp_path: Path) -> None:
    _seed(tmp_path, [_audit_entry("memories.db", datetime.now(UTC), kind="sqlite_integrity_fail", action="alarmed_unrecoverable")])
    alarms = compute_pending_alarms(tmp_path)
    assert len(alarms) == 1
    assert alarms[0].kind == "sqlite_integrity_fail"


def test_acknowledged_alarm_suppressed(tmp_path: Path) -> None:
    when = datetime.now(UTC)
    entries = [
        _audit_entry("emotion_vocabulary.json", when, action="reset_to_default"),
        # User acknowledged after the reset
        {
            "timestamp": (when + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
            "user_acknowledged": ["emotion_vocabulary.json"],
        },
    ]
    _seed(tmp_path, entries)
    alarms = compute_pending_alarms(tmp_path)
    assert alarms == []
```

- [ ] **Step 2: Run — verify FAIL**

- [ ] **Step 3: Implementation**

```python
# brain/health/alarm.py
"""Pending-alarm computation — derived from audit log; no separate state file."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.anomaly import AlarmEntry
from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.utils.time import parse_iso_utc

_IDENTITY_FILES = frozenset({
    "emotion_vocabulary.json",
    "interests.json",
    "reflex_arcs.json",
    # future: "soul.json"
})

WINDOW_DAYS = 7


def compute_pending_alarms(persona_dir: Path) -> list[AlarmEntry]:
    """Walk recent audit log; return alarms not yet acknowledged."""
    audit_path = persona_dir / "heartbeats.log.jsonl"
    cutoff = datetime.now(UTC) - timedelta(days=WINDOW_DAYS)

    # First pass: collect all anomalies + acknowledgments in window.
    anomalies: list[dict] = []
    acknowledged_files: set[str] = set()

    for entry in read_jsonl_skipping_corrupt(audit_path):
        try:
            ts = parse_iso_utc(entry["timestamp"])
        except (KeyError, ValueError, TypeError):
            continue
        if ts < cutoff:
            continue

        ack = entry.get("user_acknowledged") or []
        if isinstance(ack, list):
            for f in ack:
                if isinstance(f, str):
                    acknowledged_files.add(f)

        for a in entry.get("anomalies") or []:
            if isinstance(a, dict):
                anomalies.append(a)

    # Second pass: count per file + classify alarmable.
    by_file: dict[str, list[dict]] = {}
    for a in anomalies:
        f = a.get("file")
        if not isinstance(f, str):
            continue
        by_file.setdefault(f, []).append(a)

    alarms: list[AlarmEntry] = []
    for f, anoms in by_file.items():
        if f in acknowledged_files:
            continue

        is_alarm = False
        for a in anoms:
            if a.get("action") == "reset_to_default" and f in _IDENTITY_FILES:
                is_alarm = True
                break
            if a.get("kind") == "sqlite_integrity_fail":
                is_alarm = True
                break
        # ≥6 anomalies in window = recurring-after-adaptation
        if len(anoms) >= 6:
            is_alarm = True

        if is_alarm:
            first_seen = min(parse_iso_utc(a["timestamp"]) for a in anoms)
            kind = anoms[-1].get("kind", "json_parse_error")
            alarms.append(
                AlarmEntry(
                    file=f,
                    kind=kind,
                    first_seen_at=first_seen,
                    occurrences_in_window=len(anoms),
                )
            )

    return alarms
```

- [ ] **Step 4: Tests pass.**

- [ ] **Step 5: Commit**

```bash
git add brain/health/alarm.py tests/unit/brain/health/test_alarm.py
git commit -m "feat(health): compute_pending_alarms from audit log — Health T8"
```

---

**End of PR-1 (T1-T8): all `brain/health/` internals + SQLite integrity. Standalone, no wiring yet. Run full suite + open PR.**

---

### Task 9: Wire helpers into config + state files

**Files:**
- Modify: `brain/persona_config.py`
- Modify: `brain/user_preferences.py`
- Modify: `brain/engines/heartbeat.py` (HeartbeatConfig + HeartbeatState load/save)
- Update: existing tests for these files

For each of `PersonaConfig`, `UserPreferences`, `HeartbeatConfig`, `HeartbeatState`:
- Replace `load(path)` body with `attempt_heal(path, default_factory, schema_validator=None)`. Return data; if anomaly is not None, callers get it via a context-aware mechanism (next task wires that into the heartbeat tick).
- Replace `save(path)` with `save_with_backup(path, data, backup_count=...)` where `backup_count` comes from `compute_treatment(persona_dir, file).backup_count`.

The most natural shape: each `load` returns `(config_or_state, BrainAnomaly | None)`. Existing callers ignore the anomaly until T11 wires it. For now, expose a parallel `load_with_anomaly` and keep the existing `load` as a thin wrapper that drops the anomaly + logs it. T11 swaps callers over.

- [ ] **Step 1: Update tests for new behavior**

For each affected test file (e.g., `test_persona_config.py`, `test_user_preferences.py`, `test_heartbeat.py`'s config + state tests), add 2 new tests per file:

```python
def test_load_corrupt_file_quarantines_and_resets(tmp_path: Path) -> None:
    p = tmp_path / "persona_config.json"
    p.write_text("{not json", encoding="utf-8")
    cfg = PersonaConfig.load(p)
    # Defaults returned
    assert cfg.provider == DEFAULT_PROVIDER
    # Quarantine present
    assert any(tmp_path.glob("persona_config.json.corrupt-*"))
    # Live file rewritten with default
    assert p.exists()


def test_load_corrupt_file_restores_from_bak(tmp_path: Path) -> None:
    p = tmp_path / "persona_config.json"
    bak = tmp_path / "persona_config.json.bak1"
    bak.write_text(json.dumps({"provider": "ollama", "searcher": "noop"}), encoding="utf-8")
    p.write_text("{not json", encoding="utf-8")
    cfg = PersonaConfig.load(p)
    assert cfg.provider == "ollama"
    assert cfg.searcher == "noop"
```

- [ ] **Step 2: Run — verify FAIL** (current load methods don't quarantine, don't restore from .bak)

- [ ] **Step 3: Implementation**

For each module, the pattern:

```python
# brain/persona_config.py
from brain.health.attempt_heal import attempt_heal, save_with_backup
from brain.health.adaptive import compute_treatment


class PersonaConfig:
    @classmethod
    def load(cls, path: Path) -> PersonaConfig:
        cfg, anomaly = cls.load_with_anomaly(path)
        if anomaly is not None:
            logger.warning("PersonaConfig.load self-healed %s: %s", path, anomaly.action)
        return cfg

    @classmethod
    def load_with_anomaly(cls, path: Path) -> tuple[PersonaConfig, BrainAnomaly | None]:
        data, anomaly = attempt_heal(
            path,
            default_factory=lambda: {"provider": DEFAULT_PROVIDER, "searcher": DEFAULT_SEARCHER},
        )
        # Existing per-field type validation...
        provider = ...
        searcher = ...
        return cls(provider=provider, searcher=searcher), anomaly

    def save(self, path: Path) -> None:
        treatment = compute_treatment(path.parent, path.name)
        save_with_backup(
            path,
            {"provider": self.provider, "searcher": self.searcher},
            backup_count=treatment.backup_count,
        )
        if treatment.verify_after_write:
            self._verify_after_write(path)

    def _verify_after_write(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Re-run any validator
        except json.JSONDecodeError:
            # Restore from .bak1 (which has prior good content)
            bak = path.with_name(f"{path.name}.bak1")
            if bak.exists():
                os.replace(bak, path)
                logger.warning("PersonaConfig.save verify-after-write failed; restored from .bak1")
```

Apply the same pattern to `UserPreferences`, `HeartbeatConfig`, `HeartbeatState`.

For `HeartbeatConfig.load`, retain the existing user_preferences.json merge logic at the end of the load.

For `HeartbeatState.load`, retain back-compat fallback for `last_growth_at`.

The `default_factory` for each must produce a JSON-serialisable dict that the existing `_from_dict` parsing path happily accepts.

- [ ] **Step 4: Tests pass.** Existing config + state tests should still pass; new heal-from-bak / corrupt-resets tests pass too.

Run: `uv run pytest tests/unit/brain/test_persona_config.py tests/unit/brain/test_user_preferences.py tests/unit/brain/engines/test_heartbeat.py -q`

- [ ] **Step 5: Commit**

```bash
git add brain/persona_config.py brain/user_preferences.py brain/engines/heartbeat.py \
        tests/unit/brain/test_persona_config.py tests/unit/brain/test_user_preferences.py \
        tests/unit/brain/engines/test_heartbeat.py
git commit -m "feat(health): wire attempt_heal/save_with_backup into config + state files — Health T9"
```

---

### Task 10: Wire helpers into identity files

**Files:**
- Modify: `brain/engines/_interests.py` (InterestSet load/save)
- Modify: `brain/engines/reflex.py` (find arcs file callsites)
- Modify: `brain/emotion/persona_loader.py` (load_persona_vocabulary)
- Modify: `brain/growth/scheduler.py` (`_append_to_vocabulary` uses `save_with_backup`; `_read_current_vocabulary_names` uses `attempt_heal`)
- Update: existing tests for these modules

Same pattern as T9. The vocabulary case is the trickiest because `load_persona_vocabulary` currently returns `int` (count registered); we want to surface the anomaly. Add a parallel `load_persona_vocabulary_with_anomaly` returning `(int, BrainAnomaly | None)`; the existing function becomes a thin wrapper.

For the all-baks-corrupt path on `emotion_vocabulary.json`, the `default_factory` returns the *empty* `{"version":1,"emotions":[]}`. The reconstruction path via `reconstruct_vocabulary_from_memories` fires from a higher layer (heartbeat tick when it observes an `emotion_vocabulary.json` reset_to_default anomaly). We don't push reconstruction into `attempt_heal` itself — the heal helper stays generic. The heartbeat ticks layer applies reconstruction.

Add a test:

```python
def test_load_persona_vocabulary_corrupt_file_self_heals(tmp_path: Path, caplog) -> None:
    bak = tmp_path / "emotion_vocabulary.json.bak1"
    bak.write_text(json.dumps({"version": 1, "emotions": [
        {"name": "lingering", "description": "x", "category": "persona_extension",
         "decay_half_life_days": 7.0, "intensity_clamp": 10.0}
    ]}), encoding="utf-8")
    p = tmp_path / "emotion_vocabulary.json"
    p.write_text("{not json", encoding="utf-8")

    count, anomaly = load_persona_vocabulary_with_anomaly(p)
    assert anomaly is not None
    assert anomaly.action == "restored_from_bak1"
    assert count >= 1  # lingering registered
```

- [ ] **Step 1-5: TDD cycle as in T9.**

- [ ] **Commit**

```bash
git commit -m "feat(health): wire attempt_heal/save_with_backup into identity files — Health T10"
```

---

### Task 11: Wire `read_jsonl_skipping_corrupt` into log readers

**Files:**
- Modify: `brain/growth/log.py` (`read_growth_log` becomes a thin wrapper)
- Modify: `brain/engines/heartbeat.py` (audit log reader, if any)
- Modify: `brain/engines/research.py` (research log)
- Modify: `brain/engines/reflex.py` (reflex log)
- Modify: `brain/engines/dream.py` (dream log)

The growth log currently reimplements the per-line skip logic inline. After this task, every log reader calls `read_jsonl_skipping_corrupt(path)` and post-processes the dicts (constructing typed objects from each parsed dict).

Find each `path.read_text(encoding="utf-8").splitlines()` log-reading callsite in the brain modules above. Replace with `read_jsonl_skipping_corrupt(path)` + a per-dict transformer.

Keep the existing tests for each log reader; they should continue to pass since the behavior is identical (per-line skip + warning) just sourced from the shared helper.

- [ ] **Step 1: Run existing log-reader tests** to confirm they pass before refactor.
- [ ] **Step 2: Refactor each log reader.** Verify tests still pass after each change.
- [ ] **Step 3: Commit:** `feat(health): generalize read_jsonl_skipping_corrupt to all log readers — Health T11`

---

### Task 12: Heartbeat tick anomaly aggregation + audit log + cross-file walk

**Files:**
- Modify: `brain/engines/heartbeat.py` (substantial)
- Update: `tests/unit/brain/engines/test_heartbeat.py`

The heartbeat tick collects anomalies from all engine sub-calls during the tick, runs `walk_persona` if `len(anomalies) >= 2`, computes `pending_alarms_count`, writes both into the audit log payload, and surfaces the count in `HeartbeatResult`.

Mechanism:
- Each load function in the brain now has a `*_with_anomaly` variant returning `(data, BrainAnomaly | None)`.
- The heartbeat engine, when it constructs sub-engines or loads state, uses the anomaly-aware variants and accumulates into a per-tick `list[BrainAnomaly]`.
- After all per-tick work completes (decay, reflex, dream, research, growth) and BEFORE the audit log write, the heartbeat collected list is summed; if `>= 2`, run `walk_persona` and merge results.
- Audit log payload gets `"anomalies": [...]` (list of `to_dict()` outputs) and `"pending_alarms_count": <int>`.
- `HeartbeatResult` gains `anomalies: tuple[BrainAnomaly, ...]` (default `()`) and `pending_alarms_count: int = 0`.
- Compact CLI: if `len(result.anomalies) > 0` and no banner, print one `🩹` line. If `result.pending_alarms_count > 0`, print persistent banner above all engine output.

Tests:

```python
def test_heartbeat_audit_log_contains_anomalies_field(tmp_path: Path) -> None:
    """Every audit entry has an 'anomalies' key (empty list on clean)."""
    # ... run a clean heartbeat tick ...
    last = json.loads(audit_lines[-1])
    assert "anomalies" in last
    assert last["anomalies"] == []
    assert last["pending_alarms_count"] == 0


def test_heartbeat_corrupt_state_file_self_heals_and_records(tmp_path: Path) -> None:
    """Corrupting heartbeat_state.json triggers heal + anomaly entry."""
    # First tick to initialize
    # Corrupt the state file
    # Second tick should heal + log anomaly
    # Audit log entry has anomalies != []


def test_heartbeat_multi_anomaly_tick_triggers_walk(tmp_path: Path) -> None:
    """≥2 anomalies in one tick → walk_persona runs and merges findings."""
    # Corrupt 2 files before tick
    # Run tick
    # Audit log entry's anomalies length should reflect both + any walk-discovered


def test_heartbeat_alarm_in_pending_alarms_count(tmp_path: Path) -> None:
    """An alarm-worthy anomaly increments pending_alarms_count."""
    # Force a reset_to_default on emotion_vocabulary.json
    # Next tick: pending_alarms_count >= 1


def test_compact_cli_self_treated_line(...):
    """When self-treated, compact CLI emits 🩹 line."""


def test_compact_cli_alarm_banner(...):
    """When pending_alarms_count > 0, compact CLI prepends banner."""
```

This is a wide-touch task. Allocate the whole task to it.

- [ ] **Step 1-5: TDD cycle.**

- [ ] **Commit:** `feat(health): heartbeat tick anomaly aggregation + cross-file walk + audit log + compact CLI — Health T12`

---

### Task 13: `nell health show / check / acknowledge` CLI

**Files:**
- Modify: `brain/cli.py`
- Create: `tests/unit/brain/health/test_cli_health.py`

Three subcommands:

- **`nell health show --persona X`** — print pending alarms + recent self-treatments (last 7 days from audit log). Exit 0.
- **`nell health check --persona X`** — call `walk_persona`, print per-file ✅/⚠️/❌ status, exit 0 on healthy / self-treated, exit 2 on alarm.
- **`nell health acknowledge --persona X [--file <name>] [--all]`** — append a `user_acknowledged` entry to the audit log.

Rejected actions: `restore` is deliberately omitted; `add` / `approve` / `reject` are not part of health.

Tests follow the pattern of T7 in the Phase 2a plan:

```python
def test_cli_health_show_clean(monkeypatch, tmp_path, capsys):
    """Healthy persona → 'Brain is healthy.'"""

def test_cli_health_show_with_recent_treatments(monkeypatch, tmp_path, capsys):
    """Recent self-treatments listed; alarms section present (empty)."""

def test_cli_health_check_clean(...):

def test_cli_health_check_corrupt_file(...):

def test_cli_health_acknowledge_clears_alarm(...):
    """After acknowledge, compute_pending_alarms returns []."""

def test_cli_health_no_destructive_actions(monkeypatch, tmp_path):
    """`nell health restore` and `nell health add` raise SystemExit."""
```

- [ ] **Step 1-5: TDD cycle.**

- [ ] **Commit:** `feat(health): nell health show/check/acknowledge CLI — Health T13`

---

### Task 14: Acceptance smoke + final verification

**Files:** none (verification only).

- [ ] **Step 1: zero anthropic imports**

```bash
rg 'import anthropic' brain/health/
```
Expected: no matches.

- [ ] **Step 2: lint**

```bash
uv run ruff check brain/ tests/
uv run ruff format --check brain/ tests/
```

- [ ] **Step 3: full pytest**

```bash
uv run pytest -q
```

Expected: 534 (current) + new tests = ~590+ passing.

- [ ] **Step 4: sandbox smoke**

```bash
# 1. Clean tick — no anomalies expected
uv run nell heartbeat --persona nell.sandbox --trigger manual --provider fake
```
Expected: no `🩹` line, no banner.

```bash
# 2. Corrupt user_preferences.json manually, run tick
echo "{not json" > "$NELLBRAIN_HOME/personas/nell.sandbox/user_preferences.json"
uv run nell heartbeat --persona nell.sandbox --trigger manual --provider fake
```
Expected: tick completes, 🩹 line appears in compact output, audit log entry has `anomalies` populated.

```bash
# 3. Inspect health
uv run nell health show --persona nell.sandbox
uv run nell health check --persona nell.sandbox
```
Expected: show lists the recent self-treatment; check walks the persona and confirms healthy.

- [ ] **Step 5: open PR-2** with summary of T9-T14, link to PR-1, list all 15 acceptance criteria, request CI verification.

---

## Acceptance Criteria (from spec §8)

After all tasks ship, these must hold:

- [ ] `brain/health/` package with all modules listed.
- [ ] Every atomic-rewrite load function calls `attempt_heal`.
- [ ] Every atomic-rewrite save function calls `save_with_backup`.
- [ ] Every append-only log reader uses `read_jsonl_skipping_corrupt`.
- [ ] `MemoryStore` and `HebbianMatrix` run `PRAGMA integrity_check` on open.
- [ ] Heartbeat tick aggregates anomalies; audit log carries them; `pending_alarms_count` computed.
- [ ] Cross-file walk fires automatically when `len(anomalies) >= 2`.
- [ ] `nell health show / check / acknowledge` work.
- [ ] `reconstruct_vocabulary_from_memories` reconstructs against a real memories.db.
- [ ] Adaptive backup-depth bumps after 3 corruptions in 7 days.
- [ ] Verify-after-write activates with elevated backup count.
- [ ] Compact heartbeat CLI shows three states (silent / 🩹 / banner).
- [ ] `uv run pytest -q` green; lint + format clean.
- [ ] `rg 'import anthropic' brain/health/` zero.
- [ ] Sandbox smoke: clean tick silent; manual corruption triggers heal + 🩹.
