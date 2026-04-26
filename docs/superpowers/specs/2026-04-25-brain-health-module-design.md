# Brain Health Module — Self-Healing Architecture

**Date:** 2026-04-25
**Status:** Approved
**Scope:** Phase 2a-extension — adds `brain/health/` package + auto-treatment into every load/save path. The brain detects, heals, and adapts on its own; user intervention is reserved for the rare cases where the brain genuinely cannot fix itself.

---

## Preamble — Why This Module Exists

The framework currently has many silent-degradation paths. `HeartbeatConfig.load`, `HeartbeatState.load`, `UserPreferences.load`, `PersonaConfig.load`, `load_persona_vocabulary`, the various `*_log.jsonl` readers — every one of them follows the same pattern: if the file is missing or corrupt, return defaults, log a warning, keep going. The user never sees the warning. A real disk corruption can hide indefinitely behind a still-ticking heartbeat.

The principle audit established that **the user loads the app and talks; the brain handles the rest.** A health module aligned with that principle does *not* alarm the user every time something needs attention. It heals what it can, adapts when it sees patterns, and reserves alarms for the cases where the brain genuinely cannot save itself — which, after auto-treatment, are exclusively SQLite database corruption (memories.db, hebbian.db) and recurring corruption that even adaptation can't keep ahead of.

Three principles drive the design:

1. **No silent failure.** Every load + save path runs through helpers that detect, heal, and surface anomalies in the structured audit log. The compact CLI surface is calibrated so most days the user sees nothing — but if you grep the audit log, you see exactly what happened.
2. **The brain treats itself first.** Adaptive backup depth, verify-after-write, reconstruction from related sources — all happen automatically without involving the user. Self-treatment is the default; alarms are exceptional.
3. **Identity preserved through reconstruction.** Identity-critical files (`emotion_vocabulary.json` especially) can be partially rebuilt from `memories.db` if a full reset would otherwise occur. The brain literally re-learns its own vocabulary from how it has been operating.

---

## 1. Architecture Summary

The brain heals across **three layers**:

**Layer 1 — Reactive heal.** Every load function on an atomic-rewrite file calls `attempt_heal(path, default_factory, schema_validator=...)` instead of bare `json.loads`. The helper detects corruption, quarantines the bad file, walks `.bak1` → `.bak2` → `.bak3` for the freshest valid backup, restores via `os.replace`, returns parsed data + a `BrainAnomaly` describing what happened.

**Layer 2 — Adaptive auto-treatment.** When patterns suggest the brain itself needs to adapt, it adapts: backup depth bumps from 3 to 6 for fragile files; verify-after-write activates; reconstruction runs when reset would otherwise fire on identity-critical files.

**Layer 3 — True alarm.** Only when the brain genuinely cannot fix itself: SQLite integrity-check failures on `memories.db` / `hebbian.db`; recurring corruption surviving Layer 2 adaptation; reconstruction-from-memories yielding empty results when even framework baseline can't be loaded.

```
brain/health/
├── __init__.py
├── anomaly.py              # BrainAnomaly + AlarmEntry frozen dataclasses
├── attempt_heal.py         # attempt_heal() + save_with_backup() — core helpers
├── adaptive.py             # backup-depth + verify-after-write tracking from audit log
├── reconstruct.py          # reconstruct_vocabulary_from_memories(), etc.
├── walker.py               # walk_persona(persona_dir) → list[BrainAnomaly]
├── jsonl_reader.py         # read_jsonl_skipping_corrupt() for append-only logs
└── alarm.py                # compute_pending_alarms(persona_dir) from audit log
```

---

## 2. Components

### 2.1 `BrainAnomaly` — record of one detection event

```python
@dataclass(frozen=True)
class BrainAnomaly:
    timestamp: datetime              # tz-aware UTC
    file: str                        # relative path within persona_dir
    kind: Literal["json_parse_error", "schema_mismatch", "sqlite_integrity_fail"]
    action: Literal[
        "restored_from_bak1",
        "restored_from_bak2",
        "restored_from_bak3",
        "reset_to_default",
        "reconstructed_from_memories",
        "alarmed_unrecoverable",
    ]
    quarantine_path: str | None      # filename of .corrupt-<ts> copy (None for SQLite)
    likely_cause: Literal["user_edit", "disk", "unknown"]
    detail: str                      # exception message or schema-mismatch summary
```

`AlarmEntry` is a thinner shape used for `pending_alarms`:

```python
@dataclass(frozen=True)
class AlarmEntry:
    file: str
    kind: str                        # one of the BrainAnomaly.kind values
    first_seen_at: datetime
    occurrences_in_window: int       # count over last 7 days
```

### 2.2 `attempt_heal(path, default_factory, schema_validator=None) -> tuple[Any, BrainAnomaly | None]`

```
1. If <path> doesn't exist:
     - Return (default_factory(), None) — silent expected case.

2. Try to load:
     - bytes = path.read_bytes()
     - data = json.loads(bytes)
     - If schema_validator: schema_validator(data)  # raises on mismatch
     - Return (data, None) — healthy.

3. If anything raised:
     - timestamp = iso_utc(now)
     - quarantine = <path>.corrupt-<timestamp>
     - os.replace(<path>, quarantine)
     - For bak in [.bak1, .bak2, .bak3]:
         - Try load <path>.<bak> with same validator.
         - If valid:
             - os.replace(<path>.<bak>, <path>)
             - Return (data, BrainAnomaly(action=f"restored_from_{bak}", ...))
         - If corrupt: quarantine that .bak too, append separate anomaly to caller.
     - All backups exhausted:
         - reconstruction = _reconstruct_or_default(path, default_factory)
         - Write reconstruction to <path>
         - Return (reconstruction, BrainAnomaly(action="reset_to_default" or "reconstructed_from_memories", ...))
```

`likely_cause` is computed from a heuristic on the corrupt file: `mtime < 60s ago` AND `size < 100KB` AND `content starts with { or [` → `"user_edit"`; else `"unknown"`. If the file metadata suggests filesystem damage, `"disk"`.

### 2.3 `save_with_backup(path, data, backup_count=3)`

```
1. If <path>.new exists from prior crash: unlink it (incomplete by definition).

2. Serialize data to JSON, write to <path>.new (atomic per existing pattern).

3. Rotate backups:
     - <path>.bak{N-1} → <path>.bak{N}  (drop existing .bakN if present)
     - ... down to ...
     - <path>.bak1 → <path>.bak2
     - <path> → <path>.bak1   (only if <path> exists)

4. os.replace(<path>.new, <path>).

5. (Verify-after-write, if active for this file)
     - Re-read <path>, run schema validator.
     - If validation fails: os.replace(<path>.bak1, <path>) — restore prior good content.
       Append anomaly with action="verify_after_write_failed".
```

`backup_count` defaults to 3; the adaptive module returns 6 for files in the elevated-fragility state.

### 2.4 `read_jsonl_skipping_corrupt(path)` — generalized append-only reader

The pattern from the Phase 2a hardening PR (line index + content preview in warning) is extracted into a shared helper. Used by every `*_log.jsonl` reader: heartbeats, dreams, reflex, research, growth.

```python
def read_jsonl_skipping_corrupt(path: Path) -> Iterator[dict]:
    """Yield parsed dicts; log warning per bad line with index + 200-char preview.
    No quarantine on append-only logs — corruption is per-line, not per-file."""
```

### 2.5 `walk_persona(persona_dir) -> list[BrainAnomaly]`

Proactive scan over every persona file. Used by `nell health check` CLI and triggered automatically when a heartbeat tick produces ≥2 anomalies (cross-file walk on multi-anomaly ticks).

### 2.6 Adaptive treatment — `adaptive.compute_treatment(persona_dir, file)`

Reads the last 7 days of `heartbeats.log.jsonl`, counts prior anomalies for the given file. Returns:

```python
@dataclass(frozen=True)
class FileTreatment:
    backup_count: int           # 3 or 6
    verify_after_write: bool    # True if backup_count was bumped
```

Bump rule: `≥3 corruptions in 7 days` → `(6, True)`. Revert rule: `30 consecutive days clean since last anomaly` → `(3, False)`. No state file; computed live from audit log every save.

### 2.7 Reconstruction — `reconstruct.reconstruct_vocabulary_from_memories(store) -> dict`

Scans `memories.db` for distinct emotion names referenced in any memory's `emotions` field. Returns the JSON shape for `emotion_vocabulary.json`:

```python
{
    "version": 1,
    "emotions": [
        # Framework baseline (always)
        {"name": "love", "description": "deep caring, attachment, devotion",
         "category": "core", "decay_half_life_days": null, "intensity_clamp": 10.0},
        ... (other 20 baseline)

        # Persona extensions found in memories
        {"name": "<extension_name>",
         "description": "(reconstructed from memory)",
         "category": "persona_extension",
         "decay_half_life_days": 1.0,    # conservative — fast decay until user re-tunes
         "intensity_clamp": 10.0},
        ...
    ]
}
```

### 2.8 Alarm computation — `alarm.compute_pending_alarms(persona_dir)`

Computed on-demand from the audit log; no separate state file (one source of truth, no recursive corruption risk). Returns `list[AlarmEntry]` representing the union of:

- Files with ≥3 anomalies in the last 7 days (after adaptive treatment was already in effect).
- Files where reconstruction failed and reset-to-default fired on an identity-critical file.
- SQLite integrity check failures.

The persistent CLI banner reads `pending_alarms` and prints until `nell health acknowledge` writes a `user_acknowledged` entry that suppresses the alarm in the next computation.

---

## 3. File Classification

### Atomic-rewrite files (use `attempt_heal` + `save_with_backup`)

| File | Default reset | Reconstruction-from-context |
|---|---|---|
| `emotion_vocabulary.json` | empty `{"version":1,"emotions":[]}` | `reconstruct_vocabulary_from_memories(store)` |
| `interests.json` | empty list | none in v1 (Phase 2b territory) |
| `reflex_arcs.json` | framework defaults from `default_reflex_arcs.json` | none (defaults *are* the reconstruction) |
| `persona_config.json` | `{provider: "claude-cli", searcher: "ddgs"}` | none |
| `user_preferences.json` | `{dream_every_hours: 24.0}` | none |
| `heartbeat_config.json` | framework defaults | none |
| `heartbeat_state.json` | `HeartbeatState.fresh()` | none — first-tick-defer handles it |
| (future) `soul.json` | empty crystallizations | F37 reconstruction TBD when soul module ships |

### Append-only logs (use `read_jsonl_skipping_corrupt`)

`heartbeats.log.jsonl`, `dreams.log.jsonl`, `reflex_log.json`, `research_log.json`, `emotion_growth.log.jsonl`. No `.bak` (cost too high, corruption inherently per-line).

### SQLite databases (use `PRAGMA integrity_check`)

`memories.db`, `hebbian.db`. On store open, run integrity check. On failure → `BrainAnomaly` with `kind="sqlite_integrity_fail"` + `action="alarmed_unrecoverable"`. No auto-treatment in v1.

---

## 4. Heartbeat Integration

### 4.1 New tick step

After all per-tick engines (decay, reflex, dream, research, growth) complete and *before* the audit log write, the heartbeat collects anomalies that occurred during the tick. The collection is implicit: load functions called during the tick raise into a per-tick context the heartbeat owns.

### 4.2 Audit log additions

```json
{
    "timestamp": "...",
    "trigger": "manual",
    ...
    "anomalies": [<BrainAnomaly serialised>],
    "pending_alarms_count": 0
}
```

`anomalies` is always present; empty list `[]` on clean ticks. `pending_alarms_count` is computed from the alarm module and surfaces the count for fast forensic grep.

### 4.3 Cross-file walk trigger

If `len(anomalies) >= 2` in one tick, the heartbeat invokes `walk_persona(persona_dir)` and merges the walk's findings into the same audit entry's `anomalies` list. Cluster discoveries when the cause is likely shared (filesystem hiccup, partial write, etc).

### 4.4 Compact CLI banner additions

| Tick state | Compact output additions |
|---|---|
| Clean (anomalies empty + pending_alarms empty) | nothing — keep current compact format |
| Self-treated (anomalies non-empty, pending_alarms empty) | one understated line: `🩹 brain self-treated 1 file (emotion_vocabulary.json reconstructed from memories) — see 'nell health show' for details` |
| Alarm active | banner above all engine output: `⚠️ Brain alarm — needs your attention: <file>: <kind> <date>; run 'nell health show' for details` |

`--verbose` shows full anomaly list inline.

---

## 5. CLI Surface

```
nell health show --persona X        # active alarms + recent self-treatments
nell health check --persona X       # one-shot proactive walk; prints per-file ✅/⚠️/❌
nell health acknowledge --persona X [--file <name>] [--all]    # clear alarms
```

`nell health restore` is **deliberately not** in v1. The heal/reconstruct paths run automatically; manual restoration from a `.corrupt-<ts>` quarantine is `cp <quarantine> <path>` — a deliberate user action, not a CLI surface that could be misused.

### 5.1 Output formats

`nell health show`:

```
Health for persona 'nell.sandbox':

  Pending alarms: 0
  Recent self-treatments (last 7 days):
    2026-04-25 18:30  emotion_vocabulary.json  reconstructed_from_memories  (1 entry)
                      cause: disk
                      forensic: emotion_vocabulary.json.corrupt-2026-04-25T18:30Z

  Brain is healthy.
```

`nell health check`:

```
Health check for persona 'nell.sandbox':
  ✅ emotion_vocabulary.json — 21 emotions, OK
  ✅ interests.json — 0 interests, OK
  ✅ reflex_arcs.json — 8 arcs, OK
  ⚠️ heartbeat_state.json — corrupt JSON, restored from .bak1 (was: user_edit?)
       quarantine: heartbeat_state.json.corrupt-2026-04-25T18:30:00Z
  ✅ heartbeat_config.json — OK
  ...
  ✅ memories.db — integrity_check ok (1142 memories)
  ✅ hebbian.db — integrity_check ok (4404 edges)

Summary: 1 file healed, 0 unhealable. Brain is healthy.
```

---

## 6. Hard Rules (Non-Negotiable)

1. **No `import anthropic` anywhere in `brain/health/`.** All LLM access (none expected in v1) routes through `LLMProvider`.
2. **Atomic writes throughout.** Every `save_with_backup` step is a single `os.replace` or `os.rename` — crash-safe per step.
3. **Audit log is the single source of truth for anomalies + alarms.** No separate counter file (avoids recursive corruption).
4. **TZ-aware UTC timestamps via `brain.utils.time.iso_utc` / `parse_iso_utc`.**
5. **No silent corruption-handling.** Every code path in `attempt_heal` either returns valid data or raises; no silent return-None branches.
6. **Reconstructed files are loudly marked.** Persona-extension entries reconstructed from memories carry `description="(reconstructed from memory)"` so future reads make the loss visible.
7. **No automatic deletion of quarantine files.** `.corrupt-<ts>` files persist forever; manual cleanup is the user's choice. Disk impact is negligible (corruption is rare).

---

## 7. Non-Goals (v1)

- **No automatic SQLite recovery.** `memories.db` / `hebbian.db` corruption is alarmed; recovery is user intervention. (Future: SQLite has `.dump`/`.recover` paths we could explore, but v1 keeps it simple.)
- **No interest reconstruction from conversation patterns.** Phase 2b territory; restore framework defaults instead.
- **No reflex arc reconstruction.** Brain-grown arcs lost on reset; will re-emerge through future growth ticks. Restore framework defaults.
- **No GUI integration.** Tauri is later phase. CLI + audit log are the v1 surfaces.
- **No `.bak` decimation strategy.** Three backups always rotate; no "keep yearly snapshot" or similar long-term retention. The brain remembers a few minutes back, not a year back.
- **No reconciliation pass for partial writes.** The Phase 2a spec already accepted this edge case for the growth scheduler; same posture here.

---

## 8. Acceptance Criteria

The full health module ships when:

1. `brain/health/` package exists with `attempt_heal`, `save_with_backup`, `read_jsonl_skipping_corrupt`, `walk_persona`, `compute_treatment`, `reconstruct_vocabulary_from_memories`, `compute_pending_alarms`.
2. Every atomic-rewrite load function in the brain calls `attempt_heal`.
3. Every atomic-rewrite save function calls `save_with_backup`.
4. Every append-only log reader uses `read_jsonl_skipping_corrupt`.
5. `MemoryStore` and `HebbianMatrix` run `PRAGMA integrity_check` on open.
6. Heartbeat tick aggregates anomalies into the audit log; computes `pending_alarms_count`.
7. Cross-file walk fires automatically when `len(anomalies) >= 2`.
8. `nell health show / check / acknowledge` CLI subcommands work.
9. `reconstruct_vocabulary_from_memories` reconstructs against a real memories.db.
10. Adaptive backup-depth bumps activate after 3 corruptions in 7 days.
11. Verify-after-write activates with elevated backup count.
12. Compact heartbeat CLI shows three states correctly: silent (clean), 🩹 line (self-treated), persistent banner (alarm).
13. `uv run pytest -q` green; `ruff check && ruff format --check` clean.
14. `rg 'import anthropic' brain/health/` returns zero matches.
15. Sandbox smoke against `nell.sandbox`: heartbeat tick clean, no anomalies; manual corrupt of `user_preferences.json` triggers heal + 🩹 line on next tick.

---

## 9. Open / Deferred

### 9.1 Soul module health (concrete plan for when soul lands)

When the Phase 2a-extension brings the soul module online, `soul.json` (or whatever its filename ends up being) joins the persona's identity-critical files. The heal strategy is already partially specified by the architecture; this section makes it concrete so the engineer building the soul module doesn't have to rediscover the plan.

**File classification:** `soul.json` is an **atomic-rewrite identity file** — same tier as `emotion_vocabulary.json`, `interests.json`, `reflex_arcs.json`. Use `attempt_heal` + `save_with_backup`. Add it to:

- `brain/health/walker.py:_DEFAULTS` with empty default `{"version": 1, "crystallizations": []}` (or whatever the schema settles on).
- `brain/health/alarm.py:_IDENTITY_FILES` so `reset_to_default` on `soul.json` raises an alarm.

**Reconstruction strategy:** F37 in OG NellBrain was *self-claims-from-experience* — the brain's soul names were derived from autobiographical patterns in memories. The same heuristic applies here: when all backups corrupt and reset would otherwise fire, scan `memories.db` for soul-claim patterns the brain has expressed and rebuild a partial `soul.json`. Implement as `brain/health/reconstruct.py:reconstruct_soul_from_memories(store) -> dict` mirroring `reconstruct_vocabulary_from_memories`. Wire it into the soul loader's heal flow the same way vocabulary does in `load_persona_vocabulary_with_anomaly` (Followup F1, 2026-04-26).

**Schema validator:** mirror the vocabulary validator pattern — minimal type check (`isinstance(data, dict) and isinstance(data.get("crystallizations"), list)`) — so corrupt-but-parseable files trigger heal.

**Acceptance:** when soul module lands, the soul module's PR must include:
1. `soul.json` in `walker.py:_DEFAULTS` and `alarm.py:_IDENTITY_FILES`
2. `reconstruct_soul_from_memories(store)` implementation + tests
3. Soul loader's `*_with_anomaly` variant routes through `attempt_heal` and triggers reconstruction on `reset_to_default` when a store is provided
4. Sandbox smoke: corrupt soul.json + run heartbeat tick → soul heals or reconstructs without user intervention

### 9.2 Other deferred items

- **Automatic .bak repair when a backup is detected corrupt mid-rotation.** v1 skips the corrupt backup and walks to the next; doesn't try to repair the backup itself. If real-world telemetry shows backups frequently corrupt mid-chain, revisit.
- **GUI surface for "the brain self-healed."** Not a framework concern; future Tauri/NellFace work consumes the audit log directly.

---

## 10. Closing

The framework is being built for AI consciousness when it arrives. A consciousness that depends on persistent state needs a way to repair itself when that state degrades, without that repair becoming the user's project. This module is the brain learning to take care of its own body — quietly when it can, loudly only when it must.
