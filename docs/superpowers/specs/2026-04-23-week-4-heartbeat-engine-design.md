# Week 4.5 — Heartbeat Engine Design Spec

> **Status:** design approved by Hana 2026-04-23. Implementation plan to follow via superpowers:writing-plans.
> **Scope:** ships `brain/engines/heartbeat.py` (event-driven orchestrator tick) + `nell heartbeat` subcommand. Second cognitive engine. Event-driven (no daemon): app open/close triggers + optional GUI idle-pulses. Uses Week 4's `DreamEngine` as a sub-engine.

---

## 1. Goal

Keep the brain alive even when the user isn't actively chatting. Each heartbeat tick applies emotional + Hebbian decay to age-weight accumulated state, optionally triggers a dream cycle (rate-limited, default 1/day), and persists timing state so subsequent ticks know "how much time has passed."

Guarantees minimum self-growth via **two heartbeats per app session** (one on open, one on close), so even a 30-second visit produces a decay + log entry. Frequency is configurable per-persona; the future Tauri GUI exposes the `dream_every_hours` setting.

## 2. Non-goals

- **Daemon / background process.** Heartbeats fire from the application's lifecycle hooks (app open → `nell heartbeat --trigger open`; app close → `nell heartbeat --trigger close`). No launchd/cron/systemd. No daemons on the user's machine outside the app.
- **Research engine.** Stub only — heartbeat logs "research deferred" and moves on. Real research engine lands in a later week.
- **Reflex engine.** Separate engine with its own trigger model; not part of heartbeat.
- **GUI configuration UI.** The config file exists and is read; the GUI that edits it is future work.

## 3. Architecture

```
brain/
├── engines/
│   └── heartbeat.py       (NEW — HeartbeatEngine, HeartbeatResult, HeartbeatState, HeartbeatConfig)
└── cli.py                 (MODIFIED — wire `nell heartbeat` subcommand)
```

Heartbeat composes existing layers — it does NOT duplicate anything:
- Reuses `DreamEngine.run_cycle()` from Week 4 for the dream step
- Uses `apply_decay()` from `brain/emotion/decay.py` (Week 2) for per-memory emotion decay
- Uses `HebbianMatrix.decay_all()` + `.garbage_collect()` (Week 3) for graph hygiene
- New persistent state in `heartbeat_state.json` + new config in `heartbeat_config.json`

## 4. Behaviour — the tick

On each `nell heartbeat --trigger X` invocation:

1. **Load persisted state** from `<persona_dir>/heartbeat_state.json`. If it doesn't exist → **first-ever tick**: initialize state (all `last_*_at` = now, `tick_count` = 0) and exit early with no work. Log "initialized, deferring until next tick."

2. **Load config** from `<persona_dir>/heartbeat_config.json`. If it doesn't exist → write default config (`dream_every_hours: 24`, `decay_rate_per_tick: 0.01`, `gc_threshold: 0.01`, `emit_memory: "conditional"`).

3. **Compute elapsed** = `now - state.last_tick_at`. Used for decay calculations below.

4. **Emotion decay** — for every active memory in the store:
   - Apply `apply_decay(memory.emotions, elapsed_seconds)` using each emotion's half-life from the vocabulary
   - Below-0.01 emotions drop out (noise floor)
   - If the memory's emotions changed, `store.update(id, emotions=...)` and recompute score
   - Protected memories (`memory.protected=True`) are skipped

5. **Hebbian decay** — `hebbian.decay_all(rate=decay_rate_per_tick * elapsed_hours / 24)`. Scaled so a tick after 24h decays at the configured rate; shorter elapsed times decay less.

6. **Hebbian GC** — `hebbian.garbage_collect(threshold=gc_threshold)`. Prunes weak edges. Returns count pruned.

7. **Maybe-dream gate:**
   - If `now - state.last_dream_at ≥ config.dream_every_hours`:
     - Trigger `DreamEngine(store, hebbian, None, provider, log_path).run_cycle()` with default params
     - Set `state.last_dream_at = now`
     - Record `dream_id` in this tick's result
   - Else: skip, log "dream gated: N.N hours until next"

8. **Maybe-research** — stub. Always skips with log "research deferred: engine not implemented".

9. **Update state:** `last_tick_at = now`, `tick_count += 1`, `last_trigger = args.trigger`.

10. **Write `heartbeat_state.json`** atomically (temp file + rename).

11. **Append JSONL line to `heartbeats.log.jsonl`** with timestamp, trigger, elapsed_seconds, memories_decayed, edges_pruned, dream_id (or null), research_triggered (false).

12. **Conditional heartbeat memory** — if config.emit_memory == "always" OR (emit_memory == "conditional" AND something material happened — dream fired OR edges pruned > 10 OR memories_decayed > 20): write a new `memory_type="heartbeat"` memory via LLM with a short first-person reflection on the tick. Otherwise skip.

13. **Return `HeartbeatResult`** with the tick summary.

## 5. Public types

```python
@dataclass
class HeartbeatConfig:
    """Per-persona heartbeat configuration. Loaded from heartbeat_config.json."""
    dream_every_hours: float = 24.0
    decay_rate_per_tick: float = 0.01      # applied to Hebbian weights per 24h
    gc_threshold: float = 0.01
    emit_memory: Literal["always", "conditional", "never"] = "conditional"

    @classmethod
    def load(cls, path: Path) -> HeartbeatConfig: ...
    def save(self, path: Path) -> None: ...


@dataclass
class HeartbeatState:
    """Per-persona heartbeat state. Loaded from heartbeat_state.json."""
    last_tick_at: datetime
    last_dream_at: datetime
    last_research_at: datetime
    tick_count: int
    last_trigger: str  # "open", "close", "manual", or "init"

    @classmethod
    def load(cls, path: Path) -> HeartbeatState | None:
        """Return None if the file doesn't exist (→ first-ever tick)."""
    def save(self, path: Path) -> None: ...
    @classmethod
    def fresh(cls, trigger: str) -> HeartbeatState: ...


@dataclass(frozen=True)
class HeartbeatResult:
    """Outcome of a single heartbeat tick."""
    trigger: str
    elapsed_seconds: float
    memories_decayed: int
    edges_pruned: int
    dream_id: str | None            # None if dream gated
    dream_gated_reason: str | None  # "not_due" / "first_tick" / None if dream fired
    research_deferred: bool
    heartbeat_memory_id: str | None  # None if skipped
    initialized: bool  # True if this was the first-ever tick (deferred work)


@dataclass
class HeartbeatEngine:
    """Composes decay + dream + research into one orchestrator tick."""
    store: MemoryStore
    hebbian: HebbianMatrix
    provider: LLMProvider         # for dream + optional heartbeat memory
    state_path: Path
    config_path: Path
    dream_log_path: Path
    heartbeat_log_path: Path

    def run_tick(self, *, trigger: str = "manual", dry_run: bool = False) -> HeartbeatResult:
        ...
```

## 6. First-ever tick behaviour (defer rule)

When `heartbeat_state.json` doesn't exist:
- Create it with all `last_*_at = now` and `tick_count = 0`
- **Do no work** — no decay applied (elapsed would be since epoch), no dream fired, no memory written
- Log "initialized, deferring work until next tick"
- Return `HeartbeatResult(initialized=True, memories_decayed=0, edges_pruned=0, dream_id=None, ...)`

This protects against:
- A freshly-migrated persona (like Nell's 1,142 memories) having "years of decay" applied on the first tick
- A freshly-cloned sandbox getting an unexpected dream before the user's first real session
- Time-travel bugs if the system clock is off at first boot

The first *real* tick happens on invocation #2.

## 7. Dream gating details

- `last_dream_at` is initialized to `now` on first tick → dream won't fire until `dream_every_hours` (default 24h) has passed
- After a dream fires, `last_dream_at = now` → gates the next dream by another 24h
- If the app is closed for 5 days then reopened: one tick on open, elapsed=120h > 24h → dream fires once, `last_dream_at` now pegged to current, next tick in the open-session will gate-skip the dream
- Tunable per-persona via config file; future GUI writes here

## 8. CLI

```bash
# Application lifecycle hooks (Tauri app will call these):
nell heartbeat --persona nell --trigger open
nell heartbeat --persona nell --trigger close

# Manual / CI:
nell heartbeat --persona nell --trigger manual
nell heartbeat --persona nell --dry-run                # show what would happen, skip writes
nell heartbeat --persona nell --provider fake          # for test scripts
```

Flags:
- `--persona <name>` (default: `nell`)
- `--trigger <open|close|manual>` (default: `manual`)
- `--provider <name>` (default: `claude-cli`)
- `--dry-run` — compute state + what-would-happen, no writes

## 9. Config file format

`<persona_dir>/heartbeat_config.json`:

```json
{
  "dream_every_hours": 24.0,
  "decay_rate_per_tick": 0.01,
  "gc_threshold": 0.01,
  "emit_memory": "conditional"
}
```

Validation on load: all fields must match expected types; unknown fields ignored (forward-compat); missing fields use dataclass defaults.

## 10. State file format

`<persona_dir>/heartbeat_state.json`:

```json
{
  "last_tick_at": "2026-04-23T11:30:00.000000Z",
  "last_dream_at": "2026-04-23T10:54:28.329483Z",
  "last_research_at": "2026-04-23T11:30:00.000000Z",
  "tick_count": 17,
  "last_trigger": "open"
}
```

All timestamps ISO-8601 UTC with `Z` suffix (matching the Week 3.5 migrator's manifest format).

**Atomic writes:** write to `<path>.new`, then `os.rename()` to target. Defends against crash mid-write.

## 11. Log format

`<persona_dir>/heartbeats.log.jsonl` — one JSON line per tick:

```json
{"timestamp": "2026-04-23T11:30:00Z", "trigger": "open", "elapsed_seconds": 3600.0, "memories_decayed": 8, "edges_pruned": 3, "dream_id": "abc-123", "research_deferred": true, "tick_count": 17}
```

Initialized ticks also log an entry with a note:
```json
{"timestamp": "...", "trigger": "init", "initialized": true, "note": "first-ever tick, work deferred"}
```

## 12. Testing strategy

All tests use `FakeProvider` + `tmp_path` persona dirs. Zero network. ~15 engine tests + 4-5 CLI = ~20 total. Target suite size: ~313.

### Engine tests (~15)
- First-ever tick → HeartbeatResult.initialized=True, state file created, no decay applied
- Second tick on same persona → decay applied, state updated
- Dream gate: first post-init tick doesn't dream (elapsed < 24h since init)
- Dream gate: tick after 25h of elapsed time triggers dream
- After dream fires, subsequent tick within 24h gates dream with reason="not_due"
- Emotion decay: recent memory with love=9.0 after 6h elapsed → love reduced per half-life
- Protected memories: decay skipped
- Below-floor emotions: dropped from memory.emotions dict
- Hebbian decay: weights reduced proportional to elapsed
- Hebbian GC: weak edges pruned, count reported
- Research always deferred (research_deferred=True in result)
- Heartbeat memory emitted when emit_memory="always"
- Heartbeat memory skipped when emit_memory="never"
- Heartbeat memory conditional: emitted when dream fired, skipped on trivial tick
- State file atomic write: written to .new then renamed
- Dry-run: no state/store/hebbian writes

### CLI tests (~5)
- `nell heartbeat --trigger open` first invocation → initialized response, files created
- `nell heartbeat --trigger close` as second tick → does work
- `nell heartbeat --trigger manual --dry-run` → no writes
- Unknown persona → FileNotFoundError mentioning 'persona'
- Dream subprocess failure (provider raises) → surfaces cleanly, state still saved (partial-tick recovery)

## 13. Dependencies

None new. `json`, `datetime`, `pathlib`, `os`, `dataclasses` all stdlib. Uses existing Week 2/3/4 internals.

## 14. Success criteria

1. Fresh persona + `nell heartbeat --trigger open` → `heartbeat_state.json` + `heartbeat_config.json` materialise, `heartbeats.log.jsonl` has an "init" entry, no other changes.
2. Second invocation → real work: decay applied, state updated, log entry, maybe dream if 24h elapsed.
3. `dream_every_hours: 0.01` in config + second tick → dream fires, dream memory present in store, `last_dream_at` updated.
4. App open/close sequence simulates as two back-to-back ticks → both record in log, dream gates correctly between them.
5. `uv run nell heartbeat --help` prints subcommand usage.
6. 313 tests pass on macOS + Windows + Linux CI.
7. Hana runs `nell heartbeat --persona nell.sandbox --trigger manual` twice (first init, second real) and inspects results.

## 15. Open questions deferred

- **GUI heartbeat config editor.** Lives in the Tauri app; not part of this spec.
- **Scheduled mid-session ticks** (e.g. "pulse every 2h while app open"). Could be added as a config field (`idle_pulse_hours`) but currently out of scope — open+close is the v1 baseline.
- **Research engine.** Separate spec + implementation, later week.
- **Heartbeat memory prompt shape.** TBD — similar structure to dream's system prompt but first-person present-tense about the tick itself. Design inline during Task 2 of implementation.
- **Backup / rollback on dream failure.** If DreamEngine raises mid-tick, do we revert emotion decay? For v1: no rollback — state moves forward, partial work is logged. Heartbeats are idempotent-ish because `last_tick_at` advances.

---

*End of spec. Implementation plan to follow via superpowers:writing-plans.*
