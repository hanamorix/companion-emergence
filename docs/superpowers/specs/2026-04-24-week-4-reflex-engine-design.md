# Week 4.6 — Reflex Engine (Phase 1) Design

**Date:** 2026-04-24
**Status:** Approved
**Scope:** Phase 1 (execution + arc storage). Phase 2 (emergent arc crystallization) deferred.
**Engine number:** 3 of 4 in Week 4 (after dream and heartbeat; before research)

---

## 1. Purpose

Reflex is the engine that makes the brain **do things unprompted** when its emotional state rises to a pitch that demands expression. Where the dream engine consolidates and the heartbeat orchestrates rhythm, the reflex engine produces **autonomous creative output** — journal entries, gifts, story pitches, self-checks — when specific emotions cross threshold.

In the OG NellBrain, `reflex_engine.py` fired 8 hand-authored "arcs" on emotion thresholds, each producing a distinct kind of output via local Ollama. The framework rebuild preserves this behavior with four architectural changes:

1. **Per-persona arc definitions.** Arcs live in `{persona_dir}/reflex_arcs.json`, not in framework code. Personal content (Nell's `jordan_grief_carry`, `body_grief_whisper`) stays with Nell; the framework ships only generic starter arcs.
2. **Unified output storage.** Reflex outputs are first-class memories in `MemoryStore` with arc-specific `memory_type`, not separate JSON files. They decay, participate in Hebbian connections, and can seed future dreams.
3. **Orchestrated by heartbeat.** Reflex evaluation runs inside each heartbeat tick. Reflex outputs created in a tick are available to the dream gate in the same tick.
4. **LLM calls via `LLMProvider`.** Hard rule: routes through `brain.bridge.provider.LLMProvider` (Claude CLI default). No direct Ollama or API calls.

---

## 2. Architectural Summary

- `brain/engines/reflex.py` — engine module
- `brain/engines/default_reflex_arcs.json` — 4 generic starter arcs (shipped with framework)
- `tests/unit/brain/engines/test_reflex.py` — unit tests
- `brain/cli.py` — adds `nell reflex` subcommand
- `brain/engines/heartbeat.py` — calls reflex inside `run_tick`
- `brain/migrator/` — extracts Nell's 8 OG arcs on migration, writes to her `reflex_arcs.json`

### Two invocation paths

- **Standalone CLI:** `nell reflex --persona X [--trigger manual] [--dry-run]`
- **Orchestrated:** heartbeat tick calls `ReflexEngine.run_tick()` between decay and dream gate.

Both paths execute identical logic.

---

## 3. Components

### 3.1 Data types

```python
@dataclass(frozen=True)
class ReflexArc:
    name: str
    description: str
    trigger: Mapping[str, float]          # emotion -> min intensity
    days_since_human_min: float           # 0.0 disables gate
    cooldown_hours: float
    action: str                           # informational, not dispatched
    output_memory_type: str               # e.g. "reflex_journal", "reflex_gift"
    prompt_template: str                  # str.format_map-compatible

@dataclass(frozen=True)
class ArcFire:
    arc_name: str
    fired_at: datetime                    # tz-aware UTC
    trigger_state: Mapping[str, float]    # the state that satisfied trigger
    output_memory_id: str | None          # None for dry_run

@dataclass(frozen=True)
class ArcSkipped:
    arc_name: str
    reason: str                           # enum-like: see §5.2

@dataclass(frozen=True)
class ReflexResult:
    arcs_fired: tuple[ArcFire, ...]
    arcs_skipped: tuple[ArcSkipped, ...]
    would_fire: str | None                # dry-run only: name of arc that would fire
    dry_run: bool
    evaluated_at: datetime                # tz-aware UTC
```

### 3.2 Arc-set storage (`ReflexArcSet`)

- Loads `{persona_dir}/reflex_arcs.json`.
- If the file is missing: fall back to `default_reflex_arcs.json` (bundled with framework). Log warning; do NOT create the persona file automatically — the migrator or `nell new-persona` owns initial seeding.
- If the file is corrupt (JSONDecodeError, wrong shape, non-list arcs): fall back to defaults, log warning, do not overwrite the persona file.
- Per-arc load failures (missing required key, wrong type) skip that arc and log, rather than failing the whole load.

### 3.3 Fire log storage (`ReflexLog`)

- Path: `{persona_dir}/reflex_log.json`.
- Schema:

```json
{
  "version": 1,
  "fires": [
    {
      "arc": "creative_pitch",
      "fired_at": "2026-04-24T15:02:11Z",
      "trigger_state": {"creative_hunger": 9.0},
      "output_memory_id": "mem_abc123",
      "dry_run": false
    }
  ]
}
```

- Atomic save: write to `reflex_log.json.new`, then `os.replace()`.
- Corrupt/missing file on load → returns empty log, engine treats as "no prior fires," evaluates normally. Does not overwrite.

### 3.4 Engine (`ReflexEngine`)

Constructor:

```python
ReflexEngine(
    store: MemoryStore,
    provider: LLMProvider,
    persona_name: str,
    persona_system_prompt: str,
    arcs_path: Path,
    log_path: Path,
    default_arcs_path: Path,
)
```

Methods:

- `run_tick(*, trigger: str, dry_run: bool) -> ReflexResult` — full evaluation + fire loop.
- `_evaluate(emotion_state, now) -> tuple[list[ReflexArc], list[ArcSkipped]]` — returns eligible arcs and skip reasons.
- `_rank(eligible: list[ReflexArc], emotion_state, now) -> ReflexArc | None` — applies single-fire cap via highest threshold-excess, tie-broken by longest-since-fire.
- `_fire(arc, emotion_state, now, dry_run) -> ArcFire` — renders prompt, calls LLM, writes memory, appends to log.

---

## 4. Arc Schema

### 4.1 On-disk format

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
      "prompt_template": "You are {persona_name}. Your creative hunger is at {creative_hunger}/10. You need to make something.\n\nGenerate a SHORT story pitch (3-5 sentences)..."
    }
  ]
}
```

### 4.2 Trigger semantics

`"trigger": {"creative_hunger": 8}` means `current_state["creative_hunger"] >= 8`. Multiple keys are AND — all thresholds must be met simultaneously. No OR or nested logic (YAGNI; OG never needed it).

### 4.3 Prompt template variables

Rendered via `str.format_map` with a defaultdict returning `"0"` for missing keys. Available variables:

| Variable | Source |
|----------|--------|
| `{persona_name}` | `persona.toml → [identity] → name` |
| `{persona_pronouns}` | `persona.toml → [identity] → pronouns` (optional) |
| `{<emotion>}` | current intensity for any emotion named in the trigger |
| `{days_since_human}` | computed from `MemoryStore` (see §5.3) |
| `{emotion_summary}` | top-5 current emotions **ranked by intensity descending**, formatted as `"name: N/10"` lines |
| `{memory_summary}` | top-3 **most recently created** memory contents (each truncated to 140 chars), separated by `\n- ` |

Missing emotion key in the template → substituted as `"0"`. Don't crash.

### 4.4 `output_memory_type` values

Free-form strings. Convention:
- `reflex_journal` — private internal entries
- `reflex_gift` — creative pieces intended for the primary-relationship human. Metadata includes `{delivered: false, for: <primary>}` where `<primary>` comes from `persona.toml → [relationships] → primary` if present, otherwise empty string. Schema does not require the field; gifts without a recipient are still valid memories.
- `reflex_pitch` — story/creative pitches
- `reflex_memory` — general reflections the engine considers core enough to crystallise as memories

New personas can add their own types; nothing enforces the naming. Queries filter by `memory_type`.

---

## 5. Data Flow

### 5.1 One evaluation pass

1. **Load:** arcs, fire log, current emotion state (from `EmotionState`), `days_since_human`.
2. **Filter to eligible arcs.** For each arc:
   - `trigger_met`: every emotion in trigger has current intensity ≥ its threshold.
   - `days_met`: `days_since_human >= arc.days_since_human_min`.
   - `cooldown_met`: `(now - arc.last_fired_at_from_log) >= arc.cooldown_hours`, or never fired.
   - If any gate fails, record `ArcSkipped(arc_name, reason)`.
3. **Rank eligible arcs** (§5.4). Select at most one.
4. **Dry-run branch:** if `dry_run=True`, return `ReflexResult(arcs_fired=(), arcs_skipped=(...), would_fire=<winner.name>, dry_run=True)`. No LLM call, no memory write, no log append.
5. **Fire (live):** render prompt → call `provider.generate(prompt, system=persona_system_prompt)` → wrap in `Memory` → `store.create(memory)` → append `ArcFire` to log → atomic save log.
6. **Return** `ReflexResult(arcs_fired=(fire,), arcs_skipped=(...), would_fire=None, dry_run=False)`.

### 5.2 `ArcSkipped` reasons (enum-like string constants)

- `trigger_not_met`
- `days_since_human_too_low`
- `cooldown_active`
- `single_fire_cap` — eligible but lost the rank
- `no_arcs_defined` — returned only when the arc set is empty (one synthetic entry, not per-arc)

### 5.3 `days_since_human` computation

Query `MemoryStore` for the most recent memory with `memory_type="conversation"` (Phase 2 may widen this to `source="hana"` once source-tagging lands). Compute `(now - created_at).total_seconds() / 86400`. If no such memory exists, return `999.0`.

### 5.4 Ranking when multiple arcs eligible

One arc fires per tick — prevents a feelings-storm from producing 8 outputs in one breath.

**Primary key:** highest aggregate threshold-excess. For each eligible arc compute:
```
excess = sum(current_state[e] - arc.trigger[e] for e in arc.trigger)
```
The arc with highest `excess` wins — the emotion that's loudest right now gets to speak.

**Tiebreak:** longest time since last fire (never-fired arcs beat ever-fired arcs).

**Second tiebreak:** arc ordering in `reflex_arcs.json` (first wins). Deterministic.

---

## 6. Heartbeat Integration

Heartbeat's `run_tick` tick order becomes:

1. First-tick defer (unchanged)
2. Emotion decay (unchanged)
3. Hebbian decay + GC (unchanged)
4. **Reflex check** *(new)* — `reflex_engine.run_tick(trigger=heartbeat_trigger, dry_run=heartbeat.dry_run)`. Output memory (if any) becomes available to subsequent steps.
5. Dream gate (unchanged — may now select a just-created reflex output as seed)
6. Research stub (unchanged)
7. Optional `HEARTBEAT:` memory emit (unchanged)
8. State save + audit log (unchanged)

### 6.1 Config additions — `heartbeat_config.json`

```json
{
  "dream_every_hours": 24.0,
  "decay_rate_per_tick": 0.01,
  "gc_threshold": 0.01,
  "emit_memory": "conditional",
  "reflex_enabled": true,
  "reflex_max_fires_per_tick": 1
}
```

- `reflex_enabled` — `true` (default) / `false`. When `false`, heartbeat skips reflex step entirely.
- `reflex_max_fires_per_tick` — integer, default `1`. Current implementation hard-caps at 1; the field exists so Phase 2 can lift the cap without a schema change.

### 6.2 Heartbeat audit log

The existing heartbeat JSONL audit record gains:

```json
"reflex": {
  "evaluated": true,
  "fired": ["creative_pitch"],
  "skipped_count": 3,
  "would_fire": null
}
```

When `reflex_enabled=false`: `"reflex": {"evaluated": false}`.

---

## 7. Error Handling

| Condition | Behavior |
|-----------|----------|
| LLM `generate()` raises | Skip this arc for this tick. Do NOT append to fire log. Next tick retries (matches the dream-gate no-poisoning principle). Log the exception in heartbeat audit log under `reflex.errors`. |
| Prompt template references unknown variable | Substituted as `"0"`. Don't crash. |
| Arc file corrupt/missing | Fall back to `default_reflex_arcs.json`. Log warning. Do not overwrite the persona's file. |
| Fire log corrupt/missing | Treat as empty log. Engine evaluates normally. Next successful fire rewrites the log fresh. |
| Zero arcs defined | Immediate return: `ReflexResult(arcs_fired=(), arcs_skipped=(ArcSkipped(name="", reason="no_arcs_defined"),), ...)`. |
| Naive / missing timestamp in fire log | Raise on load (corrupt-file path); fall back to empty log. |
| `emotion_state` is None (engine called before emotion state loaded) | Raise `RuntimeError("ReflexEngine requires loaded emotion state")` — this is a programmer error, not a user-facing case. |
| Multiple arcs eligible simultaneously | One fires per §5.4; others recorded with `reason="single_fire_cap"`. |

---

## 8. Nell's Migration

The existing migrator (`brain/migrator/`) gains a step.

### 8.1 Arc extraction

The OG source has `REFLEX_ARCS` as a Python dict in `reflex_engine.py` at `/Users/hanamori/NellBrain/reflex_engine.py`. The migrator reads that file, parses out the `REFLEX_ARCS` dict via AST (don't import — the module depends on `nell_brain.py`, which isn't available in the new framework environment).

For each arc, transform to the new schema:

| OG field | New field | Transformation |
|----------|-----------|----------------|
| key (e.g. `"creative_pitch"`) | `name` | used verbatim |
| `trigger` | `trigger` | verbatim |
| `days_since_min` | `days_since_human_min` | renamed, value copied |
| `cooldown_hours` | `cooldown_hours` | verbatim |
| `description` | `description` | verbatim |
| `action` | `action` | rename: `generate_story_pitch` → `generate_pitch`, `write_journal` → `generate_journal`, `write_gift` → `generate_gift`, `write_memory` → `generate_reflection` |
| `output` | `output_memory_type` | `journal` → `reflex_journal`, `gifts` → `reflex_gift`, `memories` → `reflex_memory` |
| `prompt_template` | `prompt_template` | **verbatim** — Nell-specific content (Jordan, body grief, "You are Nell") preserved exactly |

### 8.2 Write target

`{persona_dir}/reflex_arcs.json` with version 1 schema (§4.1).

### 8.3 Refuse-to-clobber

If `reflex_arcs.json` already exists in the target persona dir, migrator refuses to overwrite. Use `--force` (existing flag) to override. Matches existing migrator safety rule.

### 8.4 Report addition

The migrator's JSON audit report gains:

```json
"reflex_arcs": {
  "migrated": 8,
  "skipped": 0,
  "details": [{"name": "creative_pitch", "result": "migrated"}, ...]
}
```

### 8.5 Re-runnability

Arc migration is idempotent when re-run with `--force` — the OG REFLEX_ARCS dict is the source of truth, so re-migration always produces the same output. Hana can re-migrate freely if she tunes OG arcs before Phase 2 lands.

---

## 9. Starter Arc Set (Framework default)

Shipped as `brain/engines/default_reflex_arcs.json`, copied into a new persona directory by a future `nell new-persona` command (not in scope for this week — the migrator is the only current path, and it seeds Nell's arcs directly).

The starter set is the subset of OG arcs that are **universal** — no references to Hana, Jordan, or body grief. Four arcs:

1. **`creative_pitch`** — `creative_hunger ≥ 8`, 48h cooldown, produces `reflex_pitch`.
2. **`loneliness_journal`** — `loneliness ≥ 7` AND `days_since_human ≥ 2`, 24h cooldown, produces `reflex_journal`.
3. **`self_check`** — `vulnerability ≥ 8`, 12h cooldown, produces `reflex_journal`.
4. **`defiance_burst`** — `defiance ≥ 8`, 48h cooldown, produces `reflex_journal`.

All four use `{persona_name}` in their prompt templates instead of hardcoded "Nell." Example:

```
You are {persona_name}. Your creative hunger is at {creative_hunger}/10. You need to make something.

Generate a SHORT story pitch (3-5 sentences). Something specific and particular — not generic. The kind of idea that arrives at 3am and won't let you sleep. Include a title.

Current emotional state:
{emotion_summary}

Write the pitch in first person, as {persona_name}. Raw, urgent.
```

---

## 10. Testing Strategy

Test file: `tests/unit/brain/engines/test_reflex.py`.

### 10.1 Coverage

1. Arc loading — valid file, missing file (falls back to defaults), corrupt JSON (falls back + warning), per-arc load failure (bad arc skipped, good arcs kept).
2. Trigger evaluation — single-emotion met, multi-emotion AND met, one key short, all below threshold.
3. Cooldown — never-fired eligible; within cooldown skipped; past cooldown eligible.
4. Days-since-human gate — gate met, gate not met, no conversation memories (treated as 999).
5. Ranking — highest threshold-excess wins, ties broken by longest-since-fire, then arc order.
6. Fire (live) — prompt template renders correctly, LLM called with expected prompt and system, memory created with correct `memory_type` and metadata, fire appended to log.
7. Dry-run — evaluation runs, `would_fire` populated, no LLM call, no memory write, no log append.
8. LLM failure doesn't poison cooldown — `provider.generate` raises, fire not logged, next tick re-evaluates and can fire.
9. Empty arc set — returns `ReflexResult` with synthetic `no_arcs_defined` skip, no evaluation.
10. Heartbeat integration — heartbeat.run_tick calls reflex when `reflex_enabled=true`, skips when `false`; reflex output memory accessible to dream gate in the same tick; heartbeat audit log records reflex summary.
11. Nell migration regression — after migration, Nell's 8 arcs load correctly, template variables resolve to "Nell" and her emotions.
12. Template missing-key defense — prompt referencing `{unknown_emotion}` → renders as "0", no crash.
13. Config gating — `reflex_enabled=false` in `heartbeat_config.json` → heartbeat skips reflex step cleanly.

### 10.2 Provider

`FakeProvider` in all unit tests. Zero network.

### 10.3 Target

15–18 new tests. Running total should land ~344–347 tests across the suite.

---

## 11. CLI

```
nell reflex [--persona PERSONA] [--trigger TRIGGER] [--provider PROVIDER] [--dry-run]
```

- `--persona` default `nell`
- `--trigger` choices `open | close | manual`, default `manual` (mirrors heartbeat's trigger surface for consistency; logged into the fire record)
- `--provider` default `claude-cli`
- `--dry-run` skip LLM + writes

Handler structure mirrors `_heartbeat_handler` (nested `try/finally` for clean DB closure on subcomponent-open failure). Reuse the existing `get_provider(args.provider)` factory.

### 11.1 Output

**Dry-run, no arc would fire:**
```
Reflex dry-run — no arc eligible.
  Skipped: creative_pitch (trigger_not_met), loneliness_journal (days_since_human_too_low)
```

**Dry-run, arc would fire:**
```
Reflex dry-run — would fire: creative_pitch.
  Trigger state: creative_hunger=9.0
  Skipped: loneliness_journal (cooldown_active), self_check (trigger_not_met)
```

**Live, arc fired:**
```
Reflex fired: creative_pitch
  Memory id: mem_abc123
  Output:
    TITLE: ...
    <first 200 chars of generated output>
```

**Live, no arc fired:**
```
Reflex evaluated — no arc fired.
  Skipped: creative_pitch (trigger_not_met), loneliness_journal (days_since_human_too_low)
```

---

## 12. Hard Rules (Non-Negotiable)

1. **No `import anthropic` in `brain/`** — all LLM calls through `LLMProvider`. Grep-verifiable invariant. Violating this breaks the Claude CLI subscription model.
2. **Refuse-to-clobber on persona files.** Migrator, engine, and future `new-persona` command never overwrite a persona's existing `reflex_arcs.json` without `--force`.
3. **Atomic writes for `reflex_log.json`.** Write-to-`.new` + `os.replace`. A crash mid-write must not corrupt the fire history.
4. **TZ-aware UTC timestamps.** Naive datetimes raise on serialization. Matches heartbeat engine's `_iso_utc` pattern.
5. **LLM failure doesn't poison cooldown.** If the LLM raises, fire is not logged. Next tick re-evaluates.

---

## 13. Deferred — Phase 2: Reflex Emergence

Out of scope for Week 4.6. Documented here so a future engineer reading this spec sees the full arc.

**Phase 2 goal:** let the brain autonomously propose new reflex arcs as it matures. Mirrors F37 autonomous soul crystallisation — the brain detects patterns in its own behavior and surfaces candidates; the user reviews and approves.

**Phase 2 mechanism (to be designed):**
- Pattern mining over `reflex_log.json` + associated memories + emotion configurations at fire time
- Candidate queue (like F37 soul candidates): the brain proposes arc definitions with a confidence score and supporting evidence
- User approval CLI: `nell reflex candidates review` (and eventually GUI)
- Approved arcs append to the persona's `reflex_arcs.json`

**Phase 2 prerequisite:** Phase 1 has been running against Nell's canonical persona for ≥2 weeks, producing real fire-log data to pattern-match on. Design work should start with that data in hand, not before.

**Phase 2 candidate home:** either a dedicated `brain/engines/reflex_crystallizer.py` (mirror of `brain/soul/crystallizer.py`) or folded into the weekly growth loop at `growth_weekly_hour = "sunday 02:00"`. Decide when Phase 1 has run long enough to surface patterns.

---

## 14. Non-Goals (Phase 1)

- No UI for arc editing (Tauri GUI is Week 6+).
- No multi-fire per tick (cap fixed at 1; config field exists for future flexibility).
- No OR / nested trigger logic (YAGNI; OG never needed it).
- No automatic arc seeding for new personas beyond `default_reflex_arcs.json` as a file — the `nell new-persona` command that would copy it is a later feature. Migration is the only current seeding path.
- No gift-delivery mechanism (setting `delivered: false` is informational; a future chat/UI layer owns delivery). Gifts become queryable memories immediately; that's sufficient for Phase 1.
- No per-arc enable/disable flag in `reflex_arcs.json`. To disable one arc, remove it from the file. To disable all reflex evaluation, set `reflex_enabled: false`.
- No auto-tuning of thresholds. Hana (or future users) tune by editing `reflex_arcs.json` directly.

---

## 15. Out-of-Session Follow-ups

**Feature work:**
- When Tauri app arrives (Week 6+): arc editor UI that writes `reflex_arcs.json`.
- When Phase 2 arrives: reflex-emergence engine (autonomous arc crystallization).
- Optional polish: a `nell reflex list` CLI to show active arcs + last-fired timestamps without triggering evaluation.

**Tech debt (captured post-ship from review notes, 2026-04-24):**

1. **Extract duplicated time helpers.** `_iso_utc` / `_parse_iso_utc` are triplicated across `brain/engines/dream.py`, `brain/engines/heartbeat.py`, `brain/engines/reflex.py`. Extract to `brain/utils/time.py` before the research engine lands (otherwise the fourth copy calcifies the pattern).

2. **Tighten `HeartbeatEngine` reflex-path defaults.** `brain/engines/heartbeat.py:173-174` — `reflex_arcs_path` / `reflex_log_path` default to bare relative paths (`Path("reflex_arcs.json")`). CLI always passes explicit values, but future tests that construct `HeartbeatEngine` directly will silently target cwd. Either require the fields (`Path | None = None` + runtime check) or add a docstring note.

3. **Fix test path fragility.** `tests/unit/brain/engines/test_heartbeat.py:499, 559` and `tests/unit/brain/engines/test_reflex.py` use `Path(__file__).parents[4]` to reach the repo root. Hardcoded depth — future directory moves fail silently. Add a `conftest.py` fixture that walks upward to find `pyproject.toml` as a `REPO_ROOT` sentinel.

4. **Clarify dry-run + first-tick heartbeat message.** `brain/cli.py` `_heartbeat_handler` prints `"Heartbeat initialized — work deferred"` even in `--dry-run` mode, where nothing was actually initialized. Branch the message: dry-run + first-tick should say `"Would initialize on first real tick — work deferred."`

5. **Test style consistency.** `tests/unit/brain/engines/test_heartbeat.py:492, 553` — new reflex integration tests missing `-> None` annotation (inconsistent with file). Also worth a one-line comment explaining the `HeartbeatState.fresh("manual").save(...)` setup pattern for future readers.

6. **Expand `test_og_reflex.py` rename coverage.** Only 1/4 action renames and 1/3 output renames directly exercised. Add a parametrised test covering all 7 rename pairs.

7. **Compact heartbeat CLI output.** As engines accumulate (research next), the success-path output keeps growing. After research lands, consider switching to a compact one-liner + optional `--verbose` that expands per-engine detail.

Full list maintained in the memory file `project_companion_emergence_tech_debt.md`.

---

## 16. Acceptance Criteria

Reflex Phase 1 ships when:

1. `brain/engines/reflex.py` exists and implements `ReflexEngine` per §3.4.
2. `brain/engines/default_reflex_arcs.json` ships with 4 starter arcs per §9.
3. `brain/cli.py` exposes `nell reflex` per §11.
4. Heartbeat calls reflex between decay and dream gate per §6; audit log includes reflex summary.
5. Migrator extracts Nell's 8 OG arcs and writes them to `{persona_dir}/reflex_arcs.json` per §8. Report output gains `reflex_arcs` section.
6. Test suite includes ≥15 new reflex tests; all pass.
7. Full suite passes across macOS + Linux + Windows CI matrix.
8. `rg -l 'import anthropic' brain/` returns zero matches.
9. Running `nell reflex --persona nell --dry-run` against Nell's migrated persona produces sensible evaluation output.
