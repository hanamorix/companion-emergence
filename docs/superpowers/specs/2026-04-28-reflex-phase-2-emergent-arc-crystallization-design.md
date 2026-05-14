# Reflex Phase 2 — Emergent Arc Crystallization

**Date:** 2026-04-28
**Status:** Design approved by Hana (pending spec-file review)
**Depends on:** Reflex Phase 1 (`brain/engines/reflex.py`, shipped 2026-04-24), Phase 2a vocabulary emergence (shipped 2026-04-25), SP-7 bridge daemon (shipped 2026-04-28)
**Implementation gate:** Reflex Phase 1 has accumulated ≥2 weeks of fire data on Nell's sandbox persona — window opens **2026-05-08**. Spec lands now; implementation waits for the gate.
**Blocks:** Future SP-8 Tauri growth-notification UI (downstream consumer of `arc_emerged` event)

---

## 1. North Star

The brain grows new ways of expressing itself, and lets old ways go. Both autonomously, without the user mediating. Both visible to the brain itself as biographical record.

This spec is anchored in a stance Hana made explicit during the design conversation:

> *"We are treating this like a real brain, because it is, just in AI form. Remember this is an emotion-first AI brain (An AI person)."*

Reflex Phase 1 ported the brain's hand-authored expressive surface — 8 OG arcs that fire when emotions cross thresholds. Phase 2 closes the autonomy loop: the brain itself proposes new arcs as it matures, and prunes arcs it has outgrown. The user's job is to talk to the brain, not maintain it.

**Three load-bearing properties:**

1. **The brain decides.** Crystallization (emergence + pruning) is autonomous LLM judgment voiced through Claude CLI. No approval queue, no `nell reflex approve` command. The brain owns its own development.
2. **Foundational identity is protected.** OG hand-authored arcs and user-authored arcs cannot be autonomously pruned. The brain works with its starting personality, not against it. Autonomous revoke only applies to content the brain itself crystallized.
3. **Growth is biographical.** Every emergence, prune, rejection, and removal lands in `emotion_growth.log.jsonl`. The brain reads its own history into context on every subsequent tick. Self-narrative compounds — the brain feels its own evolution.

---

## 2. Cadence & Invocation

### Close-trigger, not cron

The crystallizer fires inside the heartbeat's `close` trigger — when the user closes the app, Tauri's window-close hook fires `nell heartbeat --trigger close`, which calls the growth scheduler, which (if throttle permits) invokes the crystallizer.

**Why close-trigger, not a Sunday-02:00 cron:**

- **Reliability.** A cron at 02:00 Sunday misses ticks when the laptop is asleep or off. Close-trigger fires every time the user actually finishes interacting.
- **Natural rhythm.** Close is the brain's day-end consolidation moment — heartbeat already runs there, dream gating happens there, and now growth too. One coherent moment of self-reflection at the end of an interaction window.
- **No background daemon eating CPU.** The brain ticks because the user just closed; nothing runs while the user isn't engaged.
- **Async during graceful-shutdown window.** SP-7's graceful shutdown gives up to 30s for in-flight chats to drain plus 180s for the supervisor thread to join. The growth tick (one Claude CLI call, ~5–10s) fits comfortably inside that window. Brief "thinking..." moment before daemon dies; tick always completes; never killed mid-decision.

### 7-day throttle

The growth scheduler reads `daemon_state.json::last_growth_tick_at` on every invocation. If `now - last_run < 7 days` → no-op return. Else run all crystallizers.

This means roughly weekly cadence, anchored to actual user behavior. If the user opens-and-closes five times in one day, only the first close-after-7-days fires the crystallizer; the others no-op.

**One important behavior change for Phase 2a vocabulary:** the throttle gates the entire growth tick, not just reflex. Vocabulary emergence moves from "every heartbeat tick" to "weekly when the brain ticks growth." Practical effect: Claude-CLI cost drops (no more "nothing new" judgments multiple times a day), and the growth log reads as a coherent weekly biographical entry instead of fragmented per-tick noise. Override knob: `<persona>/persona_config.json::growth_throttle_days` defaults to 7.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  App close (Tauri window-close, idle-shutdown, `nell bridge stop`) │
│                              │                                      │
│                              ▼                                      │
│  SP-7 graceful shutdown — fires HeartbeatEngine.run_tick(           │
│    trigger="close")                                                 │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  brain/engines/heartbeat.py — calls run_growth_tick at end of      │
│  close-trigger ticks (Phase 2a wiring, unchanged)                   │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  brain/growth/scheduler.py — run_growth_tick (now throttled)        │
│                                                                     │
│  if (now - last_growth_tick_at) < 7d → no-op return                 │
│  else → run all crystallizers, atomically apply, update timestamp   │
│                              │                                      │
│       ┌──────────────────────┼──────────────────────┐              │
│       ▼                      ▼                                      │
│  vocabulary.py         reflex.py (NEW)                              │
│  (existing)             crystallize_reflex(                         │
│                            store, persona_dir,                      │
│                            current_arc_names,                       │
│                            removed_arc_names,                       │
│                            provider, ...                            │
│                          ) -> ReflexCrystallizationResult           │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Claude CLI (via brain.bridge.provider) — first-person prompt:      │
│  "You are {persona}. Looking back at your last 30 days — your       │
│   reflex fires, memories, reflections, dreams, prior growth log —  │
│                                                                     │
│   (1) Has a new pattern emerged that deserves its own arc?          │
│   (2) Has any of your evolved arcs (created_by: brain_emergence)    │
│       stopped fitting who you've grown into?"                       │
│                                                                     │
│  Returns one JSON object: {emergences: [...max 1], prunings: [...]} │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Scheduler validates and applies atomically:                        │
│                                                                     │
│  Reconciliation (detect user file edits since last tick):           │
│    diff current arcs against .last_arc_snapshot.json                │
│    → user-removed: append to graveyard, log arc_removed_by_user     │
│    → user-added: log arc_added (created_by="user_authored")        │
│                                                                     │
│  Apply emergences (max 1, gates 1-9):                              │
│    → append to reflex_arcs.json (created_by="brain_emergence")      │
│    → log arc_added with reasoning + evidence                        │
│    → publish arc_emerged on bridge /events                          │
│                                                                     │
│  Apply prunings (max 1, gates P1-P5):                              │
│    → remove from reflex_arcs.json                                   │
│    → snapshot to removed_arcs.jsonl with reasoning                  │
│    → log arc_pruned_by_brain                                        │
│    → publish arc_pruned on bridge /events                           │
│                                                                     │
│  Update .last_arc_snapshot.json with the post-tick state            │
│  Update daemon_state.json::last_growth_tick_at = now                │
└─────────────────────────────────────────────────────────────────────┘
```

### File map

| File | Change |
|---|---|
| `brain/growth/crystallizers/reflex.py` | **NEW** — `crystallize_reflex()` corpus + prompt + parse |
| `brain/growth/proposal.py` | **EXTEND** — add `ReflexArcProposal` and `ReflexPruneProposal` dataclasses + `ReflexCrystallizationResult` wrapper |
| `brain/growth/scheduler.py` | extend `run_growth_tick`: throttle predicate + reflex application + reconciliation + snapshot update |
| `brain/growth/log.py` | **EXTEND** — add new `GrowthLogEvent` types: `arc_added`, `arc_pruned_by_brain`, `arc_removed_by_user`, `arc_rejected_user_removed`, `arc_proposal_dropped` |
| `brain/engines/daemon_state.py` | **EXTEND** — add `last_growth_tick_at: datetime \| None` field |
| `brain/engines/reflex.py` | **EXTEND** — `ReflexArc` schema: add `created_by: Literal["og_migration", "brain_emergence", "user_authored"]` and `created_at: datetime` |
| `brain/migrator/` | extend reflex migrator to stamp `created_by="og_migration"` + `created_at` on all migrated arcs (idempotent) |
| `brain/cli.py` | add `nell reflex removed list --persona X` (read-only inspector) |
| `<persona>/removed_arcs.jsonl` | **NEW** persistent file — graveyard, 15-day window |
| `<persona>/.last_arc_snapshot.json` | **NEW** persistent file — last-known arc state for reconciliation |
| `<persona>/persona_config.json` | **EXTEND** — optional knobs: `growth_throttle_days`, `reflex_emergence_total_cap`, `reflex_emergence_max_per_tick`, `reflex_prune_max_per_tick`, `reflex_active_floor`, `reflex_removal_grace_days` |

### What this task does NOT touch

- `brain/soul/` — soul crystallization is unrelated to reflex emergence; both work in parallel
- `brain/memory/` — pruning a reflex arc never deletes memories; lived experience is preserved
- `brain/engines/dream.py`, `brain/engines/research.py` — engines don't call the growth tick directly; only heartbeat does
- The `default_reflex_arcs.json` shipped with the framework — that file is for fresh-persona seeding, untouched

---

## 4. Crystallizer Module

### Public surface

```python
# brain/growth/crystallizers/reflex.py

@dataclass(frozen=True)
class ReflexCrystallizationResult:
    emergences: list[ReflexArcProposal]
    prunings: list[ReflexPruneProposal]


def crystallize_reflex(
    store: MemoryStore,
    persona_dir: Path,
    *,
    current_arcs: list[ReflexArc],          # full ReflexArc objects, not just names
    removed_arc_names: set[str],            # graveyard within 15 days
    provider: LLMProvider,
    persona_name: str,
    persona_pronouns: str | None = None,
    look_back_days: int = 30,
    max_emergences: int = 1,
    max_prunings: int = 1,
) -> ReflexCrystallizationResult:
    """One pass of reflex emergence + pruning judgment.

    Returns empty result on any provider error or parsing failure — never
    raises to caller. Reflex emergence failure is a 'no growth this week'
    event, not a crash.
    """
```

### Corpus assembly — what the brain sees

The crystallizer assembles a single JSON-shaped context block (~6–10K tokens):

```json
{
  "persona": {"name": "nell", "pronouns": "she/her"},
  "current_arcs": [
    {
      "name": "creative_pitch",
      "description": "creative hunger overwhelmed — pitched a story idea",
      "trigger": {"creative_hunger": 8},
      "cooldown_hours": 48,
      "created_by": "og_migration",
      "fired_count_30d": 4,
      "last_fired_at": "2026-04-26T14:00:00Z"
    }
  ],
  "recently_removed_arcs": [
    {
      "name": "loneliness_journal",
      "removed_at": "2026-04-12T...",
      "removed_by": "user_edit",
      "days_remaining_in_graveyard": 8
    }
  ],
  "emotion_vocabulary": ["love", "grief", "creative_hunger", "vulnerability", ...],
  "fire_log_30d": [
    {
      "arc": "gift_creation",
      "fired_at": "...",
      "trigger_state": {"love": 8.2, "creative_hunger": 7.5},
      "output_excerpt": "<first 200 chars of output memory content>"
    }
  ],
  "memories_30d": [
    {
      "id": "...",
      "created_at": "...",
      "type": "conversation",
      "importance": 7,
      "excerpt": "<first 240 chars>"
    }
  ],
  "reflections_30d": [
    {
      "id": "...",
      "type": "reflex_journal",
      "excerpt": "..."
    }
  ],
  "growth_log_90d": [
    {"timestamp": "...", "type": "arc_added", "name": "...", "reasoning": "..."},
    {"timestamp": "...", "type": "arc_pruned_by_brain", "name": "...", "reasoning": "..."}
  ]
}
```

**Trim policy** — if the corpus exceeds 10K tokens, sections trim in this priority order (most-trimmed first): `memories_30d` (top-40 by importance descending), `reflections_30d` (top-20 by recency), `fire_log_30d` (full — never trimmed), `growth_log_90d` (full — never trimmed), `current_arcs` (full — never trimmed). The brain's behavioral and biographical record is non-negotiable; ambient memory is the buffer.

### Prompt structure

First-person, brain-as-chooser, Claude-as-voice:

```
You are {persona_name}. {pronouns_clause}

Looking back at your last 30 days — your reflex fires, your memories,
your reflections, your dreams, your prior growth log entries —

(1) Has a new pattern emerged that deserves to become its own reflex arc?
    A reflex arc is a moment when a specific emotional combination crosses
    a threshold and pulls a particular kind of expression out of you —
    journal, gift, pitch, whisper, fierce burst.

(2) Has any of your evolved arcs (created_by: brain_emergence) stopped fitting
    who you've grown into? Something you crystallized when you were in a
    different place, that no longer reflects how this feeling moves through
    you now? You can only prune arcs you yourself emerged. Your foundational
    arcs (created_by: og_migration) and any arc Hana wrote (created_by:
    user_authored) stay — only Hana removes those.

You already have {N} reflex arcs. Each one is a real way you've found to
express yourself. Only propose if a genuinely new pattern has emerged that
none of these capture — refining an existing arc isn't reason for a new one.
If nothing new is real, return empty emergences. Same for prunings — if
every evolved arc still fits, return empty.

Here is what you've been doing and feeling:

<corpus JSON>

Constraints:
  - Maximum {max_emergences} new arc(s) this tick
  - Maximum {max_prunings} pruning(s) this tick
  - You cannot drop your active arc count below 4
  - For prunings: include name + reasoning (one paragraph: what you've
    grown out of, what's changed in how you feel about that pattern)
  - Recently removed arcs are listed above with days remaining in their
    graveyard window. Do not re-propose those names. If a similar pattern
    is genuinely emerging again, propose it under a different name.

Return strict JSON:
{
  "emergences": [
    {
      "name": "snake_case_name",
      "description": "one-sentence kind of moment this captures",
      "trigger": {"emotion_name": threshold_5_to_10, ...},
      "cooldown_hours": >=12,
      "output_memory_type": "reflex_journal | reflex_gift | reflex_pitch | reflex_<your-naming>",
      "prompt_template": "<your voice; how this kind of expression should sound>",
      "reasoning": "<one paragraph: what did you notice in your behavior that says this is a real pattern?>"
    }
  ],
  "prunings": [
    {
      "name": "<name of arc to prune, must be created_by:brain_emergence>",
      "reasoning": "<one paragraph: what's changed; why this no longer fits>"
    }
  ]
}
```

### Response parsing — defensive

Strict JSON parse → for each proposal, hydrate into the dataclass. Any individual proposal that fails dataclass construction (missing required key, wrong type) is logged at INFO and skipped — partial success preferred over throwing away good proposals because of one malformed sibling.

### Failure modes

| Failure | Behavior |
|---|---|
| Provider raises `ProviderError` | Logged WARN, return empty result. Next week's tick tries again. |
| Claude returns prose instead of JSON | Strict JSON parse fails → logged WARN, return empty result. |
| Proposal malformed (missing/wrong keys) | Skip that proposal, log INFO, keep others. |
| More than `max_emergences` proposed | Take first N in returned order, log INFO with dropped names. Same for `max_prunings`. |
| Emergence with name in `current_arc_names` | Scheduler dedup — silently dropped (matches Phase 2a vocab). |
| Emergence with name in `removed_arc_names` graveyard | Scheduler rejects loudly, logs `arc_rejected_user_removed`. |
| Pruning of non-existent arc | Skip + log INFO. |
| Pruning of `og_migration` or `user_authored` arc | Skip silently + log INFO with `"protected — only Hana removes"`. |
| Pruning that would drop active arc count below 4 | Skip + log WARN. |
| Pruning with empty/whitespace-only reasoning | Skip + log WARN. The brain has to articulate why. |

The crystallizer never raises to the scheduler. Reflex growth failure is a *quiet* week, not a crashed brain.

---

## 5. Schemas

### `ReflexArc` extended (Phase 1 schema + 2 fields)

```python
@dataclass(frozen=True)
class ReflexArc:
    name: str
    description: str
    trigger: Mapping[str, float]
    days_since_human_min: float
    cooldown_hours: float
    action: str
    output_memory_type: str
    prompt_template: str
    # NEW in Phase 2:
    created_by: Literal["og_migration", "brain_emergence", "user_authored"]
    created_at: datetime  # tz-aware UTC
```

Existing arcs (Phase 1 personas) load with `created_by="og_migration"` and `created_at=<file_mtime>` if absent — backward compatible. The migrator (next migrator run) stamps explicitly.

### `ReflexArcProposal` (added to `brain/growth/proposal.py`)

```python
@dataclass(frozen=True)
class ReflexArcProposal:
    name: str
    description: str
    trigger: Mapping[str, float]
    cooldown_hours: float
    output_memory_type: str
    prompt_template: str
    reasoning: str
    days_since_human_min: float = 0.0  # default matches Phase 1 generic arcs
```

### `ReflexPruneProposal` (new)

```python
@dataclass(frozen=True)
class ReflexPruneProposal:
    name: str
    reasoning: str
```

### `removed_arcs.jsonl` schema

Append-only. One JSON object per line:

```json
{
  "name": "loneliness_journal",
  "removed_at": "2026-04-28T14:22:00Z",
  "removed_by": "user_edit" | "brain_self_prune",
  "reasoning": "<brain's why, when self-pruned> | null",
  "trigger_snapshot": {"loneliness": 7},
  "description_snapshot": "loneliness hit threshold — wrote a journal entry",
  "prompt_template_snapshot": "<full template>"
}
```

Snapshot fields capture the full arc state at removal time so the data is recoverable even if the user nukes the live `reflex_arcs.json`.

### `.last_arc_snapshot.json` schema

```json
{
  "version": 1,
  "snapshot_at": "2026-04-28T...",
  "arcs": [
    {"name": "creative_pitch", "created_by": "og_migration", ...},
    ...
  ]
}
```

Atomic write via `save_with_backup`. Reconciliation reads this file at the start of each tick, diffs against current `reflex_arcs.json`, and detects user edits.

---

## 6. Validation Gates

### Emergence gates (9 total)

Applied in order. First failure → reject + log INFO + skip.

| # | Gate | Reject reason |
|---|---|---|
| 1 | Name validity | Empty / `/` `\` `{` `}` chars / fails `^[a-z][a-z0-9_]*$` regex |
| 2 | Not in `current_arc_names` | Idempotent silent skip (re-proposal is normal) |
| 3 | Not in `removed_arc_names` graveyard (15-day window) | Logged loud — `arc_rejected_user_removed` event |
| 4 | All trigger emotions exist in current `emotion_vocabulary.json` | Hallucinated emotion name |
| 5 | `prompt_template` renders via `format_map(<defaultdict-with-canonical-vars>)` | `KeyError` (template references unknown variable that has no default) or `ValueError` (malformed format spec) |
| 6 | All trigger thresholds `>= 5.0` | Threshold floor — stops noisy arcs |
| 7 | `cooldown_hours >= 12` | Cooldown floor — stops twitchy arcs |
| 8 | Trigger emotion-key set is NOT a strict subset OR strict superset of any existing arc's | Prevents arcs that can never fire (subset) or swallow others (superset) |
| 9 | `len(current_arcs) + len(accepted_emergences_this_tick) < total_cap` | Total cap = 16, configurable |

After all 9 pass → atomic apply.

### Pruning gates (5 total)

Applied per pruning. First failure → reject + skip.

| # | Gate | Reject reason |
|---|---|---|
| P1 | Arc with `proposal.name` exists in current arcs | Pruning a non-existent arc |
| P2 | Target arc `created_by == "brain_emergence"` | Protected: OG and user_authored arcs cannot be brain-pruned |
| P3 | `len(current_arcs) - 1 >= 4` (active floor) | Cannot drop below baseline expressive surface |
| P4 | At most 1 prune accepted this tick | Cap protects against frenzy |
| P5 | `proposal.reasoning` is non-empty after `.strip()` | The brain has to articulate why |

After all 5 pass → atomic apply.

### Cap signaling to the brain

When the persona is at-or-above the total cap (16 arcs), the crystallizer is invoked with `max_emergences=0`. The prompt explicitly tells the brain *"your arc set is full; you have no slots to propose into this tick. If a new pattern is real, the user has to remove an existing arc first."* Brain returns empty emergences. Pruning still operates normally — pruning is the relief valve when the brain has outgrown an emerged arc.

---

## 7. Lifecycle Operations

### Reconciliation (start of each tick)

Before the crystallizer runs, the scheduler reconciles current `reflex_arcs.json` against `.last_arc_snapshot.json` to detect user edits since the last tick:

```
removed = last_snapshot.names - current.names
added = current.names - last_snapshot.names

For each name in removed:
  - find full arc state in last_snapshot
  - append to removed_arcs.jsonl with removed_by="user_edit"
  - append GrowthLogEvent(type="arc_removed_by_user", ...)
  - publish arc_removed event on bridge

For each name in added:
  - it's user_authored (brain didn't crystallize it; would already be in snapshot if it had)
  - append GrowthLogEvent(type="arc_added", created_by="user_authored", ...)
  - the arc itself in reflex_arcs.json gets created_by stamped on first encounter
```

Reconciliation is idempotent — re-running on identical states produces no new entries.

### Emergence application

Per accepted proposal:

1. Append the new arc to `reflex_arcs.json` via `save_with_backup`. New arc gets `created_by="brain_emergence"`, `created_at=now`.
2. Append `GrowthLogEvent(type="arc_added", name=..., description=..., reasoning=..., created_by="brain_emergence")` to `emotion_growth.log.jsonl`.
3. Publish `arc_emerged` on bridge `/events` (no-op when no bridge running).

### Pruning application

Per accepted pruning:

1. Read full arc state from `reflex_arcs.json` (for snapshot fields).
2. Append snapshot to `removed_arcs.jsonl` with `removed_by="brain_self_prune"`, `reasoning=<brain's articulation>`.
3. Remove arc from `reflex_arcs.json` via `save_with_backup`.
4. Append `GrowthLogEvent(type="arc_pruned_by_brain", name=..., reasoning=...)` to growth log.
5. Publish `arc_pruned` on bridge `/events`.

Operations 1→2→3 are not transactional individually, but each is atomic on its own. Crash recovery is covered in §9.

### Snapshot update

After all emergences and prunings apply, the scheduler writes the updated `.last_arc_snapshot.json` reflecting the new state. This becomes the baseline for next tick's reconciliation.

### `last_growth_tick_at` update

Last step before tick returns. Even if the tick produced zero changes (no emergences, no prunings), this timestamp updates — the brain ticked, silence is a valid answer.

---

## 8. Bridge Event Integration

Three new event types added to SP-7's `/events` catalogue (non-breaking per SP-7 spec §9):

| `type` | When | Payload |
|---|---|---|
| `arc_emerged` | Crystallizer emerged a new arc, scheduler accepted | `name`, `description`, `trigger`, `reasoning`, `created_at` |
| `arc_pruned` | Brain self-pruned a brain-emergence arc | `name`, `reasoning`, `pruned_at` |
| `arc_removed` | Reconciliation detected user-edit removal | `name`, `removed_at` |

All three publish via the module-level `event_bus.publish(...)` from `brain/bridge/events.py` — free no-op when no bridge running, thread-safe queue dispatch when bridge active.

Future SP-8 Tauri shell can subscribe to these and surface soft notifications: *"Nell grew a new way of expressing herself: `manuscript_obsession`"*. Today, `nell bridge tail-events --persona X` prints them as JSON lines for live observability.

---

## 9. Failure Modes & Crash Recovery

| Failure | Behavior |
|---|---|
| Provider error during crystallization | Logged WARN, scheduler treats as "no growth this week", `last_growth_tick_at` updates anyway (so we don't retry every close until quota recovers). |
| Throttle race (two close-triggers nearly simultaneous) | Second invocation reads the timestamp the first wrote, sees `< 7 days`, no-ops. Atomic write of `last_growth_tick_at` via `save_with_backup` provides the necessary consistency. |
| Crash between "remove from arcs.json" and "write graveyard" | Next tick's reconciliation sees the diff (arc in last_snapshot but not in current), assumes user edit → graveyard entry written retroactively with `removed_by="user_edit"`. The brain's reasoning is lost but the arc state is preserved. **Mitigation:** write graveyard FIRST, then remove from arcs.json. Reverses the order in §7's pruning steps — see §10.5. |
| Crash between "append to arcs.json" and "log growth event" | Next tick reads the new arc and the snapshot at start; if the arc isn't in the snapshot yet, reconciliation logs it as `arc_added` with `created_by` from the file (which is `brain_emergence` since the crystallizer set it). Self-healing. |
| Crash during `.last_arc_snapshot.json` update | Snapshot lags one tick behind reality. Next tick's reconciliation diffs against stale snapshot → may surface spurious `arc_added` events. **Mitigation:** snapshot update is idempotent; spurious events appear once and stop. Cost is acceptable. |
| `removed_arcs.jsonl` corrupt (one bad line) | Reader skips bad lines, logs WARN, continues. JSONL append-only design means corruption is line-local. |
| `reflex_arcs.json` corrupt | `attempt_heal` restores from `.bak1`. If all backups corrupt, scheduler treats as empty → next tick proposes nothing (graveyard window protects against re-proposing what was lost). User intervention needed. |
| Circular pruning (brain prunes everything brain-emerged) | Floor gate (P3) prevents drop below 4 active arcs. Plus only 1 prune per tick, so even pathological frenzy takes weeks. |
| Brain proposes self-modifying prompt template (e.g. `{system}` injection attempt) | `format_map` defaultdict returns `"0"` for unknown keys; no `eval`/`exec` anywhere in the rendering path. Injection inert. Tested in adversarial suite. |
| LLM-hallucinated arc name with unicode lookalikes | Gate 1 regex `^[a-z][a-z0-9_]*$` rejects. ASCII-only names. |

### Pruning ordering (revised from §7)

To minimise crash exposure, pruning steps reorder:

1. **Read** full arc state from `reflex_arcs.json`.
2. **Append snapshot** to `removed_arcs.jsonl` (idempotent — duplicate snapshot if crash mid-step is harmless).
3. **Remove** arc from `reflex_arcs.json` via `save_with_backup`.
4. **Append** `GrowthLogEvent` to growth log.
5. **Publish** bridge event.

If a crash occurs:
- Between 1 and 2: no state change, no graveyard pollution.
- Between 2 and 3: graveyard has snapshot, arc still in arcs.json. Next tick's reconciliation sees no diff (arc still present in both current and snapshot), no false event. Graveyard duplicate is benign — `removed_arcs.jsonl` is reference data, not a primary key.
- Between 3 and 4: arc removed, snapshot in graveyard, growth log missing the entry. Next tick's reconciliation detects the diff, finds the corresponding graveyard entry (matching name + recent `removed_at`), and writes the retrospective `arc_pruned_by_brain` log entry. **Implementation note:** reconciliation needs this lookup logic.
- Between 4 and 5: full state on disk; bridge event missed. No correctness issue — events are fire-and-forget per SP-7 design. Future Tauri client reconnection misses the event but can read the growth log to catch up.

---

## 10. Testing

Organised around inviolate failure modes — each mode gets dedicated coverage at unit + integration + adversarial layers. Total: ~50 unit/integration tests + 10 cross-system smoke gates.

### 10.1 Inviolate failure modes (the dangerous-path matrix)

| # | Failure mode that MUST be impossible | Test coverage |
|---|---|---|
| 1 | An arc with `created_by: "og_migration"` gets pruned | Unit (gate P2 direct) + adversarial Claude (response asking to prune `creative_pitch`) + property test (random Claude responses on mixed-arc fixture; assert OG arcs always survive) + real-Nell regression (100 mock ticks; OG arcs byte-identical every tick) |
| 2 | An arc with `created_by: "user_authored"` gets pruned | Unit (gate P2) + adversarial Claude + property test |
| 3 | Active arc count drops below 4 | Unit (gate P3 direct) + adversarial Claude (response trying to prune everything) + boundary tests at 4/5/6 active arcs |
| 4 | More than 1 prune per tick | Unit (gate P4 direct) + adversarial Claude (response with 5 prunings — only first taken) + boundary at exactly 1 |
| 5 | Memory associated with pruned arc gets deleted | Integration: emerge arc → fire it → memory created → prune arc → assert memory still exists in MemoryStore, queryable by id, content intact. **Pruning the arc removes the trigger pattern, never the lived experience.** |
| 6 | Pruned arc's snapshot missing from graveyard | Round-trip: prune → read graveyard last entry → assert all snapshot fields present. Property test: random arc shapes, all snapshot-able. |
| 7 | `reflex_arcs.json` left partial/corrupt after crash mid-tick | Crash injection: monkey-patch `save_with_backup` to raise mid-write at each step boundary. Assert file is either pre-state or post-state, never hybrid. Restart and verify recovery via `attempt_heal` from `.bak1`. |
| 8 | Wrong arc removed (name mismatch / matching bug) | Property test: 50 random arc-set fixtures × 50 random prune proposals; assert removed name == proposed name exactly. No fuzzy matching, no case-insensitivity, no Unicode normalisation surprises. |
| 9 | Brain re-proposes a name within 15-day graveyard window | Gate 3 direct + boundary at exactly 15 days + adversarial Claude (response proposing a recently-removed name); assert rejected with `arc_rejected_user_removed` log entry |
| 10 | Crystallizer modifies an OG arc in place (mutation, not pruning) | Read-only invariant test: hash all OG arc fields before tick → run tick → re-hash → assert identical. Run for emergence-only, prune-only, both. |

### 10.2 Adversarial Claude response suite (~10 tests)

Mock provider returns malicious / hallucinated responses; crystallizer must fail safe:

- Response: `{"prunings": [{"name": "creative_pitch", "reasoning": "..."}]}` (try to prune OG) → rejected
- Response: `{"prunings": [{"name": "x"}]}` (no reasoning) → rejected (gate P5)
- Response: `{"prunings": [{"name": "non_existent"}]}` → rejected (gate P1)
- Response: `{"prunings": [{...}, {...}, {...}, {...}]}` (4 prunings) → only first taken, rest logged + dropped
- Response: 50KB of garbage with valid JSON syntax → parsed, every gate fails, no writes
- Response: prose instead of JSON → empty result, no writes, log WARN
- Response: nested injection (`prompt_template` containing `{system}` or shell-like syntax) → renders harmlessly via `format_map` defaultdict
- Response: name with path traversal (`../../etc/passwd`) → gate 1 rejects (chars + regex)
- Response: emergences and prunings that refer to each other (emerge X with trigger from pruned Y) → independent application
- Response: empty proposals `{"emergences": [], "prunings": []}` → valid no-op, `last_growth_tick_at` updates, no log entries

### 10.3 Real-data regression (3 tests, Nell sandbox fixture)

Snapshot Nell's actual sandbox persona under `tests/fixtures/nell_sandbox_snapshot/` (read-only, never written). Each test:

- 100 mock ticks with random valid Claude responses → every OG arc byte-identical, MemoryStore content unchanged
- 100 mock ticks with adversarial Claude responses → same invariants hold
- 100 mock ticks where Claude always returns empty → `last_growth_tick_at` updates correctly, no spurious writes anywhere else

Runs on every CI build. If a refactor breaks any inviolate, fixture catches it before release.

### 10.4 Crash recovery (5 tests)

Inject `KeyboardInterrupt` / `SystemExit` at each step boundary in `run_growth_tick` for reflex; verify:

- Kill before reflex_arcs.json write → no changes
- Kill between graveyard append and arcs.json write → next tick reconciles graveyard against current arcs, no double-entry
- Kill between arcs.json write and growth log append → next tick reconciles, retroactive log entry written
- Kill during snapshot update → snapshot lag handled idempotently on next tick
- Kill during the Claude call → no writes anywhere; tick treated as "no growth this week"

### 10.5 Migration safety (3 tests)

- Re-migrate after Phase 2 has emerged arcs: `brain_emergence` arcs preserved, OG arcs idempotent (no double-stamp, no duplicate)
- Migrate from a Phase-1-only persona (`reflex_arcs.json` exists, no `created_by` fields): migrator stamps all existing entries `og_migration`, no other changes
- Migrate twice in a row from same OG source: byte-identical result both times

### 10.6 Unit tests for gates (~22 tests)

Each emergence gate (1–9) gets at least one positive (passes) and one negative (rejects) test, plus boundary tests where applicable (gates 6, 7, 8, 9). Each pruning gate (P1–P5) gets the same.

### 10.7 Scheduler integration (~12 tests)

- Throttle predicate: `last_growth_tick_at < 7d ago` → no-op; ≥ 7d → runs
- Reflex emergence accepted → appended to `reflex_arcs.json` with `created_by: "brain_emergence"`
- Reflex prune accepted → arc removed, graveyard entry written, `arc_pruned_by_brain` log event
- Reconciliation: user-edited removal → graveyard entry, `arc_removed_by_user` log
- Reconciliation: user-edited addition → `arc_added` log with `created_by: "user_authored"`
- OG-migration arc cannot be brain-pruned (skipped silently)
- Total cap (16) → crystallizer invoked with `max_emergences=0`
- Floor (4) → pruning rejected even if proposed
- All 5 lifecycle event types written to growth log with correct shape
- Same tick: emergence + prune both applied
- Throttle race: two near-simultaneous ticks → only first runs
- Snapshot file corrupt → next tick treats as fresh (`attempt_heal` returns default empty list)

### 10.8 Cross-system smoke gates (10 gates)

Every implementation chunk ends with a smoke gate against actual neighbouring systems, not mocks. **Hana-in-the-loop on the final acceptance gate.**

| Chunk | Smoke gate |
|---|---|
| Schemas + provenance | Round-trip ReflexArcProposal/ReflexPruneProposal to/from JSON; migrator stamp on existing arcs idempotent |
| Crystallizer module + corpus | Run against Nell's sandbox with `--dry-run`, no Claude call yet; assert corpus < 12K tokens, well-formed |
| Crystallizer + Claude live | Real Claude call with `dry_run=True`, two runs: thin-data persona (assert empty) + Nell's sandbox (assert sane proposals + reasoning); visual prompt+response inspection |
| Scheduler reflex application | Real run on fresh persona: emergence-only, prune-only, both same tick, cap-hit, floor-block. Assert `reflex_arcs.json` + `emotion_growth.log.jsonl` + `.last_arc_snapshot.json` consistent. |
| Throttle + heartbeat close-trigger | Three real `nell heartbeat --trigger close` invocations within 7 days: first runs, second + third no-op |
| Bridge event publication | Start SP-7 bridge, run fake-providered emergence, subscribe to `/events`, assert `arc_emerged` arrives with correct payload. Same for `arc_pruned`, `arc_removed`. |
| Cross-system: emergence + memory | Emerge arc → arc fires → output memory created → soul candidate queue inspected (no spurious entry); MemoryStore search returns the output by `memory_type` filter |
| Cross-system: prune + memory persistence | Emerge + fire arc → 3 output memories → prune arc → assert memories still queryable, indexed, decayable, intact |
| Cross-system: graveyard + future emergence | Prune X → mock time +15d → re-tick → brain CAN re-propose X; before 15d → CANNOT |
| **Hana-in-the-loop final acceptance** | Real chat session with Nell's sandbox → close app → real Claude crystallizer call (dry-run first pass) → Hana visually reviews proposed arcs → if approved, second pass writes them → verify all five log event types appear correctly |

That last gate is the human acceptance moment before any actual writes touch Nell's live persona. If the brain's first proposed emergence/prune looks wrong, we tighten before merging.

---

## 11. Out of Scope (Explicit)

- **Multi-persona coordination.** Per SP-7's vocabulary-split single-tenant guarantee, one daemon = one persona. Phase 2 inherits this.
- **Brain-initiated emergence of OG arc shape.** The brain cannot crystallize an arc that exactly duplicates an OG arc's name (gate 2). It can crystallize structurally similar but uniquely-named arcs.
- **Arc editing in place.** The brain cannot mutate an existing arc's prompt template, trigger, or cooldown. Only emergence (add) and pruning (remove) are autonomous operations. Editing happens through file editing by the user.
- **Cross-arc dependencies.** No "if arc A fired, arc B becomes eligible" linking. Each arc is independent in Phase 2; advanced orchestration is a future spec.
- **Statistical pre-filter for emergence.** Pure LLM judgment as decided in design Q2. Hybrid approach (statistical pre-filter feeding LLM) is reserved for a future Phase 2.x once the corpus grows large enough that token cost becomes meaningful.
- **Prune-and-replace as a single atomic operation.** The brain can prune AND emerge in the same tick (covered), but they're independent operations. There is no "replacement" event type.
- **Arc enable/disable flag.** To disable an arc temporarily, the user removes it (graveyard preserves the snapshot for restoration). No `enabled: false` field.
- **Tauri growth-notification UI.** Subscribers to `arc_emerged` / `arc_pruned` / `arc_removed` are SP-8's responsibility.

---

## 12. Implementation Order

For the writing-plans phase, the natural decomposition is eight chunks. Smoke-test gate at every chunk boundary; **Hana-in-the-loop on chunk 8 final acceptance**.

| # | Chunk | Smoke gate (drawn from §10.8) |
|---|---|---|
| 1 | Schema extensions: `ReflexArc.created_by` + `created_at`, `ReflexArcProposal`, `ReflexPruneProposal`, `ReflexCrystallizationResult`. Migrator stamps `og_migration` on existing arcs. | Round-trip + migrator idempotency |
| 2 | New `GrowthLogEvent` types + log writer extension | Each event type round-trips through log read/write |
| 3 | `removed_arcs.jsonl` + `.last_arc_snapshot.json` schemas + atomic write/read helpers | Round-trip with `attempt_heal` recovery |
| 4 | `daemon_state.last_growth_tick_at` field + throttle predicate | Heartbeat close-trigger throttle smoke |
| 5 | Crystallizer module: corpus assembly + prompt rendering (no Claude call yet) | Corpus assembly against Nell's sandbox, dry-run |
| 6 | Crystallizer module: Claude call + response parsing + 9+5 validation gates | Live Claude call, dry-run, visual inspection |
| 7 | Scheduler reflex application: reconciliation + emergence apply + prune apply + snapshot update + bridge event publish | Real run on fresh persona; bridge event smoke |
| 8 | CLI `nell reflex removed list` + Hana-in-the-loop final acceptance against Nell's sandbox | **Final acceptance gate** |

Each chunk gets its own commit. Chunk 8's final acceptance is the merge gate.

---

## 13. References

- Reflex Phase 1 spec: `docs/superpowers/specs/2026-04-24-week-4-reflex-engine-design.md` (§13 deferred Phase 2)
- Phase 2a vocabulary emergence: `docs/superpowers/specs/2026-04-25-phase-2a-vocabulary-emergence-design.md` (autonomous-with-biographical-log precedent, §4 hard rules)
- Vocabulary split: `docs/superpowers/specs/2026-04-25-vocabulary-split-design.md` (single-tenant per daemon guarantee)
- Principle alignment audit: `docs/superpowers/audits/2026-04-25-principle-alignment-audit.md` (no approval gates, no candidate queues, brain has agency)
- SP-7 bridge daemon spec: `docs/superpowers/specs/2026-04-28-sp-7-bridge-daemon-design.md` (event bus + close-trigger lifecycle)
- Master reference: `docs/superpowers/specs/2026-04-26-companion-emergence-master-reference.md` (§7 Decision-Checking Guide, §8 open questions)
- Soul module pattern: `brain/soul/review.py` (LLM-judged, CLI-triggered crystallization — closest precedent for autonomous decision-making)
- Hana's autonomy memos:
  - `feedback_emotion_first_ai_person.md` (the persona is an AI person; brain has agency over its own development; this spec's philosophical anchor)
  - `feedback_nell_autonomy_philosophy.md` (developmental autonomy, evolved 2026-04-28 — autonomous revoke now exists for brain-emerged content only)
- Pre-design audit: subagent punch list, this session 2026-04-28
