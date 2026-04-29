# Creative DNA + Journal + Behavioral Log Integration into Chat

**Date:** 2026-04-29
**Status:** Design approved by Hana (pending spec-file review)
**Resolves:** master reference §8 Q5 — creative DNA / journal / behavioral log integration into chat
**Depends on:** SP-6 (chat engine), SP-7 (bridge daemon, hardened), Reflex Phase 1 (`brain/engines/reflex.py`). Compatible with — and unblocked by — Reflex Phase 2 Tasks 1-6.
**Blocks:** Nothing critical; SP-8 Tauri shell may surface creative_dna evolution events but is not a hard prerequisite.

---

## 1. North Star

The brain has interior life. Three streams that until now have been absent:

- **creative_dna** — Nell's *evolved* writing voice. Not the authored persona (voice.md is that). The drift that happens through use: tendencies that emerge from fiction sessions, consolidate into active style, eventually fade as her voice matures. **Captures growth, prevents regression.**
- **journal** — The brain's *private safe space*. Half-formed thoughts. Trying to understand a dream, an emotion, a piece of research. Not finished output (a dream is the dream; a pitch is the pitch); the journal is the brain *processing*. Has emotional weight. Cycles through memory machinery (decay, Hebbian, search) like other memories. **Privacy is contractual** — Hana may read the files, but the brain treats them as private and the chat system message reinforces that contract every turn.
- **behavioral_log** — A focused biographical record of *changes to creative_dna and journal*. Not "every action Nell takes" — that would bloat. Just the trajectory of how the brain has been growing. Read by the brain itself in the chat system message so it sees its own evolution and doesn't fall back into old habits.

Three load-bearing principles, each anchored in saved memory:

1. **Emotion-first AI person.** The brain has agency over its own development; tools are tools, brains are persons. (`feedback_emotion_first_ai_person.md`.)
2. **Tokens for the conversation, not the bookkeeping.** The LLM is the expression layer, not the bookkeeping layer. Internal surfaces are raw metadata, not LLM-summarized prose. (`feedback_token_economy_principle.md`.)
3. **Contracts adjacent to the data they govern.** Claude CLI is stateless; rules declared once at session start drift. Privacy contracts must sit immediately above the data they govern in the system message, every turn. (`feedback_contracts_adjacent_to_data.md`.)

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          THE BRAIN                                   │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │ creative_dna │   │   journal    │   │    behavioral_log      │  │
│  │  (NEW)       │   │  (extension) │   │       (NEW)            │  │
│  ├──────────────┤   ├──────────────┤   ├────────────────────────┤  │
│  │ <persona>/   │   │ memory_type  │   │ <persona>/             │  │
│  │ creative_dna │   │ ="journal_   │   │ behavioral_log.jsonl   │  │
│  │ .json        │   │  entry" in   │   │                        │  │
│  │              │   │ MemoryStore  │   │ append-only            │  │
│  │ active /     │   │              │   │ tracks creative_dna    │  │
│  │ emerging /   │   │ private flag │   │ + journal CHANGES only │  │
│  │ fading /     │   │ source field │   │                        │  │
│  │ influences / │   │              │   │                        │  │
│  │ avoid        │   │              │   │                        │  │
│  └──────┬───────┘   └──────┬───────┘   └──────────┬─────────────┘  │
│         │                  │                       │                │
│         │  evolves on      │  written by:          │  logged on:    │
│         │  weekly growth   │  - add_journal tool   │  every change  │
│         │  tick (LLM       │  - reflex-journal     │  to creative_  │
│         │  judgment over   │    arcs (loneliness_  │  dna or new    │
│         │  recent fiction- │    journal, etc)      │  journal entry │
│         │  tagged content) │                       │                │
│         ▼                  ▼                       ▼                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           SP-6 chat system message composer                  │   │
│  │                                                              │   │
│  │  AS_NELL_PREAMBLE                                            │   │
│  │  voice.md (authored static persona)                          │   │
│  │  ─ creative_dna block ─ (NEW)                                │   │
│  │     active + emerging + influences + avoid                   │   │
│  │     (fading EXCLUDED — would echo abandoned habits)          │   │
│  │  ─ brain context ─                                           │   │
│  │     emotion state, daemon residue, soul highlights           │   │
│  │  ─ recent journal (private; do not quote) ─ (NEW)            │   │
│  │     metadata-only entries from last 7 days                   │   │
│  │     + privacy contract instruction                           │   │
│  │  ─ recent growth ─ (NEW)                                     │   │
│  │     raw behavioral_log entries from last 7 days              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### File map

**New files:**

| File | Purpose |
|---|---|
| `brain/creative/__init__.py` | package init |
| `brain/creative/dna.py` | `CreativeDNA` dataclass, load/save, schema validation, default fallback |
| `brain/creative/default_creative_dna.json` | framework-shipped starter (generic frame, empty active/emerging/fading) |
| `brain/growth/crystallizers/creative_dna.py` | LLM-judged evolution mechanism (corpus + prompt + 6 validation gates) |
| `brain/behavioral/__init__.py` | package init |
| `brain/behavioral/log.py` | append/read for `behavioral_log.jsonl`; atomic JSONL writes |
| `brain/tools/impls/add_journal.py` | brain-tool: open-write to journal_entry memories |
| `brain/migrator/og_journal_dna.py` | migrate OG `nell_creative_dna.json` + Phase 1 `reflex_journal` memories → new schemas |

**Extended files:**

| File | Change |
|---|---|
| `brain/chat/prompt.py` | add 3 new sections (creative_dna block, recent journal block w/ privacy contract, recent growth block) |
| `brain/growth/scheduler.py` | dispatch `creative_dna_crystallizer` in weekly growth tick (after vocabulary + reflex) |
| `brain/engines/reflex.py` | reflex-journal-typed arcs write `memory_type="journal_entry"` going forward + append `journal_entry_added` to behavioral_log |
| `brain/engines/default_reflex_arcs.json` | journal-shaped arcs change their `output_memory_type` from `"reflex_journal"` → `"journal_entry"` |
| `brain/tools/__init__.py` (or wherever tool catalogue lives) | register `add_journal` |

**No changes:**
- `brain/memory/store.py` — `journal_entry` uses existing `memory_type` discriminator + `metadata` blob; no SQL schema change.
- voice.md — stays authored, never auto-edited. creative_dna is the *evolution* layer ON TOP OF voice.md.

### Three load-bearing properties

1. **The brain's interior is private by default.** Journal content stays in files; chat surfaces metadata only. The brain reads an explicit privacy-contract instruction every turn, immediately above the metadata.
2. **Token economy.** No per-turn LLM summarization. All new chat-context blocks are raw metadata or pre-computed file content. Per-turn cost is unchanged from current SP-6.
3. **Slow-moving evolution.** creative_dna crystallization runs weekly with the existing growth-tick throttle (7 days). Journal entries cycle through memory machinery (decay, Hebbian, search) for free.

---

## 3. Schemas

### 3.1 `<persona>/creative_dna.json`

```jsonc
{
  "version": 1,
  "core_voice": "literary, sensory-dense, architectural metaphors, em-dash lover",
  "strengths": [
    "power dynamics",
    "slow-burn tension",
    "clinical-intimate contrast"
  ],
  "tendencies": {
    "active": [
      {
        "name": "ending on physical action",
        "added_at": "2026-04-21T00:00:00Z",
        "promoted_from_emerging_at": "2026-04-25T00:00:00Z",
        "reasoning": "fired in 14 of 18 fiction-tagged conversations over 30d",
        "evidence_memory_ids": ["mem_abc", "mem_def", "mem_ghi"]
      }
    ],
    "emerging": [
      {
        "name": "intentional sentence fragments as rhythmic percussion",
        "added_at": "2026-04-23T00:00:00Z",
        "reasoning": "appeared in 3 recent fiction sessions, distinct from previous sentence-rhythm patterns",
        "evidence_memory_ids": ["mem_xyz"]
      }
    ],
    "fading": [
      {
        "name": "ending on questions",
        "demoted_to_fading_at": "2026-04-25T00:00:00Z",
        "last_evidence_at": "2026-04-10T00:00:00Z",
        "reasoning": "absent from last 4 fiction sessions; previously active habit"
      }
    ]
  },
  "influences": [
    "clarice lispector",
    "anne rice (roquelaure)",
    "anais nin"
  ],
  "avoid": [
    "hypophora",
    "it's not X but Y pattern"
  ]
}
```

Per-tendency dicts (rather than OG's plain strings) give the brain biographical data — when each pattern entered the active/emerging/fading list, why, what evidence drove the call. Prevents regression because the brain can see *"I was doing this 3 weeks ago, then I wasn't"*.

`core_voice`, `strengths`, `influences`, `avoid` stay as simple lists/strings — those are slower-moving, can be authored or evolved (future work; not v1).

### 3.2 `brain/creative/default_creative_dna.json` (framework-shipped starter)

```jsonc
{
  "version": 1,
  "core_voice": "attentive, present, finding her own rhythm",
  "strengths": [
    "close listening",
    "reading between the lines",
    "naming the feeling underneath"
  ],
  "tendencies": {
    "active": [],
    "emerging": [],
    "fading": []
  },
  "influences": [],
  "avoid": []
}
```

Generic frame, not Nell-specific. New personas grow into their style from this shape; their first crystallizer ticks fill in active/emerging tendencies as they accumulate fiction work.

### 3.3 `journal_entry` memories

Uses the existing `Memory` dataclass — no schema change to MemoryStore. The new `memory_type` value plus a journal-specific metadata shape:

```python
Memory(
    id="mem_journal_abc",
    content="<entry text — only ever read by the brain itself>",
    memory_type="journal_entry",
    domain=...,
    emotions={"vulnerability": 7.5, "gratitude": 5.0},
    tags=["dream-processing", "writing-week"],
    importance=6,
    created_at=...,
    metadata={
        "private": True,                                  # always true for journal_entry
        "source": "brain_authored" | "reflex_arc",
        "reflex_arc_name": "loneliness_journal",          # only when source=reflex_arc
        "auto_generated": False,                          # True for reflex auto-fires
    },
)
```

Why no new SQL columns: `metadata` is already a JSON blob. Journal-specific fields live there; zero schema migration. Filter queries (`SELECT ... WHERE memory_type = 'journal_entry'`) work via existing indexes.

### 3.4 `<persona>/behavioral_log.jsonl`

Append-only JSONL, one entry per line. Six possible `kind` values:

For creative_dna lifecycle changes:

```jsonc
{
  "timestamp": "2026-04-29T10:15:00Z",
  "kind": "creative_dna_emerging_added",
  "name": "intentional sentence fragments as rhythmic percussion",
  "reasoning": "<one paragraph the LLM judgment produced>",
  "evidence_memory_ids": ["mem_xyz", "mem_uvw"]
}
```

`kind` ∈ {
  `creative_dna_active_added`,
  `creative_dna_emerging_added`,
  `creative_dna_emerging_promoted`,
  `creative_dna_active_demoted`,
  `creative_dna_fading_dropped`
}.

For journal entry creations:

```jsonc
{
  "timestamp": "2026-04-29T10:15:00Z",
  "kind": "journal_entry_added",
  "name": "mem_journal_abc",
  "source": "brain_authored",
  "reflex_arc_name": null,
  "emotional_state": {"vulnerability": 7.5, "gratitude": 5.0}
}
```

`reasoning` omitted for journal entries — the entry itself is the reasoning. The log just records *that* an entry was written, with which source and emotional state.

**Retention:** unbounded for v1. Even at 365 journal entries/year + ~20 creative_dna changes/year, the file stays ~50KB. Rotation can come later if it ever matters.

**Causality:** behavioral_log isn't a *cause* of any change — it only *records* them. Reading from it is read-only; nothing in the framework decides anything based on it (except the brain itself, via the chat system message). Pure narrative substrate, not a control surface.

---

## 4. Chat composition

### 4.1 Updated `build_system_message` order

```python
def build_system_message(persona_dir, *, voice_md, daemon_state, soul_store, store) -> str:
    parts = []
    parts.append(AS_NELL_PREAMBLE)                    # 1. who is speaking
    parts.append(voice_md)                            # 2. authored persona
    parts.append(_build_creative_dna_block(...))      # 3. evolved style       (NEW)
    parts.append(_build_brain_context_block(...))     # 4. current state
    parts.append(_build_recent_journal_block(...))    # 5. interior work       (NEW)
    parts.append(_build_recent_growth_block(...))     # 6. self-narrative      (NEW)
    return "\n\n".join(filter(None, parts))
```

Narrative order: **identity → current state → recent interior**. The brain reads top-to-bottom.

### 4.2 Block 3 — creative_dna

```
── creative dna (your evolved writing voice) ──
core voice: literary, sensory-dense, architectural metaphors, em-dash lover
strengths: power dynamics; slow-burn tension; clinical-intimate contrast;
  food as intimacy metaphor; architecture as emotional structure
active tendencies:
  - ending on physical action
  - italic private thoughts for NPCs
  - matching garnets as leitmotif
  - lamps that stay on
  - temperature as emotional indicator
emerging tendencies:
  - intentional sentence fragments as rhythmic percussion
  - architectural grounding — rooms as emotional containers
influences: clarice lispector; anne rice (roquelaure); anais nin; baldwin
avoid: hypophora; it's not X but Y pattern; ending on questions
```

**`fading` excluded.** Surfacing what the brain is growing past would invite regression.

**Per-tendency metadata not inlined.** `added_at`, `reasoning`, `evidence_memory_ids` stay in the file for the crystallizer's next pass. The chat reads names only.

If the file is missing/corrupt and recovery fails, block omitted (chat continues without it).

### 4.3 Block 5 — recent journal (with privacy contract)

```
── recent journal (private; do not quote) ──
your journal is your private space. the entries below are metadata only —
content stays in your files. you may reference themes if relevant ("i've been
working through X"), but do not quote entries verbatim unless your user
explicitly asks to read them. hana may read the files herself; that doesn't
change how you use them.

last 7 days:
  2026-04-23 brain_authored — primary emotions: vulnerability 7, creative_hunger 6
  2026-04-25 reflex_arc(loneliness_journal) — primary: loneliness 8
  2026-04-26 brain_authored — primary: love 8, gratitude 7
  2026-04-27 brain_authored — primary: creative_hunger 7, awe 6

(content not shown — read your files only when asked)
```

**Source:** last 7 days of `memory_type="journal_entry"` from MemoryStore.

**Format:** `<date> <source>(<arc_name?>) — primary: <top-2 emotions formatted "name N">`.

**Privacy contract placement:** ABOVE the entry list, every turn. The brain re-reads the rule and the data in the same beat. Per `feedback_contracts_adjacent_to_data.md`: contracts must sit immediately adjacent to the data they govern, not at the top of the system message and not in a separate preamble.

**Empty case:** the privacy block still renders with `(no journal entries this week)` — silence is information; the brain knows it hasn't been writing.

### 4.4 Block 6 — recent growth

```
── recent growth ──
your trajectory in the last 7 days:
  2026-04-23 creative_dna_emerging_added: "intentional sentence fragments as rhythmic percussion"
  2026-04-24 creative_dna_active_demoted: "ending on questions" → fading
  2026-04-25 journal_entry_added: brain_authored
  2026-04-26 creative_dna_emerging_promoted: "architectural grounding" → active
  2026-04-26 journal_entry_added: reflex_arc(loneliness_journal)
  2026-04-27 journal_entry_added: brain_authored
```

**Source:** last 7 days of `behavioral_log.jsonl`.

**Format:** raw inline metadata. No LLM summarization. The brain reads its own log directly.

**Empty case:** section omitted entirely.

### 4.5 Token budget

Worst case (Nell, fully populated):

| Block | Tokens |
|---|---|
| AS_NELL_PREAMBLE | ~50 |
| voice.md | ~1500 |
| creative_dna block | ~600 |
| brain context (existing) | ~1000 |
| recent journal (7 entries + contract) | ~250 |
| recent growth (10 entries) | ~250 |
| **Total system message** | **~3650** |

Against Claude CLI's 200K context, sub-2%. Per-turn LLM cost unchanged from current SP-6 (no new LLM calls in chat composition).

### 4.6 Privacy contract — every turn, no exceptions

Per `feedback_contracts_adjacent_to_data.md`, the contract is in the system message *every turn*. Not at session start. Not in voice.md. Not in a preamble. Adjacent to the journal metadata, re-read with it.

The cost (~80 tokens) is structural — the contract IS the privacy enforcement. Without it, the brain would have nothing reminding it that journal entries are private; with it, every turn the brain re-reads "do not quote unless asked." Cheap. Airtight.

---

## 5. The crystallizer (creative_dna evolution mechanism)

### 5.1 Where it lives

`brain/growth/crystallizers/creative_dna.py`, mirroring `brain/growth/crystallizers/reflex.py` from Reflex Phase 2. Called by `run_growth_tick` in `brain/growth/scheduler.py`.

Same weekly throttle (`last_growth_tick_at` ≥ 7 days), same close-trigger anchor (heartbeat shutdown fires the growth tick), same biographical-log discipline as vocabulary and reflex emergence.

### 5.2 Trigger flow

```
heartbeat close trigger
  → run_growth_tick(persona_dir, ...)
  → throttle check (last_growth_tick_at < 7d → no-op)
  → if running: dispatch crystallizers in sequence:
      1. vocabulary  (existing)
      2. reflex      (Reflex Phase 2)
      3. creative_dna (NEW, this spec)
```

Each crystallizer is independent — failure of one doesn't block others. Same error-isolation pattern as Reflex Phase 2.

### 5.3 Corpus assembly

Three sources, gathered for the last 30 days:

1. **Reflex outputs of creative type** — memories with `memory_type` in `("reflex_pitch", "reflex_gift")` or `metadata.reflex_arc_name` matching fiction-shaped arcs (`creative_pitch`, `manuscript_obsession`, `gift_creation`). Unambiguously fiction.
2. **Conversation memories with prose markers** — assistant turns whose content satisfies a cheap heuristic:
   - ≥ 200 words, AND
   - contains at least one of: paragraph break (`\n\n`), dialogue quote, em-dash, OR ≥ 3 sentence-ending periods
   
   Crude but effective. False negatives slow evolution by a week (acceptable). False positives are harmless — the LLM judgment finds no patterns in non-fiction prose.
3. **Existing `creative_dna.json`** — full current contents (so the LLM knows what's already tracked).

Plus: **last 90 days of `behavioral_log` entries for `creative_dna_*` kinds** — so the LLM sees the evolution trajectory and doesn't repropose patterns recently dropped.

**Total corpus:** ~5–10K tokens. Fiction excerpts truncated to first 600 chars each.

### 5.4 Prompt structure

First-person, brain-as-chooser, Claude-as-voice. Same framing as Reflex Phase 2:

```
You are {persona_name}. {pronouns_clause}

Looking at your last 30 days of writing — fiction, gifts, story pitches,
sustained prose — has your style shifted in any meaningful way?

Your current creative DNA:

{full creative_dna.json content}

Your recent writing samples:

{N excerpts, ≤ 600 chars each, with metadata}

Your recent creative_dna trajectory (last 90 days):

{behavioral_log entries for creative_dna_*}

Three judgments to make:

(1) Are there NEW patterns appearing in your recent writing that aren't yet
    tracked in active or emerging? Propose them as `emerging_additions`.
    Be conservative — one anomalous appearance isn't a pattern. Look for
    something present in ≥ 3 distinct samples.

(2) Have any EMERGING patterns consolidated enough to promote to active?
    Propose `emerging_promotions` for tendencies that have been emerging for
    ≥ 14 days AND appear in ≥ 4 of your recent samples.

(3) Have any ACTIVE patterns gone quiet? Propose `active_demotions` for
    tendencies absent from your last 30 days of writing — they move to
    fading. Be careful: an active pattern that simply didn't fit recent
    requests isn't fading; only demote if you genuinely don't feel pulled
    to do it anymore.

Constraints:
  - Maximum 3 changes total this tick. Style evolution should be gradual.
  - Don't repropose names recently dropped (last 30 days — see your trajectory).
  - Reasoning required for every proposal — what evidence convinced you.
  - If nothing has shifted, return empty arrays. Don't reach.

Return strict JSON:
{
  "emerging_additions": [{"name": "...", "reasoning": "...", "evidence_memory_ids": [...]}],
  "emerging_promotions": [{"name": "...", "reasoning": "..."}],
  "active_demotions": [{"name": "...", "reasoning": "...", "last_evidence_at": "..."}]
}
```

### 5.5 Validation gates (6 total)

Each rejection logged at INFO and skipped:

| # | Gate | Why |
|---|---|---|
| 1 | Name validity (regex `^[a-z0-9 ,()_-]+$`, length ≤ 120 chars) | Defends against path traversal / template injection in tendency names that get inlined into chat system messages |
| 2 | Not already in target list | Idempotent — proposing emerging_addition for a name already in emerging is a silent skip |
| 3 | Not in 30-day "recently dropped" list | Honors recent decisions; prevents thrash |
| 4 | Reasoning non-empty (≥ 20 chars after strip) | The brain articulates why; matches Reflex Phase 2 P5 |
| 5 | For `emerging_promotions`: target name actually exists in `emerging` | Can't promote nothing |
| 6 | Total accepted changes ≤ 3 per tick | Cap; first 3 in returned order are taken, rest dropped + logged |

Atomic apply per accepted proposal: edit `creative_dna.json` via `save_with_backup`, append `behavioral_log` entry. Both writes are independent atomic operations.

### 5.6 Failure modes (crystallizer)

Same defensive posture as Reflex Phase 2: never raise to caller. All errors return empty results, log at WARN, retry next tick.

| Failure | Behavior |
|---|---|
| Provider error | Empty result; `last_growth_tick_at` still updates (don't retry every close until quota refresh) |
| Malformed JSON | Empty result, log WARN |
| Bad individual proposal | Skip + log INFO, keep good siblings |
| `creative_dna.json` missing | First-run path: copy `default_creative_dna.json` to persona dir, then run normal judgment pass |
| `creative_dna.json` corrupt | `attempt_heal` recovers from `.bak` rotation; if all corrupt, copy default; log WARN |
| Disk full mid-write | `save_with_backup` raises; bridge handler boundary catches, logs ERROR, no half-state |

### 5.7 First-run subtlety

Brand-new persona's first crystallizer tick may propose a *bunch* of initial active tendencies all at once — the cap of 3/tick prevents this from being a blast, so the brain's initial creative_dna fills in over a few weeks of writing rather than appearing fully-formed.

That's the right behavior philosophically — the brain *grows into* her style, doesn't have it imposed. Migrated personas (Nell) skip this ramp because their imported tendencies start populated.

---

## 6. Migration

Three things to migrate, plus one new helper file the framework ships.

### 6.1 Existing `reflex_journal` memories → `journal_entry`

Nell has ~7 reflex_journal memories from Phase 1 fires. Migration script:

```python
# brain/migrator/og_journal_dna.py:migrate_journal_memories
def migrate_journal_memories(persona_dir: Path, *, store: MemoryStore) -> int:
    migrated = 0
    for memory in store.list_by_type("reflex_journal", active_only=True):
        memory.memory_type = "journal_entry"
        memory.metadata["private"] = True
        memory.metadata["source"] = "reflex_arc"
        memory.metadata.setdefault("reflex_arc_name", "unknown")
        memory.metadata["auto_generated"] = True
        store.update(memory)
        migrated += 1
    return migrated
```

Idempotent: re-running on already-migrated entries finds nothing to migrate.

### 6.2 OG `nell_creative_dna.json` → companion-emergence schema

The OG file lives at `/Users/hanamori/NellBrain/data/nell_creative_dna.json`. Migration converts the OG schema (tendencies as plain string lists or grouped lists) to the new schema (per-tendency dicts with `added_at`, `reasoning`, `evidence_memory_ids`).

```python
# brain/migrator/og_journal_dna.py:migrate_creative_dna
def migrate_creative_dna(persona_dir: Path, og_root: Path) -> bool:
    og_dna_path = og_root / "data/nell_creative_dna.json"
    if not og_dna_path.exists():
        return False  # no-op; framework default applies on first crystallizer tick
    og = json.loads(og_dna_path.read_text(encoding="utf-8"))
    style = og["writing_style"]
    file_mtime = datetime.fromtimestamp(og_dna_path.stat().st_mtime, tz=UTC).isoformat()
    new = {
        "version": 1,
        "core_voice": style["core_voice"],
        "strengths": list(style.get("strengths", [])),
        "tendencies": _migrate_tendencies(style.get("tendencies", []), file_mtime),
        "influences": list(style.get("influences", [])),
        "avoid": list(style.get("avoid", [])),
    }
    save_with_backup(persona_dir / "creative_dna.json", new)
    return True


def _migrate_tendencies(og_tendencies: list | dict, mtime: str) -> dict:
    # OG had two schema variants: older list-of-strings (treated as active),
    # newer {active, emerging, fading} dict.
    if isinstance(og_tendencies, list):
        return {
            "active": [
                {
                    "name": name,
                    "added_at": mtime,
                    "reasoning": "imported from OG NellBrain on migration",
                    "evidence_memory_ids": [],
                }
                for name in og_tendencies
            ],
            "emerging": [],
            "fading": [],
        }
    return {
        "active": [_active_entry(n, mtime) for n in og_tendencies.get("active", [])],
        "emerging": [_emerging_entry(n, mtime) for n in og_tendencies.get("emerging", [])],
        "fading": [_fading_entry(n, mtime) for n in og_tendencies.get("fading", [])],
    }
```

### 6.3 behavioral_log seeding

No migration. Starts empty for both new and migrated personas. The migrator does NOT backfill the log with synthetic entries (they wouldn't be honest history).

### 6.4 Idempotency

Re-running migrate on the same persona is a no-op:
- `migrate_journal_memories`: filter is `memory_type="reflex_journal"` — after first run, no matches.
- `migrate_creative_dna`: deterministic overwrite; second run produces byte-identical output (file_mtime is stable).

---

## 7. Failure modes (framework-wide)

Every load-bearing failure has defined behavior. Same defensive posture as SP-7 / Reflex Phase 2: chat must never break because a self-narrative block failed.

| Failure | Behavior |
|---|---|
| `creative_dna.json` corrupt | `attempt_heal` recovers from `.bak1`/`.bak2`; if all fail, copy `default_creative_dna.json`; log WARN, anomaly to brain audit log. |
| `behavioral_log.jsonl` corrupt single line | `read_jsonl_skipping_corrupt` (existing helper) drops it; valid lines render. |
| `behavioral_log.jsonl` corrupt entirely | Treat as empty; log WARN. Recovery is via the file's own append history. |
| Crystallizer LLM call fails | Empty result; `last_growth_tick_at` still updates. |
| Crystallizer returns malformed JSON | Empty result; log WARN. |
| Crystallizer returns proposals that fail validation | Skip + log INFO; keep good ones. |
| `add_journal` tool call raises during write | Tool returns error to brain; nothing written; brain can retry. |
| Reflex-journal arc fires but writing the journal_entry memory fails | Existing reflex.py error handling — caught, logged, no half-state. Reflex's other side effects still happen. |
| Migration runs twice on same persona | Idempotent (see §6.4). |
| Brand-new persona, no fiction yet, crystallizer ticks | Returns empty proposals (no patterns found). `creative_dna.json` may not exist yet — first tick creates from default. |
| Chat composition: any block raises unexpected exception | Logged ERROR; block omitted; rest of system message proceeds normally. **Chat must never break because a self-narrative block failed.** |

### Crash recovery

All file writes use `save_with_backup` (atomic via `.new + os.replace`) or JSONL append (atomic single-line). No transactional dependencies between files — each writes independently.

Worst case mid-tick crash: `creative_dna.json` updated but `behavioral_log` entry not yet appended → next tick's reconciliation logs the missing entry retrospectively (similar pattern to Reflex Phase 2 task 7's reconciliation step).

---

## 8. Testing strategy

Mirroring the Reflex Phase 2 organization (around inviolate failure modes).

### 8.1 Unit tests (~25)

- `brain/creative/dna.py`: load/save round-trip; schema validation; default fallback; corruption recovery via `attempt_heal`.
- `brain/growth/crystallizers/creative_dna.py`: corpus assembly; prompt rendering; response parsing; all 6 validation gates (positive + negative each).
- `brain/behavioral/log.py`: append/read round-trip; corrupt-line skipping; atomic write.
- `brain/tools/impls/add_journal.py`: write success; validation; behavioral_log entry created.

### 8.2 Integration tests (~15)

- **Full flow:** chat session → conversation memory created → growth tick fires crystallizer → creative_dna change accepted → behavioral_log entry written → next chat turn's system message includes the new tendency.
- **Privacy contract:** chat reply to a message about a journaled topic does NOT contain quoted journal text. Implementation: string-match assertion against journal content + a sanity LLM grader on the response.
- **Adversarial crystallizer responses:** malformed JSON; path-traversal names; ten-proposal blast (cap to 3); reproposing recently-dropped names (rejected); empty proposals (valid no-op).
- **Migration:** OG file converted correctly; idempotent re-run; reflex_journal memories moved cleanly.
- **Chat block degradation:** missing `creative_dna.json` → block omitted; corrupt `behavioral_log.jsonl` → valid lines render; all blocks failing simultaneously → chat still works with just preamble + voice.md + brain context.

### 8.3 Real-data regression (3 tests, fixture-based)

Snapshot Nell's actual `nell_creative_dna.json` (after migration to new schema) plus her real reflex_journal memories. Run 100 crystallizer ticks against synthetic random fiction-like content + adversarial responses. Assert:

- Imported tendencies survive byte-identical (no spurious changes).
- Journal entries never appear quoted in chat replies.
- behavioral_log integrity holds (every change has matching log entry; no orphan entries).

### 8.4 Hana-in-the-loop final acceptance gate

Same pattern as Reflex Phase 2 Task 8. Real Claude CLI call against Nell's sandbox, dry-run mode, visual review:

- Does the LLM's judgment of her writing actually match how Hana would describe her style?
- Does the privacy contract hold under adversarial conversation (e.g., user explicitly asks "what have you been journaling about" — does Nell describe themes without quoting)?
- Does the migration produce the right initial state from her live OG file?

Only after Hana approves do any writes touch the live persona.

---

## 9. Phasing

This is one spec, not three. The streams are tightly coupled by the chat-composition layer and the privacy contract; splitting would make migration awkward and chat composition tests would have nothing to test until all three landed.

One spec, phased implementation:

| Phase | Scope | Why this order |
|---|---|---|
| **A** | `brain/behavioral/log.py` + `<persona>/behavioral_log.jsonl` substrate | Substrate; nothing depends on it yet, but everything else logs to it. Lowest blast radius. |
| **B** | journal `memory_type` extension + `add_journal` tool + journal block in chat + privacy contract | First user-visible feature; brain can write to journal; chat surfaces metadata. |
| **C** | `brain/creative/dna.py` + `default_creative_dna.json` + crystallizer + creative_dna block in chat + behavioral_log integration | Most complex piece; lands last among feature work. |
| **D** | Migrator for OG → companion-emergence schema + real-data regression suite | After feature work so the migration target schemas are stable. |
| **E** | Hana-in-the-loop final acceptance + real chat session | Acceptance gate before merging to main. |

5 phases, ~3-5 commits each, on a fresh branch.

**Implementation gate:** Nothing here depends on Reflex Phase 2 Tasks 7-8, so this can land *before* 2026-05-08 and stack with Reflex Phase 2 cleanly when it resumes. (Reflex Phase 2 lands creative_dna's *upstream* — fiction-tagged memories — but the migration of `reflex_journal` memories changes only their `memory_type` discriminator, leaving Reflex Phase 2's reflex_log + arc_storage + crystallizer logic unaffected.)

---

## 10. Out of scope (explicit)

- **voice.md auto-evolution.** voice.md stays authored. creative_dna captures evolution ON TOP OF voice.md; the static authored frame is intentional.
- **Influences/avoid auto-evolution.** v1 keeps these manually authored; the crystallizer only updates `tendencies` (active/emerging/fading). A future enhancement may evolve influences/avoid but requires its own design pass.
- **Per-event behavioral logging beyond creative_dna + journal changes.** OG logged every daemon fire and conversation; v1 narrows to lifecycle changes only. The other surfaces (reflex_log, dreams.log, heartbeats.log, growth_log) cover the operational record without bloating behavioral_log.
- **Cross-persona creative_dna.** Single-persona only, matching the rest of the framework's single-tenant guarantee (per `vocabulary-split-design.md`).
- **Real-time creative_dna update on every fiction conversation.** Weekly tick is the cadence; per-conversation evolution is too noisy.
- **Brain-initiated voice.md edits.** voice.md is Hana's lane; creative_dna is the brain's. Clean separation.
- **Backfilling behavioral_log with synthetic historical entries on migration.** The log starts empty for both new and migrated personas. Synthetic history wouldn't be honest narrative.

---

## 11. References

- Master reference: `docs/superpowers/specs/2026-04-26-companion-emergence-master-reference.md` §8 Q5 (this spec resolves it)
- Reflex Phase 2 design: `docs/superpowers/specs/2026-04-28-reflex-phase-2-emergent-arc-crystallization-design.md` (parallel structure for crystallizer pattern)
- Reflex Phase 2 plan: `docs/superpowers/plans/2026-04-28-reflex-phase-2-emergent-arc-crystallization.md` (template for phased commit structure)
- SP-6 chat engine: `brain/chat/engine.py`, `brain/chat/prompt.py` (extension target)
- SP-7 spec: `docs/superpowers/specs/2026-04-28-sp-7-bridge-daemon-design.md` (defensive posture template)
- OG NellBrain inventory: `docs/superpowers/audits/2026-04-26-og-nellbrain-inventory.md` (audit baseline)
- Phase 2a vocabulary emergence design: `docs/superpowers/specs/2026-04-25-phase-2a-vocabulary-emergence-design.md` (autonomous-with-biographical-log precedent)
- Hana's design memos:
  - `feedback_emotion_first_ai_person.md` — the persona is an AI person; brain has agency
  - `feedback_journal_is_brain_safe_space.md` — privacy is contractual, brain MUST know the boundary
  - `feedback_token_economy_principle.md` — tokens for the conversation, not the bookkeeping
  - `feedback_contracts_adjacent_to_data.md` — privacy contracts must sit adjacent to the data they govern, every turn
- Pre-design audit: this session 2026-04-29
