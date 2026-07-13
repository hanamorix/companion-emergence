# Research Engine Redesign — Design

**Date:** 2026-07-13
**Status:** Approved (brainstorm with Hana; source review by ToT, 5/6 findings verified)
**Tier at ship:** EXPERIMENTAL
**Branch:** `hana/research-engine-redesign`

## 1. Problem

ToT's review of `brain/engines/research.py` (verified 2026-07-13 against code) found the engine
is a memory-vignette generator, not a research organ:

1. The only artefact per fire is a 3-5 sentence first-person "memory of having researched" —
   with `scope: "internal"` or a failed web search, the prompt runs on model weights alone
   (pure confabulation).
2. No continuation or closure: per-interest state is `pull_score` + 24h cooldown; no findings
   accumulate, nothing marks a topic done, nothing escalates a topic that got interesting.
3. No notes store — repeated fires on one topic inflate the memory DB with near-duplicates
   (compounds the dedup defect in issue #66).
4. Interests are frozen at bootstrap: new interests come ONLY from voice.md (first 2000 chars,
   5 hallucinated topics). The heartbeat hook only bumps existing ones by keyword match.
   Lived conversation never creates an interest (the Phoebe Lispector/carrot-cake case).
5. Topic selection is mechanical (highest pull eligible) while the prompt instructs
   "what pulled you to the topic today" — inviting a confabulated motive.
6. (Partly refuted: no literal JSON/no-bullets contradiction exists, but the principle stands —
   formatting belongs to mechanical code, not the creative call. Real weak spot: bootstrap
   parses creative output with bare `json.loads`.)

## 2. Goals

- One research **session** produces cumulative notes on the topic, a short felt memory
  (byproduct, not the point), and a lifecycle verdict.
- Depth comes from sessions over days (Option C), not multi-call agentic loops.
- The Kindled **chooses** the topic (and may decline), so the "why" is real.
- Interests are born from lived life: conversation, research tangents, and a slow sweep.
- Formatting handled mechanically; the creative call writes prose under markers.
- Zero change to downstream contracts: `memory_type='research'` memories (feed +
  daemon-state theme), `research_completion` initiate candidates, `ResearchLog`,
  `WebSearcher`, and the four ResearchEngine construction sites keep working.

## 3. Non-goals / out of scope

- No changes to salience, forgetting, eviction, or dedup pipelines — that is ToT's lane
  (issue #66). Research memory writes stay plain `store.create` until the dedup pipeline
  lands, then adopt it.
- No provider / prompt-serialisation changes (ToT's PR #58 hunt territory).
- No multi-round agentic search within a session (may layer on later).
- No user-facing settings. The brain owns cadence, caps, and budgets (user-surface principle).
- No notes-retention policy yet — tail-cap on read; joins the standing JSONL-retention defer.

## 4. Data model

### 4.1 Interest schema — two new optional fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `status` | `"active" \| "dormant"` | `"active"` | Dormant = closed by her verdict or retired by the sweep. Excluded from selection. A keyword-bump that crosses `pull_threshold` revives it to active. |
| `origin` | `str` | `"bootstrap"` | Where the interest came from: `bootstrap`, `conversation`, `side_quest`, `sweep`. Labels only — no branching logic on it. |

Back-compat: `Interest.from_dict` treats absent fields as defaults, so existing
`interests.json` files load unchanged. `walker.py` / `alarm.py` heal skeletons untouched
(empty list stays valid). Migrator untouched (fields optional).

### 4.2 Notes store

- Path: `<persona_dir>/research/<interest_id>.md` — one markdown file per interest.
- Free-form: facts with sources, opinions, reactions, lists — whatever fits the subject.
- Each session appends one block under a mechanically written header:
  `## Session YYYY-MM-DD` followed by the LLM's NOTES section verbatim.
- Read path: the **tail** (last ~4,000 chars, split on session headers) is injected into
  that topic's next selection summary and session prompt. Never ambient-injected into chat.
  Reachable in conversation through the existing `read_file` tool (the directory is inside
  the persona dir, already readable).
- Writes are atomic (write temp + rename), UTF-8.

### 4.3 Files summary

| File | Change |
|---|---|
| `interests.json` | +`status`, +`origin` per entry (optional) |
| `research/<id>.md` | NEW — per-topic notes |
| `interest_sweep_cadence.json` | NEW — persisted wall-clock cadence for the sweep |
| `research_log.json` | unchanged (audit trail) |

## 5. Session flow (one fire, two LLM calls)

Existing outer gates unchanged: heartbeat cadence, `days_since_human >= 1.5` or
`emo_peak >= 7.0`, per-interest 24h cooldown, `pull_threshold`, and the whole fire runs
inside `cli_throttle.background_slot()` (defer to active chat, as today).

### 5.1 Select (Haiku-class call)

Input: top ≤5 eligible **active** interests, each rendered as topic + pull + a notes-tail
summary (or "(no notes yet)") + up to 3 recent relevant memories (existing
`_build_memory_context` machinery, trimmed).

Output (this call IS the mechanical-formatting call, so strict JSON is fine here, parsed
via `extract_json_object`):

```json
{"choice": "<interest_id>" | null, "why": "<one sentence>"}
```

- `null` choice = "nothing pulls today": tick ends with reason `"declined"`, **no cooldown
  burn** on any interest, no second call.
- Parse failure → fail-soft: fall back to current behaviour (mechanical highest-pull winner,
  `why` empty).

### 5.2 Research (main provider call)

Prompt assembly: chosen topic + the real `why` line + prior notes tail + web results
(DDGS, 5 results, as today; `scope:"internal"` skips web, as today) + relevant memories +
emotion summary.

System prompt (replaces the current hard rules): the model writes plain prose under three
fixed markers, nothing else —

```
NOTES:
<free format — facts with sources, reactions, opinions, lists if lists fit.
 Continue from the prior notes; don't repeat what's already written.>

MEMORY:
<2-4 sentences, first person as {persona_name} — how the session felt, what surprised you.>

VERDICT:
<one line: continue | close | spawn: <topic 1>; <topic 2>>
```

First-person is required for MEMORY only. No JSON anywhere in this call. No structure
mandate on NOTES. No "what pulled you" instruction — the why is supplied, not invented.

### 5.3 Mechanical parse + persist

Marker split (case-insensitive, line-anchored). Fail-soft ladder:
- All three markers found → normal path.
- Markers missing → whole output appended as NOTES, verdict = `continue`, no memory created.

Then, in order:
1. Append NOTES block to `research/<id>.md` under the dated session header (atomic).
2. `store.create` the MEMORY as `memory_type='research'` with the existing metadata shape
   (`_create_research_memory`) — feed and daemon-state contracts intact. Skipped if MEMORY
   section absent.
3. Apply VERDICT:
   - `continue` — nothing beyond the normal cooldown update.
   - `close` — interest `status="dormant"` (notes retained; revivable by keyword-bump).
   - `spawn:` — create new interests (origin `side_quest`, `pull_score` seeded at
     `pull_threshold - 1.0` so they need one real bump before firing), deduped casefold
     against ALL existing topics (active + dormant), hard cap 2 per session.
4. Update `last_researched_at`, append `ResearchFire` to the log,
   `_emit_research_candidate` unchanged.

Any exception in any step: log warning, skip the rest of that step, never corrupt
`interests.json` (all saves go through the existing atomic `InterestSet.save`).

## 6. Interest spawn — three inlets

### 6.1 Pass-2 extractor field (primary, zero new calls)

The existing post-chat pass-2 extractor gains an optional output field:

```json
"interest_candidate": {"topic": "...", "keywords": ["..."], "why": "..."}
```

Mechanical guard on apply (mirrors the emotion-vocab `_filter_to_registered` pattern):
- dedupe casefold vs existing topics (active + dormant — a dormant match bumps instead
  of duplicating),
- caps: ≤1 per turn, ≤3 per day (persisted daily counter alongside the interests file),
- origin `"conversation"`, `pull_score` seeded below threshold (needs organic bumps to fire),
- fail-soft: malformed candidate dropped with a log line.

### 6.2 Side quests

VERDICT `spawn:` from §5.3 — research-born tangents, origin `"side_quest"`.

### 6.3 Weekly sweep (safety net)

- Cadence: 7 days, persisted wall-clock via a new `interest_sweep_cadence.json`
  **mirroring `brain/bridge/persisted_cadence.py`** (the generic helper — NOT
  `soul/cadence.py`), advance-after-tick regardless of outcome.
- One Haiku-class call over a bounded sample of recent conversation, dream, and monologue
  memories + the current interest list.
- Output (strict JSON, `extract_json_object`): ≤3 proposed new interests (origin `"sweep"`)
  + ≤3 proposed retirements (active → dormant). Same dedupe + below-threshold seeding
  as 6.1. Retirement only sets `status` — never deletes, never touches notes.
- Runs inside `background_slot`, counts against a small daily budget, fail-soft on
  every error.

### 6.4 Bootstrap fix

`_seed_interests_from_voice` becomes seed-from-voice-and-life: when interests.json is
empty, the prompt includes voice.md (as today) **plus** up to 10 recent conversation
memories when any exist. Parse hardened: `extract_json_object` instead of bare
`json.loads(raw)`.

## 7. §Wiring

**Reads from:** emotion aggregate state (session prompt), memory store
(`search_text` context + sweep sample over conversation/dream/monologue memories),
attunement-era conversation via pass-2 extractor, its own notes files (continuation).

**Feeds into:** memory DB (`memory_type='research'` → feed panel, daemon-state interior
theme), initiate queue (`research_completion` candidates → D-reflection → compose —
unchanged), `interests.json` (which the heartbeat keyword-bump loop keeps feeding back),
notes files (read by its own next session + reachable in chat via `read_file`).

**Organ DoD:** producer fires on the live heartbeat path (already wired); a through-path
test asserts one heartbeat-driven fire appends notes + creates the memory + applies the
verdict (§9); readers exist for every output; this section is the §Wiring entry.
`docs/maturity-manifest.md` gains the redesigned engine as EXPERIMENTAL.

## 8. Cost + failure posture

- Per fire: 2 LLM calls (select + research), down to 1 when selection declines.
  Weekly sweep: 1 call. Pass-2 field: 0 extra calls.
- Everything LLM-touching sits inside `cli_throttle.background_slot()` — interactive chat
  always wins.
- Every LLM call and every parse is fail-soft: log, degrade (fallback selection /
  notes-only append / dropped candidate), never raise out of the tick, never corrupt state.
- Salience of the select call's decline path: declining burns no cooldowns, so the same
  interests remain eligible next tick.

## 9. Testing

TDD throughout (failing test first, per phase). Highlights:

- **Through-path canary (Organ DoD):** heartbeat `run_tick` with a stub provider returning
  a marker-formatted reply → asserts notes file appended, research memory created,
  verdict applied, cooldown updated. This is the test that would have caught the
  vignette-only rot.
- Selection: picks among eligible; decline path burns no cooldown; parse-failure falls back
  to mechanical winner.
- Parse: three-marker split; missing-marker fail-soft ladder; spawn cap + casefold dedupe;
  dormant revive on bump.
- Spawn inlets: pass-2 candidate guard (caps, dedupe, dormant-bump); sweep propose/retire;
  bootstrap with + without conversation memories.
- Back-compat: old interests.json (no status/origin) loads; walker/alarm heal unaffected;
  existing ~25 research tests updated, ~18 interests tests extended.
- Full gate before merge: `uv run pytest`, ruff, `pnpm test`, `pnpm build`.

## 10. Coordination (two-dev)

- Files touched: `brain/engines/research.py`, `brain/engines/_interests.py`,
  `brain/engines/heartbeat.py` (interest hook + engine construction), pass-2 extractor
  module, `brain/bridge/persisted_cadence.py` consumer (new cadence file), tests.
  **Zero overlap** with ToT's open PRs (#65 tests/harness, #58 no files yet).
- **Fences:** do not touch provider/prompt serialisation (PR #58 territory) or
  salience/forgetting/dedup (issue #66, assigned ToT).
- GitHub issue opened + assigned to Hana before implementation; draft PR after first commit.

## 11. Deferred

(Tracked here + `project_companion_emergence_deferred.md` + next version's brainstorm.)

1. Bounded agentic multi-search within a session (Option B) — layer on if single-shot
   sessions prove too shallow in live use.
2. Notes-file retention/size policy — joins the standing JSONL-retention defer.
3. Research memory writes adopt the dedup pipeline — after issue #66 lands.
4. Dream/monologue crystallisation as additional spawn inlets (beyond the sweep's
   read-only sample) — after the three shipped inlets prove out.
5. Surfacing notes in the UI (a Research panel) — feed entries suffice for now.
