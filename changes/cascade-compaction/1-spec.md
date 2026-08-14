# 1-spec.md — Cascade Compaction (Phase 1, compaction half)

**Status:** ready for stage-3 (all loci verified against the code map, cited in §6).
**Branch:** `ThinkerOfThoughts/cascade-compaction` off diagnose base `cd29bc61`. Matched-base; no merge of main.
**Authoritative design inputs:** `~/.claude/plans/memory-dream-rework-p1-cascade-brief.md` (the granular brief;
decisions locked with owner "ToT"), `/home/zero/Documents/New_mem_system.md` Part 1 (north-star),
`~/.claude/plans/memory-dream-rework-plan.md` §"Phase 1's compaction-adjacent additions (1c, 1d)".

---

## 1. Problem

The active-chat **compaction** (short-term memory: the rolling summary of the live conversation, standalone
from `memories.db`) has four defects surfaced as bycatch of the monologue-bleed bughunt:

1. **No temporal structure (#82).** One flat `summary` row re-folded each cadence. It preserves *order* but
   not *when* — a reader cannot tell 2-hours-ago material from 3-days-ago material. Wall-clock `ts` already
   lives per-turn in the buffer but is stripped at fold time.
2. **No fold-output validation (#77).** The provider's fold output is stored with only `.strip()`. A refusal,
   a non-first-person meta-comment, or an "I won't proceed"-style completion is stored verbatim *as the
   summary* and then re-presented to the model as its own recalled past — a corruption vector.
3. **Unbounded head growth from a long day.** The 24h summary targets a *fraction* of its source; an unusually
   long day therefore folds to an unusually long summary, bloating the cache-stable head prefix without bound.
4. **Session lifecycle drops / balloons the summary.** Two coupled sub-defects:
   - **1c — rollover.** (A) After a >24h idle gap, `/sessions/active` returns `null`, the client opens a fresh
     empty session, and the prior summary is abandoned. (B) A continuously-used conversation never hits the
     24h gap, so its buffer/archive grows indefinitely. The finalize tick (24h) deletes conversation buffers,
     which is entangled with the 24h attach window.
   - **1d — archive monolith.** `append_archive` is append-only, never rotated, with **no multi-segment
     reader** — one long session's `archived_conversations/<sid>.jsonl` grows into a single unbounded file.

These compound: no rollover → one session runs forever → its archive balloons. Phase 1 fixes the compaction
core (1, 2, 3) and brings the two compaction-adjacent behaviors (1c, 1d) along in the same phase.

**Explicitly out of scope (later phases):** the dream cycle / Haiku chunker (Phase 5), the ROOT-2 relevance
overhaul (Phase 2), the interim garbage-gate + candidate queue (built separately, DRAFT #123 — NOT this
branch), ROOT-1 within-session replay bound (Phase 7). This branch is the **cascade-compaction half of Phase
1 only**.

## 2. Target design (pinned — implement, do not re-derive)

### 2a. Three-section age-stratified summary (owner's call: ONE row, three sections)

Replace the flat single-layer summary with **one `summary` row holding three age-labelled sections**
(24h / 48h / 72h). The layer's position IS the timestamp (#82 solved architecturally — no per-sentence
date-stamping). One row, not three rows (owner ToT's explicit call; confirmed *not* cache-worse — see §4).

**Current representation (from code map):** the summary row is `{"session_id", "speaker":"summary", "text":
<opaque first-person prose blob>, "ts", "compaction": {covers_until_ts, folded, gen}}`
(`compaction.py:364-374`). `text` has **no internal structure today** — it is the LLM's raw `.strip()` output.

**Target representation:** keep ONE `speaker=="summary"` row. Hold the three sections as **structured
sub-fields in the `compaction` meta** — `compaction.sections = {"24h": {...}, "48h": {...}, "72h": {...}}`,
each with its text + covered-ts span — so the cascade can re-compact a tier independently without re-parsing
prose. The row's `text` field is the **deterministic render** of the three sections in fixed order with
age-labelled headers carrying the ts spans (temporal markers, §2d). `text` is what `engine` wraps into the
head prefix. Determinism (fixed section order, no per-render nonce/live-timestamp) is what keeps the head
byte-stable between re-compactions (§3, C6). `budget.py` re-parses the head only by the `"[Earlier in this
conversation:"` **prefix** (`budget.py:28-31`) and preserves the whole block — it does not parse sections, so
its contract is unchanged as long as the prefix string is unchanged.

| Tier (label) | Fed by | Soft target | Hard cap |
|---|---|---|---|
| tier1 — "yesterday" (24h) | raw turns crossing 24h old | ≈60% of that raw batch | `_SECTION_24H_CHAR_CAP` (§5, open-Q8) |
| tier2 — "day before yesterday" (48h) | tier1 cohort, aged + re-compacted | ≈40% of tier1 | — (bounded by input) |
| tier3 — "a few days ago" (72h) — **TERMINAL** | tier2 cohort, aged + re-compacted; **re-compacted forever (fade)** | ≈20% of tier2 | `_SECTION_72H_CHAR_CAP` = 0.20 × tier1 cap |

**Owner design (2026-08-13):** human age-band labels in the render ("yesterday" / "day before yesterday" /
"a few days ago"); **tier 3 is TERMINAL** — material graduates raw→tier1→tier2→tier3 and then STAYS in tier 3,
re-compacted each cycle so the oldest content fades to an ever-briefer trace (there is no 4th tier and no
evict-out-of-summary leg). Tier 3 carries its own hard cap on top of the 20% compaction.

### 2b. Cascade pass — TRUE AGE-GATED GRADUATION, one code path, on the existing cadence

One cascade pass on the **existing daily supervisor tick** (+ a 24h-only `apply_budget` backstop). Promotion
is **age-gated by the material's ACTUAL content age each pass**, and material **graduates** raw→tier1→tier2→
tier3 one tier per cohort as it ages, then **stays in tier 3 (terminal, re-compacted forever)** — the owner's
design, New_mem_system.md Part 1 + OWNER 2026-08-13. It is neither a fixed one-position-per-tick shift NOR a
re-fold of the prior tier with fresh raw — both are proxies for age that break (see the two failure modes
below; §2b implements the fix stage-3 rounds 1 + 3 converged on, plus the owner's terminal-tier ruling). Each
pass:
1. **age-partitions eligible raw turns** by true age (`(24h,48h]`→tier1 band, `(48h,72h]`→tier2, `(>72h]`→tier3);
2. **classifies each existing section by its OLDEST covered edge** (`now − covers_from_ts`) — the oldest edge
   only advances, so a section built "yesterday" a day ago is now in the tier2 band and **graduates** there;
3. assembles oldest-first, each tier = recompact/fold of (sections that classify into that band) + (that
   band's raw group): tier3←recompact(·,20%, capped), tier2←recompact(·,40%), tier1←fold(·,60%,capped).
   **It never co-folds the prior tier1 section with fresh raw** — the prior tier1 cohort graduates to tier2;
   only just-crossed raw forms the new tier1. **Tier 3 is TERMINAL and a MULTI-INPUT fold:** under continuous
   use every cycle the new tier3 folds **two prior texts** — the **persisting prior tier3** (oldest edge >72h)
   **and the newly-graduated prior tier2** (just crossed 72h) — plus any raw crossing 72h. Because the owner
   ruled tier-3 content is never dropped, this uses the same lossless-leaning discipline as the #77 fallback:
   join all inputs oldest-first, compact to 20%, enforce `_SECTION_72H_CHAR_CAP` by sentence-boundary
   truncation, and on a double-reject fall back to the join truncated to the cap — so salient content from BOTH
   inputs survives. Tier 3 fades each cycle but never evicts (no 4th tier). After any inactivity, everything is
   >72h → the cascade rebuilds straight to tier 3 only (tiers 1 & 2 empty).

**Why not the two rejected proxies:** a *tick-count shift* mislabels every layer after a multi-day sleep
(cadence fires once, not once-per-missed-day — stage-3 round-1); *classify-by-newest-edge + re-fold prior-24h
with fresh raw* refreshes the newest edge forever, so old material is perpetually re-labelled tier-1 "recent"
and never graduates under continuous use, reproducing Problem-1 one tier down (stage-3 round-3 F3). The
oldest-edge + no-remerge mechanism satisfies both continuous-use graduation and gap robustness (criterion
C14, asserted on true content age — graduation 1→2→3 then terminal persistence in tier 3 — not on
`covers_until_ts`). One code path, one lock. Dream-ordering slot
(Phase 5) is **reserved**: raw turns stay uncompacted until *after* a reserved pre-24h-fold slot; P1 adds no
dream, only leaves the slot free.

### 2c. Fold-output validation (#77)

Before storing any fold/re-compaction output, **validate the provider output**:
- reject a refusal / non-first-person / "I won't proceed"-style completion;
- on rejection, **retry once**, then **fall back** — never store the raw `.strip()` unconditionally.
- **Fallback under the cascade (stage-3 L3):** a double-rejected *section build* must **not drop the source
  material** it was promoting. The target falls back to a lossless-leaning safe join of its **inputs** (the
  prior section prose + the age-group's rendered turns), truncated at a sentence boundary to the tier's bound,
  logged as a soft-fail — so a failed 48h build never silently discards the 24h material being promoted.
The validation rule is a documented, testable predicate (see 1.5-criteria C4).

### 2d. Temporal markers (#82) + owner age-band labels

Carry the coarse wall-clock anchors already on each turn (`ts`) into the section text/labels so age-texture
survives the fold (a section header carrying the covered date/time span). Coarse, not per-sentence. Each tier
header also carries the **owner-specified human age-band label** (OWNER 2026-08-13): tier1="yesterday",
tier2="day before yesterday", tier3="a few days ago" — static tier labels (not computed dates), so the render
stays byte-stable (§3, C6).

### 2e. 1c — session rollover (TWO triggers)

Both triggers **seed the new session with the previous session's post-compaction form** and archive the rest.

- **A) >24h idle-gap stale resume (SYNC).** On a stale resume (last turn > `_ATTACH_MAX_AGE_HOURS`),
  extract → `compact_conversation` the *entire* old conversation (`min_keep_tail=0`, fold-everything) →
  archive → **delete the old buffer + cursor + write `rolled_to.json`** (rollover owns the lifecycle) → seed
  the new session with the summary as its first **already-extracted** head row, and **return the new session
  id (not `null`)**. SYNC (owner ToT, brief §1c-A — kept, not de-synced): block the resume until the seed is
  written — correct-by-construction, costs one bounded Haiku round-trip on a rare event. **Stale selection**
  when several stale buffers exist (the endpoint has no client-supplied prior sid): roll over the
  **most-recently-active** (max last-turn ts; tie-break highest sid) — stage-3 A1.
- **B) Weekly cap (NEW).** A continuously-used conversation never hits the 24h gap, so start a new session
  every ~week, seeded with the previous session's post-compaction form = **the three compaction tiers + the
  40 most-recent messages** (`min_keep_tail`), archiving the rest. **Driven from the daily supervisor tick**
  (brief §1c-B "checked on the daily cadence" — the continuously-attached client is the whole target
  population, so an attach-only trigger would never fire for it; stage-3 F2/M1) at a **quiet-moment boundary**
  (last turn older than `_ROLLOVER_QUIET_GAP`, never mid-exchange) when session age ≥ `_WEEKLY_ROLLOVER_AGE`.
  A **successor pointer** `rolled_to.json` (written under the lock before the old buffer is deleted) redirects
  a continuous client still holding the old sid to the new session. The redirect is installed at the **real
  resolution chokepoint** — `get_or_hydrate_session` (`session.py:129-205`), which has **FIVE** call sites that
  resolve a session: `/chat`, `/stream`, `/sessions/snapshot`, `/sessions/close`, and `GET /state`. The consult
  goes in that one function, but **each handler must then rebind to the RESOLVED `sess.session_id`** (not the
  raw `req.session_id`/path id) for **every** downstream op — backend call, `in_flight_locks` key,
  `remove_session`/`in_flight_locks.pop` cleanup (`:2835-2836`), `/state` in-flight lookup, echoed id —
  otherwise close/snapshot silently no-op on the deleted old buffer and falsely report success (F1), a second
  client on the successor sid races on a split lock key (F2), the close cleanup leaks the successor's registry
  entry + lock (G1), and `/state` misreports in-flight (round-5). This whole class is closed in **one sweep**
  (round-5 F2) with a **structural guard** (criterion C21: no raw id downstream of resolution in any such
  handler) so a future 6th caller cannot silently regress it. This supersedes the round-1
  `ingest_turn`-consults-the-pointer design (which 404'd upstream — round-2 P-1). Criteria C16 (redirect at the
  write handlers), C19 (lock keying), C20 (close cleanup + /state), C21 (structural guard).

**Finalize-tick reconciliation + buffer lifecycle (both triggers):** the finalize tick
(`finalize_after_hours=24`) becomes **extraction-only** and must **stop deleting conversation buffers**;
**the rollover path now OWNS buffer deletion** — after extract → archive → seed it deletes the old buffer +
cursor + evicts the registry entry. (The prior draft removed the finalize delete but reassigned it nowhere,
leaking every rolled-over buffer into the active set that `list_active_sessions` re-iterates each tick —
stage-3 L2; criterion C15.) The 24h attach-window and 24h finalize-delete were intentionally coupled; this
decouples them. Memory extraction of old turns continues (the seed is the immediate recap; memory is
long-term recall).

### 2f. 1d — archive segmentation

`append_archive` (`buffer.py:290`) is append-only / never-rotated. **Correction from the code map:** a reader
is **not** absent — `read_archive` (`buffer.py:310-312`) already loads the whole archive file via
`read_jsonl_skipping_corrupt`. What is missing is **segment/rotation** and a **segment-aware reader**. So 1d:
**segment/rotate** the archive (per-day or size-capped rolling segments, e.g.
`archived_conversations/<sid>.<segment>.jsonl`) **and extend `read_archive`** (or add a sibling) to read
across all segments in order. Preserve the **provenance chain** (every faded summary generation + the raw
turns it superseded, in order — the archive currently appends the old summary row on a fold,
`compaction.py:380-381`) and the **archive-before-rewrite byte-count atomicity contract** (`append_archive`
returns bytes written; `compaction.py:383-394` aborts before buffer mutation on a zero-byte write). Keep
`read_archive`'s existing call sites working (back-compat: a non-segmented archive still reads). (Weekly
rollover (2e-B) bounds per-session growth; 1d bounds the on-disk archive itself.)

### 2g. Migration

Existing personas carry one single-layer summary of **accumulated (often months-old) history**. Migrate it
into the sectioned form seeding it as **tier 3 ("a few days ago"), NOT tier 1** (stage-3 round-6 MO-1): the
oldest-edge classifier (§2b) keys on the `covers_from_ts` **value**, so it would otherwise mislabel a legacy
blob "yesterday" and re-fold it as recent — reproducing the #82 defect for 100% of current production personas.
**Structural fix (round-7 MO-2):** set `covers_from_ts` to an explicit **old-floor** value (`migration_now −
96h`, `_LEGACY_AGE_FLOOR`) **unconditionally — never `covers_until_ts`**; since the classifier buckets age
>72h as terminal tier 3, this guarantees the blob lands in and stays in tier 3. (An optional primary path may
derive a real oldest ts via an **archive scan**, never a buffer check; but the default is the old-floor.) The
tolerant reader (plan §1.1) applies the same before migration runs. **Reuse the `compaction_migration.py`
backlog pattern.** Tests assert a migrated legacy persona's history classifies tier 3 (not tier 1) and stays
tier 3 across the next cascade pass (the #82-regression guard).

## 3. Invariants any build MUST honor (gate-4/criteria material)

- **Cache-stable head prefix (triple-coupled).** `engine._buffer_turns_to_messages` authors the
  "[Earlier in this conversation: …]" head; `budget.py` re-parses it by prefix. The 3-section render must keep
  authoring↔re-parse in **lockstep** and stay **byte-stable between daily re-compactions** — it may bust once
  per day on the re-compaction (already the case; acceptable).
- **Lossless-before-lossy + cursor guard + idempotent.** Never compact an un-extracted turn; archive+verify
  (byte-count) before rewrite; the pass is idempotent (existing `compaction.py` docstring invariants).
- **COMPACTION_MODEL = "haiku"** stays.
- **Don't starve interior-continuity.** `brain/monologue/ambient.py` reads `monologue_trace` on purpose — the
  cascade must not remove/starve that read path.
- **Provenance chain preserved** across archive segmentation (1d).

## 4. Why one row / three sections is not cache-worse (owner-confirmed)

The head prefix already busts on every daily re-compaction, and that is once/day. Three sections in one row
bust on the same once/day cadence, so there is no additional cache regression versus the current single fold.
(Owner ToT confirmed in the brief.)

## 5. The three open engineering points — RESOLVED IN THIS SPEC (do not escalate)

These are engineering defaults, not owner-values calls (brief §"Open points"; kickoff §"three open points").
Each is set as a **documented module const with a reasoned default + a one-line follow-up note**:

- **Open-Q8 — 24h-section hard cap.** The data-check (avg char length of a real 24h conversation) is not
  runnable here (no live bridge; Phoebe read-only/off-limits). **Resolution:** ⟨const + value + reasoning in
  plan §; documented follow-up to tune against live data⟩. The 60% is the soft target; the cap is the
  absolute ceiling that stops a huge day from bloating the head.
- **Seed asymmetry.** 24h-gap rollover seeds summary-only (`min_keep_tail=0`); weekly seeds 3 tiers + 40
  messages. **Intentional** — a >24h-old "recent 40" isn't recent. Confirmed to read right (§2e).
- **Weekly rollover fire-point during active use.** **Resolution:** fire at the **least-disruptive boundary**
  (next resume / a quiet moment, never mid-exchange). Stated as a const/predicate in the plan.

## 6. Prior art / current state (loci — verified from code map)

- **Single-layer fold** (`brain/chat/compaction.py`): module docstring stating the 4 invariants `:1-24`;
  `compact_conversation(persona_dir, session_id, *, older_than, fold_existing_summary, provider,
  min_keep_tail=40, max_compact_turns=None, now=None, lock_stale_s=600.0)` `:233-244`; per-session lock
  `:254-259`; `_split_buffer` (first `speaker=="summary"` wins) `:192-209`; cursor guard (`None`→no-op; cutoff
  clamped to cursor) `:263-273`; `min_keep_tail` protected window `:275-284`; `_FOLD_PROMPT` `:107`,
  `_SUMMARY_PROMPT` `:85`; `_TARGET_FRACTION=0.25` `:147`, `_MIN_TARGET_WORDS=40` `:149`,
  `COMPACTION_MODEL="haiku"` `:54`; provider call **`.strip()`-only, no validation** `:338` (exception→
  `"[truncated N earlier messages]"` soft-fall, NOT bad-output handling); summary-row + `compaction` meta
  `{covers_until_ts, folded, gen}` `:364-374`; archive-before-rewrite byte-count abort `:376-394`; atomic
  rewrite `[summary_row, *retained_now]` `:396-417`.
- **Archive** (`brain/ingest/buffer.py`): `append_archive` (append-only, never rotated, returns bytes)
  `:290-307`; `_archive_path` → `archived_conversations/<sid>.jsonl` `:284-287`; **whole-file reader
  `read_archive`** `:310-312`; `rewrite_session_atomic` (tmp+fsync+replace) `:265-281`; compaction lock
  acquire/release `:341-392`; `delete_session_buffer` `:162-165`. Provenance: old summary row appended to
  archive on fold (`compaction.py:380-381`).
- **Session attach/finalize**: `/sessions/active` `bridge/server.py:1203-1253`, `_ATTACH_MAX_AGE_HOURS=24.0`
  `:1224`, `>=` exclude `:1248`, returns `best_sid` or `None` `:1253`; backlog-migration startup thread
  `bridge/server.py:913-935`. Finalize delete lives in `ingest/pipeline.py:finalize_stale_sessions` `:529-626`
  (delete sites `:557-559`, `:587-590`; poison-move `:591-609`); scheduled from
  `bridge/supervisor.py:_run_finalize_tick` `:1649-1697` (finalize cadence block `:499-521`,
  `finalize_after_hours=24.0` `:129`, `finalize_interval_s=3600.0` `:130`).
- **Daily compaction tick**: `bridge/supervisor.py:_run_compaction_tick` `:1619-1646` (calls
  `compact_conversation(..., older_than=timedelta(hours=24), fold_existing_summary=True)`); daily cadence block
  `:639-662` (`compaction_interval_s=86400.0` `:138`, state file `compaction_cadence.json`).
- **Head prefix**: `engine._buffer_turns_to_messages` — summary hoist to head `:370-377`, render
  `f"[Earlier in this conversation: {summary_text}]"` inserted at index 0 `:405-408`; cache-stable region
  comment `engine.py:48-55`; `apply_budget` backstop `budget.py:42-91` (calls `compact_conversation` with
  `older_than=timedelta(0)`), re-parse const `_COMPACTION_SUMMARY_PREFIX="[Earlier in this conversation:"`
  `budget.py:28-31`, preserve-head re-parse `:94-109`; live call site `engine.respond()` `:238-245`
  (`max_tokens=80_000, preserve_tail_msgs=40`).
- **Migration pattern** (`brain/chat/compaction_migration.py`): `run_backlog_migration(...)` `:162-233`
  (marker-gated on `archived_conversations/.compat_migrated`, replays daily cadence oldest-first, reuses
  `compact_conversation`); `_drain_session` `:91-159`; `_DRAINED_REASONS={"nothing_aged","cursor_none"}` `:47`;
  `MigrationResult` `:50-60`; `_write_marker` (tmp+fsync+replace) `:69-78`.
- **Interior read** (`brain/monologue/ambient.py`): `build_interior_continuity_block` `:22-45`,
  `store.list_by_type(MONOLOGUE_TRACE_TYPE, ...)` `:34` — a **MemoryStore** read, entirely separate from the
  compaction buffer/archive. The cascade never touches `monologue_trace` rows, so C13 is preserved by
  construction (the criterion guards against an *accidental* coupling, cheap to check).

## 7. Expected touched files (declared for the cold-reviewer context — closed set)

- `brain/chat/compaction.py` — 3-section representation, cascade pass, fold validation, temporal markers, cap.
- `brain/chat/compaction_migration.py` — single-layer → 3-section migration (reuse backlog pattern).
- `brain/ingest/buffer.py` — archive segmentation + multi-segment reader (1d).
- `brain/bridge/server.py` — `/sessions/active` stale-resume → SYNC full-fold + seed + return new sid + stale
  selection (1c-A); **all FIVE `get_or_hydrate_session` handlers (`/chat`, `/stream`, `/sessions/close`,
  `/sessions/snapshot`, `/state`) rebind to the resolved `sess.session_id` for every downstream op** (backend
  call, `in_flight_locks` key, close cleanup `:2835-2836`, `/state` in-flight, echoed id) — the round-5 F2
  one-sweep close of the redirect class, guarded structurally by C21.
- `brain/chat/session.py` — **`get_or_hydrate_session` consults `rolled_to.json`** and **full-follows the
  chain to its live successor** (round-6 L-1; visited-set cycle guard, not an arbitrary depth cap; `_LOCK` is a
  reentrant RLock so recursion is safe) before returning None/404 (round-2 P-1 fix). **Hana's `brain/chat/*`
  coordination zone — flag for PR #58.**
- `brain/bridge/supervisor.py` — weekly-cap rollover on the daily tick + quiet-gap boundary (1c-B); drive the
  age-gated cascade pass; own old-buffer deletion in the rollover path.
- `brain/ingest/pipeline.py` — finalize decoupling → extraction-only (stop buffer-delete; keep memory
  extraction) (1c); `remove_session` audit (registry-evict only, not file-delete).
- `brain/ingest/buffer.py` (1c/1d) — `delete_session_buffer` used by the rollover path; `append_archive`
  segmentation + `read_archive` segment-aware reader (1d). (The successor redirect is NOT here — it lives in
  `session.py`'s `get_or_hydrate_session`, the resolution chokepoint; `ingest_turn` runs only after resolution
  succeeds. Round-2 P-1 correction.)
- `brain/chat/engine.py` — head-prefix authoring for the 3-section render (cache-stable lockstep).
- `brain/chat/budget.py` — head-prefix re-parse kept in lockstep with the new render.
- **Read-but-guard (not necessarily edited):** `brain/monologue/ambient.py` (must not starve `monologue_trace`).
- Tests: new fixtures under `tests/` for fold-validation, 3-section render/cache-stability, rollover, archive
  segmentation+reader, migration (exact paths in the plan).

## 8. Non-goals / guardrails

- No merge of `main`; matched-base discipline (Testing's live A/B stays clean).
- No push / no PR / no GitHub-issue edits (orchestrator gates all GitHub text).
- No personal details in any commit message.
- The interim garbage-gate / candidate queue is a **separate** branch (#123) — do not touch it here.
