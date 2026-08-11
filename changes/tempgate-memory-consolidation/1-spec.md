# 1 — Spec: Temporary memory-consolidation gate (Root 2 stopgap) — SEPARATE-QUEUE ARCHITECTURE

> **v2 architecture (owner ruling, Roy, relayed 2026-08-11).** The earlier in-`memories.db`
> `consolidation_state` column + `committed_only` recall filter is **removed**. The candidate/pending
> queue is a **separate store OUTSIDE `memories.db`**; a candidate is **not a stored memory**. See the
> ratification record at the end. This dissolves the "terminal fate of a rejected candidate" axis that
> bounced the plan-red-team three times: a rejected candidate was never a DB row, so grief / forgetting
> / hebbian — all `memories.db` concepts — never apply to it.

## Problem

The companion's memory store has **no separator between generated and genuine memory**. Every
automatic engine (monologue, dream, reflex, research, heartbeat, initiate, conversation
auto-ingest) writes straight into the same `memories.db` that the recall channel reads from, so
generated content re-surfaces into the next prompt, the engines read it and generate more, and the
volume snowballs — a generated-memory **flood** (Root 2 of the memory/dream rework).

## The architecture (owner-pinned)

A **separate pending-candidate queue**, outside `memories.db`. Everything the system writes
**automatically** is **enqueued** to this pending store — it does **not** become a `memories.db`
row. A consolidation gate on the idle heartbeat tick drains the queue and, per candidate, either:

- **REJECT** → **discard the entry from the pending queue.** No graveyard, no grief, no hebbian
  cleanup — those are `memories.db` concepts and a pending item was never a memory.
- **PROMOTE** (survivor / genuinely new) → **`store.create()` it into `memories.db`** at that point.
- **MERGE** (near-duplicate that adds info) → **archive a lossless pre-image of the existing
  `memories.db` memory**, apply the **smallest surgical edit** to that memory, then **discard the
  candidate** from the queue.
- **CORRECTION / CONTINUATION** → promote the candidate into `memories.db` and seed a **weight-5
  hebbian edge** to the referent memory (both now real `memories.db` rows).

**Recall reads `memories.db`, which only ever holds genuine + promoted memories.** No recall filter
is needed or added — candidates are simply never in the store recall reads. The forgetting/decay
pass only walks `memories.db`, so it never touches the queue; no exemption logic is needed.
**Pass 1 / Pass 2 read `memories.db` read-only** to compare candidates against existing memories.

### Why gate-at-commit, not filter-at-read (rejected alternative, unchanged)

A read-side "hide monologue types at recall" filter is a bandaid: the garbage stays in the store
and the re-generation snowball keeps spinning. The separate queue is the strongest form of
gate-at-commit — the generated content is not in `memories.db` at all until vetted.

## The gate (two passes), on the idle heartbeat tick

- **Drain (fast, under the queue lock):** read all pending entries and truncate the queue file;
  process the drained batch **lock-free**. New candidates enqueued during processing land in the
  now-empty queue and are picked up next tick (the proven soul-candidate read-modify-rewrite
  pattern — `brain/soul/review.py:532-540` holds `file_lock` over the queue's r-m-w window).
- **Pass 1 — free/automatic (firehose → garden-hose), on the drained batch:**
  - **Drop exact repeats** (identical after normalizing whitespace + case) — within the batch and
    against existing `memories.db` memories (read-only compare). Exact-identical text can't mean two
    different things, so dropping it is always safe (the twin/genuine memory carries the content).
  - **Drop low-salience filler** — **scoped to the monologue-EPISODE types only**
    (`SALIENCE_ELIGIBLE_TYPES = {monologue, monologue_emotion, monologue_soul_candidate}`), which
    carry a real 0..10 importance signal (`extractor.py:344` sets `importance = w.salience * 10`).
    **All other gated types are EXEMPT from salience-drop** and go through **dedup only** (Pass 1
    exact-dup + Pass 2 semantic). This is the owner-directed salience guard (Roy, sign-off
    2026-08-11): a single flat floor is wrong because dreams set no importance → it auto-derives as
    `sum(emotion_intensities)/10` (`store.py:126`; verified `dream.py:266-277` passes no
    `importance`), **near-ZERO for emotionally-flat dreams**; `research`/`heartbeat`/`reflex`/
    `initiate` use the same emotion-derived default; `monologue_trace` is pinned at **0.3**
    (`trace.py:17`, explicitly "not part of the salience formula"). A blanket floor would nuke flat
    dreams + traces while keeping high-emotion noise — so salience-drop is **structurally confined**
    to the episode types (independent of the floor magnitude, which stays conservative/OFF pending
    L-B). Threshold still `SALIENCE_FLOOR` (default OFF, see Tuning); it only ever applies to
    `SALIENCE_ELIGIBLE_TYPES`.
  - Compute a **cheap fragment-similarity** used **ONLY to cluster** related candidates for Pass 2.
    It **never deletes anything non-identical** ("the dog ate cat food" vs "the dog ate the cat as
    food" share nearly all fragments but mean opposite things).
  - "Drop" here = **discard the pending entry** (it was never a memory; no side effects).
- **Pass 2 — Haiku, per cluster (injected classifier; tests inject a stub):** given a related batch
  of candidates + the existing `memories.db` memories they resemble, decide per candidate:
  `duplicate`→discard; `merge`→archive pre-image + surgical edit the existing memory + discard
  candidate; `distinct`/`new`→promote (`store.create`); `correction`→promote + weight-5 edge to the
  corrected memory, flagged as a correction (never silently merged); `continuation`→promote +
  weight-5 edge to part A. Split rule: compatible added info consolidates (merge); conflicting info
  binds-but-stays-visible (correction, linked not merged).
- **Ordering:** the gate runs **first** on the idle tick — before reflex/dream/research
  (`brain/engines/heartbeat.py` `run_tick`, after `persona_dir` is resolved ~line 489, before reflex
  at line 491) — so each idle cycle consolidates the accumulated candidates, then the generative
  engines run and produce the next cycle's candidates.

## The pending queue (new component)

- **`brain/memory/pending.py` — NEW `PendingQueue`** over `persona_dir / "pending_candidates.jsonl"`,
  guarded by the existing `brain/utils/file_lock.py` `file_lock` (the proven separate-queue pattern
  used for `soul_candidates.jsonl`). Operations: `enqueue(mem, *, source)` (append
  `mem.to_dict()` + provenance under lock); `read_recent(memory_type, limit)` (lock-free read via
  `read_jsonl_skipping_corrupt`, newest-first — for interior-continuity); `drain()` (under lock:
  read-all + truncate, return entries). A candidate carries the same fields as a `Memory`
  (`to_dict`), so promotion is `store.create(Memory.from_dict(entry))`.
- A candidate is transient JSONL, **never a `memories.db` row** — so it never appears in recall,
  forgetting, hebbian, graveyard, or any `list_by_type` main-DB read.

## The write seam — automatic writers ENQUEUE (the new choke behavior)

Under v1 the choke point was `store.create` defaulting to candidate. Under v2 the automatic firehose
is **enqueued instead of written to `memories.db`**. **Routing is by `memory_type`, NOT by call site**
(the round-4 red-team MAJOR: `engines/reflex.py:452` emits a *variable* `arc.output_memory_type`, and
one configured value is `journal_entry` — a **deliberate** type consumed by
`chat/prompt.py:1079`; a site-level "reflex → enqueue" rule would wrongly gate journal entries and
strip them from the weekly self-narrative block). So a candidate is enqueued **iff its
`memory_type ∈ GATED_TYPES`**, else it is `store.create`d directly. This makes the partition robust to
the two variable-typed sites (`reflex.py:452` `arc.output_memory_type`; `ingest/commit.py:69`
`item.label`).

Realization: a helper `route_write(store, pending, mem, source)` — `if mem.memory_type in
GATED_TYPES: pending.enqueue(mem, source) else: store.create(mem)`. Each **automatic** `store.create`
site is switched to `route_write` (so a variable-typed site self-classifies per row); deliberate
sites keep `store.create`. `MemoryStore` gains a `db_path`/`persona_dir` accessor so function-style
writers can locate the queue.

**`GATED_TYPES` (the generated firehose that floods recall) — conservative default, and the SCOPE
knob for owner sign-off:** `monologue_trace`, `monologue`, `monologue_emotion`,
`monologue_soul_candidate`, `dream`, `research`, `heartbeat`, plus the reflex arc output types that
are *not* `journal_entry`, plus the gated conversation-ingest labels. **Excluded (write direct):**
`journal_entry` (deliberate; feeds `prompt.py:1079`), and — by default — `initiate_outbound` (its own
dedup at `initiate/memory.py:105` needs immediate `memories.db` visibility, else a double-send; see
consumer section). The exact membership is the **scope decision** surfaced at sign-off
(full-firehose vs monologue-family-only). Fail-safe direction: a type wrongly *excluded* lets
un-vetted content into recall (defeats the change) → the `GATED_TYPES` set is the fidelity item the
red-team/owner must confirm; a type wrongly *included* only delays/over-gates (recoverable).

**Automatic write sites switched to `route_write`:** `monologue/trace.py:31`,
`chat/extractor.py:348/408/546`, `engines/dream.py:277`, `engines/reflex.py:462` (self-classifies:
`journal_entry` → direct, other arc types → gated), `engines/research.py:327`,
`initiate/memory.py:156` (`initiate_outbound` excluded by default → direct), `engines/heartbeat.py:943`,
`ingest/commit.py:69` (self-classifies by `item.label`).

**Deliberate / import / restore → `store.create` direct (unchanged):**
`tools/impls/add_memory.py:71`, `tools/impls/add_journal.py:51`, `body/events.py:75`,
`migrator/*` (`emergence_kit.py:265`, `cli.py:183`), `recovery/engine.py:128/132`,
`files/commit.py:23`, `grief/breadcrumb.py:138`, `self_model/reconcile.py:123`/`resolve.py:157`,
`maker/wiring.py:69`. (`soul_store` / `kindled_link` are separate stores — out of scope.)

## The consumer-migration surface (NEW — the primary risk this architecture introduces)

Pulling candidates out of `memories.db` changes what **every** main-DB `list_by_type` reader of a
firehose type sees — not just recall. Each such consumer now sees **only promoted** items of that
type. Enumerated (verified by grep), with the resolution for each:

- **Interior-continuity — MUST read the queue (hard constraint).** `monologue/ambient.py:34`
  (`build_interior_continuity_block`) and `monologue/recall.py:29` read recent `monologue_trace` from
  main DB via `list_by_type`. Under v2 traces are in the **queue**, so these read
  **`PendingQueue.read_recent(MONOLOGUE_TRACE_TYPE, limit)`** or continuity **starves**. Faithful
  realization (the queue *is* the recent-raw short-term substrate). **Owner directive (Roy, sign-off
  2026-08-11): KEEP interior-continuity AS-IS for this run** — do **NOT** change `_AMBIENT_LIMIT`
  (stays 5) or the block's semantics; the **only** change is the read-source move (queue instead of
  `list_by_type`), which is necessary because `monologue_trace` is now gated. The interior-continuity
  rework (ultimately ≤2 ephemeral thoughts) is **deferred to Phase 4** (the monologue-system rework).
  So this consumer is **behavior-preserving except the read-source for steady-state content** — with
  one owner-accepted timing change: because the gate drains the queue on the idle tick, immediately
  after a drain the block reads a near-empty queue (v1 read the last 5 traces persisted in the DB).
  That is the R2/R3-accepted "short-term = live buffer" effect, rework deferred to Phase 4. Note
  (round-4): `recall.py:42`'s
  `store.get(id)` "keep-sharp" `recall_count` bump becomes a **no-op** for a queue trace (its id is
  not a `memories.db` row) — harmless (no crash), the side effect is dropped; accepted. Residual: an
  idle-tick drain clears the queue, so recent traces are briefly gone for a resumed session —
  acceptable (short-term = live buffer + tiered compaction, R2/R3).
- **`chat/prompt.py:1079` weekly self-narrative (`journal_entry`) — UNAFFECTED by construction.**
  `journal_entry` is **excluded** from `GATED_TYPES`, so reflex- and tool-authored journal entries
  write direct to `memories.db` and still appear here. (This is the round-4 MAJOR, resolved by
  type-based routing.)
- **Initiate dedup — RESOLVED by default.** `initiate/memory.py:105` reads `initiate_outbound` from
  main DB to avoid a repeat send. Default: `initiate_outbound` is **excluded** from `GATED_TYPES`
  (writes direct), so the dedup sees every outbound → **no double-send**. (Alternative, if the owner
  wants outbounds gated: initiate's dedup must also read the queue — C17a tests whichever is chosen.)
- **Bridge feed (`brain/bridge/feed.py`) — behavior change, FLAG for sign-off.** With the default
  `GATED_TYPES`, `dream` (:99) and `research` (:247) feeds show only **promoted** (consolidated)
  items; `initiate_outbound` (:174) and `file_write` (:135) are **unaffected** (not gated /
  deliberate). Whether the feed should show only consolidated dreams/research (arguably an
  improvement) or all raw items is a **product decision** — surfaced at stage-4 sign-off (tied to the
  scope knob).
- **`research.py:518/522` self-dedup (ADVISORY).** Research reads main DB to avoid re-researching a
  topic; a topic sitting un-promoted in the queue is invisible → at most one wasted research cycle.
  Benign; noted for completeness.
- **Crystallizers / interest-sweep / notes / maker** (`growth/crystallizers/*`,
  `engines/interest_sweep.py:44`, `notes/runner.py:30`, `maker/sources.py:55`,
  `heartbeat.py:982/1014`) — operate on genuine memory; reading **promoted-only** is benign-to-correct.
- **`chat/extractor.py:39`'s `list_by_type("monologue")`** and `initiate/ambient.py:92`
  (`conversation`) — re-check at build whether they need pre-promotion visibility.
- **Soul-crystallization source deref (`monologue_soul_candidate`) — accepted fail-soft.** Gating
  `monologue_soul_candidate` (per the BROAD scope) means the placeholder's `mem.id` handed to
  `queue_soul_candidate` (`extractor.py:540-556`) is in the pending queue, so `soul/review.py:314-323`
  `_source_memory_snippet`'s `store.get(id)` returns `None` at review time → the reviewer loses the
  DB source snippet. **Fail-soft** (no crash; the soul candidate's own `text` field carries the
  content). Accepted within the confirmed BROAD scope; flagged to the owner (if it matters,
  `monologue_soul_candidate` can be excluded from `GATED_TYPES` — a one-line refinement, since it is a
  soul-queue placeholder, not recall-flood content).
- **`monologue/recall.py` (`recall_monologues` tool) — migrates to the queue too.** It reads recent
  `monologue_trace` from the DB (`recall.py:29`, `limit=None` token-search); under v2 it reads the
  queue via `read_recent`. Consequence: it sees un-promoted recent traces (the common case); a
  promoted trace (rare) is missed — accepted. C3 covers **both** `ambient.py` and `recall.py`.

**Scope option for the owner (surfaced, not decided):** the brief's scope is the *full* firehose. If
the consumer-migration blast radius above is undesired, a **narrower scope** — route only the
**monologue family** (`monologue_trace` / `monologue` / `monologue_emotion` /
`monologue_soul_candidate`, the actual diagnosed "monologue bleed") through the queue and leave
`dream`/`research`/`reflex`/`initiate` writing direct to `memories.db` — would sharply reduce the
blast radius while still cutting the monologue flood. Presented at stage-4 sign-off; default is the
pinned full scope.

## What is KEPT from the red-teamed v1 (survived review)

- **Gate runs first** on the idle tick (ordering above).
- **Non-blocking persona gate file-lock** for the multi-thread `run_tick` (verified: background
  `supervisor.py:1085`, session-close worker `server.py:651`, CLI — separate threads/connections;
  only close-vs-close is debounced). Still required because PROMOTE/MERGE **write `memories.db`**, so
  two overlapping gate runs could double-fold a merge target. Add a `blocking=False` mode to
  `file_lock` (it is otherwise blocking-only). Contended run **skips**.
- **Weight-5 hebbian edges** for corrections/continuations — via a `hebbian.set_edge_weight(a,b,w)`
  UPSERT to `MAX(existing, 5.0)` (bare `ensure_edge` no-ops over a pre-existing ~0.5 edge from
  `ingest/commit.py:86`). Applies to PROMOTED memories (real `memories.db` rows), which is correct.
- **Merge = archive lossless pre-image (plain append-only file, NOT the grief graveyard) + surgical
  edit**; wrap the merge `store.update` in try/except KeyError (a concurrent forgetting `LOSE` could
  delete the target — forgetting runs in the supervisor thread, `supervisor.py:423`, not the tick).

## Constraints (hard)

- **Base branch `ThinkerOfThoughts/memory-rework-groundwork`** (diagnose base `cd29bc61`). **Do NOT
  `git merge main`.**
- **Don't starve interior-continuity** (now realized by reading the queue). **Preserve the
  prompt-cache-stable prefix** (`chat/engine.py:48-55`) — this change does not touch
  `build_static_system_message` or the history-replay prefix (recall block is unchanged now).
- **Mark `TEMP (Root 2 stopgap — remove when Phase 2 relevance overhaul lands)`** in code; file the
  removal as an explicit tracked Phase-2 dependency.
- **No personal details in any GitHub-facing text.** Phoebe is never a fixture; synthetic user =
  **Bob**, persona = **Canary**, model = **Claude**.
- Local CI before done: `uv run ruff check .` + `uv run pytest -m "not live and not
  requires_claude_cli and not integration"` (py3.12). **Commit LOCALLY only** — no push/PR.

## Deferred (do NOT block on)

- **Dedup/merge MAGNITUDE** (salience cutoff — default OFF, and only ever over
  `SALIENCE_ELIGIBLE_TYPES` = the monologue-episode types; fragment unit + cluster threshold; merge
  sensitivity) — gated on the L-B test read. Build parameterized with a conservative default,
  confirmed after L-B. Association weight starts at **5**. The salience **scoping** (episode-types
  only; dreams/research/trace exempt) is **structural, not a magnitude knob** — it holds regardless
  of the eventual floor value.

## Prior art

- `brain/soul/review.py` — the separate-queue-with-`file_lock` precedent (`soul_candidates.jsonl`;
  `_load_soul_candidates`/`_save_soul_candidates` + `file_lock` over the r-m-w window,
  `review.py:274-292,532-540`). The `PendingQueue` mirrors it.
- `chat/compaction.py:376-381` — archive-before-mutate (lossless-before-lossy) the merge step mirrors.
- `brain/memory/hebbian.py` — `strengthen`/`ensure_edge`, `MAX_WEIGHT=10.0` (`:29`).
- `chat/engine.py:48-55` — the cache-stable prefix (untouched here).

## Expected touched files

**New:** `brain/memory/pending.py` (`PendingQueue`).
**Store:** `brain/memory/store.py` — add a `db_path`/`persona_dir` accessor (so writers/queue locate
the persona dir); `hebbian.py` — `set_edge_weight`; `brain/utils/file_lock.py` — `blocking=False` mode.
**Gate:** `brain/engines/consolidation.py` (NEW) — drain + Pass 1/Pass 2 + promote/merge/reject +
gate lock; `brain/engines/heartbeat.py` — call it first in `run_tick`.
**Writers → enqueue:** `monologue/trace.py`, `chat/extractor.py`, `engines/{dream,reflex,research}.py`,
`initiate/memory.py`, `engines/heartbeat.py` (its own memory write), `ingest/commit.py`, plus the
plumbing to give each a `PendingQueue`/persona_dir.
**Consumers → queue:** `brain/monologue/ambient.py`, `brain/monologue/recall.py` (read traces from the
queue). Others (`feed.py`, `initiate/memory.py:105`, crystallizers…) per the consumer-migration
section — resolved or flagged.
**Tests:** `tests/` per criterion (see 1.5).
**Non-code:** the Phase-2 removal dependency record.

**NOTE — no recall-channel edits.** `brain/chat/prompt.py` `_build_recall_block`,
`brain/forgetting/recall.py`, `brain/tools/impls/search_memories.py` are **unchanged** (the
`committed_only` filter is removed from scope). Recall/forgetting are untouched by construction.

## Stage-4 owner sign-off — RESOLVED (Roy, 2026-08-11; see R3)

1. **Scope = BROAD (confirmed).** Full automatic firehose gates: monologue family + `dream` +
   `research` + `heartbeat` + non-`journal_entry` reflex/ingest output. `journal_entry` + deliberate
   writes bypass (direct). Type-based routing kept.
2. **Feeds = consolidated-only (accepted).** `feed.py` dream/research keep reading `memories.db` →
   promoted-only. No code change (C17b resolved).
3. **Interior-continuity = keep AS-IS** (`_AMBIENT_LIMIT` stays 5; semantics unchanged; only the
   read-source moves). The ≤2-thought rework is deferred to Phase 4.
4. **Salience guard (directive 4)** applied above — salience-drop confined to `SALIENCE_ELIGIBLE_TYPES`
   (episode types); dreams/research/trace exempt; new criterion C18 proves legitimate content survives.
5. **One extra cold spec pass** on the round-4 fixes + this salience change runs before build
   (directive 5).
(The reject-fate axis is RESOLVED by R1; within-session recall behavior is RATIFIED by R2.)

---

## Ratification records (RAT1)

### R1 — Architecture: candidate queue is a SEPARATE store outside `memories.db`
- **Flagged axis:** where the candidate/pending state lives, and what "reject a candidate" means
  against grief/forgetting/hebbian (the axis that bounced the plan-red-team 3×).
- **Owner response (Roy, verbatim intent as relayed by the coordinator, 2026-08-11):** "the
  pending/candidate queue is a SEPARATE store, OUTSIDE the main memory database. A candidate is NOT a
  stored memory. REJECT = just discard the entry from the pending queue. No graveyard, no grief, no
  hebbian cleanup… PROMOTE (survivor) = write it into the main memory DB… MERGE = archive a lossless
  pre-image of the existing main-DB memory, apply the surgical edit to it, then discard the candidate…
  RECALL reads the main memory DB… the committed_only / consolidation_state recall-filter apparatus is
  UNNECESSARY; remove it… The forgetting/decay pass only walks the main DB, so it never touches the
  queue — no exemption logic needed."
- **Durable source:** the coordinator ruling message received during this gc run, 2026-08-11 (relaying
  Roy, "OWNER RULING (Roy, confirmed)"). Locus: this run's inbound coordinator message.
- **Mapping:** selects a **fourth option** beyond the three v1 options (discard / inert-held /
  decay) — *the candidate is never a memories.db row at all*, which dissolves the axis. This spec
  implements exactly that. **Elaboration audited (RAT2):** the operative terms added here (JSONL
  `PendingQueue` + `file_lock`, enqueue-at-writers, read-recent-from-queue for interior-continuity)
  are realizations of Roy's "separate store outside the DB" and "candidate is not a stored memory";
  none adds a commitment beyond his stated mechanism. The consumer-migration consequence is flagged
  (not silently resolved) for sign-off.

### R2 — Within-session recall behavior is intended (ratified downward per owner instruction)
- **Owner response (Roy, relayed 2026-08-11):** "the within-session behavior — content formed
  mid-session doesn't re-enter recall until the next idle consolidation — is INTENDED (short-term
  memory = the live conversation + tiered compaction). Write it into the spec as confirmed."
- **Source:** same coordinator ruling, 2026-08-11. Owner instructed **not** to re-ask; recorded as
  confirmed. (Was OPEN #2 in v1; now closed.)

### R3 — Stage-4 sign-off directives (Roy, relayed 2026-08-11)
- **Flagged axes:** scope of `GATED_TYPES`; bridge-feed behavior; interior-continuity change extent;
  the salience-guard design.
- **Owner response (Roy, verbatim intent as relayed by the coordinator, 2026-08-11):** "SCOPE =
  BROAD… The full automatic firehose gates: monologue family + dream + research + heartbeat +
  non-journal reflex/ingest. journal_entry and deliberately-created memories… BYPASS the gate…
  FEEDS: consolidated-only is accepted… INTERIOR-CONTINUITY: KEEP AS-IS for this run. Do NOT change
  `_AMBIENT_LIMIT`… deferring the interior-continuity rework… to Phase 4… SALIENCE GUARD… The Pass-1
  salience cut must NOT wholesale-discard legitimate gated content… Extend your existing
  destructive-salience guard to cover DREAMS + monologue_trace, and either calibrate the salience cut
  PER-TYPE or scope salience-drop to just the flood/monologue types (letting dream/research go
  through dedup only)… Add a measurable CRITERION proving legitimate dream/research/trace content
  survives… run the one extra cold spec pass on the round-4 fixes… then proceed."
- **Durable source:** the coordinator sign-off message received during this gc run, 2026-08-11
  ("OWNER SIGN-OFF (Roy)"). Locus: this run's inbound coordinator message.
- **Mapping:** scope→BROAD `GATED_TYPES` (kept type-based routing); feeds→consolidated-only (no
  change); interior-continuity→read-source move only, `_AMBIENT_LIMIT`=5 preserved, rework→Phase 4;
  salience→scoped-to-episode-types design + C18. **RAT2:** the elaboration (scope salience to the
  episode types, exempt dreams/research/trace) is one of the two options Roy explicitly offered
  ("scope salience-drop to just the flood/monologue types, letting dream/research go through dedup
  only") — no inflation beyond his stated choice.
