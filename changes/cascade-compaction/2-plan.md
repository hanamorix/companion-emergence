# 2-plan.md — Cascade Compaction

Build plan for the spec (`1-spec.md`) against the criteria (`1.5-criteria.md`). **Measurement posture:** no
replay/held workload exists on this VM (no live bridge), so regression metrics are **advisory** (A1); the
gating criteria (C1–C13) are verified by **scratch pytest fixtures** that exercise the real code paths — this
is the route-(a) representative pre-ship harness (stage-8 H5). Each such fixture is itself subject to H6 (must
be shown able to fail on the pre-change / known-bad state) before its PASS counts.

---

## 0. Three open engineering points — RESOLVED (documented consts; not escalated)

### Q8 — 24h-section hard cap → `_SECTION_24H_CHAR_CAP = 12_000` (chars)
**Reasoning.** The soft target is 60% of the raw batch; the cap only bites when a huge day makes 60%-of-raw
exceed the ceiling. 12 000 chars ≈ 2 000 words ≈ ~3 000 tokens. (The full-head worst case, with the tier-3
hard cap, is computed once below — ≈19.2 k chars — not repeated here.) This sits comfortably inside the live
prompt budget (`engine.respond` caps the assembled prompt at `max_tokens=80_000` with 40 preserved tail turns,
`budget.py`/`engine.py:238-245`), so even a pathological day cannot let the head crowd out the live tail.
**Enforcement:** the 24h fold requests `target_words = min(0.60·raw, cap-derived word budget)`, and the fold
validator (C4 path) additionally rejects+re-requests (then truncates at a sentence boundary) any 24h output
exceeding the char cap. **Tier-3 hard cap (OWNER 2026-08-13):** because tier 3 is terminal (re-compacted
forever), it gets its own hard ceiling on top of the 20% compaction: `_SECTION_72H_CHAR_CAP = 0.20 ×
_SECTION_24H_CHAR_CAP` (= 2 400 chars at the 12 000 default), enforced the **same way** (sentence-boundary
truncation + validator re-request). **Tier 2 stays bounded-by-input** (no separate hard cap): it is transient
(graduates to tier 3 each cycle) and its input is the tier-1-capped section; the only unbounded-ish case is a
one-time multi-day-gap catch-up where `G48` is a full day of raw, which self-corrects next cycle as it
graduates into the capped tier 3 — a documented accepted transient (owner leaned against a tier-2 cap absent a
concrete reason). Worst-case steady head ≈ 12 000 (t1) + ~4 800 (t2, 40% of t1) + 2 400 (t3 capped) ≈ 19.2 k
chars ≈ ~4.8 k tokens, well inside the 80 k prompt budget. **Follow-up (open-Q8):** measure the average char
length of a real 24h conversation on live data and re-tune the cap; the const carries a `# FOLLOW-UP(open-Q8)`
comment.

### Weekly-rollover fire-point → driven from the DAILY SUPERVISOR TICK at a quiet-moment boundary
**(Revised per stage-3 F2/M1 — the daily-cadence MECHANISM is brief-pinned (§1c-B line 68 "checked on the
daily cadence"); attach-only firing was a defect because trigger B's entire target population is the
continuously-used client that never re-attaches.)**

**Consts:** `_WEEKLY_ROLLOVER_AGE = timedelta(days=7)`; `_ROLLOVER_QUIET_GAP = timedelta(minutes=30)` (the
engineering-default boundary I own — "quiet moment, not mid-exchange"). On the **daily supervisor tick**
(`_run_compaction_tick`, already holding the per-session compaction lock), if a session's age
(oldest-turn → now) ≥ `_WEEKLY_ROLLOVER_AGE` **and** its last turn is older than `_ROLLOVER_QUIET_GAP` (nobody
mid-exchange), **execute the weekly rollover then and there** (under the lock): seed a new session with the 3
tiers + 40 messages, archive+delete the old buffer (§2.3 L2), and write a **successor pointer**
`rolled_to.json` on the old session id. If the last turn is *within* the quiet gap, defer to the next tick
(re-checked daily).

**Successor pointer (`rolled_to.json`) closes the continuous-client repoint race — redirect installed at the
REAL locus (revised per stage-3 round-2 P-1 BLOCKER).** The round-1 draft claimed `ingest_turn` consults the
pointer, but a continuous client's next `POST /chat` (`server.py:2360-2366`) / `WS /stream`
(`server.py:2410-2428`) calls **`get_or_hydrate_session`** (`session.py:129-205`) FIRST and 404s
(`server.py:2365-2367`) before `ingest_turn` is ever reached — so the redirect never ran and the turn was
lost. **Fix (option a — server-contained, transparent):** `get_or_hydrate_session` consults `rolled_to.json`.
When the requested sid is neither in `_SESSIONS` nor has buffer turns (the exact post-swap state: evicted +
deleted), it reads `rolled_to.json` for that sid and, if present, **follows the pointer chain to its END**
(**full-follow — stage-3 round-6 L-1**, not a single hop): a client dormant across *many* weekly rollovers
(sid1→sid2→sid3→…) must still resolve to the **current** successor, and the pointer reads are cheap. Follow
until a sid has no `rolled_to.json` (the live successor), guarding against a **corrupt cyclic pointer** with a
**visited-set cycle check** (NOT an arbitrary small depth cap — a legitimately long chain must resolve; only a
true cycle aborts, to a 404-then-reattach fallback). Returns the terminal **successor** `SessionState`
(session_id == the current sid). **Close the whole redirect class in ONE sweep (stage-3 round-5 F2 — stop the per-round whack-a-mole).**
`get_or_hydrate_session` has **FIVE** call sites (grep `get_or_hydrate_session(` in `server.py`):

| Site | Handler | Raw id used downstream where it must be resolved? |
|---|---|---|
| `:1261` | `GET /state/{session_id}` | **YES** — `in_flight` lookup uses the raw path `session_id` (`:1265`) → misreports in_flight for a redirected old sid (the `session_id` field already echoes `sess.session_id`). It is **C20's own oracle**, so it must be correct. |
| `:2365` | `POST /chat` | backend uses `sess` (ok); **lock key + response echo** use `req.session_id`. |
| `:2424` | `WS /stream` | uses the raw path `session_id` for lock key. |
| `:2696` | `POST /sessions/snapshot` | backend `_snapshot_session_blocking` + lock use `req.session_id` (F1 no-op). |
| `:2753` | `POST /sessions/close` | backend + lock + **cleanup `remove_session`/`in_flight_locks.pop`** (`:2835-2836`) use `req.session_id` (F1 no-op + G1 leak). |

**Chosen fix — uniform rebind + a structural guard (so round 6 cannot find a 6th site):** immediately after a
successful `get_or_hydrate_session`, each handler **rebinds its local id to the canonical `sid = sess.session_id`
and uses `sid` for EVERYTHING downstream** — backend pipeline call, `in_flight_locks` key, `remove_session`,
`/state` in-flight lookup, and any echoed `session_id`. This one mechanical rule covers all five current sites
and any future caller. To make it enforceable (not just enumerated), **criterion C21** is a **structural check**
over `server.py`: for every handler body that calls `get_or_hydrate_session`, no raw request/path `session_id`
may be used downstream for those load-bearing operations — the resolved `sess.session_id` must be. (A stronger
type-level root fix — wrapping the id in a `NewType` and threading the canonical value so a caller *cannot name*
the stale id — was considered and rejected as a larger, riskier refactor across the bridge for this change; the
uniform-rebind rule + the C21 structural guard gives the same "round-6-can't-find-a-6th-site" guarantee at a
fraction of the surface. Recorded as the engineering call.) Consequences this closes:
- **F1 (close/snapshot silent no-op):** with the rebind, `/sessions/close` + `/sessions/snapshot` operate on
  the successor, not the deleted old buffer's empty-guard (`pipeline.py:150-154`/`:326`) → no false-success.
- **F2 (`in_flight_locks` key-split, `server.py:816`):** all sites key by the resolved sid, so old-sid +
  successor-sid traffic serialise on one key.
- **G1 (close cleanup leak, `:2835-2836`):** `remove_session`/`in_flight_locks.pop` use the resolved sid → no
  successor registry/lock leak.
- **/state (round-5):** `in_flight` reads the resolved sid → correct report (and it is C20's oracle).

The engine writes turns to the successor buffer (`engine.respond` uses the returned `sess.session_id`); a
handler's response may still echo the client's original sid where that is what the client sent (the client
keeps its sid, each turn re-reads the cheap pointer) — transparent, **no client contract change**. The pointer
is written atomically under the compaction lock **before** the old buffer is deleted, so there is no window
where the old sid resolves to nothing. **`_LOCK` is a reentrant `RLock` (`session.py:97`)**, so the full-follow
pointer chase inside `get_or_hydrate_session` is deadlock-safe; a **visited-set cycle guard** (not a depth cap)
aborts only a corrupt cyclic pointer (→ 404-then-reattach), while a legitimately long chain resolves.
**Touched-files:** `brain/chat/session.py` (the resolve) **+ `brain/bridge/server.py`** (rebind to
`sess.session_id` at all **five** sites — already a touched file). Both are/border Hana's `brain/chat/*` zone
— **noted for PR #58**.
New criterion **C16** (broadened) asserts the redirect through the **real** path at the write handlers incl.
close/snapshot operating on the successor (not a false-success no-op); **C19** asserts the resolved-sid
lock keying (two clients, one lock key); **C20** the close-cleanup (G1); **C21** the structural guard (no
raw-id downstream use in any `get_or_hydrate_session` handler) so the class stays closed; **/state** in-flight
is covered as part of C20's oracle correctness.

### Seed asymmetry → confirmed intentional, encoded in the two call sites
1c-A (idle >24h): `compact_conversation(..., min_keep_tail=0)` — fold **everything**, seed summary-only (a
>24h-old "recent 40" is not recent). 1c-B (weekly): seed the **3 tiers + 40 most-recent messages**
(`min_keep_tail=40`). The asymmetry is a direct consequence of *why* each fired (staleness vs. size) and is
documented at both sites.

---

## 1. Representation & core mechanics

### 1.1 Sectioned summary row
The single `summary` row gains structured sections in its `compaction` meta; `text` becomes a deterministic
render of them.

```
summary_row = {
  "session_id", "speaker": "summary", "ts",
  "text": <deterministic render, see 1.2>,
  "compaction": {
     "gen": int,                       # unchanged: monotonic
     "folded": bool,                   # unchanged
     "covers_until_ts": str,           # unchanged (newest covered ts, back-compat)
     "sections": {                     # NEW
        "24h": {"text": str, "covers_from_ts": str, "covers_until_ts": str},
        "48h": {"text": str, "covers_from_ts": str, "covers_until_ts": str},
        "72h": {"text": str, "covers_from_ts": str, "covers_until_ts": str},
     },
  },
}
```
A section absent/empty (early life, or a quiet day) is represented by an omitted/empty entry and rendered as
nothing. **Legacy rows (no `sections`) default to TIER 3 with an explicit OLD-FLOOR `covers_from_ts`
(stage-3 round-6 MO-1 / round-7 MO-2 — critical, structural).** A legacy single-layer summary is *accumulated
history* (often months old); the oldest-edge classifier (§1.3) keys on the `covers_from_ts` **value**
(age = `now − covers_from_ts`), so a *recent* value reclassifies the blob back to tier 1 ("yesterday") on the
next pass — the #82 defect this phase kills. The structurally-correct fix: the tolerant reader reads a legacy
row as `sections={"72h": {text: <legacy text>, covers_until_ts: <existing>, covers_from_ts: <now −
`_LEGACY_AGE_FLOOR`>}}` where **`_LEGACY_AGE_FLOOR = timedelta(hours=96)` (>72h with margin) — set
UNCONDITIONALLY, never falling back to `covers_until_ts`.** Because the classifier buckets age >72h as terminal
tier 3, an old `covers_from_ts` **structurally guarantees** the blob lands in and STAYS in tier 3, pass after
pass. The exact age of months-old summarized content is both unknowable and irrelevant — "old" is the correct,
safe answer. This makes old rows behave correctly before migration runs (defensive); migration formalizes the
same mapping (§4, C12).

### 1.2 Deterministic render (temporal markers #82; cache-stability C6)
`_render_sections(sections) -> str`: fixed order tier1 → tier2 → tier3; each non-empty section emitted as
`"[<label> — <coarse span>] <text>"`. The `<label>` is the **owner-specified human age-band label** (OWNER
2026-08-13): tier1 (24h)=**"yesterday"**, tier2 (48h)=**"day before yesterday"**, tier3 (72h)=**"a few days
ago"** — a static tier label (NOT a computed date), so it stays byte-stable. `<coarse span>` is derived from
the section's `covers_from_ts`/`covers_until_ts` at **coarse** granularity (date, or date+part-of-day), never
a per-render clock read. No nonces, no `now()` in the render. `engine._buffer_turns_to_messages` wraps the whole render in the unchanged
`f"[Earlier in this conversation: {summary_text}]"` head prefix (`engine.py:405-408`); `budget.py` still
matches only `_COMPACTION_SUMMARY_PREFIX` and preserves the whole block (`budget.py:28-31,94-109`) — **the
prefix string is unchanged, so budget's contract is untouched.** Determinism is what makes the head
byte-stable between re-compactions (C6a); the prefix-invariance is what keeps authoring↔re-parse in lockstep
(C6b).

### 1.3 Cascade = true AGE-GATED GRADUATION (NOT a tick-shift, NOT prior-tier re-fold) — robust to skips AND to continuous use
**(Revised per stage-3 round-1 F1/L1 and round-3 F3 — two distinct failure modes of a proxy-for-age.)**
Two things a correct mechanism must satisfy at once: **(i)** after a skipped/multi-day tick, material lands in
the tier its ACTUAL age dictates (round-1: the cadence fires **once** after a gap, not once-per-missed-day —
`supervisor.py:635-662`, `persisted_cadence.advance` — so a fixed one-position shift would dump 3 days of raw
into "24h"); **(ii)** under continuous daily use with no gap, each cohort **graduates from tier 1 down toward
tier 3 by true content age** (round-3 F3: classifying by the *newest* edge and re-folding the prior 24h section
with fresh raw refreshes that edge forever, so genuinely-old material is perpetually re-labelled tier-1
"recent" — reproducing Problem-1 one tier down). The mechanism below classifies by the **oldest** covered edge
and **never co-folds the prior 24h section with fresh raw**, satisfying both. **Tier 3 is terminal-with-fade**
(OWNER 2026-08-13): material graduates raw→tier1→tier2→tier3 and then stays in tier 3, re-compacted each cycle
(step 4) — there is no evict-out-of-summary leg.

New function `cascade_conversation(persona_dir, session_id, *, provider, now=None, min_keep_tail=40)`, run by
the daily supervisor tick in place of the current single fold. Boundaries as `timedelta`s from `now`:
`_AGE_24H=24h`, `_AGE_48H=48h`, `_AGE_72H=72h`. Under the existing per-session compaction lock:

1. **Read the pre-pass state:** the existing sections (each with its `covers_from_ts`/`covers_until_ts`) and
   the eligible raw turns (extracted, beyond `min_keep_tail`, `ts ≤ now − _AGE_24H`).
2. **Age-partition the eligible raw turns** by `now − ts`:
   `G24 = (24h,48h]`, `G48 = (48h,72h]`, `G72 = (>72h]`. (Turns ≤24h stay raw/live.) This is what makes a
   post-sleep pass correct: 3-day-old raw turns go straight to `G72`, not into "24h".
3. **Age-classify each existing section by the age of its OLDEST covered ts** (`now − covers_from_ts`) — NOT
   the newest edge (revised per stage-3 round-3 F3). `bucket_of(sec)`: `24h` if `age ≤ 48h`, else `48h` if
   `age ≤ 72h`, else `72h`. Using the oldest edge is what makes a cohort **graduate** on schedule: the oldest
   edge only advances (a cohort's oldest material never gets younger), so a section built as "24h" a day ago
   has `covers_from_ts` age ≈ 48h now → it classifies into the **48h** band and graduates. (The newest edge
   was wrong precisely because step 4 could refresh it — see the anti-merge rule next.)
4. **Assemble oldest-first — never co-fold the prior 24h section with fresh raw (F3), tier 3 is TERMINAL
   (OWNER 2026-08-13).** The prior 24h section is a *cohort*; under continuous daily use it has aged into the
   48h band (step 3) and graduates there, NOT re-merged with today's fresh raw (which would refresh its newest
   edge and launder stale content as "recent" — the F3 defect). Each tier = recompact/fold of **(the sections
   that classify into that band by step 3) + (that band's raw age-group)**:
   - `new_72 = recompact( [secs bucket==72h] + G72 , _FRACTION_72H=0.20 )`, bounded by `_SECTION_72H_CHAR_CAP`
   - `new_48 = recompact( [secs bucket==48h] + G48 , _FRACTION_48H=0.40 )`  *(bounded-by-input; no hard cap)*
   - `new_24 = fold(     [secs bucket==24h] + G24 , _FRACTION_24H=0.60 )`, bounded by `_SECTION_24H_CHAR_CAP`.
   **Under steady daily use `[secs bucket==24h]` is empty** (yesterday's 24h cohort graduated to the 48h band),
   so `new_24 = fold(G24)` — only genuinely just-crossed-24h material — and yesterday's 24h becomes `new_48`.
   Material therefore *graduates* raw→tier1→tier2→tier3 by true content age (the owner's design,
   New_mem_system.md Part 1). **Tier 3 (72h band) is TERMINAL** — there is **no 4th tier and no evict-out-of-
   summary leg**: the prior tier-3 section classifies into the 72h band (its oldest edge is >72h) and is
   included in `new_72` *together with* anything newly crossing 72h, so tier 3 is **re-compacted every cycle
   (20% + the hard cap) and the oldest content fades to an ever-briefer trace** — the intended gradual fade,
   not a leak (OWNER: "re-compacted forever" is the design; the round-3 "72h→gone" premise was wrong). Two
   cohorts landing in the same band ARE merged (correct — same true age). Merging reuses the existing
   `fold_existing_summary=True` machinery. Each new section records its `covers_from_ts`/`covers_until_ts`
   from its inputs (temporal markers §1.2).

   **Terminal tier-3 is a MULTI-INPUT fold — the steady-state NORM, specified explicitly (stage-3 round-5
   F1).** Under continuous daily use, EVERY cycle `new_72` folds **two prior texts**: the **persisting prior
   tier-3 section** + the **newly-graduated prior tier-2 section** (both classify into the 72h band), plus any
   `G72` raw. The owner ruled tier-3 content must never be dropped, so this fold uses the **same lossless-
   leaning discipline as the #77 cascade fallback (§1.4)**: **(1)** build the fold input as an explicit
   ordered, lossless-leaning join of ALL inputs (prior tier-3 prose, then the graduated tier-2 prose, then any
   `G72` rendered turns — oldest-first so the fade preferentially compresses the oldest); **(2)** compact that
   joined input to the 20% target; **(3)** enforce `_SECTION_72H_CHAR_CAP` by sentence-boundary truncation;
   **(4)** if the fold double-rejects (§1.4), fall back to the lossless-leaning join truncated to the cap
   rather than dropping either input. This guarantees salient content from BOTH the persisting tier-3 and the
   just-arrived tier-2 survives into the new tier-3 within the cap. (Tier-1 and tier-2 are likewise multi-input
   whenever a band holds a graduated section + a raw age-group; the same join discipline applies, but only
   tier-3 combines two *sections* every steady-state cycle, so it is called out here.) C14/C3 force-exercise
   this: assert salient markers from both inputs survive into the new tier-3 within the cap across steady-state
   cycles.
   **Inactivity/catch-up (OWNER item 5):** after any length of inactivity, all raw is >72h → it partitions to
   `G72`, all existing sections classify to the 72h band, so the cascade **rebuilds straight down to tier 3
   only** (tiers 1 & 2 empty because there is no recent material). The age-bucketing handles this with no
   special case (C14 asserts it).

**Atomic single write (stage-3 round-2 minor).** All three tiers are computed from the **pre-pass snapshot**
(the sections + eligible raw turns read in step 1), then the new summary row + retained tail are installed in
**one** `rewrite_session_atomic` call (`buffer.py:265`, tmp+fsync+replace) — NOT three sequential
archive+rewrite cycles. So a crash cannot leave a half-updated row with, say, a new 72h but an old 24h: either
the whole new row lands or the pre-pass row remains. The archive-before-rewrite byte-count verify still gates
the single rewrite. New criterion **C17** asserts this atomicity (a simulated failure before the atomic
replace leaves the pre-pass row intact).

**Idempotence (C7):** a pass with no newly-crossed material and no section needing to graduate makes no change.
**Graduation + terminal persistence (C14, per F3 + OWNER 2026-08-13):** the criterion asserts against **actual
source-content age**, NOT `covers_until_ts` (the classifier's own input → tautological). Method: seed
identifiable raw turns at known ts and use a fake provider whose fold **preserves the markers** (identity/
concatenative); then across many consecutive daily passes assert a marker sown on day 0 is in tier 1
("yesterday") after 1 pass, tier 2 after 2, tier 3 after 3, and then **PERSISTS in tier 3 (fading, still
present) on passes 4, 5, …** — NOT gone/archived (tier 3 is terminal). Plus: a **long-inactivity** fixture →
**only tier 3 populated** (tiers 1 & 2 empty); and a **multi-day-gap** sequence lands markers in the tier
their true age dictates.

**#77 fallback under the cascade (revised per stage-3 L3):** if building a target section double-rejects the
validator (§1.4), the fallback **must not drop the source material**. The target's new value =
a lossless-leaning safe join of its *inputs* (the prior section prose + the age-group's rendered turns),
truncated at a sentence boundary to `_SECTION_24H_CHAR_CAP` for the 24h tier (or the tier's derived bound),
recorded as a soft-fail. This guarantees that a failed 48h build never silently discards the 24h material it
was promoting (the L3 gap), because that material is carried into the section verbatim-ish rather than lost.

Sub-ops reused by both the daily cascade and the `apply_budget` backstop:
- `_fold_into_section(inputs, fraction, cap=None)` — the existing fold machinery (`_render_transcript`/
  `fold_existing_summary`, provider call, **fold validation §1.4**, archive-before-rewrite, atomic rewrite).
- `apply_budget` **backstop** (`budget.py:42-91`, `older_than=timedelta(0)`) calls a **24h-only** emergency
  fold (`fold( raw + existing 24h, 0.60, cap )`) to bound the live head in-turn; it does **not** run the full
  age-gated re-bucket (that stays on the daily tick). Both paths write the sectioned row, hold the lock, and
  archive-before-rewrite. (Preserves the backstop's job of bounding the live prompt while the age-gated
  promotion stays on cadence.)

### 1.4 Fold-output validation (#77) — `_validate_fold_output(text) -> str | None`
Rejects (returns `None`) when the provider output is: empty/whitespace after strip; a refusal / policy
completion (matches a documented predicate — e.g. leading "I won't/ I cannot/ I'm sorry" + no first-person
recollection; or an assistant-meta frame like "Here is the summary:"/ "As an AI"); or non-first-person for a
persona whose fold is first-person by construction (heuristic: absence of any first-person pronoun in a
non-trivial output). On `None`: **retry once** with the same prompt; on a second `None`, **fall back** =
**keep the prior section unchanged** (do not overwrite with garbage; do not store bare `.strip()`), and record
a soft-fail note in the result. Replaces the raw `.strip()` store at `compaction.py:338`. The predicate is a
module-level, unit-testable function (C4). (Scope note: this is deliberately conservative — a false-reject
merely keeps last cycle's section, which is safe; the cost of a false-accept, storing a refusal as memory, is
the failure #77 names.)

## 2. Session rollover (1c)

### 2.1 1c-A — idle >24h stale resume (SYNC), at `/sessions/active`
When the handler finds **no** attach-eligible (<`_ATTACH_MAX_AGE_HOURS`) session but a stale one exists:
synchronously (a) run a final memory-**extraction** pass on the old buffer (so nothing is lost — §2.3),
(b) `compact_conversation(..., min_keep_tail=0)` the entire old conversation, (c) archive it, (d) **delete the
old buffer + its cursor + write `rolled_to.json`** (§2.3 L2 — rollover owns the lifecycle), (e) create a new
session, write the seed as its first head **summary** row (a `summary` row — un-extractable via the
speaker-filter — with the new session's cursor set so it is never re-compacted as if it were a raw turn; 1c-A
seeds summary-only, `min_keep_tail=0`, so there is no carried raw tail here — the raw-tail cursor-carry is the
1c-B case, C18/UA-2), and (f) **return the new session
id (not `null`)**. Blocks until the seed is written (SYNC — owner-ratified, brief §1c-A). Reuses
`compact_conversation` + `rewrite_session_atomic`. `_ATTACH_MAX_AGE_HOURS` (`server.py:1224`) stays the
threshold.

**Stale-session SELECTION when several stale buffers exist (stage-3 A1-b — endpoint has no client-supplied
prior sid).** Rule: roll over the **most-recently-active** stale session (max last-turn `ts`) — the one the
user most plausibly continues; its seed becomes the new session's head. Other, older stale buffers are not
seeded from; they are reaped by the daily finalize-extraction + the same rollover reaper (they too get a
weekly-age rollover on the daily tick, or are extracted-then-removed) so the active set stays bounded (§2.3,
C15). Deterministic tie-break: highest `ts`, then lexicographically-largest sid.

**Sync-work bound (stage-3 A1-a — blocks one threadpool worker for the full fold).** The full-fold is a single
Haiku call whose input is bounded by the same `_SECTION_24H_CHAR_CAP` truncation applied to the rendered
transcript before the provider call; the handler holds the per-session compaction lock only for its own
session. This keeps the rare-event block to one bounded Haiku round-trip (a few seconds — the owner-accepted
cost), not an unbounded fold. Documented as an accepted, bounded synchronous cost.

**Extraction concurrency — corrected claim (stage-3 round-3 F4).** The round-2 draft claimed "the cursor, not
a lock, serialises extraction." That over-claims: `extract_session_snapshot` (`pipeline.py:317-318`) does an
**unguarded read-then-later-write of the cursor**, so two concurrent callers (a finalize tick on the
supervisor thread + a SYNC 1c-A rollover extraction on a request worker) *can* both read the same pre-advance
cursor and double-process; the embedding-cosine dedupe downstream is a **soft net, not a guarantee**. This is
a **pre-existing** property of `extract_session_snapshot` — this change adds the rollover as one more caller
but **does not worsen** it (and never *loses* a turn; the failure mode is a possible duplicate, caught softly).
A proper fix (guard the cursor read→write) is **out of scope** for this change and recorded as a follow-up.
C10 asserts the buffer/seed survive the interleaving; it does **not** claim exactly-once extraction.

### 2.2 1c-B — weekly cap, driven + executed on the DAILY SUPERVISOR TICK (§0)
On the daily tick, if session age ≥ `_WEEKLY_ROLLOVER_AGE` and last turn older than `_ROLLOVER_QUIET_GAP`,
execute the swap under the compaction lock: seed a new session with **3 tiers + 40 messages**
(`min_keep_tail=40`), extract → archive → **delete** the old buffer, write `rolled_to.json`. Same atomic
primitives as 2.1; the seed shape differs (§0 seed asymmetry). No attach-seam dependency — the continuous
client is redirected by `rolled_to.json` (§0, C16). If within the quiet gap, defer to the next daily tick.

**Same-tick ordering: cascade fold FIRST, then the weekly-rollover check (M2, stage-3 round-4/5 minor).** The
daily tick does both the cascade `cascade_conversation` and the weekly-rollover check on a session. Order is
**cascade-fold-then-rollover**: the fold brings the 3 tiers current, then the rollover — if it fires — seeds
the new session from those **just-updated** tiers (+ 40 messages). Doing it the other way would seed from stale
pre-fold tiers. **The real lock mechanism (corrected per stage-3 round-6 F3 — NOT a single continuous hold).** The per-session
compaction lock (`buffer.py:341-392`) is a **non-reentrant, self-contained per-call primitive**: `cascade_
conversation` and the rollover each **acquire it internally and release at the end of their own call**
(mirroring `compact_conversation` `:254-259`). So the tick does **cascade-fold (acquire→release) THEN
rollover (acquire→release)** — two separate acquisitions, and `apply_budget` is a **confirmed real concurrent
caller** of the same lock that can interpose between them. Correctness does **not** rest on lock continuity;
it rests on **re-read**: the rollover, after acquiring the lock, **re-reads the current committed summary row**
and seeds from it. If `apply_budget` interposed, it only touched the 24h tier (bounded, §1.3) — the rollover
reads whatever the latest committed tiers are, so the seed is always current, never stale or torn. Ordering
(fold before rollover-check) is by sequence in `_run_compaction_tick`; interposition-safety is by re-read,
matching the existing re-read-by-identity discipline (`compaction.py:396-417`).

### 2.3 Finalize-tick decoupling + rollover owns the buffer lifecycle (stage-3 L2 — Major)
`finalize_stale_sessions` (`pipeline.py:529-626`) currently **extracts then deletes** the buffer + cursor
(success path `:587-590`, empty path `:557-559`). The prior draft removed the delete but **reassigned it
nowhere** → every rolled-over buffer would leak into `active_conversations/`, and `list_active_sessions` is
re-iterated every tick by `_run_compaction_tick` (`supervisor.py:1636`), `/sessions/active`
(`server.py:1227`), and finalize (`pipeline.py:554`) — cost growing with all sessions ever created. Fix:

- **Finalize becomes extraction-only** — it still runs the final memory extraction on every eligible session
  (memory of old turns continues, brief), but **never deletes a conversation buffer**. This applies to **both**
  the success path (`:587-590`) **and** the empty-buffer branch (`pipeline.py:556-560` `if not turns:`) — the
  latter's buffer delete is removed too (an empty live buffer is bounded and becomes rollover-eligible; the
  rollover reaps it). The poison-move path (`:591-609`, corrupt input) is **unchanged** (a poisoned buffer is
  still quarantined out of the active set).
- **Rollover OWNS delete of a superseded NON-empty buffer.** Both triggers (2.1, 2.2), after extract →
  archive → seed, call `delete_session_buffer` (`buffer.py:162`) on the old sid, delete its cursor, evict it
  from the in-memory registry, and write `rolled_to.json`. **Corrected enumeration (stage-3 round-3 minor):**
  the rollover is not the *only* buffer deleter — `snapshot_stale_sessions` (`pipeline.py:475-526`, scheduled
  ~5-min via `daemon.py:108`/`server.py:684`/`supervisor.py:319`) still **ghost-deletes a truly-empty buffer**
  (`pipeline.py:504-506`, `if not turns: delete_session_buffer`). That is harmless and NOT a conflict: it only
  fires on an already-empty buffer, whereas the rollover reaps a *non-empty* superseded buffer under the lock.
  So the accurate claim is "a **superseded non-empty** buffer is deleted exactly by the rollover; a
  **truly-empty ghost** buffer may also be reaped by the ~5-min ghost-cleanup." Both are enumerated in §5.
- **`remove_session` is ALREADY registry-only (no build work here — stage-3 round-3 correction).**
  `remove_session` (`session.py:222-231`) today only pops `_SESSIONS`; it performs **no** file delete. So there
  is nothing to change there — the prior draft's "change `remove_session` to retain only registry eviction"
  was a no-op mis-stated as work. A session that is neither rolled over nor poisoned keeps its buffer (still
  live) — correct, not a leak; it is bounded by the daily cascade + 1d segmentation and becomes
  rollover-eligible at the 24h-idle or weekly boundary. **C15** asserts the active set stays bounded: a
  rolled-over old buffer is gone next tick.
- **Dead code note:** `close_stale_sessions` (`pipeline.py:629-673`) has **no call site** (dead; docstring
  stale). This change does not touch it; noted so a future reader need not re-verify.

## 3. Archive segmentation (1d)

Segment `archived_conversations/<sid>.jsonl` → rolling segments `archived_conversations/<sid>.<NNN>.jsonl`
(size-capped, const `_ARCHIVE_SEGMENT_MAX_BYTES = 5 * 1024 * 1024`; a hard byte bound is chosen over per-day
so a bursty day cannot make one segment huge). `append_archive` rolls to a new segment when the active one
would exceed the cap; each individual append stays atomic (fsync of the file) and still returns bytes written
(byte-count contract, C7/C11). Extend `read_archive` (`buffer.py:310`) to enumerate all segments for a sid
**in order** (numeric segment sort) and concatenate (skipping corrupt lines), reconstructing the full
provenance chain; a **legacy single-file archive with no segment suffix still reads** (back-compat — glob both
`<sid>.jsonl` and `<sid>.<NNN>.jsonl`). Preserve the provenance chain (faded summaries + superseded raw turns
in order).

**Crash durability (stage-3 concurrency-ii + C11 re-point).** The reviewer correctly noted that a concurrent
append during a roll is **lock-precluded** (all archive writers hold the per-session compaction lock, §5), so
that is not the real risk — the real risk is a **crash mid-roll / partial write**. Mitigations: (1) when a new
segment file is created, **fsync the containing directory** so the new segment's directory entry is durable
(not just the file contents) — **reuse the existing posix-guarded pattern at `brain/health/attempt_heal.py:250-266`**
(`if os.name == "posix": dir_fd = os.open(str(parent), os.O_DIRECTORY); os.fsync(dir_fd); os.close(dir_fd)`,
wrapped in try/except/finally) so **Windows CI does not break** (this repo supports Windows — cf. `buffer.py`
`_unlink_with_retry`/`_pid_alive`); (2) the reader tolerates a torn final line in the active segment
(`read_jsonl_skipping_corrupt` already skips corrupt lines) and a zero-length trailing segment. C11's oracle
is re-pointed from "inject an append during a roll" to "**simulate a crash mid-roll** (a created-but-unsynced
segment / a torn final line) and assert the segment-aware reader still returns the complete prior provenance
chain."

## 4. Migration (reuse `compaction_migration.py` backlog pattern)

Add a one-time migration (marker-gated exactly like `run_backlog_migration`, its own marker file e.g.
`.sections_migrated`) that, per persona/session with a legacy single-layer summary row, rewrites it into the
sectioned form. **The legacy `text` → `sections["72h"]` (TIER 3) with `covers_from_ts = migration_now −
`_LEGACY_AGE_FLOOR` (96h)`, set UNCONDITIONALLY (stage-3 round-6 MO-1 / round-7 MO-2).** Never fall back to
`covers_until_ts` — a recent value would let the classifier reclassify the blob to tier 1 next pass (#82). An
explicit old value structurally guarantees it lands in and stays in tier 3. **Optional primary path
(nice-to-have, NOT required):** if you want a real oldest ts, "derivable" means an **archive scan** for the
true oldest turn (`read_archive`) — **never a buffer check** (the buffer's oldest is always recent and would
defeat the point); if the archive scan yields a genuinely older ts, use it, else the 96h old-floor. Idempotent:
the marker + an "already-sectioned" check make a re-run a no-op (C12). Wire it into the same startup
backlog-migration thread (`server.py:913-935`) so it runs before the first daily cascade tick. **Tests
(extend C12):** (a) migrated legacy persona (months-old history) → cascade classifies tier 3, not tier 1;
(b) **the fallback branch** (legacy blob, no derivable ts) → tier 3 AND **survives the NEXT cascade pass still
tier 3** (not reclassified to tier 1) — the load-bearing #82-regression guard.

**Interaction with the unchanged `run_backlog_migration` (round-2 minor; corrected round-8 MO-3).**
`run_backlog_migration` is NOT modified; it still calls legacy `compact_conversation`, which writes a summary
row with **no `sections` key** (`compaction.py:364-374`). If a backlog drain runs after a sectioned row already
exists (e.g. its marker was withheld across a restart on a transient `locked`/`archive_failed`,
`compaction_migration.py:44-47`), that drain would flatten the row back to single-layer. This is **not
data-loss**: the **tolerant reader (§1.1)** reads a section-less row as
`sections={"72h": {text: legacy text, covers_until_ts: <existing>, covers_from_ts: now − _LEGACY_AGE_FLOOR
(96h)}}` — i.e. **tier 3 with the old-floor, NOT tier 1** (this passage previously said `{"24h"}` and omitted
`covers_until_ts`, contradicting §1.1/§4 — the round-8 MO-3 + round-9 N2 fix) — and the **next daily cascade
tick** re-establishes the full sectioned form, still tier 3. Confirmed self-healing; noted so an implementer
does not "fix" it by editing the legacy backlog path (which would enlarge the change surface unnecessarily).
**C12 covers this exact interaction** (already-sections-migrated persona hit by a delayed backlog retry → stays
tier 3, not tier 1).

**Startup migration order (round-9 N1 — pinned, not left inferable).** In the shared startup backlog-migration
thread (`server.py:913-935`), run the **new sections-migration BEFORE `run_backlog_migration`** (the legacy
compaction backlog drain). Rationale: sections-migration only rewrites the *existing* summary row's shape
(cheap, in-place); running it first means any row the legacy drain then touches is already sectioned, and even
if the legacy drain later flattens one, the tolerant reader + next cascade tick self-heal it to tier 3
(above). The order is pinned for determinism; correctness holds either way (self-healing), but pinning removes
the inferable ambiguity.

## 5. Concurrency accessor enumeration (ST2b — the change adds writers/RMW windows over shared state)

Shared mutable state and every accessor (reader R / writer W), the guard, and whether the guard covers it:

| State | Accessor | R/W | Synchronized by | Covered by guard? |
|---|---|---|---|---|
| **session buffer** `active_conversations/<sid>.jsonl` | turn-time write path: `get_or_hydrate_session` (session resolve; **now consults `rolled_to.json`**) → `engine.respond` → `ingest_turn` (append) | W | append-only + compaction re-read-by-identity; **redirect installed in `get_or_hydrate_session` (session.py:129-205), the shared 404 chokepoint — NOT in `ingest_turn`** (P-1) | yes — the fold re-reads & subtracts by identity (`compaction.py:396-417`); a post-swap `/chat`//stream resolves the old sid to the successor before the 404 (§0, C16); **preserved** |
| | `cascade_conversation` / `_fold_into_section` (daily tick + backstop + migration) | W | per-session **compaction lock** (`buffer.py:341-392`) + atomic rewrite | yes — **all writers acquire the same lock** |
| | rollover swap — 1c-A at `/sessions/active`, 1c-B on the daily tick (extract→archive→**delete**→seed→`rolled_to`) | W | compaction lock (both paths acquire it) | yes — **both rollover paths hold the compaction lock across the whole swap** |
| | `finalize_stale_sessions` | R (extract only) | — | **change removes the delete** (extraction-only); delete moves to the rollover path → no leak (L2), no race |
| | `snapshot_stale_sessions` ghost-delete (`pipeline.py:504-506`, ~5-min cadence) | W (empty only) | none (unlocked) | fires **only on a truly-empty buffer** → harmless; not a conflict with the rollover (which reaps a *non-empty* superseded buffer under the lock). Enumerated per stage-3 round-3 |
| **`in_flight_locks`** (`server.py:816`, `dict[str,asyncio.Lock]`) (F2 + round-5) | all **5** `get_or_hydrate_session` handlers: `/chat`,`/stream`,`/sessions/close`,`/sessions/snapshot` (W/R lock) + `/state` (R, in-flight report) | W/R | asyncio per-key lock | **all keyed by resolved `sess.session_id`** (uniform rebind, §0) so old-sid + successor-sid traffic serialise on ONE key and `/state` reports the successor's in-flight correctly (else key-split F2 / `/state` misreport). C19, C20 (/state oracle), **C21 structural guard** |
| **`_SESSIONS` registry** (`session.py`) | `/sessions/close` cleanup `remove_session` + `in_flight_locks.pop` (`server.py:2835-2836`) | W | `_LOCK` (RLock) | **G1 fix (stage-3 round-4, OWNER-flagged):** these used the RAW `req.session_id` → the **successor**'s registry entry + lock **leak forever** and `/state/{successor}` reports it live after "closed". **Fixed to use resolved `sess.session_id`** (C20). |
| | finalize `remove_session(r.session_id)` (`supervisor.py:1686`) under extraction-only finalize | W | `_LOCK` | **M1 (enumerated, harmless):** registry-cache eviction only; a still-live session evicted here **re-hydrates from disk** via `get_or_hydrate_session` on next access. No buffer effect (finalize no longer deletes buffers). |
| **ingest cursor** sidecar | `write_cursor` (extraction), `read_cursor` (fold guard); rollover deletes old-sid cursor + **carries the extraction state of the 40-msg raw tail** to the new-sid cursor | W/R | **cursor read→write is UNGUARDED in `extract_session_snapshot` (`pipeline.py:317-318`)** | pre-existing; two concurrent extractors (finalize tick + rollover extraction) *can* double-process (soft-deduped downstream) — **not worsened** by this change, exactly-once fix out of scope (F4). The seed *summary* row is un-extractable via the speaker-filter; the real cursor risk is the **carried raw tail** — its extraction state must transfer so it is neither re-extracted nor lost (C18, UA-2) |
| **archive** `<sid>.<NNN>.jsonl` | `append_archive` (fold, rollover, migration) | W | compaction lock (callers) + file fsync + **new-segment directory fsync** | yes — all appenders hold the lock (append-during-roll is lock-precluded); the real risk is a crash mid-roll, mitigated by the directory fsync + corrupt-line-tolerant reader (§3, C11) |
| | `read_archive` (segment-aware) | R | reads committed segments; tolerates a torn trailing line | yes — globs the full segment set, numeric-ordered; skips corrupt lines |
| **`rolled_to.json`** (NEW successor pointer) | rollover (write, under lock, before old-buffer delete); `get_or_hydrate_session` + `/sessions/active` (read) | W/R | atomic tmp+fsync+replace; written before the delete so old-sid never resolves to nothing | **full-follow chain** to the live successor (visited-set cycle guard) at the real resolve chokepoint; **no window** where the old sid is unresolvable; multi-generation (C16) |

**The two lost-update criteria** (C10 finalize↔rollover; C11 archive roll) are verified by **injected
interleavings** that must **fail against the unguarded/pre-change version** (H4).

## 6. Measurement — how each criterion is verified (scratch pytest fixtures; the route-(a) harness)

All under `tests/chat/` (+ `tests/ingest/`) with a **fake `LLMProvider`** (returns scripted fold outputs, incl.
injectable bad outputs) and a `tmp_path` persona dir. Each fixture is shown able to fail (H6) by first running
it against the pre-change code / a known-bad scratch variant.

| Crit | Test | Path exercised | Shown-able-to-fail |
|---|---|---|---|
| C1 | `test_cascade_three_sections` | cascade over >72h fixture → parse row | vs pre-change single-layer |
| C2 | `test_cascade_age_gated_wiring` | age-partition of raw + section age-classify recorded; assert each tier's inputs match their age bucket | vs a youngest-first / tick-shift scratch |
| C3 | `test_tier_caps` | over-cap raw batch → assert tier1 ≤ `_SECTION_24H_CHAR_CAP` AND terminal tier3 ≤ `_SECTION_72H_CHAR_CAP` (= 0.20×tier1 cap) **after many steady-state cycles** (tier3 folds 2 inputs/cycle — F1) | vs each cap removed (tier3 grows past cap when the multi-input fold isn't capped) |
| C4 | `test_fold_validation` (single + **cascade double-reject**) | **drives the real `cascade_conversation` path** (not just the `_validate_fold_output` unit): inject refusal/non-first-person; assert reject+retry+preserve; assert a double-rejected 48h build keeps the 24h source (L3) | vs validator disabled / vs prior "drop on double-reject" |
| C5 | `test_temporal_markers_and_labels` | known-ts turns → assert the coarse span in the header AND the owner labels ("yesterday"/"day before yesterday"/"a few days ago") on tiers 1/2/3 | vs ts stripped / vs bare "24h/48h/72h" labels |
| C6 | `test_head_prefix_bytestable_and_reparse` | render×2 byte-eq; feed to `apply_budget` reparse | vs injected nonce / broken prefix |
| C7 | `test_invariants_preserved` | un-extracted tail no-op; byte-verify-before-rewrite; double-run no-op | vs cursor guard removed |
| C8 | `test_idle_rollover_sync_and_selection` | drives the **real `/sessions/active` endpoint** (FastAPI TestClient) + the rollover fn: stale last-turn → non-null sid, archived, old buffer deleted, seed head, sync; **multi-stale → most-recently-active selected** | vs pre-change handler (null) |
| C9 | `test_weekly_rollover_daily_tick` | drives the **real daily-tick function** (`_run_compaction_tick`), not a bespoke helper: age≥cap at a quiet moment → swap seeds 3 tiers+40; within-quiet-gap defers (no mid-exchange fire) | vs "fire immediately/mid-exchange" scratch |
| C10 | `test_finalize_no_delete_interleave` | inject finalize tick into rollover window; seed survives; **finalize does not delete**; extraction ran | vs pre-change finalize (deletes) |
| C11 | `test_archive_segments_reader_crash` | append past cap → >1 segment; reader in-order; provenance; **simulate crash mid-roll (unsynced segment / torn line)** → reader still returns full chain | vs newest-segment-only reader / byte-verify off |
| C12 | `test_sections_migration_idempotent` | legacy row → sectioned; re-run no-op | vs assertion on legacy shape |
| C13 | `test_interior_not_starved` | seed monologue_trace; run cascade; ambient read still returns | vs trace store nulled |
| **C14** | `test_graduation_terminal_multiinput` | **fake provider preserves markers**: sow marked turns on **consecutive** days, run many daily passes; assert each marker in tier1 after 1 pass, tier2 after 2, tier3 after 3, then **PERSISTS in tier3 on 4,5,…** (terminal); **crucially, assert markers from BOTH the persisting tier3 AND the newly-graduated tier2 co-survive in the new tier3 each steady-state cycle (multi-input fold — F1) within the cap**; + long-inactivity → only tier3 populated; + multi-day-gap → markers by true age. Asserts on **actual marker content, NOT `covers_until_ts`** | vs newest-edge+remerge (marker never leaves tier1); vs "72h→evict" (gone after pass 4); vs a single-input tier3 fold that drops one of the two inputs; vs tick-shift (gap mislabel) |
| **C15** | `test_active_set_bounded_after_rollover` | roll over a session; assert the **old buffer is deleted** and `list_active_sessions` does not grow with rolled-over sessions | vs the "finalize delete removed, unassigned" draft (leaks) |
| **C16** | `test_post_rollover_redirect_all_sites` | after a weekly swap, `get_or_hydrate_session(old_sid)` resolves to the **successor**; the four **operation** handlers via TestClient with old_sid → operate on the successor: `/chat` → 200 + turn in successor buffer; **`/sessions/close` + `/sessions/snapshot` → act on the successor (NOT a false-success `committed=0` no-op)** (F1); `/stream` resolves too (`/state` read handler is C20/C21) | vs no redirect (404) / vs close-snapshot passing raw `req.session_id` (silent no-op reported success) |
| **C17** | `test_cascade_write_atomic` | compute 3 tiers from pre-pass snapshot; **one `rewrite_session_atomic`**; simulate failure before the atomic replace → pre-pass row intact (no half-updated tiers) | vs a 3-sequential-rewrite scratch (partial row survives) |
| **C18** | `test_carried_raw_tail_extraction_state` (re-pointed, UA-2) | after a **1c-B** rollover seeding a 40-msg raw tail with mixed old-session extraction state, run the new session's extraction → assert already-extracted carried msgs are NOT re-extracted (no dup memory) and unextracted ones ARE. (The seed *summary* row is already unconditionally un-extractable via the speaker-filter — not the real risk.) | vs seed written without carrying the cursor/extraction state (dup memories from the carried tail) |
| **C19** | `test_inflight_lock_keyed_by_resolved_sid` (F2) | after a swap, drive traffic on **both** the old sid (redirected) and the successor sid; assert both acquire the **same** `in_flight_locks` key (resolved `sess.session_id`) → serialised, no cross-key concurrent buffer write | vs keying by raw `req.session_id` (two keys → concurrent write to one buffer) |
| **C20** | `test_close_cleanup_uses_resolved_sid` (G1) | after a swap, `POST /sessions/close` with the **old** sid via TestClient → assert the **successor**'s registry entry is removed and its `in_flight_locks` key popped (no leak); **`GET /state` on the old sid reports the successor's in-flight correctly (round-5)** | vs cleanup keyed by raw `req.session_id` (successor left live + lock leaked; `/state` misreports in-flight) |
| **C21** | `test_no_raw_sid_downstream_of_resolution` (F2 structural guard) | **static check** over `server.py`: for every handler that calls `get_or_hydrate_session`, assert no raw request/path `session_id` is used downstream for the load-bearing ops (backend call, `in_flight_locks` key, `remove_session`, `/state` in-flight, echoed id) — the resolved `sess.session_id` must be. Closes the redirect class so a 6th site can't regress silently | vs the current code (all 5 sites use the raw id) → the check fires on every unfixed site |
| **C22** | `test_apply_budget_sectioned_row` (F3 minor) | drive the real `apply_budget` backstop against a **sectioned** summary row: assert the 24h-only emergency fold reads/updates the sectioned row correctly (does not corrupt sections / does not run the full re-bucket) and the head stays re-parseable | vs a scratch where the backstop assumes a flat single-layer row (mis-parses sections) |

**Instrumentation added to scope:** `cascade_conversation` returns a result carrying, per tier, the
`(inputs, age_bucket, source_sections, raw_group, target_fraction, fell_soft, validated)` record so C2/C4/C14
are observable without prints; the fold validator is a pure function so C4 is a direct unit test. No new
production JSONL telemetry is required (measurement is advisory; the config `metrics` block is unchanged — a
`# FOLLOW-UP` notes that a replay workload would make A1 gating).

## 7. Severity → routing thresholds (this change)
Standard SEV1 table. Gating criteria C1–C13 must be `verified=yes` by execution at stage 8 (no proxy/defer).
A red CI (`ruff` / `pytest -m "not live and not requires_claude_cli and not integration"`) blocks the build.
Advisory A1 regression deltas are surfaced, never auto-bounce. A stage-8 gating miss routes per the table
(blocker→1, major→human).

## 8. Build order (stage 5 — ordered commits on the one branch)
1. Sectioned representation + deterministic render (owner labels "yesterday"/"day before yesterday"/"a few
   days ago") + tolerant reader (C1, C5, C6) — no behavior change to callers yet.
2. Fold validator (#77) + cascade double-reject fallback (C4), swap in at the `.strip()` site.
3. `cascade_conversation` = **age-gated GRADUATION with a TERMINAL, MULTI-INPUT tier 3** (classify by oldest
   edge `covers_from_ts`; never co-fold prior tier1 with fresh raw; tier3 folds persisting-tier3 + graduated-
   tier2 each cycle via the lossless-leaning join → 20% → `_SECTION_72H_CHAR_CAP`) + per-tier fractions +
   tier1/tier3 caps (C2, C3, C7, **C14 graduation+terminal+multi-input**); wire the daily tick + the 24h-only
   backstop (**C22 apply_budget × sectioned row**).
4. 1d archive segmentation + segment-aware reader + crash durability (C11).
5. Migration + tolerant reader: legacy blob → **tier 3** with `covers_from_ts` = old-floor (`now − 96h`)
   **unconditionally** (never `covers_until_ts`; optional archive-scan primary path) — the structural
   #82-regression guard for existing personas (C12, **MO-1/MO-2** incl. the fallback-branch survives-next-pass
   test).
6. 1c-A idle rollover at `/sessions/active` + stale selection + sync bound + **old-buffer delete**; carried
   raw-tail cursor state (**C18/UA-2**, mainly the 1c-B path); cascade write proven atomic (**C17**).
7. Finalize decoupling → extraction-only (incl. the empty-buffer branch); rollover owns delete (note:
   `remove_session` already registry-only) (C10, **C15**); 1c-B weekly swap on the daily tick + `rolled_to.json`.
8. Redirect wiring (close the class in ONE sweep): `get_or_hydrate_session` consults `rolled_to.json` +
   **full-follows the chain to the live successor** (visited-set cycle guard; L-1 multi-generation)
   (`session.py`); **all FIVE handlers (`/chat`, `/stream`, `/sessions/close`, `/sessions/snapshot`, `/state`)
   rebind to the resolved `sess.session_id` for every downstream op** (backend call, `in_flight_locks` key,
   `remove_session`/`in_flight_locks.pop` `:2835-2836`, `/state` in-flight, echoed id) — C16 (+multi-gen),
   **C19** (lock keying), **C20** (close cleanup + /state / G1), **C21** (structural guard — no raw id downstream).
9. C13 guard + full CI.
Each commit: imperative mood, no gc-process noise, no personal details.
