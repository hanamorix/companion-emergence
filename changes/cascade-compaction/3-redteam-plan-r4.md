# Stage-3 Plan Red-Team — Round 4 — Cascade Compaction (cold, independent)

Reviewer: cold subagent, model claude-sonnet-5 (per orchestrator's usage directive — all cold reviewers on
sonnet since gate-4 round 1). No shared context with the author; first read of the round-4 artifact set.

## Provenance

Artifacts (sha256, full-file):
```
1201f8bdee521914d94bcfd10daae5ccaba5cccae21e0113e0bbf8fb777731c6  1-spec.md
678a67b881e85a32025e59a1ea847a9d9ee789f3cab72d51295a0100eb5d5f71  1.5-criteria.md
621f78217335aa69c7bad938e1816910f54c7062ce3943acc214971c19ad5414  2-plan.md
46890b678c6c364f1ac4c12ca22c27eaa4ed633a1755b69900c65625a545d638  decisions.md
bcce93ecd382484709895cffb675b5729d3dfd679240027c85d423c2fdb71d96  3-redteam-plan-r3.md
edb8945841cd99cc162c0ddc443c44dfb89a46511e32f87d592fe19560efc632  3-redteam-plan.md
```
Note: `1-spec.md`/`1.5-criteria.md`/`2-plan.md` hashes are **identical** to the round-3 author-revision hashes
recorded in `3-redteam-plan.md` round-3 section (`8f1f42b3…800c`, `a1c05b15…18f9`, `a4c0caa1…3b78e` — matches
exactly). The artifacts under review this round are the SAME text round 3 approved-with-findings; this is
confirmed by the orchestrator's own `decisions.md` (round-3 gate-4 entry cites the same hash prefixes). I
treat round 3's findings as the operative carried-forward set.

Source read (whole-file sha256, base `cd29bc61`, branch `ThinkerOfThoughts/cascade-compaction`):
```
cc9b6e22ac3cf05aa1109e84abb5d8217619153d85cf172ae1b84644b1aad7fb  brain/chat/compaction.py
f0b0b715746bc9f8e27964ec7c24301a14b6b80dab4d38db8f8b533d1df62d7e  brain/bridge/server.py
773f1e0b0ae3dad2cbdc4f316e16e65194d09ab047665ff740259eace1a4dc34  brain/chat/session.py
cfe8b63b3d642dabe998f52b52a087ccc4c0acbd8bab5b32bba177f0b309d331  brain/ingest/pipeline.py
cefb079963884fbafea3a0d8125c74bdc3a9e889894731329f482f49f93da56b  brain/ingest/buffer.py
ca6eeba8070959cf502e76177a3a635832ecc0377e12e43773b3cf9629c116b2  brain/bridge/supervisor.py
d359be520046a66c502ca5f0b56a0c61e8f4a13fbef93868e6fe127bed0d1260  brain/chat/engine.py
b71ed34d8f0d50108739fad682a3698989cfb5bd1278f0e4a3c7a011d873c7ca  brain/chat/budget.py
7d32a50b29f47b34645bd48c04b6cae3bb62e2dd15e031712b2b752fd59e1d77  brain/chat/compaction_migration.py
74dd5ba03614872c430fd2f3d2e40f23d4f70349a69816309ea0a4e7b8ee054e  brain/monologue/ambient.py
```
Plus `brain/health/attempt_heal.py:235-267` (dir-fsync pattern) and `brain/bridge/persisted_cadence.py`
(full — `advance`/`is_due` semantics) read for targeted verification. All source hashes are **identical** to
those cited in round 2 and round 3 (no code has changed — expected, since this is a plan-stage change; the
build has not started). I independently confirm the touched-file set is readable and matches `decisions.md`'s
CFG3 record.

## Carried-forward findings resolution (round-3 → round-4)

| # | Finding | Resolved? | Evidence |
|---|---|---|---|
| F1 (Major, redirect incomplete for close/snapshot) | **PARTIALLY** — see NEW finding G1 below | Plan §0 now states each handler must use `sess.session_id` for its "backend call ... and its `in_flight_locks` key" (2-plan.md:53-55). Verified against `server.py`: `/sessions/close` (2753/2771) and `/sessions/snapshot` (2696/2703) currently pass `req.session_id` to `_close_session_blocking`/`_snapshot_session_blocking` — the plan's fix, if implemented as literally described, closes this. **But** the plan's own citations (§0, §5) never mention `server.py:2835` (`remove_session(req.session_id)`) or `:2836` (`s.in_flight_locks.pop(req.session_id, None)`), two more `req.session_id`-keyed accessors inside the SAME `/sessions/close` handler, executed AFTER the (now-fixed) backend call. See G1. |
| F2 (Major, `in_flight_locks` key-split) | **YES for the four `setdefault` sites** | Confirmed at `server.py:2368,2429,2699,2765` — plan's fix (key by `sess.session_id`) is a straightforward substitution at each. C19's oracle (acquire-side only) would pass. See G1 for the residual `.pop()` gap this doesn't cover. |
| F3 (Major, age-laundering — classify-by-newest-edge) | **YES, the specific mechanism is fixed** — but see NEW finding G2 (a different defect in the SAME locus) | Confirmed: plan §1.3 step 3 now classifies by `covers_from_ts` (oldest edge), and step 4 explicitly forbids co-folding the prior 24h section with fresh raw. I traced the arithmetic for continuous daily use (worked example in G2 below) and confirmed the *specific* F3 mechanism (newest-edge classify + prior-24h remerge) is gone. However, the replacement mechanism has its own defect at the OTHER end of the pipeline — see G2. |
| F4 (Minor, extraction-cursor race overclaim) | YES | Plan §2.1 "corrected claim" now states the race is pre-existing/unguarded, not worsened, exactly-once fix out of scope. Confirmed against `pipeline.py:317-318` (`extract_session_snapshot`) — unguarded `read_cursor` at :317 / `write_cursor` at :453, no lock. Accurate. |
| snapshot_stale enumeration | YES | Plan §2.3 now enumerates `snapshot_stale_sessions` ghost-delete (`pipeline.py:504-506`) as a 5th, harmless, empty-only deleter. Confirmed against source: `if not turns: delete_session_buffer(...)` at pipeline.py:504-506, unlocked, matches. |
| C4 wording tightened | YES | 1.5-criteria.md C4 now explicitly says "drives the real `cascade_conversation` path, not only the `_validate_fold_output` unit." |
| cap ack (only 24h has a hard ceiling) | YES | Plan §0 Q8 explicitly acknowledges this with reasoning (worst-case ≈17.8k chars). |
| `remove_session` no-op correction | YES | Confirmed against `session.py:222-231` — `remove_session` only does `_SESSIONS.pop(session_id, None)`, no file I/O. Plan's claim is accurate. |
| dead code (`close_stale_sessions`) note | YES | Plan §2.3 notes it; confirmed no call site via grep (only defined, never invoked outside its own module/tests). |

**Bottom line on carried findings: 6 of 7 fully resolved; the 2 Major mechanism fixes (F1, F3) each leave a
residual defect in the same locus they were meant to close** (G1, G2 below). Per `decisions.md`'s own
iteration-cap note, both **cascade-promotion** (round-1 F1 + round-3 F3) and **rollover/redirect-lifecycle**
(round-2 P-1 + round-3 F1/F2) are already at 2 bounces; G1 and G2 are new findings in exactly those two
classes, which is the situation the cap was written to flag. I return this now rather than soften the
severity — see Bottom line.

## Per-lens findings

### Lens 1 — Factual

**G1 (Major) — `/sessions/close`'s registry/lock cleanup is not covered by the redirect fix, leaking the
successor session after every post-rollover close.**
`server.py:2739-2850` (`sessions_close`). The plan's F1/F2 fix (§0, cited lines 2368/2429/2699/2765 for lock
keying, 2703/2771 for the backend call) only touches the **acquire-side** uses of `req.session_id`. Two more
uses of `req.session_id` exist in the SAME handler, both executed AFTER the (fixed) backend call succeeds:
- `server.py:2835` — `remove_session(req.session_id)`
- `server.py:2836` — `s.in_flight_locks.pop(req.session_id, None)`

Neither is cited anywhere in `1-spec.md`, `1.5-criteria.md`, or `2-plan.md` (grepped all three; zero hits for
"2835", "2836", or `remove_session` outside the plan's unrelated §2.3 note about `remove_session` already
being registry-only). Trace: after a weekly rollover, `old_sid` is evicted from `_SESSIONS` and its buffer
deleted; `get_or_hydrate_session(old_sid)` follows `rolled_to.json` and returns the successor's already-
registered `SessionState` (`_SESSIONS[successor_sid]`, created when the rollover seeded the new session). A
client still holding `old_sid` calls `POST /sessions/close`. With F1's fix applied, `_close_session_blocking`
is called with `sess.session_id` (== `successor_sid`) — correctly closes/extracts the successor's buffer,
which `close_session` (`pipeline.py:256-259`) then deletes on success. But line 2835 still calls
`remove_session(req.session_id)` — i.e. `remove_session(old_sid)`, a no-op (`session.py:230-231`, `_SESSIONS`
has nothing under `old_sid`, already evicted by the rollover). **`_SESSIONS[successor_sid]` is never removed.**
Line 2836 pops `s.in_flight_locks[old_sid]` — with F2's fix, nothing was ever stored under that key (the lock
was acquired at `successor_sid`), so this is also a no-op, and **`in_flight_locks[successor_sid]` is never
removed.**
Consequence: a session that the client believes is closed (buffer file gone, `committed`/`closed:true`
returned) remains registered in `_SESSIONS` with a stale `SessionState`, and its `in_flight_locks` entry is
never reclaimed. `GET /state/{successor_sid}` (`server.py:1255-1264`, via `get_or_hydrate_session` returning
the still-registered entry from `_SESSIONS` before ever checking disk) would report the session as live even
though its buffer is gone — an observable inconsistency, plus an unbounded per-rollover-then-close memory leak
(one orphaned `SessionState` + one orphaned `asyncio.Lock` per occurrence, for the process lifetime).
This is squarely in the SAME finding class round 3 flagged as F1 ("redirect fix incomplete for close/
snapshot") — it is not a new class, it is round 3's fix not reaching every accessor in the handler it targeted.
C16's oracle (1.5-criteria.md:182-189) checks that close/snapshot "operate on the successor... NOT a
`committed=0` false-success no-op" — it does not assert `_SESSIONS`/`in_flight_locks` post-state, so a
C16 PASS would not catch this.
**Fail-demo:** patch `server.py` to change only lines 2703/2706/2771/2773 to `sess.session_id` (the literal
scope the plan's prose describes) and leave 2835/2836 untouched; drive a weekly swap + `POST /sessions/close`
with `old_sid`; assert `successor_sid not in _SESSIONS` after the call → fails (still present).
Severity: **Major** (state leak + stale-read exposure via `/state`, not data loss — no turn is lost, the
successor's memories ARE extracted before the leak). Same class as round-3 F1 (rollover/redirect-lifecycle);
this is a genuine 3rd surfacing.

**G2 (Major, arguably higher — see reasoning) — the graduation mechanism has no path OUT of the 72h tier;
C14's own "gone after 4 passes" assertion is unsatisfiable by the mechanism as specified.**
`2-plan.md` §1.3, `bucket_of(sec)` (line 146-150): *"`24h` if `age ≤ 48h`, else `48h` if `age ≤ 72h`, else
`72h`."* This is a 3-bucket classifier with an **unbounded terminal bucket**: once a section's oldest edge
(`covers_from_ts`) ages past 72h, `bucket_of` returns `"72h"` — and will **continue to return `"72h"` on every
subsequent pass forever**, because `covers_from_ts` only grows older and there is no fourth band or exit
condition anywhere in step 3 or step 4 to catch "already spent a pass in the 72h band." Per step 4, `new_72 =
recompact([secs bucket==72h] + G72, 0.20)` — a section already classified 72h is *always* a member of
`[secs bucket==72h]` on every future pass, so it is *recompacted forever* (fraction 0.20 applied again and
again), never dropped from the summary row.

I worked this against C14's own oracle (1.5-criteria.md:151-162, plan §1.3 "Layer↔age correspondence"): "a
fake provider that preserves markers (identity/concatenative fold); sow an identifiable turn... assert the
marker's tier by pass number (1→24h, 2→48h, 3→72h, 4→**gone/archived**)." Under the mechanism as literally
specified, pass 4 would classify the (already-72h) section into `bucket_of == "72h"` again (its
`covers_from_ts` is now even older, still `> 72h` → still the terminal `else` branch) and recompact it AGAIN
with the identity/concatenative fake provider — which by C14's own methodology **preserves the marker
verbatim**. The marker does not disappear; it is still present, now inside a *second* 72h-bucket recompaction.
C14's assertion "gone from the summary (archived) after 4" has no corresponding step in §1.3 to produce that
outcome. This is not a rare edge case — it is the literal, unconditional behavior of the 3-bucket classifier
for *any* pass beyond the third.

I also checked whether this reproduces the original Problem-3 ("unbounded head growth") one tier down: it does
NOT — the recompaction is contractive (each pass's `new_72` is ≈0.20×(prior_72 + new G72), which converges to
a bounded steady-state, not unbounded growth; I verified the fixed point algebraically: steady-state
`old_72 = 0.25×(per-pass G72 size)`). So this is not a resource-bound violation. What it IS: a fidelity gap
against the owner's pinned design — spec §2b's own words are "graduates raw→24h→48h→72h→**out**" (also plan
§1.3:161-162, "Material therefore *graduates*... instead of being perpetually re-folded inside '24h'") — the
mechanism delivers graduation through 24h→48h→72h correctly (confirmed by the arithmetic below) but never
implements the "→out" step; "perpetually re-folded" is exactly what happens to the 72h tier instead, just at a
smaller, bounded scale than the original 24h-tier defect. Given CH8-6's own claim ("material aging past 72h
leaves the summary — verified SAFE... the dropped prose stays in the archive + long-term memory") is
FALSE as the mechanism is specified — nothing is ever dropped from the summary, so there is no "dropped prose"
event to have been safe about.

**Worked arithmetic (24h→48h graduation, confirming F3 IS fixed for this leg, for contrast with the 72h gap):**
Idealized ~24h-exact daily cadence, continuous chat. Pass at `T=100`: `G24` covers turns aged `(24,48]`, i.e.
`ts ∈ [52,76)`; under continuous activity the oldest surviving member approaches `ts→52` (age→48h⁻, strictly
under 48h — anything with `ts≤52` was already swept at the prior pass, `T=76`, whose eligibility cutoff was
`76−24=52`). So the new 24h section's `covers_from_ts→52⁺`. Next pass `T=124`: age of that edge
`=124−52⁺=72⁻ <72` → classifies `48h` bucket (rule: `else 48h if age≤72h`) → **graduates on schedule.** This
confirms round-3 F3 is genuinely fixed for the 24h→48h leg. The SAME reasoning applied to the 48h→72h leg also
graduates on schedule (same contractive structure). It is **specifically the terminal 72h→out leg** that has
no corresponding step.

**Missed opportunity noted in the plan itself, unresolved:** `covers_from_ts` is used correctly for 24h/48h
graduation, but the plan does not define ANY threshold beyond 72h (e.g. `_AGE_96H` or similar) at which a
72h-bucket section is dropped from the summary row entirely (and, presumably, only its *archive* copy
persists — already true, since `append_archive` records every fold's superseded content per §1d). A one-line
fourth rule — e.g. "a section already classified 72h on a PRIOR pass (not newly arrived) is archived and
omitted from the new summary row, rather than recompacted again" — would close this, but needs its own
criterion (a stronger C14 assertion already asks for this; the *mechanism* to satisfy it is what's missing).

Both G1 and G2 land in the two classes `decisions.md` flags as already at 2 bounces each
(rollover/redirect-lifecycle; cascade-promotion). See Bottom line for the SEV4 implication.

**Other Lens-1 checks (clean):** `remove_session` (session.py:222-231) confirmed registry-only, no file I/O —
plan's claim accurate. `_LOCK = threading.RLock()` at session.py:97 confirmed (redirect-recursion
deadlock-safety claim holds). `attempt_heal.py:256-266` posix-guarded `os.open(..., os.O_DIRECTORY)` +
`os.fsync` + `finally: os.close` pattern confirmed exactly as the plan describes (cited as :250-266; the guard
block itself is :256-266, comment starts :250 — accurate to within the comment/code boundary). `budget.py:31`
`_COMPACTION_SUMMARY_PREFIX` and :94-118 preserve-head re-parse confirmed unchanged; `engine.py:238-245`
(`max_tokens=80_000, preserve_tail_msgs=40`) and :405-408 (head f-string) confirmed unchanged and exactly as
cited. `compaction.py:364-374` summary_row schema confirmed to have **no** `covers_from_ts` field today — the
plan's proposed schema addition (§1.1) is genuinely new, not already present. `_respond_blocking` (server.py:
241-248) already takes the whole `sess` object (not `session_id`) — confirms why F1's fix at `/chat`/`/stream`
is a non-issue (those two already operate on the resolved object) while close/snapshot's blocking wrappers
take a bare `session_id: str` positional (server.py:474-478, 508-511) — the exact asymmetry the plan's fix
targets, consistent with round 3's diagnosis.

### Lens 2 — Logical

**M1 (Minor).** `_run_finalize_tick` (`supervisor.py:1649-1697`) calls `remove_session(r.session_id)`
(line 1686) for **every** report `finalize_stale_sessions` returns — a third, un-enumerated `_SESSIONS`
accessor beyond the four HTTP handlers, and one the plan doesn't touch. Under the plan's fix (finalize becomes
extraction-only, no longer deletes the buffer), this call still fires on every 24h-silent session, evicting
its in-memory `SessionState` even though the buffer stays on disk and the session is NOT superseded. I traced
the consequence: `get_or_hydrate_session` re-hydrates cleanly from disk on the next access (the buffer still
exists, `read_session` returns turns) — this is the already-supported Phase-B sticky-session path, so I do
NOT believe this loses data or breaks a contract. It is, however, an unenumerated accessor the plan's §5 table
doesn't list, and it means a live session that goes quiet for exactly 24h gets silently re-created as a new
`SessionState` object on its next turn (informational fields like `history`/`turns` get recomputed from disk,
which the docstring already says is safe) — worth a one-line acknowledgment so a future reader doesn't have to
re-derive that it's harmless. Not gating.

**M2 (Minor).** Neither `1-spec.md` nor `2-plan.md` states whether the weekly-rollover check (§2.2, run
"on the daily tick... under the compaction lock") happens before or after `cascade_conversation` for the SAME
session within one `_run_compaction_tick` iteration, when a session is eligible for both in the same tick
(age ≥7d AND has material crossing tier boundaries). If rollover runs first, the seed is built from
pre-cascade tiers (arguably stale by up to one cascade pass); if cascade runs first, the seed reflects the
freshly-graduated tiers (more correct). Plan §8's numbered list is BUILD order (commit sequence), not runtime
order within a tick, so it doesn't resolve this. Low-impact (both orders are defensible; the seed is "a recap,
not the record" per spec §2e), but unstated.

### Lens 3 — Missed opportunity

- The 72h-exit gap (G2) is itself the clearest missed-opportunity finding — noted above with the concrete fix
  shape (a fourth "already-72h-last-pass → archive out" rule).
- `covers_from_ts` for a MERGED cohort (two sections + a raw group landing in the same band after a gap) is
  stated to be "recorded from its inputs" (plan §1.3:164-165) but the exact aggregation rule (min across all
  inputs' `covers_from_ts`, presumably) is never spelled out as explicitly as the graduation classification
  logic is. Low-risk (min() is the only sane choice) but worth a one-line pin given how much weight
  `covers_from_ts` now carries for correctness.

### Lens 4 — Unstated assumptions & risks

- The plan's "transparent, no client contract change" framing (§0, carried from round 2/3) still implicitly
  assumes the ONLY consumers of `req.session_id` post-redirect are response echoes. G1 shows that assumption
  is false for `/sessions/close`'s cleanup calls — those are not echoes, they're state mutations that need the
  resolved id.
- The 24h→48h/48h→72h graduation-timing guarantee ("graduates after exactly 1 pass") implicitly assumes each
  tier's raw cohort spans close to the full tier width at formation time (true under dense/continuous chat,
  per my worked arithmetic). Under **narrow/bursty** cohorts (e.g. an evening-only chatter, whose G24 batch at
  formation clusters entirely near the fresh edge of `(24h,48h]` rather than spanning it) `covers_from_ts` is
  closer to `T_{N-1}-24h` than `T_{N-1}-48h`, and one pass later its age is closer to 48h than to 72h — still
  inside the inclusive `age≤48h` boundary is NOT reached in this case (I re-checked the algebra: fresh-edge
  cohort → next-pass age → 48h⁻, still graduates, so this particular worry does not materialize; I flag it only
  as an assumption worth a one-line note, not a finding, since my own check clears it — recorded so a future
  reviewer doesn't have to re-derive it).
- Q8's worst-case head budget (≈17.8k chars) is computed assuming the 48h/72h tiers are simultaneously at
  their fractional maxima the SAME day as a maximal 24h day — plausible but not the only assumption; not
  re-litigated here since round 3 already accepted this as an engineering default.

### Lens 5 — Fidelity

Loaded terms and what they pin to:
- **"graduates raw→24h→48h→72h→out"** (spec §2b, plan §1.3) — the owner's mechanism (New_mem_system.md Part 1)
  is the ONLY authority cited for this phrase. Faithful for the 24h→48h→72h legs (confirmed by arithmetic
  above); **not faithful for the "→out" leg** — no mechanism produces it (G2). The term as used in the plan
  describes a property the specified mechanism does not have.
- **"correct-by-construction"** (plan §0, re: the redirect at all four handlers) — faithful for the read/lock-
  acquire/backend-call surface; **not faithful** for the close-handler's cleanup surface (G1) — the plan's own
  scope statement ("each handler must use the resolved session_id for its backend call and its in_flight_locks
  key") is narrower than what "correct-by-construction" implies, and doesn't cover the accessors G1 names.
- **"classify by the OLDEST covered edge... the oldest edge only advances"** (spec §2b) — accurate description
  of `covers_from_ts`'s monotonicity; the CLAIM about what this achieves ("graduates... instead of being
  perpetually re-folded") is true through 72h and false at 72h→out.
- **"one code path, one lock"** — confirmed: `cascade_conversation` runs under the single per-session
  compaction lock (`buffer.py:341-392`), same as today's `compact_conversation`. No issue.
- **COMPACTION_MODEL="haiku"** — confirmed unchanged, `compaction.py:54`.
- Dream-ordering slot — confirmed a true P1 no-op (nothing in §1.3 touches ordering relative to raw-turn
  retention; ties are unaffected).

### Position lens (fires) — No issue.
`engine.py:405-408`'s head f-string is unchanged by anything in this round's revision; `budget.py:28-31/94-118`
prefix-match and preserve-head logic likewise unchanged. Section-render determinism (plan §1.2, "no nonces, no
`now()` in the render") is still the stated mechanism for C6a/C6b. Neither G1 nor G2 touches the render/re-parse
path — G1 is a registry/lock-cleanup gap, G2 is a content-graduation gap; both are orthogonal to position/cache
stability. C6 as specified remains sound.

### Concurrency lens (fires)
Enumerated beyond plan §5:
1. **G1's accessors** (`server.py:2835` `remove_session`, `:2836` `in_flight_locks.pop`) — new, not in §5's
   table, mutate the SAME two structures (`_SESSIONS` via `session.py`, `in_flight_locks`) the table already
   claims are "fixed to key by resolved `sess.session_id`... at all four sites" — the table's claim is accurate
   for the sites it lists but the list is incomplete for `/sessions/close`.
2. **M1's accessor** (`supervisor.py:1686` `remove_session`) — a third `_SESSIONS` writer beyond the four HTTP
   handlers, not enumerated; traced above as probably-harmless but unlisted.
3. Lock-ordering: no issue found — `rolled_to.json` is a plain file read inside `_LOCK`/the compaction lock,
   no additional lock is acquired inside another lock's critical section anywhere I traced (session.py's
   `_LOCK`, buffer.py's file-based compaction lock, and `in_flight_locks`' asyncio locks are never nested into
   each other in the code I read).

## Coverage challenge (CH8, new this round)

1. **(Major, G1)** Post-rollover `POST /sessions/close` leaves the successor session registered in `_SESSIONS`
   and its `in_flight_locks` entry orphaned forever; observable via `GET /state/{successor_sid}` reporting a
   live session after the client was told it closed. No criterion (C10/C15/C16/C19) asserts `_SESSIONS`/
   `in_flight_locks` post-state after a close on a redirected sid.
2. **(Major, G2)** No criterion can observe "material genuinely leaves the summary at 72h" succeeding, because
   the mechanism doesn't implement it — C14 as worded requires this but the fail-demo I ran mentally against
   the described mechanism shows it can't pass. This is the "no criterion catches it" case inverted: the
   criterion tries to catch it and is right to, but the plan's own mechanism can't satisfy the criterion it
   wrote for itself.
3. **(Minor, M1)** `_run_finalize_tick`'s `remove_session` call (supervisor.py:1686) under extraction-only
   finalize — traced as probably harmless (re-hydration recovers) but unenumerated in plan §5.
4. **(Minor, M2)** Same-tick ordering of weekly-rollover-check vs. cascade fold for a session eligible for
   both — unspecified.
5. **(Low, carried, still open)** `run_backlog_migration` × new sectioned rows across restarts — plan §4
   documents this as self-healing via the tolerant reader; I re-confirmed the tolerant-reader claim is
   plausible (§1.1's fallback "sections={'24h': {text: legacy}}" is a reasonable read of an un-sectioned row)
   but no criterion directly exercises "migration self-heals a flattened row on the next cascade tick" as an
   integration scenario — still just prose, not tested. Not new this round; noting it remains open.

## Label audit

- **C14** — targets the real graduation mechanism (non-tautological, asserts against the marker not
  `covers_until_ts`) — the ORACLE is well-constructed; the problem is the SYSTEM UNDER TEST (§1.3's mechanism)
  cannot pass it as specified (G2). This is a mechanism defect, not a proxy/label defect — C14 is doing its
  job correctly by being unsatisfiable.
- **C16** — governs the real path (`get_or_hydrate_session` + all 4 handlers) for the properties it states
  (200 + successor buffer write; close/snapshot operate on successor, not false-success). It does NOT claim
  anything about `_SESSIONS`/`in_flight_locks` post-state, so it is not mislabeled — it's simply narrower than
  the actual blast radius of the redirect fix. Recommend broadening C16 (or a new C20) to assert
  `successor_sid not in _SESSIONS` is FALSE... i.e. assert the successor's registry+lock entries are correctly
  present/absent as appropriate after close (currently unobserved — this is CH8-1 above).
- **C19** — governs lock-KEYING at the four acquire sites (`setdefault`) correctly; does not (and was never
  claimed to) cover the `.pop()` cleanup site. Not mislabeled, just doesn't reach G1.
- **C1–C13, C17, C18** — re-checked against source this round for any new drift: none found. Citations remain
  accurate (spot-checked compaction.py:364-374, buffer.py:265/290/310/341-392, engine.py:238-245/405-408,
  budget.py:28-31/94-118 — all exact).
- **A1/A2** — advisory labels remain justified (no live/replay workload; CI-green is build-hygiene).

## Assumptions/risks challenge (A1/A2)
A1 (regression metrics advisory) and A2 (CI-green advisory) are correctly scoped given the stated "no live
bridge" constraint; nothing in this round's revision changes that constraint. No challenge.

## Bottom line

Worst severity: **Major**, tied between **G1** (close-handler registry/lock cleanup not redirected — a
residual of round-3 F1, same class) and **G2** (no mechanism for material to actually leave the 72h tier — a
residual of round-3 F3, same class; C14 as written is unsatisfiable by §1.3 as written). Six of seven round-3
carried findings are genuinely, fully resolved — the fixes for F1 (redirect at 4 sites) and F3 (oldest-edge
classify, no-remerge) are real and correctly reasoned for the specific mechanisms they targeted; both new
findings are narrower residuals in the SAME two loci, not new classes of problem, and both have a small,
mechanical fix:
- G1: also switch `server.py:2835` (`remove_session`) and `:2836` (`in_flight_locks.pop`) to
  `sess.session_id`; extend C16 (or add C20) to assert `_SESSIONS`/`in_flight_locks` cleanup targets the
  resolved id.
- G2: add a fourth classification outcome — a section already in the 72h bucket on a PRIOR pass (not newly
  arrived this pass) is archived and dropped from the new summary row instead of being recompacted again;
  strengthen C14's own pass-4 assertion is already correct, it just needs the mechanism built under it.

**Routing: back to stage 2, narrowly scoped** — same disposition class as rounds 1–3 (both fixes are small,
targeted, and don't touch the parts of the design already validated: position lens, C1-C13/17/18, the
resolved 6/7 carried findings).

**Iteration-cap flag (SEV4), reported explicitly per the charter and `decisions.md`'s own round-3 note:**
`decisions.md` records BOTH the rollover/redirect-lifecycle class (round-2 P-1 + round-3 F1/F2 = 2 bounces)
and the cascade-promotion class (round-1 F1 + round-3 F3 = 2 bounces) as already at the 2-bounce cap, with the
explicit note that "a third bounce on either class in round 4 triggers the human tie-break." G1 is a new
finding in the rollover/redirect-lifecycle class; G2 is a new finding in the cascade-promotion class. Both are
narrower/smaller in scope than their round-3 predecessors (each is a specific missing accessor/rule, not a
wrong mechanism), but by the letter of the cap as `decisions.md` states it, **this is a third bounce on BOTH
classes simultaneously** — the orchestrator/owner should treat this as the human-tie-break trigger rather than
routing a fourth narrow revision straight back to stage 2 on the reviewer's authority alone.
