# 3-redteam-plan-r6.md — Cascade Compaction, stage-3 plan red-team, ROUND 6

Cold, independent reviewer. No shared context with the author beyond the charter text handed to me. All
citations below were verified by directly reading the named file/line in the worktree
`/home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction` at the time of this review
(date confirmed via `date`: 2026-08-13). The owner-pinned design (3 tiers/labels, terminal tier-3, oldest-edge
graduation, tier-3 hard cap = 0.20×tier1) is treated as GIVEN — I verified conformance, not merits.

---

## Provenance

Worktree: `/home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction`
Branch: `ThinkerOfThoughts/cascade-compaction` (HEAD `cd29bc61`)

sha256 of every file read for this round:

```
0a085db46019f968f63f73480135ac780fa00d4524844d14b8a6af075bba818f  changes/cascade-compaction/1-spec.md
1959d6383565b37f7706619aa22c1b11b2dd4c90fcaeff70bd2ad354ee3fc1c0  changes/cascade-compaction/1.5-criteria.md
ef890b0fa84adf20bae3e051fb09d6aca7b004b88ab31f56ce4ad1c9635ca8d8  changes/cascade-compaction/2-plan.md
cc9b6e22ac3cf05aa1109e84abb5d8217619153d85cf172ae1b84644b1aad7fb  brain/chat/compaction.py
773f1e0b0ae3dad2cbdc4f316e16e65194d09ab047665ff740259eace1a4dc34  brain/chat/session.py
f0b0b715746bc9f8e27964ec7c24301a14b6b80dab4d38db8f8b533d1df62d7e  brain/bridge/server.py
cfe8b63b3d642dabe998f52b52a087ccc4c0acbd8bab5b32bba177f0b309d331  brain/ingest/pipeline.py
ca6eeba8070959cf502e76177a3a635832ecc0377e12e43773b3cf9629c116b2  brain/bridge/supervisor.py
cefb079963884fbafea3a0d8125c74bdc3a9e889894731329f482f49f93da56b  brain/ingest/buffer.py
d359be520046a66c502ca5f0b56a0c61e8f4a13fbef93868e6fe127bed0d1260  brain/chat/engine.py
b71ed34d8f0d50108739fad682a3698989cfb5bd1278f0e4a3c7a011d873c7ca  brain/chat/budget.py
7d32a50b29f47b34645bd48c04b6cae3bb62e2dd15e031712b2b752fd59e1d77  brain/chat/compaction_migration.py
74dd5ba03614872c430fd2f3d2e40f23d4f70349a69816309ea0a4e7b8ee054e  brain/monologue/ambient.py
ddfbec5d20d3b4655afe7c14ce0c2d24564f09fa3e8f78f79d6560eb67051d90  brain/health/attempt_heal.py
```

Also read (context, not cited with sha256): `changes/cascade-compaction/decisions.md` (gate log, all 5 prior
rounds' dispositions).

Additional verification performed beyond reading: `grep -rn "get_or_hydrate_session(" --include=*.py .` over
the whole worktree (not just `server.py`) to independently confirm the round-5 F2 claim of exactly FIVE
production call sites and rule out a 6th caller anywhere else in the codebase.

---

## Carried-forward resolution table (round 5 → round 6)

| ID | What round 5 demanded | Verdict this round | Evidence |
|---|---|---|---|
| **F1** (Major, terminal tier-3 multi-input fold) | Plan §1.3 must specify the every-cycle join of persisting-tier3 + graduated-tier2 (+ G72) → 20% → cap → sentence-truncation → double-reject fallback that drops neither input; C14/C3 must force both inputs to survive within the cap. | **RESOLVED.** `2-plan.md` §1.3 lines 198-216 give the explicit 4-step mechanism (ordered lossless-leaning join → compact to 20% → cap by sentence truncation → double-reject fallback = join truncated to cap, never a single-input fold). `1.5-criteria.md` C14 and C3 both explicitly assert "markers from BOTH the persisting tier3 AND the newly-graduated tier2 co-survive... within the cap" across "many steady-state cycles," with the fail-demo `(c) a single-input tier3 fold that drops one of the two inputs`. No path found that silently drops an input; the cap applies to the OUTPUT only (post-join, post-fold), not to either input independently. | plan.md:198-216; criteria.md C3 (45-53), C14 (159-176) |
| **F2** (Major, redirect one-sweep + structural guard, possible 6th caller) | §0 must enumerate ALL 5 `get_or_hydrate_session` callers, rebind each to `sess.session_id` for every downstream op; C21 structural guard; independently grep for a 6th caller. | **RESOLVED.** Independent `grep -rn "get_or_hydrate_session(" --include=*.py .` over the entire worktree returns exactly 5 production hits, all in `brain/bridge/server.py` (:1261 `/state`, :2365 `/chat`, :2424 `/stream`, :2696 `/sessions/snapshot`, :2753 `/sessions/close`) plus 7 hits in `tests/unit/brain/chat/test_session.py` (test code, not a handler, out of C21's scope). No 6th production caller exists anywhere. Directly read all 5 handler bodies; each currently uses the RAW id downstream exactly as the plan's table (§0, lines 59-65) claims — `/state`'s `in_flight` lookup at line 1264 uses raw `session_id` (not `sess.session_id`), `/chat`'s lock key + echo at 2368/2398 use raw `req.session_id`, `/stream`'s lock key at 2429 uses raw `session_id`, `/sessions/snapshot`'s backend call + lock at 2699/2705/2714/2721/2729 use raw `req.session_id`, `/sessions/close`'s cleanup at exactly `:2835`/`:2836` (`remove_session(req.session_id)`, `in_flight_locks.pop(req.session_id, None)`) matches the plan's own line-number citations byte-for-byte. C21's fail-demo ("the current (pre-sweep) code → the check fires on every unfixed site") is independently CONFIRMED true by this read — every one of the 5 sites is presently unfixed, so a structural scanner built against today's tree would genuinely trip on all 5, which is exactly what "shown able to fail" requires. | server.py:1261,1264,2365,2368,2398,2424,2429,2696,2699,2753,2835,2836; criteria.md C21 (224-232) |
| **N1** (minor, stale 960-char Q8 figure) | Remove/correct. | **RESOLVED.** `grep -n "960"` across all three artifacts returns zero hits. |
| **M-1** (minor, multi-input fold semantics documented) | — | **RESOLVED** — see F1 row; §1.3 is explicit. |
| **M-2** (minor, apply_budget×sectioned-row test) | — | **RESOLVED.** Criteria C22 added (criteria.md:234-239), plan §1.3 sub-ops note (246-250) and build order §8 step 3 both wire it in. |
| **F3** (minor, same-tick lock-continuity MECHANISM stated, not just asserted) | — | **NOT FULLY RESOLVED — see Concurrency-lens finding below.** A paragraph titled "The lock-continuity mechanism (stated, not just asserted)" was added (plan.md:311-318), but its content asserts an OUTCOME ("`_run_compaction_tick` acquires the... lock once... does not release between the two") that does not reconcile with the actual lock primitive I read in `buffer.py`/`compaction.py`: the per-session compaction lock is acquired and released SELF-CONTAINED inside `compact_conversation` (compaction.py:254 acquire / :425-426 release-in-finally), is a plain non-reentrant `O_CREAT\|O_EXCL` pid-file (not a handle the caller can hold across two calls), and `_run_compaction_tick` itself (supervisor.py:1619-1646) currently acquires no lock of its own — it just loops calling the fold function. The plan does not say which of the two possible fixes (externalize lock acquisition to the tick, or fold the rollover check inside `cascade_conversation`'s own existing lock scope) it intends, and no build-order step or touched-file note mentions this refactor. This is a real, still-open specification gap on the exact carried item — ranked at its original severity (Minor) since it's readily fixable and no criterion depends on it being wrong, but it is not resolved as claimed. | plan.md:311-318 vs compaction.py:254,425-426 vs supervisor.py:1619-1646 vs budget.py:71-86 |
| Owner design conformance | 3 tiers/labels, terminal tier3, caps, graduation by oldest edge | **CONFORMS.** plan.md §1.3 step 3 classifies by `covers_from_ts` (oldest edge), never co-folds prior-tier1-with-fresh-raw, tier3 is terminal/re-compacted forever with its own 0.20×tier1 cap; render uses static labels "yesterday"/"day before yesterday"/"a few days ago" (plan.md:140-142). |
| G1 (close cleanup + `/state`) | C20 + structural guard | **RESOLVED** — see F2 row; confirmed by direct read of server.py:2835-2836 and :1264-1265. |
| M1 (finalize `remove_session` enumerated) | — | **RESOLVED.** plan.md §5 table row and decisions.md both correctly describe `supervisor.py:1686`'s `remove_session(r.session_id)` as harmless registry-only eviction; confirmed by reading `session.py:222-231` — `remove_session` only pops `_SESSIONS`, no file I/O, matching the plan's claim exactly. |

**Bottom line on carried items: 6 of 7 fully resolved; 1 (F3, same-tick lock-continuity mechanism) is still an
open specification gap, carried forward at its original Minor severity.**

---

## Lens 1 — Factual

Extensive direct-source verification was performed (not spot-checking): every line-locus cited in `1-spec.md`
§6/§7 and `2-plan.md` §0/§5 for `compaction.py`, `session.py`, `server.py`, `pipeline.py`, `supervisor.py`,
`buffer.py`, `engine.py`, `budget.py`, `compaction_migration.py`, `ambient.py`, and `attempt_heal.py` was read
against the actual file content. Every citation checked (>40 individual line-locus claims) matched the source
exactly — function names, line ranges, and quoted code fragments are all accurate. Notable precise matches:
`compaction.py:254` (lock acquire), `:338` (`.strip()`), `:364-374` (summary row shape), `:425-426` (lock
release); `session.py:97` (`_LOCK = threading.RLock()`), `:129-205` (`get_or_hydrate_session`), `:222-231`
(`remove_session`); `server.py:1261/1264-1265/2365/2368/2424/2429/2696/2753/2835-2836`; `pipeline.py:150-154`
(close_session empty guard), `:273-331` (`extract_session_snapshot`, cursor read/write at :317-318 unguarded,
confirmed), `:475-526` (`snapshot_stale_sessions`), `:504-506` (ghost-delete), `:529-626`
(`finalize_stale_sessions`), `:557-559`/`:587-590` (delete sites), `:591-609` (poison-move); `supervisor.py:129`
(`finalize_after_hours=24.0`), `:138` (`compaction_interval_s=86400.0`), `:639-662` (cadence block),
`:1619-1646` (`_run_compaction_tick`), `:1649-1697` (`_run_finalize_tick`), `:1686` (finalize `remove_session`);
`buffer.py:162-165` (`delete_session_buffer`), `:265-281` (`rewrite_session_atomic`), `:290-307`
(`append_archive`), `:310-312` (`read_archive`); `attempt_heal.py:250-266` (directory-fsync pattern, byte-for-
byte reusable). No factual misstatement found in this round.

**No issue found** on this lens beyond what's captured under Concurrency/Fidelity below.

## Lens 2 — Logical

**Finding L-1 (Major) — "one hop" and "recurse... cap the chain to a small N" describe two different
mechanisms, and the plan never resolves which one is intended.**
`2-plan.md:55` and `:90-92`, and `1-spec.md:271-272`, all describe the successor-pointer follow as a
**"one hop"** redirect — but `plan.md:55` in the same breath says `get_or_hydrate_session` **"recurse[s]
into `get_or_hydrate_session(successor_sid)`... cap the chain to a small N to bound a pathological pointer
loop."** These are logically different designs:
- If it is genuinely capped at exactly one hop (as "one hop" literally reads and as `1.5-criteria.md:191`'s
  "follows one hop" and `plan.md:411`'s "one-hop redirect (chain capped)" both restate), then a client whose
  original sid has been superseded by **two or more** successive weekly rollovers (old→successor1→successor2)
  cannot be resolved: `get_or_hydrate_session(original_sid)` would follow the pointer to `successor1`, find
  `successor1` itself evicted+deleted (because it too rolled over), and — if capped at ONE hop — return `None`
  instead of continuing to `successor2`. That is a 404 for a client that never restarted, never re-attached,
  and did nothing wrong — precisely the "continuously-used conversation" population 1c-B exists to serve
  (spec.md:138-142: "a continuously-used conversation never hits the 24h gap... the continuously-attached
  client is the whole target population").
- If instead the mechanism is real multi-level recursion bounded by "a small N" (which the word "recurse" and
  the RLock-reentrancy justification at `plan.md:90-92` both suggest), the plan never names a concrete `N`,
  never states whether N is measured in rollover-generations or something else, and — critically — **no
  criterion tests a 2-hop chain.** Every OTHER engineering default resolved in this spec (`_SECTION_24H_CHAR_CAP
  = 12_000`, `_WEEKLY_ROLLOVER_AGE = 7 days`, `_ROLLOVER_QUIET_GAP = 30 min`) is given a concrete value plus a
  reasoning paragraph in plan §0. The chain-cap constant is the one exception: it is named only descriptively
  ("a small N"), never assigned a value, and never justified against the feature's own stated multi-month/
  multi-year operational lifetime (a companion instance is not expected to be torn down after one week).
This is not a hypothetical: for the intended population (a client that keeps the same sid forever and is
redirected transparently), a SECOND weekly rollover is not an edge case — it is the second of an unbounded
sequence that will keep happening for as long as the persona is used. If a genuinely small N (e.g., a literal
1, matching the "one hop" language) is what gets built, this is a **live-population regression** that ships
invisibly, because C16/C9's tests only ever exercise a single rollover generation.
- **Fail-demo I can construct:** implement `get_or_hydrate_session`'s redirect as `if raw not resolvable: check
  rolled_to.json once, no recursion` (a literal reading of "one hop") — a fixture that performs two sequential
  weekly rollovers on the same persona, then resolves the ORIGINAL (generation-0) sid, would 404. No criterion
  in `1.5-criteria.md` would catch this, because C16/C9's fixtures (as specified) only perform one swap.
- **Severity:** Major — it's an internally contradictory description of a mechanism serving the feature's core
  population, with an unspecified bound and zero test coverage past one generation.

## Lens 3 — Missed opportunity

**Finding MO-1 (Major) — migration/tolerant-reader's `covers_from_ts := covers_until_ts` approximation is
unacknowledged and untested, and directly risks reproducing the defect (#82) this phase exists to fix.**
Both the migration (`plan.md:380-381`: "legacy `text` → `sections['24h']`, spans from `covers_until_ts`") and
the defensive tolerant-reader for not-yet-migrated rows (`spec.md:132-135`: "Legacy rows... read as
`sections={'24h': {text: <legacy text>, spans from covers_until_ts}}`") derive a synthesized `24h` section
whose only available timestamp is the legacy row's `covers_until_ts` (the NEWEST covered ts — legacy rows never
recorded an oldest-edge ts; confirmed by reading `compaction.py:364-374`, the current summary-row shape, which
has only `covers_until_ts`, no `covers_from_ts`). The natural implementation sets the synthesized section's
`covers_from_ts` equal to that same value — i.e., treats a summary that may cover WEEKS or MONTHS of
compressed history as if its oldest content were only as old as its newest. Since the cascade's own
graduation mechanism classifies a section's tier by `now − covers_from_ts` (the OLDEST edge, per plan.md:172-
177, explicitly chosen over the newest edge to avoid exactly this kind of age-laundering — this is the F3/round-
3 fix), a freshly-migrated legacy summary — genuinely arbitrarily old — will classify as "24h"/"yesterday" on
the very next cascade pass, i.e., it is rendered to the model under the label "yesterday" when it may be months
old. This is not a rare edge case: **every existing persona** carries exactly this shape today (spec.md §2g:
"Existing personas carry one single-layer summary") and will hit this on migration. It reproduces the
mislabeling defect (#82 — "a reader cannot tell 2-hours-ago material from 3-days-ago material") that this
entire phase's Problem 1 exists to close, specifically at the migration boundary.
- Unlike Q8/weekly-age/quiet-gap (each given a "documented module const with a reasoned default + a one-line
  follow-up note" per spec §5's own stated policy), this default gets no such treatment anywhere in the three
  artifacts — no reasoning paragraph, no accepted-tradeoff note, no follow-up marker.
- **No criterion tests this.** C12 only asserts migration produces the 3-section *shape* and that a re-run is a
  no-op — it never runs a cascade pass on the migrated output to check tier placement. C14's oracle explicitly
  starts from a **fresh, marker-tracked, day-0** scenario (a fake provider sowing turns "on consecutive days"),
  never from a migrated/legacy-seeded section. The interaction between migration and graduation is unexercised.
- A concrete, cheap-to-state fix exists (e.g., set the synthesized `covers_from_ts` to something conservatively
  OLD — the persona's creation date, or simply omit it so the section is treated as already fully-aged into
  tier 3 on first cascade — rather than defaulting to "as young as possible"), but the plan doesn't discuss the
  choice at all, so I can't tell whether the underspecification is an oversight or a considered (undocumented)
  tradeoff.
- **Severity:** Major — silent, universal (100% of existing personas), and reproduces the phase's own target
  defect at exactly the moment migration is supposed to fix it.

## Lens 4 — Unstated assumptions & risks

**Finding UA-1 (Minor, ties to carried F3) — "single lock hold across fold-then-rollover" assumes a lock
primitive that can be held across two calls; the one that exists cannot, without a change nowhere specified.**
See the carried-findings table above for the full citation trail. In short: `acquire_compaction_lock`/
`release_compaction_lock` (buffer.py) is a plain `O_CREAT|O_EXCL` pid-file lock, acquired and released
entirely inside `compact_conversation` (soon `cascade_conversation`) with no handle exposed to the caller, and
it is **not reentrant** — the reap logic checks "is the holder pid alive," and since the holder in a same-
process double-acquire IS alive, a second acquire from the same process while the first is still held would
be treated as **busy, not reentrant**, and return `False`. `_run_compaction_tick` (supervisor.py) currently
holds no lock of its own. `apply_budget` (budget.py:71-86) is a real, independent, concurrently-reachable
caller of the same self-contained fold/lock unit (triggered synchronously mid-turn from a worker thread,
wholly asynchronous to the daily-tick thread) — so the race the "single lock hold" claim exists to close
(apply_budget's 24h-only backstop re-folding the 24h section between the tick's cascade-fold and its own
rollover-seed-read) is real, not decorative. The plan's revised paragraph (plan.md:311-318) states the desired
outcome but not the code-level restructuring (who now owns lock acquisition; whether `cascade_conversation`'s
internal acquire/release is removed in favor of an external hold, or the rollover check is folded inside the
existing lock scope) needed to make it true against this primitive.
- **Severity:** Minor (unchanged from round 5's F3) — resolvable with an obvious-enough refactor, and no
  criterion currently depends on the claim being true (C9 tests only the successful-swap outcome, not
  resistance to an apply_budget interposition specifically at the fold→rollover boundary), so it doesn't block
  build, but it is not "stated" to the standard the carried item asked for.

**Finding UA-2 (Minor) — C18 names the wrong protected artifact, and its fixture choice is unpinned, so the
oracle may not be shown-able-to-fail on the scenario that actually needs it.**
Both `1.5-criteria.md` (C18: "...the seed's cursor set so the seed row is not re-extracted as if it were a
fresh raw turn") and `2-plan.md:273` describe the cursor-set as protecting **the seed summary row itself**
from re-extraction. Reading `pipeline.py:273-331` (`extract_session_snapshot`) shows this protection already
exists unconditionally and independently of any cursor state: `turns = [t for t in turns if t.get("speaker")
!= "summary"]` (the line immediately following the cursor-gated read) drops every `speaker=="summary"` row
before extraction proceeds, regardless of the cursor. So the *seed row* can never be mis-extracted whether or
not its cursor is set — that part of C18's stated rationale is not actually contingent on the new mechanism it
is meant to be testing. The **actual** cursor-dependent risk is the raw tail 1c-B's seed carries forward (the
"3 tiers + 40 most-recent messages," `min_keep_tail=40` — ordinary `speaker=="user"/"assistant"` rows, NOT
filtered by the summary-only guard): without the new session's cursor pre-set past their timestamps, THOSE 40
already-extracted messages would be picked up as apparently-new by the new session's first extraction pass,
risking duplicate memories (the plan's own F4 admits the downstream embedding-dedupe is "a soft net, not a
guarantee"). 1c-A's seed (`min_keep_tail=0`, summary-only, per §0 "Seed asymmetry") carries no raw tail at all,
so cursor-setting is causally inert there — the fail-demo ("write the seed without the cursor-set → oracle
flags the seed re-extracted") would **not** actually flag anything if the C18 fixture is built against the
1c-A shape, because the structural speaker-filter already prevents the symptom with or without the fix. Since
C18 is discussed in the plan's §2.1 (the 1c-A section) and its own wording talks about "the seed" generically,
an implementer following the prose is more likely to reach for the 1c-A fixture — which would make C18 pass
even against a broken/missing cursor-set implementation, failing ST1.5f's "oracle shown able to fail" bar for
the one scenario (1c-B, 40-message tail) where the risk is real.
- **Severity:** Minor — the underlying protection is sound and needed for 1c-B; this is a criterion-wording/
  fixture-pinning gap, fixable by naming the 1c-B/40-tail scenario explicitly as C18's fixture.

## Lens 5 — Fidelity (owner mechanism vs proxy)

Loaded terms pinned and checked against the plan's actual mechanism:

- **"Terminal"** → pins to: material graduates 1→2→3 then is re-compacted forever in tier 3, no eviction leg,
  no 4th tier. Plan §1.3 step 4 (lines 189-193) implements exactly this — the prior tier-3 section always
  classifies back into the 72h band and is folded again every cycle. **Conforms.**
- **"Multi-input"** → pins to: the terminal tier-3 fold combines the persisting prior tier-3 text AND the
  newly-graduated prior tier-2 text (plus any raw crossing 72h) every steady-state cycle, with neither
  silently dropped. Plan §1.3 lines 198-216 specify this exactly, with an explicit ordered join + fallback that
  never drops either source. C14/C3 assert both markers survive. **Conforms**, and — per the carried-findings
  table — this is the F1 item that is now genuinely resolved, not merely asserted.
- **"Labels"** → pins to: static, byte-stable, human age-band strings ("yesterday"/"day before yesterday"/"a
  few days ago"), not computed dates. Plan §1.2 line 140 states this explicitly ("a static tier label (NOT a
  computed date), so it stays byte-stable"), matching C5, C6. **Conforms.**
- **"Caps"** → pins to: tier1 hard cap (12 000 chars, documented reasoning) + tier3 hard cap (0.20× tier1,
  owner-specified), tier2 bounded-by-input only. Plan §0 gives the full worst-case arithmetic (12000 + 4800 +
  2400 ≈ 19.2k chars ≈ 4.8k tokens — I recomputed this sum and it is correct) inside the live 80k-token prompt
  budget. **Conforms.**
- **"Structural guard"** → pins to: a static/scan-based check over `server.py`, not a runtime probe, that fires
  on every one of the 5 pre-fix sites. Independently verified via direct read of all 5 handler bodies (see the
  Provenance/F2 row above) — the check as specified would genuinely fire on today's tree. **Conforms, and
  independently confirmed rather than merely trusted.**

**No fidelity mismatch found for any of the five pinned loaded terms.** The two Major findings above (L-1,
MO-1) are not fidelity mismatches against the owner's pinned design — they are gaps in an author-owned
engineering default (chain-cap N; migration's `covers_from_ts` synthesis) that the owner never ruled on and
that the "resolved-in-this-spec" policy (spec §5) was supposed to, but didn't, cover for these two.

---

## Position lens (fires — ST1.5d)

Checked: do the static human labels stay byte-stable, and does the `"[Earlier in this conversation:"` head
prefix + `budget.py` re-parse stay intact?
- Labels are static strings selected by tier index, never computed from `now()` (plan.md:140-142) — byte-stable
  by construction.
- `engine.py:405-408` still authors `f"[Earlier in this conversation: {summary_text}]"` unchanged; `budget.py:31`
  still matches only `_COMPACTION_SUMMARY_PREFIX = "[Earlier in this conversation:"` and preserves the whole
  block (`budget.py:94-118`, read in full this round). The 3-section render is confined to what `summary_text`
  contains — the prefix contract is untouched. C6 exercises this by execution (render×2 byte-equality +
  re-parse), not by inspection.
**No issue found.** Clean pass — the prefix/parse coupling I verified by reading both sides (`engine.py` and
`budget.py`) directly is unchanged.

## Concurrency lens (fires — ST1.5e)

Enumerated accessors (session buffer, archive, cursor, `rolled_to.json`, `in_flight_locks`, `_SESSIONS`) were
checked against plan.md §5's own table by re-deriving each row from source rather than trusting the table:
- Session buffer: confirmed the compaction lock (buffer.py, file-based, non-reentrant) genuinely serializes
  `compact_conversation`/cascade/rollover writers against each other; confirmed `finalize_stale_sessions` no
  longer needs to delete (decoupling is real, since its ONLY current delete sites are the two enumerated by
  the plan, both confirmed at the cited lines).
- `in_flight_locks`/`_SESSIONS`/close-cleanup: confirmed all 5 handlers currently use the raw id (F2 table row)
  — the planned uniform-rebind fix genuinely closes what it claims to close.
- `rolled_to.json`: the plan's claim that it's "written before the delete so old-sid never resolves to
  nothing" is architecturally sound as a SINGLE-hop guarantee, but see Logical-lens Finding L-1 — the
  MULTI-generation chain behavior of this exact pointer is the open question, not the single-hop write-order
  claim (which I have no issue with).
- The two mandatory lost-update criteria (C10, C11) both correctly specify a fail-demo that must fail against
  the unguarded/pre-change version, satisfying H4.
- **Finding (see UA-1 above, carried F3):** the same-tick single-lock-hold claim for fold-then-rollover does
  not reconcile with the actual (self-contained, non-reentrant) lock primitive. This is the one concurrency
  item I'd call still-open; ranked Minor as it doesn't currently threaten any gating criterion's correctness,
  only the plan's own stated guarantee about ordering.

---

## Coverage challenge (CH8)

Ranked by how directly each threatens the shipped behavior of the target population, worst first:

1. **(Major — restates Logical L-1)** Multi-generation rollover chain (client holding a sid from 2+ rollovers
   ago). Concrete scenario: a persona used continuously for 3+ months accumulates >10 weekly rollovers; no
   fixture in `1.5-criteria.md` exercises more than one swap. If the chain-follow is capped smaller than the
   number of rollovers a long-lived persona will accumulate, the original client sid eventually 404s
   permanently. Not observed by C8, C9, or C16 as written.
2. **(Major — restates Missed-opportunity MO-1)** Migration/tolerant-reader `covers_from_ts` synthesis feeding
   directly into the cascade's own age-classification. Concrete scenario: any of today's existing personas,
   post-migration, has its accumulated (possibly months-old) history mislabeled "yesterday" on the very next
   cascade pass. Not observed by C12 (shape-only) or C14 (fresh-start-only).
3. **(Minor)** Same-tick lock-continuity (UA-1/carried F3): whether `apply_budget`'s independent backstop can
   genuinely interpose between a daily tick's cascade-fold and its own rollover-seed-read for the SAME session.
   No criterion drives this interleaving (C10's injected-interleaving oracle is scoped to finalize vs rollover,
   not backstop vs cascade-then-rollover).
4. **(Minor)** C18's fixture-scenario ambiguity (UA-2): whether the seed-cursor test is built against the
   1c-A (vacuously-passing) or 1c-B (actually-discriminating) seed shape is unpinned in both the plan and the
   criteria text.
5. **(Nitpick, informational — not a new gap)** The plan's own §2.1 "sync-work bound" (A1-a) and the SYNC
   full-fold cost are explicitly accepted-not-gated (no criterion bounds the blocking Haiku round-trip's
   wall-clock time) — this is already flagged as a deliberate, owner-accepted cost in the plan text itself, so
   I list it only for completeness, not as new residue.

## Label audit (CH9/CH10)

Walked all 22 gating criteria + 2 advisory against "does it exercise the real path it governs, not a proxy":

- **C1, C2, C5, C6, C7, C8, C9, C10, C11, C12, C13, C15, C16, C17, C19, C20, C21:** each names a real-path
  oracle (TestClient / real function / structural scan over the real file), each has a fail-demo that would
  genuinely fail against the pre-change or a plausible-wrong scratch. No proxy substitution found in this
  round for any of these seventeen.
- **C3 — "both caps over steady-state":** exercises `_fold_into_section` machinery across "many steady-state
  cycles" including the 2-input tier-3 fold; non-tautological (asserts against the char cap constant, not
  against the classifier's own output). **Clean.**
- **C4 — cascade double-reject fallback:** explicitly stated to drive the real `cascade_conversation` path,
  not only the `_validate_fold_output` unit — confirmed by the criteria text itself distinguishing (a) the
  unit-level single-fold case from (b) the cascade double-reject case. **Clean, non-proxy.**
- **C14 — "non-tautological":** correctly asserts against actual marker content (not `covers_until_ts`, the
  classifier's own input) — genuinely non-tautological for what it tests (plumbing correctness: does the join
  route both sources into the fold call and does the cap/truncation logic work). **One scope note, not a
  defect:** it necessarily uses "a fake provider that preserves markers (identity/concatenative)" rather than
  the real haiku model, so it cannot and does not test whether the REAL model's summarization actually retains
  salient content from BOTH inputs when compressing for real (as opposed to correctly routing both inputs INTO
  the compression call). Given the project's own stated measurement posture (A1: no live bridge, so LLM-output
  quality is advisory-only project-wide, not specific to this change), this is a consistent, not a novel, scope
  boundary — flagged for completeness under CH9, not counted as a finding.
- **C18:** see Unstated-assumptions Finding UA-2 above — the wrong artifact is named as protected, and the
  fixture that would make the oracle discriminating (1c-B) is not pinned by the criterion text. This is the
  one criterion in the whole file I'd call NOT cleanly passing the "shown able to fail on the scenario that
  matters" bar.
- **C21 — "shown able to fail on the pre-fix tree":** independently re-verified (not just trusted) — see the
  Provenance/F2 table row. **Confirmed, not merely asserted.**
- **C22 — apply_budget × sectioned row:** real path (`apply_budget` itself), asserts non-corruption + head
  re-parseability. **Clean.**
- **Advisory A1/A2:** reasons given are consistent with the project's stated measurement posture (no live
  replay workload) and standard CI-hygiene framing respectively. No challenge.

---

## Bottom line

Six of the seven round-5 carried items are genuinely resolved on this reading, including both prior Majors
(F1 terminal multi-input fold, F2 redirect one-sweep + structural guard) — both independently re-verified
against source, not merely re-read in the plan text. The seventh carried item (F3, same-tick lock-continuity
mechanism) is still open, unchanged in severity (Minor).

This round surfaces two NEW findings I rank Major: (L-1) the successor-pointer redirect's "one hop" vs.
"recurse... cap the chain to a small N" language is internally inconsistent, the chain-cap constant is the one
engineering default in the whole spec never given a concrete value or reasoning, and no criterion exercises
more than one rollover generation — a real risk for the feature's own target population (a client that never
re-attaches and will accumulate many rollovers over a long-lived persona's life); and (MO-1) the migration/
tolerant-reader's `covers_from_ts := covers_until_ts` approximation is undocumented, untested against the
cascade's own age-classifier, and would mislabel every existing persona's accumulated history as "yesterday"
on first migration — reproducing the exact #82 defect this phase exists to fix, at the one moment (migration)
that is guaranteed to affect 100% of current production data.

Neither finding is a fidelity mismatch against the owner-pinned design, and neither requires re-litigating any
owner ruling — both are gaps in author-owned engineering defaults that the plan's own stated policy (§5:
"documented module const with a reasoned default + a one-line follow-up note") was supposed to cover and, for
these two, didn't.

**Routing: back to stage 2 (narrow) — not clean, not minors-only.** Scope for the revision: (1) pin a concrete
chain-cap constant with reasoning, and reconcile the "one hop" phrasing with the actual (apparently multi-hop
capable) recursive mechanism, and add a 2-generation-rollover test to whichever criterion is the natural home
(C16 extension, or a new C23); (2) specify and document the migration/tolerant-reader `covers_from_ts`
synthesis choice (either accept the age-underestimate explicitly with reasoning, matching the treatment every
other engineering default gets in §0, or pick a more conservative default) and add a test that runs a cascade
pass on migrated/tolerant-read output; (3) while in there, close F3 (state which function now owns the lock
across fold-then-rollover) and tighten C18 to name its actual protected content and pin the 1c-B fixture. None
of the four items requires touching the owner-pinned design (tiers, terminal, labels, caps, graduation
mechanism) — this is a narrow, mechanical revision, not a re-open of anything already settled.
