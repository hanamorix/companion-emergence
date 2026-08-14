# 3-redteam-plan-r9.md — Cascade Compaction, stage-3 plan red-team, ROUND 9

Cold, independent reviewer. No shared context with the author beyond the charter text handed to me for this
round. All citations below were verified by directly reading the named file/line in the worktree
`/home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction` at the time of this review (date
confirmed via `date`: **Thu Aug 13 04:05:44 PM EDT 2026**). Per the orchestrator's charter, the cascade +
redirect + migration MECHANISMS are treated as GIVEN/settled (cold-verified in rounds 1-8); this round's job is
to confirm round-8's narrow fixes (MO-3 doc fix, C12 sub-case, the one-time full-consistency sweep) and to
independently hunt for anything the sweep missed, plus re-verify all carried items are unregressed.

---

## Provenance

Worktree: `/home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction`
Branch: `ThinkerOfThoughts/cascade-compaction`

sha256 of every file read for this round:

```
724ccd170e827417d5385832b25332e3b302682ac79b71f45af697a47d54dc18  changes/cascade-compaction/1-spec.md
6a12be35e71558237412e8faad5956960c0117bc45ebe3fed42fdee50166a914  changes/cascade-compaction/1.5-criteria.md
c10bdf75f63829c152709f05a3153d925f2405300bbe14f84462cc631971b0bb  changes/cascade-compaction/2-plan.md
6bd5de55b2200be38b9ecd2d2563977447e1db7f171b223b3e9cdbb2118957b1  changes/cascade-compaction/3-redteam-plan-r8.md
09eeb7e9f92e91db20bc3c4d4b9384b8c4efef211ac8ec51bf0b72000cda4773  changes/cascade-compaction/decisions.md
cc9b6e22ac3cf05aa1109e84abb5d8217619153d85cf172ae1b84644b1aad7fb  brain/chat/compaction.py
7d32a50b29f47b34645bd48c04b6cae3bb62e2dd15e031712b2b752fd59e1d77  brain/chat/compaction_migration.py
cefb079963884fbafea3a0d8125c74bdc3a9e889894731329f482f49f93da56b  brain/ingest/buffer.py
773f1e0b0ae3dad2cbdc4f316e16e65194d09ab047665ff740259eace1a4dc34  brain/chat/session.py
f0b0b715746bc9f8e27964ec7c24301a14b6b80dab4d38db8f8b533d1df62d7e  brain/bridge/server.py
cfe8b63b3d642dabe998f52b52a087ccc4c0acbd8bab5b32bba177f0b309d331  brain/ingest/pipeline.py
ca6eeba8070959cf502e76177a3a635832ecc0377e12e43773b3cf9629c116b2  brain/bridge/supervisor.py
b71ed34d8f0d50108739fad682a3698989cfb5bd1278f0e4a3c7a011d873c7ca  brain/chat/budget.py
d359be520046a66c502ca5f0b56a0c61e8f4a13fbef93868e6fe127bed0d1260  brain/chat/engine.py
74dd5ba03614872c430fd2f3d2e40f23d4f70349a69816309ea0a4e7b8ee054e  brain/monologue/ambient.py
ddfbec5d20d3b4655afe7c14ce0c2d24564f09fa3e8f78f79d6560eb67051d90  brain/health/attempt_heal.py
```

**Independently confirmed:** every `brain/*.py` source file above has the identical sha256 to the value round 8
recorded (round 8's own provenance block lists the same hex for all ten files it hashed; I additionally hashed
`brain/health/attempt_heal.py`, not in round 8's list, and read it directly — see Lens 1 below). The source tree
has not changed one byte since round 6/7/8. Only `1-spec.md`, `1.5-criteria.md`, and `2-plan.md` changed since
round 8 (all three sha256 differ from round 8's recorded values), confirming round-8's revision landed.

Also read in full: `decisions.md` (all 8 prior gate-4/stage-3 rounds' dispositions, including the retired
redirect-class stop-condition, the migration-class stop-condition, and the orchestrator's override + revised
stop-conditions after round 8) and `3-redteam-plan-r8.md` (the immediately-prior round's full review, to know
exactly what MO-3/C12/the sweep were supposed to fix).

Additional verification performed beyond reading:
- `grep -n '"24h"|"48h"|"72h"|24h\b|48h\b|72h\b'` over all three artifacts, filtered for non-mechanical
  contexts — no stray bare-hour-label render language found (all matches are internal tier keys / spans /
  correctly-negated descriptions of the rejected mechanisms).
- Targeted greps for every stale term the charter listed: `graduates out`, `evict`, `newest edge`, `one hop`,
  `chain cap`, `four sites`, `else.*covers_until_ts`, `seed cursor prevents re-extraction`, `single lock hold`,
  `single continuous lock`, `falls off the bottom`, `archived out`, `conservative floor`, plus the three human
  labels. Every hit was read in context; all are either the correct current mechanism, an explicitly-negated
  description of a rejected proxy, or (for "seed cursor prevents re-extraction") a heading correctly explaining
  *why* that criterion no longer needs to exist (C18's own re-pointing rationale, unchanged since round 6).
- `grep -n 'sections={'` over `2-plan.md` — now exactly two hits (`:142`, `:418`), both `{"72h": ...}`,
  confirming the MO-3 contradiction is gone (round 8 had one `{"72h"}` and one stale `{"24h"}`).
- Full re-read of `2-plan.md` (all 516 lines) and `1-spec.md` (all 304 lines) end to end; targeted re-read of
  `1.5-criteria.md` C1–C22 + A1/A2 (305 lines) with fresh scrutiny on C12 (the criterion whose text changed)
  and the footer label-audit note.
- Direct source re-verification (spot, not exhaustive, since source is byte-identical to round 6-8's
  cold-verified state): `compaction.py:230-427` (summary-row shape `:364-374`, `.strip()` site `:338`,
  archive-before-rewrite `:376-394`, re-read-by-identity rewrite `:396-417`, lock acquire `:254-259`);
  `compaction_migration.py` (full file — `_DRAINED_REASONS` line 47, `_write_marker` `:69-78`,
  `run_backlog_migration` `:162-233`, confirms it still writes a summary row with no `sections` key);
  `buffer.py:158-166,260-320,338-392` (`delete_session_buffer`, `rewrite_session_atomic`, `append_archive`,
  `read_archive`, `acquire_compaction_lock` — confirmed **non-reentrant, per-call** via `O_CREAT|O_EXCL`,
  matching F3's claim exactly); `session.py:90-100,129-205,220-231` (`_LOCK = threading.RLock()`,
  `get_or_hydrate_session`'s current pre-build behavior, `remove_session` registry-only); `server.py` (grepped
  `get_or_hydrate_session(` — confirmed exactly 5 call sites at `:1261,2365,2424,2696,2753`; `in_flight_locks`
  decl `:816`, all raw-id usages `:1264,2368,2429,2699,2765,2836` in the pre-build state exactly as the plan
  describes needing fixing; backlog-migration thread `:913-935`); `pipeline.py` (speaker-filter drop before
  extraction, `finalize_stale_sessions` delete sites); `supervisor.py:635-662,1619-1646,1686`
  (`_run_compaction_tick`, persisted-cadence single-fire-per-gap, finalize `remove_session`); `budget.py:25-109`
  (`_COMPACTION_SUMMARY_PREFIX`, `apply_budget`'s `older_than=timedelta(0)` 24h-only backstop call, preserve-head
  re-parse); `engine.py:238-245,404-408` (head-prefix insert, `apply_budget` call site with the cited
  `max_tokens=80_000`/`preserve_tail_msgs=40`); `ambient.py:1-45` (`build_interior_continuity_block`,
  `MONOLOGUE_TRACE_TYPE` read); `attempt_heal.py:245-266` (the posix directory-fsync pattern §3 cites as
  reusable — confirmed real, confirmed try/except/finally-wrapped as described).

---

## Carried-forward resolution table (round 8 → round 9)

| ID | What round 8 demanded | Verdict this round | Evidence |
|---|---|---|---|
| **MO-3 fix (1): `plan.md:412-419` tolerant-read reference `{"24h"}`→`{"72h"}` old-floor** | Correct the stale §4 restatement to match §1.1's authoritative `{"72h"}` old-floor shape | **RESOLVED.** `plan.md:417-419` now reads `sections={"72h": {text: legacy text, covers_from_ts: now − _LEGACY_AGE_FLOOR (96h)}}`, matching §1.1's tier and `covers_from_ts` value exactly, and the passage explicitly self-annotates the fix ("this passage previously said `{"24h"}`, contradicting §1.1 §4 — the round-8 MO-3 fix"). Only two `sections={` literals remain in the whole document (`:142`, `:418`), both `"72h"`. **See new Nitpick N2 below for one residual field-level abbreviation** (not the tier-key contradiction MO-3 was about). | plan.md:136-148 (§1.1), 412-423 (§4, fixed) |
| **MO-3 fix (2): C12 sub-case — already-sections-migrated persona hit by a delayed `run_backlog_migration` retry → still tier 3, not tier 1** | Add a criterion sub-case exercising exactly this interaction | **RESOLVED.** `criteria.md:154-157` (C12 sub-case (c)): "a persona already sections-migrated, THEN hit by a delayed `run_backlog_migration` retry that flattens the row to single-layer → the tolerant reader + next cascade re-establish it as **tier 3, not tier 1**; run migration again → no change." This is the exact scenario `plan.md:412-423` (§4) describes, word-for-word the same framing ("delayed... retry"). | criteria.md:146-159 (C12, all 3 sub-cases); plan.md:412-423 |
| **MO-3 fix (3): one-time full internal-consistency sweep of spec+plan+criteria** | Sweep all three docs against every current authoritative decision; fix every stale leftover | **INDEPENDENTLY RE-SWEPT, CONFIRMED CLEAN for the items the charter named** (terminal tier3+no-evict, human labels, tier3 cap=0.20×tier1, old-floor 96h migration, full-chain redirect+5-site rebind+C21, oldest-edge classifier, multi-input terminal fold). Also independently confirmed the two other round-8-claimed sweep fixes: `spec.md` §7 no longer has a stale `ingest_turn`-follows-`rolled_to` line (redirect correctly attributed to `get_or_hydrate_session`/`session.py`, `:280-283`); `criteria.md`'s footer now says "C16→4 operation handlers" (`:300`), matching C16's own header (`criteria.md:196`, "the four OPERATION handlers"). **No new stale-term contradiction found** in this round's independent grep sweep — see Lens 2. **But see new Finding N1 (Minor)**: the sweep fixed the *documented* interaction (delayed retry) but did not add a one-line pin for the migration-vs-backlog-migration **execution order** within their shared startup thread, which is a distinct, untested ordering assumption in the same feature area. | Full re-read of all three docs; spec.md:280-283; criteria.md:196,300 |
| **L-1, F3, UA-2, owner-conformance/G1-C20/F4 group** | Should still hold | **STILL HOLD.** No text touched in the redirect-class sections (`plan.md:33-104` §0), the lock-continuity passage (`plan.md:322-336` §2.2), or C18's re-pointing (`criteria.md:262-273`) since round 8 — confirmed by direct re-read, byte-for-byte consistent with round 8's own quotes. Source-level re-verification (session.py's `RLock`, buffer.py's non-reentrant per-call lock primitive, server.py's 5 call sites, `remove_session` registry-only) all independently reconfirmed this round, not merely trusted from round 8's say-so. | plan.md:33-104,163-254,305-313,322-336; criteria.md:196-274; decisions.md:64-80 |

**Bottom line on carried items: all three of round-8's specific fixes are genuinely, verifiably in place, and
the carried mechanism findings (L-1/F3/UA-2/owner-conformance/G1-C20/F4) are unregressed.** This round's
independent sweep did not surface a fresh instance of the MO-3 contradiction pattern (two descriptions of one
mechanism, one stale) anywhere else in the document set. It did surface one new Minor (N1, an unstated
execution-order assumption in the same migration feature area, self-healing by the tolerant reader either way)
and one new Nitpick (N2, a field dropped from an abbreviated restatement).

---

## Lens 1 — Factual

**No Major/Blocker found.** Representative direct-source re-verification (full list in Provenance) confirms
every cited line number and code shape in the touched sections of spec/plan/criteria still matches the
byte-identical source tree.

**Finding N2 (Nitpick — factual, migration class).** `plan.md:142` (§1.1, authoritative) specifies the tolerant
reader's output as `sections={"72h": {text: <legacy text>, covers_until_ts: <existing>, covers_from_ts: <now −
`_LEGACY_AGE_FLOOR`>}}` — three fields. `plan.md:418` (§4, fixed this round) restates the **same** function's
output as `sections={"72h": {text: legacy text, covers_from_ts: now − _LEGACY_AGE_FLOOR (96h)}}` — **two**
fields; `covers_until_ts` is dropped. This is the identical *pattern* MO-3 was about (two descriptions of one
mechanism, not byte-identical) but not the identical *defect*: MO-3 was a **tier-key** contradiction (a
behavioral divergence — the classifier would key on the wrong data and misclassify). This is a **field-count**
abbreviation in prose — `covers_from_ts` (the field the classifier actually keys on, per §1.1's own sentence:
"the oldest-edge classifier... keys on the `covers_from_ts` **value**") is present and correct in both
places; only `covers_until_ts` (used by the render's coarse-span computation, §1.2, not by classification) is
elided in §4's restatement. §4 explicitly cross-references "(§1.1)" as the authoritative definition immediately
before giving the shape, so an implementer building the tolerant reader would build from §1.1, not reverse-
engineer it from §4's illustrative aside. **Severity: Nitpick** — no behavioral consequence, does not reproduce
#82 or any other defect, purely a terser restatement of a mechanism whose authoritative, complete definition sits
one section away and is explicitly pointed to. Does not warrant a stage-2 bounce; fix inline if convenient
(add `, covers_until_ts: <existing>` to `plan.md:418` for perfect textual mirroring) or accept as-is.

## Lens 2 — Logical

**No new logical contradiction found.** I specifically re-ran the same check round 8 used to find MO-3 — hunt
for two descriptions of one mechanism that diverge — across the full, freshly-swept document set (not just the
migration section) and found only N2 above (non-behavioral). The primary migration procedure (§4:395-410), the
tolerant reader (§1.1), spec §2g, and C12 are now mutually consistent on the tier key, the `covers_from_ts`
value, and the "unconditional, never `covers_until_ts`" discipline. The redirect chain's full-follow
description (`plan.md:54-60`), the 5-site table (`plan.md:63-69`), C16/C19/C20/C21, and spec §2e/§7 all agree on
"FIVE" call sites and the uniform-rebind rule — no drift from round 6-8's converged language.

**Finding N1 restated (the logical-consistency angle):** the document establishes, in two co-located passages
(`plan.md:406-407` and `:412-423`), that the new sections-migration and the pre-existing `run_backlog_migration`
share one startup thread, and separately establishes (via the tolerant reader, §1.1) that reading the two
migrations' outputs in *either* relative order is safe. But it never states which order they actually run in,
and the only interaction scenario given prose treatment (§4) is framed around a **delayed retry** of the *old*
migration (an already-completed-then-re-triggered case), not the **normal, first-time** case where an
implementer might simply call the new migration before the existing one in the shared thread body. Both
readings are logically consistent with everything else in the document — this is an omission, not a
contradiction, which is why it's filed under Lens 3/4 and the coverage challenge rather than as a second MO-3.

## Lens 3 — Missed opportunity

**Finding N1 (Minor, migration class, new this round).** The plan wires the new sections-migration "into the
same startup backlog-migration thread (`server.py:913-935`) so it runs before the first daily cascade tick"
(`plan.md:406-407`) but never pins the migration's execution order **relative to** `run_backlog_migration`
itself within that shared thread. I read `server.py:913-935` directly (confirmed this round): the existing
`_run_backlog_migration()` closure is the thread's sole target function, calling `run_backlog_migration(...)`
once; the plan's instruction to "wire in" the new migration most naturally means adding a second call inside
that same closure, but does not say before or after. Why this matters, independent of the tolerant reader's
safety net: if an implementer places the new sections-migration call **before** `run_backlog_migration(...)` in
that function body (a plausible ordering — "modernize the row shape, then drain backlog into it" is a coherent-
sounding rationale even though it is not what the design intends), then **every** persona with both a legacy
single-layer summary *and* undrained backlog would hit exactly the flatten-then-tolerant-read interaction §4
describes — on the very first, normal startup, not as a rare delayed-retry edge case. The **consequence is
still bounded and safe** (the tolerant reader defends it regardless of order, per §1.1's own "this makes old
rows behave correctly before migration runs (defensive)" — the mechanism is explicitly designed to be
order-agnostic), so this is not a data-loss or #82-reproduction risk on its own. But it means the "self-heal
over one cascade cycle" cost — currently documented and tested (C12(c)) as a rare, delayed-retry-only event —
could silently become the **normal** cost path for every migrated persona, with **no criterion distinguishing
the two orderings** (C12(c)'s fixture is explicitly a delayed-retry scenario, not a first-run reversed-order
one). This is the same risk *shape* as the round-4 carried CH-1 finding (an inferable-but-unpinned aggregation
rule) — "only one sane reading, worth a one-line pin" — not a design defect. **Fix (one line):** state the
intended order explicitly in `plan.md` §4 or the build-order note (§8 step 5), e.g. "the new migration must run
**after** `run_backlog_migration` completes within the shared thread, so it converts an already-backlog-drained
row" — and optionally extend C12 with a fixture that runs both migrations in a single pass in the documented
order to confirm no reliance on the tolerant reader's self-heal is needed in the common case.

**CH-1 (Minor, cascade class, carried from round 4, still unresolved) — unchanged this round.** The multi-input
tier-3 fold's `covers_from_ts` aggregation rule (`plan.md:208-209`, "records its `covers_from_ts`/
`covers_until_ts` from its inputs") is still not spelled out as a formula (`min()` across inputs, implied but
unstated). Text is byte-identical to round 8's; not touched by this round's revision. Still Minor per round 8's
calibration (C14 provides indirect general coverage; the term leaves only one sane reading).

No other missed-opportunity finding.

## Lens 4 — Unstated assumptions & risks

**Finding N1 restated (the assumption angle):** the plan implicitly assumes an implementer will infer the
correct migration ordering from context (you cannot meaningfully "section" a legacy summary until the backlog
that would otherwise still be sitting in raw turns has been folded into it) rather than from an explicit
statement. This is the same category of assumption as round 7/8's MO-2/MO-3 chain — reasoning that is likely
correct but was previously not written down until a reviewer forced it into text. Filed as Minor per Lens 3
above, not repeated here as a separate item.

**One assumption checked and confirmed sound (not a finding):** the daily cascade tick and the migration
threads run concurrently with no explicit ordering barrier between them (the backlog-migration thread is a
`daemon=True` thread started independently of the supervisor thread, `server.py:930-940`), which could in
principle let a cascade tick observe a not-yet-migrated (or half-migrated) row. I checked whether this is a real
race: both `cascade_conversation`/`compact_conversation` and the migration path acquire the **same per-session
compaction lock** (`buffer.py:341-392`, confirmed non-reentrant per-call this round), so writes serialize; and
`spec.md:192` explicitly states "the tolerant reader (plan §1.1) applies the same **before migration runs**" —
i.e., the design is explicitly stated to be safe for a cascade tick to observe a not-yet-migrated row, not just
an already-migrated-then-flattened one. This closes what would otherwise be an obvious concurrency question;
it is the same design property that also (partially) covers Finding N1's ordering concern, which is why N1 is
ranked Minor rather than Major.

No other new unstated-assumption finding.

## Lens 5 — Fidelity (owner mechanism vs proxy)

- **"Unconditionally" / "never `covers_until_ts`"** (`plan.md:143-145,400,419`, `spec.md:188-189`,
  `criteria.md:149`) — the term this finding-chain (MO-1→MO-2→MO-3→now) has pinned across four rounds. **Now
  conforms document-wide.** The one remaining place it could have failed to conform (§4's restated shape) now
  states the correct tier and the correct `covers_from_ts` computation; N2's field omission does not touch this
  term (`covers_until_ts`'s *presence or absence* in the restated shape is orthogonal to whether the mechanism
  *falls back to using it as the age source*, which is what "never `covers_until_ts`" actually pins — and it
  does not, in either §1.1 or §4's versions).
- **"Terminal"**, **"Multi-input"**, **"Full-follow"/visited-set guard**, **"Structural guard"**, **"Labels"**,
  **"Caps"**, **"Self-healing"** — unchanged since round 7/8; spot-re-read confirms no textual drift and no new
  fidelity gap. **"Self-healing"** in particular (`plan.md:420-421`, the term round 7 flagged as failing to
  conform against the *pre-fix* §4 text) now genuinely conforms: the passage it describes now actually
  reclassifies to the correct (old, safe) tier before the next cascade tick, which is what "heals" requires.
- No new loaded term identified this round that fails to conform to its underlying mechanism.

---

## Position lens (fires — ST1.5d)

Unchanged since round 6/7/8 (no text touched in the render/prefix sections this round — the revision was
confined to migration + the cross-doc sweep, none of which touched §1.2/`engine.py`/`budget.py` citations).
Re-confirmed by direct re-read this round: `plan.md:150-161` (§1.2) still authors the unchanged
`f"[Earlier in this conversation: {summary_text}]"` head prefix in lockstep with `budget.py`'s
`_COMPACTION_SUMMARY_PREFIX` re-parse (independently re-verified against `budget.py:28-31,94-109` and
`engine.py:404-408` this round — exact string match). No per-render nonce/live-timestamp introduced. **No issue
found.**

## Concurrency lens (fires — ST1.5e)

Unchanged since round 6/7/8 (§5's accessor table, `plan.md:425-442`, is textually identical to what round 8
verified; independently re-verified this round against `server.py`'s live `in_flight_locks`/`_SESSIONS` state
and `buffer.py`'s lock primitive). Finding N1 (migration-vs-backlog-migration ordering) is a **data-correctness/
documentation-completeness** gap, not a race or lost-update: both migrations serialize on the same per-session
compaction lock regardless of which runs first (confirmed this round, `buffer.py:338-392`), so there is no two-
writer collision — only an unpinned *order* between two lock-serialized, non-concurrent operations. Filed under
Lens 3/4, not here, consistent with how MO-2/MO-3 were filed in rounds 7-8. **No new concurrency-lens issue
found.**

---

## Coverage challenge (CH8)

Ranked by how directly each threatens shipped behavior for the target population, worst first:

1. **(Minor — new this round, restates Lens-3/4 Finding N1)** The **normal-order** (non-retry) interaction
   between the new sections-migration and the pre-existing `run_backlog_migration` — i.e., what happens if an
   implementer wires the new migration to run *before* `run_backlog_migration` in their shared startup thread
   — is not observed by any criterion. C12(c) tests only the **delayed-retry** framing (an already-migrated
   persona hit by a *later* backlog-drain retry). No criterion constructs "both migrations run once, in the
   wrong relative order, on the same fresh startup." Concrete scenario: any persona with both a legacy summary
   and undrained backlog, on a build where the new migration was wired in before the old one, hits the
   flatten-then-tolerant-read path on literally its first migration run — safe (self-heals via the tolerant
   reader within one cascade cycle, per §1.1's explicit "defensive... before migration runs" design) but
   untested as the *common* case rather than the documented rare one.
2. **(Minor — new this round, low materiality)** The migration's **optional** archive-scan primary path
   (`plan.md:403-405`, "if the archive scan yields a genuinely older ts, use it, else the 96h old-floor") is
   not exercised by C12's three sub-cases (all three assert the old-floor fallback path or its persistence/
   idempotency; none constructs an archive with a genuinely-derivable older ts and asserts that value is used
   instead of the floor). Low materiality: the plan itself labels this path "nice-to-have, NOT required"
   (`plan.md:402`), and the floor-only path is the one actually load-bearing for #82-safety.
3. **(Minor — restates round 7/8's carried CH8 item, unchanged)** Multi-generation chain interaction with the
   carried-cursor mechanism (C18): still specified against a single 1c-B rollover only
   (`criteria.md:262-273`); a second/third successive weekly rollover's carried-tail extraction-state transfer
   remains untested. Unchanged assessment from round 7/8 (low materiality — uniform per-rollover mechanism, a
   bug would likely also show on generation 1).
4. **(Minor — restates Lens-3 finding CH-1, unchanged)** The multi-input tier-3 fold's `covers_from_ts`
   aggregation rule (min-of-inputs, implied but never spelled out as a formula) remains exercised only
   generally by C14, not specifically for migrated content across multiple post-migration cascade cycles.
5. **(Nitpick, informational — restates round 6/7/8's item, still true, still explicitly owner-accepted)** The
   SYNC full-fold's wall-clock blocking cost (A1-a) remains ungated by any criterion; already flagged in the
   plan text itself as a deliberate, accepted cost.

No new coverage gap found in the weekly-rollover/successor-redirect class or the core cascade mechanics — those
areas were not touched by this round's revision and were independently re-derived (not merely trusted) via the
source spot-checks in Provenance, with no discrepancy found.

## Label audit (CH9/CH10)

Walked all 22 gating criteria + 2 advisory, focusing fresh scrutiny on **C12** (the one criterion whose text
changed since round 8) and independently spot-re-confirming a representative sample of the rest against source:

- **C12 — migration idempotent/#82-safe:** **Cleanly resolves round 8's MO-3 gap.** Sub-case (c)
  (`criteria.md:154-157`) now drives exactly the delayed-backlog-retry interaction §4 describes and asserts the
  tolerant-reader + next-cascade re-establishment lands at tier 3. **Does NOT cover the N1 scenario** (normal-
  order-reversed, first-run interaction) — see Coverage challenge item 1. This is the same "criterion exercises
  the documented case, not every case the surrounding prose's mechanism could produce" pattern this review
  chain has now caught four times (UA-2 round 6, MO-2's fallback-branch gap round 7, MO-3's interaction gap
  round 8, N1's ordering gap round 9) — each time in the migration/tolerant-reader feature area specifically,
  which is worth naming as a pattern for whoever builds this: the migration area keeps producing "one more
  untested interaction of the same safe mechanism," not new unsafe mechanisms.
- **C16, C19, C20, C21 (redirect class):** independently re-verified against source this round (not just
  re-read of the criteria text) — `server.py`'s 5 `get_or_hydrate_session(` call sites (`:1261,2365,2424,2696,
  2753`), all raw-id downstream usages at the cited lines, `session.py`'s `RLock`/`remove_session` — all match
  the criteria's claims about the pre-build state exactly. No regression.
- **C1–C11, C13–C15, C17, C18, C22:** no text changed since round 8; spot-re-read against fresh source citations
  (compaction.py's summary-row/lock/rewrite lines, buffer.py's archive/lock functions, budget.py/engine.py's
  head-prefix contract) found the same clean, real-path, fail-demo-equipped shape prior rounds documented. No
  regression found.
- **Advisory A1/A2:** unchanged, reasons still consistent with the project's stated measurement posture. No
  challenge.

---

## Bottom line

**All three of round-8's demanded fixes are genuinely in place and independently verified this round, not just
re-asserted:** (1) `plan.md:417-419`'s tolerant-read restatement now matches §1.1's authoritative `{"72h"}`
old-floor shape exactly (the MO-3 tier-key contradiction is gone); (2) `criteria.md`'s C12 gained sub-case (c),
which drives precisely the delayed-backlog-retry scenario §4 describes; (3) the one-time full-consistency sweep
holds up against an independent, fresh grep-and-read pass across all three documents — every stale term the
charter listed (`24h/48h/72h` bare labels, `evict`, `newest edge`, `one hop`/`chain cap`, `four sites`,
`else covers_until_ts`, `single lock hold`, `conservative floor`, `falls off the bottom`, etc.) was checked in
context and found either correctly-current or correctly-negated (a rejected-proxy description). L-1, F3, UA-2,
and the owner-conformance/G1-C20/F4 carried group are all confirmed unregressed against source, not merely
against round 8's prose.

**This round surfaced two new findings, both Minor/Nitpick, both in the migration feature area, both
explicitly self-healing by the same tolerant-reader mechanism that MO-1/MO-2/MO-3 already established:**
- **N1 (Minor):** the plan never pins the execution **order** of the new sections-migration relative to the
  pre-existing `run_backlog_migration` within their shared startup thread. Getting it "backwards" doesn't cause
  data loss or a #82 regression (the tolerant reader is explicitly designed to be order-agnostic, per §1.1's
  own "defensive... before migration runs" language, independently confirmed this round), but it would silently
  convert the documented "rare, delayed-retry-only" self-heal cost into the "every persona, every normal
  startup" common case, and no criterion (including C12(c), which is scoped to the delayed-retry framing only)
  distinguishes the two. Same risk shape as the already-accepted, carried CH-1 finding: an inferable-but-unpinned
  detail, one-line fix.
- **N2 (Nitpick):** `plan.md:418`'s restated tolerant-reader output shape drops the `covers_until_ts` field that
  §1.1's authoritative definition (`plan.md:142`) includes. Does not affect the classifier (which keys only on
  `covers_from_ts`, present and correct in both places) or reproduce any defect; §4 explicitly points back to
  "(§1.1)" as authoritative immediately before the abbreviated restatement.

**Worst finding this round: Minor (N1).** Per the REVISED stop-conditions the orchestrator recorded after round
8 (`decisions.md:169-171`: "Going forward HALT+report ONLY on a new Major that is a genuine MECHANISM/design/
correctness failure in any class; a pure doc/text-consistency finding AFTER this sweep is fixed inline, not
halted"), **neither N1 nor N2 triggers the migration-class stop-condition** — N1 is not a mechanism failure (the
tolerant reader already defends both orderings; this is a testing/documentation completeness gap on an
already-safe mechanism) and N2 is pure prose terseness with no behavioral consequence. No owner-pinned design
element needs re-opening, and no carried finding regressed.

**Routing: minors-only — clean enough to proceed to BUILD.** Recommended (not gating): add the one-line
ordering pin to `plan.md` §4/§8-step-5 (N1) and optionally extend C12 with a same-startup-normal-order fixture;
optionally mirror `covers_until_ts` into `plan.md:418` for perfect textual symmetry with §1.1 (N2). Neither
blocks the build; both are cheap to fold into the build commit itself rather than warranting another red-team
round.
