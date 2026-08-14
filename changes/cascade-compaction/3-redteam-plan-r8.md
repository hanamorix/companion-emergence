# 3-redteam-plan-r8.md — Cascade Compaction, stage-3 plan red-team, ROUND 8

Cold, independent reviewer. No shared context with the author beyond the charter text handed to me. All
citations below were verified by directly reading the named file/line in the worktree
`/home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction` at the time of this review
(date confirmed via `date`: **Thu Aug 13 03:48:07 PM EDT 2026**). The owner-pinned design (3 tiers/labels,
terminal tier-3, oldest-edge graduation, tier-3 hard cap = 0.20×tier1, cascade + redirect classes as
CLASS-CLOSED per round 7) is treated as GIVEN — I verified conformance and internal consistency, not merits.
The scope of this round, per the orchestrator's charter, is narrow: confirm round-7's carried finding MO-2
(migration old-floor fix) and re-verify all other carried items have not regressed.

---

## Provenance

Worktree: `/home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction`
Branch: `ThinkerOfThoughts/cascade-compaction` (HEAD `cd29bc61`)

sha256 of every file read for this round:

```
12c6f215e8bfdfd206596ac2947e5458cfeea3767c2896719042515b76c19b7f  changes/cascade-compaction/1-spec.md
66dd9eda0f94e3c443e82c5028a7a135639b4df645a8ad2a15603220078decec  changes/cascade-compaction/1.5-criteria.md
341b3dd514ad962f04cbfd58206c64a23a5fef349036afa80c099dbdae4c0a60  changes/cascade-compaction/2-plan.md
cc9b6e22ac3cf05aa1109e84abb5d8217619153d85cf172ae1b84644b1aad7fb  brain/chat/compaction.py
7d32a50b29f47b34645bd48c04b6cae3bb62e2dd15e031712b2b752fd59e1d77  brain/chat/compaction_migration.py
cefb079963884fbafea3a0d8125c74bdc3a9e889894731329f482f49f93da56b  brain/ingest/buffer.py
773f1e0b0ae3dad2cbdc4f316e16e65194d09ab047665ff740259eace1a4dc34  brain/chat/session.py
f0b0b715746bc9f8e27964ec7c24301a14b6b80dab4d38db8f8b533d1df62d7e  brain/bridge/server.py
ca6eeba8070959cf502e76177a3a635832ecc0377e12e43773b3cf9629c116b2  brain/bridge/supervisor.py
cfe8b63b3d642dabe998f52b52a087ccc4c0acbd8bab5b32bba177f0b309d331  brain/ingest/pipeline.py
d359be520046a66c502ca5f0b56a0c61e8f4a13fbef93868e6fe127bed0d1260  brain/chat/engine.py
b71ed34d8f0d50108739fad682a3698989cfb5bd1278f0e4a3c7a011d873c7ca  brain/chat/budget.py
74dd5ba03614872c430fd2f3d2e40f23d4f70349a69816309ea0a4e7b8ee054e  brain/monologue/ambient.py
```

**Independently confirmed:** every `brain/*.py` source file above has the identical sha256 to the value round 7
recorded — the source tree has not changed one byte since round 7 (or round 6). Only `1-spec.md`,
`1.5-criteria.md`, and `2-plan.md` changed (all three sha256 differ from round 7's recorded values, confirming
a real revision landed). Also read in full: `decisions.md` (all 7 prior gate-4/stage-3 rounds' dispositions,
including both HARD STOP-CONDITIONS — the retired redirect-class one and the still-live migration-class one)
and `3-redteam-plan-r7.md` (the immediately-prior round's full review, to know exactly what MO-2 demanded).

Additional verification performed beyond reading:
- `grep -n "covers_from_ts\|conservative floor\|else.*covers_until_ts"` over all three artifacts — traced
  every remaining reference to the migration/tolerant-reader default value, which is how Finding **MO-3**
  below was surfaced (a direct textual contradiction, not inferred).
- `grep -n 'sections={'` over `2-plan.md` — confirms exactly two literal `sections={...}` code-shape
  statements exist in the document, describing the SAME mechanism (the tolerant reader) with two different,
  incompatible tier keys (`"72h"` at line 142 vs `"24h"` at line 417).
- Full re-read of `brain/chat/compaction_migration.py` (234 lines, the unmodified existing backlog-migration
  pattern this phase's new sections-migration is supposed to reuse) and `brain/ingest/buffer.py:300-320`
  (`append_archive`, `read_archive` — confirms `read_archive` is still the pre-change whole-file reader,
  matching the plan's own description of the pre-build state).
- Full re-read of `brain/chat/compaction.py` (426 lines, in full) — confirms the summary-row shape
  (`:364-374`), archive-before-rewrite (`:376-394`), and re-read-by-identity rewrite (`:396-417`) citations
  used throughout spec/plan match the source exactly, and confirms the source has NO `sections`/
  `covers_from_ts` implementation yet (correctly describing the pre-build state a stage-3 plan review should
  see).

---

## Carried-forward resolution table (round 7 → round 8)

| ID | What round 7 demanded | Verdict this round | Evidence |
|---|---|---|---|
| **MO-2** (Major, migration — old-floor `covers_from_ts` must be unconditional, never `covers_until_ts`; archive-scan primary path must never read the buffer; C12 must exercise the fallback branch) | (i) old-floor mechanism structurally correct; (ii) written as a concrete value at migration time, tolerant reader yields an always-old value at read time, no inconsistency between the two; (iii) "never `covers_until_ts`" stated consistently across §1.1/§4/spec §2g/C12; (iv) 96h-floor interaction with a genuinely-recent legacy persona argued. | **RESOLVED for the mechanism itself — but see NEW Finding MO-3, a fresh internal contradiction in the SAME feature area, one level below where MO-2 lived.** (i) Verified structurally sound: `covers_from_ts = migration_now − 96h` gives `age = now − covers_from_ts ≥ 96h` forever (a persisted historical value only grows older with real time), and `bucket_of()`'s `else 72h` branch (`plan.md:186`) catches any age > 72h unconditionally — there is no path back to tier1/tier2 from this value. (ii) Confirmed: migration writes a **fixed, persisted** value once (`plan.md:399-400`); the tolerant reader (`plan.md:142-144`) instead **recomputes `now − _LEGACY_AGE_FLOOR` fresh on every invocation** (it is not persisted pre-migration) — mechanically different (fixed-and-aging vs. recomputed-and-constant-at-96h) but **outcome-consistent**: both always yield age ≥ 96h > 72h at the moment of use, so both always classify tier3. No inconsistency between the two **as specified in §1.1/spec §2g/plan §4's primary migration procedure (`plan.md:395-410`)**. (iii) The primary migration procedure and §1.1 are internally consistent and both say "unconditionally... never `covers_until_ts`" — **but §4's OWN secondary passage (the `run_backlog_migration`-interaction note, `plan.md:412-419`) contradicts §1.1 by describing the tolerant reader's output as `sections={"24h": legacy text}` instead of `sections={"72h": ...}`** — see MO-3. (iv) The 96h-floor-vs-genuinely-recent-legacy-persona tradeoff is sound and unchanged from round 7's accepted argument: a bounded, one-time, safe-direction (over-compression/mislabel-as-older, never under-compression/mislabel-as-newer) error that self-heals as new raw content accrues normally; I independently re-derived this and it holds — **also note the archive-scan optional path (`plan.md:403-405`, "if the archive scan yields a genuinely older ts, use it, else the 96h old-floor") is a `min()`-with-floor combination that can only ever push `covers_from_ts` OLDER, never younger than the floor — this structurally closes round 7's compounding concern #5 (a buggy "derivable" implementation that checks the buffer instead of the archive) even better than an explicit instruction alone would: a buffer-derived ts is essentially always <24h old, so it can never be "genuinely older" than the floor and therefore can never override it, regardless of which source an implementer mistakenly reads.** | plan.md:136-148 (§1.1), 395-410 (§4 primary), 412-419 (§4 interaction note — the contradiction); spec.md:182-194 (§2g); criteria.md:146-156 (C12); compaction.py (fresh full re-read, confirms pre-build state) |
| **L-1** (Major, redirect chain — full-follow, multi-generation) | Should still hold (redirect class CLOSED per round 7) | **STILL HOLD.** No text changed in the redirect-class sections (`plan.md:33-104` §0) since round 7 — confirmed by direct re-read; the full-follow + visited-set-guard mechanism, the uniform-rebind rule, and C16/C19/C20/C21 are textually identical to what round 7 verified clean. | plan.md:33-104; criteria.md C16/C19/C20/C21 (unchanged) |
| **F3** (minor, same-tick lock-continuity mechanism) | Should still hold | **STILL HOLD.** `plan.md:322-336` (§2.2) unchanged since round 7 — re-read-based correctness against the actual non-reentrant per-call lock primitive, not lock continuity. | plan.md:322-336 |
| **UA-2** (minor, C18 re-pointed to carried raw-tail) | Should still hold | **STILL HOLD.** `criteria.md:259-270` (C18) unchanged since round 7 — still framed around the 1c-B carried 40-msg raw tail, with the speaker-filter rationale intact. | criteria.md:259-270 |
| Prior carried (owner conformance: 3 tiers/labels/terminal tier3+cap/oldest-edge graduation/multi-input tier3 fold; G1/C20; F4 cursor-race scoping) | Should still hold | **STILL HOLD.** No text changed in `plan.md §1.3` (163-254, cascade mechanics), `plan.md:65,82-88` (G1/C20), or `plan.md:305-313` (§2.1, F4 extraction-concurrency scoping) since round 6/7. Cross-checked against the owner design rulings recorded in `decisions.md` (lines 64-80) — all five numbered rulings are still faithfully reflected in the current text. | plan.md:163-254, 65, 82-88, 305-313; decisions.md:64-80 |

**Bottom line on carried items: MO-2's own mechanism is genuinely, structurally resolved (all four
sub-questions the orchestrator posed check out). L-1, F3, UA-2, and the five-item "should still hold" group
are all confirmed unregressed. The catch is a NEW finding, MO-3, discovered while tracing MO-2's own
consistency — a stale, unswept passage elsewhere in the SAME document (§4) that still describes the
pre-fix tolerant-reader behavior and contradicts §1.1's authoritative, fixed statement.**

---

## Lens 1 — Factual

**Finding MO-3 lives here first: a direct, citable factual contradiction.** `plan.md:136-148` (§1.1) states,
as the load-bearing, explicitly-labelled fix ("stage-3 round-6 MO-1 / round-7 MO-2 — critical, structural"):

> "the tolerant reader reads a legacy row as `sections={"72h": {text: <legacy text>, covers_until_ts:
> <existing>, covers_from_ts: <now − `_LEGACY_AGE_FLOOR`>}}`"

`plan.md:412-419` (§4, "Interaction with the unchanged `run_backlog_migration`" — dated "stage-3 round-2
minor", i.e. written in round 2, well before the MO-1/MO-2 fix existed) states, describing the **same named
mechanism** ("the tolerant reader (§1.1)"):

> "the **tolerant reader (§1.1)** reads a section-less row as `sections={"24h": legacy text}`"

These are two different claims about the output of one function for the same input shape (a section-less
row). One of them is wrong. Given §1.1's version is the one explicitly labelled as the round-6/7 structural
fix, cited by number, and consistent with spec.md §2g/C12, I take §4's "24h" version to be the stale one —
but the document as written does not say so; it simply asserts both, in two places, about the same mechanism.
An implementer who reads §4's interaction note in isolation (a plausible thing to do — it is the one paragraph
that specifically addresses a delayed `run_backlog_migration` retry hitting an already-sectioned persona) would
build the tolerant reader to default a section-less row to **tier 1 ("yesterday")**, not tier 3, for exactly
the scenario that paragraph describes — reproducing #82 for that scenario. **No `covers_from_ts` value is
even given in §4's rendering** (`sections={"24h": legacy text}` has no ts field at all), so it also fails to
carry forward the "unconditionally... never `covers_until_ts`" discipline C12/§1.1/spec §2g established — it
simply omits the field that discipline is about.

The scenario §4 itself describes is not far-fetched: it requires (a) the new sections-migration to have
already converted a persona's summary row to sectioned form (its own marker written), and (b) the pre-existing,
unmodified `run_backlog_migration`'s *separate* marker (`.compat_migrated`) to have been withheld on an
earlier, unrelated startup due to a transient `locked`/`archive_failed` miss (`compaction_migration.py:44-47`,
confirmed this round by direct re-read — `_DRAINED_REASONS = frozenset({"nothing_aged", "cursor_none"})`, so
anything else, e.g. `locked`/`archive_failed`, correctly withholds the marker and forces a retry on the next
restart per the module's own docstring). Because these are two independent migrations gated by two independent
markers, sharing only the startup thread they're both wired into (`plan.md:407` cites `server.py:913-935` for
the new migration; the old one already lives there), there is nothing that prevents (a) and (b) from both
being true for the same persona on the same restart. This is a real, reachable interaction, not a
manufactured edge case — which is exactly why §4 exists to address it in the first place; it just resolves the
address incorrectly.

Beyond MO-3, representative direct-source re-verification (source tree unchanged since round 6/7, so a
sampled re-check suffices rather than a full re-derivation): `compaction.py:230-427` (full file — summary-row
shape at `:364-374`, archive-before-rewrite at `:382-394`, re-read-by-identity at `:396-417`, all match cited
line numbers exactly); `compaction_migration.py` (full file — confirms `_write_marker` tmp+fsync+replace
pattern, `_DRAINED_REASONS`, and that `run_backlog_migration` writes rows with no `sections` key, matching the
plan's own claim at `plan.md:398-399`); `buffer.py:300-320` (`append_archive`/`read_archive` — confirms
`read_archive` is still the pre-change whole-file reader, matching spec.md §2f's description of the current
state that 1d is meant to extend).

**Severity: Major** (Factual lens). Same reasoning as MO-1/MO-2's Major ranking — a stale/incorrect default
for the tolerant reader, in the exact scenario the surrounding paragraph exists to cover, reproduces the
phase's own target defect (#82) if built as literally written.

## Lens 2 — Logical

**Finding MO-3 restated (the logical-consistency angle on the same gap):** the document asserts two mutually
exclusive behaviors for one function (the tolerant reader, defined once in §1.1) without reconciling them or
even acknowledging the second exists as a special case. This is structurally the same failure pattern round 6
caught and closed for the redirect chain ("one hop" vs. "cap N", both describing `get_or_hydrate_session`) —
**a mechanism gets a fixed, load-bearing definition in one place, and an older, unrevised restatement survives
elsewhere, un-swept.** I specifically looked for other instances of this pattern (two descriptions of one
mechanism) across the revised sections and found none beyond MO-3 — the primary migration procedure (§4:395-
410), spec §2g, and C12 are all mutually consistent with §1.1's authoritative statement; only the §2-vintage
interaction-note paragraph was missed.

No other logical issue found. The multi-input tier-3 fold description (unchanged since round 5/6) remains
internally consistent on this re-read; the same-tick lock-continuity fix (F3, unchanged since round 6)
remains internally consistent.

## Lens 3 — Missed opportunity

**Finding CH-1 (Minor, cascade class, carried from round 4, still unresolved) — the multi-input tier-3 fold's
`covers_from_ts` aggregation rule is still not spelled out as a formula.** `plan.md:208-209` says merely "Each
new section records its `covers_from_ts`/`covers_until_ts` from its inputs" — round 4 flagged this exact gap
("the exact aggregation rule (min across all inputs' `covers_from_ts`, presumably) is never spelled out...
Low-risk (min() is the only sane choice) but worth a one-line pin"). It was never fixed in rounds 5-7 (the
revisions in those rounds targeted other findings) and remains unfixed this round. I re-derived why this
matters specifically for MO-2/MO-3's topic: the migrated tier3 section's persistence across MANY cascade
cycles (not just the one pass C12(b) tests) depends on the merged section's `covers_from_ts` staying pinned to
the oldest input's value (i.e. `min()`) every time it's re-folded with a newly-graduated tier2 cohort — if an
implementer instead computed some other aggregate (e.g., the newest input, or an average), the old-floor value
would drift younger over repeated cycles and could eventually cross back under 72h, silently re-introducing
#82 several cycles after migration rather than on the very next one. This is **not untested** in the general
case — C14's oracle ("assert each marker... PERSISTS in tier3 on passes 4, 5, …") exercises the identical
code path for ordinary (non-migrated) content and would catch a broken aggregation rule generally — but C14's
fixture is fresh-sown turns, not migrated legacy content, so it does not specifically confirm the migrated
floor value survives multiple post-migration cascade cycles under the SAME code path. Still ranked Minor,
not Major: the term "OLDEST covered ts" (used to define `covers_from_ts` at `plan.md:185`) leaves only one
sane reading (`min()`), and C14 provides indirect coverage of the general mechanism.

No other missed-opportunity finding beyond MO-3 (filed under Factual/Logical above, per this review chain's
established convention of filing the load-bearing instance under whichever lens surfaces it first) and CH-1.

## Lens 4 — Unstated assumptions & risks

**No new finding.** I looked specifically for a fresh instance of "asserted outcome that doesn't reconcile
with the actual mechanism" (the pattern MO-2/MO-3 and the old F3/UA-1 carried findings share) elsewhere in the
revised text and found none beyond what's captured above. One assumption worth naming explicitly (not a
finding, since I verified it holds): the migration's optional archive-scan primary path (`plan.md:403-405`)
implicitly assumes `read_archive` is already the segment-aware, full-provenance-chain version by the time
migration runs — this is true by construction, since build order (`plan.md:487-496`, §8) places 1d archive
segmentation (step 4) before migration (step 5), so an archive scan invoked from the migration code is
guaranteed to see the full chain, not just the newest segment. This closes what would otherwise be a residual
piece of round-7's MO-2 finding-item-5 concern; I checked it because it was exactly the kind of
sequencing assumption that could silently fail, and it does not.

## Lens 5 — Fidelity (owner mechanism vs proxy)

- **"Unconditionally" / "never `covers_until_ts`"** (`plan.md:143-145,400`, `spec.md:188-189`,
  `criteria.md:149`) — the loaded term this whole finding-chain (MO-1→MO-2→now) has been pinning. It
  **conforms** in the primary migration procedure, §1.1's own authoritative statement, spec §2g, and C12 — but
  it does **not** conform document-wide, because §4's secondary interaction-note (line 412-419) describes an
  output for the identical mechanism that has no `covers_from_ts` value at all (let alone an unconditional
  old-floor one) and uses the pre-fix tier key. A term pinned as "unconditional" that has an un-swept exception
  elsewhere in the same document is not actually unconditional as specified — this is the fidelity angle on
  MO-3, filed here for completeness per this chain's established practice of cross-referencing the same gap
  under multiple lenses when it fits more than one.
- **"Terminal"**, **"Multi-input"**, **"Full-follow"/visited-set guard**, **"Structural guard"**, **"Labels"**,
  **"Caps"** — unchanged since round 7; spot-re-read confirms no textual drift. **Conform.**
- **"Self-healing"** (`plan.md:417-418`, describing the outcome of the very passage MO-3 flags) — this term
  does **not** conform to the mechanism it's applied to, if that mechanism is read as literally described in
  the same sentence: a tolerant-reader default that mislabels old content as tier1 does not "heal" on the next
  cascade tick in the safe direction — it does the opposite of what the rest of the document calls healing
  (re-establishing the CORRECT sectioned form), because the reclassification would be based on a wrong-tier
  seed. This is the same fidelity failure pattern round 7 caught in "conservative floor" (a reassuring word
  applied to a mechanism that, on inspection, doesn't earn it).

---

## Position lens (fires — ST1.5d)

Unchanged since round 6/7 (no text touched in the render/prefix sections this round — the revision was
confined to migration). Re-confirmed by direct re-read: `plan.md:150-161` (§1.2) still authors the unchanged
`f"[Earlier in this conversation: {summary_text}]"` head prefix in lockstep with `budget.py`'s
`_COMPACTION_SUMMARY_PREFIX` re-parse; no per-render nonce/live-timestamp introduced. **No issue found.**

## Concurrency lens (fires — ST1.5e)

Unchanged since round 6/7 (§5's accessor table, `plan.md:421-438`, is textually identical to what round 7
verified). MO-3 is a data-correctness gap (wrong tier key / missing ts fed to a classifier on a specific
migration-interaction path), not a race or lost-update — no two writers contend over it, so it is filed under
Factual/Logical, not here, consistent with how MO-2 itself was filed in round 7. **No new concurrency-lens
issue found.**

---

## Coverage challenge (CH8)

Ranked by how directly each threatens shipped behavior for the target population, worst first:

1. **(Major — restates Factual/Logical Finding MO-3)** The `run_backlog_migration`-interaction scenario
   (`plan.md:412-419`) is **not observed by any criterion.** C12 (`criteria.md:146-156`) tests the
   sections-migration function's own idempotency and its fallback branch's persistence across one cascade
   pass — it does not construct the "sectioned row gets flattened back to single-layer by a delayed legacy
   backlog drain, then re-read by the tolerant reader" scenario that §4's own paragraph exists to describe. No
   other criterion touches `run_backlog_migration` at all. Concrete scenario: any persona whose
   `.compat_migrated` marker was withheld on an earlier startup (transient lock/archive-write failure — a
   real, already-handled-elsewhere failure mode, `compaction_migration.py:44-47`) and which has since been
   converted to sectioned form by the new migration, has its history flattened-then-misclassified on the next
   startup's backlog-drain retry, reproducing #82 for that persona.
2. **(Minor — restates round 7's carried CH8 item 2)** Multi-generation chain interaction with the
   carried-cursor mechanism (C18): still specified against a single 1c-B rollover only
   (`criteria.md:259-270`); whether the carried-tail extraction-state-transfer logic is correctly re-exercised
   on a second or third successive weekly rollover remains untested. Low materiality (unchanged assessment
   from round 7 — the mechanism is stated to be uniform per-rollover, so a bug would likely also show on
   generation 1).
3. **(Minor — restates Lens-3 finding CH-1)** The multi-input tier-3 fold's `covers_from_ts` aggregation rule
   (min-of-inputs, implied but never spelled out as a formula) is exercised generally by C14 but not
   specifically for migrated content across multiple post-migration cascade cycles.
4. **(Nitpick, informational — restates round 6/7's item, still true, still explicitly owner-accepted)** The
   SYNC full-fold's wall-clock blocking cost (A1-a) remains ungated by any criterion; already flagged in the
   plan text itself as a deliberate, accepted cost.

No new coverage gap found in the weekly-rollover/successor-redirect class or the core cascade mechanics beyond
what round 7 already documented — those areas were not touched by this round's revision and were not
re-derived from scratch this round (source-identical, text-identical to round 7's clean audit).

## Label audit (CH9/CH10)

Walked all 22 gating criteria + 2 advisory, focusing fresh scrutiny on C12 (the one criterion whose text
changed since round 7) and spot-re-confirming the rest did not regress:

- **C12 — migration idempotent/#82-safe:** **Cleanly resolves round 7's specific complaint** (the fixture now
  explicitly names sub-case (b), "a legacy blob with no derivable oldest ts," forcing the fallback branch to
  fire and asserting it survives the next cascade pass at tier 3 — this is a real, discriminating test for the
  scenario MO-2 was about). **However, C12 does NOT cover the MO-3 scenario** (the `run_backlog_migration`
  interaction) — that is a distinct code path (a flattening event followed by a tolerant-reader read) that no
  sub-case of C12, or any other criterion, drives. This is the same "criterion exercises the documented case,
  not every case the surrounding prose claims to handle" pattern this review chain has now caught three times
  (UA-2 in round 6, the MO-2 fallback-branch gap in round 7, and this MO-3 interaction gap in round 8) — each
  time in the migration/tolerant-reader feature area specifically.
- **C1–C11, C13–C22:** no text changed since round 7; spot-re-read a sample (C6's head-prefix oracle via
  fresh `engine.py`/`budget.py`-referencing text, C21's structural-guard fail-demo language) and found the
  same clean, real-path, fail-demo-equipped shape round 7 documented in full for all of these. No regression
  found.
- **Advisory A1/A2:** unchanged, reasons still consistent with the project's stated measurement posture. No
  challenge.

---

## Bottom line

Round 7's carried finding, **MO-2, is genuinely resolved on its own terms** — I independently re-derived all
four sub-questions the orchestrator posed (structural soundness of the old-floor value, write-time vs.
read-time consistency, the "never `covers_until_ts`" discipline, and the genuinely-recent-legacy-persona
tradeoff) and each checks out against the primary specification (§1.1, spec §2g, plan §4's main migration
procedure, and C12). L-1, F3, UA-2, and the owner-conformance/G1-C20/F4 carried items are all confirmed
unregressed (source-identical, text-identical to round 7 where not touched).

**But tracing MO-2's consistency surfaced a new, adjacent defect in the same document: Finding MO-3.** A
secondary paragraph in the same section (`plan.md:412-419`, the `run_backlog_migration`-interaction note,
textually dated to round 2 — written well before the MO-1/MO-2 fix existed) still describes the tolerant
reader defaulting a section-less row to **tier 1 ("24h")**, directly contradicting §1.1's authoritative,
explicitly-labelled fixed statement that the SAME mechanism defaults to **tier 3 ("72h")**. This paragraph was
never swept when the tolerant reader's default changed in rounds 6-7. If built as literally written, it
reproduces #82 for a real (not manufactured) interaction scenario the paragraph itself exists to describe, and
that scenario is untested by C12 or any other criterion (Coverage challenge item 1 / Label audit).

**Worst finding this round: Major (MO-3).** It is squarely in the **migration** class — the same class as
MO-1 and MO-2, concerning the same tolerant-reader mechanism, one paragraph away from where the round-7 fix
landed. **This explicitly triggers the round-7 owner HARD STOP-CONDITION** recorded in `decisions.md`
(line 147-149: *"if round 8 surfaces ANOTHER new Major in the migration/tolerant-reader class, HALT + report
(signals the migration approach needs a rethink or scope carve-out)"*) — flagging for the orchestrator to
apply that condition rather than routing silently back to stage 2.

That said, the concrete fix is narrow and mechanical, not a rethink of the design: (1) correct `plan.md:417`'s
`sections={"24h": legacy text}` to `sections={"72h": {text: legacy text, covers_from_ts: <now −
_LEGACY_AGE_FLOOR>, covers_until_ts: <existing>}}`, matching §1.1's authoritative definition exactly (the
mechanism does not need to differ for this interaction case — it's the same tolerant reader, same input
shape); (2) add a C12 sub-case (or a new criterion) that drives the flatten-then-tolerant-read interaction
scenario end-to-end and asserts it, too, survives at tier 3. No owner-pinned design element (tiers, terminal,
labels, caps, oldest-edge graduation, redirect chain, multi-input fold) needs re-opening.

**Routing: back to stage 2 (narrow) — not clean, not minors-only — AND report the stop-condition trigger to
the orchestrator per the owner's standing instruction, rather than treating this as a normal bounce.**
