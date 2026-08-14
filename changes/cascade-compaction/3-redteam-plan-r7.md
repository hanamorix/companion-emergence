# 3-redteam-plan-r7.md — Cascade Compaction, stage-3 plan red-team, ROUND 7

Cold, independent reviewer. No shared context with the author beyond the charter text handed to me. All
citations below were verified by directly reading the named file/line in the worktree
`/home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction` at the time of this review
(date confirmed via `date`: **Thu Aug 13 03:36:59 PM EDT 2026**). The owner-pinned design (3 tiers/labels,
terminal tier-3, oldest-edge graduation, tier-3 hard cap = 0.20×tier1) is treated as GIVEN — I verified
conformance, not merits.

---

## Provenance

Worktree: `/home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction`
Branch: `ThinkerOfThoughts/cascade-compaction` (HEAD `cd29bc61`)

sha256 of every file read for this round:

```
b405ed97c0ead91addfc21cd806ac7ed8e8d1785ccf813ec6fec00401187af8c  changes/cascade-compaction/1-spec.md
59b5e538e17e70a7d769c9814a621d4766060ba6498473ba2fd5040536ecb8f5  changes/cascade-compaction/1.5-criteria.md
fec1b114277b8782aaa43debd548d1f8f6a886145ef612548a0791341ec9adee  changes/cascade-compaction/2-plan.md
773f1e0b0ae3dad2cbdc4f316e16e65194d09ab047665ff740259eace1a4dc34  brain/chat/session.py
f0b0b715746bc9f8e27964ec7c24301a14b6b80dab4d38db8f8b533d1df62d7e  brain/bridge/server.py
cc9b6e22ac3cf05aa1109e84abb5d8217619153d85cf172ae1b84644b1aad7fb  brain/chat/compaction.py
7d32a50b29f47b34645bd48c04b6cae3bb62e2dd15e031712b2b752fd59e1d77  brain/chat/compaction_migration.py
cfe8b63b3d642dabe998f52b52a087ccc4c0acbd8bab5b32bba177f0b309d331  brain/ingest/pipeline.py
ca6eeba8070959cf502e76177a3a635832ecc0377e12e43773b3cf9629c116b2  brain/bridge/supervisor.py
cefb079963884fbafea3a0d8125c74bdc3a9e889894731329f482f49f93da56b  brain/ingest/buffer.py
d359be520046a66c502ca5f0b56a0c61e8f4a13fbef93868e6fe127bed0d1260  brain/chat/engine.py
b71ed34d8f0d50108739fad682a3698989cfb5bd1278f0e4a3c7a011d873c7ca  brain/chat/budget.py
74dd5ba03614872c430fd2f3d2e40f23d4f70349a69816309ea0a4e7b8ee054e  brain/monologue/ambient.py
ddfbec5d20d3b4655afe7c14ce0c2d24564f09fa3e8f78f79d6560eb67051d90  brain/health/attempt_heal.py
```

Also read (context, not cited with sha256): `changes/cascade-compaction/decisions.md` (full gate log, all 6
prior rounds' dispositions, including the owner's round-6 HARD STOP-CONDITION for the weekly-rollover/
successor-redirect class only).

**Important provenance fact, independently checked, not just inherited:** every one of the 11 `brain/*.py`
source files above has the **identical sha256** to the value recorded in round 6's review
(`3-redteam-plan-r6.md`). The source tree has not changed one byte since round 6 — only `1-spec.md`,
`1.5-criteria.md`, and `2-plan.md` changed. This means round 6's exhaustive line-locus factual verification
(>40 individually checked citations against source) is still valid evidence for any claim that did not change
text between round 6 and round 7; I re-verified a representative sample directly myself (see Lens 1) rather
than re-deriving the entire citation set from scratch, and I independently re-ran the round-6 F2 grep to
confirm the caller-count claim still holds.

Additional verification performed beyond reading:
- `grep -rn "get_or_hydrate_session(" --include=*.py .` over the whole worktree (production hits only:
  `server.py:1261,2365,2424,2696,2753`, `session.py:129` the definition — exactly 5 callers, no 6th).
- `grep -n -i "one hop\|one-hop\|single hop\|cap the chain\|small N\|depth cap"` over all three artifacts —
  confirms the round-6 L-1 "one hop / cap N" contradiction is fully purged (only contrastive phrasing
  "not a single hop" / "not a depth cap" remains).
- `grep -n "derivable\|covers_from_ts"` over all three artifacts — traced every use of the migration/
  tolerant-reader fallback value (this is what surfaces Finding MO-2 below).
- Direct reads of `session.py:85-234` (`get_or_hydrate_session`, `_LOCK`, `remove_session` — all unchanged
  from round 6, confirmed byte-identical and re-read fresh), `compaction_migration.py` (full file, 234 lines
  — the *existing*, unmodified backlog-migration pattern this change's new sections-migration is supposed to
  reuse), `buffer.py:255-393` (`rewrite_session_atomic`, `append_archive`, `read_archive`,
  `acquire_compaction_lock`/`release_compaction_lock` — confirms the lock is a plain non-reentrant
  `O_CREAT|O_EXCL` pid-file, acquired/released self-contained per call, matching plan.md's F3-corrected claim),
  `compaction.py:230-424` (`compact_conversation` in full — lock acquire at `:254`, cursor guard, summary-row
  shape at `:364-374`, archive-before-rewrite at `:376-394`, re-read-by-identity at `:396-417` — all match
  cited line numbers exactly), `server.py:1255-1274` and `server.py:2820-2844` (the `/state` in-flight lookup
  and the `/sessions/close` cleanup — confirmed both still use the raw, unresolved id at exactly the cited
  line numbers `:1264`, `:2835`, `:2836`, i.e., the pre-fix tree the plan describes and C21's fail-demo
  depends on is accurately described).

---

## Carried-forward resolution table (round 6 → round 7)

| ID | What round 6 demanded | Verdict this round | Evidence |
|---|---|---|---|
| **MO-1** (Major, migration — legacy blob must default to tier 3 not tier 1) | Fix the tolerant reader + migration to seed a legacy blob as tier 3 (derive `covers_from_ts` from oldest turn ts if available), and add a C12 cascade-on-migrated-output #82-guard. Also: argue whether tier-3-default is acceptable even for a legacy persona whose history really IS recent. | **PARTIALLY RESOLVED — see new Finding MO-2 below.** The *placement* fix is real and correctly specified: `1-spec.md:182-191` (§2g) and `2-plan.md:136-145` (§1.1 tolerant reader) + `2-plan.md:392-403` (§4 migration) now unconditionally seed the legacy blob into `sections["72h"]`, not `sections["24h"]`, and `1.5-criteria.md` C12 (146-155) adds the "run cascade on migrated output → assert stays tier3" oracle exactly as demanded. However, the **fallback branch** of the very same fix (`covers_from_ts := covers_until_ts` when the oldest turn ts is not derivable) still carries the *original* #82-reproducing defect, and the plan's own claim that this fallback is a safe "conservative floor" is demonstrably false against the plan's own cascade reclassification mechanism (§1.3 step 3). This is not a new category of bug — it's the identical MO-1 mechanism, now scoped to the fallback branch instead of the primary branch, and it is untested by C12 as currently specified. Ranked as its own finding (MO-2) rather than "MO-1 unresolved" because the *primary* path (derivable case) genuinely is fixed. The requested "argue whether tier3-default is ever wrong for a genuinely-recent legacy persona" question was not explicitly re-litigated in the artifacts, but is low-stakes (a genuinely-recent legacy persona would just fade one cycle slower than optimal — a conservative-direction error, unlike the topic of MO-2). | spec.md:182-191; plan.md:136-145,392-403; criteria.md:146-155 (C12); see Lens 3 finding MO-2 |
| **L-1** (Major, redirect chain — full-follow with visited-set guard, multi-generation test) | Plan §0 + spec §2e-B/§7 + C16 must specify full-follow to the live successor (visited-set cycle guard, not a depth cap) with a multi-generation test (sid1→sid2→sid3). Verify consistent everywhere (no residual "one hop"/"cap N"). | **RESOLVED.** `2-plan.md:52-59` ("follows the pointer chain to its END... full-follow — stage-3 round-6 L-1, not a single hop... visited-set cycle check (NOT an arbitrary small depth cap — a legitimately long chain must resolve; only a true cycle aborts...)"), reaffirmed at `plan.md:95` and `spec.md:278` ("visited-set cycle guard, not an arbitrary depth cap"). `1.5-criteria.md` C16 (192-211) adds the explicit multi-generation oracle: "perform **three** successive rollovers sid1→sid2→sid3; a client still holding **sid1** → `get_or_hydrate_session(sid1)` resolves to **sid3**... A **cyclic** pointer (corrupt) → aborts to 404 (visited-set guard), not a hang." Independently grepped all three artifacts for "one hop"/"cap the chain"/"small N"/"depth cap" — zero residual contradictory instances; every remaining hit is contrastive language explicitly ruling the old phrasing out. Genuinely closed, not just re-worded. | plan.md:52-59,95; spec.md:278; criteria.md:192-211 (C16); grep (zero residual contradictions) |
| **F3** (minor, same-tick lock-continuity mechanism must be stated, not asserted) | State which function now owns the lock across fold-then-rollover, reconciling with the actual non-reentrant per-call lock primitive. | **RESOLVED.** `2-plan.md:322-333` ("The real lock mechanism (corrected per stage-3 round-6 F3 — NOT a single continuous hold)... `cascade_conversation` and the rollover each **acquire it internally and release at the end of their own call**... two separate acquisitions, and `apply_budget` is a **confirmed real concurrent caller**... Correctness does **not** rest on lock continuity; it rests on **re-read**: the rollover, after acquiring the lock, **re-reads the current committed summary row** and seeds from it.") This is coherent with the actual primitive: I directly re-read `buffer.py:341-392` this round (`acquire_compaction_lock`/`release_compaction_lock`, a plain `O_CREAT\|O_EXCL` pid-file, no handle exposed across calls, reaped only via dead-pid/stale-mtime) and `compaction.py:254`/`:425-426` (acquire/release self-contained inside `compact_conversation`) — the plan's corrected claim now matches the code exactly, and the fix (re-read, not lock-continuity) is a real, sufficient correctness argument given the confirmed re-read-by-identity discipline already used elsewhere (`compaction.py:396-417`). | plan.md:322-333; buffer.py:341-392 (fresh read); compaction.py:254,396-417,425-426 |
| **UA-2** (minor, C18 re-pointed from the seed summary row to the carried 40-msg raw tail) | Re-point C18 to the real risk (the 1c-B carried raw tail), since the seed summary row is unconditionally un-extractable via the speaker-filter. | **RESOLVED.** `1.5-criteria.md` C18 (258-269) is now explicitly framed around "The seed **summary** row cannot be re-extracted regardless of cursor — `pipeline.py`'s speaker-filter blocks a `speaker=="summary"` row unconditionally... The **real** cursor-dependent risk is **1c-B's carried 40-message RAW tail**." I independently re-read `pipeline.py:317-324` this round: `cursor = read_cursor(...)`, `turns = read_session_after(...)`, then the very next line `turns = [t for t in turns if t.get("speaker") != "summary"]` — confirms the speaker-filter claim byte-for-byte (it drops every `speaker=="summary"` row unconditionally, independent of cursor state, exactly as C18's rationale now states). The oracle and fail-demo (criteria.md:265-269) are now pinned to the 1c-B/40-tail fixture, not the vacuously-passing 1c-A shape. | criteria.md:258-269 (C18); pipeline.py:317-324 (fresh read, confirms speaker-filter) |
| Prior carried (F1 multi-input tier3, F2 five-site redirect + C21, owner conformance, G1/C20, tier caps, labels) | Should still hold | **STILL HOLD.** No text changed in the artifacts for any of these since round 6 (confirmed by diffing round-6's cited line ranges against this round's read); source is byte-identical to round 6. Re-spot-checked the F2/C21 "exactly 5 callers, no 6th" claim independently this round (grep above) — still true. | grep (5 callers); no artifact text changes in the relevant sections |

**Bottom line on carried items: 3 of 4 fully resolved (L-1, F3, UA-2). MO-1 is resolved for its primary
(derivable) path but its own fallback branch reproduces the identical defect it was meant to fix — filed as a
new finding, MO-2, rather than "still open," since the mechanism demanded by round 6 was genuinely built; it
just has an unaddressed edge in the exact same family.**

---

## Lens 1 — Factual

Representative direct-source re-verification performed this round (not a full re-derivation of all >40
round-6 citations, since the source tree is byte-identical to round 6 — see Provenance): `session.py:97`
(`_LOCK = threading.RLock()`), `:129-205` (`get_or_hydrate_session`, current pre-change behavior: registry
lookup then buffer hydration, returns `None` on both misses — no `rolled_to.json` consultation yet, correctly
describing the *pre-build* state the plan proposes to change), `:222-231` (`remove_session`, registry-pop
only, no file I/O — matches the plan's "already registry-only, no build work" claim exactly);
`server.py:1261,1264-1265` (`/state`, raw `session_id` in-flight lookup), `:2835-2836` (`/sessions/close`
cleanup, raw `req.session_id`); `buffer.py:265-393` (`rewrite_session_atomic`, `append_archive`,
`read_archive`, the compaction lock pair); `compaction.py:230-424` (`compact_conversation` end to end,
including the archive-before-rewrite abort at `:382-394` and the re-read-by-identity rewrite at `:396-417`);
`compaction_migration.py` (full file — confirms `run_backlog_migration`/`_drain_session` write **no**
`sections` key, matching plan.md:405-412's claim that a backlog drain after sectioning would flatten the row
and rely on the tolerant reader + next cascade tick to self-heal). Every citation checked this round matched
the source exactly. No factual misstatement found.

**No issue found** on this lens beyond what's captured under Missed-opportunity/Concurrency below.

## Lens 2 — Logical

**No issue found.** The round-6 L-1 internal contradiction ("one hop" vs. "recurse... cap to a small N") is
fully resolved (see carried table) with a single, consistently-restated mechanism (full-follow + visited-set
cycle guard) and no residual ambiguity in any of the three artifacts. I looked specifically for a
new instance of the same failure pattern (two different mechanisms described for one behavior) elsewhere in
the revised text and did not find one — the same-tick lock-continuity fix (F3) is internally consistent
(acquire→release, acquire→release, correctness by re-read), and the multi-input tier-3 fold description
(carried from round 5/6, unchanged) remains internally consistent on this re-read.

## Lens 3 — Missed opportunity

**Finding MO-2 (Major, migration class) — the migration/tolerant-reader's fallback (`covers_from_ts :=
covers_until_ts`) still reproduces #82 exactly, and the plan's claim that it is a safe "conservative floor" is
false against its own cascade reclassification mechanism; the primary (derivable) path may also be
underspecified enough to hit the same bug.**

Both the tolerant reader (`plan.md:136-145`) and the one-time migration (`plan.md:392-403`, `spec.md:182-191`)
derive the synthesized tier-3 section's `covers_from_ts` two ways: **(a)** "the actual oldest available turn
ts (buffer/archive) if derivable" — the genuine fix; **(b)** "else `covers_until_ts`" — an explicit fallback,
described at `plan.md:143-145` as: *"absent that, tier 3 is the conservative floor — it can only fade, never
mislabel-as-recent."* `spec.md:188-189` and `plan.md:399` both go further and claim: *"either way it lands in
tier 3 by the classifier."*

Both claims are false for branch (b), by the plan's **own** mechanism, cited two paragraphs earlier in the
same document:

1. The cascade's classifier does not consult which dict key a section currently sits under. It **re-derives
   the tier from content age on every pass**: `plan.md:182-187` (§1.3 step 3) — *"Age-classify each existing
   section by the age of its OLDEST covered ts (`now − covers_from_ts`)... `bucket_of(sec)`: `24h` if
   `age ≤ 48h`, else `48h` if `age ≤ 72h`, else `72h`."* This is the exact mechanism the plan itself cites as
   the fix for the *original* MO-1 defect (mislabeling-as-recent) — it operates purely on the `covers_from_ts`
   **value**, not on which key the tolerant reader/migration happened to write the section into.
2. For any actively-used, continuously-running persona — which `spec.md:12-14`'s own Problem-1 description
   names as the primary target population ("accumulated (often months-old) history... [under] the OLD
   single-layer system... re-folded each cadence") — `covers_until_ts` is the **newest** covered ts, updated
   to roughly "yesterday" on *every* daily fold, right up until the moment migration runs. It is not a proxy
   for how old the *accumulated* text is; it is a proxy for how *recently the summary was last touched*,
   which for a healthy, actively-used persona is always recent, almost by definition.
3. Therefore: if branch (b) fires, the very next cascade pass after migration computes
   `now − covers_from_ts = now − covers_until_ts ≈` a day or less, and `bucket_of()` returns `24h`
   ("yesterday") — **not** 72h. The section is pulled straight back out of tier 3 and re-labeled "yesterday,"
   reproducing #82 for exactly the population this phase's Problem 1 describes, at the one moment (migration)
   the plan itself calls "for 100% of current production personas" (`spec.md:187`, `plan.md:397`, referring
   to the original MO-1 defect — the same sentence-level framing applies unchanged to this fallback).
4. This is not confined to the one-time migration function. The **tolerant reader** (`plan.md:136-145`) is a
   *live, hot-path* defensive read applied to "a section-less row... before migration runs" — i.e., on every
   read of an unmigrated persona until the startup migration thread completes. The identical fallback and the
   identical false "conservative floor" claim appear there too (`plan.md:143-145`), so the exposure window is
   not "one migration run" but "every live session for every persona until migration finishes."
5. **Additional, compounding risk in the "derivable" primary path itself:** the plan never specifies *how*
   "the actual oldest available turn ts (buffer/archive)" is determined, nor gives buffer/archive precedence.
   The **buffer** (`active_conversations/<sid>.jsonl`) only ever holds turns younger than the existing daily
   fold's `older_than=24h` cutoff — i.e., for an actively-used persona, the buffer's oldest turn is itself
   almost always <24h old (confirmed by re-reading `compaction.py:275-284`, the `min_keep_tail`-protected
   window, and the daily-tick call site described at `spec.md:249-251`, `older_than=timedelta(hours=24)`).
   If an implementer reads "derivable" as "the buffer has *a* turn, use its ts" (a natural reading — the
   buffer is the cheaper, more obvious source to check first, and it will almost always have *some* turn), the
   "primary," non-fallback path would **also** synthesize a too-recent `covers_from_ts` for essentially every
   active persona, not just the ones that hit the explicit fallback — silently defeating the MO-1 fix
   entirely rather than merely leaving a narrow edge case. Only a full **archive** scan (all the way back to
   persona creation) gives a genuinely old value, and the plan never states that the archive — not the buffer
   — is the required/preferred source, nor that the scan must cover the full provenance chain rather than
   just the most recent segment.

**Untested:** `1.5-criteria.md` C12 (146-155) asserts "run the cascade on the migrated output → assert it
stays tier 3... NOT tier 1" — but its stated fixture is "legacy-form fixture (months-old history)" with no
mention of which derivation branch (a or b) the fixture is built to exercise, nor of what the fixture's
archive/buffer contents are. If the fixture (as is natural to build) supplies a full, coherent archive with a
genuinely old first-turn ts, C12 will pass while validating only the derivable path — exactly the
"criterion exercises a proxy, not the scenario that actually needs it" failure pattern round 6 caught in
UA-2, now recurring one level up in the same feature area. Neither the fallback branch nor the buffer-vs-
archive precedence question is exercised by any criterion in the file.

- **Severity: Major.** Same population-scale and same-defect-reproduction reasoning that made MO-1 Major
  (spec's own words: "the exact #82 defect this phase kills, for 100% of current production personas") applies
  here — the difference is that round 6's MO-1 fix closed the *documented* case (derivable path) but left
  either an unaddressed fallback (if buffer/archive precedence is implemented as intended) or, in the more
  concerning reading, may have left the *entire* fix inert if "derivable" is naturally implemented as
  "check the buffer." Either reading reproduces the phase's own target defect; I cannot rule out the more
  severe reading from the text as written, and the plan gives this default none of the "documented module
  const with a reasoned default + a one-line follow-up note" treatment it explicitly promises every other
  engineering default (`spec.md §5`).
- **Class (per the orchestrator's classification request): migration.** Not cascade (the cascade's
  reclassification mechanism itself is correct and unchanged — it is doing exactly what round 3/5 fixed it to
  do; the bug is in what timestamp migration/tolerant-reader feeds it), and not weekly-rollover/successor-
  redirect (unrelated subsystem). **The round-6 owner HARD STOP-CONDITION applies only to a new Major in the
  weekly-rollover/successor-redirect class — this finding does not trigger it.**
- **Concrete fix directions** (not prescribing one): (i) require the archive scan explicitly and specify it
  must cover the full segment/provenance chain, not just the most recent file; (ii) if no genuinely-old
  timestamp is derivable at all (a truly corrupt/absent archive), use an explicitly *conservative* floor that
  is actually conservative under the classifier's own mechanism — e.g., the persona directory's creation
  time, or an epoch/very-old sentinel — rather than `covers_until_ts`, so the "can only fade, never
  mislabel-as-recent" claim becomes true instead of aspirational; (iii) add a C12 sub-case that explicitly
  drives the fallback branch (no derivable oldest-turn-ts) and asserts it, too, survives the cascade at tier
  3, not just the happy path.

## Lens 4 — Unstated assumptions & risks

**No new finding.** I looked for a fresh instance of the "asserted outcome that doesn't reconcile with the
actual mechanism" pattern (round 6's carried F3/UA-1) elsewhere in the revised text and found none — the F3
fix itself is now internally consistent (see Carried table), and I did not find an analogous gap in the L-1
or UA-2 revisions. (MO-2 above is filed under Missed-opportunity rather than here because it fits that lens's
established pattern in this review chain — an under-examined engineering default — more precisely than an
"unstated assumption"; the assumption in question, that `covers_until_ts` is a safe stand-in for content age,
*is* stated, just wrong.)

## Lens 5 — Fidelity (owner mechanism vs proxy)

Loaded terms re-pinned and checked against the plan's actual (round-7) mechanism:

- **"Terminal"** and **"Multi-input"** (tier-3 fold) — unchanged text since round 6; conformance re-confirmed
  by re-reading `plan.md:189-222`. **Conforms.**
- **"Full-follow" / visited-set cycle guard** (new pin this round, replacing round 6's flagged ambiguity) →
  pins to: the chain resolution walks every `rolled_to.json` hop to the live successor with no depth
  ceiling, aborting only on a genuine revisit (cycle), never on chain length. `plan.md:52-59` and `:95`
  implement exactly this, and `criteria.md` C16's multi-generation oracle (192-211) is the correct
  discriminating test for it (3 generations, not 1). **Conforms.**
- **"Labels"**, **"Caps"**, **"Structural guard"** — unchanged text since round 6; no re-litigation needed,
  spot-re-read to confirm no textual drift. **Conform.**
- **"Conservative floor" / "safe default"** (the tolerant-reader/migration fallback language, `plan.md:143-145`,
  `spec.md:188-189`) — this is the term I pin fresh this round, and it does **not** conform: the plan uses
  "conservative" and "safe" to mean "placed under the tier-3 dict key," but the owner-pinned mechanism this
  entire phase is built around (the oldest-edge classifier, `plan.md:182-187`) determines a section's actual
  tier by its `covers_from_ts` **value**, not its storage location. A value that isn't actually old is not
  conservative under that mechanism, regardless of which key it's initially filed under. This is the basis
  for Finding MO-2 above — flagged here as the lens-5 fidelity angle on the same underlying gap (a
  loaded/reassuring term ["conservative"] applied to a mechanism it doesn't actually describe).

---

## Position lens (fires — ST1.5d)

Unchanged since round 6 (no text touched in the render/prefix sections). Re-confirmed by direct re-read:
`engine.py:405-408` still authors the unchanged `f"[Earlier in this conversation: {summary_text}]"` prefix;
`budget.py`'s `_COMPACTION_SUMMARY_PREFIX` match and preserve-head logic are untouched by this round's edits
(all of which are in the migration/redirect/lock sections). **No issue found.**

## Concurrency lens (fires — ST1.5e)

Re-enumerated this round with fresh source reads rather than trusting the plan's own table:
`buffer.py:341-392` (compaction lock — non-reentrant, file-based, self-contained per call, confirmed);
`server.py:1261-1272,2820-2844` (in-flight lookup and close cleanup — confirmed both still use the raw,
pre-redirect id in the current pre-build tree, matching the plan's own "these are the 5 unfixed sites"
description and giving C21 a genuine, currently-failing target); `session.py:96-97,222-231` (`_SESSIONS`,
`_LOCK`, `remove_session` — confirmed registry-only, no file I/O, matching the plan's "no build work needed"
claim). The F3 same-tick lock-continuity fix (re-read, not continuity) is coherent against this primitive
(see carried table). **No new concurrency-lens issue found** beyond MO-2, which is a data-correctness gap
(wrong timestamp fed to a classifier) rather than a race/lost-update — filed under Missed-opportunity, not
here, since no two writers contend over it.

---

## Coverage challenge (CH8)

Ranked by how directly each threatens shipped behavior for the target population, worst first:

1. **(Major — restates Missed-opportunity MO-2)** Migration-fallback / tolerant-reader `covers_from_ts`
   synthesis feeding the cascade's own classifier a too-recent value. Concrete scenario: any actively-used
   existing persona, post-migration (or read live via the tolerant reader before migration completes), has
   its accumulated history mislabeled "yesterday" on the very next cascade pass — reproducing #82 at exactly
   the moment this phase exists to prevent it. **Not observed by C12** (fixture doesn't pin which derivation
   branch it exercises) **or by C14** (fresh-start-only, per its own documented scope in round 6's label
   audit, unchanged this round).
2. **(Minor)** Multi-generation chain interaction with the carried-cursor mechanism (C18): C18's oracle
   (criteria.md:265-269) is specified against a single 1c-B rollover. Whether the carried-tail
   extraction-state-transfer logic is re-exercised correctly on a *second* or *third* successive weekly
   rollover (i.e., does generation N's seed correctly carry forward generation N-1's already-partially-carried
   cursor state) is not tested by any criterion. Low materiality — the mechanism is stated to be uniform
   per-rollover, so a bug here would likely also show up on generation 1 — listed for completeness, not as an
   independent risk.
3. **(Nitpick, informational — restates round 6's item 5, still true, still explicitly owner-accepted)** The
   SYNC full-fold's wall-clock blocking cost (A1-a) remains ungated by any criterion; already flagged in the
   plan text itself as a deliberate, accepted cost, so listed only for completeness.

No new coverage gap found in the weekly-rollover/successor-redirect class or the core cascade mechanics beyond
what's listed above — those areas were extensively re-verified this round (Concurrency lens, Fidelity lens)
and hold.

## Label audit (CH9/CH10)

Walked all 22 gating criteria + 2 advisory against "does it exercise the real path it governs, not a proxy,"
focusing fresh scrutiny on the three criteria whose text changed since round 6 (C12, C16, C18) and
spot-re-confirming the rest did not regress:

- **C12 — migration idempotent/#82-safe:** **NOT cleanly passing.** As detailed in Finding MO-2, the fixture
  is under-specified with respect to which derivation branch it exercises, and the "run cascade on migrated
  output" oracle — the load-bearing #82-regression guard — would pass even if the fallback branch (or a
  buffer-based misreading of "derivable") is broken, because nothing pins the fixture to a scenario that
  forces that branch. This is the one criterion I'd flag as not meeting the "shown able to fail on the
  scenario that matters" bar this round — structurally the same failure pattern round 6 caught in UA-2 (now
  resolved for C18), recurring in C12.
- **C16 — post-rollover redirect, multi-generation:** **Clean, and now discriminating where round 6 found it
  wasn't.** The added 3-generation oracle (sid1→sid2→sid3) directly targets the exact scenario Logical-lens
  L-1 identified last round as untested; I confirmed by re-reading the criterion text that it names concrete
  generation counts and a concrete assertion (`get_or_hydrate_session(sid1)` resolves to `sid3`), not a vague
  "chain resolves" claim.
- **C18 — carried raw-tail extraction state:** **Clean.** Re-pointed exactly as UA-2 demanded; independently
  confirmed the underlying speaker-filter claim in its rationale against source (`pipeline.py:317-324`) this
  round, so the criterion's own stated reasoning is accurate, not just plausible.
- **C1–C11, C13–C15, C17, C19–C22:** no text changed since round 6; spot-re-read a sample (C6's head-prefix
  oracle via fresh `engine.py`/`budget.py` reads, C21's structural-guard fail-demo via fresh `server.py`
  reads) and found the same clean, real-path, fail-demo-equipped shape round 6 documented in full for all of
  these. No regression found.
- **Advisory A1/A2:** unchanged, reasons still consistent with the project's stated measurement posture. No
  challenge.

---

## Bottom line

Three of the four round-6 carried items are genuinely, cleanly resolved this round: **L-1** (redirect chain —
now a coherent full-follow-with-visited-set-guard mechanism, multi-generation-tested, zero residual "one hop"
language anywhere in the three artifacts), **F3** (lock-continuity — now correctly stated as re-read-based
correctness against the actual non-reentrant per-call lock primitive, verified against fresh reads of
`buffer.py` and `compaction.py`), and **UA-2** (C18 — correctly re-pointed to the real risk, with the
speaker-filter rationale independently confirmed against source). The fourth, **MO-1**, is resolved for the
path it was explicitly built to fix (derivable oldest-turn-ts → tier 3) but leaves its own fallback branch —
and, on one plausible reading, potentially its primary branch too, depending on how "derivable" gets
implemented — carrying the *identical* defect it was meant to close. This is filed as a new finding, **MO-2**,
rather than "MO-1 still open," because genuine, verified progress was made; the gap is a real residue in the
same family, not a rejection of the round-6 fix.

**Worst finding this round: Major (MO-2).** It is squarely in the **migration** class (the migration/
tolerant-reader default that feeds `covers_from_ts` to the unchanged, correct cascade classifier), not the
cascade class (the classifier itself is correct and unchanged) and not the weekly-rollover/successor-redirect
class (a different subsystem entirely — that class is now clean: L-1 resolved, C16 multi-generation-tested,
C21 structural guard independently re-confirmed). **The round-6 owner HARD STOP-CONDITION — halt on a new
Major in the weekly-rollover/successor-redirect class — is explicitly NOT triggered by this finding.**

**Routing: back to stage 2 (narrow) — not clean, not minors-only.** Scope for the revision, all confined to
the migration/tolerant-reader default (no owner-pinned design element — tiers, terminal, labels, caps, oldest-
edge graduation, redirect chain — needs re-opening): (1) specify explicitly that `covers_from_ts` derivation
must scan the **archive** (full provenance chain, not just the most recent segment), not the buffer, and state
why the buffer is unsuitable (its oldest turn is bounded by the existing 24h fold cutoff); (2) replace the
`else covers_until_ts` fallback with a value that is actually conservative against the classifier's own
oldest-edge mechanism (e.g., persona-creation time, or an explicit very-old sentinel) rather than a value that
routinely reproduces #82 one cascade pass later; (3) extend C12 with an explicit sub-case that pins the
fallback (non-derivable) branch and asserts it also survives the cascade at tier 3 — closing the same
"criterion exercises the happy path only" gap round 6 caught (and fixed) in C18, now recurring in C12.
