# Stage-3 Plan Red-Team — Round 3 — Cascade Compaction (verbatim cold-review record)

> Filed by the orchestrator from the round-3 sonnet cold reviewer's returned text (the reviewer returned
> to main as text; this is the durable record). Model: claude-sonnet-5, cold, first read of the artifact set.

## Provenance
Artifacts (sha256): 1-spec.md 8f1f42b3…800c; 1.5-criteria.md a1c05b15…18f9; 2-plan.md a4c0caa1…3b78e.
Branch base: ThinkerOfThoughts/cascade-compaction @ cd29bc61.
Source read (whole-file sha noted in the live review): session.py 773f1e0b…, server.py f0b0b715…,
compaction.py cc9b6e22…, buffer.py cefb0799…, pipeline.py cfe8b63b…, supervisor.py ca6eeba8…,
engine.py d359be52…, budget.py b71ed34d…, compaction_migration.py 7d32a50b…, ambient.py 74dd5ba0…,
attempt_heal.py ddfbec5d…, daemon.py 8fbcd4a7…, ingest/__init__.py cfa37eb9…, test_pipeline.py e2ab1108…,
New_mem_system.md ad4c76ff… (Parts 1-2).

## Carried-forward resolution
- **P-1 (round-2 BLOCKER):** RESOLVED for its named scope (/chat, /stream). session.py:129-205 structure
  confirmed; _LOCK = threading.RLock() (session.py:97) reentrant → redirect recursion deadlock-safe (plan
  doesn't cite this but it's true); C16 now drives get_or_hydrate_session directly + a /chat TestClient POST,
  bare ingest_turn explicitly disqualified. NEW residual surfaced → F1/F2 below.
- crash mid-cascade-write: RESOLVED (plan §1.3 "atomic single write", all tiers from pre-pass snapshot →
  one rewrite_session_atomic; C17 tests it; rewrite_session_atomic @ buffer.py:265).
- dir-fsync Windows-safe: RESOLVED (plan §3 cites attempt_heal.py:250-266; posix guard real).
- migration ordering: RESOLVED (plan §4 self-heal via tolerant reader + next tick; run_backlog_migration
  unmodified @ compaction_migration.py:162).
- C8/C9 locus: RESOLVED (criteria state real /sessions/active endpoint + real _run_compaction_tick).
- CH8-4 cursor: RESOLVED (C18 asserts seed cursor prevents re-extraction).
- empty/corrupt finalize branch nitpick: RESOLVED (plan §2.3 covers both :587-590 success + :556-560 empty;
  poison-move :591-609 unchanged).

## Per-lens — NEW findings

**F1 (Major).** Redirect fix is incomplete. grep `sess = get_or_hydrate_session` in server.py = FOUR hits,
not two: /chat (2365), /stream (2424), /sessions/snapshot (2696), /sessions/close (2753). close+snapshot use
`sess` only for the None-gate, then call the backend pipeline fn with the RAW pre-redirect req.session_id
(2699/2703 snapshot; 2765/2771 close). After a rollover the old buffer is deleted → close_session hits the
"empty session" guard (pipeline.py:150-154) → silently deletes an already-gone buffer, returns committed=0,
handler reports "closed": True. The client's real live conversation (under the successor sid) is never
closed/extracted — silent no-op that looks like success. Same for /sessions/snapshot (empty-guard
pipeline.py:326). Plan's "single-locus, no handler edits needed" claim is false for these two.

**F2 (Major).** in_flight_locks (server.py:816, dict[str, asyncio.Lock]) is a 4th unenumerated shared
accessor, keyed by the PRE-redirect id at all four sites (2368/2699/2765; 2429). Single continuous client
resubmitting old sid self-serializes (fine). But /sessions/active (server.py:1203-1253) returns the
SUCCESSOR real sid to a fresh client post-rollover; a 2nd client (2nd window/device/admin tool) using the
successor sid directly writes the same buffer under a DIFFERENT in_flight_locks key than the original
client's redirected old-sid traffic. Not in plan §5's accessor table.

**F3 (Major) — the meaty one.** Age-classification proxy permits indefinite age-laundering under continuous
activity. Plan §1.3 step 3 reclassifies/merges-down by `now − covers_until_ts` (NEWEST covered ts) only;
step 4 recomputes the rebuilt section's covers_until_ts from all inputs incl. the fresh raw group. For a
session chatting daily with no >24h gap: each pass folds [prior 24h section] + G24 → new 24h section whose
covers_until_ts refreshes to ≈"now−24h", so `now − covers_until_ts` stays perpetually ~24h and the
merge-down condition (age > 48h) NEVER fires while G24 is non-empty. Material that is really many days old
keeps being re-folded (lossy, 60%) into a section perpetually labeled "24h — recent," never migrating to
48h/72h. This is the COMMON case (an actively-used companion) and reproduces the spec's original Problem-1
defect ("can't tell 2-hours-ago from 3-days-ago material") one tier down. Not caught by C14 (its steady-daily
fixture asserts against covers_until_ts — the classifier's own input field → tautological).

**F4 (Minor).** Plan §2.1 "cursor, not a lock, serializes extraction" not supported: extract_session_snapshot
(pipeline.py:317-318) does unguarded read-then-later-write of the cursor; two concurrent callers (finalize
tick on supervisor thread + SYNC 1c-A rollover extraction on a request worker thread) can both read the same
pre-advance cursor and double-process; embedding-cosine dedupe is a soft net, not a guarantee. C10 asserts
buffer/seed survive, not that extraction ran exactly once. May be pre-existing, but the plan's explicit safety
CLAIM is unbacked.

## Missed opportunity
- covers_from_ts is already in the §1.1 schema but unused by the §1.3 classification → cheap fix for F3
  (classify by span/covers_from_ts, or force merge-down when covers_from_ts alone crosses the boundary).
- snapshot_stale_sessions (pipeline.py:475-526) is a live, scheduled (daemon.py:108, server.py:684,
  supervisor.py:319, 5-min cadence) buffer accessor with its OWN unlocked ghost-buffer delete
  (pipeline.py:504-506) → a 5th deleter, falsifying plan §2.3's "a buffer is deleted EXACTLY by the path that
  supersedes it." Low harm (only fires on truly-empty buffer) but the enumeration/claim is wrong.
- close_stale_sessions (pipeline.py:629-673) is dead code (no call site); its docstring is stale — one-line
  plan note would save a future reader the verification.

## Assumptions/risks
- "Transparent, no client contract change" assumes exactly ONE live client per session lineage ever; F1/F2
  show that's false once a 2nd discovery path touches the successor sid.
- remove_session (session.py:222-231) ALREADY only pops _SESSIONS today (no file-delete exists there) → the
  plan's "change remove_session to retain only registry eviction" is a no-op; don't miscount as build work.
- Q8 cap: only the 24h section has a hard cap (_SECTION_24H_CHAR_CAP); 48h/72h are "bounded by input" (the
  uncapped prior section). Low risk (fractions contractive 0.40/0.20) but no independent hard ceiling — one-line ack.

## Fidelity
- "layer IS the timestamp" — faithful (structural render, §1.2). "correct-by-construction" — accurate for
  /chat,/stream; overclaimed globally (F1). "24h layer IS recent material" — the clearest fidelity gap: the
  implemented mechanism is "classify by an always-refreshable proxy for age," not "classify by true content
  age" (F3). Dream-ordering slot — faithful P1 no-op. COMPACTION_MODEL="haiku" — confirmed compaction.py:54.

## Position lens (fires) — No issue.
engine.py:405-408 head f-string unchanged; budget.py:28-31 prefix + :94-109 re-parse (matches index 1)
unchanged; determinism arg correct; C6 tests by execution. F3 is a temporal-CORRECTNESS defect, not a
position/cache defect — render stays byte-stable + re-parseable even while content is stale-mislabeled.

## Concurrency lens (fires)
Unenumerated beyond §5: (1) in_flight_locks keyed pre-redirect (F2); (2) snapshot_stale_sessions ghost-delete
unlocked, 5-min cadence (Minor); (3) extraction-cursor race (F4). _LOCK reentrancy confirmed sound. No
lock-ordering issue from reading rolled_to.json under _LOCK (plain file read).

## Coverage challenge (CH8-new)
1 (Major, F1) close/snapshot post-rollover silent no-op reported as success. 2 (Major, F2) cross-key
concurrent write via dual sid discovery. 3 (Major, F3) age-laundering under continuous no-gap activity — C14
tautological. 4 (Minor) snapshot_stale ghost-delete × in-progress rollover. 5 (Minor) finalize/rollover
double-extraction via unlocked cursor. (Round-1 gaps C14/C15/C8-multistale/C16 spot-checked present.)

## Label audit
C16 now targets the real path (sound) but scope narrower than blast radius — only /chat, not close/snapshot
(F1); broaden or add a sibling. C17/C18/C8/C9 sound, real paths. C4(b) Minor: wording doesn't unambiguously
require driving the full cascade_conversation path vs a narrower _fold_into_section/predicate unit — tighten
(mirror how C16 was tightened). A1/A2 advisory reasoned. No other proxy-labeling issues.

## Bottom line
Worst-severity = Major (not Blocker) — materially stronger than round 2; P-1 genuinely fixed at the correct
locus incl. the RLock detail; code-map citations excellent (no fabrication across ~11 files). Three NEW
Majors from reading past the cited lines: F1 (redirect breaks close/snapshot silently), F2 (in_flight_locks
key-split), F3 (age-classification launders stale content into "24h" under continuous use, reproducing
Problem-1 one tier down, C14 tautological). Routing: back to stage 2 (plan) — spec target design unchanged.
(a) extend the get_or_hydrate_session redirect to correct close/snapshot's use of sess.session_id, or
scope-fence with a criterion proving the no-op is at least SAFE (not silently reported as success); (b) key
in_flight_locks (+ compaction/rollover lock) consistently off resolved sess.session_id; (c) revise §1.3
age-classification to use covers_from_ts (true-span) not covers_until_ts alone, and strengthen C14 to assert
against something other than the classifier's own input field. Minors (F4, snapshot_stale enumeration, C4(b)
wording) fold in, not independently blocking.

---

## Author disposition round 3 (orchestrator-routed; gate-4 entry in `decisions.md`)
Route: **Major → return to stage 2 (narrow).** All round-2 carried findings confirmed resolved; three new
Majors + minors addressed:
- **F1 (Major, redirect incomplete):** `get_or_hydrate_session` has FOUR call sites; each handler now uses the
  **resolved `sess.session_id`** for its backend call so `/sessions/close` + `/sessions/snapshot` act on the
  successor (not a false-success `committed=0` no-op). Plan §0, spec §2e-B, §7; **C16 broadened to all 4
  handlers**.
- **F2 (Major, `in_flight_locks` key-split):** all 4 handlers key `in_flight_locks` (+ the rollover/compaction
  lock) by the **resolved sid**. Plan §0/§5; **new C19**.
- **F3 (Major, age-laundering — the meaty one):** cascade reworked to true **graduation** — classify by the
  **oldest** edge (`covers_from_ts`), and **never co-fold the prior 24h section with fresh raw** (that refresh
  was the launder). Under continuous use each cohort graduates raw→24h→48h→72h→out. Plan §1.3, spec §2b;
  **C14 strengthened** to assert graduation on **true content age** via a marker-preserving fake provider
  (non-tautological), across many consecutive passes.
- **Minors:** F4 corrected (extraction cursor race is pre-existing/unguarded in `extract_session_snapshot`,
  not worsened, exactly-once fix out of scope); `snapshot_stale_sessions` ghost-delete enumerated as a 5th
  (harmless, empty-only) deleter + the "deleted exactly by superseding path" claim corrected; C4 tightened to
  the real `cascade_conversation` path; only-24h-has-a-hard-cap acked; `remove_session` already registry-only
  (not build work) corrected; `close_stale_sessions` dead-code noted.

## Iteration-cap status (SEV4) — reported to the orchestrator
Three stage-3 rounds logged at gate 4. Two distinct **finding classes** have each bounced **twice**:
- **Cascade-promotion mechanism** — round-1 F1 (tick-shift) + round-3 F3 (age-laundering) = 2 bounces.
- **Rollover/redirect lifecycle** — round-2 P-1 (redirect locus) + round-3 F1/F2 (redirect scope + lock key)
  = 2 bounces.
Per SEV4 ("after 2 bounces at the same gate on the same finding class, stop and a human breaks the tie"), a
**third** bounce on **either** class in round 4 triggers a human tie-break. This lap resolved both at their
mechanism roots (graduation by true age; resolved-sid at all four handler sites), so the intent is to clear
them, not re-bounce — but the orchestrator/owner should be aware we are at the cap on both.
