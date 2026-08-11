# 3 — Red-team ROUND 2 (verbatim record)

## Provenance
- **Reviewer:** cold subagent, `general-purpose`, model **opus** (adversarial re-review of a
  concurrency + data-lifecycle replan). Charter = charter.md core (five lenses + discipline + the
  firing position & concurrency lenses) + stage-3 CH8/CH9/CH10, quoted verbatim in the prompt, with
  the round-1 findings carried forward as context (closed-set rule).
- **Context (closed set):** the three revised artifacts + `3-redteam-plan.md` (round-1 record) +
  priority-ordered worktree source. Reviewer-reported sha256 (round-2 measured):
  1-spec `09067a6bcfe4bb860fe3f5f8f9376b56e3194ba4e3f09465d7d7e5d481f73ee3`;
  1.5-criteria `0bdc6aa3c520399ac7064cdc054cb779377ebbbb78f936b95ecf9621389bf7a6`;
  2-plan `5d35b48b320b7a16431a70ce23533c9b0e94bc2cddb35d42b0717efaf058a7fd`;
  store.py `d6276059…`, hebbian.py `50776e00…a326ac804…` (round-2 measured), graveyard.py `40756ecf…`,
  supervisor.py `ca6eeba8…`, server.py `f0b0b715…`, prompt.py `d904f423…`, recall.py `b096b9f4…`,
  search_memories.py `9b31524b…`, ambient.py `74dd5ba0…`, heartbeat.py `248cf2e6…`, compaction.py
  `cc9b6e22…`, reflex.py `df351ae1…`, dream.py `b4cb00bf…`, research.py `fa65566f…`, extractor.py
  `27252fe7…`, commit.py `8f52f4a1…`. (`brain/utils/file_lock.py` read; blocking-only confirmed.)

## Verbatim reviewer output (key content)

**Bottom line: MAJOR → replan.** Four of six carried round-1 findings FULLY RESOLVED
(F-1 salience-destructive, A-2 grief-graveyard, F-2 ensure_edge, CH8-1 continuity-source, CH8-5
recall-churn — each verified against source). Two MAJORs remain/introduced:

- **M1 [MAJOR]** — the A-1 fix's guard is specified as "reuse the `file_lock` helper" *as* a
  "non-blocking, skip-if-contended" lock, but `brain/utils/file_lock.py` is **blocking-only**
  (`fcntl.flock(fd, LOCK_EX)`, docstring "Blocks until acquired"). A non-blocking acquire
  (`LOCK_NB` → catch `BlockingIOError` → return `skipped`) must be *written*; **C17's assertion
  `return skipped` is unsatisfiable** with the blocking helper (the second acquire would block/
  deadlock the test). Right direction, mis-specified mechanism — stop calling it a reuse.
- **A [MAJOR]** — the non-destructive replan leaves every `duplicate`, merged-source, and
  low-salience candidate in the pool **forever**, re-read by `list_candidates(limit=None)`,
  re-clustered, and re-sent to the Pass-2 classifier **every idle tick**. (i) *certain:* the
  candidate set grows monotonically and per-tick classifier work grows without bound (also
  lengthening lock-hold → more skips); the plan's "bounded cost" (2-plan) is **false**. (ii)
  *conditional:* a merged-but-kept source W (left `candidate`, textually ≠ target Y) re-reaches
  Pass-2 next tick → possible re-merge → content drift (the exact thing "smallest surgical edit"
  prevents) + unbounded archive growth. No criterion observes multi-tick stability (all tests are
  single-cycle; C17 covers overlapping, not sequential re-fold). Cheap mitigation: a
  "processed/held" marker so decided candidates leave the pool, or cap `list_candidates`.

**Lens verdicts:** Factual mostly earned-clean (crux citations all TRUE; one FALSE claim =
`file_lock` non-blocking, M1); recall channel confirmed to be EXACTLY the three sites by grep (no
missed leak) — but the spec's "reviewed, not filtered" list was non-exhaustive (5 more callers,
none a leak) [advisory]. Logical sound. Fidelity: all terms pin cleanly EXCEPT "gate
lock/non-blocking/skip" (fails the pin → M1). Position-sensitivity CLEAN (continuity protected at
read AND source; cache prefix untouched via C13 diff-gate). Concurrency: guard direction right,
`flock` per-fd arbitration correct for both threads and cross-process, skip-on-contention does NOT
starve (triggers not phase-locked) — but M1 blocks delivery. Label audit: C7/C8 stub-check
legitimate; C18/C19 sound; C17 well-designed for the *intended* guard but inconsistent with the
specified blocking helper. Advisories: gate-merge vs concurrent forgetting-`fade` on one row is
un-locked but **non-corrupting** (different columns, atomic statements); classifier return contract
(target/referent id) unspecified.

## Author disposition (see decisions.md gate-3-round-2 entry)
MAJOR → replan. Both correctable without a respec; fixes in the round-3 artifacts, re-reviewed in
`3-redteam-plan-v3.md`.
