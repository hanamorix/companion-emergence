# 3 — Red-team ROUND 3 (verbatim record)

## Provenance
- **Reviewer:** cold subagent, `general-purpose`, model **opus**. Charter = charter.md core + the
  firing position/concurrency lenses + stage-3 CH8/CH9/CH10, quoted verbatim in the prompt, with
  rounds 1–2 records carried forward. Focus directed at the two round-2 fixes (non-blocking lock;
  the new `held` value) and whether `held` introduced a new leak.
- **Context (closed set):** the three revised artifacts + rounds-1/2 records + priority-ordered
  worktree source. Reviewer-reported sha256 (round-3 measured): 1-spec
  `681ebff2c0fe7e04d4f313a90562eb17d76db20b29140f0609f341d047987c0c`; 1.5-criteria
  `620648f72bbaa95f4c2cf035b1a23e5f1551f306bcc6e3f510cd6d5cf8af31c0`; 2-plan
  `4c1e9f0bea9452c02670a14a219c5b08f4754807c3cd63b1f81641d36ccff992`; plus store.py, file_lock.py
  `263d0e3bbef8…`, hebbian.py `50776e0063bf…`, prompt.py, search_memories.py, ambient.py,
  heartbeat.py, research.py, review.py `c9541a0a82de…`, ingest/commit.py, feed.py `e7e4588c74c…`.

## Verbatim reviewer output (key content)

**Bottom line: MAJOR → replan.** **M1 RESOLVED** (non-blocking `flock(LOCK_EX|LOCK_NB)` /
`msvcrt LK_NBLCK` yielding a bool composes cleanly with `@contextmanager`; C17 satisfiable in-process
via per-fd `flock`; skip-on-contention non-starving). **A PARTIALLY RESOLVED:** re-merge/re-process
bound is clean (`list_candidates` returns only `candidate`; nothing transitions `held→candidate`);
recall filter correctly excludes candidate AND held (C2 asserts it); and — decisively — `held`
introduces **NO new leak into non-recall consumers** (no non-recall consumer filters on
`consolidation_state`, so `candidate` and `held` are already treated identically by research seeding
`research.py:519/522`, review `review.py:341`, hebbian linking `commit.py:81`, feeds `feed.py`).

**NEW MAJOR — held rows ride the forgetting loss path → spurious grief for deduped content.**
`forgetting/__init__.py:135` selects `WHERE state IN ('active','fading')` with **no**
`consolidation_state` filter; the `LOSE` transition (:188-216) does unconditional
`graveyard.append` → `hard_delete` → `grief.handle_drop`. A `held` duplicate/merged-source is
permanently out of recall → never recall-bumped → salience decays with no floor → **structurally
guaranteed** to fade then be lost past the 30-day recent-buffer exemption (`policy.py:126`;
`is_exempt` honors only soul-crystallised / under-review / recent-buffer — **not** `protected`). Its
loss writes a graveyard entry and, when `emotion_at_ingest ≥ THRESHOLD`, a grief breadcrumb — grief
for content that was *deduped, not lost*. This is the round-1 **A-2** class reopened on a delay, and
it is change-introduced (pre-change the duplicate sat in recall and could be recall-rescued; the
gate now guarantees it decays). Spec frames "held decays via forgetting → bounds accumulation" as
purely beneficial and omits this terminal cost. No criterion observes it (C18/C19 cover only a
single gate cycle, not a downstream forgetting pass).

**Minors:** (i) plan concurrency table says forgetting "runs earlier in the same tick (heartbeat
477-480)" — FALSE; those are emotion/hebbian decay; the forgetting fade/lose pass runs in the
**supervisor thread** (`supervisor.py:423`) on its own cadence/connection. (ii) gate-vs-forgetting on
a merge target is therefore cross-thread and unguarded: a concurrent forgetting `LOSE` hard-deleting
target Y between the gate's read and its merge-write makes `store.update(Y)` raise `KeyError`
(`store.py:405`) — fault-isolated but unhandled, leaving an orphaned pre-merge archive record. Fix:
wrap the merge `update` in try/except KeyError.

CLEAN + earned: recall gate, cache-prefix isolation (C13), bounded re-processing, C2/C20 labels,
the choke point, the non-blocking lock.

## Author disposition (see decisions.md gate-3-round-3 entry)
Third MAJOR at gate 3; the recurring class is **the terminal fate of a gate-rejected candidate**
(round-1 A-2 grief-on-drop → round-2 A accumulation → round-3 grief-on-decay). Iteration cap reached
on this class + it is a genuine owner-decision the brief's "drop" does not resolve against the
grief/forgetting/referential-integrity interactions. **STOP and escalate to owner** (do not patch a
4th time). The two minors are fixed on this pass; the terminal-fate axis is marked OPEN in 1-spec
pending owner ratification.
