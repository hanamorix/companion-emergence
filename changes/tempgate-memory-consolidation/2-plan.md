# 2 — Plan (separate-queue architecture): how + measurement + instrumentation + thresholds

## Build order

### Step 0 — `file_lock` non-blocking mode (`brain/utils/file_lock.py`)
Add `file_lock(path, *, blocking: bool = True)`: POSIX `LOCK_EX | LOCK_NB` catching `BlockingIOError`,
Windows `msvcrt LK_NBLCK` single attempt; yields a bool "acquired". Existing blocking behavior is the
default (soul-candidate callers unchanged). Used by the gate's non-blocking persona lock.

### Step 1 — `PendingQueue` (`brain/memory/pending.py`, NEW)
JSONL queue at `persona_dir / "pending_candidates.jsonl"`, mirroring the proven `soul_candidates.jsonl`
pattern (`brain/soul/review.py:274-292,532-540`). Header carries the **TEMP marker** (C14).
- `enqueue(mem: Memory, *, source: str)` — append `{**mem.to_dict(), "_source": source,
  "_enqueued_at": iso}` under `file_lock` (append is short; safe against a concurrent drain).
- `read_recent(memory_type: str, *, limit: int) -> list[dict]` — lock-free read via
  `brain/health/jsonl_reader.read_jsonl_skipping_corrupt` (tolerates a partial last line from a
  concurrent append), filter by `memory_type`, newest-first, `limit`. For interior-continuity.
- `drain() -> list[dict]` — **under `file_lock`**: read all entries, then truncate the file, return
  entries. Fast (no slow work under the lock); Pass 1/2 process the returned batch lock-free, so
  candidates enqueued during processing land in the emptied file and are caught next tick.
  **Durability note (round-4, accepted for the stopgap):** truncate-before-process means a crash mid
  Pass-2 loses the drained batch (removed from the file, not yet promoted). This differs from the
  soul precedent, which rewrites survivors as the *last* step. Accepted because firehose content is
  regenerable and this is a TEMP stopgap; named here, not silently carried. (If undesired, switch to
  process-then-rewrite-survivors — a build option.)
- A candidate is `Memory`-shaped (`to_dict`), so promotion = `store.create(Memory.from_dict(entry))`.

### Step 2 — Store + hebbian helpers
- `brain/memory/store.py` — add a `db_path` / `persona_dir` accessor (writers + queue locate the
  persona dir; today `db_path` isn't exposed). **No `consolidation_state` column, no `committed_only`
  param, no `bump` change** (all v1 apparatus dropped — recall reads are unchanged).
- `brain/memory/hebbian.py` — add `set_edge_weight(a, b, weight)` (UPSERT to `MAX(existing, weight)`,
  capped 10.0) for correction/continuation seeding at 5.0.

### Step 3 — Route writes by `memory_type` (the write seam — TYPE-based, not site-based)
Add a helper `route_write(store, pending, mem, source)`: `if mem.memory_type in GATED_TYPES:
pending.enqueue(mem, source) else: store.create(mem)`. Define `GATED_TYPES` as a named constant
(spec write-seam section; the scope knob). Switch each **automatic** `store.create(mem)` site to
`route_write`, threading a `PendingQueue`/`persona_dir` into the engines that lack one:
`monologue/trace.py:31`, `chat/extractor.py:348/408/546`, `engines/dream.py:277`,
`engines/reflex.py:462`, `engines/research.py:327`, `initiate/memory.py:156`,
`engines/heartbeat.py:943`, `ingest/commit.py:69`. Because it keys on `memory_type`, the two
variable-typed sites (`reflex.py:452` `arc.output_memory_type`; `ingest/commit.py:69` `item.label`)
self-classify per row — a reflex `journal_entry` writes direct, a reflex gated-type enqueues
(round-4 MAJOR fix). Deliberate/import/restore sites keep `store.create`. Per-site + `GATED_TYPES`
membership verified at build (fidelity — the red-team/owner audits the set).

### Step 4 — The gate (`brain/engines/consolidation.py`, NEW)
`run_consolidation(store, pending, *, persona_dir, classifier=None, config=...) -> Result`. TEMP
marker.
- **Enter:** acquire the non-blocking persona gate lock (`file_lock(gate_lock_path, blocking=False)`);
  not acquired → return `skipped`.
- **Drain:** `batch = pending.drain()`.
- **Pass 1 (pure):** normalize (`casefold` + collapse whitespace + strip); **discard** exact repeats
  (within batch or identical to an existing `memories.db` memory — read-only compare via a
  non-recall-bumping read); **discard** low-salience **only for `SALIENCE_ELIGIBLE_TYPES =
  {monologue, monologue_emotion, monologue_soul_candidate}`** (the episode types with a real 0..10
  importance signal), floor `SALIENCE_FLOOR` default OFF — **all other gated types
  (dream/research/heartbeat/reflex/monologue_trace) are EXEMPT from salience-drop** and go
  (note: `initiate_outbound` is excluded from `GATED_TYPES` entirely, so it never reaches Pass 1 —
  it is not "gated-but-exempt")
  through dedup only (owner directive 4 / R3; prevents nuking flat dreams at importance≈0 and traces
  at 0.3); fragment-similarity **clusters only**, never discards non-identical.
- **Pass 2 (injected classifier; stub in tests):** per candidate `∈ {duplicate, merge, distinct,
  correction, continuation, new}` + a target/referent id for merge/correction/continuation:
  - `duplicate` → discard.
  - `merge` → **re-read the target immediately before editing; skip the merge (leave candidate for
    next tick) if it is missing or `state=='fading'`** — this closes the merge-vs-concurrent-`fade`
    lost-update window (round-4 minor: a `fade` in the supervisor thread rewrites the target's
    `content`→summary between the gate's read and its `update`; skipping-if-fading avoids clobbering
    it, and skipping-if-missing avoids the `KeyError`). Then archive the (re-read) pre-image to the
    **plain append-only archive file** (mirroring `compaction.py:376-381` `append_archive`) **before**
    editing; surgical `store.update`; discard the candidate. Belt-and-braces: still wrap the `update`
    in try/except KeyError (fault-isolated skip+log).
  - `distinct`/`new` → `store.create(Memory.from_dict(entry))` (promote).
  - `correction`/`continuation` → promote + `hebbian.set_edge_weight(new_id, referent_id, 5.0)`;
    `correction` flagged in metadata.
- **Reject** is implicit: a discarded candidate was removed by `drain()`'s truncate and is simply not
  acted upon — **no `memories.db`, graveyard, grief, or hebbian touch** (C4).
- **Instrumentation:** one structured log line per run — batch size, exact-discarded, clusters,
  merged, promoted, corrections, continuations, `skipped`. The observable that promotions ≪ batch.

### Step 5 — Wire into the idle tick (`brain/engines/heartbeat.py`)
In `run_tick`, after `persona_dir` (~489) and before reflex (491), call
`run_consolidation(self.store, self._pending, persona_dir=persona_dir, ...)`, fault-isolated
(`if not dry_run`). (C9.)

### Step 6 — Consumer migration
- **Interior-continuity → queue:** `brain/monologue/ambient.py:34` and `brain/monologue/recall.py:29`
  read recent `monologue_trace` from `PendingQueue.read_recent`, not `store.list_by_type`. (C3.)
- **Bridge feed / initiate dedup / others:** per the spec consumer-migration section — the
  crystallizer/interest/notes/maker reads (promoted-only) are benign; **`feed.py` behavior and
  `initiate/memory.py:105` dedup are FLAGGED for the stage-4 owner ruling** (C17). The build applies
  the ruled resolution.

### Step 7 — Removal dependency (C14)
Record the Phase-2 removal dependency in `~/.claude/plans/memory-dream-rework-plan.md` (Phase-2 gains
"remove the temp consolidation gate + pending queue" as predecessor cleanup). In-code TEMP marker is
the in-repo signal.

### Step 8 — Tests (per criterion). Pass-2 via injected stub; C10 injects an enqueue into the drain
window; C11 invokes a second gate run against the held lock. Every oracle first shown to fail against
a known-violating variant (ST1.5f).

## Measurement
All criteria verified by the **test suite** (the project harness for a behavioral change; the config's
telemetry metrics are advisory-only — no replay workload). Stage 8 runs the tests + CI and fills the
per-criterion table. C16 (advisory) is a post-sign-off live sample judged by Roy. C17 is finalized
once the stage-4 consumer ruling lands.

## Instrumentation (added)
- The gate's per-run structured log line (Step 4) — advisory signal that the flood is cut at source.
- No new `memories.db` columns or telemetry are needed for the gating criteria (all
  unit/integration-checkable). CP3/CFG6: no *gating* signal is unmeasurable, so none is added beyond
  the gate log.

## Concurrency: accessor enumeration + guard scope (ST2b / CP7)
**Two** distinct shared states now:

**(A) The pending queue** (`pending_candidates.jsonl`) — new shared mutable state.

| Accessor | R/W | Synchronized by | Covered? |
|---|---|---|---|
| firehose `enqueue` (chat/extractor/engines, concurrent threads) | append | `file_lock` (short append) | ✔ |
| gate `drain` (read-all + truncate) | R+W | `file_lock` over the r-m-w window (the soul-candidate P2-2 pattern) | ✔ — an enqueue during the window either lands before the truncate (drained) or after (next tick); never lost. C10. |
| interior-continuity `read_recent` | R | lock-free; `read_jsonl_skipping_corrupt` tolerates a partial concurrent-append line | ✔ (read-only; worst case skips one corrupt line) |

**(B) `memories.db` merge target** — the gate now WRITES the main DB (promote/merge).

| Accessor | R/W | Synchronized by | Covered? |
|---|---|---|---|
| gate promote (`store.create`) / merge (`store.update` on the existing memory) | W | non-blocking persona **gate lock** (only one gate run at a time) + per-statement WAL | ✔ overlapping-run: the gate lock skips a contended run so a merge target is never double-folded. C11. |
| a second overlapping `run_tick` gate | R+W | the gate lock (skip if contended) | ✔ |
| forgetting `fade`/`unfade`/`LOSE` (supervisor thread, `supervisor.py:423`) | W | NOT the gate lock (different thread/cadence) | Partial: a concurrent `LOSE` can `hard_delete` a merge target between the gate's read and its `update` → `KeyError` (`store.py:405`). Low-prob (LOSE hits >30-day low-salience rows; merge targets are salient) + fault-isolated; the build **wraps the merge `update` in try/except KeyError**. |
| deliberate writers `store.create` | W | per-statement WAL | orthogonal (append). |

The candidate/reject side has **no `memories.db` accessor at all** — the v1 lost-update/grief/forgetting
interactions are dissolved by construction (a candidate is not a DB row).

## Severity → routing thresholds
- **Blocker (→ stage 1):** a candidate reaches recall (C1/C2 fail — e.g. an automatic writer not
  redirected); interior-continuity starved (C3); reject causes a `memories.db`/grief side effect (C4);
  lost candidate on drain (C10) or double-fold on overlap (C11); recall/forgetting/cache code touched
  (C13). These defeat the change or regress behavior.
- **Major (→ stage 5):** merge not archived / not surgical / to the grief graveyard (C7); corrections
  merged or wrong/weakened weight (C8); gate not first (C9); a deliberate writer wrongly enqueued or an
  automatic writer wrongly direct (C12 / Step-3 misclassification); a consumer regression (C17) once
  the ruling is known.
- **Minor (fix-and-proceed):** log fields; test-seam ergonomics; marker wording.
- **Advisory:** C16 live quality; the gate log; magnitude tuning (L-B).

## Deferred defaults (parameterized, pending L-B)
`SALIENCE_FLOOR = None` (salience discard OFF — no numeric floor safe pre-L-B; when enabled it only
discards a *pending* entry, never a memory); `CLUSTER_SIM = <conservative>`; merge = surgical;
`ASSOC_WEIGHT = 5.0`. Named constants so L-B retunes without a structural change. Default
pending-confirmation per the brief.
