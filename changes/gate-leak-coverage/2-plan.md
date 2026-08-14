# 2 — Plan

## How (code changes)
Each of the 4 GATE loci already holds a `MemoryStore store` with a real `persona_dir`.
Replace the direct create with `route_write`; import locally (matching the existing
in-function import style used everywhere `route_write` is already called).

1. **`brain/self_model/reconcile.py:123`**
   `store.create(mem)` → `from brain.memory.pending import route_write; route_write(store, mem, source="self_model_reconcile")`.
   Return value unused.

2. **`brain/self_model/resolve.py:157`**
   `store.create(mem); memory_id = mem.id` →
   `from brain.memory.pending import route_write; memory_id = route_write(store, mem, source="self_model_resolved")`.
   `route_write` returns `mem.id`, so `memory_id` is preserved and the downstream
   `queue_soul_candidate(memory_id=…)` is unchanged. (Same accepted pattern as
   `extractor.py:552`.) `source` label matches the type `self_model_resolved` (provenance-only
   metadata; does not affect gating, which keys on `memory_type`).

3. **`brain/maker/wiring.py:69`** (`write_making_memory`)
   `store.create(mem)` → `route_write(store, mem, source="maker")`. Keep the surrounding
   try/except.

4. **`brain/files/commit.py:23`** (`_wire_memory`)
   `store.create(Memory.create_new(...))` → build the `Memory` into a local, then
   `route_write(store, mem, source="file_write")`. Keep the try/except.

No change to `GATE_BYPASS_TYPES`, `route_write`, `consolidation.py`, or the 4 BYPASS loci.

## Measurement (how each criterion is verified) — all via pytest in default CI
- **G1–G4** per-locus: new tests drive each real function against a `tmp_path` persona and
  assert (queue has the type) ∧ (DB has zero of the type). Each includes a shown-able-to-fail
  note (the pre-change direct-create form lands in DB) demonstrated by an inline ungated
  control write.
- **G5** live-path integration: new test drives `capture_monologue` (→ `write_trace_memory`)
  and the **public** `apply_side_effects(ExtractorOutput(memory_writes=…, emotion_delta=…),
  persona_dir=…)` wrapper (the one the live pass-2 worker calls at `tool_loop.py:78`, one layer
  above `_apply_memory_writes`, so the `_safely` dispatch is covered) against a `tmp_path`
  persona; asserts `pending_candidates.jsonl` exists and is non-empty and DB has zero gated
  rows. Unmarked (runs in default CI). No provider/Claude CLI (neither function calls a
  provider in its write path).
- **G6** coverage guard: new test enumerates **all** `\.create\(` lines under `brain/**/*.py`
  (line-regex; matches any receiver, excludes `.create_new(`) and asserts the set is a subset
  of the pinned union of (1) KNOWN-NON-MEMORY excludes = the `pending.create(` calls in
  `tools/impls/propose_write.py` (`brain.files.pending`, NOT a MemoryStore — flagged by the
  stage-3 reviewer), and (2) the ALLOWED-DIRECT memory-write allowlist below. Matching ALL
  `.create(` and subtracting only pinned lists means a novel/aliased receiver
  (`s.create(`, etc.) fails safe. Inline negative sub-checks demonstrate it can fail.
- **G7** bypass-unchanged: allowlist lists the 4 bypass writers as allowed-direct; a positive
  test writes a `grief_event`/`kindled_peer` via direct `store.create` and asserts it reaches
  the DB (documents the intended bypass).
- **G8** CI: run ruff + the marker-scoped pytest locally.

## Instrumentation
None needed — the gate already exposes the observable signals (the queue file + the DB). No
new logging/telemetry. The regression metrics in the project config (cost/cache/num_turns)
are **advisory-only and irrelevant** to this correctness change; stage 8 is **conformance-only**
(no stage-0 baseline: this is not a cost/behavioral-telemetry change).

## The pinned lists for G6 — verified against a full enumeration (21 `.create(` lines today)

**List 1 — KNOWN-NON-MEMORY `.create(` (excluded; not MemoryStore writes):**
- `brain/tools/impls/propose_write.py:74` and `:91` — `pending.create(...)` =
  `brain.files.pending.create` (the file-write-approval request queue). Unrelated to
  `MemoryStore`/`PendingQueue`. (Stage-3 reviewer catch.)

**List 2 — ALLOWED-DIRECT memory writes (legitimately direct):**
- `brain/recovery/engine.py:128, :132` — restores real backed-up memories.
- `brain/body/events.py:75` — `journal_entry` (bypass type).
- `brain/grief/breadcrumb.py:138` — `grief_event` (owner bypass ruling).
- `brain/kindled_link/relationship.py:415` — `kindled_peer` (owner bypass ruling; `mem_store`).
- `brain/engines/consolidation.py:298` — the gate's OWN promote of a vetted candidate INTO DB.
- `brain/memory/pending.py:132` — `route_write`'s bypass branch (`store.create` for bypass types).
- `brain/tools/impls/add_memory.py:71`, `add_journal.py:51`, `crystallize_soul.py:82` — explicit
  USER writes.
- `brain/soul/review.py:419` — promotion of a vetted soul candidate.
- `brain/migrator/cli.py:183, :300`, `brain/migrator/emergence_kit.py:181, :265` — migration
  tooling (not per-turn automatic generation).

The pinned lists are keyed by **`(file, exact line-text)`** (not line numbers), so unrelated
line shifts inside an allowlisted file don't false-fail the guard (round-2 reviewer note).

Enumeration counts (the guard works on a **set** of `(relpath, stripped-text)` tuples, so
physically-identical lines in the same file collapse): `grep` finds **21 physical** `.create(`
lines today; `recovery/engine.py`'s two identical `store.create(mem)` lines collapse to one
tuple → **20 unique tuples** = List1 (2 propose_write) ∪ List2 (14 unique; 15 physical) ∪ the 4
routed loci (4). **After this change** the 4 routed loci call `route_write` (no `.create(`), so
the enumerated set becomes exactly List1 ∪ List2 = **16 unique tuples** (17 physical lines). The
guard asserts `enumerated ⊆ List1 ∪ List2`; it fails if any automatic writer appears with a
direct create (any receiver name — fails safe), or if a routed locus regresses to `store.create`.
The guard scans `brain/` only (scripts/ and tests/ are out of scope).

## Blast radius accepted (CH8 — surfaced, not fixed; owner-ordered gating consequences)
- **file_write exact-dup (A2; corrected scope):** `_wire_memory` content is fixed
  low-cardinality (`"you let me write to {path}"`/`"...declined..."`). Pass-1 exact-dup checks
  within-batch AND against **all existing active `memories.db` rows** (`_has_exact_existing` →
  `store.search_text`, NO time bound, `consolidation.py:176-179`). So after the first
  `file_write` memory for a (path, outcome) is promoted, every later identical one — any tick,
  any session — is dropped **permanently** from recall (full history stays in `audit.jsonl`).
  Designed gate behavior; arguably desirable (dedup), accepted advisory but **surfaced to the
  owner** with true scope so file_write's GATE ruling can be revisited if undesired.
- **self_model_reconcile visibility latency (A3):** the emotion-delta memory is delayed to the
  next idle tick and contingent on promotion. Verified the gap `acknowledged`/cooldown state is
  NOT coupled to the memory write (`reconcile.py:177-186`: keyed on `gap is not None`, not on the
  write), so only the supplementary memory's visibility is delayed — inherent to gating any type,
  which the owner explicitly ordered. Surfaced to the owner in the completion report.

## Thresholds (severity → routing)
- **blocker** (a routed write does not actually enqueue; a bypass write wrongly enqueued; the
  guard cannot enumerate call sites; CI red) → back to build/plan.
- **major** (guard allowlist is incomplete/incorrect such that a real leak is missed; live-path
  test doesn't actually exercise the real functions / can't fail) → back to build.
- **minor** (naming, `source=` label wording, test placement) → fix-and-proceed.
- gating criteria: G1–G8. advisory: A1.

## Risk notes for the red-team to challenge
- **Coverage completeness** — did the audit enumerate ALL `store.create`/`*.create` memory
  writers in `brain/`? The allowlist + G6 guard is the durable answer; challenge whether the
  enumerator's regex misses a call form (e.g. aliased store, `mem_store.create`,
  multi-line calls).
- **Does G5 genuinely prove the gate fires end-to-end** (not another isolation test)? Challenge
  whether driving `capture_monologue` + `_apply_memory_writes` is representative of the live
  bridge write path, and whether it fails against an ungated stand-in.
- **resolve.py soul-candidate** — confirm routing doesn't break `queue_soul_candidate` and that
  referencing a pending (uncommitted) id matches the shipped extractor precedent.
