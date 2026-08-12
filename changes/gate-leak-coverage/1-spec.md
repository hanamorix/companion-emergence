# 1 — Spec: close the direct-`store.create` leaks in the Phase-1 consolidation gate

## Problem
The Phase-1 memory-consolidation gate (Root-2 flood stopgap) routes automatic memory
writes through `route_write` (`brain/memory/pending.py`), which **enqueues** a gated type
as a candidate rather than writing it straight into `memories.db`. Several **automatic**
memory writers bypass `route_write` and call `store.create` (or `mem_store.create`)
directly, so their output reaches the recall channel ungated — the gate leaks for those
types.

## Scope of THIS change (owner-fixed, 2026-08-12)
Route these **automatic** writers through `route_write` (so they GATE / enqueue):
- `brain/self_model/reconcile.py:123` — type `self_model_reconcile`
- `brain/self_model/resolve.py:157` — type `self_model_resolved`
- `brain/maker/wiring.py:69` — type `making`
- `brain/files/commit.py:23` — type `file_write`

Leave these as direct `store.create` (BYPASS — owner rulings):
- `brain/recovery/engine.py:128 & 132` — **restores** real backed-up memories (arbitrary
  original types), not the generated firehose. Gating a restore would wrongly divert a
  genuine memory into the candidate queue.
- `brain/body/events.py:75` — emits `journal_entry`, already a `GATE_BYPASS_TYPES` member.
- `brain/grief/breadcrumb.py:138` — type `grief_event`: significant + low-volume, must NOT
  be dedup-dropped (owner ruling).
- `brain/kindled_link/relationship.py:415` — type `kindled_peer`: a peer's genuine words,
  not her generated firehose (owner ruling).

## What is explicitly NOT the fix here (diagnosis, verified this session)
The *dominant* live symptom in Testing's Run #1 (monologue-family flood of ~223 rows into
`memories.db`, `pending_candidates.jsonl` never created) is **NOT** a brain coverage gap.
The monologue-family writers (`brain/chat/extractor.py::_apply_memory_writes`,
`brain/monologue/trace.py::write_trace_memory`, `brain/ingest/commit.py`) already route
through `route_write`, and were **proven** this session to enqueue correctly at HEAD
(`3e98ee57`): driving them against a temp persona populates `pending_candidates.jsonl` and
writes zero gated rows to `memories.db`. The live no-op was an **apparatus** fault — the
test server imported the *ungated* main-checkout brain (main working tree is on branch
`ThinkerOfThoughts/diagnose-monologue-bleed-memory-gap`, whose brain has no `pending.py`,
no `consolidation.py`, zero `route_write`) because `live_server.py`'s `PHASE1_BRAIN_REPO`
redirect was ineffective. **That fix belongs to the Testing chat's harness lane and is out
of scope here.** This change (a) closes the genuine, smaller brain-side leaks the audit
surfaced, and (b) adds the in-CI coverage that would have caught *both* a missed writer and
the apparatus regression.

## Constraints / prior art
- `route_write` keys on `memory_type`, not the call site (docstring, `pending.py:18`). The 4
  routed types are NOT in `GATE_BYPASS_TYPES`, so `route_write` enqueues them — no change to
  `GATE_BYPASS_TYPES` needed. The 4 bypass call sites never call `route_write`, so their
  types staying out of `GATE_BYPASS_TYPES` is correct; they remain call-site bypasses.
- Each of the 4 routed call sites already holds a `MemoryStore` whose `persona_dir`
  (`db_path.parent`) locates the queue — `route_write(store, mem, source=...)` works with no
  new plumbing.
- `resolve.py` uses `mem.id` after the write to queue a soul candidate. `route_write`
  returns `mem.id` (both branches), and the "route then `queue_soul_candidate(memory_id=…)`"
  pattern is already the accepted design in `extractor.py:552` (monologue_soul_candidate).
  Routing `resolve.py` inherits that accepted behavior (the soul candidate references a
  pending id until the gate promotes it — identical to the shipped extractor path).
- No push / no PR / local-only. Branch `ThinkerOfThoughts/memory-rework-groundwork`.

## Expected touched files
- `brain/self_model/reconcile.py` (route the write)
- `brain/self_model/resolve.py` (route the write)
- `brain/maker/wiring.py` (route the write)
- `brain/files/commit.py` (route the write)
- `tests/test_consolidation_gate.py` (new tests: per-locus routing, live-path integration,
  call-site allowlist guard) — or a new sibling test module `tests/test_gate_coverage.py`.
