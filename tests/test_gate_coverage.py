"""Coverage tests for the Phase-1 consolidation gate (Root-2 stopgap).

TEMP — companion to tests/test_consolidation_gate.py. These tests close the coverage the
isolation tests missed (change: changes/gate-leak-coverage/):

- G1-G4: four automatic writers routed through ``route_write``. G2-G4
  (self_model_resolved / making / file_write) are GATED — a candidate lands in the pending
  queue and ZERO rows reach memories.db. G1 (self_model_reconcile) is a later owner-decision
  BYPASS (GATE_BYPASS_TYPES in brain/memory/pending.py) — it commits straight to memories.db.
- G5: a LIVE-PATH integration test proving the gate FIRES end-to-end when the real per-turn
  write path runs (monologue capture + the public pass-2 ``apply_side_effects`` wrapper) — the
  coverage that would have caught the Run-#1 apparatus regression at unit level.
- G6: a direct-write coverage guard — enumerates every ``.create(`` under brain/ and asserts
  it is a subset of a pinned allowlist, so a NEW automatic writer that bypasses the gate (any
  receiver name) or a regression of a routed locus fails CI.
- G7: the owner-BYPASS writers (grief_event / kindled_peer) stay direct — guards the owner
  ruling against an over-eager "route everything" edit.
- A2 (advisory): documents the file_write exact-dup cross-tick drop (designed gate behavior).

None of these are marked ``integration``/``live``/``requires_claude_cli`` — they must run in the
default CI selection (that is the whole point: the missed coverage runs in CI).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from brain.emotion import vocabulary
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore

_BRAIN_ROOT = Path(__file__).resolve().parents[1] / "brain"
_TEST_CHANNEL = "gatecoverage_channel"


# ── helpers ──────────────────────────────────────────────────────────────────
def _db_count(persona_dir: Path, memory_type: str) -> int:
    db = persona_dir / "memories.db"
    if not db.exists():
        return 0
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM memories WHERE memory_type=?", (memory_type,)
        ).fetchone()[0]
    finally:
        conn.close()


def _queue_types(persona_dir: Path) -> list[str]:
    import json

    q = persona_dir / "pending_candidates.jsonl"
    if not q.exists():
        return []
    out = []
    for line in q.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line).get("memory_type"))
    return out


def _mem(content: str, mtype: str) -> Memory:
    return Memory.create_new(content=content, memory_type=mtype, domain="test")


@pytest.fixture
def registered_channel():
    """Register a test emotion channel (registry is process-global; clean up after)."""
    vocabulary._unregister(_TEST_CHANNEL)
    vocabulary.register(
        vocabulary.Emotion(
            name=_TEST_CHANNEL,
            description="test-only channel for gate-coverage tests",
            category="persona_extension",
            decay_half_life_days=None,
        )
    )
    yield _TEST_CHANNEL
    vocabulary._unregister(_TEST_CHANNEL)


# ── G1: self_model_reconcile BYPASSES the gate (owner decision); G2-G4: each
# remaining newly-routed automatic writer enqueues, none reach memories.db ──
def test_g1_reconcile_bypasses_gate(tmp_path, registered_channel):
    """self_model_reconcile BYPASSES the gate (owner decision, GATE_BYPASS_TYPES in
    brain/memory/pending.py): it commits straight to memories.db so a self-authored
    emotional nudge affects felt state immediately."""
    from brain.self_model.reconcile import _write_self_authored_delta

    wrote = _write_self_authored_delta(tmp_path, registered_channel, 0.5)
    assert wrote is True
    assert _db_count(tmp_path, "self_model_reconcile") == 1
    assert _queue_types(tmp_path).count("self_model_reconcile") == 0  # bypasses the gate


def test_g2_resolve_routes_and_still_queues_soul(tmp_path):
    from brain.self_model.gap import Gap
    from brain.self_model.resolve import _emit_resolution

    gap = Gap(per_channel={"warmth": 0.5}, magnitude=0.5, unnamed_pressure=0.0)
    _emit_resolution(tmp_path, gap, resolution_path="reconcile", session_id="s1")
    assert _queue_types(tmp_path).count("self_model_resolved") == 1
    assert _db_count(tmp_path, "self_model_resolved") == 0
    # routing did not break the soul-candidate side-effect (importance 9 >= threshold)
    assert (tmp_path / "soul_candidates.jsonl").exists()


def test_g3_maker_routes_to_queue(tmp_path):
    from brain.maker.maker import Making
    from brain.maker.wiring import write_making_memory

    store = MemoryStore(tmp_path / "memories.db")
    try:
        making = Making(type="poem", title="untitled", content="x", disposition="private")
        write_making_memory(store, making, emotions={})
    finally:
        store.close()
    assert _queue_types(tmp_path).count("making") == 1
    assert _db_count(tmp_path, "making") == 0


def test_g4_file_write_routes_to_queue(tmp_path):
    from datetime import UTC, datetime

    from brain.files import pending as files_pending
    from brain.files.commit import decline_write

    store = MemoryStore(tmp_path / "memories.db")
    try:
        rid = files_pending.create(
            tmp_path,
            op="create",
            resolved_path=str(tmp_path / "note.txt"),
            content="hi",
            now=datetime.now(UTC),
        )
        res = decline_write(tmp_path, rid, store=store)
    finally:
        store.close()
    assert res["ok"] is True
    assert _queue_types(tmp_path).count("file_write") == 1
    assert _db_count(tmp_path, "file_write") == 0


# ── G5: LIVE-PATH integration — the gate demonstrably FIRES end-to-end ──────────
def test_g5_live_path_gate_fires_end_to_end(tmp_path, registered_channel):
    """Drive the real per-turn write path in-process; prove the gate fires.

    monologue capture path (the real record_monologue handler → write_trace_memory) +
    the public apply_side_effects wrapper (the one tool_loop.py's pass-2 worker calls).
    """
    from brain.chat.extractor import ExtractorOutput, MemoryWrite, apply_side_effects
    from brain.chat.monologue_capture import capture_monologue

    store = MemoryStore(tmp_path / "memories.db")
    try:
        capture_monologue(
            persona_dir=tmp_path,
            store=store,
            monologue="Bob mentioned his knee PT is going well.",
            feed_digest="thinking about Bob's PT",
        )
    finally:
        store.close()

    out = ExtractorOutput(
        memory_writes=[MemoryWrite(episode="Bob's knee PT is improving.", salience=0.7)],
        emotion_delta={registered_channel: 0.3},
    )
    apply_side_effects(out, persona_dir=tmp_path)

    # The gate FIRED: the pending queue was created and populated ...
    pend = tmp_path / "pending_candidates.jsonl"
    assert pend.exists(), "pending_candidates.jsonl was never created — gate is a no-op"
    qtypes = _queue_types(tmp_path)
    assert qtypes, "pending queue is empty — nothing was enqueued"
    assert "monologue_trace" in qtypes  # from capture_monologue
    assert "monologue" in qtypes  # from apply_side_effects memory_writes
    # ... and NO gated type leaked straight into the recall channel.
    for gated in ("monologue", "monologue_trace"):
        assert _db_count(tmp_path, gated) == 0, f"{gated} leaked into memories.db"
    # monologue_emotion BYPASSES the gate (owner decision, GATE_BYPASS_TYPES in
    # brain/memory/pending.py): it commits straight to memories.db so per-turn
    # emotional nudges affect felt state immediately, not queued for consolidation.
    assert _db_count(tmp_path, "monologue_emotion") == 1, "monologue_emotion bypass didn't fire"
    assert "monologue_emotion" not in qtypes, "monologue_emotion should bypass the queue"

    # Shown-able-to-fail: the oracle distinguishes a gated write that DOES reach the DB
    # (an ungated direct store.create — the pre-gate behavior). If this assertion could not
    # fire, the zero-checks above would be meaningless.
    s2 = MemoryStore(tmp_path / "memories.db")
    try:
        s2.create(_mem("ungated monologue stand-in", "monologue"))
    finally:
        s2.close()
    assert _db_count(tmp_path, "monologue") == 1


# ── G6: direct-write coverage guard (fails safe on any novel/aliased receiver) ──
# Pinned by (relpath, exact stripped line-text) so unrelated line shifts don't false-fail.
# List 1 — KNOWN-NON-MEMORY .create( (brain.files.pending, NOT a MemoryStore):
_KNOWN_NON_MEMORY = {
    ("tools/impls/propose_write.py",
     "rid = pending.create(persona_dir, op=op, resolved_path=str(g.resolved),"),
    ("tools/impls/propose_write.py",
     "rid = pending.create(persona_dir, op=op, resolved_path=str(g.resolved), content=content,"),
}
# List 2 — ALLOWED-DIRECT memory writes (legitimately direct; see 2-plan.md):
_ALLOWED_DIRECT = {
    ("body/events.py", "store.create(journal)"),
    ("engines/consolidation.py", "store.create(cand)"),
    ("grief/breadcrumb.py", "return store.create(memory)"),
    ("kindled_link/relationship.py", "mem_store.create(mem)"),
    ("memory/pending.py", "return store.create(mem)"),
    ("migrator/cli.py", "store.create(mem)"),
    ("migrator/cli.py", "soul_store.create(crystal)"),
    ("migrator/emergence_kit.py", "soul_store.create(crystal)"),
    ("migrator/emergence_kit.py", "store.create(mem)"),
    ("recovery/engine.py", "store.create(mem)"),
    ("soul/review.py", "soul_store.create(c)"),
    ("tools/impls/add_journal.py", "store.create(memory)"),
    ("tools/impls/add_memory.py", "store.create(memory)"),
    ("tools/impls/crystallize_soul.py", "soul_store.create(c)"),
}
_PINNED = _KNOWN_NON_MEMORY | _ALLOWED_DIRECT
_ROUTED_LOCI = {
    "self_model/reconcile.py",
    "self_model/resolve.py",
    "maker/wiring.py",
    "files/commit.py",
}


def _enumerate_create_sites(root: Path = _BRAIN_ROOT) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for py in root.rglob("*.py"):
        rel = py.relative_to(root).as_posix()
        for raw in py.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if ".create(" in line and ".create_new(" not in line:
                found.add((rel, line))
    return found


def test_g6_no_ungated_direct_create_writer():
    found = _enumerate_create_sites()
    unpinned = found - _PINNED
    assert not unpinned, (
        "Un-allowlisted direct .create( site(s) found — an automatic memory writer may be "
        "bypassing the consolidation gate. Route it through brain.memory.pending.route_write, "
        "or (if a legitimate bypass) add it to the pinned allowlist in this test with a reason:\n"
        + "\n".join(f"  {rel}: {txt}" for rel, txt in sorted(unpinned))
    )
    # A routed locus must NOT reappear as a direct .create( (regression tripwire).
    for rel, _txt in found:
        assert rel not in _ROUTED_LOCI, f"routed locus regressed to direct .create(: {rel}"


def test_g6_guard_can_fail(tmp_path):
    """Shown-able-to-fail: run the REAL enumerator against a synthetic novel-receiver writer
    and confirm the guard would flag it (un-pinned) while ignoring .create_new(."""
    fake_root = tmp_path / "fakebrain"
    (fake_root / "sub").mkdir(parents=True)
    (fake_root / "sub" / "new_writer.py").write_text(
        "def f(s, mem):\n"
        "    s.create(mem)        # novel aliased receiver — must be caught\n"
        "    Memory.create_new()  # must NOT be caught (different method)\n",
        encoding="utf-8",
    )
    found = _enumerate_create_sites(fake_root)
    assert ("sub/new_writer.py", "s.create(mem)        # novel aliased receiver — must be caught") in found
    # the create_new line is excluded
    assert not any("create_new" in txt for _rel, txt in found)
    # and such a novel writer is NOT in the pinned allowlist → test_g6 would FAIL on it
    assert (found - _PINNED)


# ── G7: owner-BYPASS writers stay direct (guards the owner ruling) ──────────────
def test_g7_bypass_writers_remain_direct():
    grief = (_BRAIN_ROOT / "grief" / "breadcrumb.py").read_text(encoding="utf-8")
    kindled = (_BRAIN_ROOT / "kindled_link" / "relationship.py").read_text(encoding="utf-8")
    # grief_event: significant + low-volume, must NOT be dedup-dropped → direct store.create.
    assert "return store.create(memory)" in grief
    assert "route_write" not in grief
    # kindled_peer: a peer's genuine words, not her firehose → direct mem_store.create.
    assert "mem_store.create(mem)" in kindled
    assert "route_write" not in kindled


# ── A2 (advisory): document the file_write exact-dup cross-tick drop ─────────────
def test_a2_file_write_identical_content_dropped_cross_tick(tmp_path):
    """Advisory (not gating): identical file_write content is dropped by the gate's exact-dup
    across ticks/sessions (DB-wide, no time bound). Documents the accepted, designed behavior."""
    from brain.engines.consolidation import run_consolidation

    store = MemoryStore(tmp_path / "memories.db")
    try:
        q = PendingQueue(tmp_path)
        content = "you let me write to /home/bob/notes.txt"
        # tick 1: first identical → promoted (degraded promote-all classifier, no provider)
        q.enqueue(_mem(content, "file_write"), source="file_write")
        run_consolidation(store, persona_dir=tmp_path)
        assert _db_count(tmp_path, "file_write") == 1
        # tick 2 (a LATER drain): identical content → dropped by _has_exact_existing (DB-wide)
        q.enqueue(_mem(content, "file_write"), source="file_write")
        run_consolidation(store, persona_dir=tmp_path)
        assert _db_count(tmp_path, "file_write") == 1  # still 1 — permanent cross-tick drop
    finally:
        store.close()
