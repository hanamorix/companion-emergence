"""Coverage tests for the Phase-1 consolidation gate (Root-2 stopgap).

TEMP — companion to tests/test_consolidation_gate.py. These tests close the coverage the
isolation tests missed (change: changes/gate-leak-coverage/):

- G2-G4: three automatic writers routed through ``route_write`` (self_model_resolved / making /
  file_write). They are GATED — a candidate lands in the pending queue and ZERO rows reach
  memories.db. (G1, the self_model_reconcile writer, was removed with the emotion-only row writes.)
- G5: a LIVE-PATH integration test proving the gate FIRES end-to-end when the real per-turn
  write path runs (monologue capture + the public pass-2 ``apply_side_effects`` wrapper) — the
  coverage that would have caught the Run-#1 apparatus regression at unit level.
- G6: a direct-write coverage guard — enumerates every ``.create(`` under brain/ and asserts
  it is a subset of a pinned allowlist, so a NEW automatic writer that bypasses the gate (any
  receiver name) or a regression of a routed locus fails CI.
- G6b: the same pin for ``.update(..., check_names=False)`` (a write that skips the check on a memory's feeling names).
- G7: the owner-BYPASS writers (grief_event / kindled_peer) stay direct — guards the owner
  ruling against an over-eager "route everything" edit.
- A2 (advisory): documents the file_write exact-dup cross-tick drop (designed gate behavior).

None of these are marked ``integration``/``live``/``requires_claude_cli`` — they must run in the
default CI selection (that is the whole point: the missed coverage runs in CI).
"""

from __future__ import annotations

import ast
import sqlite3
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import pytest

from brain.emotion import vocabulary
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore
from tests.source_tree import brain_sources

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


# ── G2-G4: each newly-routed automatic writer enqueues, none reach memories.db ──
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
    # The pass-2 emotion_delta no longer writes a row at all: no monologue_emotion row in
    # memories.db and none queued.
    assert _db_count(tmp_path, "monologue_emotion") == 0, "emotion_delta must not write a row"
    assert "monologue_emotion" not in qtypes, "emotion_delta must not queue a candidate"

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
    ("migrator/cli.py", "store.create(mem, fold=False, check_names=False)"),
    ("migrator/cli.py", "soul_store.create(crystal)"),
    ("migrator/emergence_kit.py", "soul_store.create(crystal)"),
    ("migrator/emergence_kit.py", "store.create(mem, fold=False, check_names=False)"),
    ("recovery/engine.py", "store.create(mem, fold=False, check_names=False)"),
    ("soul/review.py", "soul_store.create(c)"),
    ("tools/impls/add_journal.py", "store.create(memory)"),
    ("tools/impls/add_memory.py", "store.create(memory, check_names=False)"),
    ("tools/impls/crystallize_soul.py", "soul_store.create(c)"),
}
_PINNED = _KNOWN_NON_MEMORY | _ALLOWED_DIRECT
_ROUTED_LOCI = {
    "self_model/reconcile.py",
    "self_model/resolve.py",
    "maker/wiring.py",
    "files/commit.py",
}


def _enumerate_create_sites(root: Path | None = None) -> set[tuple[str, str]]:
    """Every ``.create(`` line as (path relative to brain/, stripped line). ``root`` None reads the real ``brain/`` from
    the process-wide copy in tests/source_tree.py; a path scans that tree (the synthetic one in the can-fail test)."""
    if root is None:
        files = {rel.removeprefix("brain/"): text for rel, text in brain_sources().items()}
    else:
        files = {py.relative_to(root).as_posix(): py.read_text(encoding="utf-8") for py in root.rglob("*.py")}
    found: set[tuple[str, str]] = set()
    for rel, text in files.items():
        for raw in text.splitlines():
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


# ── G6b: the same for `.update(..., check_names=False)` (#803 step 17) ──────────────────────
# `MemoryStore.update` checks a memory's feeling names unless `check_names=False`. A write that turns the check off is pinned, like
# the `.create(` writers above, by (relpath, exact stripped line) and a count, with a reason. The shapes looked at: an `.update(`
# call, a `dict(...)` call, a dict literal and a subscript assignment that set `check_names` to anything but the constant True.
# Any other callee (an alias of `update`, `functools.partial(s.update, ...)`, a reader such as `_resolved_faded`) is caught when it is
# given the constant False; the readers that do are pinned as reads. Not caught: a `**kwargs` dict built somewhere this scan does not
# look (a name, a function's return), a value held in a variable (`check_names=flag` into an `update`-named call is caught, into any
# other callee it is not), a `check_names` set with `setattr`.
_CHECK_NAMES_OFF_PINNED: dict[tuple[str, str], tuple[int, str]] = {
    ("memory/store.py", "fields.update(emotions=_merge_emotions(target._stored_emotions, candidate._stored_emotions), fold=False, check_names=False)"): (
        1,
        "the consolidation merge builds the kwargs dict that `self.update(target.id, **fields)` takes on the next line (this line is the "
        "dict, not the store call): the merged map comes from two stored rows whose names the store already wrote tidied",
    ),
    ("emotion/engine.py", "emotions = self._resolved_faded(memory, check_names=False)  # the store checked the emotions before writing them"): (
        1,
        "a READ of a row's feelings by the engine's hook after the store checked and wrote the row (`_resolved_faded`'s docstring); writes nothing",
    ),
    ("emotion/engine.py", "emotions = self._resolved_faded(memory, check_names=False)  # the store checked the row before writing it"): (
        1,
        "a READ of a row's feelings by the engine's hook after the store checked and wrote the row (`_resolved_faded`'s docstring); writes nothing",
    ),
    ("emotion/engine.py", "return self._resolved_faded(memory, check_names=False) if counted_kind(memory) is not None else {}"): (
        1,
        "a READ of a stored row's feelings replayed under the engine lock (spec section 7, `_resolved_faded`'s docstring); writes nothing",
    ),
}


def _is_true(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _enumerate_check_names_off(files: Mapping[str, str]) -> Counter[tuple[str, str]]:
    """Every place that sets `check_names` to something other than True, as (path relative to brain/, stripped line)."""
    found: Counter[tuple[str, str]] = Counter()
    for rel, text in files.items():
        lines = text.splitlines()
        for node in ast.walk(ast.parse(text)):
            hit = False
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                # `update` and `dict` (the kwargs of an update) with any value but True; any other callee (an alias `u = s.update`,
                # `functools.partial(s.update, ...)`, `getattr(s, "update")(...)`) with the constant False. `create` is G6's.
                # A variable forwarded to a callee (`check_names=check_names`) is not turning the check off by itself.
                hit = name != "create" and any(
                    k.arg == "check_names" and (not _is_true(k.value) if name in ("update", "dict") else isinstance(k.value, ast.Constant) and not _is_true(k.value))
                    for k in node.keywords
                )
            elif isinstance(node, ast.Dict):
                hit = any(isinstance(k, ast.Constant) and k.value == "check_names" and not _is_true(v) for k, v in zip(node.keys, node.values, strict=True))
            elif isinstance(node, ast.Assign):
                hit = not _is_true(node.value) and any(
                    isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == "check_names" for t in node.targets
                )
            if hit:
                found[(rel.removeprefix("brain/"), lines[node.lineno - 1].strip())] += 1
    return found


def test_g6b_no_unpinned_update_that_skips_the_name_check():
    found = _enumerate_check_names_off({rel: text for rel, text in brain_sources().items() if "check_names" in text})
    unpinned = {k: n for k, n in found.items() if k not in _CHECK_NAMES_OFF_PINNED or n > _CHECK_NAMES_OFF_PINNED[k][0]}
    assert not unpinned, (
        "A write that turns off the check on a memory's feeling names is new. Let the store check them, or pin it in "
        "_CHECK_NAMES_OFF_PINNED with the reason:\n" + "\n".join(f"  {rel}: {txt}" for (rel, txt) in sorted(unpinned))
    )
    stale = sorted(k for k, (count, _reason) in _CHECK_NAMES_OFF_PINNED.items() if found.get(k, 0) < count)
    assert not stale, f"pins with no hit left, remove them: {stale}"
    assert all(reason.strip() for _count, reason in _CHECK_NAMES_OFF_PINNED.values())


@pytest.mark.no_base_failure(reason="self-test of the guard on synthetic source; passes on any base by design")
def test_g6b_guard_can_fail():
    """The REAL enumerator on synthetic bypasses: each shape is found, a checked write and a constant True are not."""
    bypasses = {
        "a.py": "def f(s, i):\n    s.update(i, emotions={}, check_names=False)\n",
        "b.py": "def f(s, i, flag):\n    s.update(i, check_names=flag)\n",
        "c.py": "def f(s, i):\n    kw = dict(check_names=False)\n    s.update(i, **kw)\n",
        "d.py": "def f(s, i):\n    s.update(i, **{'check_names': False})\n",
        "e.py": "def f(s, i, fields):\n    fields['check_names'] = False\n    s.update(i, **fields)\n",
        "f.py": "def f(s, i):\n    u = s.update\n    u(i, check_names=False)\n",
        "g2.py": "import functools\ndef f(s):\n    return functools.partial(s.update, check_names=False)\n",
        "g3.py": "def f(s, i):\n    getattr(s, 'update')(i, check_names=False)\n",
    }
    clean = {
        "g.py": "def f(s, i):\n    s.update(i, emotions={})\n    s.update(i, check_names=True)\n",
        "h.py": "def f(s, i):\n    x = {'check_names': True}\n    return s.create(i, check_names=False)\n",
    }
    found = _enumerate_check_names_off(bypasses | clean)
    assert {rel for rel, _txt in found} == set(bypasses)
    assert not (set(found) & set(_CHECK_NAMES_OFF_PINNED))


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
