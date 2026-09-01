"""Conformance tests for cascade-compaction session rollover.

Maps to changes/cascade-compaction/1.5-criteria.md:
  C10  Finalize decoupling: no-lost-update over the session buffer (CONCURRENCY)
  C15  Active-set boundedness: old buffer reaped by the rollover path
  C18  Carried raw-tail extraction state preserved across rollover

Drives the REAL ``perform_rollover`` / ``finalize_stale_sessions`` — no HTTP,
no bespoke helpers standing in for the production functions.
"""

from __future__ import annotations

import json
import re
import subprocess
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.chat.rollover import perform_rollover
from brain.ingest.buffer import (
    ingest_turn,
    list_active_sessions,
    read_cursor,
    read_session,
    write_cursor,
)
from brain.ingest.pipeline import extract_session_snapshot, finalize_stale_sessions
from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.pending import PendingQueue
from brain.memory.store import MemoryStore

_PERSONA_NAME = "persona"


def _persona(tmp_path: Path) -> Path:
    p = tmp_path / "persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    # provider="fake" so build_compaction_provider (called internally by
    # perform_rollover for the fold) never shells out to a real CLI.
    (p / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "fake"}), encoding="utf-8"
    )
    return p


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _seed(
    persona_dir: Path,
    sid: str,
    n: int,
    *,
    base: datetime,
    step: timedelta,
    marker: str | None = None,
) -> list[str]:
    """Write n user/assistant turns, oldest first. Returns the ts written."""
    tss: list[str] = []
    for i in range(n):
        ts = _iso(base + step * i)
        tss.append(ts)
        speaker = "user" if i % 2 == 0 else "assistant"
        text = f"{marker}-{i}" if marker else f"turn {i}"
        ingest_turn(
            persona_dir,
            {"session_id": sid, "speaker": speaker, "text": text, "ts": ts},
        )
    return tss


class _ExtractOnlyProvider:
    """generate() answers extraction prompts with one valid item.

    Only ever handed in as the EXTRACTION provider (the ``provider=`` kwarg
    to finalize_stale_sessions / perform_rollover) — the fold step inside
    perform_rollover always builds its own provider via
    build_compaction_provider (see the module docstring's NOTE), so this
    stub never sees a fold prompt in the tests below.
    """

    def __init__(self) -> None:
        self.extract_calls = 0

    def name(self) -> str:
        return "extract-stub"

    def healthy(self) -> bool:
        return True

    def complete(self, prompt: str) -> str:
        return ""

    def generate(self, prompt=None, *, system=None, **kw) -> str:
        prompt = prompt if prompt is not None else kw.get("prompt", "")
        if "extracting durable memories" in prompt:
            self.extract_calls += 1
            return json.dumps(
                [
                    {
                        "text": "stub extracted memory",
                        "label": "note",
                        "importance": 5,
                        "emotions": {},
                    }
                ]
            )
        return "I remember this."


class _MarkerExtractProvider:
    """Extraction-only stub: one item per distinct MARK-N token in the
    prompt, echoing the marker itself as the memory text, so committed
    memories are individually identifiable back to their source turn."""

    def name(self) -> str:
        return "marker-stub"

    def healthy(self) -> bool:
        return True

    def complete(self, prompt: str) -> str:
        return ""

    def generate(self, prompt=None, *, system=None, **kw) -> str:
        prompt = prompt if prompt is not None else kw.get("prompt", "")
        marks = sorted(set(re.findall(r"MARK-\d+", prompt)))
        items = [
            {"text": m, "label": "note", "importance": 5, "emotions": {}} for m in marks
        ]
        return json.dumps(items)


def _load_old_pipeline_module() -> types.ModuleType:
    """Load the PRE-CHANGE brain/ingest/pipeline.py (commit c2154a97's parent,
    before finalize decoupling landed) as a live module, bound to the
    CURRENT brain.ingest.buffer (unchanged by this revision) via its own
    top-of-file imports. Used only for the H6 fail-demo below — never
    imported by, or used to modify, anything under brain/."""
    repo_root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        ["git", "show", "c2154a97^:brain/ingest/pipeline.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    mod = types.ModuleType("old_pipeline_pre_finalize_decoupling")
    exec(compile(result.stdout, "<old_pipeline_c2154a97^>", "exec"), mod.__dict__)
    return mod


# --------------------------------------------------------------------------- C10
def test_c10_finalize_no_delete_interleave(tmp_path: Path) -> None:
    """A finalize tick interleaved (here: run BEFORE) a rollover on the same
    session must NOT delete the buffer out from under the rollover's seed —
    only the rollover owns buffer deletion; finalize is extraction-only."""
    persona_dir = _persona(tmp_path)
    sid = "sess_c10"
    base = datetime.now(UTC) - timedelta(hours=3)
    _seed(persona_dir, sid, 6, base=base, step=timedelta(minutes=1))

    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    embeddings = EmbeddingCache(persona_dir / "embeddings.db", FakeEmbeddingProvider(dim=256))
    provider = _ExtractOnlyProvider()
    try:
        # Interleave: the finalize tick fires into the window BEFORE the
        # rollover's fold/seed (the allowed "before/around" interleaving).
        reports = finalize_stale_sessions(
            persona_dir,
            finalize_after_hours=0.0,
            store=store,
            hebbian=hebbian,
            provider=provider,
            embeddings=embeddings,
        )
        assert len(reports) == 1
        assert reports[0].committed == 1
        assert provider.extract_calls == 1
        # Extraction ran (a memory landed) AND the buffer was NOT deleted —
        # this is the exact assertion that fails against the pre-change
        # finalize (see test_c10_pre_change_finalize_deleted_buffer_fail_demo).
        assert sid in list_active_sessions(persona_dir)
        assert read_cursor(persona_dir, sid) is not None
        # TEMP (Root-2 consolidation-gate stopgap — see brain/memory/pending.py
        # and brain/monologue/ambient.py, which disagree on whether the gate
        # retires at "Phase 4" or "Phase 5"; revert to store.count() when the
        # dream cycle lands and retires the gate): extraction lands a gated
        # "note" candidate in the pending queue, not memories.db.
        pending_notes = PendingQueue(persona_dir).read_recent("note", limit=1000)
        assert len(pending_notes) == 1
        assert pending_notes[0].content == "stub extracted memory"

        # Now the rollover runs; its seed must survive the finalize that
        # already ran inside its window.
        new_sid = perform_rollover(
            persona_dir,
            sid,
            _PERSONA_NAME,
            seed_mode="summary_only",
            provider=provider,
            store=store,
            hebbian=hebbian,
            embeddings=embeddings,
        )
        assert new_sid is not None
        seed_turns = read_session(persona_dir, new_sid)
        assert seed_turns, "rollover seed must survive the interleaved finalize"
        assert seed_turns[0]["speaker"] == "summary"
        # Only the rollover deletes the OLD buffer — finalize alone (above)
        # left it in place.
        assert sid not in list_active_sessions(persona_dir)
    finally:
        store.close()
        hebbian.close()
        embeddings.close()


def test_c10_pre_change_finalize_deleted_buffer_fail_demo(tmp_path: Path) -> None:
    """H6: run the ACTUAL pre-change finalize_stale_sessions (commit
    c2154a97's parent, which deleted the buffer unconditionally on a clean
    pass) against an identical fixture and confirm it DOES delete — proving
    the C10 assertion above (buffer survives finalize) is a real regression
    guard, not a vacuous one."""
    persona_dir = _persona(tmp_path)
    sid = "sess_c10_fail_demo"
    base = datetime.now(UTC) - timedelta(hours=3)
    _seed(persona_dir, sid, 6, base=base, step=timedelta(minutes=1))

    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    embeddings = EmbeddingCache(persona_dir / "embeddings.db", FakeEmbeddingProvider(dim=256))
    provider = _ExtractOnlyProvider()
    try:
        old_pipeline = _load_old_pipeline_module()
        old_pipeline.finalize_stale_sessions(
            persona_dir,
            finalize_after_hours=0.0,
            store=store,
            hebbian=hebbian,
            provider=provider,
            embeddings=embeddings,
        )
        assert sid not in list_active_sessions(persona_dir), (
            "pre-change finalize_stale_sessions deleted the buffer on a clean "
            "pass — the C10 'buffer survives finalize' assertion would have "
            "failed against this code, confirming it is discriminating"
        )
    finally:
        store.close()
        hebbian.close()
        embeddings.close()


# --------------------------------------------------------------------------- C15
def test_c15_active_set_bounded_after_rollover(tmp_path: Path, monkeypatch) -> None:
    """A rolled-over old session's buffer is deleted by the rollover path, so
    list_active_sessions never accumulates rolled-over sids — neither across
    a chain of successive rollovers on one lineage, nor across many
    independent sessions each rolled over once."""
    persona_dir = _persona(tmp_path)
    base = datetime.now(UTC) - timedelta(hours=2)

    def _seed_with_cursor(sid: str) -> None:
        tss = _seed(persona_dir, sid, 4, base=base, step=timedelta(minutes=1))
        write_cursor(persona_dir, sid, tss[-1])

    # Chain 3 successive rollovers over the SAME lineage: sid0 -> sid1 -> sid2 -> sid3.
    sid0 = "sess_c15_chain0"
    _seed_with_cursor(sid0)
    chain = [sid0]
    for _ in range(3):
        new_sid = perform_rollover(persona_dir, chain[-1], _PERSONA_NAME, seed_mode="summary_only")
        assert new_sid is not None
        chain.append(new_sid)

    active = set(list_active_sessions(persona_dir))
    assert active == {chain[-1]}
    for old in chain[:-1]:
        assert old not in active
        assert not (persona_dir / "active_conversations" / f"{old}.jsonl").exists()

    # Independent sessions, each rolled over once, don't accumulate either.
    extra_olds: list[str] = []
    for i in range(3):
        esid = f"sess_c15_extra{i}"
        _seed_with_cursor(esid)
        extra_olds.append(esid)
        assert perform_rollover(persona_dir, esid, _PERSONA_NAME, seed_mode="summary_only") is not None

    active_after = set(list_active_sessions(persona_dir))
    assert active_after.isdisjoint(extra_olds)
    # Bounded by "sessions currently live", NOT by "sessions ever created":
    # chain[-1] + 3 fresh rollover targets = 4, regardless of the 4 rollovers
    # already performed above.
    assert len(active_after) == 4

    # H6 fail-demo: if the rollover path stopped deleting the old buffer, the
    # active set WOULD grow. Prove the "gone" assertions above are
    # discriminating by monkeypatching delete_session_buffer to a no-op for
    # one more rollover and observing the active set fail to shrink.
    import brain.chat.rollover as rollover_mod

    monkeypatch.setattr(rollover_mod, "delete_session_buffer", lambda *a, **kw: None)
    leaked_sid = "sess_c15_leak"
    _seed_with_cursor(leaked_sid)
    before = set(list_active_sessions(persona_dir))
    new_sid2 = perform_rollover(persona_dir, leaked_sid, _PERSONA_NAME, seed_mode="summary_only")
    assert new_sid2 is not None
    after = set(list_active_sessions(persona_dir))
    assert leaked_sid in after, (
        "with delete_session_buffer patched to a no-op, the old buffer WOULD "
        "persist in the active set — confirming the real assertions above "
        "(old sid gone / active-set bounded) are capable of catching a "
        "regression, not vacuously true"
    )
    assert len(after) > len(before)


# --------------------------------------------------------------------------- C18
def test_c18_carried_raw_tail_extraction_state(tmp_path: Path) -> None:
    """The new session's cursor carries the old session's extraction state so
    that, of the carried 40-message raw tail, already-extracted messages are
    NOT re-extracted (no duplicate memory) while not-yet-extracted messages
    still are."""
    persona_dir = _persona(tmp_path)
    sid = "sess_c18"
    # Recent turns (well under 24h old) so the cascade fold never removes
    # anything — this test isolates the seed's carried-tail + cursor-carry
    # behavior, not the age-gated fold (that's C2/C14's job).
    base = datetime.now(UTC) - timedelta(hours=2)
    n = 50
    tss = _seed(persona_dir, sid, n, base=base, step=timedelta(minutes=1), marker="MARK")

    # Cursor sits at turn 25: turns 0..25 (26 turns) are already-extracted;
    # turns 26..49 (24 turns) are not.
    cursor_ts = tss[25]
    write_cursor(persona_dir, sid, cursor_ts)

    new_sid = perform_rollover(persona_dir, sid, _PERSONA_NAME, seed_mode="tiers_plus_tail")
    assert new_sid is not None
    # The carried cursor is what makes the guard below work.
    assert read_cursor(persona_dir, new_sid) == cursor_ts

    new_turns = read_session(persona_dir, new_sid)
    raw_new = [t for t in new_turns if t.get("speaker") != "summary"]
    assert len(raw_new) == 40
    # The carried tail is raw[-40:] of the original buffer == turns 10..49.
    carried_texts = {t["text"] for t in raw_new}
    assert carried_texts == {f"MARK-{i}" for i in range(10, 50)}

    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    embeddings = EmbeddingCache(persona_dir / "embeddings.db", FakeEmbeddingProvider(dim=256))
    provider = _MarkerExtractProvider()
    try:
        report = extract_session_snapshot(
            persona_dir,
            new_sid,
            store=store,
            hebbian=hebbian,
            provider=provider,
            embeddings=embeddings,
        )
        assert report.errors == 0
        # TEMP (Root-2 consolidation-gate stopgap — see brain/memory/pending.py
        # and brain/monologue/ambient.py, which disagree on whether the gate
        # retires at "Phase 4" or "Phase 5"; revert to store.list_active() when
        # the dream cycle lands and retires the gate): extraction lands gated
        # "note" candidates in the pending queue, not memories.db. limit=1000
        # (well above the true count) — read_recent's limit has no default and
        # returns only the newest `limit` entries, so a small limit (e.g. the
        # unrelated _AMBIENT_LIMIT=5) would silently drop the oldest entries.
        committed_texts = {
            m.content for m in PendingQueue(persona_dir).read_recent("note", limit=1000)
        }
        # Already-extracted carried turns (10..25) must NOT be re-extracted;
        # not-yet-extracted carried turns (26..49) must be.
        already_extracted = {f"MARK-{i}" for i in range(10, 26)}
        not_yet_extracted = {f"MARK-{i}" for i in range(26, 50)}
        assert committed_texts == not_yet_extracted
        assert committed_texts.isdisjoint(already_extracted)
    finally:
        store.close()
        hebbian.close()
        embeddings.close()


def test_c18_without_carried_cursor_would_reextract_fail_demo(tmp_path: Path) -> None:
    """H6: seed the SAME carried tail but WITHOUT carrying the cursor
    (simulating the pre-fix gap) and confirm the already-extracted carried
    turns DO get re-extracted (duplicate memories) — proving the carried-
    cursor guard above is load-bearing, not incidental."""
    persona_dir = _persona(tmp_path)
    sid = "sess_c18_nocursor"
    base = datetime.now(UTC) - timedelta(hours=2)
    n = 50
    tss = _seed(persona_dir, sid, n, base=base, step=timedelta(minutes=1), marker="MARK")
    cursor_ts = tss[25]
    write_cursor(persona_dir, sid, cursor_ts)

    new_sid = perform_rollover(persona_dir, sid, _PERSONA_NAME, seed_mode="tiers_plus_tail")
    assert new_sid is not None
    assert read_cursor(persona_dir, new_sid) == cursor_ts

    # Simulate "seed without carrying the cursor" (the pre-fix gap the
    # criterion's fail-demo names) by dropping the new session's cursor.
    from brain.ingest.buffer import delete_cursor

    delete_cursor(persona_dir, new_sid)
    assert read_cursor(persona_dir, new_sid) is None

    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    embeddings = EmbeddingCache(persona_dir / "embeddings.db", FakeEmbeddingProvider(dim=256))
    provider = _MarkerExtractProvider()
    try:
        report = extract_session_snapshot(
            persona_dir,
            new_sid,
            store=store,
            hebbian=hebbian,
            provider=provider,
            embeddings=embeddings,
        )
        assert report.errors == 0
        # TEMP (Root-2 consolidation-gate stopgap — see brain/memory/pending.py
        # and brain/monologue/ambient.py, which disagree on whether the gate
        # retires at "Phase 4" or "Phase 5"; revert to store.list_active() when
        # the dream cycle lands and retires the gate): extraction lands gated
        # "note" candidates in the pending queue, not memories.db. limit=1000
        # (well above the true count) — read_recent's limit has no default and
        # returns only the newest `limit` entries, so a small limit (e.g. the
        # unrelated _AMBIENT_LIMIT=5) would silently drop the oldest entries.
        committed_texts = {
            m.content for m in PendingQueue(persona_dir).read_recent("note", limit=1000)
        }
        already_extracted = {f"MARK-{i}" for i in range(10, 26)}
        # Without the carried cursor, extraction starts from scratch and
        # re-extracts the already-extracted carried turns too.
        assert already_extracted.issubset(committed_texts), (
            "dropping the carried cursor should cause re-extraction of the "
            "already-extracted carried tail turns — if this fails, the "
            "no-cursor fixture no longer demonstrates the gap the real "
            "carried-cursor behavior closes"
        )
    finally:
        store.close()
        hebbian.close()
        embeddings.close()


def test_c1_mid_rollover_window_redirects_not_resurrect(tmp_path: Path) -> None:
    """C-1 regression (stage-6 concurrency): during the mid-rollover window — the
    ``rolled_to`` pointer is already written (the rollover writes it FIRST), but the
    old sid is still cached in ``_SESSIONS`` (registry not yet evicted) AND its
    buffer is not yet deleted — a request for the old sid MUST resolve to the
    successor, never the stale-cached old session.

    Pre-fix, ``get_or_hydrate_session`` short-circuited on the ``_SESSIONS`` cache
    hit BEFORE consulting the pointer, so it returned the old session; the caller
    then appended via ``ingest_turn`` (append-creates the file), resurrecting the
    deleted old buffer and orphaning the turn. The assertion
    ``resolved.session_id == new_sid`` fails against that pre-fix behavior — this is
    the shown-able-to-fail regression guard. The whole C1–C22 suite missed this
    race because none of them held a stale cache entry across the redirect.
    """
    from brain.chat.session import (
        create_session,
        get_or_hydrate_session,
        get_session,
        reset_registry,
    )
    from brain.ingest.buffer import read_session, write_rolled_to

    persona = _persona(tmp_path)
    reset_registry()
    try:
        # Old session: live, cached in _SESSIONS, with a real buffer.
        old = create_session(_PERSONA_NAME)
        old_sid = old.session_id
        ingest_turn(persona, {"session_id": old_sid, "speaker": "user", "text": "hi"})
        ingest_turn(
            persona, {"session_id": old_sid, "speaker": "assistant", "text": "yo"}
        )

        # Successor session: seeded and present on disk.
        new = create_session(_PERSONA_NAME)
        new_sid = new.session_id
        ingest_turn(
            persona, {"session_id": new_sid, "speaker": "summary", "text": "[seed]"}
        )

        # --- Enter the mid-rollover window ---
        # Pointer down (written first by the rollover); registry NOT yet evicted
        # (old still cached) and the old buffer NOT yet deleted.
        write_rolled_to(persona, old_sid, new_sid)
        assert get_session(old_sid) is not None, "precondition: old still cached"
        assert read_session(persona, old_sid), "precondition: old buffer still present"

        # A request arriving now must resolve to the successor, not the stale cache.
        resolved = get_or_hydrate_session(persona, _PERSONA_NAME, old_sid)
        assert resolved is not None
        assert resolved.session_id == new_sid, (
            "mid-rollover request resolved to the stale-cached OLD session — "
            "the C-1 race (resurrectable old buffer) is not closed"
        )

        # A turn on the resolved session lands in the successor buffer, and the old
        # buffer is never re-grown beyond its two pre-window turns.
        ingest_turn(
            persona,
            {"session_id": resolved.session_id, "speaker": "user", "text": "cont"},
        )
        assert any(t.get("text") == "cont" for t in read_session(persona, new_sid))
        assert len(read_session(persona, old_sid)) == 2, "old buffer must not grow"
    finally:
        reset_registry()


# ------------------------------------------------------- resolve-persist race (r4)
#
# Stage-6 round-3 (6-redteam-code-r3.md) found a residual TOCTOU: the idle-gate
# checks ``is_session_busy`` once before ``perform_rollover``, but the rollover then
# spends multiple seconds in extract+cascade before deleting the old buffer. A live
# request that resolves the OLD sid and persists inside that window used to orphan
# its turn — ``ingest_turn`` append-creates (resurrects) the just-deleted buffer.
#
# The round-4 fix makes the destructive section (seed re-read → rolled_to pointer →
# registry evict) hold ``session.registry_lock()``, the SAME lock the live persist
# (``session.persist_turns_following_successor``) holds, and routes the engine's
# persist through that redirect. These three tests pin the fix and demonstrate the
# pre-fix failure.


def test_persist_after_rollover_redirects_to_successor(tmp_path: Path) -> None:
    """A live-turn persist for an already-rolled-over sid threads into the successor
    (via the ``rolled_to`` redirect) instead of resurrecting the deleted old buffer.
    This is the mechanism that closes the resolve-persist resurrection Major."""
    from brain.chat.session import persist_turns_following_successor, reset_registry

    reset_registry()
    persona_dir = _persona(tmp_path)
    old_sid = "sess_redir"
    base = datetime.now(UTC) - timedelta(hours=3)
    _seed(persona_dir, old_sid, 6, base=base, step=timedelta(minutes=1))

    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    embeddings = EmbeddingCache(persona_dir / "embeddings.db", FakeEmbeddingProvider(dim=256))
    provider = _ExtractOnlyProvider()
    try:
        new_sid = perform_rollover(
            persona_dir, old_sid, _PERSONA_NAME,
            seed_mode="summary_only", provider=provider,
            store=store, hebbian=hebbian, embeddings=embeddings,
        )
        assert new_sid is not None
        assert old_sid not in list_active_sessions(persona_dir)

        # A request that still holds the OLD sid persists its turn now.
        persist_turns_following_successor(persona_dir, [
            {"session_id": old_sid, "speaker": "user", "text": "late-user"},
            {"session_id": old_sid, "speaker": "assistant", "text": "late-asst"},
        ])

        # It landed in the successor, NOT on a resurrected old buffer.
        assert old_sid not in list_active_sessions(persona_dir), (
            "old buffer was resurrected — the persist redirect did not fire"
        )
        succ_texts = [r.get("text") for r in read_session(persona_dir, new_sid)]
        assert "late-user" in succ_texts and "late-asst" in succ_texts
    finally:
        store.close()
        hebbian.close()
        embeddings.close()
        reset_registry()


def test_plain_ingest_after_rollover_resurrects_fail_demo(tmp_path: Path) -> None:
    """Fail-demo: the PRE-FIX persist path (a plain ``ingest_turn`` on the old sid,
    with no successor redirect — exactly what engine._persist_turn did before r4)
    resurrects the deleted old buffer and orphans the turn OUTSIDE the successor
    chain. Proves the redirect guard above is a real regression guard, not vacuous."""
    from brain.chat.session import reset_registry
    from brain.ingest.buffer import ingest_turn

    reset_registry()
    persona_dir = _persona(tmp_path)
    old_sid = "sess_resurrect"
    base = datetime.now(UTC) - timedelta(hours=3)
    _seed(persona_dir, old_sid, 6, base=base, step=timedelta(minutes=1))

    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    embeddings = EmbeddingCache(persona_dir / "embeddings.db", FakeEmbeddingProvider(dim=256))
    provider = _ExtractOnlyProvider()
    try:
        new_sid = perform_rollover(
            persona_dir, old_sid, _PERSONA_NAME,
            seed_mode="summary_only", provider=provider,
            store=store, hebbian=hebbian, embeddings=embeddings,
        )
        assert new_sid is not None
        assert old_sid not in list_active_sessions(persona_dir)

        # The OLD (unredirected) persist path — append straight to the old sid.
        ingest_turn(persona_dir, {"session_id": old_sid, "speaker": "user", "text": "orphan"})

        # Bug reproduced: the old buffer is resurrected and the turn is orphaned —
        # it is NOT in the successor. (The fixed path routes through
        # persist_turns_following_successor, which redirects; see the test above.)
        assert old_sid in list_active_sessions(persona_dir), (
            "expected the unredirected ingest to resurrect the old buffer"
        )
        succ_texts = [r.get("text") for r in read_session(persona_dir, new_sid)]
        assert "orphan" not in succ_texts
    finally:
        store.close()
        hebbian.close()
        embeddings.close()
        reset_registry()


def test_persist_during_rollover_destructive_window_not_orphaned(
    tmp_path: Path, monkeypatch
) -> None:
    """The exact r3 window, exercised concurrently: a live persist that registers
    for the OLD sid WHILE ``perform_rollover`` is mid-destructive-section (pointer
    written, buffer not yet deleted) must NOT orphan its turn — it threads into the
    successor. The rollover runs on one thread, blocked at the buffer-delete step;
    the racing persist runs on the main thread in that window.

    Without the round-4 fix the racing persist (a) is not serialized against the
    delete and (b) does not redirect, so its turn lands on the old buffer and is
    deleted with it (or resurrects it). With the fix it redirects to the live
    successor under registry_lock()."""
    import threading

    from brain.chat import rollover as rollover_mod
    from brain.chat.session import persist_turns_following_successor, reset_registry

    reset_registry()
    persona_dir = _persona(tmp_path)
    old_sid = "sess_window"
    base = datetime.now(UTC) - timedelta(hours=3)
    _seed(persona_dir, old_sid, 6, base=base, step=timedelta(minutes=1))

    at_delete = threading.Event()
    release_delete = threading.Event()
    real_delete = rollover_mod.delete_session_buffer

    def blocking_delete(pd, sid):
        # perform_rollover has finished its destructive section (rolled_to pointer
        # written, registry evicted) and is about to delete the old buffer. Pause
        # here so the racing persist executes in exactly this window.
        at_delete.set()
        release_delete.wait(5)
        real_delete(pd, sid)

    monkeypatch.setattr(rollover_mod, "delete_session_buffer", blocking_delete)

    # store/hebbian/embeddings are omitted: the best-effort memory extraction is
    # irrelevant to the buffer-lifecycle race under test, and its SQLite handles
    # can't cross the thread boundary. The fold + destructive section still run.
    provider = _ExtractOnlyProvider()
    result: dict = {}

    def run_rollover() -> None:
        # tiers_plus_tail is the real weekly-rollover (1c-B) mode r3's scenario
        # targets; it seeds from the raw tail, so it reaches the destructive delete
        # without depending on a memory-store-backed summary.
        result["new_sid"] = perform_rollover(
            persona_dir, old_sid, _PERSONA_NAME,
            seed_mode="tiers_plus_tail", provider=provider,
        )

    t = threading.Thread(target=run_rollover, name="rollover-under-test")
    try:
        t.start()
        assert at_delete.wait(5), "rollover never reached the buffer-delete step"

        # Racing live persist for the OLD sid, exactly as engine._persist_turn does.
        persist_turns_following_successor(persona_dir, [
            {"session_id": old_sid, "speaker": "user", "text": "raced-user"},
            {"session_id": old_sid, "speaker": "assistant", "text": "raced-asst"},
        ])

        release_delete.set()
        t.join(5)
        assert not t.is_alive(), "rollover thread did not finish"
    finally:
        release_delete.set()
        t.join(5)
        reset_registry()

    new_sid = result.get("new_sid")
    assert new_sid is not None
    # No resurrection / no orphan: old buffer gone, raced turn in the successor.
    assert old_sid not in list_active_sessions(persona_dir), (
        "old buffer was resurrected by the racing persist"
    )
    succ_texts = [r.get("text") for r in read_session(persona_dir, new_sid)]
    assert "raced-user" in succ_texts and "raced-asst" in succ_texts, (
        "racing persist's turn was orphaned instead of threading into the successor"
    )


def test_summary_only_carries_residual_raw_not_dropped(
    tmp_path: Path, monkeypatch
) -> None:
    """Round-4 regression (6-redteam-code-r4.md): a ``summary_only`` rollover must
    carry raw turns still present after the fold into the successor. Pre-fix the seed
    was ``[summary]`` only, so a fresh turn that raced into the old buffer after the
    fold (or a turn the cursor-guarded fold left unfolded) was DROPPED — deleted with
    the old buffer, never folded, never archived: permanent data loss.

    The fold is stubbed to a no-op and the buffer is pre-seeded with a summary plus
    two residual raw turns (the post-fold-with-race state). Without the fix the two
    raw turns never reach the successor."""
    from brain.chat import rollover as rollover_mod
    from brain.chat.session import reset_registry
    from brain.ingest.buffer import ingest_turn

    reset_registry()
    persona_dir = _persona(tmp_path)
    old_sid = "sess_sumonly"
    # Fold no-op: the buffer already reflects a post-fold state with residual raw.
    monkeypatch.setattr(rollover_mod, "compact_conversation", lambda *a, **k: None)
    ingest_turn(persona_dir, {"session_id": old_sid, "speaker": "summary", "text": "[folded]"})
    ingest_turn(persona_dir, {"session_id": old_sid, "speaker": "user", "text": "fresh-user"})
    ingest_turn(persona_dir, {"session_id": old_sid, "speaker": "assistant", "text": "fresh-asst"})
    try:
        new_sid = perform_rollover(
            persona_dir, old_sid, _PERSONA_NAME, seed_mode="summary_only", provider=None,
        )
        assert new_sid is not None
        texts = [r.get("text") for r in read_session(persona_dir, new_sid)]
        assert "[folded]" in texts, "the folded summary must be carried"
        assert "fresh-user" in texts and "fresh-asst" in texts, (
            "residual raw turns were dropped — summary_only rollover data loss (r4)"
        )
        # The summary still leads the seed (order preserved).
        assert texts[0] == "[folded]"
        assert old_sid not in list_active_sessions(persona_dir)
    finally:
        reset_registry()


def test_persist_during_summary_only_rollover_window_not_orphaned(
    tmp_path: Path, monkeypatch
) -> None:
    """Concurrent sibling of the tiers_plus_tail window test, for summary_only mode
    (closes the r4 re-review's noted coverage gap). A live persist that registers in
    the destructive window (pointer written, buffer not yet deleted) of a
    summary_only rollover threads into the successor, not orphaned. The fold is a
    no-op over a pre-seeded summary so the rollover proceeds without needing the
    memory stores across the thread boundary."""
    import threading

    from brain.chat import rollover as rollover_mod
    from brain.chat.session import persist_turns_following_successor, reset_registry
    from brain.ingest.buffer import ingest_turn

    reset_registry()
    persona_dir = _persona(tmp_path)
    old_sid = "sess_sowin"
    monkeypatch.setattr(rollover_mod, "compact_conversation", lambda *a, **k: None)
    ingest_turn(persona_dir, {"session_id": old_sid, "speaker": "summary", "text": "[folded]"})

    at_delete = threading.Event()
    release_delete = threading.Event()
    real_delete = rollover_mod.delete_session_buffer

    def blocking_delete(pd, sid):
        at_delete.set()
        release_delete.wait(5)
        real_delete(pd, sid)

    monkeypatch.setattr(rollover_mod, "delete_session_buffer", blocking_delete)

    result: dict = {}

    def run_rollover() -> None:
        result["new_sid"] = perform_rollover(
            persona_dir, old_sid, _PERSONA_NAME, seed_mode="summary_only", provider=None,
        )

    t = threading.Thread(target=run_rollover, name="summary-only-rollover")
    try:
        t.start()
        assert at_delete.wait(5), "rollover never reached the buffer-delete step"
        persist_turns_following_successor(persona_dir, [
            {"session_id": old_sid, "speaker": "user", "text": "so-raced-user"},
            {"session_id": old_sid, "speaker": "assistant", "text": "so-raced-asst"},
        ])
        release_delete.set()
        t.join(5)
        assert not t.is_alive(), "rollover thread did not finish"
    finally:
        release_delete.set()
        t.join(5)
        reset_registry()

    new_sid = result.get("new_sid")
    assert new_sid is not None
    assert old_sid not in list_active_sessions(persona_dir), "old buffer resurrected"
    texts = [r.get("text") for r in read_session(persona_dir, new_sid)]
    assert "so-raced-user" in texts and "so-raced-asst" in texts, (
        "racing persist orphaned instead of threading into the successor"
    )
