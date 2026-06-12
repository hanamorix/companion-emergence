"""Tests for brain.health.soul_candidate_repair — one-time startup migration.

TDD: one test at a time, per tdd-guard policy.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

from brain.memory.store import Memory, MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "memories.db"), integrity_check=False)


def _write_candidates(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "soul_candidates.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _read_candidates(tmp_path: Path) -> list[dict]:
    path = tmp_path / "soul_candidates.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _monologue_pending(memory_id: str, text: str = "Ordinary trust") -> dict:
    return {
        "memory_id": memory_id,
        "text": text,
        "label": "monologue_soul_candidate",
        "importance": 8,
        "session_id": "monologue",
        "queued_at": "2026-06-01T10:00:00+00:00",
        "status": "auto_pending",
        "defer_count": 5,
        "last_deferred_at": "2026-06-10T08:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Test 1: stuck fragment with memory present → text backfilled, counters reset
# ---------------------------------------------------------------------------

def test_repair_backfills_context_from_memory(tmp_path: Path) -> None:
    """Stuck monologue fragment with a real memory → text combines old+content, counters reset."""
    from brain.health.soul_candidate_repair import run_soul_candidate_repair

    store = _make_store(tmp_path)
    try:
        mem = Memory.create_new(
            content="Ordinary trust — she let the goodnight be ordinary",
            memory_type="monologue_soul_candidate",
            domain="interior",
            emotions={},
        )
        store.create(mem)

        _write_candidates(tmp_path, [_monologue_pending(mem.id, text="Ordinary trust")])

        report = run_soul_candidate_repair(tmp_path, store=store)
    finally:
        store.close()

    assert report.repaired == 1
    assert report.expired == 0
    assert report.status == "complete"

    records = _read_candidates(tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["status"] == "auto_pending"
    assert rec["defer_count"] == 0
    assert "last_deferred_at" not in rec
    assert "Ordinary trust — she let the goodnight be ordinary" in rec["text"]
    # Must not double-combine
    assert rec["text"].count("Ordinary trust") == 1


def test_repair_uses_memory_content_directly_when_it_already_contains_old_text(
    tmp_path: Path,
) -> None:
    """Post-S1 placeholder memories already hold the combined string.
    When memory content already contains the old text, use it directly.
    """
    from brain.health.soul_candidate_repair import run_soul_candidate_repair

    store = _make_store(tmp_path)
    try:
        combined = "Ordinary trust — she let the goodnight be ordinary"
        mem = Memory.create_new(
            content=combined,
            memory_type="monologue_soul_candidate",
            domain="interior",
            emotions={},
        )
        store.create(mem)

        # Candidate text already equals the combined string (post-fix memory)
        _write_candidates(tmp_path, [_monologue_pending(mem.id, text=combined)])

        report = run_soul_candidate_repair(tmp_path, store=store)
    finally:
        store.close()

    records = _read_candidates(tmp_path)
    rec = records[0]
    # Should use memory content directly, no double-combine
    assert rec["text"] == combined
    assert rec["text"].count("Ordinary trust") == 1
    assert report.repaired == 1


# ---------------------------------------------------------------------------
# Test 3: memory missing → expired
# ---------------------------------------------------------------------------

def test_repair_expires_when_memory_missing(tmp_path: Path) -> None:
    """memory_id not in store → status='expired' with reason + expired_at set."""
    from brain.health.soul_candidate_repair import run_soul_candidate_repair

    store = _make_store(tmp_path)
    try:
        _write_candidates(tmp_path, [_monologue_pending("nonexistent-id-abc")])
        report = run_soul_candidate_repair(tmp_path, store=store)
    finally:
        store.close()

    assert report.repaired == 0
    assert report.expired == 1
    assert report.status == "complete"

    records = _read_candidates(tmp_path)
    rec = records[0]
    assert rec["status"] == "expired"
    assert "pre-fix fragment" in rec["reason"]
    assert "expired_at" in rec


def test_repair_expires_when_memory_content_empty(tmp_path: Path) -> None:
    """memory_id exists but has empty content → expired."""
    from brain.health.soul_candidate_repair import run_soul_candidate_repair

    store = _make_store(tmp_path)
    try:
        mem = Memory.create_new(
            content="   ",
            memory_type="monologue_soul_candidate",
            domain="interior",
            emotions={},
        )
        store.create(mem)

        _write_candidates(tmp_path, [_monologue_pending(mem.id)])
        report = run_soul_candidate_repair(tmp_path, store=store)
    finally:
        store.close()

    records = _read_candidates(tmp_path)
    rec = records[0]
    assert rec["status"] == "expired"
    assert report.expired == 1  # empty content


# ---------------------------------------------------------------------------
# Test 5: non-targets left untouched
# ---------------------------------------------------------------------------

def test_repair_leaves_non_targets_untouched(tmp_path: Path) -> None:
    """ingest-sourced auto_pending + accepted + rejected → byte-identical after repair."""
    from brain.health.soul_candidate_repair import run_soul_candidate_repair

    def _ingest_pending(mid: str) -> dict:
        return {
            "memory_id": mid,
            "text": "She trusted the night",
            "label": "conversation",
            "importance": 9,
            "session_id": "sess-abc123",
            "queued_at": "2026-06-01T11:00:00+00:00",
            "status": "auto_pending",
        }

    def _accepted(mid: str) -> dict:
        return {
            "memory_id": mid,
            "text": "Something crystallised",
            "label": "monologue_soul_candidate",
            "importance": 8,
            "session_id": "monologue",
            "queued_at": "2026-06-01T09:00:00+00:00",
            "status": "accepted",
        }

    def _rejected(mid: str) -> dict:
        return {
            "memory_id": mid,
            "text": "Rejected fragment",
            "label": "monologue_soul_candidate",
            "importance": 8,
            "session_id": "monologue",
            "queued_at": "2026-06-01T09:00:00+00:00",
            "status": "rejected",
        }

    store = _make_store(tmp_path)
    try:
        original_records = [
            _ingest_pending("mem-ingest-1"),
            _accepted("mem-accepted-1"),
            _rejected("mem-rejected-1"),
        ]
        _write_candidates(tmp_path, original_records)

        report = run_soul_candidate_repair(tmp_path, store=store)
    finally:
        store.close()

    assert report.repaired == 0
    assert report.expired == 0

    after = _read_candidates(tmp_path)
    assert len(after) == 3
    by_id = {r["memory_id"]: r for r in after}
    assert by_id["mem-ingest-1"]["status"] == "auto_pending"
    assert by_id["mem-ingest-1"]["text"] == "She trusted the night"
    assert by_id["mem-accepted-1"]["status"] == "accepted"
    assert by_id["mem-rejected-1"]["status"] == "rejected"


# ---------------------------------------------------------------------------
# Test 6: idempotent
# ---------------------------------------------------------------------------

def test_repair_idempotent(tmp_path: Path) -> None:
    """Second run is a no-op: state marker present, file mtime unchanged."""
    from brain.health.soul_candidate_repair import (
        run_soul_candidate_repair,
        should_run_soul_candidate_repair,
    )

    store = _make_store(tmp_path)
    try:
        mem = Memory.create_new(
            content="Ordinary trust — she let the goodnight be ordinary",
            memory_type="monologue_soul_candidate",
            domain="interior",
            emotions={},
        )
        store.create(mem)
        _write_candidates(tmp_path, [_monologue_pending(mem.id)])

        report1 = run_soul_candidate_repair(tmp_path, store=store)
        assert report1.status == "complete"
        assert report1.repaired == 1

        state_path = tmp_path / "soul_candidate_repair_state.json"
        assert state_path.exists()
        state_data = json.loads(state_path.read_text(encoding="utf-8"))
        assert state_data["status"] == "complete"
        assert state_data["repaired"] == 1

        assert should_run_soul_candidate_repair(tmp_path) is False

        candidates_path = tmp_path / "soul_candidates.jsonl"
        mtime_before = candidates_path.stat().st_mtime

        report2 = run_soul_candidate_repair(tmp_path, store=store)
        assert report2.status == "complete"
        assert report2.repaired == 1  # from state, not re-run
        assert candidates_path.stat().st_mtime == mtime_before
    finally:
        store.close()


def test_should_run_false_when_no_candidates_file(tmp_path: Path) -> None:
    """No soul_candidates.jsonl → should_run False."""
    from brain.health.soul_candidate_repair import should_run_soul_candidate_repair
    assert should_run_soul_candidate_repair(tmp_path) is False


def test_should_run_false_when_no_targets(tmp_path: Path) -> None:
    """Candidates file exists but no monologue auto_pending → should_run False."""
    from brain.health.soul_candidate_repair import should_run_soul_candidate_repair
    records = [
        {"memory_id": "m1", "text": "x", "session_id": "sess-1", "status": "auto_pending"},
        {"memory_id": "m2", "text": "y", "session_id": "monologue", "status": "accepted"},
    ]
    _write_candidates(tmp_path, records)
    assert should_run_soul_candidate_repair(tmp_path) is False


def test_should_run_true_when_targets_present(tmp_path: Path) -> None:
    """Monologue auto_pending present + no state file → should_run True."""
    from brain.health.soul_candidate_repair import should_run_soul_candidate_repair
    _write_candidates(tmp_path, [_monologue_pending("m1")])
    assert should_run_soul_candidate_repair(tmp_path) is True


def test_should_run_false_when_state_complete(tmp_path: Path) -> None:
    """Complete state file → should_run False regardless of candidates."""
    from brain.health.soul_candidate_repair import should_run_soul_candidate_repair
    _write_candidates(tmp_path, [_monologue_pending("m1")])
    state = {"status": "complete", "repaired": 1, "expired": 0, "completed_at": "2026-06-12T00:00:00Z"}
    (tmp_path / "soul_candidate_repair_state.json").write_text(json.dumps(state), encoding="utf-8")
    assert should_run_soul_candidate_repair(tmp_path) is False


# ---------------------------------------------------------------------------
# Test 7: through-supervisor-startup path
# ---------------------------------------------------------------------------

def test_repair_runs_from_supervisor_startup_once(tmp_path: Path) -> None:
    """The supervisor startup block calls should_run + run exactly once.

    Patch should_run to True, patch run to record the call, spin run_folded
    with stop immediately set — mirrors the vocab_repair seam pattern.
    """
    from brain.bridge.events import EventBus
    from brain.bridge.provider import FakeProvider
    from brain.bridge.supervisor import run_folded

    persona_dir = tmp_path / "test-persona"
    persona_dir.mkdir()
    (persona_dir / "active_conversations").mkdir()
    (persona_dir / "persona_config.json").write_text(
        '{"provider": "fake", "searcher": "noop"}'
    )

    called: list[dict] = []

    def fake_should_run(pd: Path) -> bool:
        return True

    def fake_run(pd: Path, *, store: object) -> object:
        called.append({"persona_dir": pd})

        class _FakeReport:
            status = "complete"
            repaired = 0
            expired = 0
            completed_at = "2026-06-12T00:00:00Z"

        return _FakeReport()

    stop = threading.Event()
    stop.set()

    with (
        patch(
            "brain.bridge.supervisor._soul_candidate_repair_should_run",
            side_effect=fake_should_run,
        ),
        patch(
            "brain.bridge.supervisor._soul_candidate_repair_run",
            side_effect=fake_run,
        ),
    ):
        run_folded(
            stop,
            persona_dir=persona_dir,
            provider=FakeProvider(),
            event_bus=EventBus(),
            tick_interval_s=0.1,
            heartbeat_interval_s=None,
        )

    assert len(called) == 1, "soul_candidate_repair must be called exactly once at startup"
    assert called[0]["persona_dir"] == persona_dir


# ---------------------------------------------------------------------------
# Test 8: rewrite goes through file_lock
# ---------------------------------------------------------------------------

def test_repair_uses_file_lock_for_rewrite(tmp_path: Path) -> None:
    """The read-modify-rewrite cycle must go through the soul_candidates.jsonl file_lock."""
    from brain.health.soul_candidate_repair import run_soul_candidate_repair
    from brain.utils.file_lock import file_lock as real_file_lock

    store = _make_store(tmp_path)
    try:
        mem = Memory.create_new(
            content="Ordinary trust — she let the goodnight be ordinary",
            memory_type="monologue_soul_candidate",
            domain="interior",
            emotions={},
        )
        store.create(mem)
        _write_candidates(tmp_path, [_monologue_pending(mem.id)])

        lock_paths: list[Path] = []

        def recording_lock(path: Path):
            lock_paths.append(path)
            return real_file_lock(path)

        with patch("brain.health.soul_candidate_repair.file_lock", side_effect=recording_lock):
            run_soul_candidate_repair(tmp_path, store=store)
    finally:
        store.close()

    candidates_path = tmp_path / "soul_candidates.jsonl"
    assert any(p == candidates_path for p in lock_paths), (
        "soul_candidate_repair must acquire file_lock on soul_candidates.jsonl"
    )
