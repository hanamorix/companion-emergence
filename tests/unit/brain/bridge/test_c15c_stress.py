"""ram-spike-fix INC-10 — C15(c): heartbeat passes and gated-job items
driven for >= 60s on F-bob20k with concurrent chat-path writes from another
thread -> zero "database is locked" (2-plan §4.3/§4.2's whole point: the
30s busy timeout + batched decay removes the O16 starvation pattern even
under real, item-level, multi-minute background activity).

Stage-6 red-team MAJOR, addressed: C15(a)/(b) (job-granularity ordering and
heartbeat/job mutual exclusion) were proven at INC-9; this file proves
C15(c) specifically — the ITEM-level interleaving this increment (INC-10)
introduces, under real concurrent chat-path writes, for the full 60s+ the
criterion names.

Local Linux only, real F-bob20k fixture (~20k rows, 107MB, not committed) —
skips gracefully (not a failure) if the fixture isn't present on this
machine, mirroring test_search_via_bridge.py's convention. Runs `run_folded`
in a background thread for >= 60s wall-clock with the heartbeat and the
maintenance (forgetting) job firing continuously, while a SEPARATE thread
hammers `store.bump_recall` on a second connection to the SAME memories.db
the whole time — the real chat-path write this criterion means by
"concurrent chat-path writes."
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

import pytest

_FIXTURE = (
    # This file lives at .../.claude/worktrees/ram-spike-fix/tests/unit/brain/bridge/;
    # parents[4] is the ram-spike-fix worktree root, parents[5] is the shared
    # .claude/worktrees/ directory that also holds the dragonfly-ram-spike
    # worktree (a sibling worktree, not nested under ram-spike-fix).
    Path(__file__).resolve().parents[5]
    / "dragonfly-ram-spike"
    / "changes"
    / "dragonfly-ram-spike"
    / "persona"
    / "Bob"
    / "memories.db"
)

_STRESS_DURATION_S = 60.0


class _LockErrorCapture(logging.Handler):
    """Captures any log record whose message OR attached exception mentions
    a database lock — the observable signature of the O16 starvation
    pattern this fix removes, across every logger in the process
    (heartbeat, forgetting, store, supervisor).

    Round-2 red-team MAJOR, fixed: `record.getMessage()` alone is NOT
    enough — `brain/bridge/supervisor.py`'s heartbeat/forgetting call sites
    catch with a bare `except Exception: logger.exception("<generic
    label>")`, which never interpolates the caught exception's own text
    into the message (verified: `LogRecord.getMessage()` returns only the
    literal format string, never `exc_info`/`exc_text`). A real "database
    is locked" raised on either of those paths would previously go
    completely undetected by this oracle. `logger.exception(...)` attaches
    `exc_info` to the record; format it (via `self.format`, which renders
    the traceback + `repr(exception)` through the handler's own formatter)
    and scan THAT too, not just the bare message.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.hits: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        haystacks = [record.getMessage()]
        if record.exc_info is not None:
            # self.format() renders the message + the formatted traceback
            # (which includes str(exception), e.g. "OperationalError:
            # database is locked") through this handler's own formatter.
            haystacks.append(self.format(record))
        text = " ".join(haystacks).lower()
        if "database is locked" in text or ("locked" in text and "database" in text):
            # Round-3 red-team MINOR, fixed: record the actual matched text
            # (including the exception's own "database is locked" message
            # when that's what fired), not just the generic log label --
            # a future failure message should be debuggable at a glance.
            self.hits.append(text if len(haystacks) > 1 else record.getMessage())


@pytest.mark.skipif(not _FIXTURE.exists(), reason=f"F-bob20k fixture not present at {_FIXTURE}")
def test_c15c_heartbeat_and_gated_items_60s_with_concurrent_writes_no_lock_errors(
    tmp_path: Path,
) -> None:
    import brain.bridge.provider as provider_mod
    from brain.bridge.events import EventBus
    from brain.bridge.supervisor import run_folded
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "active_conversations").mkdir()
    # I11: copy into tmp_path, never open the fixture itself for write.
    shutil.copy(_FIXTURE, persona_dir / "memories.db")

    lock_capture = _LockErrorCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(lock_capture)

    write_errors: list[Exception] = []
    stop_writer = threading.Event()

    def _concurrent_chat_writer() -> None:
        """A real chat-path write on a SECOND connection to the same
        memories.db, hammering continuously for the whole stress window —
        this is what C15(c) means by "concurrent chat-path writes"."""
        writer_store = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
        try:
            row = writer_store._conn.execute("SELECT id FROM memories LIMIT 1").fetchone()
            memory_id = row["id"] if row is not None else None
            while not stop_writer.is_set() and memory_id is not None:
                try:
                    writer_store.bump_recall(memory_id, 0.01)
                except Exception as exc:  # noqa: BLE001 — record, don't crash the writer thread
                    write_errors.append(exc)
                time.sleep(0.01)
        finally:
            writer_store.close()

    writer_thread = threading.Thread(target=_concurrent_chat_writer, daemon=True)
    writer_thread.start()

    stop_event = threading.Event()
    supervisor_thread = threading.Thread(
        target=run_folded,
        kwargs={
            "stop_event": stop_event,
            "persona_dir": persona_dir,
            "provider": provider_mod.FakeProvider(),
            "event_bus": EventBus(),
            "tick_interval_s": 0.05,
            "heartbeat_interval_s": 1.0,
            "soul_review_interval_s": 0.0,  # maintenance (forgetting) due every pass
            "finalize_interval_s": None,
            "log_rotation_interval_s": None,
            "initiate_review_interval_s": None,
            "voice_reflection_interval_s": None,
            "self_model_interval_s": None,
            "compaction_interval_s": None,
            "calibration_interval_s": None,
            "interest_sweep_interval_s": None,
            "judge_selftune_interval_s": None,
            "clustering_interval_s": None,
            "vocab_repair_interval_s": None,
            "maker_enabled": False,
            "notes_enabled": False,
            "kindled_link_enabled": False,
        },
        daemon=True,
    )
    supervisor_thread.start()
    try:
        # Drive heartbeat + gated-job items for the full window the
        # criterion names, real wall-clock time (not a fake/injected
        # clock — this is a stress test of real contention).
        time.sleep(_STRESS_DURATION_S)
    finally:
        stop_writer.set()
        stop_event.set()
        supervisor_thread.join(timeout=30.0)
        writer_thread.join(timeout=10.0)
        root_logger.removeHandler(lock_capture)

    assert not write_errors, (
        f"the concurrent chat-path writer hit {len(write_errors)} exception(s), "
        f"first: {write_errors[0] if write_errors else None}"
    )
    assert lock_capture.hits == [], (
        f"'database is locked' (or similar) logged during the {_STRESS_DURATION_S}s "
        f"stress window: {lock_capture.hits[:5]}"
    )
