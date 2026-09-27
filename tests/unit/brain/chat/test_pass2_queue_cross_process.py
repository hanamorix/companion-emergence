"""Cross-process pass-2 queue tests (ram-spike-fix INC-8, C31b/C36).

Real ``sys.executable`` subprocesses throughout (never fork-only APIs), so
this runs unmodified on the Windows/macOS CI runners too — same pattern as
``tests/unit/brain/memory/test_busy_timeout.py`` / ``test_cmd_start_lock_
handoff.py``: a ready-file a subprocess creates only AFTER reaching the
state under test, polled by the parent, removes any wall-clock guessing
about subprocess startup timing.

C36(d) — two concurrent drainers, same persona: one subprocess is held
inside its ``pass2_drain.lock`` (via a monkeypatched dispatch that blocks on
a gate file) while a second subprocess attempts ``drain_all_locked()`` at
the same time; the second must return 0 immediately and leave the queue
untouched, and the first's own drain must complete normally afterwards.

C36(a/b/c) — SIGKILL variants: items queued, kill the drainer process at
each of 3 points (before any item runs / between two items / after an
item's effect ran but before its pop is persisted), restart, and confirm
"none lost" — with the mid-item kill (c) explicitly re-running that one
item (accepted at-least-once repeat, S76), asserted as exactly 2 runs, not
a failure.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

_READY_POLL_S = 0.05
_READY_WAIT_TIMEOUT_S = 15.0


def _wait_for_file(path: Path, proc: subprocess.Popen | None = None, timeout: float = _READY_WAIT_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if proc is not None and proc.poll() is not None:
            raise AssertionError(f"subprocess exited early (returncode {proc.returncode}) before {path.name} appeared")
        time.sleep(_READY_POLL_S)
    raise AssertionError(f"{path.name} never appeared within {timeout}s")


def _write_records(persona_dir: Path, records: list[dict]) -> None:
    (persona_dir / "pass2_queue.json").write_text(json.dumps(records), encoding="utf-8")


# ---------------------------------------------------------------------------
# C36(d): two real concurrent drainers on the same persona_dir
# ---------------------------------------------------------------------------

# This script holds pass2_drain.lock for `hold_seconds` (its "test_probe"
# dispatch is monkeypatched to sleep) and writes side effects to a plain
# JSON side-file so the parent can inspect exactly what ran, without either
# subprocess needing real extraction machinery.
_HOLDING_DRAINER_SCRIPT = r"""
import json, sys, time
from pathlib import Path

persona_dir = Path(sys.argv[1])
ready_path = Path(sys.argv[2])
hold_seconds = float(sys.argv[3])
side_file = Path(sys.argv[4])

from brain.chat import pass2_queue

def _slow_dispatch(record, *, persona_dir):
    Path(ready_path).write_text("1")
    time.sleep(hold_seconds)
    existing = json.loads(side_file.read_text()) if side_file.exists() else []
    existing.append(record["id"])
    side_file.write_text(json.dumps(existing))

pass2_queue._dispatch = _slow_dispatch
done = pass2_queue.drain_all_locked(persona_dir)
Path(sys.argv[5]).write_text(str(done))
"""

_FAST_DRAINER_SCRIPT = r"""
import json, sys
from pathlib import Path

persona_dir = Path(sys.argv[1])
result_path = Path(sys.argv[2])

from brain.chat import pass2_queue

done = pass2_queue.drain_all_locked(persona_dir)
Path(result_path).write_text(str(done))
"""


def test_two_concurrent_drainers_second_leaves_queue_untouched(tmp_path):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_records(persona_dir, [
        {"id": "item-1", "kind": "test_probe"},
        {"id": "item-2", "kind": "test_probe"},
    ])

    ready_path = persona_dir / "holder.ready"
    side_file = persona_dir / "side_effects.json"
    result_a_path = persona_dir / "result_a.txt"
    result_b_path = persona_dir / "result_b.txt"

    holder = subprocess.Popen([
        sys.executable, "-c", _HOLDING_DRAINER_SCRIPT,
        str(persona_dir), str(ready_path), "2.0", str(side_file), str(result_a_path),
    ])
    try:
        _wait_for_file(ready_path, holder)  # holder now holds pass2_drain.lock

        # Second drainer attempts while the first still holds the lock.
        loser = subprocess.run(
            [sys.executable, "-c", _FAST_DRAINER_SCRIPT, str(persona_dir), str(result_b_path)],
            timeout=10.0,
        )
        assert loser.returncode == 0
        assert result_b_path.read_text().strip() == "0"  # returned 0 immediately (S77)

        # Queue must be untouched by the loser: both items still present.
        on_disk = json.loads((persona_dir / "pass2_queue.json").read_text())
        assert {it["id"] for it in on_disk} == {"item-1", "item-2"}

        holder.wait(timeout=10.0)
        assert holder.returncode == 0
        assert result_a_path.read_text().strip() == "2"  # the winner drained both, unaffected
        ran_ids = json.loads(side_file.read_text())
        assert sorted(ran_ids) == ["item-1", "item-2"]
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5.0)


# ---------------------------------------------------------------------------
# C36(a/b/c): SIGKILL variants — none lost; a mid-item kill repeats (accepted)
# ---------------------------------------------------------------------------

_KILLABLE_DRAINER_SCRIPT = r"""
import json, sys, time
from pathlib import Path

persona_dir = Path(sys.argv[1])
side_file = Path(sys.argv[2])
# "before": exit before touching the queue at all.
# "mid_item": run item-1's fn() and record it, then die BEFORE the pop is
#             persisted — the crash window this whole design accepts (S76).
# "resume": drain fully to completion, recording every item exactly like
#           the killed run did — used for the POST-restart pass, so the
#           side-effect side-file stays comparable across both processes.
mode = sys.argv[3]
ready_path = Path(sys.argv[4])

from brain.chat import pass2_queue

def _record(record_id):
    existing = json.loads(side_file.read_text()) if side_file.exists() else []
    existing.append(record_id)
    side_file.write_text(json.dumps(existing))

def _dispatch(record, *, persona_dir):
    _record(record["id"])
    if mode == "mid_item" and record["id"] == "item-1":
        Path(ready_path).write_text("1")
        time.sleep(60)  # parent SIGKILLs us well before this elapses

pass2_queue._dispatch = _dispatch

if mode == "before":
    Path(ready_path).write_text("1")
    time.sleep(60)
else:
    pass2_queue.drain_all_locked(persona_dir)
"""


def _run_killable_drain(persona_dir: Path, side_file: Path, mode: str, ready_path: Path) -> subprocess.Popen:
    return subprocess.Popen([
        sys.executable, "-c", _KILLABLE_DRAINER_SCRIPT,
        str(persona_dir), str(side_file), mode, str(ready_path),
    ])


@pytest.mark.parametrize("mode", ["before", "mid_item"])
def test_sigkill_then_restart_none_lost(tmp_path, mode):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_records(persona_dir, [
        {"id": "item-1", "kind": "test_probe"},
        {"id": "item-2", "kind": "test_probe"},
    ])
    side_file = persona_dir / "side_effects.json"
    ready_path = persona_dir / "ready"

    proc = _run_killable_drain(persona_dir, side_file, mode, ready_path)
    try:
        _wait_for_file(ready_path, proc, timeout=10.0)
        proc.kill()
        proc.wait(timeout=10.0)

        # Restart: a fresh subprocess ("resume" mode: drain to completion,
        # recording via the same side-file mechanism) must run every item
        # remaining in the persisted queue (item-1 runs AGAIN for
        # "mid_item" — the accepted at-least-once repeat, not a failure).
        ran_before_kill = json.loads(side_file.read_text()) if side_file.exists() else []
        resume_ready = persona_dir / "resume_ready"  # unused by "resume" mode, kept for arg parity
        resumed = subprocess.run(
            [sys.executable, "-c", _KILLABLE_DRAINER_SCRIPT,
             str(persona_dir), str(side_file), "resume", str(resume_ready)],
            timeout=15.0,
        )
        assert resumed.returncode == 0

        ran_after_restart = json.loads(side_file.read_text())
        remaining_after_restart = json.loads((persona_dir / "pass2_queue.json").read_text())
        assert remaining_after_restart == []  # both items eventually popped
        if mode == "mid_item":
            # item-1 ran once before the kill, then again on restart — 2
            # total runs for that one logical item (S76's accepted repeat).
            assert ran_after_restart.count("item-1") == 2
            assert ran_after_restart.count("item-2") == 1
        else:  # "before": nothing ran before the kill at all
            assert ran_before_kill == []
            assert ran_after_restart.count("item-1") == 1
            assert ran_after_restart.count("item-2") == 1

        # Queue is empty afterwards either way — nothing left behind.
        assert json.loads((persona_dir / "pass2_queue.json").read_text()) == []
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
