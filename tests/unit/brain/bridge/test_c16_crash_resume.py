"""ram-spike-fix INC-10 — C16: real subprocess SIGKILL mid-item, restart,
resume from saved progress with completed items not redone (spec §4,
S14/S31/S32/S36).

Targets the FORGETTING job specifically (2-plan §4.1 row 5) because its
crash-resume mechanism is genuinely NEW in this increment: pre-INC-10, its
per-memory counters were only ever persisted at PASS END (module docstring
in brain/forgetting/__init__.py), so a kill mid-pass lost the whole pass's
progress — there was no per-item cursor to resume from at all. This test's
own fail-first (a temp worktree at the pre-INC-10 commit) fails at
collection (`brain.bridge.job_progress` does not exist there yet), the
strongest form — confirmed before this file was added to the harness.

Real subprocess kill (not a simulated exception) — a genuine process death,
not interpreter cleanup, not a `finally`, not atexit. A marker file written
from inside a patched, slow salience.score gives a real, observable
per-item sync point for exactly when to kill, rather than a sleep-based
race.

Windows-safe note: `signal.SIGKILL` does not exist on Windows (`os.kill`
there only supports a handful of signals plus CTRL_C/CTRL_BREAK events).
`Popen.kill()` is the portable hard-kill: on POSIX it sends SIGKILL, on
Windows it calls `TerminateProcess` — structurally the same "the OS ends
the process with no interpreter cleanup" kill, not re-implemented per-OS
in this file (matching the project's own convention in
test_daemon_extras.py's C18 tests).
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

from brain.bridge import job_progress

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _persona(tmp_path: Path) -> Path:
    p = tmp_path / "persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    return p


def test_c16_forgetting_sigkill_mid_item_resumes_without_redoing_done(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    marker = tmp_path / "scored.log"

    script = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(_REPO_ROOT)!r})
        from pathlib import Path
        from brain.memory.store import Memory, MemoryStore
        from brain.forgetting import run_pass
        import brain.forgetting.policy as policy_mod
        import brain.forgetting.salience as salience_mod
        from brain.bridge.events import EventBus

        persona_dir = Path({str(persona_dir)!r})
        marker = Path({str(marker)!r})

        # Fresh test memories are exempt by policy.is_exempt's own "recent
        # buffer" grace -- bypass so the loop actually reaches salience.score.
        policy_mod.is_exempt = lambda *a, **kw: False
        policy_mod.is_within_import_grace = lambda *a, **kw: False

        store = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
        ids = []
        for i in range(3):
            m = Memory.create_new(
                content=f"crash-resume forgetting content {{i}}", memory_type="conversation", domain="us"
            )
            store.create(m)
            ids.append(m.id)
        ids.sort()
        store.close()

        calls = {{"n": 0}}

        def _slow_score(memory, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                # Item 1: return immediately. run_pass's loop then persists
                # this item's counters + the NEW job_progress cursor BEFORE
                # calling us again for item 2.
                return 5.0
            # Item 2: write the marker (a real sync point proving item 1's
            # counters/cursor already landed on disk), THEN sleep -- a wide,
            # deterministic window for the parent to SIGKILL mid-item-2.
            with open(marker, "a") as f:
                f.write("item2_score_called\\n")
            time.sleep(10.0)
            return 5.0

        salience_mod.score = _slow_score
        run_pass(persona_dir, event_bus=EventBus())
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-P", "-c", script],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if marker.exists():
                break
            time.sleep(0.05)
        else:
            proc.kill()
            raise AssertionError("subprocess never reached item 2 within 20s")

        proc.kill()
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    progress = job_progress.load_progress(persona_dir, "forgetting")
    assert progress, "the NEW job_progress cursor must have survived the kill"
    scored_id_1 = progress["last_id"]
    assert scored_id_1, "item 1's cursor must be a real memory id"

    # Restart: a fresh process resumes from the saved cursor. Item 1 is not
    # re-scored; items 2/3 complete; the cursor clears on a clean finish.
    from brain.bridge.events import EventBus
    from brain.forgetting import run_pass

    scored_order: list[str] = []
    import brain.forgetting.policy as policy_mod
    import brain.forgetting.salience as salience_mod

    def _fake_is_exempt(*a, **kw):
        return False

    def _counting_score(memory, **kw):
        scored_order.append(memory.id)
        return 5.0

    policy_mod.is_exempt = _fake_is_exempt
    policy_mod.is_within_import_grace = lambda *a, **kw: False
    salience_mod.score = _counting_score
    try:
        run_pass(persona_dir, event_bus=EventBus())
    finally:
        # Restore module functions for any other test in this process.
        import importlib

        importlib.reload(policy_mod)
        importlib.reload(salience_mod)

    assert scored_id_1 not in scored_order, "item 1 (already scored+persisted before the kill) must not be redone"
    assert len(scored_order) == 2, "exactly the 2 remaining items run on resume"
    assert job_progress.load_progress(persona_dir, "forgetting") == {}, "a clean finish clears the cursor"
