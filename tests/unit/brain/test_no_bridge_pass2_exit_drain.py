"""C23 (S78/S80): `nell chat --no-bridge`'s exit-drain never waits on the
lull/throttle, is bounded by `PASS2_NOBRIDGE_DRAIN_BUDGET_S`, shows CLI
progress, and leaves any budget-cutoff remainder in the durable queue.

Runs `brain.cli._drain_pass2_at_exit` — the exact function both of
`_chat_direct_mode`'s call sites (one-shot post-flush, REPL exit `finally`)
invoke — in a real subprocess, with `cli_throttle.is_chat_idle` /
`acquire_background` monkeypatched to ALWAYS deny inside that subprocess.
This is the fail-first oracle from round-3/round-4's history: the prior
`drain_pending()`-after-`respond()` design was found INERT because it
consulted exactly this always-denying gate; `drain_all_locked()` (S78) must
never consult it at all. Isolating `_drain_pass2_at_exit` directly (rather
than driving full chat turns through a real LLM provider) keeps this test
free of network/provider fixtures while still exercising the REAL
production call path cli.py uses.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

_NO_LULL_DRAIN_SCRIPT = r"""
import json, sys
from pathlib import Path

persona_dir = Path(sys.argv[1])

from brain.bridge import cli_throttle
from brain import cli as brain_cli

# Simulate "chat is never idle" — the exact condition round-4's cold
# red-team found made the round-3 fix inert.
cli_throttle.is_chat_idle = lambda **_: False
cli_throttle.acquire_background = lambda **_: False

brain_cli._drain_pass2_at_exit(persona_dir)
"""

_BOUNDED_DRAIN_SCRIPT = r"""
import json, sys, time
from pathlib import Path

persona_dir = Path(sys.argv[1])
per_item_sleep = float(sys.argv[2])

from brain.chat import pass2_queue
from brain import dev_constants, cli as brain_cli

def _slow_dispatch(record, *, persona_dir):
    time.sleep(per_item_sleep)

pass2_queue._dispatch = _slow_dispatch
brain_cli._drain_pass2_at_exit(persona_dir)
"""


def _write_records(persona_dir: Path, n: int) -> None:
    records = [{"id": f"item-{i}", "kind": "test_probe"} for i in range(n)]
    (persona_dir / "pass2_queue.json").write_text(json.dumps(records), encoding="utf-8")


def test_exit_drain_completes_with_no_lull_even_when_throttle_always_denies(tmp_path):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_records(persona_dir, 2)

    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", _NO_LULL_DRAIN_SCRIPT, str(persona_dir)],
        capture_output=True, text=True, timeout=15.0,
    )
    elapsed = time.monotonic() - start
    assert result.returncode == 0, result.stderr

    remaining = json.loads((persona_dir / "pass2_queue.json").read_text())
    assert remaining == []  # both items drained — the throttle was never consulted
    # Generous bound: this must be fast (no lull, no polling wait), not
    # merely "eventually" — a few seconds of process-startup overhead is
    # the only cost, never anywhere near a real lull (minutes).
    assert elapsed < 10.0, f"exit-drain took {elapsed:.2f}s — looks lull-gated, not immediate"
    assert "pass-2: 1 of 2 done" in result.stdout
    assert "pass-2: 2 of 2 done" in result.stdout


def test_exit_drain_is_bounded_and_progress_shown_remainder_stays_queued(tmp_path, monkeypatch):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    n_items = 5
    _write_records(persona_dir, n_items)

    # A small per-item sleep and a monkeypatched dev constant (via env, since
    # the subprocess re-imports dev_constants fresh) so the budget trips
    # partway through the backlog, deterministically.
    per_item_sleep = 0.3
    budget_s = 0.5  # allows ~1-2 items before tripping

    script = _BOUNDED_DRAIN_SCRIPT.replace(
        "from brain import dev_constants, cli as brain_cli",
        f"from brain import dev_constants, cli as brain_cli\ndev_constants.PASS2_NOBRIDGE_DRAIN_BUDGET_S = {budget_s}",
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(persona_dir), str(per_item_sleep)],
        capture_output=True, text=True, timeout=15.0,
    )
    assert result.returncode == 0, result.stderr

    remaining = json.loads((persona_dir / "pass2_queue.json").read_text())
    assert 0 < len(remaining) < n_items, (
        f"expected the budget to stop the drain partway through; remaining={len(remaining)}"
    )
    assert "pass-2: 1 of 5 done" in result.stdout


def test_single_item_exceeding_the_whole_budget_still_completes(tmp_path):
    """round-7 minor: one item whose own duration exceeds the ENTIRE budget
    still runs to completion (never interrupted mid-item) before the next
    check ever fires — exactly 1 item drained, not 0 and not 2+."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_records(persona_dir, 2)

    per_item_sleep = 1.0
    budget_s = 0.2  # much shorter than one item's own duration

    script = _BOUNDED_DRAIN_SCRIPT.replace(
        "from brain import dev_constants, cli as brain_cli",
        f"from brain import dev_constants, cli as brain_cli\ndev_constants.PASS2_NOBRIDGE_DRAIN_BUDGET_S = {budget_s}",
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(persona_dir), str(per_item_sleep)],
        capture_output=True, text=True, timeout=15.0,
    )
    assert result.returncode == 0, result.stderr

    remaining = json.loads((persona_dir / "pass2_queue.json").read_text())
    assert len(remaining) == 1  # exactly 1 drained, not 0 (started before budget seen exceeded)
