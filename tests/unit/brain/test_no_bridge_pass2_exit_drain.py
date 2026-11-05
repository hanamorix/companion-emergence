"""C23 (S78/S80): `nell chat --no-bridge`'s exit-drain never waits on the
lull/throttle, is bounded by `PASS2_NOBRIDGE_DRAIN_BUDGET_S`, shows CLI
progress, and leaves any budget-cutoff remainder in the durable queue.

Runs `brain.cli._drain_pass2_at_exit` — the exact function both of
`_chat_direct_mode`'s call sites (one-shot post-flush, REPL exit `finally`)
invoke — in this process, with `cli_throttle.is_chat_idle` /
`acquire_background` monkeypatched to ALWAYS deny.
This is the fail-first oracle from round-3/round-4's history: the prior
`drain_pending()`-after-`respond()` design was found INERT because it
consulted exactly this always-denying gate; `drain_all_locked()` (S78) must
never consult it at all. The budget is checked against a fake clock the fake
dispatch advances, so no test sleeps. Isolating `_drain_pass2_at_exit` directly
(rather than driving full chat turns through a real LLM provider) keeps this test
free of network/provider fixtures while still exercising the REAL
production call path cli.py uses.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain import cli as brain_cli
from brain import dev_constants
from brain.bridge import cli_throttle
from brain.chat import pass2_queue


def _write_records(persona_dir: Path, n: int) -> None:
    records = [{"id": f"item-{i}", "kind": "test_probe"} for i in range(n)]
    (persona_dir / "pass2_queue.json").write_text(json.dumps(records), encoding="utf-8")


def _slow_items(monkeypatch: pytest.MonkeyPatch, per_item_s: float, budget_s: float) -> None:
    """Every dispatched item takes ``per_item_s`` of fake time; the exit-drain's budget is ``budget_s``."""
    now = [0.0]
    monkeypatch.setattr(pass2_queue, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def _slow_dispatch(record, *, persona_dir):
        now[0] += per_item_s

    monkeypatch.setattr(pass2_queue, "_dispatch", _slow_dispatch)
    monkeypatch.setattr(dev_constants, "PASS2_NOBRIDGE_DRAIN_BUDGET_S", budget_s)


def test_exit_drain_completes_with_no_lull_even_when_throttle_always_denies(tmp_path, monkeypatch, capsys):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_records(persona_dir, 2)
    # Simulate "chat is never idle" — the exact condition round-4's cold
    # red-team found made the round-3 fix inert.
    monkeypatch.setattr(cli_throttle, "is_chat_idle", lambda **_: False)
    monkeypatch.setattr(cli_throttle, "acquire_background", lambda **_: False)

    brain_cli._drain_pass2_at_exit(persona_dir)

    remaining = json.loads((persona_dir / "pass2_queue.json").read_text())
    assert remaining == []  # both items drained — the throttle was never consulted
    out = capsys.readouterr().out
    assert "pass-2: 1 of 2 done" in out
    assert "pass-2: 2 of 2 done" in out


def test_exit_drain_is_bounded_and_progress_shown_remainder_stays_queued(tmp_path, monkeypatch, capsys):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    n_items = 5
    _write_records(persona_dir, n_items)
    # 0.3 s per item against a 0.5 s budget: the budget trips after the second item (0.6 s > 0.5 s).
    _slow_items(monkeypatch, per_item_s=0.3, budget_s=0.5)

    brain_cli._drain_pass2_at_exit(persona_dir)

    remaining = json.loads((persona_dir / "pass2_queue.json").read_text())
    assert len(remaining) == n_items - 2, "the budget should stop the drain after exactly two items"
    assert "pass-2: 1 of 5 done" in capsys.readouterr().out


def test_single_item_exceeding_the_whole_budget_still_completes(tmp_path, monkeypatch):
    """round-7 minor: one item whose own duration exceeds the ENTIRE budget
    still runs to completion (never interrupted mid-item) before the next
    check ever fires — exactly 1 item drained, not 0 and not 2+."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_records(persona_dir, 2)
    _slow_items(monkeypatch, per_item_s=1.0, budget_s=0.2)  # budget much shorter than one item

    brain_cli._drain_pass2_at_exit(persona_dir)

    remaining = json.loads((persona_dir / "pass2_queue.json").read_text())
    assert len(remaining) == 1  # exactly 1 drained, not 0 (started before budget seen exceeded)
