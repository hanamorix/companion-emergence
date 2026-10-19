"""A gated pass stops at its next item when the supervisor is told to stop (#437, #445).

``run_folded`` read ``stop_event`` only at the top of its loop and in the
inter-tick wait. A pausable gated job (forgetting, the backfills, finalize and
so on) takes its pause signal from the supervisor's ``_between_items`` hook,
which ignored ``stop_event``: once a long pass had started, shutdown waited for
the whole pass (the bridge's 180 s join). These tests pin the fix.

Guard 1: ``_between_items`` also reports pause when ``stop_event`` is set, so a
pass saves its progress, returns PAUSED and the loop exits at the next item,
exactly as for a chat pause.

Guard 2: ``_maybe_run_heartbeat`` does not start a heartbeat once ``stop_event``
is set (``run_central_pass`` calls the heartbeat hook after a PAUSED job, before
it ends the sequence, so without this a stop still waits one heartbeat).

The tests run the real ``run_folded``. The only stand-in is the forgetting
pass: a spy that behaves like the real one (a loop of items, ``should_pause()``
after each), runs on the supervisor thread, and sets ``stop_event`` itself at a
known item. Everything the assertions compare is therefore recorded on one
thread, in order, so there is no cross-thread race and no off-by-one.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import brain.bridge.provider as provider_mod
import brain.bridge.supervisor as supervisor_mod
from brain.bridge import cli_throttle
from brain.bridge.events import EventBus

_ITEMS = 1000
_STOP_AT_ITEM = 3
_JOIN_TIMEOUT_S = 10.0


def _loop_kwargs(
    persona_dir: Path, stop_event: threading.Event, *, heartbeat_interval_s: float | None
) -> dict:
    """The complete explicit kwargs of the C15(c) stress test (every optional
    cadence off, maker/notes/kindled link off) plus the pending-write
    reconcile cadence, so no cadence can build a provider that spawns a real
    Claude CLI."""
    return {
        "stop_event": stop_event,
        "persona_dir": persona_dir,
        "provider": provider_mod.FakeProvider(),
        "event_bus": EventBus(),
        "tick_interval_s": 0.01,
        "heartbeat_interval_s": heartbeat_interval_s,
        "soul_review_interval_s": 0.0,  # maintenance (forgetting) due from the second pass
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
        "pending_reconcile_interval_s": None,
        "maker_enabled": False,
        "notes_enabled": False,
        "kindled_link_enabled": False,
    }


@dataclass
class _Record:
    """What the spies saw. Written on the supervisor thread only; read by the
    test after the thread has been joined."""

    items: int = 0
    items_at_set: int | None = None
    hb_calls: int = 0
    hb_calls_at_set: int | None = None
    pause_answers: list[bool] = field(default_factory=list)
    reached: threading.Event = field(default_factory=threading.Event)


def _run_with_spy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    heartbeat_interval_s: float | None,
    set_stop_at_item: int | None,
    chat_active_at_item: int | None = None,
) -> tuple[_Record, bool]:
    """Run the real ``run_folded`` with the forgetting pass replaced by a spy.

    The spy loops over ``_ITEMS`` items, calls ``should_pause()`` after each as
    the real pass does and returns paused as soon as it answers True. At item
    ``set_stop_at_item`` it first sets the loop's own ``stop_event`` (just
    before that item's ``should_pause()`` call) and snapshots the counters; at
    ``chat_active_at_item`` it first marks chat active. In every variant the
    spy ends the run itself, so the thread always exits.

    Returns the record and whether the supervisor thread was still alive after
    the join timeout."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "active_conversations").mkdir()
    stop_event = threading.Event()
    record = _Record()

    def _counting_heartbeat(*_args, **_kwargs):
        record.hb_calls += 1
        return None  # "no attempt": last_heartbeat_at stays put, so one is due on every hook call

    def _spy_forgetting(persona_dir, *, event_bus, intensity_drivers, should_pause, progress_out):
        for _ in range(_ITEMS):
            time.sleep(0.001)
            record.items += 1
            if set_stop_at_item is not None and record.items == set_stop_at_item:
                stop_event.set()
                record.items_at_set = record.items
                record.hb_calls_at_set = record.hb_calls
            if chat_active_at_item is not None and record.items == chat_active_at_item:
                cli_throttle.note_user_message()
            answer = should_pause()
            record.pause_answers.append(answer)
            record.reached.set()
            if answer:
                if progress_out is not None:
                    progress_out["paused"] = True
                if set_stop_at_item is None:
                    stop_event.set()  # chat-pause variant: end the run once the pause was seen
                return {}
        stop_event.set()
        return {}

    monkeypatch.setattr(supervisor_mod, "forgetting_run_pass", _spy_forgetting)
    monkeypatch.setattr(supervisor_mod, "_heartbeat_and_felt_time", _counting_heartbeat)

    thread = threading.Thread(
        target=supervisor_mod.run_folded,
        kwargs=_loop_kwargs(persona_dir, stop_event, heartbeat_interval_s=heartbeat_interval_s),
        daemon=True,
    )
    thread.start()
    try:
        thread.join(timeout=_JOIN_TIMEOUT_S)
        still_alive = thread.is_alive()
    finally:
        stop_event.set()
        if thread.is_alive():
            thread.join(timeout=_JOIN_TIMEOUT_S)
    return record, still_alive


def test_gated_pass_stops_on_the_item_where_stop_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1: stop set mid-pass -> the very next ``should_pause()`` is True, the
    pass processes no further item, and the supervisor thread exits."""
    record, still_alive = _run_with_spy(
        tmp_path, monkeypatch, heartbeat_interval_s=None, set_stop_at_item=_STOP_AT_ITEM
    )

    assert record.items_at_set == _STOP_AT_ITEM, (
        f"the spy never reached item {_STOP_AT_ITEM} (items={record.items}); the maintenance "
        "job did not run, so this test proved nothing"
    )
    assert record.pause_answers[_STOP_AT_ITEM - 1] is True, (
        "should_pause() did not report pause on the item where stop_event was set"
    )
    assert record.items == record.items_at_set, (
        f"{record.items - record.items_at_set} item(s) were processed after stop_event was set"
    )
    assert not still_alive, f"the supervisor thread did not exit within {_JOIN_TIMEOUT_S}s"


def test_no_heartbeat_starts_after_stop_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2 (guard 2): with a heartbeat due on every hook call, none is started
    after stop_event is set, neither inside the pause hook nor by the heartbeat
    that ``run_central_pass`` runs after a PAUSED job."""
    record, still_alive = _run_with_spy(
        tmp_path, monkeypatch, heartbeat_interval_s=0.01, set_stop_at_item=_STOP_AT_ITEM
    )

    assert record.hb_calls_at_set is not None and record.hb_calls_at_set >= 1, (
        "no heartbeat was attempted before stop_event was set, so this test proved nothing"
    )
    assert record.hb_calls == record.hb_calls_at_set, (
        f"{record.hb_calls - record.hb_calls_at_set} heartbeat(s) started after stop_event was set"
    )
    assert not still_alive, f"the supervisor thread did not exit within {_JOIN_TIMEOUT_S}s"


def test_chat_pause_behaviour_is_unchanged_when_stop_is_not_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P3: with stop_event unset the hook still answers False while chat is
    idle, True once chat is active, and the heartbeat still runs when due."""
    record, still_alive = _run_with_spy(
        tmp_path,
        monkeypatch,
        heartbeat_interval_s=0.01,
        set_stop_at_item=None,
        chat_active_at_item=5,
    )

    assert record.reached.is_set(), "the maintenance job never ran, so this test proved nothing"
    assert record.pause_answers[:4] == [False] * 4, (
        f"should_pause() answered {record.pause_answers[:4]} while chat was idle and stop unset"
    )
    assert record.pause_answers[4] is True, "chat became active but should_pause() did not pause"
    assert len(record.pause_answers) == 5, "the pass kept going after the chat pause"
    assert record.hb_calls >= 1, "the heartbeat no longer runs between items when stop is unset"
    assert not still_alive, f"the supervisor thread did not exit within {_JOIN_TIMEOUT_S}s"
