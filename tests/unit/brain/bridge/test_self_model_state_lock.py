"""#558: the self-model tick and the reconcile tool both load, change and save self_model_state.json.

Without a lock spanning the whole read-modify-write, a reconcile that lands while the tick is mid-flight is lost
(the tick saves over it from its stale snapshot), and vice versa. These tests force the interleavings with real
threads through the real tick (`_run_self_model_tick`, live engine path) and the real `reconcile_self_read`.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from brain.bridge import supervisor
from brain.bridge.provider import FakeProvider
from brain.self_model import state as sm_state
from brain.self_model.gap import Gap
from brain.self_model.reconcile import is_channel_in_cooldown, reconcile_self_read

_JOIN_S = 10.0


class _Bus:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish(self, event: dict) -> None:
        self.events.append(event)


def _persona(tmp_path: Path) -> Path:
    p = tmp_path / "persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text(
        '{"provider": "fake", "searcher": "noop"}', encoding="utf-8"
    )
    return p


def _open_grief_gap(persona_dir: Path) -> None:
    """A saved open grief gap the live tick continues (same channel, engine never reset)."""
    gap = Gap(
        per_channel={"grief": 2.0},
        magnitude=2.0,
        unnamed_pressure=0.0,
        status="open",
        sustained_ticks=1,
        directions={"grief": 1},
        reset_seen=None,
    )
    sm_state.save(persona_dir, sm_state.SelfModelState(current_gap=gap, gap_history=[]))


def _live_engine(stub) -> None:
    from brain.emotion.engine import Reading

    stub.live = True
    stub.readings = {"grief": Reading(short=6.0, long=4.0, difference=2.0, shift=1)}


def _in_thread(fn) -> threading.Thread:
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t


def _cadence_outcome(persona_dir: Path) -> str:
    """What the last tick told the cadence: "deferred" retries within about a minute, "clean" waits hours."""
    from datetime import UTC, datetime, timedelta

    from brain.self_model import cadence as sm_cadence

    state = sm_cadence.load(persona_dir)
    assert state.next_reflection_at is not None, "the tick did not advance the cadence"
    ahead = state.next_reflection_at - datetime.now(UTC)
    if ahead <= timedelta(minutes=2):
        return "deferred"
    if ahead >= timedelta(hours=1) and state.consecutive_failures == 0:
        return "clean"
    return f"other ({ahead}, failures={state.consecutive_failures})"


def test_a_state_file_holding_nan_does_not_wedge_the_tick_into_deferred_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, emotion_engine_stub
) -> None:
    """A NaN (old json.dumps allowed it; a hand edit can add it) never equals itself, so the tick's under-lock
    re-load never matched its snapshot: every tick ended "deferred" and retried a minute later, paying for an
    articulate call each time. The loader treats a non-finite number as a malformed gap, so the tick runs clean."""
    persona_dir = _persona(tmp_path)
    (persona_dir / "self_model_state.json").write_text(
        '{"current_gap": {"per_channel": {"grief": 2.0}, "magnitude": NaN, "unnamed_pressure": 0.0,'
        ' "status": "open", "sustained_ticks": 1, "directions": {"grief": 1}, "reset_seen": null},'
        ' "gap_history": []}',
        encoding="utf-8",
    )
    _live_engine(emotion_engine_stub)
    monkeypatch.setattr(supervisor, "sm_articulate", lambda gap, *, provider, persona_dir: "a note")

    supervisor._run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=_Bus())

    assert _cadence_outcome(persona_dir) == "clean"
    state, recovered = sm_state.load_or_recover(persona_dir)
    assert not recovered, "the tick's save should have replaced the NaN-bearing gap"
    assert state.current_gap is not None and state.current_gap.magnitude == 2.0


def test_a_reconcile_during_the_ticks_articulate_call_is_not_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, emotion_engine_stub
) -> None:
    """She dismisses grief while the tick is waiting on its articulate (Haiku) call. The tick must not then save its
    stale snapshot over her dismissal: the grief cooldown she set has to survive."""
    persona_dir = _persona(tmp_path)
    _open_grief_gap(persona_dir)
    _live_engine(emotion_engine_stub)

    # The tick swallows a raise inside it, so the hook records and the test asserts.
    still_waiting: list[bool] = []

    def articulate_while_she_reconciles(gap, *, provider, persona_dir):
        t = _in_thread(
            lambda: reconcile_self_read(persona_dir=persona_dir, action="dismiss", channel="grief")
        )
        t.join(_JOIN_S)
        still_waiting.append(t.is_alive())
        return "a note"

    monkeypatch.setattr(supervisor, "sm_articulate", articulate_while_she_reconciles)
    supervisor._run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=_Bus())

    assert still_waiting == [False], "reconcile must not wait on the tick's articulate call"
    gap = sm_state.load_or_recover(persona_dir)[0].current_gap
    assert gap is not None
    assert gap.status == "dismissed", gap
    assert is_channel_in_cooldown(gap, "grief"), gap.channel_cooldowns


def test_a_reconcile_that_lands_while_the_tick_saves_waits_for_it_and_both_writes_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, emotion_engine_stub
) -> None:
    """She dismisses grief just as the tick is saving. The reconcile must take the same lock for its own load → change
    → save, so it waits, then loads the tick's new gap and dismisses that: the tick's update (sustained_ticks 2, its
    note) and her dismissal both survive. Without the lock she saves first and the tick's save wipes her out."""
    persona_dir = _persona(tmp_path)
    _open_grief_gap(persona_dir)
    _live_engine(emotion_engine_stub)
    monkeypatch.setattr(supervisor, "sm_articulate", lambda gap, *, provider, persona_dir: "a note")

    real_save = sm_state.save
    reconciles: list[threading.Thread] = []
    # Whether she was still waiting after half a second: under the lock she must be (the tick holds it), so a
    # build without the lock cannot pass just because her reconcile happened to run slowly.
    blocked: list[bool] = []

    def save_while_she_reconciles(pd, st):
        t = _in_thread(
            lambda: reconcile_self_read(persona_dir=pd, action="dismiss", channel="grief")
        )
        reconciles.append(t)
        t.join(0.5)
        blocked.append(t.is_alive())
        real_save(pd, st)

    monkeypatch.setattr(sm_state, "save", save_while_she_reconciles)
    supervisor._run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=_Bus())
    for t in reconciles:
        t.join(_JOIN_S)
        assert not t.is_alive()

    assert blocked == [True], "the reconcile did not wait for the tick's lock"

    gap = sm_state.load_or_recover(persona_dir)[0].current_gap
    assert gap is not None
    assert gap.status == "dismissed", gap
    assert is_channel_in_cooldown(gap, "grief"), gap.channel_cooldowns
    assert gap.sustained_ticks == 2 and gap.note == "a note", gap


def test_a_reconcile_before_the_spare_rule_saves_is_not_cleared_by_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, emotion_engine_stub
) -> None:
    """After an engine rebuild the tick's spare rule clears an open gap from its snapshot. If she dismissed that gap
    after the tick loaded it, the spare rule must not clear her dismissal (a dismissed gap held for the sustained
    ticks is one the rule keeps)."""
    persona_dir = _persona(tmp_path)
    gap = Gap(
        per_channel={"grief": 2.0},
        magnitude=2.0,
        unnamed_pressure=0.0,
        status="open",
        sustained_ticks=3,
        directions={"grief": 1},
    )  # reset_seen never stored → the live tick runs the spare rule
    sm_state.save(persona_dir, sm_state.SelfModelState(current_gap=gap, gap_history=[]))
    _live_engine(emotion_engine_stub)

    real_reset_seen = supervisor._self_model_reset_seen
    # The tick swallows a raise inside it, so the hook records and the test asserts.
    still_waiting: list[bool] = []

    def reset_seen_while_she_reconciles(reads, prior_gap):
        t = _in_thread(
            lambda: reconcile_self_read(persona_dir=persona_dir, action="dismiss", channel="grief")
        )
        t.join(_JOIN_S)
        still_waiting.append(t.is_alive())
        return real_reset_seen(reads, prior_gap)

    monkeypatch.setattr(supervisor, "_self_model_reset_seen", reset_seen_while_she_reconciles)
    supervisor._run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=_Bus())

    assert still_waiting == [False]
    saved = sm_state.load_or_recover(persona_dir)[0].current_gap
    assert saved is not None, "the spare rule cleared a gap she had dismissed"
    assert saved.status == "dismissed"
    assert is_channel_in_cooldown(saved, "grief"), saved.channel_cooldowns
    # The rule did not run, so it is retried soon, as the tick's own mismatch is (not after the 6 h clean interval).
    assert _cadence_outcome(persona_dir) == "deferred"


@pytest.mark.no_base_failure(reason="control for the race test above: a spare-rule run with no reconcile already ends clean on the base")
def test_the_spare_rule_with_no_reconcile_in_between_ends_clean(
    tmp_path: Path, emotion_engine_stub
) -> None:
    """The control for the race above: with nothing changing the file, the spare rule clears the open gap and the
    tick schedules the normal interval."""
    persona_dir = _persona(tmp_path)
    gap = Gap(
        per_channel={"grief": 2.0},
        magnitude=2.0,
        unnamed_pressure=0.0,
        status="open",
        sustained_ticks=3,
        directions={"grief": 1},
    )  # reset_seen never stored → the live tick runs the spare rule
    sm_state.save(persona_dir, sm_state.SelfModelState(current_gap=gap, gap_history=[]))
    _live_engine(emotion_engine_stub)

    supervisor._run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=_Bus())

    assert sm_state.load_or_recover(persona_dir)[0].current_gap is None
    assert _cadence_outcome(persona_dir) == "clean"


def test_the_startup_repair_waits_for_a_reconcile_in_flight_so_its_clear_is_not_undone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one-time startup repair clears the current gap. If it saves while a reconcile is between its load and its
    save, the reconcile then writes the old gap back. The repair takes the same lock, so it clears after her save."""
    from brain.health.self_model_repair import run_self_model_repair
    from brain.self_model import reconcile

    persona_dir = _persona(tmp_path)
    _open_grief_gap(persona_dir)

    real_load = reconcile.load_or_recover
    repairs: list[threading.Thread] = []
    # Whether the repair was still waiting after half a second: under the lock it must be (the reconcile holds it).
    blocked: list[bool] = []

    def load_while_the_repair_runs(pd):
        loaded = real_load(pd)
        t = _in_thread(lambda: run_self_model_repair(pd))
        repairs.append(t)
        t.join(0.5)
        blocked.append(t.is_alive())
        return loaded

    monkeypatch.setattr(reconcile, "load_or_recover", load_while_the_repair_runs)
    reconcile_self_read(persona_dir=persona_dir, action="dismiss", channel="grief")
    for t in repairs:
        t.join(_JOIN_S)
        assert not t.is_alive()

    assert blocked == [True], "the repair did not wait for the reconcile's lock"

    assert sm_state.load_or_recover(persona_dir)[0].current_gap is None, (
        "the reconcile wrote the cleared gap back"
    )


# ── the wait is bounded (#730 review) ──────────────────────────────────────────────────────────────────────────────
# `file_lock` blocks forever, so a holder stalled inside its save (a hung disk) would hang the next chat turn's
# reconcile with no error. Every writer waits at most dev_constants.SELF_MODEL_LOCK_WAIT_S, then gives up cleanly.

_SHORT_WAIT_S = 0.2


def _hold_the_state_lock(persona_dir: Path):
    """The lock as a stalled holder keeps it: taken on its own file handle, so this process's waiters contend for it."""
    from brain.utils.file_lock import file_lock

    return file_lock(persona_dir / "self_model_state.json")


def _run_bounded(fn) -> tuple[list, bool]:
    """Run fn in a thread; (its result in a list, whether it was still waiting after _JOIN_S)."""
    out: list = []
    t = _in_thread(lambda: out.append(fn()))
    t.join(_JOIN_S)
    return out, t.is_alive()


def test_a_reconcile_that_cannot_get_the_lock_gives_up_and_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """She dismisses grief while a stalled holder keeps the lock. The reconcile returns an error within the wait
    instead of hanging the chat turn, and writes nothing: the gap is still open, so trying again is safe."""
    from brain import dev_constants
    from brain.self_model.resolve import load_audit

    monkeypatch.setattr(dev_constants, "SELF_MODEL_LOCK_WAIT_S", _SHORT_WAIT_S, raising=False)
    persona_dir = _persona(tmp_path)
    _open_grief_gap(persona_dir)

    with _hold_the_state_lock(persona_dir):
        out, still_waiting = _run_bounded(
            lambda: reconcile_self_read(persona_dir=persona_dir, action="dismiss", channel="grief")
        )
        assert not still_waiting, "the reconcile waited on the lock with no bound"

    assert "error" in out[0], out
    gap = sm_state.load_or_recover(persona_dir)[0].current_gap
    assert gap is not None and gap.status == "open", gap
    assert load_audit(persona_dir)["reconciles_called"] == 0


def test_a_tick_that_cannot_get_the_lock_writes_nothing_and_retries_soon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, emotion_engine_stub
) -> None:
    """A stalled holder keeps the lock while the tick reaches its save. The tick gives up within the wait, leaves the
    file as it was and ends "deferred", so the next run (about a minute later) tries again."""
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "SELF_MODEL_LOCK_WAIT_S", _SHORT_WAIT_S, raising=False)
    persona_dir = _persona(tmp_path)
    _open_grief_gap(persona_dir)
    _live_engine(emotion_engine_stub)
    monkeypatch.setattr(supervisor, "sm_articulate", lambda gap, *, provider, persona_dir: "a note")

    with _hold_the_state_lock(persona_dir):
        _out, still_waiting = _run_bounded(
            lambda: supervisor._run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=_Bus())
        )
        assert not still_waiting, "the tick waited on the lock with no bound"

    gap = sm_state.load_or_recover(persona_dir)[0].current_gap
    assert gap is not None and gap.sustained_ticks == 1 and gap.note is None, gap
    assert _cadence_outcome(persona_dir) == "deferred"


def test_a_spare_rule_that_cannot_get_the_lock_writes_nothing_and_retries_soon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, emotion_engine_stub
) -> None:
    """The spare rule's save takes the same lock: with a stalled holder it gives up within the wait, the open gap
    stays as it was, and the tick ends "deferred" so the rule runs again soon."""
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "SELF_MODEL_LOCK_WAIT_S", _SHORT_WAIT_S, raising=False)
    persona_dir = _persona(tmp_path)
    gap = Gap(
        per_channel={"grief": 2.0},
        magnitude=2.0,
        unnamed_pressure=0.0,
        status="open",
        sustained_ticks=3,
        directions={"grief": 1},
    )  # reset_seen never stored → the live tick runs the spare rule
    sm_state.save(persona_dir, sm_state.SelfModelState(current_gap=gap, gap_history=[]))
    _live_engine(emotion_engine_stub)

    with _hold_the_state_lock(persona_dir):
        _out, still_waiting = _run_bounded(
            lambda: supervisor._run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=_Bus())
        )
        assert not still_waiting, "the spare rule waited on the lock with no bound"

    saved = sm_state.load_or_recover(persona_dir)[0].current_gap
    assert saved is not None and saved.status == "open", saved
    assert _cadence_outcome(persona_dir) == "deferred"


def test_a_startup_repair_that_cannot_get_the_lock_leaves_its_marker_unwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one-time repair gives up within the wait when a stalled holder keeps the lock. It has not cleared the gap,
    so it must not write its one-time marker either: the next start runs it again."""
    from brain import dev_constants
    from brain.health.self_model_repair import run_self_model_repair, should_run_self_model_repair

    monkeypatch.setattr(dev_constants, "SELF_MODEL_LOCK_WAIT_S", _SHORT_WAIT_S, raising=False)
    persona_dir = _persona(tmp_path)
    _open_grief_gap(persona_dir)

    with _hold_the_state_lock(persona_dir):
        _out, still_waiting = _run_bounded(lambda: run_self_model_repair(persona_dir))
        assert not still_waiting, "the repair waited on the lock with no bound"

    assert sm_state.load_or_recover(persona_dir)[0].current_gap is not None
    assert should_run_self_model_repair(persona_dir), "the marker was written though the repair did not run"


def test_an_accept_that_cannot_get_the_lock_writes_no_delta_so_trying_again_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """accept/revise change the gap before they write the emotion delta: when the lock wait gives up, the delta has
    not been written, so the same call made again does not write it twice."""
    from brain import dev_constants
    from brain.self_model import reconcile

    monkeypatch.setattr(dev_constants, "SELF_MODEL_LOCK_WAIT_S", _SHORT_WAIT_S, raising=False)
    deltas: list[str] = []
    monkeypatch.setattr(reconcile, "_write_self_authored_delta", lambda pd, ch, d: deltas.append(ch) or True)
    persona_dir = _persona(tmp_path)
    _open_grief_gap(persona_dir)

    with _hold_the_state_lock(persona_dir):
        out, still_waiting = _run_bounded(
            lambda: reconcile_self_read(persona_dir=persona_dir, action="accept", channel="grief", delta=0.3)
        )
        assert not still_waiting

    assert "error" in out[0], out
    assert deltas == [], "the delta was written though the reconcile gave up"
    again = reconcile_self_read(persona_dir=persona_dir, action="accept", channel="grief", delta=0.3)
    assert again.get("ok") is True and deltas == ["grief"]
