"""`read()` and `status()` of the emotion engine (Engine 4, #579).

Reads never write, never recover and never take the lock. `read()` answers `Reading(short, long, difference, shift)` for the
present emotions whose names are in the vocabulary registry; `{}` while the state cannot be used. It asks the persona loader
to refresh the registry first, inside its own `try`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from brain.emotion import engine, engine_files, persona_loader, vocabulary
from brain.emotion.engine_utils import Average, EmotionState, EngineState
from brain.emotion.vocabulary import Emotion
from tests.unit.brain.emotion._engine_support import (
    LockSpy,
    install,
    local_day,
    local_noon,
    lose_the_state,
    make_live,
    make_persona,
    save,
    save_as_backup,
    simple_state,
    tree,
)

D = [local_noon(local_day(n)) for n in range(40)]
NOVEL = "zz_engine_read_test_feeling"


@pytest.fixture
def persona(tmp_path: Path) -> Path:
    return make_persona(tmp_path / "persona")


def _day(n: int) -> str:
    return local_day(n).isoformat()


def _two_emotion_state() -> EngineState:
    state = simple_state(saved_at=D[3], joy=4.0, day=_day(3))
    state.emotions["joy"].short = Average(6.0, 2.0)  # short 6, long 4: difference +2
    state.emotions["grief"] = EmotionState(short=Average(1.0, 1.0), long=Average(3.0, 5.0), last_entry_day=_day(3))
    state.emotions["ghost"] = EmotionState(short=Average(2.0, 1.0), long=Average(2.0, 0.0), last_entry_day=_day(1))  # no weight left
    return state


def _no_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(persona_loader, "refresh_if_changed", raising=False)


# --- read() (Rd1, Rd4) --------------------------------------------------------------------------------------


def test_read_returns_the_stored_averages_their_difference_and_only_the_present_emotions(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_refresh(monkeypatch)
    save(persona, _two_emotion_state())
    readings = install(persona).read()
    assert set(readings) == {"joy", "grief"}  # `ghost` has no long weight: not present
    joy, grief = readings["joy"], readings["grief"]
    assert (joy.short, joy.long, joy.difference, joy.shift) == (6.0, 4.0, 2.0, 0)
    assert (grief.short, grief.long, grief.difference) == (1.0, 3.0, -2.0)


def test_read_changes_no_file_takes_no_lock_and_returns_fresh_values_each_time(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_refresh(monkeypatch)
    save(persona, _two_emotion_state())
    eng = install(persona)
    before = tree(persona)
    spy = LockSpy(monkeypatch)
    first = eng.read()
    first.clear()  # a caller changing its result must not reach the next read
    second = eng.read()
    assert set(second) == {"joy", "grief"}
    eng.status()
    assert (spy.calls, spy.within, spy.acquisitions) == (0, [], [])
    assert tree(persona) == before


def test_read_is_empty_while_the_state_is_missing_unreadable_or_flagged_and_takes_no_lock_for_it(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_refresh(monkeypatch)
    eng = install(persona)
    spy = LockSpy(monkeypatch)
    assert eng.read() == {}  # missing
    save(persona, _two_emotion_state())
    assert set(eng.read()) == {"joy", "grief"}  # the control: the same state is readable until it is damaged
    for suffix in ("", ".bak1", ".bak2", ".bak3"):
        target = engine_files.state_path(persona).with_name(engine_files.state_path(persona).name + suffix)
        if target.exists():
            target.write_bytes(b"\x00")
    assert eng.read() == {}  # unreadable
    assert spy.calls == 0  # a lock-taking read fails here even though the answer is empty


def test_read_is_empty_while_a_catch_up_failed(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_refresh(monkeypatch)
    make_live(persona, D[1])
    eng = install(persona, D[8])
    assert eng.is_live()  # the running engine has seen the switch-over; the state is gone afterwards
    lose_the_state(persona)
    eng.note_active(D[8])  # a live persona with no state: flagged
    save(persona, _two_emotion_state())
    assert eng.read() == {}
    assert eng.status().caught_up is False


def test_the_shift_is_worked_out_once_per_state_load_and_kept_with_the_cached_state(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_refresh(monkeypatch)
    save(persona, _two_emotion_state())
    calls: list[int] = []
    real = engine.shift_for
    monkeypatch.setattr(engine, "shift_for", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    eng = install(persona)
    eng.read()
    eng.read()
    assert len(calls) == 1
    changed = _two_emotion_state()
    changed.saved_at = D[4]
    save(persona, changed)
    eng.read()
    assert len(calls) == 2


def test_the_shift_comes_from_the_snapshot_and_the_start_gate(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A direct check that the readings carry `shift_for`'s answer for the emotion: the gate is closed with no start date."""
    _no_refresh(monkeypatch)
    state = _two_emotion_state()
    state.persona_start_at = None
    save(persona, state)
    seen: dict[str, object] = {}
    monkeypatch.setattr(engine, "shift_for", lambda snaps, start: (seen.update(start=start), {"joy": 1, "grief": -1})[1])
    readings = install(persona).read()
    assert seen["start"] is None
    assert (readings["joy"].shift, readings["grief"].shift) == (1, -1)


# --- the registry filter (Rd2) ----------------------------------------------------------------------------------


def test_a_name_not_in_the_registry_is_left_out_logged_once_and_appears_once_it_is_registered(
    persona: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _no_refresh(monkeypatch)
    state = _two_emotion_state()
    state.emotions[NOVEL] = EmotionState(short=Average(3.0, 1.0), long=Average(3.0, 1.0), last_entry_day=_day(3))
    save(persona, state)
    eng = install(persona)
    with caplog.at_level(logging.WARNING, logger="brain.emotion.engine"):
        assert NOVEL not in eng.read()
        assert NOVEL not in eng.read()
    assert len([r for r in caplog.records if NOVEL in r.getMessage()]) == 1  # once per name per instance
    assert NOVEL in engine_files.read_state(persona).state.emotions  # the stored state keeps it
    vocabulary.register(Emotion(name=NOVEL, description="a test feeling", category="persona_extension", decay_half_life_days=14.0))
    assert NOVEL in eng.read()


# --- the refresh (Rd3) ---------------------------------------------------------------------------------------


def test_a_name_registered_by_another_process_appears_on_the_next_read_and_the_refresh_gets_the_vocabulary_path(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _two_emotion_state()
    state.emotions[NOVEL] = EmotionState(short=Average(3.0, 1.0), long=Average(3.0, 1.0), last_entry_day=_day(3))
    save(persona, state)
    asked: list[Path] = []

    def refresh(path: Path) -> None:
        asked.append(path)
        if vocabulary.get(NOVEL) is None:
            vocabulary.register(Emotion(name=NOVEL, description="registered elsewhere", category="persona_extension", decay_half_life_days=14.0))

    monkeypatch.setattr(persona_loader, "refresh_if_changed", refresh, raising=False)
    assert NOVEL in install(persona).read()
    assert asked == [engine_files.vocabulary_path(persona)]


def test_with_no_refresh_function_the_call_does_nothing_and_the_read_returns_the_registrys_names(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_refresh(monkeypatch)
    save(persona, _two_emotion_state())
    assert set(install(persona).read()) == {"joy", "grief"}


@pytest.mark.parametrize("failure", ["raises", "wrong arguments"])
def test_a_refresh_that_fails_is_logged_once_and_the_read_still_returns_the_registrys_names_not_empty(
    persona: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: str
) -> None:
    save(persona, _two_emotion_state())
    if failure == "raises":

        def refresh(path: Path) -> None:
            raise RuntimeError("the vocabulary file is damaged")

    else:

        def refresh() -> None:  # takes no argument: the call fails with a TypeError
            return None

    monkeypatch.setattr(persona_loader, "refresh_if_changed", refresh, raising=False)
    eng = install(persona)
    with caplog.at_level(logging.WARNING, logger="brain.emotion.engine"):
        assert set(eng.read()) == {"joy", "grief"}
        assert set(eng.read()) == {"joy", "grief"}
    assert len([r for r in caplog.records if "vocabulary refresh failed" in r.getMessage()]) == 1  # once per distinct failure


# --- status (Rd5) -------------------------------------------------------------------------------------------------


def test_status_for_a_missing_state_a_main_state_a_backup_and_an_unreadable_state(persona: Path) -> None:
    eng = install(persona)
    assert eng.status() == engine.EngineStatus(True, None)  # nothing to catch up
    state = simple_state(saved_at=D[3], day=_day(3))
    state.reset_at = D[2]
    save(persona, state)
    assert eng.status() == engine.EngineStatus(True, D[2])
    save_as_backup(persona, state)
    assert eng.status() == engine.EngineStatus(False, D[2])  # a backup copy is as old as its saved_at
    for suffix in ("", ".bak1", ".bak2", ".bak3"):
        target = engine_files.state_path(persona).with_name(engine_files.state_path(persona).name + suffix)
        if target.exists():
            target.write_bytes(b"\x00")
    assert eng.status() == engine.EngineStatus(False, None)


def test_status_is_false_and_read_empty_for_any_failure_of_the_call_itself(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_refresh(monkeypatch)
    eng = install(persona)
    monkeypatch.setattr(eng._state_cache, "get", lambda: (_ for _ in ()).throw(RuntimeError("a bug")))
    assert eng.status() == engine.EngineStatus(False, None)
    assert eng.read() == {}
