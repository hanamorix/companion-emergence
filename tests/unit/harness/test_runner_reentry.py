"""G4 — the runner state machine: checkpoint/resume, park-on-trip, refuse nothing silently.

Driven by a FAKE engine (``drive_fn``), so no server + no tokens. The fake returns a scripted exit
code per (arm, invocation) to exercise the branches.
"""

from __future__ import annotations

from pathlib import Path

from tests.harness import (
    EXIT_DONE,
    EXIT_INVALID,
    EXIT_LIMIT,
    EXIT_REVIEW,
    ArmSpec,
    PersonaSpec,
    Runner,
)
from tests.harness.runner import RUN_COMPLETE, RUN_HELD, RUN_STALLED


def _arm(name: str) -> ArmSpec:
    return ArmSpec(name=name, persona=PersonaSpec(), bob=object(), detector=object())


def test_all_arms_complete(tmp_path: Path) -> None:
    arms = [_arm("A"), _arm("B"), _arm("C")]
    calls: list[str] = []

    def drive(arm: ArmSpec, resume: bool) -> int:
        calls.append(f"{arm.name}:{resume}")
        return EXIT_DONE

    r = Runner(arms, tmp_path / "state.json", drive)
    assert r.run_once() == RUN_COMPLETE
    assert calls == ["A:False", "B:False", "C:False"]
    assert not (tmp_path / "state.json").exists()  # cleared on complete


def test_stall_then_resume_continues_same_arm(tmp_path: Path) -> None:
    arms = [_arm("A"), _arm("B")]
    seen: list[str] = []
    # A stalls first time, then completes on resume; B completes.
    script = {"A": [EXIT_LIMIT, EXIT_DONE], "B": [EXIT_DONE]}

    def drive(arm: ArmSpec, resume: bool) -> int:
        seen.append(f"{arm.name}:{resume}")
        return script[arm.name].pop(0)

    state = tmp_path / "state.json"
    r1 = Runner(arms, state, drive)
    assert r1.run_once() == RUN_STALLED
    assert state.exists()  # checkpoint persists across the "re-invocation"

    # Re-invoke (fresh Runner reading the state file) — must RESUME arm A, not restart from B or 0.
    r2 = Runner(arms, state, drive)
    assert r2.run_once() == RUN_COMPLETE
    assert seen == ["A:False", "A:True", "B:False"]


def test_detector_trip_parks_and_continues(tmp_path: Path) -> None:
    arms = [_arm("A"), _arm("B")]

    def drive(arm: ArmSpec, resume: bool) -> int:
        return EXIT_REVIEW if arm.name == "A" else EXIT_DONE

    r = Runner(arms, tmp_path / "state.json", drive)
    assert r.run_once() == RUN_COMPLETE  # parks A, continues to B, done


def test_invalid_holds(tmp_path: Path) -> None:
    arms = [_arm("A"), _arm("B")]

    def drive(arm: ArmSpec, resume: bool) -> int:
        return EXIT_INVALID

    r = Runner(arms, tmp_path / "state.json", drive)
    assert r.run_once() == RUN_HELD
    assert (tmp_path / "state.json").exists()  # held state persists (does NOT clear)


def test_corrupt_state_refuses(tmp_path: Path) -> None:
    """L1 (oracle-can-fail): a corrupt/unparseable state file must REFUSE (HELD), never restart."""
    arms = [_arm("A")]
    called: list[str] = []

    def drive(arm: ArmSpec, resume: bool) -> int:
        called.append(arm.name)  # must NOT be reached
        return EXIT_DONE

    state = tmp_path / "state.json"
    state.write_text("{ this is not valid json ")
    r = Runner(arms, state, drive)
    assert r.run_once() == RUN_HELD
    assert called == [], "must not drive any arm off a corrupt checkpoint"


def test_state_load_tolerates_extra_keys(tmp_path: Path) -> None:
    """L2: a forward-compat state (unknown keys / missing keys) loads gracefully, no splat-crash."""
    arms = [_arm("A")]

    def drive(arm: ArmSpec, resume: bool) -> int:
        return EXIT_DONE

    state = tmp_path / "state.json"
    # idx present, unknown future key present, some known keys absent.
    state.write_text('{"idx": 0, "stage": "setup", "future_field": 42}')
    r = Runner(arms, state, drive)
    assert r.run_once() == RUN_COMPLETE


def test_non_dict_state_refuses(tmp_path: Path) -> None:
    """L1/L2: a structurally-wrong (non-object) state refuses rather than crashing."""
    arms = [_arm("A")]

    def drive(arm: ArmSpec, resume: bool) -> int:
        return EXIT_DONE

    state = tmp_path / "state.json"
    state.write_text("[1, 2, 3]")
    r = Runner(arms, state, drive)
    assert r.run_once() == RUN_HELD


def test_headless_run_waits_for_reset(tmp_path: Path) -> None:
    arms = [_arm("A")]
    script = [EXIT_LIMIT, EXIT_DONE]
    waited: list[int] = []

    def drive(arm: ArmSpec, resume: bool) -> int:
        return script.pop(0)

    def wait_for_reset() -> None:
        waited.append(1)

    r = Runner(arms, tmp_path / "state.json", drive)
    assert r.run(wait_for_reset=wait_for_reset) == RUN_COMPLETE
    assert waited == [1]  # waited once on the stall
