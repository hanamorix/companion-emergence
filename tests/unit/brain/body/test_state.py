"""Unit tests for brain/body/state.py — compute_body_state pure function.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §3.1.

Spec section 7 (inviolate property 4): "compute_body_state blocks chat composition because it's slow: pure function,
sub-ms. No I/O. No LLM call." The spec asked for a timing test; a wall-clock budget cannot tell a fast I/O call from a
pure one and flakes under load (removed in #540), so the property is held structurally by the last two tests: the
module imports only pure modules and calls nothing that does I/O (static, branch-independent), and running the
function over a grid that reaches every branch of both projections fires no I/O audit event (behaviour). The inputs
the caller reads (store rows, session hours, words) are the caller's; they are outside the property.
"""

from __future__ import annotations

import ast
import sys
import threading
from datetime import UTC, datetime

import pytest

from brain.body.state import BodyState, compute_body_state
from tests.source_tree import brain_trees


def _now() -> datetime:
    return datetime(2026, 4, 30, 12, 0, 0, tzinfo=UTC)


def test_body_state_to_dict_shape():
    bs = BodyState(
        energy=8,
        temperature=4,
        exhaustion=0,
        session_hours=0.0,
        days_since_contact=0.0,
        body_emotions={
            "arousal": 0.0,
            "desire": 0.0,
            "climax": 0.0,
            "touch_hunger": 0.0,
            "comfort_seeking": 0.0,
            "rest_need": 0.0,
        },
        computed_at=_now(),
    )
    d = bs.to_dict()
    assert d["loaded"] is True
    assert d["energy"] == 8
    assert d["temperature"] == 4
    assert d["exhaustion"] == 0
    assert d["session_hours"] == 0.0
    assert d["days_since_contact"] == 0.0
    assert set(d["body_emotions"].keys()) == {
        "arousal",
        "desire",
        "climax",
        "touch_hunger",
        "comfort_seeking",
        "rest_need",
    }
    assert "computed_at" in d
    assert d["computed_at"].endswith("+00:00") or d["computed_at"].endswith("Z")


def test_energy_baseline_no_inputs():
    bs = compute_body_state(
        emotions={},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    assert bs.energy == 8


def test_energy_session_band_60_to_120():
    bs = compute_body_state(
        emotions={},
        session_hours=1.5,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # band: -1 (>60min); continuous: -0.75 → energy 6.25 → round 6
    assert bs.energy == 6


def test_energy_session_band_120_to_180():
    bs = compute_body_state(
        emotions={},
        session_hours=2.5,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # band: -2 (>120min); continuous: -1.25 → energy 4.75 → round 5
    assert bs.energy == 5


def test_energy_session_band_over_180():
    bs = compute_body_state(
        emotions={},
        session_hours=3.5,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # band: -3 (>180min); continuous: -1.75 → energy 3.25 → round 3
    assert bs.energy == 3


def test_energy_words_drain():
    bs = compute_body_state(
        emotions={},
        session_hours=0.5,
        words_written=2500,
        days_since_contact=0.0,
        now=_now(),
    )
    # band: 0 (not >60min); continuous: -0.25; words: -1.0 → energy 6.75 → round 7
    assert bs.energy == 7


def test_energy_high_emotional_load_drain():
    emotions = {f"emo{i}": 8.0 for i in range(7)}  # 7 high emotions
    bs = compute_body_state(
        emotions=emotions,
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # 8 baseline - 1 (>6 high emotions) = 7
    assert bs.energy == 7


def test_energy_rest_need_drain():
    bs = compute_body_state(
        emotions={"rest_need": 8.0},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # 8 - 1 (rest_need >= 7) = 7
    assert bs.energy == 7


def test_energy_peace_restoration_fresh_session():
    bs = compute_body_state(
        emotions={"peace": 8.0},
        session_hours=0.5,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # 8 baseline - 0.25 (continuous) + 1 (peace, fresh) = 8.75 → round 9
    assert bs.energy == 9


def test_energy_peace_no_restoration_old_session():
    bs = compute_body_state(
        emotions={"peace": 8.0},
        session_hours=2.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # peace bonus blocked (session_hours >= 1); band: 0 (120min not >120min);
    # continuous: -1.0; actual energy 6
    assert bs.energy == 6


def test_energy_clamped_at_1():
    bs = compute_body_state(
        emotions={"rest_need": 8.0},
        session_hours=10.0,
        words_written=20000,
        days_since_contact=0.0,
        now=_now(),
    )
    assert bs.energy == 1


def test_energy_clamped_at_10():
    """No path through current formula reaches 10, but clamp must hold
    against a future tweak that adds another bonus term."""
    # Pretend a future term added +5; today we just hand-check the upper clamp:
    bs = compute_body_state(
        emotions={"peace": 8.0},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # 8 + 1 (peace, fresh) = 9 → still under 10
    assert bs.energy == 9
    assert bs.energy <= 10


def test_temperature_baseline():
    bs = compute_body_state(
        emotions={},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    assert bs.temperature == 4


def test_temperature_arousal_warm():
    bs = compute_body_state(
        emotions={"arousal": 8.0},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    assert bs.temperature == 5  # 4 + 1


def test_temperature_full_warmth_stack():
    bs = compute_body_state(
        emotions={
            "arousal": 8.0,
            "desire": 8.0,
            "belonging": 9.0,
            "love": 9.0,
            "climax": 6.0,
        },
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # 4 + 1 + 1 + 1 + 1 + 1 = 9 (max)
    assert bs.temperature == 9


def test_temperature_body_grief_cold():
    bs = compute_body_state(
        emotions={"body_grief": 8.0},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    assert bs.temperature == 3  # 4 - 1


def test_temperature_long_no_contact():
    bs = compute_body_state(
        emotions={},
        session_hours=0.0,
        words_written=0,
        days_since_contact=8.0,
        now=_now(),
    )
    assert bs.temperature == 2  # 4 - 2 (>7 days)


def test_temperature_medium_no_contact():
    bs = compute_body_state(
        emotions={},
        session_hours=0.0,
        words_written=0,
        days_since_contact=4.0,
        now=_now(),
    )
    assert bs.temperature == 3  # 4 - 1 (>3 days)


def test_temperature_clamped_at_1():
    bs = compute_body_state(
        emotions={"body_grief": 8.0, "touch_hunger": 8.0},
        session_hours=0.0,
        words_written=0,
        days_since_contact=10.0,
        now=_now(),
    )
    # 4 - 1 - 1 - 2 = 0 → clamp 1
    assert bs.temperature == 1


def test_temperature_asymmetric_range_top_is_9():
    """Spec: temperature 1-9, NOT 1-10."""
    bs = compute_body_state(
        emotions={
            "arousal": 8.0,
            "desire": 8.0,
            "belonging": 9.0,
            "love": 9.0,
            "climax": 6.0,
            "extra1": 0.0,
        },
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    assert bs.temperature <= 9


def test_exhaustion_derivation():
    bs = compute_body_state(
        emotions={},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # energy 8 → exhaustion max(0, 7-8) = 0
    assert bs.exhaustion == 0


def test_exhaustion_high_when_energy_low():
    bs = compute_body_state(
        emotions={"rest_need": 8.0},
        session_hours=10.0,
        words_written=20000,
        days_since_contact=0.0,
        now=_now(),
    )
    # energy clamped at 1 → exhaustion 6
    assert bs.energy == 1
    assert bs.exhaustion == 6


def test_body_emotions_dict_includes_all_six_with_zero_default():
    bs = compute_body_state(
        emotions={"arousal": 5.0},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    expected = {"arousal", "desire", "climax", "touch_hunger", "comfort_seeking", "rest_need"}
    assert set(bs.body_emotions.keys()) == expected
    assert bs.body_emotions["arousal"] == 5.0
    assert bs.body_emotions["desire"] == 0.0
    assert bs.body_emotions["climax"] == 0.0


# ---------------------------------------------------------------------------
# Spec section 7, inviolate property 4: compute_body_state does no I/O and makes no LLM call
# ---------------------------------------------------------------------------

_STATE_FILE = "brain/body/state.py"
# What the module may import: nothing that reaches a file, a socket, a process, a database, a clock-wait or a model.
_PURE_MODULES = frozenset({"__future__", "collections.abc", "dataclasses", "datetime", "typing"})
# A brain module may be added here only after checking that it does no I/O; it is then scanned by the same rule.
_PURE_BRAIN_MODULES: tuple[str, ...] = ()
_FORBIDDEN_CALLS = frozenset({"__import__", "open", "input", "exec", "eval", "compile", "print"})


def impure_findings(tree: ast.AST, allowed: frozenset[str]) -> list[str]:
    """Every import outside ``allowed`` (anywhere in the file, relative imports included) and every call of a builtin
    that reads, writes or runs code, as ``line N: what``."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [f"line {node.lineno}: import {a.name}" for a in node.names if a.name not in allowed]
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            if node.level or module not in allowed:
                found.append(f"line {node.lineno}: from {module} import ...")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            found.append(f"line {node.lineno}: call of {node.func.id}()")
    return found


@pytest.mark.slow(reason='1.9 s on windows in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
@pytest.mark.no_base_failure(reason='#540 audit guard, snapshot or registry of what already exists (it replaces removed tests); passes on the old code by design')
def test_the_body_state_module_imports_only_pure_modules_and_calls_no_io() -> None:
    trees = brain_trees()
    assert _STATE_FILE in trees, f"{_STATE_FILE} moved: point this guard at the module that holds compute_body_state"
    assert compute_body_state.__module__ == "brain.body.state", "compute_body_state moved out of the guarded module"
    allowed = _PURE_MODULES | set(_PURE_BRAIN_MODULES)
    scanned = [_STATE_FILE] + [m.replace(".", "/") + ".py" for m in _PURE_BRAIN_MODULES]
    problems = [
        f"{rel}: {finding}"
        for rel in scanned
        for finding in impure_findings(trees[rel], allowed)
    ]
    assert not problems, (
        "compute_body_state must stay a pure function (body spec section 7, property 4: it must not block chat "
        "composition: no I/O, no LLM call). Read the inputs in the caller and pass them in; a pure standard-library "
        "module (math, re, enum) may be added to _PURE_MODULES after checking it:\n  " + "\n  ".join(problems)
    )


_AUDIT: dict = {"installed": False, "on": False, "thread": 0, "events": []}
_IO_EVENTS = (
    "open", "socket.", "subprocess.", "sqlite3.", "os.", "shutil.", "time.sleep", "urllib.", "http.", "ssl.", "glob.",
    "tempfile.", "ctypes.", "import", "exec", "compile", "fcntl.", "mmap.",
)  # fmt: skip


def _audit_hook(event: str, args: tuple) -> None:
    if _AUDIT["on"] and threading.get_ident() == _AUDIT["thread"] and event.startswith(_IO_EVENTS):
        _AUDIT["events"].append(event)


@pytest.mark.no_base_failure(reason='#540 audit guard, snapshot or registry of what already exists (it replaces removed tests); passes on the old code by design')
def test_compute_body_state_raises_no_io_audit_event() -> None:
    if not _AUDIT["installed"]:  # an audit hook cannot be removed, so it is installed once, here, and inert otherwise
        sys.addaudithook(_audit_hook)
        _AUDIT["installed"] = True
    every = (
        "love", "joy", "grief", "peace", "belonging", "arousal", "desire", "climax", "touch_hunger", "comfort_seeking",
        "rest_need", "body_grief",
    )  # fmt: skip
    emotions_grid = [
        {},
        *(dict.fromkeys(every, level) for level in (5.0, 7.0, 8.0)),  # every threshold of both projections, and > 6 loud
        {"peace": 8.0},
        {"body_grief": 7.0, "touch_hunger": 7.0},
        {"rest_need": 7.0},
    ]
    grid = [
        (emotions, hours, words, days)
        for emotions in emotions_grid
        for hours in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)  # both sides of 60, 120 and 180 minutes
        for words in (0, 5000)
        for days in (0.0, 3.0, 5.0, 7.0, 40.0)  # both sides of 3 and 7 days
    ]
    now = _now()
    # One call first, outside the window: a tracer in this thread (coverage) reads and compiles state.py the first time
    # it sees it, which is not the function's own I/O.
    compute_body_state(emotions={}, session_hours=0.0, words_written=0, days_since_contact=0.0, now=now)
    _AUDIT.update(events=[], thread=threading.get_ident(), on=True)
    try:
        for emotions, hours, words, days in grid:
            compute_body_state(
                emotions=emotions, session_hours=hours, words_written=words, days_since_contact=days, now=now
            )
    finally:
        _AUDIT["on"] = False
    assert _AUDIT["events"] == [], (
        "compute_body_state did I/O (body spec section 7, property 4: it must not block chat composition): "
        f"{sorted(set(_AUDIT['events']))}"
    )
