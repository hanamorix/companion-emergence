"""Unit tests for brain/body/state.py — the body projection rules shared by every body state.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §3.1.

The session, words and contact-gap rules, the clamps and the exhaustion derivation are tested here through
`body_state_from_short`, the one function that builds a body state. Which feelings count as high is the engine's
predicate (`feels_high`); its per-feeling effects are in test_body_state_from_short.py.

Spec section 7 (inviolate property 4): "the body projection blocks chat composition because it's slow: pure function,
sub-ms. No I/O. No LLM call." The spec asked for a timing test; a wall-clock budget cannot tell a fast I/O call from a
pure one and flakes under load (removed in #540), so the property is held structurally by the last two tests: the
module imports only pure modules and calls nothing that does I/O (static, branch-independent), and running the
function over a grid that reaches every branch of the projection fires no I/O audit event (behaviour). The inputs
the caller reads (store rows, session hours, words) are the caller's; they are outside the property.
"""

from __future__ import annotations

import ast
import sys
import threading
from datetime import UTC, datetime

import pytest

from brain.body.state import BodyState, body_state_from_short
from tests.source_tree import brain_trees


def _now() -> datetime:
    return datetime(2026, 4, 30, 12, 0, 0, tzinfo=UTC)


def _body(
    *high: str,
    session_hours: float = 0.0,
    words_written: int = 0,
    days_since_contact: float = 0.0,
) -> BodyState:
    """The body state with the named feelings high and every other feeling not high."""
    return body_state_from_short(
        {},
        feels_high=lambda name: name in high,
        session_hours=session_hours,
        words_written=words_written,
        days_since_contact=days_since_contact,
        now=_now(),
    )


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
    assert _body().energy == 8


def test_energy_session_band_60_to_120():
    # band: -1 (>60min); continuous: -0.75 → energy 6.25 → round 6
    assert _body(session_hours=1.5).energy == 6


def test_energy_session_band_120_to_180():
    # band: -2 (>120min); continuous: -1.25 → energy 4.75 → round 5
    assert _body(session_hours=2.5).energy == 5


def test_energy_session_band_over_180():
    # band: -3 (>180min); continuous: -1.75 → energy 3.25 → round 3
    assert _body(session_hours=3.5).energy == 3


def test_energy_words_drain():
    # band: 0 (not >60min); continuous: -0.25; words: -1.0 → energy 6.75 → round 7
    assert _body(session_hours=0.5, words_written=2500).energy == 7


def test_energy_clamped_at_1():
    assert _body("rest_need", session_hours=10.0, words_written=20000).energy == 1


def test_temperature_baseline():
    assert _body().temperature == 4


def test_temperature_full_warmth_stack():
    # 4 + one for each of arousal, desire, belonging, love, climax = 9 (the top of the 1-9 range, not 10)
    assert _body("arousal", "desire", "belonging", "love", "climax").temperature == 9


def test_temperature_long_no_contact():
    assert _body(days_since_contact=8.0).temperature == 2  # 4 - 2 (>7 days)


def test_temperature_medium_no_contact():
    assert _body(days_since_contact=4.0).temperature == 3  # 4 - 1 (>3 days)


def test_temperature_clamped_at_1():
    # 4 - 1 - 1 - 2 = 0 → clamp 1
    assert _body("body_grief", "touch_hunger", days_since_contact=10.0).temperature == 1


def test_exhaustion_derivation():
    # energy 8 → exhaustion max(0, 7-8) = 0
    assert _body().exhaustion == 0


def test_exhaustion_high_when_energy_low():
    bs = _body("rest_need", session_hours=10.0, words_written=20000)
    # energy clamped at 1 → exhaustion 6
    assert bs.energy == 1
    assert bs.exhaustion == 6


# ---------------------------------------------------------------------------
# Spec section 7, inviolate property 4: the body projection does no I/O and makes no LLM call
# ---------------------------------------------------------------------------

_STATE_FILE = "brain/body/state.py"
# What the module may import: nothing that reaches a file, a socket, a process, a database, a clock-wait or a model.
_PURE_MODULES = frozenset({"__future__", "collections.abc", "dataclasses", "datetime", "typing"})
# A brain module may be added here only after checking that it does no I/O; it is then scanned by the same rule.
_PURE_BRAIN_MODULES: tuple[str, ...] = ("brain.dev_constants",)  # constants only: no import but `__future__`, no call
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


@pytest.mark.slow(reason='1.6 s of CPU in its call phase on a 4-core dev VM, 1.6 s wall (serial run for #540, J12)')
@pytest.mark.no_base_failure(reason='#540 audit guard, snapshot or registry of what already exists (it replaces removed tests); passes on the old code by design')
def test_the_body_state_module_imports_only_pure_modules_and_calls_no_io() -> None:
    trees = brain_trees()
    assert _STATE_FILE in trees, f"{_STATE_FILE} moved: point this guard at the module that holds body_state_from_short"
    assert body_state_from_short.__module__ == "brain.body.state", "body_state_from_short moved out of the guarded module"
    allowed = _PURE_MODULES | set(_PURE_BRAIN_MODULES)
    scanned = [_STATE_FILE] + [m.replace(".", "/") + ".py" for m in _PURE_BRAIN_MODULES]
    problems = [
        f"{rel}: {finding}"
        for rel in scanned
        for finding in impure_findings(trees[rel], allowed)
    ]
    assert not problems, (
        "the body projection must stay a pure function (body spec section 7, property 4: it must not block chat "
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


@pytest.mark.no_base_failure(reason='#540 audit guard, renamed from the audit test of the deleted fixed-level projection and now driving body_state_from_short; the property it checks holds on the old code by design')
def test_body_state_from_short_raises_no_io_audit_event() -> None:
    if not _AUDIT["installed"]:  # an audit hook cannot be removed, so it is installed once, here, and inert otherwise
        sys.addaudithook(_audit_hook)
        _AUDIT["installed"] = True
    every = (
        "love", "joy", "grief", "peace", "belonging", "arousal", "desire", "climax", "touch_hunger", "comfort_seeking",
        "rest_need", "body_grief",
    )  # fmt: skip
    short = dict.fromkeys(every, 5.0)
    high_grid = [
        frozenset(),
        frozenset(every),  # every feeling high: more than six loud, every warming and cooling feeling
        frozenset({"peace"}),
        frozenset({"body_grief", "touch_hunger"}),
        frozenset({"rest_need"}),
        frozenset({"arousal", "desire", "belonging", "love", "climax"}),
    ]
    grid = [
        (shorts, high, hours, words, days)
        for shorts in ({}, short)  # an empty read and a full one
        for high in high_grid
        for hours in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)  # both sides of 60, 120 and 180 minutes
        for words in (0, 5000)
        for days in (0.0, 3.0, 5.0, 7.0, 40.0)  # both sides of 3 and 7 days
    ]
    now = _now()
    # One call first, outside the window: a tracer in this thread (coverage) reads and compiles state.py the first time
    # it sees it, which is not the function's own I/O.
    body_state_from_short(
        {}, feels_high=lambda name: False, session_hours=0.0, words_written=0, days_since_contact=0.0, now=now
    )
    _AUDIT.update(events=[], thread=threading.get_ident(), on=True)
    try:
        for shorts, high, hours, words, days in grid:
            body_state_from_short(
                shorts, feels_high=high.__contains__, session_hours=hours, words_written=words,
                days_since_contact=days, now=now,
            )
    finally:
        _AUDIT["on"] = False
    assert _AUDIT["events"] == [], (
        "body_state_from_short did I/O (body spec section 7, property 4: it must not block chat composition): "
        f"{sorted(set(_AUDIT['events']))}"
    )
