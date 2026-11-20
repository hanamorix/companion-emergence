"""The conftest fixture `_no_real_rebuild_child` makes `register_command_run` a no-op even for a command that imports
`brain.emotion.engine_files` after the fixture ran (Hana's 11-18 point 6 on #633).

The fixture used to patch the function only when the module was already in `sys.modules`: the first test in a process whose
command imported it lazily got the real function, which wrote a real `command_runs/<pid>.json` into its temp persona and
registered an `atexit` hook that later ran against a folder that was gone. Nothing in the suite removes the module from
`sys.modules`, so these tests do it themselves, before the conftest fixture runs, to be the "first test in its process".
"""

from __future__ import annotations

import atexit
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE = "brain.emotion.engine_files"


@pytest.fixture
def _engine_files_not_imported_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `brain.emotion.engine_files` look never imported.

    Teardown puts back what was there (the `sys.modules` entry and the package attribute, LIFO undo of one monkeypatch), so the
    original module object comes back only when it had been imported before this test. When it had not, there is nothing to put
    back: the copy the test imports stays in `sys.modules` and as the package attribute, which is the state an isolated run of
    this file is in anyway.
    """
    import brain.emotion

    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    monkeypatch.delattr(brain.emotion, "engine_files", raising=False)


@pytest.fixture(autouse=True)
def _no_real_rebuild_child(_engine_files_not_imported_yet: None, _no_real_rebuild_child: Any) -> Any:  # noqa: F811 - overrides the conftest fixture on purpose
    """Same fixture, requested after the module was removed from `sys.modules`: the conftest one then sees a fresh process."""
    return _no_real_rebuild_child


@pytest.fixture
def atexit_calls(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every `atexit.register` call, kept and not performed (a real hook would run at the end of the whole test session)."""
    calls: list[Any] = []
    monkeypatch.setattr(atexit, "register", lambda func, *args, **kwargs: calls.append((func, args)))
    return calls


@pytest.mark.no_base_failure(
    reason="the defect is in tests/conftest.py, which the check keeps at the PR head and runs against base product code; "
    "the red run is manual: revert only the fixture's sys.modules guard and this test fails (D-111 in the run records)"
)
def test_a_lazily_imported_register_command_run_writes_no_record_and_registers_no_exit_hook(
    tmp_path: Path, atexit_calls: list[Any]
) -> None:
    persona = tmp_path / "persona"
    persona.mkdir()
    engine_files = importlib.import_module(MODULE)  # the command's lazy import, after the fixture ran
    engine_files.register_command_run(persona)
    assert not (persona / "command_runs").exists() or not list((persona / "command_runs").iterdir()), "a real run record was written"
    assert atexit_calls == [], "an atexit hook was registered for a record that should not exist"


@pytest.mark.real_rebuild_step(reason="the real register_command_run, which the conftest otherwise makes a no-op")
@pytest.mark.no_base_failure(reason="control: the real function writes its record and registers its hook wherever the product has it, base included")
def test_the_real_function_does_write_the_record_and_register_the_hook(tmp_path: Path, atexit_calls: list[Any]) -> None:
    persona = tmp_path / "persona"
    persona.mkdir()
    engine_files = importlib.import_module(MODULE)
    engine_files.register_command_run(persona)
    assert len(list((persona / "command_runs").iterdir())) == 1
    assert len(atexit_calls) == 1
