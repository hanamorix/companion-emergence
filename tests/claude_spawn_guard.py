"""A test must not start a real `claude` in CI (#540, J11).

The product spawns the claude CLI by its bare name (`subprocess.run(["claude", ...])`). On a CI runner there is no
`claude`, so a test that reaches that spawn without patching it does nothing useful: the spawn fails with
FileNotFoundError, and a test that tolerates the failure passes without having tested what it claims. This guard turns
the attempt into a loud failure of that test, with the three ways out in the message: patch the spawn, put a fake
`claude` under the test's tmp_path, or mark the test `@pytest.mark.requires_claude_cli`.

What counts as an attempt: a `subprocess.Popen` (so `subprocess.run`, `check_output` and asyncio subprocesses too) whose
program is the bare name `claude` (case-insensitive, `.exe`/`.cmd`/`.bat` stripped on Windows). A spawn by explicit
path is never blocked. A bare name is allowed only when it resolves, on the PATH of the env passed to the call (else the
process PATH), to a file under the system temp directory or pytest's base temp (a fake the test wrote). A bare name that
resolves to nothing is an attempt: the test relies on the spawn failing.

Where it applies: when `GITHUB_ACTIONS` is set (CI), or `KINDLED_GUARD_CLAUDE` is set (to try it locally), for every test
without the `requires_claude_cli` marker. A patched `subprocess.Popen` raises no audit event, so mocks are untouched by
construction. Limits: a spawn made by a child process of the test, or by a module- or session-scoped fixture, or by a
thread that was already running when the test began, is not seen or not blamed on the test.

Mechanism: one `sys.addaudithook` (cannot be removed; inert while no guarded test is running), a stack of active records
(the innermost test is blamed), and an autouse fixture that fails the test at teardown when the spawn error was swallowed.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

_SUFFIXES = (".exe", ".cmd", ".bat")

HELP = (
    "this test starts a real `claude` (the CLI is not installed on CI, so the spawn can only fail). Patch the spawn "
    "(monkeypatch subprocess.Popen or the provider's spawn helper), put a fake `claude` script under tmp_path and spawn it "
    "by explicit path or put its directory first on PATH, or mark the test @pytest.mark.requires_claude_cli."
)


def _first_token(command: str) -> str:
    """The program of a Windows command line: a leading double-quoted token, else up to the first whitespace."""
    command = command.lstrip()
    if command.startswith('"'):
        end = command.find('"', 1)
        return command[1:end] if end != -1 else command[1:]
    return command.split(None, 1)[0] if command.split(None, 1) else ""


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return os.fsdecode(value)
    return os.fspath(value) if isinstance(value, os.PathLike) else str(value)


def _program(executable: Any, argv: Any) -> str:
    """The program a Popen event starts. On Windows `argv` is already one command-line string and `executable` None."""
    if executable:
        return _as_text(executable)
    if isinstance(argv, (str, bytes)):
        return _first_token(_as_text(argv))
    if isinstance(argv, Sequence) and argv:
        return _as_text(argv[0])
    return ""


def _path_of(env: Mapping[str, str] | None, environ: Mapping[str, str]) -> str | None:
    source = env if env is not None else environ
    for key, value in source.items():
        if _as_text(key).upper() == "PATH":
            return _as_text(value)
    return None


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def violation(
    executable: Any,
    argv: Any,
    env: Mapping[str, str] | None,
    temp_roots: Iterable[str],
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Why this Popen event is an attempt to start the real `claude`, or None."""
    program = _program(executable, argv)
    if not program or os.sep in program or "/" in program or (os.altsep and os.altsep in program):
        return None
    name = program.lower()
    for suffix in _SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name != "claude":
        return None
    found = shutil.which(program, path=_path_of(env, os.environ if environ is None else environ))
    if found is None:
        return f"it spawns `{program}` by name and no such program exists on the PATH of the call"
    real = _norm(found)
    for root in temp_roots:
        if real == root or real.startswith(root + os.sep):
            return None
    return f"it spawns `{program}` by name and the name resolves to {found}"


@dataclass
class _Record:
    nodeid: str
    thread: int
    threads_before: frozenset[int]
    violations: list[str] = field(default_factory=list)


_ACTIVE: list[_Record] = []
_BASETEMP: list[str] = []
_INSTALLED = [False]


def _temp_roots() -> list[str]:
    return [_norm(tempfile.gettempdir()), *_BASETEMP]


def _where() -> str:
    """The innermost few frames outside the standard library, pytest and this file: where the spawn comes from."""
    skip = (os.sep + "lib" + os.sep + "python", os.sep + "site-packages" + os.sep, os.path.abspath(__file__))
    frames = [f for f in traceback.extract_stack() if not any(part in f.filename for part in skip)]
    return "\n".join(f"  from {f.filename}:{f.lineno} in {f.name}" for f in frames[-4:])


def _audit(event: str, args: tuple) -> None:
    if event != "subprocess.Popen" or not _ACTIVE:
        return
    try:
        record = _ACTIVE[-1]
        me = threading.get_ident()
        if me != record.thread and me in record.threads_before:
            return  # a thread that was already running when this test began is not this test's doing
        executable, argv, _cwd, env = args
        message = violation(executable, argv, env, _temp_roots())
    except Exception:  # noqa: BLE001 - an analysis bug must never break an unrelated spawn
        return
    if message:
        message = f"{message}\n{_where()}"
        record.violations.append(message)
        raise RuntimeError(f"{record.nodeid}: {message}\n{HELP}")


def install() -> None:
    """Add the audit hook once per process (an audit hook can never be removed)."""
    if not _INSTALLED[0]:
        sys.addaudithook(_audit)
        _INSTALLED[0] = True


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return bool(env.get("GITHUB_ACTIONS") or env.get("KINDLED_GUARD_CLAUDE"))


@pytest.fixture(autouse=True)
def no_real_claude_in_ci(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Fail the test that starts a real `claude` in CI (see the module docstring)."""
    if not enabled() or request.node.get_closest_marker("requires_claude_cli"):
        yield
        return
    install()
    base = _norm(str(tmp_path_factory.getbasetemp()))
    if base not in _BASETEMP:
        _BASETEMP.append(base)
    me = threading.get_ident()
    record = _Record(
        nodeid=request.node.nodeid,
        thread=me,
        threads_before=frozenset(t.ident for t in threading.enumerate() if t.ident is not None and t.ident != me),
    )
    _ACTIVE.append(record)
    try:
        yield
    finally:
        _ACTIVE.remove(record)
    if record.violations:
        pytest.fail(f"{record.nodeid}: {record.violations[0]}\n{HELP}", pytrace=False)
