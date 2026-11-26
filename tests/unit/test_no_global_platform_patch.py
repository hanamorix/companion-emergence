"""#710: no test patches the global ``sys.platform``.

Patching it reaches the stdlib and process-wide caches, not just the code under
test: #662 was a test faking Windows that poisoned ``own_token`` for every test
after it. A test that needs another platform patches the module's own seam instead
(a small ``_platform()`` or ``_is_windows()`` function, as ``respawn_guard`` and
``provider`` have).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[1]
_SELF = Path(__file__).resolve()

# setattr(sys, "platform", ...), setattr("sys.platform", ...), patch("sys.platform", ...), patch.object(sys, "platform", ...),
# and each with the module path in front (``kill.sys``, ``"brain.bridge.kill.sys.platform"``): a module's ``sys`` is the
# same global object.
_PATCH = re.compile(
    r"""(?:setattr|patch(?:\.object)?)\(\s*(?:[\w.]*\bsys\s*,\s*["']platform["']|["'][\w.]*\bsys\.platform["'])"""
)

# Files left alone on purpose, each with why.
_ALLOWED: set[str] = set()


def _offenders() -> list[str]:
    found = []
    for path in sorted(_TESTS.rglob("*.py")):
        if path.resolve() == _SELF:
            continue
        rel = path.relative_to(_TESTS).as_posix()
        if rel in _ALLOWED:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _PATCH.search(line):
                found.append(f"tests/{rel}:{n}: {line.strip()}")
    return found


@pytest.mark.no_base_failure(
    reason="test-suite machinery (#710): it checks the test files, which the base-code swap keeps at the head; on main's own test files it finds 47 global patches"
)
def test_no_test_patches_the_global_sys_platform():
    offenders = _offenders()
    assert not offenders, (
        "These tests patch the global sys.platform; patch the module's platform seam instead (#662, #710):\n"
        + "\n".join(offenders)
    )


@pytest.mark.no_base_failure(
    reason="test-suite machinery (#710): checks the guard's own pattern, which no product code affects"
)
def test_the_pattern_catches_each_form():
    for line in (
        'monkeypatch.setattr(sys, "platform", "win32")',
        "monkeypatch.setattr('sys.platform', 'darwin')",
        'mock.patch("sys.platform", "linux")',
        'mock.patch.object(sys, "platform", "linux")',
        'scoped.setattr(sys, "platform", name)',
        # A module's own ``sys`` is the same global object (ToT's review of #715).
        'monkeypatch.setattr(kill.sys, "platform", "win32")',
        'monkeypatch.setattr("brain.bridge.kill.sys.platform", "win32")',
    ):
        assert _PATCH.search(line), line
    assert not _PATCH.search('monkeypatch.setattr(mod, "_platform", lambda: "linux")')
    assert not _PATCH.search(
        'monkeypatch.setattr("brain.service.current_platform", lambda: "darwin")'
    )
