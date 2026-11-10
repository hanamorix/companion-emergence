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

_TESTS = Path(__file__).resolve().parents[1]
_SELF = Path(__file__).resolve()

# setattr(sys, "platform", ...), setattr("sys.platform", ...), patch("sys.platform", ...), patch.object(sys, "platform", ...)
_PATCH = re.compile(
    r"""(?:setattr|patch(?:\.object)?)\(\s*(?:sys\s*,\s*["']platform["']|["']sys\.platform["'])"""
)

# Files left alone on purpose, each with why.
_ALLOWED = {
    # The script it tests is deleted by #635 (emotion rework Engine 7), which must delete this test with it.
    "unit/scripts/test_time_old_emotion_paths_host_line.py",
}


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


def test_no_test_patches_the_global_sys_platform():
    offenders = _offenders()
    assert not offenders, (
        "These tests patch the global sys.platform; patch the module's platform seam instead (#662, #710):\n"
        + "\n".join(offenders)
    )


def test_the_pattern_catches_each_form():
    for line in (
        'monkeypatch.setattr(sys, "platform", "win32")',
        "monkeypatch.setattr('sys.platform', 'darwin')",
        'mock.patch("sys.platform", "linux")',
        'mock.patch.object(sys, "platform", "linux")',
        'scoped.setattr(sys, "platform", name)',
    ):
        assert _PATCH.search(line), line
    assert not _PATCH.search('monkeypatch.setattr(mod, "_platform", lambda: "linux")')
    assert not _PATCH.search('monkeypatch.setattr("brain.service.current_platform", lambda: "darwin")')
