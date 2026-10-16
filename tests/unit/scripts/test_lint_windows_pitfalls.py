"""`scripts/lint_windows_pitfalls.py` — blocking, not advisory (#363).

The lint existed but nothing ran it, so it could not stop anything: on 2026-10-12 it already had a
real finding (`schtasks` read in text mode with no `encoding=`). It now runs in pytest and in
`scripts/release_preflight.sh --strict`, so `brain/` has to stay clean of unreviewed findings.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_SCRIPT = REPO / "scripts" / "lint_windows_pitfalls.py"


def _load():
    spec = importlib.util.spec_from_file_location("lint_windows_pitfalls", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lint_windows_pitfalls"] = mod
    spec.loader.exec_module(mod)
    return mod


lint = _load()


def test_brain_tree_has_no_unreviewed_windows_pitfalls() -> None:
    assert lint.scan() == []


def _tree(tmp_path: Path, source: str) -> Path:
    (tmp_path / "brain").mkdir(parents=True)
    (tmp_path / "brain" / "mod.py").write_text(source, encoding="utf-8")
    return tmp_path


def test_a_reviewed_text_mode_spawn_with_a_marker_is_not_flagged(tmp_path) -> None:
    flagged = "import subprocess\nsubprocess.run(['x'], text=True)\n"
    marked = (
        "import subprocess\n"
        "subprocess.run(  # windows-lint: ok -- x only exists on Linux\n"
        "    ['x'], text=True)\n"
    )
    assert len(lint.scan(_tree(tmp_path / "a", flagged))) == 1
    assert lint.scan(_tree(tmp_path / "b", marked)) == []


def test_an_unmarked_os_kill_is_flagged_and_a_marked_one_is_not(tmp_path) -> None:
    flagged = "import os\ndef stop(pid, sig):\n    os.kill(pid, sig)\n"
    marked = "import os\ndef stop(pid, sig):\n    os.kill(pid, sig)  # windows-lint: ok -- reviewed\n"
    assert len(lint.scan(_tree(tmp_path / "a", flagged))) == 1
    assert lint.scan(_tree(tmp_path / "b", marked)) == []


def test_a_marker_on_a_neighbouring_spawn_does_not_exempt_an_unmarked_one(tmp_path) -> None:
    # #398 review: the check used a fixed 500-character window after `subprocess.run(`, so a
    # reviewed spawn within 500 characters below an unreviewed one switched the lint off for it.
    source = (
        "import subprocess\n"
        "def a():\n"
        "    subprocess.run(['x'], text=True)\n"
        "def b():\n"
        "    subprocess.run(  # windows-lint: ok -- b only exists on Linux\n"
        "        ['y'], text=True)\n"
    )
    findings = lint.scan(_tree(tmp_path, source))
    assert len(findings) == 1
    assert findings[0].startswith("brain/mod.py:3:")


def test_encoding_on_a_neighbouring_spawn_does_not_exempt_an_unmarked_one(tmp_path) -> None:
    # same window, same hole: `encoding=` belonging to the next call hid this one
    source = (
        "import subprocess\n"
        "def a():\n"
        "    subprocess.run(['x'], text=True)\n"
        "def b():\n"
        "    subprocess.run(['y'], text=True, encoding='utf-8')\n"
    )
    findings = lint.scan(_tree(tmp_path, source))
    assert len(findings) == 1
    assert findings[0].startswith("brain/mod.py:3:")


def test_a_marker_cannot_exempt_an_os_kill_liveness_probe(tmp_path) -> None:
    # signal 0 is CTRL_C_EVENT on Windows; the only sanctioned probe is state_file.pid_is_alive
    source = "import os\ndef alive(pid):\n    os.kill(pid, 0)  # windows-lint: ok -- reviewed\n"
    findings = lint.scan(_tree(tmp_path, source))
    assert len(findings) == 1
    assert "liveness probe" in findings[0]
