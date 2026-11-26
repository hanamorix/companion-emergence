"""`scripts/pytest_verdict.sh` — how release_preflight.sh judges the full pytest run (#363).

The old step grepped only `^FAILED` and then accepted any last line containing "passed", never
reading pytest's exit status. A run ending `ERROR tests/x.py::t - boom` / `5 passed, 1 error in 1.2s`
reported green here, then went red in release.yml's `validate` job after the tag existed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash script; the preflight is mac/linux")

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "pytest_verdict.sh"
BASH = "/bin/bash"


def _verdict(tmp_path: Path, log: str, rc: int) -> subprocess.CompletedProcess:
    f = tmp_path / "pytest.log"
    f.write_text(log, encoding="utf-8")
    return subprocess.run(
        [BASH, str(SCRIPT), str(f), str(rc)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_an_error_only_run_is_red_even_though_the_summary_says_passed(tmp_path) -> None:
    log = (
        "tests/x.py .....\n"
        "=========== short test summary info ===========\n"
        "ERROR tests/x.py::t - boom\n"
        "5 passed, 1 error in 1.2s\n"
    )
    r = _verdict(tmp_path, log, rc=1)
    assert r.returncode == 1
    assert "ERROR tests/x.py::t - boom" in r.stderr


def test_a_failed_line_is_red(tmp_path) -> None:
    r = _verdict(tmp_path, "FAILED tests/x.py::t - assert 1 == 2\n1 failed, 4 passed in 1s\n", rc=1)
    assert r.returncode == 1
    assert "FAILED tests/x.py::t" in r.stderr


def test_a_nonzero_exit_without_failed_or_error_lines_is_red(tmp_path) -> None:
    # interrupted / usage error / nothing collected: no FAILED line, but not a green run either
    r = _verdict(tmp_path, "4 passed in 1s\n", rc=2)
    assert r.returncode == 1
    assert "exited 2" in r.stderr


def test_a_clean_run_is_green_and_shutdown_noise_after_the_summary_is_ignored(tmp_path) -> None:
    log = "6591 passed, 17 skipped, 5 xfailed in 481.2s\nAttributeError: resource_tracker shutdown\n"
    r = _verdict(tmp_path, log, rc=0)
    assert r.returncode == 0
    assert "6591 passed" in r.stdout


def test_an_error_line_is_red_even_if_the_exit_status_claims_success(tmp_path) -> None:
    # belt and braces: a wrapper or plugin that swallows the status must not hide an ERROR row
    r = _verdict(tmp_path, "ERROR tests/x.py::t - boom\n5 passed, 1 error in 1.2s\n", rc=0)
    assert r.returncode == 1
    assert "ERROR tests/x.py::t - boom" in r.stderr
