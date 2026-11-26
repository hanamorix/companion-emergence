"""`scripts/release_preflight.sh` must actually call the gates that can fail (#363).

The preflight is a long bash script that can't be run in a unit test, so these pin the wiring a
later edit could silently drop: the pytest step goes through `pytest_verdict.sh` with pytest's exit
status, and the Windows lint runs as a blocking `--strict` step.
"""

from __future__ import annotations

from pathlib import Path

PREFLIGHT = (Path(__file__).resolve().parents[3] / "scripts" / "release_preflight.sh").read_text(
    encoding="utf-8"
)


def test_pytest_step_is_judged_by_the_verdict_script_with_the_exit_status() -> None:
    assert 'scripts/pytest_verdict.sh "$LOG_DIR/pytest.log" "$pytest_rc"' in PREFLIGHT
    assert "pytest_rc=$?" in PREFLIGHT
    assert "tail -1" not in PREFLIGHT.split("backend pytest", 1)[1].split("step \"ruff", 1)[0]


def test_windows_lint_runs_strict_in_the_preflight() -> None:
    assert "scripts/lint_windows_pitfalls.py --strict" in PREFLIGHT
