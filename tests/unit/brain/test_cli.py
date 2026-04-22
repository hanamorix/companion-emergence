"""Tests for brain.cli entry point."""

from __future__ import annotations

import pytest

from brain import cli


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    """`nell --version` prints version and exits with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "companion-emergence" in captured.out


def test_no_args_prints_help_and_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`nell` with no args shows help and returns 1."""
    result = cli.main([])
    assert result == 1
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()


STUB_COMMANDS = [
    "supervisor",
    "dream",
    "heartbeat",
    "reflex",
    "status",
    "rest",
    "soul",
    "memory",
    "works",
    "migrate",
]


@pytest.mark.parametrize("name", STUB_COMMANDS)
def test_stub_subcommand_runs_and_reports_not_implemented(
    capsys: pytest.CaptureFixture[str], name: str
) -> None:
    """Every stub subcommand exits 0 and prints 'not implemented yet'."""
    result = cli.main([name])
    assert result == 0
    captured = capsys.readouterr()
    assert "not implemented" in captured.out.lower()
    assert name in captured.out


def test_stub_subcommand_help_works(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each stub subcommand supports --help without crashing."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["supervisor", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "supervisor" in captured.out.lower()
