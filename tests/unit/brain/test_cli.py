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
