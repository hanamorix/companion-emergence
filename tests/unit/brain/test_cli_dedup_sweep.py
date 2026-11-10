"""#502 (from #431): `nell dedup-sweep` reports an unreadable soul-exemption set in one closing line, not a traceback.

The loader's own warnings (logged with exc_info) still reach a terminal through Python's last-resort handler,
because the CLI configures no logging; that is the separate CLI-wide issue. This pins the part #502 names: the
fail-closed error is caught, reported in one line, and the command exits non-zero.
"""

from __future__ import annotations

import pytest

from brain.cli import main


def test_dedup_sweep_exits_nonzero_with_one_line_when_exemptions_cannot_be_read(
    tmp_path, monkeypatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    persona_dir = tmp_path / "personas" / "canary"
    persona_dir.mkdir(parents=True)
    (persona_dir / "soul_candidates.jsonl").mkdir()  # the soul-candidate read fails, so the sweep fails closed

    exit_code = main(["dedup-sweep", "--persona", "canary"])

    err = capsys.readouterr().err
    assert exit_code != 0
    assert err.strip().splitlines()[-1].startswith("dedup-sweep stopped, nothing was deleted: "), err
    assert "SoulExemptionsUnavailableError" not in err  # not an uncaught traceback
