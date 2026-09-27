"""`nell update` (#286 slice 2; closes #256)."""

from __future__ import annotations

import json
import shutil
import sys

import pytest

from brain import cli
from brain.update import overlay

# The bash handoff is POSIX-only: Windows has no updater yet (#255), so there
# the handler refuses instead of exec'ing bash.
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="no bash updater on Windows (#255)")


def test_status_reports_no_overlay(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    assert cli.main(["update", "--status"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["active"] is None and out["previous"] is None
    assert out["install_kind"] == "source" and out["supported"] is False


def _entry(d):
    return {"dir": d, "commit": d * 10, "brain_version": "0.0.42", "bundle_id": "b"}


def test_revert_and_rollback(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    root = overlay.overlay_root()
    overlay.activate(root, _entry("aaaa"))
    overlay.activate(root, _entry("bbbb"))
    assert cli.main(["update", "--rollback"]) == 0
    assert overlay.read_state(root)["active"]["dir"] == "aaaa"
    assert cli.main(["update", "--revert"]) == 0
    assert overlay.read_state(root) == {"active": None, "previous": _entry("aaaa")}


def test_revert_help_mentions_the_no_overlay_escape_hatch(capsys):
    with pytest.raises(SystemExit):
        cli.main(["update", "--help"])
    assert "KINDLED_NO_OVERLAY" in capsys.readouterr().out


def test_revert_works_with_kindled_no_overlay_set(tmp_path, monkeypatch, capsys):
    """KINDLED_NO_OVERLAY is a hook-activation escape hatch, not a CLI gate: --revert
    must still clear the active overlay when it's set (the documented recovery path
    for a bundle whose overlay brain can't start)."""
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    monkeypatch.setenv("KINDLED_NO_OVERLAY", "1")
    root = overlay.overlay_root()
    overlay.activate(root, _entry("aaaa"))
    assert cli.main(["update", "--revert"]) == 0
    assert overlay.read_state(root) == {"active": None, "previous": _entry("aaaa")}


def test_bundled_without_wheel_args_is_a_usage_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_install_kind", lambda: "bundled")
    assert cli.main(["update"]) == 2
    assert "--wheel" in capsys.readouterr().err


@posix_only
def test_source_install_execs_update_sh_with_passthrough_args(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: calls.append((path, argv)) or 0)
    cli.main(["update", "--", "--persona", "nell", "--dry-run"])
    assert calls and calls[0][0] == (shutil.which("bash") or "/bin/bash")
    assert calls[0][1][1].endswith("scripts/update.sh") or calls[0][1][1].endswith("scripts\\update.sh")
    assert calls[0][1][2:] == ["--persona", "nell", "--dry-run"]


@posix_only
def test_source_install_falls_back_to_bin_bash_when_bash_is_not_on_path(tmp_path, monkeypatch):
    """NixOS has no /bin/bash; shutil.which must be tried first (T4)."""
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: calls.append((path, argv)) or 0)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/bash" if name == "bash" else None)
    cli.main(["update", "--", "--dry-run"])
    assert calls and calls[0][0] == "/usr/bin/bash"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only refusal (#255)")
def test_source_install_on_windows_refuses_instead_of_exec(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: calls.append((path, argv)) or 0)
    assert cli.main(["update", "--", "--dry-run"]) == 2
    assert not calls
    assert "no bash updater" in capsys.readouterr().err


def test_oserror_from_the_overlay_lock_is_reported_not_a_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))

    def boom(src, dst, *a, **kw):
        raise OSError("no hard links on this filesystem")

    monkeypatch.setattr(overlay.os, "link", boom)
    rc = cli.main(["update", "--revert"])
    assert rc == 1
    assert "nell update:" in capsys.readouterr().err


def test_bundled_update_failure_exits_1_and_says_why(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_install_kind", lambda: "bundled")
    rc = cli.main(["update", "--wheel", str(tmp_path / "x.whl"),
                   "--requirements", str(tmp_path / "r.txt"), "--commit", "a" * 40])
    assert rc == 1 and "predates" in capsys.readouterr().err  # the dev venv has no _ce_bundle_id


def test_cli_reports_a_missing_requirements_file_cleanly(tmp_path, monkeypatch, capsys):
    from brain.update import install
    from tests.unit.brain.update.test_install import BRAIN_OK, _fake_bundle, _wheel

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_install_kind", lambda: "bundled")
    site = _fake_bundle(tmp_path, {})
    monkeypatch.setattr(install, "bundle_site_dir", lambda: site)
    brain_whl = _wheel(tmp_path, "companion-emergence", "9.9.9", BRAIN_OK)

    rc = cli.main(["update", "--wheel", str(brain_whl),
                   "--requirements", str(tmp_path / "missing.txt"), "--commit", "a" * 40])
    assert rc == 1
    assert "update failed" in capsys.readouterr().err
