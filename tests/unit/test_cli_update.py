"""`nell update` (#286 slice 2; closes #256)."""

from __future__ import annotations

import json

from brain import cli
from brain.update import overlay


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


def test_bundled_without_wheel_args_is_a_usage_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_install_kind", lambda: "bundled")
    assert cli.main(["update"]) == 2
    assert "--wheel" in capsys.readouterr().err


def test_source_install_execs_update_sh_with_passthrough_args(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: calls.append((path, argv)) or 0)
    cli.main(["update", "--", "--persona", "nell", "--dry-run"])
    assert calls and calls[0][0] == "/bin/bash"
    assert calls[0][1][1].endswith("scripts/update.sh") or calls[0][1][1].endswith("scripts\\update.sh")
    assert calls[0][1][2:] == ["--persona", "nell", "--dry-run"]


def test_bundled_update_failure_exits_1_and_says_why(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_install_kind", lambda: "bundled")
    rc = cli.main(["update", "--wheel", str(tmp_path / "x.whl"),
                   "--requirements", str(tmp_path / "r.txt"), "--commit", "a" * 40])
    assert rc == 1 and "predates" in capsys.readouterr().err  # the dev venv has no _ce_bundle_id
