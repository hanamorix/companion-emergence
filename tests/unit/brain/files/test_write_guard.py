from pathlib import Path

import pytest

from brain.files.write_guard import GuardResult, check_write_target


def _home(tmp_path) -> Path:
    return tmp_path / "home"


def test_create_in_ordinary_dir_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: _home(tmp_path))
    target = _home(tmp_path) / "Documents" / "note.md"
    r = check_write_target(str(target), op="create", persona_dir=tmp_path / "persona")
    assert r.ok and r.resolved == target.resolve()


@pytest.mark.parametrize("rel", [
    ".zshrc", ".bashrc", ".bash_profile", ".profile", ".zprofile", ".zshenv",
    ".ssh/authorized_keys", ".ssh/config", ".aws/credentials", ".gnupg/x", ".netrc",
    "Library/LaunchAgents/x.plist",
])
def test_home_sensitive_paths_refused(tmp_path, monkeypatch, rel):
    h = _home(tmp_path); monkeypatch.setattr(Path, "home", lambda: h)
    (h / Path(rel).parent).mkdir(parents=True, exist_ok=True)
    (h / rel).write_text("x")  # exist so it's an append target too
    r = check_write_target(str(h / rel), op="append", persona_dir=tmp_path / "persona")
    assert not r.ok and "denied" in r.error.lower()


@pytest.mark.parametrize("p", ["/etc/hosts", "/usr/bin/x", "/System/x", "/bin/x", "/Library/x"])
def test_system_roots_refused(tmp_path, monkeypatch, p):
    monkeypatch.setattr(Path, "home", lambda: _home(tmp_path))
    r = check_write_target(p, op="append", persona_dir=tmp_path / "persona")
    assert not r.ok


def test_persona_dir_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: _home(tmp_path))
    persona = tmp_path / "persona"; persona.mkdir()
    r = check_write_target(str(persona / "bridge.json"), op="create", persona_dir=persona)
    assert not r.ok and "denied" in r.error.lower()


def test_dotdot_escape_into_ssh_refused(tmp_path, monkeypatch):
    h = _home(tmp_path); monkeypatch.setattr(Path, "home", lambda: h)
    (h / ".ssh").mkdir(parents=True, exist_ok=True)
    sneaky = h / "Documents" / ".." / ".ssh" / "authorized_keys"
    r = check_write_target(str(sneaky), op="append", persona_dir=tmp_path / "persona")
    assert not r.ok  # resolves into ~/.ssh


def test_symlink_escape_into_ssh_refused(tmp_path, monkeypatch):
    h = _home(tmp_path); monkeypatch.setattr(Path, "home", lambda: h)
    (h / ".ssh").mkdir(parents=True, exist_ok=True)
    (h / ".ssh" / "authorized_keys").write_text("k")
    link = h / "Documents"; link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(h / ".ssh")  # Documents → ~/.ssh
    r = check_write_target(str(link / "authorized_keys"), op="append", persona_dir=tmp_path / "persona")
    assert not r.ok  # realpath lands in ~/.ssh


def test_create_refused_if_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: _home(tmp_path))
    p = _home(tmp_path) / "f.txt"; p.parent.mkdir(parents=True); p.write_text("x")
    r = check_write_target(str(p), op="create", persona_dir=tmp_path / "persona")
    assert not r.ok and "exists" in r.error.lower()


def test_append_refused_if_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: _home(tmp_path))
    r = check_write_target(str(_home(tmp_path) / "missing.txt"), op="append",
                           persona_dir=tmp_path / "persona")
    assert not r.ok and "exist" in r.error.lower()
