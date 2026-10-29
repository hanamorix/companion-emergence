"""#597: read_file and list_directory refuse credential and secret locations.

Driven through the two tools' real entry points with a temp HOME and KINDLED_HOME
(no mocked state files). The deny list always wins; ordinary files still read.
"""
import json
from pathlib import Path

import pytest

from brain.tools.impls.list_directory import list_directory
from brain.tools.impls.read_file import read_file


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    kh = tmp_path / "kindled"
    persona = kh / "personas" / "canary"
    for d in (home, persona):
        d.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("KINDLED_HOME", str(kh))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return {"home": home, "kh": kh, "persona": persona, "tmp": tmp_path}


def _plant(path: Path, data: bytes = b"secret") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _refused(out: dict) -> bool:
    return "content" not in out and "image" not in out and "protected" in out.get("error", "")


def test_ssh_private_key_refused(env):
    f = _plant(env["home"] / ".ssh" / "id_rsa")
    assert _refused(read_file(path=str(f), persona_dir=env["persona"]))


@pytest.mark.parametrize("rel", [
    ".claude/.credentials.json",
    ".claude/projects/x/session.jsonl",
    ".aws/credentials",
    ".gnupg/private-keys-v1.d/k",
    ".config/gh/hosts.yml",
    ".docker/config.json",
    ".kube/config",
    ".password-store/site.gpg",
    ".local/share/keyrings/login.keyring",
    "Library/Keychains/login.keychain-db",
    "Library/Cookies/Cookies.binarycookies",
    "Library/Application Support/Google/Chrome/Default/Cookies",
    "Library/Application Support/Firefox/Profiles/p/logins.json",
    ".mozilla/firefox/p/logins.json",
    ".config/google-chrome/Default/Cookies",
    ".config/chromium/Default/Cookies",
    "AppData/Local/Google/Chrome/User Data/Default/Cookies",
    "AppData/Roaming/Mozilla/Firefox/Profiles/p/logins.json",
    "AppData/Roaming/Microsoft/Credentials/x",
    "AppData/Roaming/Microsoft/Protect/x",
])
def test_home_secret_dirs_refused(env, rel):
    f = _plant(env["home"] / rel)
    assert _refused(read_file(path=str(f), persona_dir=env["persona"]))


@pytest.mark.parametrize("rel", [".claude.json", ".netrc", ".git-credentials", ".pgpass", ".npmrc", ".pypirc"])
def test_home_secret_files_refused(env, rel):
    f = _plant(env["home"] / rel)
    assert _refused(read_file(path=str(f), persona_dir=env["persona"]))


@pytest.mark.parametrize("where", ["brain", "upstream"])
def test_claude_config_dir_refused_wherever_it_lives(env, monkeypatch, where):
    if where == "brain":
        cfg = env["kh"] / "claude-config"  # the brain's own CLAUDE_CONFIG_DIR
    else:
        cfg = env["tmp"] / "elsewhere" / "cfg"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg))
    f = _plant(cfg / ".credentials.json")
    assert _refused(read_file(path=str(f), persona_dir=env["persona"]))


@pytest.mark.parametrize("persona", ["canary", "other"])
@pytest.mark.parametrize("rel", ["bridge.json", "kindled_link/identity_ed25519.key"])
def test_persona_secrets_refused(env, persona, rel):
    f = _plant(env["kh"] / "personas" / persona / rel)
    assert _refused(read_file(path=str(f), persona_dir=env["persona"]))


@pytest.mark.parametrize("name", [
    ".env", ".env.local", ".ENV", "server.pem", "tls.key", "TLS.KEY", "cert.p12", "cert.pfx",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "login.keychain-db",
])
def test_secret_file_names_refused_anywhere(env, name):
    f = _plant(env["home"] / "proj" / name)
    assert _refused(read_file(path=str(f), persona_dir=env["persona"]))


@pytest.mark.parametrize("rel", [
    "home/Desktop/notes.txt",
    "home/proj/environment.md",
    "home/proj/keys.txt",
    "home/proj/.envrc.example.md",
    "home/.ssh_notes.txt",
    "home/proj/id_ed25519.pub",
    "kindled/personas/canary/files/" + "a" * 64,  # a file the user shared in chat
])
def test_ordinary_files_still_read(env, rel):
    f = _plant(env["tmp"] / rel, b"hello")
    assert read_file(path=str(f), persona_dir=env["persona"]).get("content") == "hello"


@pytest.mark.parametrize("rel", ["home/.ssh", "home/.claude", "home/.aws/sub", "kindled/personas/canary/kindled_link"])
def test_list_directory_secret_dir_refused(env, rel):
    d = env["tmp"] / rel
    _plant(d / "x")
    out = list_directory(path=str(d), persona_dir=env["persona"])
    assert "entries" not in out and "protected" in out.get("error", "")


def test_case_variant_refused(env):
    _plant(env["home"] / ".claude" / ".credentials.json")
    variant = env["home"] / ".CLAUDE" / ".credentials.json"
    if not variant.exists():
        pytest.skip("case-sensitive filesystem: the variant names a different, absent path")
    assert _refused(read_file(path=str(variant), persona_dir=env["persona"]))


def test_symlink_into_secret_dir_refused(env):
    target = _plant(env["home"] / ".aws" / "credentials")
    link = env["home"] / "Desktop" / "innocent.txt"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable for this user/platform")
    assert _refused(read_file(path=str(link), persona_dir=env["persona"]))


def test_image_in_secret_dir_refused_before_the_image_branch(env):
    f = _plant(env["home"] / ".ssh" / "shot.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    assert _refused(read_file(path=str(f), persona_dir=env["persona"]))


def test_missing_file_in_secret_dir_does_not_suggest_its_siblings(env):
    _plant(env["home"] / ".ssh" / "id_work")
    out = read_file(path=str(env["home"] / ".ssh" / "id_wrk"), persona_dir=env["persona"])
    assert _refused(out) and "did_you_mean" not in out


def test_refusal_is_audited(env):
    f = _plant(env["home"] / ".ssh" / "id_rsa")
    read_file(path=str(f), persona_dir=env["persona"])
    log = env["persona"] / "file_access.jsonl"
    line = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert line["tool"] == "read_file" and line["ok"] is False and "protected" in line["error"]


def test_listing_home_still_works(env):
    _plant(env["home"] / ".ssh" / "id_rsa")
    _plant(env["home"] / "Desktop" / "a.txt")
    out = list_directory(path="~", persona_dir=env["persona"])
    assert {e["name"] for e in out["entries"]} >= {"Desktop", ".ssh"}
