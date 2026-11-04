"""#597: read_file and list_directory refuse credential and secret locations.

Driven through the two tools' real entry points with a temp HOME and KINDLED_HOME
(no mocked state files). The deny list always wins; ordinary files still read.
"""
import json
import sys
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


_KEY_PEM = b"-----BEGIN PRIVATE KEY-----\nMIIEvQ\n-----END PRIVATE KEY-----\n"


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
    # review of #608: other browsers, cloud CLIs, Windows stores
    "Library/Application Support/BraveSoftware/Brave-Browser/Default/Cookies",
    "Library/Application Support/Microsoft Edge/Default/Cookies",
    "Library/Application Support/com.operasoftware.Opera/Cookies",
    "Library/Application Support/Vivaldi/Default/Cookies",
    "Library/Application Support/Chromium/Default/Cookies",
    "Library/Safari/History.db",
    "Library/Containers/com.apple.Safari/Data/Library/Cookies/Cookies.binarycookies",
    ".config/BraveSoftware/Brave-Browser/Default/Cookies",
    ".config/microsoft-edge/Default/Cookies",
    ".config/opera/Cookies",
    ".config/vivaldi/Default/Cookies",
    ".config/gcloud/credentials.db",
    ".azure/msal_token_cache.json",
    "AppData/Local/BraveSoftware/Brave-Browser/User Data/Default/Cookies",
    "AppData/Local/Microsoft/Edge/User Data/Default/Cookies",
    "AppData/Roaming/Opera Software/Opera Stable/Cookies",
    "AppData/Local/Vivaldi/User Data/Default/Cookies",
    "AppData/Local/Chromium/User Data/Default/Cookies",
    "AppData/Local/Microsoft/Credentials/x",
    "AppData/Roaming/GitHub CLI/hosts.yml",
    "AppData/Roaming/gcloud/credentials.db",
    "AppData/Roaming/Microsoft/Windows/PowerShell/PSReadLine/ConsoleHost_history.txt",
])
def test_home_secret_dirs_refused(env, rel):
    f = _plant(env["home"] / rel)
    assert _refused(read_file(path=str(f), persona_dir=env["persona"]))


@pytest.mark.parametrize("rel", [
    ".claude.json", ".netrc", ".git-credentials", ".pgpass", ".npmrc", ".pypirc",
    ".cargo/credentials.toml", ".cargo/credentials", ".gem/credentials", ".terraform.d/credentials.tfrc.json",
    # shell and REPL history: pasted tokens and `export KEY=...` lines
    ".bash_history", ".zsh_history", ".python_history", ".psql_history", ".mysql_history", ".node_repl_history",
    ".local/share/fish/fish_history",
])
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


@pytest.mark.parametrize("rel", ["memories.db", "voice.md", "files/" + "b" * 64])
def test_another_personas_directory_refused(env, rel):
    """Review of #608: another kindled on the same machine is private to it, not just its bridge token and link key."""
    f = _plant(env["kh"] / "personas" / "other" / rel)
    assert _refused(read_file(path=str(f), persona_dir=env["persona"]))
    assert _refused(list_directory(path=str(f.parent), persona_dir=env["persona"]))


@pytest.mark.parametrize("name", [
    ".env", ".env.local", ".ENV", "server.pem", "tls.key", "TLS.KEY", "cert.p12", "cert.pfx",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "login.keychain-db",
])
def test_secret_file_names_refused_anywhere(env, name):
    f = _plant(env["home"] / "proj" / name, _KEY_PEM)
    assert _refused(read_file(path=str(f), persona_dir=env["persona"]))


def test_a_public_certificate_pem_still_reads(env):
    """Review of #608: `.pem` is refused only when it holds private-key material; a certificate is public."""
    cert = b"-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n"
    f = _plant(env["home"] / "proj" / "cert.pem", cert)
    assert read_file(path=str(f), persona_dir=env["persona"]).get("content") == cert.decode()
    bundle = _plant(env["home"] / "proj" / "bundle.pem", cert + b"-----BEGIN EC PRIVATE KEY-----\nMHc\n")
    assert _refused(read_file(path=str(bundle), persona_dir=env["persona"]))


@pytest.mark.no_base_failure(reason="#608 review over-blocking guard: main has no read guard, so this reads there too")
def test_a_dotenv_directory_still_lists(env):
    """Review of #608: `.env/` is a common virtualenv name; only the dotenv FILE is a secret."""
    _plant(env["home"] / "proj" / ".env" / "pyvenv.cfg", b"home = /usr")
    out = list_directory(path=str(env["home"] / "proj" / ".env"), persona_dir=env["persona"])
    assert [e["name"] for e in out["entries"]] == ["pyvenv.cfg"]


@pytest.mark.no_base_failure(reason="#608 review over-blocking guard: main has no read guard, so this reads there too")
@pytest.mark.parametrize("name", [".env.example", ".env.sample", ".ENV.Template", ".env.dist"])
def test_dotenv_templates_still_read(env, name):
    """Review of #608: a dotenv template holds placeholders, not secrets."""
    f = _plant(env["home"] / "proj" / name, b"hello")
    assert read_file(path=str(f), persona_dir=env["persona"]).get("content") == "hello"


@pytest.mark.skipif(sys.platform == "win32", reason="no procfs: /proc/... resolves to an ordinary file on the current drive")
@pytest.mark.parametrize("path", ["/proc/self/environ", "/proc/1/environ", "/proc/1/task/1/environ"])
def test_process_environment_refused(env, path):
    """Review of #608: the environment can hold tokens and keys. On Linux these exist; on macOS the refusal still comes first."""
    assert _refused(read_file(path=path, persona_dir=env["persona"]))


@pytest.mark.no_base_failure(reason="#597 over-blocking guard: ordinary files read on the old code too, and must keep reading")
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


@pytest.mark.no_base_failure(reason="#597: needs a case-insensitive volume; it skips on case-sensitive CI, and on macOS fails on the old code")
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


@pytest.mark.no_base_failure(reason="#597 over-blocking guard: listing home works on the old code too, and must keep working")
def test_listing_home_still_works(env):
    _plant(env["home"] / ".ssh" / "id_rsa")
    _plant(env["home"] / "Desktop" / "a.txt")
    out = list_directory(path="~", persona_dir=env["persona"])
    assert "Desktop" in {e["name"] for e in out["entries"]}


def test_a_listing_leaves_out_refused_entries(env):
    """Review of #608: listing a parent must not show the names inside it that a read would refuse."""
    _plant(env["home"] / ".ssh" / "id_rsa")
    _plant(env["home"] / ".bash_history")
    _plant(env["home"] / "proj" / ".env")
    _plant(env["home"] / "proj" / "tls.key")
    _plant(env["home"] / "proj" / "main.py")
    _plant(env["kh"] / "personas" / "other" / "memories.db")
    names = lambda d: {e["name"] for e in list_directory(path=str(d), persona_dir=env["persona"])["entries"]}  # noqa: E731
    assert ".ssh" not in names(env["home"]) and ".bash_history" not in names(env["home"])
    assert names(env["home"] / "proj") == {"main.py"}
    assert names(env["kh"] / "personas") == {"canary"}
