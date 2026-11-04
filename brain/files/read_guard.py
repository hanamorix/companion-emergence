"""brain.files.read_guard — credential/secret deny list for the read tools (#597).

read_file and list_directory may open any file the user asks about, except these.
Checked right after the path is resolved, before anything is opened, listed or
suggested. Directories are compared by file identity (os.path.samestat) on every
existing ancestor, so a case variant on a case-insensitive volume or a symlink
cannot reach around a rule the way a string-prefix check can.

The lists are best-effort: they name the common credential stores per platform,
not every one. A hard link to a listed single file is refused (same identity);
shortcut: a hard link to a file INSIDE a denied directory is not, since that would
mean indexing every file under those trees — the read tools can't make links, so
upgrade only if a write path ever can.
"""
from __future__ import annotations

import os
from pathlib import Path

from brain.bridge.state_file import STATE_FILENAME
from brain.paths import get_home

PROTECTED = "protected location (credentials or secrets)"

# Home-relative directories that hold credentials, keys or session secrets.
_HOME_DENY_DIRS = (
    ".claude",  # the Claude CLI's default config: login credentials and the user's own transcripts
    ".ssh", ".aws", ".gnupg", ".config/gh", ".docker", ".kube", ".password-store",
    ".local/share/keyrings",
    "Library/Keychains", "Library/Cookies",
    # browser profiles (saved logins, cookies)
    "Library/Application Support/Google/Chrome", "Library/Application Support/Firefox",
    ".mozilla", ".config/google-chrome", ".config/chromium",
    "AppData/Local/Google/Chrome", "AppData/Roaming/Mozilla",
    "AppData/Roaming/Microsoft/Credentials", "AppData/Roaming/Microsoft/Protect",
    "Library/Application Support/BraveSoftware", "Library/Application Support/Microsoft Edge",
    "Library/Application Support/com.operasoftware.Opera", "Library/Application Support/Vivaldi",
    "Library/Application Support/Chromium", "Library/Safari", "Library/Containers/com.apple.Safari",
    ".config/BraveSoftware", ".config/microsoft-edge", ".config/opera", ".config/vivaldi",
    "AppData/Local/BraveSoftware", "AppData/Local/Microsoft/Edge", "AppData/Roaming/Opera Software",
    "AppData/Local/Vivaldi", "AppData/Local/Chromium",
    "AppData/Local/Microsoft/Credentials",
    "AppData/Roaming/Microsoft/Windows/PowerShell/PSReadLine",  # PowerShell history
    # cloud CLIs
    ".config/gcloud", ".azure", "AppData/Roaming/gcloud", "AppData/Roaming/GitHub CLI",
)
# Home-relative single files holding tokens or passwords.
_HOME_DENY_FILES = (
    ".claude.json", ".netrc", ".git-credentials", ".pgpass", ".npmrc", ".pypirc",
    ".cargo/credentials.toml", ".cargo/credentials", ".gem/credentials", ".terraform.d/credentials.tfrc.json",
    # shell and REPL history: pasted tokens and `export KEY=...` lines
    ".bash_history", ".zsh_history", ".python_history", ".psql_history", ".mysql_history", ".node_repl_history",
    ".local/share/fish/fish_history",
)
# Secret file names, refused wherever they are (matched casefolded).
_DENY_NAMES = frozenset({".env", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"})
_DENY_SUFFIXES = frozenset({".key", ".p12", ".pfx", ".keychain", ".keychain-db"})


_DOTENV_TEMPLATES = frozenset({".example", ".sample", ".template", ".dist"})


def _secret_name(p: Path) -> bool:
    if p.is_dir():  # the names are secret FILES; `.env/` is a common virtualenv
        return False
    name = p.name.casefold()
    if name.startswith(".env.") and Path(name).suffix in _DOTENV_TEMPLATES:  # placeholders, not secrets
        return False
    if Path(name).suffix == ".pem":
        return _holds_private_key(p)
    return name in _DENY_NAMES or name.startswith(".env.") or Path(name).suffix in _DENY_SUFFIXES


def _holds_private_key(p: Path) -> bool:
    """A .pem is often a public certificate; refuse it only if it holds a private key, or can't be checked."""
    try:
        with p.open("rb") as fh:
            return b"PRIVATE KEY" in fh.read(1 << 20)
    except OSError:
        return True


def _denied_paths(persona_dir: Path) -> list[Path]:
    home = Path.home()
    paths = [home / rel for rel in (*_HOME_DENY_DIRS, *_HOME_DENY_FILES)]
    # the brain's own Claude CLI config dir, and any upstream one the CLI is pointed at
    paths.append(get_home() / "claude-config")
    if os.environ.get("CLAUDE_CONFIG_DIR"):
        paths.append(Path(os.environ["CLAUDE_CONFIG_DIR"]).expanduser())
    # this persona's bridge bearer token and Kindled-link private key (kindled_link/identity.py's _KEY_DIRNAME)
    paths += [persona_dir / STATE_FILENAME, persona_dir / "kindled_link"]
    # every other persona on the machine, whole
    own = _stat(persona_dir)
    personas_root = get_home() / "personas"
    for pd in personas_root.iterdir() if personas_root.is_dir() else ():
        st = _stat(pd)
        if st is not None and not (own is not None and os.path.samestat(st, own)):
            paths.append(pd)
    return paths


def _stat(p: Path) -> os.stat_result | None:
    try:
        return os.stat(p)
    except OSError:
        return None


def denied_stats(persona_dir: Path) -> list[os.stat_result]:
    """The protected locations that exist, for checking many paths against one snapshot (a directory listing)."""
    return [st for st in map(_stat, _denied_paths(persona_dir)) if st is not None]


def is_read_denied(resolved: Path, *, persona_dir: Path, denied: list[os.stat_result] | None = None) -> bool:
    """True if `resolved` (already .resolve()d) is, or is inside, a protected location."""
    if _secret_name(resolved):
        return True
    if resolved.parts[:2] == ("/", "proc") and resolved.name == "environ":  # a process's environment: tokens, keys
        return True
    if denied is None:
        denied = denied_stats(persona_dir)
    for anc in (resolved, *resolved.parents):
        st = _stat(anc)
        if st is not None and any(os.path.samestat(st, d) for d in denied):
            return True
    return False
