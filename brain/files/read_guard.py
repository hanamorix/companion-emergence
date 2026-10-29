"""brain.files.read_guard — credential/secret deny list for the read tools (#597).

read_file and list_directory may open any file the user asks about, except these.
Checked right after the path is resolved, before anything is opened, listed or
suggested. Directories are compared by file identity (os.path.samestat) on every
existing ancestor, so a case variant on a case-insensitive volume or a symlink
cannot reach around a rule the way a string-prefix check can.
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
)
# Home-relative single files holding tokens or passwords.
_HOME_DENY_FILES = (".claude.json", ".netrc", ".git-credentials", ".pgpass", ".npmrc", ".pypirc")
# Secret file names, refused wherever they are (matched casefolded).
_DENY_NAMES = frozenset({".env", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"})
_DENY_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".keychain", ".keychain-db"})


def _secret_name(p: Path) -> bool:
    name = p.name.casefold()
    return name in _DENY_NAMES or name.startswith(".env.") or Path(name).suffix in _DENY_SUFFIXES


def _denied_paths(persona_dir: Path) -> list[Path]:
    home = Path.home()
    paths = [home / rel for rel in (*_HOME_DENY_DIRS, *_HOME_DENY_FILES)]
    # the brain's own Claude CLI config dir, and any upstream one the CLI is pointed at
    paths.append(get_home() / "claude-config")
    if os.environ.get("CLAUDE_CONFIG_DIR"):
        paths.append(Path(os.environ["CLAUDE_CONFIG_DIR"]).expanduser())
    # every persona's bridge bearer token and Kindled-link private key (kindled_link/identity.py's _KEY_DIRNAME)
    personas_root = get_home() / "personas"
    personas = [persona_dir, *(personas_root.iterdir() if personas_root.is_dir() else ())]
    for pd in personas:
        paths += [pd / STATE_FILENAME, pd / "kindled_link"]
    return paths


def _stat(p: Path) -> os.stat_result | None:
    try:
        return os.stat(p)
    except OSError:
        return None


def is_read_denied(resolved: Path, *, persona_dir: Path) -> bool:
    """True if `resolved` (already .resolve()d) is, or is inside, a protected location."""
    if _secret_name(resolved):
        return True
    denied = [st for st in map(_stat, _denied_paths(persona_dir)) if st is not None]
    for anc in (resolved, *resolved.parents):
        st = _stat(anc)
        if st is not None and any(os.path.samestat(st, d) for d in denied):
            return True
    return False
