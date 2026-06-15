"""brain.files.write_guard — the file-write security spine (LOAD-BEARING).

Resolves a proposed write target to its real path and refuses (a) any path
inside a hard deny-list (sensitive/persistence/system/persona-substrate), (b)
a create whose path already exists, (c) an append whose path is missing. Run
at BOTH propose and commit (TOCTOU). Pure + exhaustively tested."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_MAX_WRITE_BYTES = 1 * 1024 * 1024          # 1 MB per proposal
_MAX_RESULT_FILE_BYTES = 25 * 1024 * 1024   # 25 MB resulting-file ceiling (append)

# Home-relative sensitive prefixes (login/persistence/credentials).
_HOME_DENY = (
    ".zshrc", ".bashrc", ".bash_profile", ".profile", ".zprofile", ".zshenv", ".zlogin",
    ".ssh", ".aws", ".gnupg", ".netrc", ".config/gh",
    "Library/LaunchAgents",
)
# Absolute system roots.
_ABS_DENY = ("/etc", "/usr", "/bin", "/sbin", "/System", "/Library", "/private/etc",
             "/Applications", "/Library/LaunchDaemons")


@dataclass
class GuardResult:
    ok: bool
    resolved: Path | None = None
    error: str | None = None


def _is_within(child: Path, parent: Path) -> bool:
    """True if child == parent or is under it, comparing casefolded real paths
    (macOS is case-insensitive — block ~/.SSH too)."""
    try:
        c = os.path.normcase(os.path.realpath(child))
        p = os.path.normcase(os.path.realpath(parent))
    except OSError:
        return False
    return c == p or c.startswith(p + os.sep)


def is_within_authorized(resolved: Path, folder: Path | None) -> bool:
    """True only if `resolved`'s real path is inside `folder`. Escape-proof
    (realpath collapses .. and symlinks). None folder → always False (notes
    disabled = nothing authorized)."""
    if folder is None:
        return False
    return _is_within(resolved, folder)


def _denied(resolved: Path, persona_dir: Path) -> bool:
    home = Path.home()
    for rel in _HOME_DENY:
        if _is_within(resolved, home / rel):
            return True
    for ab in _ABS_DENY:
        if _is_within(resolved, Path(ab)):
            return True
    # her own substrate — never writable through this tool
    if _is_within(resolved, persona_dir):
        return True
    return False


def check_write_target(raw_path: str, *, op: str, persona_dir: Path) -> GuardResult:
    try:
        resolved = Path(os.path.expandvars(os.path.expanduser(raw_path))).resolve()
    except (OSError, ValueError) as exc:
        return GuardResult(ok=False, error=f"bad path: {exc}")
    if _denied(resolved, persona_dir):
        return GuardResult(ok=False, error=f"denied: {resolved} is in a protected location")
    if op == "create":
        if resolved.exists():
            return GuardResult(ok=False, error=f"path already exists (create won't overwrite): {resolved}")
    elif op == "append":
        if not resolved.exists() or not resolved.is_file():
            return GuardResult(ok=False, error=f"append target must exist as a file: {resolved}")
    else:
        return GuardResult(ok=False, error=f"unknown op: {op!r}")
    return GuardResult(ok=True, resolved=resolved)


def check_size(content: str, *, op: str, resolved: Path) -> GuardResult:
    """Byte caps: per-write + (append) resulting-file ceiling."""
    nbytes = len(content.encode("utf-8"))
    if nbytes > _MAX_WRITE_BYTES:
        return GuardResult(ok=False, error=f"too large ({nbytes} > {_MAX_WRITE_BYTES} cap)")
    if op == "append" and resolved.exists():
        if resolved.stat().st_size + nbytes > _MAX_RESULT_FILE_BYTES:
            return GuardResult(ok=False, error="append would exceed the file-size ceiling")
    return GuardResult(ok=True, resolved=resolved)
