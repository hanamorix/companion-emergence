"""brain.files.write_guard — the file-write security spine (LOAD-BEARING).

Resolves a proposed write target to its real path and refuses (a) any path
inside a hard deny-list (sensitive/persistence/system/persona-substrate), (b)
a create whose path already exists, (c) an append whose path is missing. Run
at BOTH propose and commit (TOCTOU). Pure + exhaustively tested."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from brain.files.path_identity import stat_or_none, under_any

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


def _real(p: Path) -> Path | None:
    try:
        return Path(os.path.realpath(p))
    except (OSError, ValueError):
        return None


def _is_within(child: Path, parent: Path) -> bool:
    """Deny-list test: True if child is parent or under it (#607).

    By file identity when `parent` exists, so a case or Unicode-normalisation
    variant on a case-insensitive volume still matches. Also by casefolded
    spelling, because commit creates missing parents: ``~/.AWS/credentials`` with
    no ``~/.aws`` yet has nothing to compare by identity but would create the
    directory the AWS CLI then reads. On a case-sensitive volume that refuses a
    differently-cased sibling too, which is the safe side for a deny list."""
    c, p = _real(child), _real(parent)
    if c is None or p is None:
        return False
    st = stat_or_none(p)
    if st is not None and under_any(c, [st]):
        return True
    cs, ps = os.path.normcase(str(c)).casefold(), os.path.normcase(str(p)).casefold()
    return cs == ps or cs.startswith(ps.rstrip(os.sep) + os.sep)


def is_within_authorized(resolved: Path, folder: Path | None) -> bool:
    """True only if `resolved`'s real path is inside `folder`, by file identity
    (#607), so the folder may be given in any case the volume accepts. Escape-proof
    (realpath collapses .. and symlinks before the ancestors are compared). No
    spelling fallback: this is an allow-list, and a folder that does not exist
    authorizes nothing. None folder → always False (notes disabled = nothing
    authorized)."""
    if folder is None:
        return False
    c, st = _real(resolved), stat_or_none(folder)
    return c is not None and st is not None and under_any(c, [st])


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
