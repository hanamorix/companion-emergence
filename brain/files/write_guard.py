"""brain.files.write_guard — the file-write security spine (LOAD-BEARING).

Resolves a proposed write target to its real path and refuses (a) any path
inside a hard deny-list (sensitive/persistence/system/persona-substrate), (b)
a create whose path already exists, (c) an append whose path is missing. Run
at BOTH propose and commit (TOCTOU). Pure + exhaustively tested."""
from __future__ import annotations

import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from brain.files.path_identity import stat_or_none, under_any, unusable

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


def _is_windows() -> bool:
    """Seam for tests: patching ``sys.platform`` globally reaches the stdlib (#662)."""
    return sys.platform == "win32"


def _fold(name: str) -> str:
    """Canonical caseless form: equal for any case or Unicode-normalisation variant,
    and on Windows for the trailing dots and spaces Win32 drops from a segment
    (``.aws.`` creates ``.aws``)."""
    if _is_windows():
        name = name.rstrip(". ")
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", name).casefold())


def _is_within(child: Path, parent: Path) -> bool:
    """Deny-list test: True if child is parent or under it (#607).

    `parent` may not exist yet (commit creates missing parents, so writing
    ``~/.AWS/credentials`` with no ``~/.aws`` would create the directory the AWS
    CLI then reads). So split `parent` at its nearest existing ancestor: that
    anchor is matched against child's ancestors by file identity, which holds for
    any case or Unicode-normalisation variant the volume accepts, and the missing
    tail is matched by folded spelling. When `parent` exists the tail is empty and
    this is a plain identity check. On a case-sensitive volume the folding also
    refuses a differently-cased sibling of a missing directory, the safe side for
    a deny list."""
    c, p = _real(child), _real(parent)
    if c is None or p is None:
        return False
    anchor, tail = p, ()
    while stat_or_none(anchor) is None and anchor != anchor.parent:
        anchor, tail = anchor.parent, (anchor.name, *tail)
    st = stat_or_none(anchor)
    if st is None:
        return False
    for anc in (c, *c.parents):
        s = stat_or_none(anc)
        if s is not None and os.path.samestat(s, st):
            rest = c.parts[len(anc.parts):len(anc.parts) + len(tail)]
            return len(rest) == len(tail) and all(_fold(a) == _fold(b) for a, b in zip(rest, tail, strict=True))
    return False


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
    except (OSError, ValueError, RuntimeError) as exc:  # RuntimeError: a symlink loop on Python 3.12 (#695)
        return GuardResult(ok=False, error=f"bad path: {exc}")
    try:  # a lone surrogate (from a JSON argument) isn't text; the OS may take it, the audit and card ids can't (#695)
        str(resolved).encode("utf-8")
    except UnicodeEncodeError:
        return GuardResult(ok=False, error="bad path: not valid text (lone surrogate)")
    if _is_windows() and any(":" in part for part in resolved.parts[1:]):
        # an NTFS stream (``.aws::$INDEX_ALLOCATION`` creates the directory) slips past both comparisons
        return GuardResult(ok=False, error=f"bad path: {resolved} names an NTFS stream")
    # before the deny-list scan, which stats every ancestor per protected location: a 2600-part path
    # took 3 s of CPU there. Either way it's refused.
    if (why := unusable(resolved)) is not None:
        return GuardResult(ok=False, error=f"bad path: {why}")
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
    if op == "append":
        try:
            size = resolved.stat().st_size if resolved.exists() else 0
        except OSError as exc:  # #695: the path stopped being usable since check_write_target
            return GuardResult(ok=False, error=f"bad path: {exc.strerror or type(exc).__name__}")
        if size + nbytes > _MAX_RESULT_FILE_BYTES:
            return GuardResult(ok=False, error="append would exceed the file-size ceiling")
    return GuardResult(ok=True, resolved=resolved)
