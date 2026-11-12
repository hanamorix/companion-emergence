"""brain.files.path_identity — is a path inside a directory, by file identity.

Shared by the read guard (#597) and the write guard (#607). Comparing spellings
fails on a case-insensitive volume (``realpath`` keeps the case the caller typed,
so ``~/.SSH`` and ``~/.ssh`` differ as strings but name one directory) and on
APFS's normalisation-insensitive names. ``os.path.samestat`` on each existing
ancestor catches both, and a not-yet-existing tail is covered by its nearest
existing parent.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path


def stat_or_none(p: Path) -> os.stat_result | None:
    """None if `p` doesn't exist or can't be statted. A path that can't be a path
    at all (NUL byte) raises ValueError, so a caller can't mistake it for "absent"."""
    try:
        return os.stat(p)
    except OSError:
        return None


def unusable(p: Path) -> str | None:
    """Why the OS can't use `p` as a path (too long, permission denied, a loop, a NUL
    byte, a lone surrogate), or None if it can: it exists, or simply doesn't yet.
    Guards check this right after resolving, so a later ``exists()`` or ``is_dir()``
    can't raise mid-check (#695)."""
    try:
        os.stat(p)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        return exc.strerror or type(exc).__name__
    except ValueError:  # NUL byte, or a surrogate the filesystem encoding can't take (UnicodeEncodeError)
        return "not a valid path"
    return None


def under_any(resolved: Path, targets: Iterable[os.stat_result]) -> bool:
    """True if `resolved`, or any existing ancestor of it, is one of `targets`.

    `resolved` must already be a real path (``Path.resolve()``): ``os.stat``
    follows symlinks, so an unresolved ancestor that is a link would be compared
    as the place it points to while its own parents still match the original tree.
    """
    targets = list(targets)
    if not targets:
        return False
    for anc in (resolved, *resolved.parents):
        st = stat_or_none(anc)
        if st is not None and any(os.path.samestat(st, t) for t in targets):
            return True
    return False
