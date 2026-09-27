"""Brain-overlay activation hook (#286 §3.4).

Installed into the bundled runtime's site-packages as `_ce_overlay.py`, with a
`_ce_overlay.pth` that runs `activate(SITE)` at every interpreter start and a
`_ce_bundle_id` naming this bundle. If `<home>/brain-overlay/current.json` has an
active overlay built for this bundle, its folder goes on sys.path just before the
bundle's site-packages, so its packages (and `brain`) win.

MUST NOT import `brain` (the bundle's copy would be pinned in sys.modules before
the overlay is on the path) and MUST NOT raise: any problem means "no overlay",
i.e. the release brain runs. Imports: stdlib (hashlib, json, os, re, shutil, sys,
pathlib), plus platformdirs lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

HOOK_MODULE = "_ce_overlay"
BUNDLE_ID_FILE = "_ce_bundle_id"
PTH_LINE = "import _ce_overlay; _ce_overlay.activate(_ce_overlay.SITE)\n"
SITE = Path(__file__).parent
_PLAIN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def home() -> Path:
    """Same answer as brain.paths.get_home(); keep the two in step (parity test)."""
    override = os.environ.get("KINDLED_HOME") or os.environ.get("NELLBRAIN_HOME")
    if override:
        return Path(override).expanduser().resolve()
    from platformdirs import PlatformDirs

    return PlatformDirs(appname="companion-emergence", appauthor="hanamorix").user_data_path.resolve()


def activate(site_dir: Path) -> str | None:
    try:
        root = home() / "brain-overlay"
        state = json.loads((root / "current.json").read_text(encoding="utf-8"))
        active = state.get("active") if isinstance(state, dict) else None
        if not isinstance(active, dict):
            return None
        bundle_id = (Path(site_dir) / BUNDLE_ID_FILE).read_text(encoding="utf-8").strip()
        if not bundle_id or active.get("bundle_id") != bundle_id:
            return None
        name = active.get("dir")
        if not isinstance(name, str) or not _PLAIN_NAME.fullmatch(name) or name in (".", ".."):
            return None
        folder = root / name
        if not folder.is_dir():
            return None
        site_real = os.path.realpath(site_dir)
        index = next((i for i, p in enumerate(sys.path) if os.path.realpath(p) == site_real), None)
        if index is None:
            return None  # can't place it before the bundle; running the release brain is safe
        sys.path.insert(index, str(folder))
        return str(folder)
    except Exception:
        return None


def compute_bundle_id(requirements: Path, wheel_name: str) -> str:
    digest = hashlib.sha256(Path(requirements).read_bytes())
    digest.update(b"\0" + wheel_name.encode("utf-8"))
    return digest.hexdigest()


def install_hook(site_dir: Path, bundle_id: str) -> None:
    site_dir = Path(site_dir)
    shutil.copyfile(Path(__file__), site_dir / f"{HOOK_MODULE}.py")
    (site_dir / f"{HOOK_MODULE}.pth").write_text(PTH_LINE, encoding="utf-8")
    (site_dir / BUNDLE_ID_FILE).write_text(bundle_id + "\n", encoding="utf-8")
