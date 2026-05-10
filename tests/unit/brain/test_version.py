"""Version consistency tests.

Audit 2026-05-10 found ``brain.__version__`` was hard-coded to "0.0.1"
even though pyproject was at 0.0.4 and the Tauri bundle reported 0.0.4 —
so `nell --version` from the v0.0.4-alpha DMG reported "0.0.1",
confusing users + breaking release assets that include the CLI version
in their filenames.

Fix: ``brain/__init__.py`` derives ``__version__`` from package metadata
via importlib.metadata.version("companion-emergence"). This test pins
the contract — a future refactor that re-introduces a hard-coded
constant will break here, and a pyproject bump that forgets to re-lock
will also surface here.
"""
from __future__ import annotations

import re
from importlib.metadata import version as pkg_version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _read_pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert match, "pyproject.toml has no version line"
    return match.group(1)


def test_brain_version_matches_package_metadata():
    """brain.__version__ must equal importlib.metadata's view of the
    installed package version. Anything else means the import is reading
    a stale hard-coded constant."""
    import brain

    assert brain.__version__ == pkg_version("companion-emergence"), (
        f"brain.__version__={brain.__version__!r} but "
        f"importlib.metadata says {pkg_version('companion-emergence')!r}"
    )


def test_brain_version_matches_pyproject():
    """brain.__version__ must equal what pyproject.toml declares.

    The wheel install during `uv sync` writes pyproject's version into
    importlib.metadata, so an import-time mismatch here means the env
    is stale (uv sync wasn't re-run after a version bump)."""
    import brain

    assert brain.__version__ == _read_pyproject_version(), (
        f"brain.__version__={brain.__version__!r} but "
        f"pyproject says {_read_pyproject_version()!r} — "
        "did you forget `uv sync` after the version bump?"
    )


def test_tauri_conf_version_matches_pyproject():
    """The Tauri bundle's about/version dialog reads tauri.conf.json's
    "version" field. It must match pyproject so the .app/.dmg/.exe
    metadata aligns with the wheel."""
    import json

    pyproject_version = _read_pyproject_version()
    conf = json.loads(
        (REPO_ROOT / "app" / "src-tauri" / "tauri.conf.json").read_text(
            encoding="utf-8"
        )
    )
    assert conf["version"] == pyproject_version, (
        f"tauri.conf.json version={conf['version']!r} but "
        f"pyproject says {pyproject_version!r} — "
        "release tag-time bumps must touch all three files in lockstep "
        "(pyproject.toml, app/src-tauri/Cargo.toml, app/src-tauri/tauri.conf.json)."
    )


def test_cargo_toml_version_matches_pyproject():
    """The nellface crate version. Same lockstep contract."""
    pyproject_version = _read_pyproject_version()
    cargo_text = (REPO_ROOT / "app" / "src-tauri" / "Cargo.toml").read_text(
        encoding="utf-8"
    )
    # First `version = "..."` line is the [package] block.
    match = re.search(r'^version\s*=\s*"([^"]+)"', cargo_text, flags=re.MULTILINE)
    assert match, "Cargo.toml has no version line"
    assert match.group(1) == pyproject_version, (
        f"Cargo.toml version={match.group(1)!r} but "
        f"pyproject says {pyproject_version!r}."
    )
