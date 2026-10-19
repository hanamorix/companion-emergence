"""#488: no test may write into the real per-user data directory.

At least one test (any that builds the bridge app: its lifespan rewrites `tunables.json`) resolved the data home to the platformdirs
default, the real `~/.local/share/companion-emergence` of whoever ran the suite. The autouse fixture `_isolate_user_data_home`
(tests/conftest.py) now gives every test its own empty default.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from brain import paths
from brain.bridge.server import build_app
from tests.conftest import REAL_PLATFORM_DIRS


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def test_the_default_home_is_not_the_real_one(monkeypatch):
    monkeypatch.delenv("KINDLED_HOME", raising=False)
    monkeypatch.delenv("NELLBRAIN_HOME", raising=False)
    for resolved in (paths.get_home(), paths.get_cache_dir(), paths.get_log_dir()):
        for real in REAL_PLATFORM_DIRS.values():
            assert not _is_under(resolved, real)
    assert "companion-emergence" in str(paths.get_home()).lower()


def test_xdg_variables_point_away_from_the_real_home():
    for var in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
        value = os.environ.get(var)
        assert value, var
        assert not _is_under(Path(value).resolve(), REAL_PLATFORM_DIRS['user_data_path'])


def test_a_fresh_platformdirs_agrees_with_the_brain_paths_default(monkeypatch):
    from platformdirs import PlatformDirs

    monkeypatch.delenv("KINDLED_HOME", raising=False)
    monkeypatch.delenv("NELLBRAIN_HOME", raising=False)
    assert PlatformDirs(appname="companion-emergence", appauthor="hanamorix").user_data_path.resolve() == paths.get_home()


def test_a_tests_default_home_starts_empty():
    home = paths.get_home()
    assert not home.exists() or not any(home.iterdir())


def test_building_the_app_leaves_the_real_data_home_untouched(tmp_path, monkeypatch):
    monkeypatch.delenv("KINDLED_HOME", raising=False)
    monkeypatch.delenv("NELLBRAIN_HOME", raising=False)
    real = REAL_PLATFORM_DIRS['user_data_path']
    before = _snapshot(real)
    persona = tmp_path / "p"
    persona.mkdir()
    app = build_app(persona_dir=persona, client_origin="tests")
    with TestClient(app):
        pass
    assert _snapshot(real) == before
    # the lifespan did write its tunables file: into the test's own home
    assert (paths.get_home() / "tunables.json").exists()
