"""#488: no test may write into the real per-user data directory.

At least one test (any that builds the bridge app: its lifespan rewrites `tunables.json`) resolved the data home to the platformdirs
default, the real `~/.local/share/companion-emergence` of whoever ran the suite. The autouse fixture `_isolate_user_data_home`
(tests/conftest.py) now gives every test its own empty default.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain import paths
from brain.bridge.server import build_app
from tests.conftest import REAL_PLATFORM_DIRS, REAL_XDG_CACHE_HOME


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
    for resolved in (paths.get_home(), paths.get_log_dir()):
        for real in REAL_PLATFORM_DIRS.values():
            assert not _is_under(resolved, real)
    assert "companion-emergence" in str(paths.get_home()).lower()


def test_xdg_variables_point_away_from_the_real_home():
    for var in ("XDG_DATA_HOME", "XDG_STATE_HOME"):
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


def test_the_isolated_dirs_keep_the_appauthor_distinction():
    """Code review 3a, NB5: on Windows and macOS the real class puts the app author in the path, and the harness guard test
    `test_guarded_roots_includes_appauthor_correct_home` exists because `PlatformDirs(app)` (no author) and the engine's
    `PlatformDirs(app, author)` are DIFFERENT directories there. The fixture must not make them one."""
    from platformdirs import PlatformDirs

    with_author = PlatformDirs(appname="companion-emergence", appauthor="hanamorix").user_data_path
    other_author = PlatformDirs(appname="companion-emergence", appauthor="someone-else").user_data_path
    without = PlatformDirs("companion-emergence").user_data_path
    assert with_author != other_author
    assert with_author != without
    assert paths._dirs.user_data_path == with_author  # the engine's own instance


# --- the model cache is NOT isolated (CI: judge release RSS, test_c3a) -------------------------------------------------------


def test_the_model_cache_stays_where_it_is():
    """The first version of the fixture also moved the cache directory and XDG_CACHE_HOME. The cache dir is where the downloaded
    models live (a test that needs a cached model, or a subprocess it starts, looks there), not user state; with it moved the judge
    release RSS job's C3a found no cached judge in its subprocess, fell back to the network and failed on all three OSes. platformdirs
    reads XDG_CACHE_HOME on Linux AND macOS, and huggingface_hub derives HF_HOME from it everywhere."""
    if not (os.environ.get("KINDLED_HOME") or os.environ.get("NELLBRAIN_HOME")):
        assert paths.get_cache_dir() == REAL_PLATFORM_DIRS["user_cache_path"]
    assert os.environ.get("XDG_CACHE_HOME") == REAL_XDG_CACHE_HOME


def test_a_subprocess_resolves_the_same_model_cache_as_the_run_started_with():
    """What C3a does: it starts a fresh interpreter that inherits this test's environment and expects `get_cache_dir()` there to be the
    real cache holding the judge. No model is loaded here."""
    import subprocess
    import sys

    if os.environ.get("KINDLED_HOME") or os.environ.get("NELLBRAIN_HOME"):
        pytest.skip("the run was started with an explicit home override")
    repo_root = Path(__file__).resolve().parents[3]
    out = subprocess.run(
        [sys.executable, "-P", "-c", "from brain.paths import get_cache_dir; print(get_cache_dir())"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip()).resolve() == REAL_PLATFORM_DIRS["user_cache_path"]
