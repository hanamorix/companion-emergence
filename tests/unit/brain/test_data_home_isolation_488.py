"""#488: no test may write into the real per-user data directory.

At least one test (any that builds the bridge app: its lifespan syncs the settings file) resolved the data home to the platformdirs
default, the real `~/.local/share/companion-emergence` of whoever ran the suite. The autouse fixture `_isolate_user_data_home`
(tests/conftest.py) now gives every test its own empty default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain import paths, process_identity
from brain.bridge.server import build_app
from brain.bridge.state_file import ALIVE, STATE_FILENAME, BridgeState, bridge_liveness
from brain.state_compat import from_known_fields
from tests.conftest import REAL_PLATFORM_DIRS, REAL_XDG_CACHE_HOME


def _snapshot(root: Path) -> dict[str, tuple[int, int] | None]:
    """mtime and size of every file under `root`, plus a key per persona directory (so a new empty one shows). SQLite sidecars inside a
    persona are skipped: whoever opens its database creates them, so they say nothing about who wrote it (#600)."""
    if not root.exists():
        return {}
    snap: dict[str, tuple[int, int] | None] = {}
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        in_persona = rel.parts[0] == "personas" and len(rel.parts) >= 2
        if in_persona and len(rel.parts) == 2 and p.is_dir():
            snap[f"personas/{rel.parts[1]}/"] = None
            continue
        if in_persona and p.name.endswith(("-wal", "-shm", "-journal")):
            continue
        try:
            st = p.stat()
        except FileNotFoundError:  # removed by that persona's supervisor mid-walk
            continue
        if p.is_file():
            snap[rel.as_posix()] = (st.st_mtime_ns, st.st_size)
    return snap


def _live_personas(root: Path) -> set[str]:
    """Personas whose own supervisor is running now (the installed app on a dev machine, #600): their files are theirs to write.
    bridge.json is parsed here, not through state_file.read, whose heal of a corrupt file would itself write into the real home."""
    live = set()
    for d in (root / "personas").glob("*"):
        try:
            state = from_known_fields(BridgeState, json.loads((d / STATE_FILENAME).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        if bridge_liveness(state, d) == ALIVE:
            live.add(d.name)
    return live


def _outside(snap: dict[str, tuple[int, int] | None], personas: set[str]) -> dict[str, tuple[int, int] | None]:
    """`snap` without the files of `personas`; their directory keys stay, so removing one still shows."""
    return {k: v for k, v in snap.items() if k.endswith("/") or not (k.startswith("personas/") and k.split("/")[1] in personas)}


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
    live_before, before = _live_personas(real), _snapshot(real)
    persona = tmp_path / "p"
    persona.mkdir()
    app = build_app(persona_dir=persona, client_origin="tests")
    with TestClient(app):
        pass
    # #600: a persona whose supervisor was live at either check writes its own files; everything else must be untouched
    live = live_before | _live_personas(real)
    assert _outside(_snapshot(real), live) == _outside(before, live)
    # the lifespan did write its tunables file: into the test's own home
    assert (paths.get_home() / "tunables.toml").exists()


def _bridge_record(persona_dir: Path, pid_token: str | None) -> None:
    """A bridge.json for a bridge running as this test process (`pid_token=own_token()`), or a stale one (another token)."""
    (persona_dir / "bridge.json").write_text(
        json.dumps({
            "persona": persona_dir.name, "pid": os.getpid(), "port": 1, "started_at": "2026-11-01T00:00:00+00:00",
            "stopped_at": None, "shutdown_clean": False, "client_origin": "launchd", "pid_token": pid_token, "identity": "token",
        }),
        encoding="utf-8",
    )


def _changed(real: Path, before: dict, live_before: set[str]) -> bool:
    """What the isolation test asserts: a change outside the personas whose supervisor was live at either check."""
    live = live_before | _live_personas(real)
    return _outside(_snapshot(real), live) != _outside(before, live)


@pytest.mark.no_base_failure(reason="#600 test infrastructure: it checks this file's own snapshot helpers, not brain code")
def test_a_live_supervisor_writing_its_own_persona_is_not_a_leak(tmp_path):
    """#600: the installed app's supervisor writes `personas/<name>/` (and its SQLite sidecars) whenever it likes. That must not read
    as this test writing into the real home; a new persona directory still must."""
    real = tmp_path / "home"
    live = real / "personas" / "live"
    live.mkdir(parents=True)
    _bridge_record(live, process_identity.own_token())
    (live / "hebbian.db").write_bytes(b"a")
    live_before, before = _live_personas(real), _snapshot(real)
    assert live_before == {"live"}

    (live / "hebbian.db").write_bytes(b"ab")
    (live / "hebbian.db-wal").write_bytes(b"")
    (live / "new_cadence.json").write_text("{}")
    assert not _changed(real, before, live_before)

    (real / "personas" / "p").mkdir()
    assert _changed(real, before, live_before)


@pytest.mark.no_base_failure(reason="#600 guard: only a live supervisor's own persona is skipped")
@pytest.mark.parametrize("record", ["none", "stale"])
def test_a_write_into_a_persona_without_a_live_supervisor_is_a_leak(tmp_path, record):
    """ToT's review of #617: the first fix skipped every persona's files on every machine. A persona with no bridge.json, or one whose
    recorded bridge is not running, gets the full comparison."""
    real = tmp_path / "home"
    idle = real / "personas" / "idle"
    idle.mkdir(parents=True)
    if record == "stale":
        _bridge_record(idle, "not-this-process")
    (idle / "memories.db").write_bytes(b"a")
    live_before, before = _live_personas(real), _snapshot(real)
    assert live_before == set()

    (idle / "memories.db").write_bytes(b"ab")
    assert _changed(real, before, live_before)


@pytest.mark.no_base_failure(reason="#600 guard: the sidecar filter is limited to persona directories")
def test_a_sidecar_outside_a_persona_is_a_leak(tmp_path):
    real = tmp_path / "home"
    real.mkdir()
    before = _snapshot(real)
    (real / "settings.db-wal").write_bytes(b"")
    assert _changed(real, before, set())


@pytest.mark.no_base_failure(reason="#600 guard: the narrower snapshot must still see the #488 leak, which the old one saw too")
def test_a_top_level_write_is_still_a_leak(tmp_path):
    real = tmp_path / "home"
    real.mkdir()
    (real / "tunables.toml").write_text("x = 1\n")
    before = _snapshot(real)
    (real / "tunables.toml").write_text("x = 22\n")
    assert _changed(real, before, set())


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


# --- a test's own monkeypatch.undo() must not reopen the real home (#530) ---------------------------------------------------


def test_a_tests_monkeypatch_undo_does_not_put_back_the_real_data_home(monkeypatch):
    """The fixture's patches are held by its own MonkeyPatch. A test of the model setter called ``monkeypatch.undo()`` and then
    ``set_job_models``, which wrote ``model.interactive_chat`` into the real install's ``tunables.toml``. After an ``undo()`` the
    default home must still be the test's own, a write must land there, and the real file must be byte-for-byte what it was."""
    from brain.bridge.model_tier import TIER_INTERACTIVE_CHAT, set_job_models
    from brain.tunables_migration import LEGACY_FILE_NAME

    real_files = [REAL_PLATFORM_DIRS["user_data_path"] / name for name in ("tunables.toml", LEGACY_FILE_NAME)]
    before = [f.read_bytes() if f.exists() else None for f in real_files]
    monkeypatch.delenv("KINDLED_HOME", raising=False)
    monkeypatch.delenv("NELLBRAIN_HOME", raising=False)
    monkeypatch.undo()
    monkeypatch.delenv("KINDLED_HOME", raising=False)
    monkeypatch.delenv("NELLBRAIN_HOME", raising=False)
    home = paths.get_home()
    assert not _is_under(home, REAL_PLATFORM_DIRS["user_data_path"])
    for var in ("XDG_DATA_HOME", "XDG_STATE_HOME"):
        assert not _is_under(Path(os.environ[var]).resolve(), REAL_PLATFORM_DIRS["user_data_path"])
    set_job_models({TIER_INTERACTIVE_CHAT: "claude-opus-5"})  # a value that differs from the default, so a file is written
    assert (home / "tunables.toml").exists()
    after = [f.read_bytes() if f.exists() else None for f in real_files]
    assert after == before
