"""Tests for scripts/update.sh via --dry-run against a fake `nell` shim (#179).

The shim answers `nell paths <key>` with canned values and echoes everything
else, so no real git/uv/wheel/bridge is touched. Windows is #255.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash script; Windows is #255")

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "update.sh"
BASH = "/bin/bash"  # absolute: one test empties PATH


def _shim(tmp_path: Path, install_root: Path, kind: str) -> Path:
    shim = tmp_path / "bin" / "nell"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(
        "#!/bin/sh\n"
        'case "$1 $2" in\n'
        f"  'paths install_root') echo '{install_root}';;\n"
        f"  'paths install_kind') echo '{kind}';;\n"
        f"  'paths persona_dir') echo '{tmp_path}';;\n"
        "  '--version ') echo 'nell 0.0.42';;\n"
        "esac\n",
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return shim


def _src(tmp_path: Path, name: str = "src") -> Path:
    src = tmp_path / name
    src.mkdir(parents=True, exist_ok=True)
    (src / "pyproject.toml").write_text('version = "0.0.42"\n', encoding="utf-8")
    return src


def _run(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run([BASH, str(SCRIPT), *args], capture_output=True, text=True, env=env)


def _plan(cp: subprocess.CompletedProcess) -> list[str]:
    return [line[len("plan: ") :] for line in cp.stdout.splitlines() if line.startswith("plan: ")]


def test_missing_tool_is_a_clear_preflight_failure(tmp_path):
    shim = _shim(tmp_path, tmp_path / "repo" / ".venv", "source")
    empty = tmp_path / "emptybin"
    empty.mkdir()
    cp = _run("--nell", str(shim), "--persona", "p", "--dry-run", env_extra={"PATH": str(empty)})
    assert cp.returncode == 2
    assert "git" in cp.stderr and "uv" in cp.stderr


def test_source_kind_plan(tmp_path):
    repo = _src(tmp_path, "repo")
    (repo / ".venv").mkdir()
    shim = _shim(tmp_path, repo / ".venv", "source")
    cp = _run("--nell", str(shim), "--persona", "p", "--dry-run")
    assert cp.returncode == 0, cp.stderr
    plan = _plan(cp)
    stop, start = f"{shim} supervisor stop --persona p", f"{shim} supervisor start --persona p"
    pull = f"git -C {repo} pull --ff-only"
    assert stop in plan and start in plan and pull in plan
    assert any("uv sync --all-extras" in line for line in plan)
    assert plan.index(stop) < plan.index(pull) < plan.index(start)


def test_no_restart_skips_supervisor_calls(tmp_path):
    repo = _src(tmp_path, "repo")
    (repo / ".venv").mkdir()
    shim = _shim(tmp_path, repo / ".venv", "source")
    cp = _run("--nell", str(shim), "--persona", "p", "--dry-run", "--no-restart")
    assert cp.returncode == 0, cp.stderr
    assert not [line for line in _plan(cp) if " supervisor " in line]


def _bundled_root(root: Path) -> Path:
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "nell").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "bin" / "python3").write_text("#!/bin/sh\n", encoding="utf-8")
    return root


def test_bundled_kind_plan_uses_wheel_and_restores_wrapper(tmp_path):
    root = _bundled_root(tmp_path / "python-runtime")
    src = _src(tmp_path)
    shim = _shim(tmp_path, root, "bundled")
    cp = _run("--nell", str(shim), "--persona", "p", "--dry-run", "--source", str(src))
    assert cp.returncode == 0, cp.stderr
    joined = "\n".join(_plan(cp))
    assert "uv build --wheel" in joined
    assert "--require-hashes" in joined
    assert f"--python '{root}/bin/python3' --no-deps" in joined
    assert f"cp {root}/bin/nell {root}/bin/nell.orig" in joined
    restore = f"mv {root}/bin/nell.orig {root}/bin/nell"
    assert restore in joined
    assert joined.index("--no-deps") < joined.index(restore)  # restore AFTER pip clobbers it
    assert "git clone" not in joined  # --source given


def test_bundled_without_source_clones_ref(tmp_path):
    root = _bundled_root(tmp_path / "python-runtime")
    shim = _shim(tmp_path, root, "bundled")
    cp = _run("--nell", str(shim), "--persona", "p", "--dry-run", "--ref", "v0.0.42")
    assert cp.returncode == 0, cp.stderr
    assert any(
        line.startswith("git clone --depth 1 --branch v0.0.42 https://github.com/hanamorix/companion-emergence")
        for line in _plan(cp)
    )
    assert any("pyproject.toml" in line for line in _plan(cp))  # the check it can't run yet


@pytest.mark.skipif(getattr(os, "geteuid", lambda: -1)() == 0, reason="root can write anything")
def test_unwritable_root_plans_sudo(tmp_path):
    """Only the writes into the runtime run as root, as `sudo -H` with an absolute uv:
    Ubuntu's secure_path drops ~/.local/bin (uv's default home), and -H keeps root's
    uv cache out of the user's ~/.cache. Build + export stay unprivileged, and the
    password is asked for before anything is stopped."""
    root = _bundled_root(tmp_path / "python-runtime")
    src = _src(tmp_path)
    root.chmod(0o555)
    try:
        shim = _shim(tmp_path, root, "bundled")
        cp = _run("--nell", str(shim), "--persona", "p", "--dry-run", "--source", str(src),
                  env_extra={"HTTPS_PROXY": "http://proxy.test:3128"})
    finally:
        root.chmod(0o755)
    assert cp.returncode == 0, cp.stderr
    plan = _plan(cp)
    uv = shutil.which("uv")
    # env_reset drops proxy/CA/UV_* settings; the root uv must still get them.
    root_uv = [line for line in plan if f"{uv} pip install --python {root}/bin/python3 --require-hashes" in line]
    assert root_uv and root_uv[0].startswith("sudo -H env ") and "HTTPS_PROXY=http://proxy.test:3128" in root_uv[0]
    wheel = [line for line in plan if "--no-deps" in line]
    assert wheel and wheel[0].startswith("sudo -H env ") and f"sh -c '{uv}' pip install" in wheel[0]
    assert plan.index("sudo -v") < plan.index(f"{shim} supervisor stop --persona p")
    assert any(line.startswith("sudo -H ") and line.endswith(f"cp {root}/bin/nell {root}/bin/nell.orig") for line in plan)
    assert any(line.startswith("sudo -H ") and line.endswith(f"mv {root}/bin/nell.orig {root}/bin/nell") for line in plan)
    assert not [line for line in plan if "uv build" in line and line.startswith("sudo")]
    assert not [line for line in plan if "--no-restart" in line]  # no whole-script re-exec


def test_app_bundle_refused_without_flag(tmp_path):
    root = _bundled_root(tmp_path / "Companion Emergence.app" / "Contents" / "Resources" / "python-runtime")
    src = _src(tmp_path)
    shim = _shim(tmp_path, root, "bundled")
    cp = _run("--nell", str(shim), "--persona", "p", "--dry-run", "--source", str(src))
    assert cp.returncode == 2
    assert "--allow-app-rewrite" in cp.stderr
    cp2 = _run("--nell", str(shim), "--persona", "p", "--dry-run", "--source", str(src), "--allow-app-rewrite")
    assert cp2.returncode == 0, cp2.stderr


def test_unknown_install_kind_is_rejected(tmp_path):
    shim = _shim(tmp_path, tmp_path / "x", "weird")
    cp = _run("--nell", str(shim), "--persona", "p", "--dry-run")
    assert cp.returncode == 2 and "install_kind" in cp.stderr


def test_every_nell_command_the_script_calls_exists_in_the_real_cli():
    """The shim answers anything, so check each `"$NELL" <cmd> <action>` call site
    (plan steps AND the failure trap) against the real parser (#285: the script
    called `nell service stop`, which no nell version has ever had)."""
    calls = set(re.findall(r'"\$NELL" ([a-z-]+) ([a-z-]+)', SCRIPT.read_text(encoding="utf-8")))
    assert calls, "no nell call sites found — did the quoting change?"
    for cmd, action in sorted(calls):
        cp = subprocess.run(
            [sys.executable, "-c", "import sys; from brain.cli import main; sys.exit(main(sys.argv[1:]))",
             cmd, action, "--help"],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert cp.returncode == 0, f"`nell {cmd} {action}` is not a real command:\n{cp.stderr}"


def test_restart_without_persona_is_refused(tmp_path):
    repo = _src(tmp_path, "repo")
    (repo / ".venv").mkdir()
    shim = _shim(tmp_path, repo / ".venv", "source")
    cp = _run("--nell", str(shim), "--dry-run")
    assert cp.returncode == 2 and "--persona" in cp.stderr  # `nell supervisor` requires it
    assert _run("--nell", str(shim), "--dry-run", "--no-restart").returncode == 0


def _old_nell(bindir: Path) -> Path:
    """A pre-#179 nell (every release up to v0.0.42): rejects the install keys."""
    bindir.mkdir(parents=True, exist_ok=True)
    nell = bindir / "nell"
    nell.write_text(
        "#!/bin/sh\n"
        'case "$1 $2" in\n'
        "  'paths install_root'|'paths install_kind') echo \"unknown key '$2'\" >&2; exit 2;;\n"
        f"  'paths persona_dir') echo '{bindir}';;\n"
        "  '--version ') echo 'companion-emergence 0.0.42';;\n"
        "esac\n",
        encoding="utf-8",
    )
    nell.chmod(nell.stat().st_mode | stat.S_IEXEC)
    return nell


def test_old_nell_install_found_via_python_beside_it(tmp_path):
    root = tmp_path / "python-runtime"
    nell = _old_nell(root / "bin")
    py = root / "bin" / "python3"
    py.write_text(f"#!/bin/sh\nprintf '%s\\n%s\\n' '{root}' bundled\n", encoding="utf-8")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    link = tmp_path / "localbin" / "nell"  # the app's ~/.local/bin/nell symlink
    link.parent.mkdir()
    link.symlink_to(nell)
    src = _src(tmp_path)
    cp = _run("--nell", str(link), "--persona", "p", "--dry-run", "--source", str(src))
    assert cp.returncode == 0, cp.stderr
    assert f"--python '{root}/bin/python3' --no-deps" in "\n".join(_plan(cp))


def test_python_probe_reports_the_real_interpreter_install(tmp_path):
    """Through-path for the probe snippet itself: a real interpreter, a real `brain`."""
    checkout = Path(sys.prefix).parent
    if not (checkout / "pyproject.toml").exists():
        pytest.skip("test interpreter's venv is not inside a checkout")
    nell = _old_nell(tmp_path / "bin")
    py = tmp_path / "bin" / "python3"
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    cp = _run("--nell", str(nell), "--persona", "p", "--dry-run")
    assert cp.returncode == 0, cp.stderr
    assert f"git -C {checkout} pull --ff-only" in _plan(cp)


def test_source_without_version_line_is_refused_before_stopping(tmp_path):
    root = _bundled_root(tmp_path / "python-runtime")
    src = tmp_path / "src"
    src.mkdir()
    (src / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    shim = _shim(tmp_path, root, "bundled")
    cp = _run("--nell", str(shim), "--persona", "p", "--dry-run", "--source", str(src))
    assert cp.returncode == 2 and "version" in cp.stderr
    assert not _plan(cp)


@pytest.mark.skipif(getattr(os, "geteuid", lambda: -1)() == 0, reason="root can write anything")
def test_failed_sudo_says_the_update_was_not_applied(tmp_path):
    root = _bundled_root(tmp_path / "python-runtime")
    src = _src(tmp_path)
    shim = _shim(tmp_path, root, "bundled")
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    sudo = fakebin / "sudo"
    sudo.write_text("#!/bin/sh\necho 'sudo: a password is required' >&2\nexit 1\n", encoding="utf-8")
    sudo.chmod(sudo.stat().st_mode | stat.S_IEXEC)
    root.chmod(0o555)
    try:  # a real (non-dry) run: it stops at the sudo step, before any install
        cp = _run("--nell", str(shim), "--persona", "p", "--source", str(src),
                  env_extra={"PATH": f"{fakebin}{os.pathsep}{os.environ['PATH']}"})
    finally:
        root.chmod(0o755)
    assert cp.returncode != 0
    assert "NOT applied" in cp.stderr


def test_help_prints_the_whole_header():
    cp = _run("--help")
    assert cp.returncode == 0
    assert cp.stdout.rstrip().endswith("executes nothing past preflight.")
    assert "set -euo" not in cp.stdout


def test_nell_runs_from_a_neutral_cwd_even_from_a_checkout(tmp_path):
    """Bundled nell runs `python3 -c`, which puts the cwd on sys.path: from a checkout
    (how the README says to run this) it imports the checkout's brain/ and misreports
    install_kind and --version. Every nell call must run from `/`; relative
    --nell/--source must still resolve."""
    root = _bundled_root(tmp_path / "python-runtime")
    _src(tmp_path)
    log = tmp_path / "cwd.log"
    shim = tmp_path / "bin" / "nell"
    shim.parent.mkdir()
    shim.write_text(
        "#!/bin/sh\n"
        f"pwd >> '{log}'\n"
        'case "$1 $2" in\n'
        f"  'paths install_root') echo '{root}';;\n"
        "  'paths install_kind') echo bundled;;\n"
        f"  'paths persona_dir') echo '{tmp_path}';;\n"
        "esac\n",
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    cp = subprocess.run(
        [BASH, str(SCRIPT), "--nell", "bin/nell", "--source", "src", "--persona", "p", "--dry-run"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert cp.returncode == 0, cp.stderr
    assert set(log.read_text(encoding="utf-8").split()) == {"/"}
    assert f"--no-deps --quiet '{tmp_path}/src'/dist/*.whl" in "\n".join(_plan(cp))


def test_unknown_persona_is_refused_before_stopping(tmp_path):
    """`supervisor stop` says "not running" (exit 0) for a persona with no state, so a
    typo would replace the runtime under the live bridge. Refuse it up front."""
    root = _bundled_root(tmp_path / "python-runtime")
    src = _src(tmp_path)
    shim = _shim(tmp_path, root, "bundled")
    text = shim.read_text(encoding="utf-8").replace(f"echo '{tmp_path}';;", f"echo '{tmp_path}/nope';;")
    shim.write_text(text, encoding="utf-8")
    cp = _run("--nell", str(shim), "--persona", "nel", "--dry-run", "--source", str(src))
    assert cp.returncode == 2 and "nel" in cp.stderr and "persona" in cp.stderr
    assert not _plan(cp)


def _logging_shim(tmp_path: Path, root: Path, *, stop_rc: int = 0, start_rc: int = 0) -> tuple[Path, Path]:
    """A bundled-kind nell that logs every supervisor call and exits as told."""
    log = tmp_path / "calls.log"
    shim = tmp_path / "bin" / "nell"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(
        "#!/bin/sh\n"
        'case "$1 $2" in\n'
        f"  'paths install_root') echo '{root}';;\n"
        "  'paths install_kind') echo bundled;;\n"
        f"  'paths persona_dir') echo '{tmp_path}';;\n"
        f"  'supervisor stop') echo stop >> '{log}'; exit {stop_rc};;\n"
        f"  'supervisor start') echo start >> '{log}'; exit {start_rc};;\n"
        "  '--version ') echo 'companion-emergence 0.0.42';;\n"
        "esac\n",
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return shim, log


def test_stop_timeout_still_restarts_the_brain(tmp_path):
    """`supervisor stop` exits 1 on its timeout while the bridge may still go down:
    the trap must restart it (spec: never leave the brain down)."""
    root = _bundled_root(tmp_path / "python-runtime")
    src = _src(tmp_path)
    shim, log = _logging_shim(tmp_path, root, stop_rc=1)
    cp = _run("--nell", str(shim), "--persona", "p", "--source", str(src))  # real run: dies at stop
    assert cp.returncode != 0
    assert log.read_text(encoding="utf-8").split() == ["stop", "start"]


def _fakebin(tmp_path: Path, **tools: str) -> Path:
    """A PATH dir of stub tools: name -> shell body. Each call is logged to calls.log."""
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir(exist_ok=True)
    for name, body in tools.items():
        f = fakebin / name
        f.write_text(f"#!/bin/sh\necho {name} \"$@\" >> '{tmp_path}/tools.log'\n{body}\n", encoding="utf-8")
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return fakebin


def test_bridge_back_up_after_update_is_restarted_onto_new_code(tmp_path):
    """The app can relaunch the bridge mid-update (app start / Restart button); it then
    runs old or mixed code, and `supervisor start` exits 2. Restart it."""
    root = _bundled_root(tmp_path / "python-runtime")
    src = _src(tmp_path)
    shim, log = _logging_shim(tmp_path, root, start_rc=2)
    shim.write_text(
        shim.read_text(encoding="utf-8").replace(
            "esac\n", f"  'supervisor restart') echo restart >> '{log}';;\nesac\n"
        ),
        encoding="utf-8",
    )
    fakebin = _fakebin(tmp_path, uv="exit 0")
    cp = _run("--nell", str(shim), "--persona", "p", "--source", str(src),
              env_extra={"PATH": f"{fakebin}{os.pathsep}{os.environ['PATH']}"})
    assert cp.returncode == 0, cp.stderr
    assert log.read_text(encoding="utf-8").split() == ["stop", "start", "restart"]


def test_bare_nell_name_is_looked_up_on_path(tmp_path):
    repo = _src(tmp_path, "repo")
    (repo / ".venv").mkdir()
    shim = _shim(tmp_path, repo / ".venv", "source")
    cp = _run("--nell", "nell", "--persona", "p", "--dry-run",
              env_extra={"PATH": f"{shim.parent}{os.pathsep}{os.environ['PATH']}"})
    assert cp.returncode == 0, cp.stderr
    assert f"{shim} supervisor stop --persona p" in _plan(cp)


def test_exported_cdpath_does_not_redirect_relative_source(tmp_path):
    root = _bundled_root(tmp_path / "python-runtime")
    _src(tmp_path)
    decoy = tmp_path / "decoy"
    _src(decoy)  # decoy/src exists too: `cd src` via CDPATH would land there
    shim = _shim(tmp_path, root, "bundled")
    cp = subprocess.run(
        [BASH, str(SCRIPT), "--nell", str(shim), "--source", "src", "--persona", "p", "--dry-run"],
        capture_output=True, text=True, cwd=tmp_path, env={**os.environ, "CDPATH": str(decoy)},
    )
    assert cp.returncode == 0, cp.stderr
    assert f"--no-deps --quiet '{tmp_path}/src'/dist/*.whl" in "\n".join(_plan(cp))


def test_hardened_umask_keeps_the_restored_wrapper_executable(tmp_path):
    """sudo unions the caller's umask with its own, so under umask 077 the cp/mv
    dance left bin/nell 0700 root:root — unrunnable by the user. Pin umask 022."""
    root = _bundled_root(tmp_path / "python-runtime")
    (root / "bin" / "nell").chmod(0o755)
    src = _src(tmp_path)
    shim = _shim(tmp_path, root, "bundled")
    fakebin = _fakebin(tmp_path, uv="exit 0")
    cp = subprocess.run(
        [BASH, "-c", 'umask 077; exec "$0" "$@"', str(SCRIPT),
         "--nell", str(shim), "--source", str(src), "--no-restart"],
        capture_output=True, text=True,
        env={**os.environ, "PATH": f"{fakebin}{os.pathsep}{os.environ['PATH']}"},
    )
    assert cp.returncode == 0, cp.stderr
    assert stat.S_IMODE((root / "bin" / "nell").stat().st_mode) == 0o755


def test_appimage_runtime_is_refused_before_stopping(tmp_path):
    """An AppImage's runtime lives in a read-only squashfs mount: no sudo can write it."""
    root = _bundled_root(tmp_path / ".mount_NellFaAbC123" / "usr" / "lib" / "python-runtime")
    src = _src(tmp_path)
    shim = _shim(tmp_path, root, "bundled")
    cp = _run("--nell", str(shim), "--persona", "p", "--dry-run", "--source", str(src))
    assert cp.returncode == 2 and "AppImage" in cp.stderr
    assert not _plan(cp)


def test_restart_racing_the_apps_own_start_is_success(tmp_path):
    """restart's start exits 2 when another starter holds the lock — that starter
    launches after the install finished, so it is on the new code."""
    root = _bundled_root(tmp_path / "python-runtime")
    src = _src(tmp_path)
    shim, log = _logging_shim(tmp_path, root, start_rc=2)
    shim.write_text(
        shim.read_text(encoding="utf-8").replace(
            "esac\n", f"  'supervisor restart') echo restart >> '{log}'; exit 2;;\nesac\n"
        ),
        encoding="utf-8",
    )
    fakebin = _fakebin(tmp_path, uv="exit 0")
    cp = _run("--nell", str(shim), "--persona", "p", "--source", str(src),
              env_extra={"PATH": f"{fakebin}{os.pathsep}{os.environ['PATH']}"})
    assert cp.returncode == 0, cp.stderr
    assert log.read_text(encoding="utf-8").split() == ["stop", "start", "restart"]
