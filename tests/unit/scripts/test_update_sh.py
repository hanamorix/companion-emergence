"""Tests for scripts/update.sh via --dry-run against a fake `nell` shim (#179).

The shim answers `nell paths <key>` with canned values and echoes everything
else, so no real git/uv/wheel/bridge is touched. Windows is #255.
"""

from __future__ import annotations

import os
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
    stop, start = f"{shim} service stop --persona p", f"{shim} service start --persona p"
    pull = f"git -C {repo} pull --ff-only"
    assert stop in plan and start in plan and pull in plan
    assert any("uv sync --all-extras" in line for line in plan)
    assert plan.index(stop) < plan.index(pull) < plan.index(start)


def test_no_restart_skips_service_calls(tmp_path):
    repo = _src(tmp_path, "repo")
    (repo / ".venv").mkdir()
    shim = _shim(tmp_path, repo / ".venv", "source")
    cp = _run("--nell", str(shim), "--persona", "p", "--dry-run", "--no-restart")
    assert cp.returncode == 0, cp.stderr
    assert not [line for line in _plan(cp) if " service " in line]


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


@pytest.mark.skipif(getattr(os, "geteuid", lambda: -1)() == 0, reason="root can write anything")
def test_unwritable_root_plans_sudo(tmp_path):
    root = _bundled_root(tmp_path / "python-runtime")
    src = _src(tmp_path)
    root.chmod(0o555)
    try:
        shim = _shim(tmp_path, root, "bundled")
        cp = _run("--nell", str(shim), "--persona", "p", "--dry-run", "--source", str(src))
    finally:
        root.chmod(0o755)
    assert cp.returncode == 0, cp.stderr
    sudo = [line for line in _plan(cp) if line.startswith("sudo ")]
    assert sudo and "--no-restart" in sudo[0] and f"--source {src}" in sudo[0]


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
    cp = _run("--nell", str(shim), "--dry-run")
    assert cp.returncode == 2 and "install_kind" in cp.stderr
