"""Offline unit tests for the drop-in code-ingestion capability (tests.harness.dropin).

Covers the gating criteria C1, C2, C12, C3, C4a/b/c, C5a/b and the C5-fires (ST1.5f) self-test.
ALL offline / network-free: a tiny zero-dependency ``fakebrain`` tree stands in for the real
``brain`` package, and every venv is a BARE venv (``deps="none"``) built with ``uv venv`` /
``python -m venv --without-pip`` — both verified offline. The ``.pth`` startup hook is exercised in
a REAL venv ``site-packages`` (a ``.pth`` does NOT fire from a bare ``PYTHONPATH`` dir), which is the
same ``site.py`` mechanism a ``-m brain.mcp_server`` child uses.

The subprocess driver loads ``dropin.py`` STANDALONE (by file path) rather than importing the
``tests.harness`` package, because a bare sandbox venv has none of the harness's third-party
dependencies installed; ``dropin.py`` imports only the standard library, so it loads cleanly there.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.harness import (
    DropinBuild,
    DropinMismatch,
    assert_brain_under_dropin,
    ingest_version,
)
from tests.harness import dropin as dropin_mod

DROPIN_PATH = Path(dropin_mod.__file__).resolve()

# A standalone driver: loads dropin.py by path (no harness package import) and runs the explicit
# prompt-side guard, exiting 3 on DropinMismatch so the parent can distinguish it from a crash.
_DRIVER = """\
import importlib.util
import sys
from pathlib import Path

dropin_path, repo, pyexe = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("dropin_standalone", dropin_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod  # register so @dataclass can resolve the module by __module__
spec.loader.exec_module(mod)
build = mod.DropinBuild(repo=Path(repo), python=Path(pyexe), source=Path(repo))
try:
    mod.assert_brain_under_dropin(build)
except mod.DropinMismatch as exc:
    sys.stderr.write("DROPIN_MISMATCH: " + str(exc))
    sys.exit(3)
print("PASS")
"""


# --- helpers -------------------------------------------------------------------------------------


def _make_fakebrain(base: Path) -> Path:
    """Write a zero-dependency ``brain`` package under ``base`` (with an inner ``mcp_server`` sub)."""
    pkg = base / "brain"
    (pkg / "mcp_server").mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("# fake brain package\n", encoding="utf-8")
    (pkg / "mcp_server" / "__init__.py").write_text("# fake brain.mcp_server\n", encoding="utf-8")
    return base


def _source_repo(tmp_path: Path) -> Path:
    """A source tree containing a fake ``brain/`` (what a real version-under-test looks like)."""
    src = tmp_path / "src"
    src.mkdir()
    _make_fakebrain(src)
    return src


def _neutral_cwd(tmp_path: Path) -> Path:
    """An empty dir to use as subprocess cwd so ``brain`` never resolves from the working dir."""
    d = tmp_path / "neutral"
    d.mkdir(exist_ok=True)
    return d


def _driver_file(tmp_path: Path) -> Path:
    p = tmp_path / "driver.py"
    p.write_text(_DRIVER, encoding="utf-8")
    return p


def _run(argv: list[str], *, cwd: Path, pythonpath: Path) -> subprocess.CompletedProcess:
    env = _clean_env()
    env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(argv, cwd=str(cwd), env=env, capture_output=True, text=True)


def _clean_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.pop("CE_DROPIN_EXPECT_ROOT", None)  # never let an outer override leak into a subprocess test
    return env


# --- C1: copy independence -----------------------------------------------------------------------


def test_c1_copy_independent_of_source(tmp_path: Path) -> None:
    src = _source_repo(tmp_path)
    dest = tmp_path / "dest"
    build = ingest_version(src, dest, deps="none", install_guard=False)

    assert (build.repo / "brain" / "__init__.py").is_file()
    # A sentinel written into the copy must not appear in the source (independence, both ways).
    (build.repo / "SENTINEL").write_text("copy-only", encoding="utf-8")
    assert not (src / "SENTINEL").exists()


# --- C2: prior-copy lifecycle --------------------------------------------------------------------


def test_c2_delete_replaces_prior_copy(tmp_path: Path) -> None:
    src = _source_repo(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "OLD_MARKER").write_text("stale", encoding="utf-8")

    build = ingest_version(src, dest, on_existing="delete", deps="none", install_guard=False)

    assert not (dest / "OLD_MARKER").exists()  # old copy gone
    assert (build.repo / "brain" / "__init__.py").is_file()  # new copy present


def test_c2_archive_zips_prior_copy_and_excludes_venv(tmp_path: Path) -> None:
    src = _source_repo(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "OLD_MARKER").write_text("stale", encoding="utf-8")
    # A fake prior drop-in venv that must NOT bloat the archive.
    (dest / ".dropin-venv" / "bin").mkdir(parents=True)
    (dest / ".dropin-venv" / "bin" / "python").write_text("junk", encoding="utf-8")

    build = ingest_version(src, dest, on_existing="archive", deps="none", install_guard=False)

    archives = list(tmp_path.glob("dest.*.zip"))
    assert len(archives) == 1, archives
    archive = archives[0]
    assert zipfile.is_zipfile(archive)
    names = zipfile.ZipFile(archive).namelist()
    assert any(n.endswith("OLD_MARKER") for n in names), names
    assert not any(".dropin-venv" in n for n in names), names  # venv excluded from archive
    assert (build.repo / "brain" / "__init__.py").is_file()  # new copy installed


def test_c2_prompt_without_tty_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = _source_repo(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "OLD_MARKER").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with pytest.raises(DropinMismatch, match="on_existing"):
        ingest_version(src, dest, on_existing="prompt", deps="none", install_guard=False)


# --- C12: destructive-path safety ----------------------------------------------------------------


def test_c12_dest_equals_source_raises_source_intact(tmp_path: Path) -> None:
    src = _source_repo(tmp_path)
    with pytest.raises(DropinMismatch):
        ingest_version(src, src, deps="none", install_guard=False)
    assert (src / "brain" / "__init__.py").is_file()


def test_c12_dest_under_source_raises_source_intact(tmp_path: Path) -> None:
    src = _source_repo(tmp_path)
    with pytest.raises(DropinMismatch):
        ingest_version(src, src / "nested_dest", deps="none", install_guard=False)
    assert (src / "brain" / "__init__.py").is_file()


def test_c12_source_under_dest_raises_source_intact(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    src = dest / "inner_src"
    src.mkdir()
    _make_fakebrain(src)
    with pytest.raises(DropinMismatch):
        ingest_version(src, dest, deps="none", install_guard=False)
    assert (src / "brain" / "__init__.py").is_file()


# --- C3: dedicated sandbox venv ------------------------------------------------------------------


def test_c3_builds_dedicated_venv_under_dest(tmp_path: Path) -> None:
    src = _source_repo(tmp_path)
    dest = tmp_path / "dest"
    build = ingest_version(src, dest, deps="none", install_guard=False)

    assert build.python.exists()
    # Lexical containment: the interpreter PATH is inside the copy. NOT realpath — a venv bin/python
    # is a symlink to the base interpreter (outside dest), so realpath would (correctly, for the
    # guard) point outside; here we care that the launch path lives under the copy.
    assert dest.resolve() in build.python.parents
    # Unresolved inequality: it is NOT the caller's interpreter path. (Resolving both would collapse
    # the two symlinks onto the same base cpython and wrongly compare equal.)
    assert build.python != Path(sys.executable)
    assert build.venv_root == build.python.parent.parent


# --- C4a / C5a / C5b: the EXPLICIT prompt-side guard (via subprocess) -----------------------------


def test_c4a_explicit_call_passes_when_correctly_wired(tmp_path: Path) -> None:
    src = _source_repo(tmp_path)
    dest = tmp_path / "dest"
    build = ingest_version(src, dest, deps="none", install_guard=False)
    driver = _driver_file(tmp_path)

    # Run UNDER the built venv (sys.prefix == venv_root) with brain resolving under the copy.
    proc = _run(
        [str(build.python), str(driver), str(DROPIN_PATH), str(build.repo), str(build.python)],
        cwd=_neutral_cwd(tmp_path),
        pythonpath=build.repo,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PASS" in proc.stdout


def test_c5a_explicit_call_raises_on_wrong_venv(tmp_path: Path) -> None:
    src = _source_repo(tmp_path)
    dest = tmp_path / "dest"
    build = ingest_version(src, dest, deps="none", install_guard=False)
    driver = _driver_file(tmp_path)

    # Run under a DIFFERENT interpreter (the test's own), even though brain resolves under the copy:
    # the venv-identity check must still fail (the exact original wrong-interpreter incident).
    proc = _run(
        [sys.executable, str(driver), str(DROPIN_PATH), str(build.repo), str(build.python)],
        cwd=_neutral_cwd(tmp_path),
        pythonpath=build.repo,
    )
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    assert "NOT running under the sandbox venv" in proc.stderr
    assert str(build.venv_root.resolve()) in proc.stderr  # names the expected venv root
    assert str(Path(sys.prefix).resolve()) in proc.stderr  # names the running venv root


def test_c5b_explicit_call_raises_on_brain_outside_copy(tmp_path: Path) -> None:
    src = _source_repo(tmp_path)
    dest = tmp_path / "dest"
    build = ingest_version(src, dest, deps="none", install_guard=False)
    driver = _driver_file(tmp_path)
    outside = _make_fakebrain(tmp_path / "outside")  # a brain NOT under the copy

    proc = _run(
        [str(build.python), str(driver), str(DROPIN_PATH), str(build.repo), str(build.python)],
        cwd=_neutral_cwd(tmp_path),
        pythonpath=outside,
    )
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    assert "NOT under the sandbox copy" in proc.stderr
    assert str(build.repo.resolve()) in proc.stderr


# --- C4b / C5b: the venv STARTUP HOOK (.pth fires at interpreter start) ---------------------------


def _build_with_hook(tmp_path: Path, name: str = "dest") -> DropinBuild:
    src = _source_repo(tmp_path)
    dest = tmp_path / name
    return ingest_version(src, dest, deps="none", install_guard=True)


def test_c4b_hook_fires_and_passes_when_brain_under_copy(tmp_path: Path) -> None:
    build = _build_with_hook(tmp_path)
    proc = _run(
        [str(build.python), "-c", "print('ok')"],
        cwd=_neutral_cwd(tmp_path),
        pythonpath=build.repo,  # brain resolves under the expected root
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stdout.strip() == "ok"


def test_c5b_hook_exits_70_when_brain_outside_copy(tmp_path: Path) -> None:
    build = _build_with_hook(tmp_path)
    outside = _make_fakebrain(tmp_path / "outside")  # brain shadowed from OUTSIDE the expected root
    proc = _run(
        [str(build.python), "-c", "print('should-not-print')"],
        cwd=_neutral_cwd(tmp_path),
        pythonpath=outside,
    )
    assert proc.returncode == 70, (proc.returncode, proc.stdout, proc.stderr)
    assert "ce-dropin guard" in proc.stderr
    assert "should-not-print" not in proc.stdout  # user code never ran


# --- C4c: no tool name in the shipped guard or the generated hook --------------------------------


def test_c4c_no_tool_names_in_dropin_or_generated_hook(tmp_path: Path) -> None:
    shipped = DROPIN_PATH.read_text(encoding="utf-8")
    hook_text = dropin_mod._hook_source(tmp_path / "dest")
    haystacks = {"dropin.py": shipped, "generated-hook": hook_text}

    forbidden = ["NELL_TOOL_NAMES", "roster", "list_tools"]
    try:
        from brain.tools import NELL_TOOL_NAMES

        forbidden.extend(NELL_TOOL_NAMES)
    except Exception:  # noqa: BLE001 — if brain isn't importable, still enforce the literal checks
        pass

    for where, text in haystacks.items():
        for token in forbidden:
            assert token not in text, f"{token!r} found in {where}"


# --- C5-fires (ST1.5f): the oracle demonstrably fires on known-bad inputs -------------------------


def test_c5_fires_on_known_bad_inputs(tmp_path: Path) -> None:
    """ST1.5f self-test: the guard is shown to FAIL on the mis-wired states, so a clean positive is
    trustworthy. Exercises both carriers directly (no subprocess) against known-bad inputs."""
    src = _source_repo(tmp_path)
    dest = tmp_path / "dest"
    build = ingest_version(src, dest, deps="none", install_guard=False)

    # Explicit call, wrong venv: the CURRENT interpreter is not the sandbox venv -> must raise.
    assert Path(sys.prefix).resolve() != build.venv_root.resolve()
    with pytest.raises(DropinMismatch, match="sandbox venv"):
        assert_brain_under_dropin(build)

    # Containment predicate fires on a brain outside the copy, and passes on one inside it.
    outside_origin = _make_fakebrain(tmp_path / "outside") / "brain" / "__init__.py"
    inside_origin = build.repo / "brain" / "__init__.py"
    assert not dropin_mod._is_under(outside_origin, build.repo)
    assert dropin_mod._is_under(inside_origin, build.repo)
