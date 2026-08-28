"""Drop-in code ingestion + a version-agnostic startup guard against split-brain runs.

The live-test harness must be able to test an ALTERNATE version of ``brain`` honestly. The
prior approach only *pointed* a redirect at the version-under-test, and that pointer reached
only ONE of the two processes that import ``brain`` during a run: the prompt/bridge side ran
the new code while the ``brain-tools`` MCP subprocess (a fresh ``sys.executable`` child that
does not inherit the parent's in-memory ``sys.path``) served a stale one. Prompt built from the
new build, tools served from the old install, and nothing detected it.

This module makes "drop in and run" literal:

1. **Copy-in + lifecycle** (:func:`ingest_version`) — copy the version-under-test into a stable
   sandbox ``dest`` and build a dedicated venv ON that copy, so ``sys.executable`` carries the
   correct interpreter into the MCP child by construction. One install, one interpreter, every
   process, mirroring production.
2. **A version-agnostic startup guard** — cheap regression insurance that fails LOUD if the
   harness is ever mis-wired so a process resolves ``brain`` from OUTSIDE the sandbox copy
   anyway (a launcher not using the sandbox venv python, a stray ``PYTHONPATH``, an editable
   ``.pth`` shadowing the copy). It is a startup assertion that the resolved ``brain`` package
   origin is under the sandbox copy. It references NO tool names, no tool inventory, no auth, no
   live turn, so it is version-agnostic by construction. Two carriers:
   - :func:`assert_brain_under_dropin` — the explicit prompt-side call. Additionally asserts the
     prompt process is itself running under the sandbox venv (the check the original incident
     lacked), raising :class:`DropinMismatch` loudly at startup before any child is spawned.
   - A venv startup hook (``.pth`` + guard module) installed by :func:`ingest_version` into the
     sandbox venv's ``site-packages``. ``site.py`` runs it at every non-``-S`` interpreter start; it
     registers a ``sys.meta_path`` finder that validates ``brain`` when ``brain`` is imported (after
     all ``.pth`` path setup, so it is independent of ``.pth`` ordering), hard-exiting a mis-wired
     process. It fires for the MCP child and the prompt process alike.

This module composes with :mod:`tests.harness.sandbox` (which isolates STATE dirs); the two are
orthogonal — sandbox isolates state, drop-in ingestion isolates code. It imports only the standard
library so it can be loaded standalone (by file path) inside a bare sandbox venv that has no
harness dependencies installed.

**Serial-use precondition (CP7).** ``dest`` is a stable, cross-process shared location. Do NOT
re-ingest or delete/archive a ``dest`` while a run against it is live: ``on_existing`` in
``{"delete", "archive"}`` ``rmtree``s ``dest``, and pulling the module tree out from under a
running MCP child is a teardown race. This matches the operator's standing "one harness server at
a time" rule; a full cross-process lock is out of scope.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_ON_EXISTING = ("prompt", "archive", "delete")
_DEPS = ("auto", "none")
_VENV_DIRNAME = ".dropin-venv"
_HOOK_MODULE = "_ce_dropin_guard.py"
_HOOK_PTH = "_ce_dropin_guard.pth"

# Files/dirs never copied into the sandbox copy: VCS metadata, caches, any pre-existing venv, the
# drop-in venv itself, and the guarded-change stage folder.
_IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", ".venv", "venv", _VENV_DIRNAME, "changes", "*.pyc"
)


class DropinMismatch(RuntimeError):  # noqa: N818 — public API name (spec/criteria); not a SandboxLeak
    """Raised when the harness detects a split-brain drop-in state (or refuses an unsafe ingest).

    A distinct type, deliberately NOT a subclass of :class:`tests.harness.sandbox.SandboxLeak`, so a
    caller catching one does not accidentally swallow the other. Raised by
    :func:`assert_brain_under_dropin` when the calling process is not under the sandbox venv or
    resolves ``brain`` outside the sandbox copy, and by :func:`ingest_version` for a destructive-path
    violation or a non-interactive ``on_existing="prompt"``.
    """


@dataclass(frozen=True)
class DropinBuild:
    """Handle returned by :func:`ingest_version`: the copied code + its dedicated interpreter.

    - ``repo`` — the copied-in code root (contains ``brain/``); the containment root the guard checks.
    - ``python`` — the dedicated sandbox venv interpreter every run process must launch under.
    - ``source`` — the original source the copy was taken from (provenance only).
    """

    repo: Path
    python: Path
    source: Path

    @property
    def venv_root(self) -> Path:
        """The sandbox venv's prefix (``= python.parent.parent``).

        Used for the venv-identity check ``Path(sys.prefix).resolve() == venv_root.resolve()``.
        ``sys.prefix`` (not resolved ``sys.executable``) is compared because a venv ``bin/python`` is
        usually a symlink to the base interpreter, so resolving it would collapse distinct venvs to
        the same system python and pass wrongly; ``sys.prefix`` is per-venv distinct.
        """
        return self.python.parent.parent


# --- guard: shared, side-effect-free containment logic -------------------------------------------
# ``_is_under`` and the ``_DropinImportGuard`` finder below are emitted VERBATIM (via
# inspect.getsource) into the generated venv hook, so the containment logic has a single source of
# truth. They import only the standard library. ``_resolve_module_origin`` is used only by the
# explicit prompt-side call (which runs after interpreter startup, when import resolution is final);
# the venv hook does NOT resolve at startup (see ``_DropinImportGuard`` for why).


def _resolve_module_origin(module: str = "brain") -> Path | None:
    """Resolve a TOP-LEVEL package's on-disk origin WITHOUT importing/executing it.

    ``importlib.util.find_spec`` does not execute a top-level package's ``__init__.py`` (no import
    side effects, no dependency cost); the returned ``origin`` is the path ``module.__file__`` would
    hold. Returns None if the module is unresolvable or has no file origin (e.g. a namespace package),
    either of which the guard treats as a failure. Checking only ``brain`` is sufficient:
    ``brain.mcp_server`` lives inside the same ``brain/`` package dir, so containment of ``brain``
    covers it (and resolving ``brain.mcp_server`` would import the parent ``brain``, which this avoids).
    """
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def _is_under(path: Path, root: Path) -> bool:
    """True if ``path`` equals ``root`` or lies within it, compared by resolved real paths."""
    rp = os.path.realpath(path)
    rr = os.path.realpath(root)
    return rp == rr or rp.startswith(rr + os.sep)


class _DropinImportGuard:
    """A ``sys.meta_path`` finder that validates ``brain`` resolves under the sandbox copy AT IMPORT.

    The generated venv hook installs one of these at interpreter startup. Crucially it does NOT resolve
    ``brain`` during ``.pth`` processing: ``site.py`` runs ``.pth`` files in filename-sorted order, and
    an editable install (``uv pip install -e``) adds the copy to ``sys.path`` via its OWN ``.pth`` (e.g.
    ``_editable_impl_*.pth``). Our hook's ``.pth`` can sort BEFORE that one, so a startup-time
    ``find_spec("brain")`` would see an incomplete ``sys.path``, resolve nothing, and hard-exit on a
    correctly-wired run. Instead this finder validates when ``brain`` is actually imported — after
    ``site.py`` has finished all path setup — making the check independent of ``.pth`` ordering.

    Registered at the FRONT of ``sys.meta_path``. On an ``import brain`` it consults every OTHER finder
    to obtain the real spec, then checks the resolved origin is under ``expect_root``. If the origin is
    outside the copy, unresolvable, or a ``None``-origin namespace package, it writes a loud diagnostic
    and hard-exits with ``os._exit(70)`` (a raise here would surface only as an ``ImportError`` a caller
    could swallow); otherwise it returns the genuine spec so the import proceeds normally. Only the
    top-level ``brain`` is intercepted: ``brain.mcp_server`` lives inside the same package dir, so once
    ``brain`` is validated its submodules are covered.
    """

    def __init__(self, expect_root: str, module: str = "brain") -> None:
        self._expect = expect_root
        self._module = module

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001,ANN201 — import-protocol sig
        if fullname != self._module:
            return None
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                spec = find(fullname, path, target)
            except Exception:  # noqa: BLE001 — a broken finder must not mask the guard; try the next
                spec = None
            if spec is not None:
                origin = getattr(spec, "origin", None)
                if origin is None or not _is_under(Path(origin), Path(self._expect)):
                    sys.stderr.write(
                        "ce-dropin guard: this process resolved the package "
                        f"{fullname!r} to {origin!s}, which is NOT under the expected sandbox copy "
                        f"{self._expect!s}. This is a split-brain drop-in; refusing to start. "
                        "Exiting 70.\n"
                    )
                    sys.stderr.flush()
                    os._exit(70)
                return spec
        return None


def _install_import_guard(expect_root: str, module: str = "brain") -> None:
    """Register a :class:`_DropinImportGuard` at the front of ``sys.meta_path`` (idempotent)."""
    for existing in sys.meta_path:
        if type(existing).__name__ == "_DropinImportGuard":
            return
    sys.meta_path.insert(0, _DropinImportGuard(expect_root, module))


def assert_brain_under_dropin(build: DropinBuild, *, module: str = "brain") -> None:
    """Explicit prompt-side startup guard. Raises :class:`DropinMismatch` on a split-brain state.

    Asserts, in order:

    1. **Venv identity** — ``Path(sys.prefix).resolve() == build.venv_root.resolve()``. This is the
       load-bearing check that closes the original incident: the bug was a wrong-interpreter launch
       (a ``uv run python`` prompt process on the main-clone venv), and since the MCP child is spawned
       as ``sys.executable -m brain.mcp_server``, a prompt process on the wrong venv spawns a stale,
       unguarded child. Raising here aborts the run at prompt startup, before any child spawns.
       Containment alone is insufficient: a wrong interpreter with a ``sys.path`` insert can still
       resolve ``brain`` under the copy, so venv identity is checked FIRST.
    2. **Copy containment** — the resolved ``module`` origin is under ``build.repo``.

    Call it once at ``live_server`` startup after the prompt process is (meant to be) running under
    ``build.python``. The venv hook installed by :func:`ingest_version` is the defense-in-depth net
    for the fresh MCP child, which cannot check venv identity (it has no ``build`` to compare against).
    """
    running = Path(sys.prefix).resolve()
    expected = build.venv_root.resolve()
    if running != expected:
        raise DropinMismatch(
            "prompt process is NOT running under the sandbox venv: "
            f"sys.prefix {running} != build.venv_root {expected}. Launch it with the sandbox venv "
            "interpreter (build.python), not the caller's or main-clone interpreter. A wrong-venv "
            "prompt process would spawn a stale, unguarded brain-tools child."
        )
    origin = _resolve_module_origin(module)
    if origin is None or not _is_under(origin, build.repo):
        raise DropinMismatch(
            f"package {module!r} resolved to {origin}, which is NOT under the sandbox copy "
            f"{build.repo.resolve()}. A process resolving brain from outside the copy is a "
            "split-brain run."
        )


# --- copy-in + lifecycle + venv build ------------------------------------------------------------


def ingest_version(
    source: Path,
    dest: Path,
    *,
    on_existing: str = "prompt",
    deps: str = "auto",
    install_guard: bool = True,
) -> DropinBuild:
    """Copy the version-under-test into ``dest`` and build a dedicated venv on the copy.

    Args:
        source: the code root to copy (must contain ``brain/`` for a real run).
        dest: the STABLE sandbox copy location. Serial-use precondition (CP7): do not re-ingest or
            delete/archive ``dest`` while a run against it is live (an ``rmtree`` under a running MCP
            child is a teardown race). This matches the operator's "one harness server at a time" rule.
        on_existing: prior-copy lifecycle when ``dest`` already exists — ``"prompt"`` (default) asks
            archive/delete/abort on a tty and RAISES :class:`DropinMismatch` with no tty (autonomous
            runs never block); ``"archive"`` zips the old copy (excluding the drop-in venv) to
            ``<dest>.<UTCstamp>.zip`` beside ``dest`` then removes it; ``"delete"`` removes it.
        deps: ``"auto"`` (default) installs the copy editable into the venv (``uv pip install -e`` —
            real deps, NEEDS NETWORK, not run in CI); ``"none"`` builds a bare venv only (offline).
        install_guard: install the venv startup hook (``.pth`` + guard module) into the venv's
            ``site-packages`` (default on).

    Returns a :class:`DropinBuild`. Raises :class:`DropinMismatch` on a destructive-path violation or
    a non-interactive ``on_existing="prompt"``; ``ValueError`` on a bad ``on_existing``/``deps`` value.
    """
    source = Path(source).resolve()
    dest = Path(dest).resolve()

    # Step 0 (FIRST, before any rmtree/copytree): destructive-path safety. Rejecting an overlapping
    # dest here is what makes the on_existing="delete"/"archive" rmtree(dest) unable to touch source.
    if dest == source:
        raise DropinMismatch(f"dest must not equal source (both resolve to {dest})")
    if _is_under(source, dest):
        raise DropinMismatch(
            f"source {source} is nested under dest {dest} — refused (an rmtree of dest would delete "
            "part of the source tree)"
        )
    if _is_under(dest, source):
        raise DropinMismatch(
            f"dest {dest} is nested under source {source} — refused (an rmtree of dest would delete "
            "part of the source tree, and the copy would recurse into itself)"
        )

    if on_existing not in _ON_EXISTING:
        raise ValueError(f"on_existing must be one of {_ON_EXISTING!r}, got {on_existing!r}")
    if deps not in _DEPS:
        raise ValueError(f"deps must be one of {_DEPS!r}, got {deps!r}")

    if dest.exists():
        _handle_existing(dest, on_existing)

    shutil.copytree(source, dest, ignore=_IGNORE)

    venv_python = _build_venv(dest, deps)
    if install_guard:
        _install_guard_hook(venv_python, dest)

    return DropinBuild(repo=dest, python=venv_python, source=source)


def _handle_existing(dest: Path, on_existing: str) -> None:
    """Apply the prior-copy lifecycle. ``"prompt"`` with no tty RAISES (autonomous runs never block)."""
    if on_existing == "prompt":
        if not sys.stdin.isatty():
            raise DropinMismatch(
                f"dest {dest} already exists and on_existing='prompt', but there is no interactive "
                "terminal. Pass on_existing='archive' or on_existing='delete' explicitly so an "
                "autonomous run never blocks."
            )
        answer = input(
            f"A prior drop-in copy exists at {dest}. [a]rchive to zip, [d]elete, or a[b]ort? "
        ).strip().lower()
        if answer in ("a", "archive"):
            on_existing = "archive"
        elif answer in ("d", "delete"):
            on_existing = "delete"
        else:
            raise DropinMismatch(f"aborted by user; prior drop-in copy at {dest} left untouched")

    if on_existing == "archive":
        _archive_copy(dest)
    shutil.rmtree(dest)


def _archive_copy(dest: Path) -> Path:
    """Zip the code copy at ``dest`` (EXCLUDING the drop-in venv) to ``<dest>.<UTCstamp>.zip`` beside it.

    Excluding ``.dropin-venv`` keeps the archive to code (not a fat, partly symlinked venv). Arcnames
    are relative to ``dest.parent`` so the zip contains a top-level ``<dest.name>/`` tree. Returns the
    archive path.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = dest.parent / f"{dest.name}.{stamp}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(dest):
            dirs[:] = [d for d in dirs if d != _VENV_DIRNAME]
            for name in files:
                fp = Path(root) / name
                zf.write(fp, fp.relative_to(dest.parent).as_posix())
    return archive


def _venv_python(venv_dir: Path) -> Path:
    """The venv interpreter path (``Scripts/python.exe`` on Windows, ``bin/python`` elsewhere)."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _build_venv(dest: Path, deps: str) -> Path:
    """Build the dedicated venv under ``<dest>/.dropin-venv`` and return its interpreter.

    ``deps="auto"`` installs the copy editable (real deps; NEEDS NETWORK — not run in CI).
    ``deps="none"`` builds a bare venv only (offline). Prefers ``uv`` when present, falling back to the
    stdlib ``venv`` module.
    """
    venv_dir = dest / _VENV_DIRNAME
    uv = shutil.which("uv")

    if deps == "auto":
        if uv:
            _run([uv, "venv", str(venv_dir)])
            venv_py = _venv_python(venv_dir)
            _run([uv, "pip", "install", "--python", str(venv_py), "-e", str(dest)])
        else:
            _run([sys.executable, "-m", "venv", str(venv_dir)])
            venv_py = _venv_python(venv_dir)
            _run([str(venv_py), "-m", "pip", "install", "-e", str(dest)])
    else:  # "none" — bare venv, offline
        if uv:
            _run([uv, "venv", str(venv_dir)])
        else:
            _run([sys.executable, "-m", "venv", "--without-pip", str(venv_dir)])
        venv_py = _venv_python(venv_dir)

    if not venv_py.exists():
        raise DropinMismatch(f"venv build did not produce an interpreter at {venv_py}")
    return venv_py


def _run(argv: list[str]) -> None:
    """Run a build subprocess, surfacing its output on failure."""
    proc = subprocess.run(argv, capture_output=True, text=True)  # noqa: S603 — fixed argv, harness use
    if proc.returncode != 0:
        raise DropinMismatch(
            f"command failed ({proc.returncode}): {' '.join(argv)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


# --- venv startup hook ---------------------------------------------------------------------------


def _venv_purelib(venv_python: Path) -> Path:
    """The venv's PURELIB ``site-packages`` dir, queried from the venv interpreter itself."""
    proc = subprocess.run(  # noqa: S603 — fixed argv, harness use
        [str(venv_python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise DropinMismatch(
            f"could not resolve the venv purelib for {venv_python}: {proc.stderr}"
        )
    return Path(proc.stdout.strip())


def _hook_source(expect_root: Path) -> str:
    """Generate the standalone ``_ce_dropin_guard.py`` text for a venv with baked ``expect_root``.

    The hook is a self-contained, stdlib-only module (the sandbox venv has no harness code installed).
    It embeds ``_is_under`` and the ``_DropinImportGuard`` finder VERBATIM (via ``inspect.getsource``)
    so the guard logic has one source of truth, and ends with a top-level ``_install_import_guard(<abs
    expect_root>)`` call. That runs at interpreter startup when ``site.py`` processes the companion
    ``.pth``, but only REGISTERS the finder; the containment check itself fires later, when ``brain`` is
    imported (after all ``.pth`` path setup), so it is independent of ``.pth`` processing order.
    """
    root_literal = repr(os.fspath(expect_root))
    body = "\n".join(
        [
            '"""Generated harness startup hook. Do not edit — installed by tests.harness.dropin."""',
            "from __future__ import annotations",
            "",
            "import os",
            "import sys",
            "from pathlib import Path",
            "",
            inspect.getsource(_is_under),
            inspect.getsource(_DropinImportGuard),
            inspect.getsource(_install_import_guard),
            "",
            f"_install_import_guard({root_literal}, 'brain')",
            "",
        ]
    )
    return body


def _install_guard_hook(venv_python: Path, dest: Path) -> None:
    """Write ``_ce_dropin_guard.py`` + ``_ce_dropin_guard.pth`` into the venv's ``site-packages``.

    The ``.pth`` holds a single ``import _ce_dropin_guard`` line (a ``.pth`` line beginning with
    ``import`` is executed by ``site.py`` at interpreter startup). It fires only from a REGISTERED site
    dir (a real venv ``site-packages``), not a bare ``PYTHONPATH`` dir — which is why every run process
    launches under the venv interpreter.
    """
    purelib = _venv_purelib(venv_python)
    purelib.mkdir(parents=True, exist_ok=True)
    (purelib / _HOOK_MODULE).write_text(_hook_source(dest.resolve()), encoding="utf-8")
    (purelib / _HOOK_PTH).write_text("import _ce_dropin_guard\n", encoding="utf-8")
