"""Install a newer brain into the overlay (#286 §3.3).

Only packages whose locked version differs from the bundle's own dist-info
records are installed (the rest come from the bundle), plus the brain wheel.
pip runs straight from ensurepip's bundled wheel — the bundle has no pip and is
never written to. Staging → import smoke → atomic swap; any failure leaves
current.json untouched.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from brain.update import overlay
from brain.update.overlay_hook import BUNDLE_ID_FILE
from brain.utils.time import iso_utc

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"  # == pyproject's pytorch-cpu index (drift test)
NEVER_INSTALL = frozenset({"pip", "setuptools", "wheel"})  # the build strips them on purpose
SMOKE_MODULES = ("brain.cli", "brain.bridge.server", "brain.chat.engine")

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^]]*\])?==([^\s;\\]+)")
_SAFE_COMMIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_OPTION_PREFIXES = ("--index-url", "--extra-index-url", "-i", "--find-links", "-f")


class UpdateError(RuntimeError):
    """The update did not install; the active overlay is unchanged."""


class Requirement(NamedTuple):
    name: str
    version: str
    block: str


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(text: str) -> tuple[list[str], list[Requirement]]:
    options: list[str] = []
    reqs: list[Requirement] = []
    current: list[str] | None = None
    head: re.Match | None = None

    def close():
        if current is not None and head is not None:
            reqs.append(Requirement(_norm(head.group(1)), head.group(3), "\n".join(current) + "\n"))

    for line in text.splitlines():
        if line[:1].isspace() and current is not None:
            current.append(line)
            continue
        close()
        current, head = None, None
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _PIN.match(stripped)
        if m:
            current, head = [line], m
            continue
        if stripped.startswith(_OPTION_PREFIXES):
            options.append(stripped)
            continue
        raise UpdateError(f"unrecognised requirement line: {line!r}")
    close()
    return options, reqs


def installed_versions(site_dir: Path) -> dict[str, str]:
    return {_norm(d.metadata["Name"]): d.version for d in md.distributions(path=[str(site_dir)])}


def diff_requirements(reqs: list[Requirement], installed: dict[str, str]) -> list[Requirement]:
    return [r for r in reqs if r.name not in NEVER_INSTALL and installed.get(r.name) != r.version]


def bundle_site_dir() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def _pip() -> list[str]:
    import ensurepip

    wheel = next(Path(ensurepip.__file__).parent.glob("_bundled/pip-*.whl"), None)
    if wheel is None:
        raise UpdateError("this Python has no ensurepip pip wheel")
    return [sys.executable, "-P", str(wheel / "pip"), "install", "--quiet",
            "--disable-pip-version-check", "--no-input", "--no-deps"]


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise UpdateError(f"pip failed ({r.returncode}): {(r.stderr or r.stdout)[-2000:]}")


def _smoke(folder: Path, modules: Sequence[str]) -> None:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import importlib, brain\n"
        "for m in sys.argv[2:]: importlib.import_module(m)\n"
        "import os; assert os.path.realpath(brain.__file__).startswith(os.path.realpath(sys.argv[1])), brain.__file__\n"
    )
    # KINDLED_NO_OVERLAY: new folder + bundle only, as it will run once active — not
    # stacked on the currently active overlay the hook would load (#303).
    r = subprocess.run([sys.executable, "-P", "-c", code, str(folder), *modules],
                       capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "KINDLED_NO_OVERLAY": "1"})
    if r.returncode != 0:
        raise UpdateError(f"the new brain does not load on this machine: {(r.stderr or r.stdout)[-2000:]}")


def apply_update(*, wheel: Path, requirements: Path, commit: str, site_dir: Path, root: Path,
                 pip_extra: Sequence[str] = (), smoke_modules: Sequence[str] = SMOKE_MODULES) -> dict:
    if not _SAFE_COMMIT.fullmatch(commit or ""):
        raise UpdateError(f"refusing unsafe commit label {commit!r}")
    wheel, requirements, site_dir = Path(wheel), Path(requirements), Path(site_dir)
    try:
        bundle_id = (site_dir / BUNDLE_ID_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        raise UpdateError("this install predates overlay updates; update the app first") from None
    parts = wheel.name.split("-")
    if len(parts) < 2 or not wheel.is_file():
        raise UpdateError(f"not a wheel: {wheel}")
    # bundle_id is folded into the folder name: an app upgrade that changes it (same
    # commit re-applied against a new bundle) installs into a NEW folder rather than
    # colliding with one built for the old bundle.
    base = f"{commit[:12]}-{bundle_id[:8]}"
    entry = {"dir": base, "commit": commit, "brain_version": parts[1], "bundle_id": bundle_id}
    with overlay.overlay_lock(root):
        state = overlay.read_state(root)
        named = {e["dir"] for e in (state["active"], state["previous"]) if e}
        target = root / entry["dir"]
        try:
            stamp = json.loads((target / "stamp.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stamp = {}
        matches = (isinstance(stamp, dict) and stamp.get("commit") == commit
                   and stamp.get("bundle_id") == bundle_id and target.name in named)
        if not matches:
            if target.exists() and target.name in named:
                # `target` is current.json's active or previous folder, just with a
                # stamp that doesn't match (e.g. corrupted) — never rename or delete
                # a folder current.json names; install into a fresh one instead. Pick
                # the first counter not already named (not pid-based: a prior run's
                # `-r<pid>` folder could still be named when the OS reuses that pid).
                n = 1
                while f"{base}-r{n}" in named:
                    n += 1
                entry["dir"] = f"{base}-r{n}"
                target = root / entry["dir"]
            staging = root / f".staging-{entry['dir']}-{os.getpid()}"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            try:
                options, reqs = parse_requirements(requirements.read_text(encoding="utf-8"))
                todo = diff_requirements(reqs, installed_versions(site_dir))
                if todo:
                    diff_file = staging.parent / f"{staging.name}.req.txt"
                    diff_file.write_text("\n".join(options) + "\n" + "".join(r.block for r in todo),
                                         encoding="utf-8")
                    try:
                        _run([*_pip(), "--require-hashes", "--extra-index-url", TORCH_CPU_INDEX,
                              "--target", str(staging), *pip_extra, "-r", str(diff_file)])
                    finally:
                        diff_file.unlink(missing_ok=True)
                # the brain wheel is installed without a hash: its integrity is the
                # caller's — built locally by scripts/update.sh, or sha256-checked
                # against the signed manifest by the app in slice 4.
                _run([*_pip(), "--target", str(staging), *pip_extra, str(wheel)])
                _smoke(staging, smoke_modules)
                stamp = {**entry, "installed_at": iso_utc(datetime.now(UTC))}
                (staging / "stamp.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
                # Checked, not assumed: the named case above should already have
                # redirected `target` away from anything current.json names. This is
                # unreachable in the happy path; it's the guard.
                if target.name in named:
                    raise UpdateError(f"refusing to replace {target.name}: current.json names it")
                if target.exists():
                    # `target` is NOT named by current.json — it's leftover garbage
                    # from an earlier crashed/partial install. Safe to remove outright;
                    # if it can't be removed, fail loudly rather than silently
                    # swallowing it (current.json is still untouched).
                    try:
                        shutil.rmtree(target)
                    except OSError as e:
                        raise UpdateError(f"could not remove leftover overlay folder {target}: {e}") from e
                os.replace(staging, target)
            except OSError as exc:
                shutil.rmtree(staging, ignore_errors=True)
                raise UpdateError(f"update failed: {exc}") from exc
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        overlay.activate(root, entry)
        overlay.prune(root)
    return entry
