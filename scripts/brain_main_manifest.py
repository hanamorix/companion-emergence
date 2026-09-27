#!/usr/bin/env python3
"""Build and gate the brain-main channel's manifest (#286 slice 3, spec §3.1).

`write`: the manifest the app verifies (signature → manifest → file hashes)
before it downloads anything.

Raising MIN_BUNDLE_VERSION is how a future non-additive persisted-state change
stops older apps from taking `main` builds (spec §7).

Stdlib only at module level: CI's signing job runs this with a bare python3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = 1
PYTHON = "3.13"  # the bundled runtime's major.minor (build_python_runtime.sh PY_VERSION; drift test)
# The first release carrying slice 1 (tolerant readers) and slice 2 (the overlay hook).
# Never lower it: an older bundle can't safely read what a main brain writes, and has no hook.
MIN_BUNDLE_VERSION = "0.0.43"

_COMMIT = re.compile(r"[0-9a-f]{40}")
_WHEEL = re.compile(r"companion_emergence-([^-]+)-py3-none-any\.whl")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(*, wheel: Path, requirements: Path, commit: str, built_at: datetime) -> dict:
    if not _COMMIT.fullmatch(commit):
        raise ValueError(f"commit must be a 40-hex sha, got {commit!r}")
    m = _WHEEL.fullmatch(wheel.name)
    if not m:
        raise ValueError(f"not a companion_emergence wheel: {wheel.name}")
    from brain.update.install import parse_requirements  # the parser `nell update` runs

    _, reqs = parse_requirements(requirements.read_text(encoding="utf-8"))
    if not reqs:
        raise ValueError(f"{requirements.name} has no pinned requirements")
    unhashed = [r.name for r in reqs if "--hash=sha256:" not in r.block]
    if unhashed:
        raise ValueError(f"requirements without hashes: {', '.join(unhashed)}")
    return {
        "schema": SCHEMA,
        "commit": commit,
        "brain_version": m.group(1),
        "built_at": built_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": PYTHON,
        "min_bundle_version": MIN_BUNDLE_VERSION,
        "wheel": {"name": wheel.name, "sha256": _sha256(wheel)},
        "requirements": {"name": requirements.name, "sha256": _sha256(requirements)},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="brain-main manifest (#286 slice 3)")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="write manifest.json for a built wheel + exported lock")
    w.add_argument("--wheel", type=Path, required=True)
    w.add_argument("--requirements", type=Path, required=True)
    w.add_argument("--commit", required=True)
    w.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    try:
        manifest = build_manifest(wheel=args.wheel, requirements=args.requirements,
                                  commit=args.commit, built_at=datetime.now(UTC))
    except (ValueError, RuntimeError, OSError) as exc:  # RuntimeError: UpdateError from the parser
        print(f"brain_main_manifest: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(manifest, indent=2) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
