#!/usr/bin/env python3
"""organ_wireback_audit.py — the two mechanizable wire-back greps, codified.

The brain's characteristic failure mode is the SILENT half-wired organ: a writer
whose reader is dead-on-arrival, or a memory minted without emotion so it can
never feed the emotional loops. docs/maturity-manifest.md says to re-run two
cheap greps each minor release to seed the manifest refresh. This makes them one
runnable, deterministic (AST-based) check.

  1. emotions={} detector — Memory.create_new(...) calls with NO emotions= kwarg
     (emotion is how a memory feeds body/dream/felt-time/salience; a memory
     minted without it is inert to those loops). Caught the W7 ingest gap.
  2. write-only detector — a *.jsonl basename written somewhere under brain/ but
     read nowhere under brain/ (excluding tests). Caught draft_space W1 and the
     reflex-crystallizer W2. Heuristic (string-literal filenames), so it is
     ADVISORY: a hit is a prompt to look, not proof of a dead reader.

Usage:
  python scripts/organ_wireback_audit.py            # report (exit 0 always)
  python scripts/organ_wireback_audit.py --strict   # exit 1 if any finding
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAIN = ROOT / "brain"


def _brain_py_files() -> list[Path]:
    return [p for p in BRAIN.rglob("*.py") if "__pycache__" not in p.parts]


# --- 1. emotions={} detector (AST) -----------------------------------------
def find_emotionless_memory_creates() -> list[tuple[Path, int]]:
    hits: list[tuple[Path, int]] = []
    for path in _brain_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Match `Memory.create_new(...)` (attribute) or a bare `create_new(...)`.
            is_create_new = (
                isinstance(func, ast.Attribute) and func.attr == "create_new"
            ) or (isinstance(func, ast.Name) and func.id == "create_new")
            if not is_create_new:
                continue
            # Attribute form must be on `Memory` to avoid unrelated create_new.
            if isinstance(func, ast.Attribute) and not (
                isinstance(func.value, ast.Name) and func.value.id == "Memory"
            ):
                continue
            kwnames = {kw.arg for kw in node.keywords if kw.arg}
            if "emotions" not in kwnames:
                hits.append((path, node.lineno))
    return hits


# --- 2. write-only detector (jsonl filename heuristic) ----------------------
_JSONL_RE = re.compile(r'["\']([A-Za-z0-9_.\-]+\.jsonl)["\']')


_WRITE_HINTS = ('"a"', "'a'", '"w"', "'w'", "append", "write_text", "dump", ".write(")
_READ_HINTS = ("read", "load", "iter", "scan", "tail", "stream", '"r"', "'r'")


def find_write_only_jsonl() -> list[str]:
    """jsonl basenames that appear in brain/ but only ever next to a write
    operation (never a read). Coarse substring heuristic → ADVISORY: a hit means
    'look here', not 'proven dead reader'."""
    seen: dict[str, dict[str, bool]] = {}
    for path in _brain_py_files():
        for line in path.read_text(encoding="utf-8").splitlines():
            names = _JSONL_RE.findall(line)
            if not names:
                continue
            low = line.lower()
            wrote = any(h in low for h in _WRITE_HINTS)
            read = any(h in low for h in _READ_HINTS)
            for name in names:
                s = seen.setdefault(name, {"write": False, "read": False})
                s["write"] = s["write"] or wrote
                s["read"] = s["read"] or read
    return sorted(n for n, s in seen.items() if s["write"] and not s["read"])


def main() -> int:
    strict = "--strict" in sys.argv
    emo = find_emotionless_memory_creates()
    woj = find_write_only_jsonl()

    print("== organ wire-back audit ==\n")
    print(f"1. Memory.create_new without emotions= : {len(emo)} site(s)")
    for path, lineno in emo:
        print(f"     {path.relative_to(ROOT)}:{lineno}")
    print(f"\n2. jsonl written-but-not-read (advisory) : {len(woj)} file(s)")
    for name in woj:
        print(f"     {name}")
    print(
        "\nBoth are seeds for the docs/maturity-manifest.md refresh — review each "
        "hit: is the memory intentionally inert, is the writer's reader really "
        "dead, or is a new organ silently half-wired?"
    )
    if strict and (emo or woj):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
