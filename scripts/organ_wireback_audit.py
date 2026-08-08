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
     reflex-crystallizer W2. AST + scope-aware: the filename literal and the
     operation are normally on different lines, and an earlier same-line
     heuristic therefore saw only 5 of 32 basenames — a 46 MB write-only
     gate_rejections.jsonl was invisible for months. Still ADVISORY: a hit is a
     prompt to look, not proof of a dead reader. Anything unclassifiable is
     REPORTED (3), never silently dropped — that omission was the real defect.

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


# --- 2. write-only detector (AST, scope-aware) ------------------------------
_JSONL_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.jsonl$")

_WRITE_ATTRS = {"write", "writelines", "write_text", "writestr"}
_READ_ATTRS = {"read", "readline", "readlines", "read_text", "iterdir"}
_WRITE_MODES = {"a", "w", "a+", "w+", "ab", "wb", "at", "wt"}
_READ_MODES = {"r", "r+", "rb", "rt"}
_WRITE_NAMES = ("append", "dump", "write", "log_", "_log", "emit", "record")
_READ_NAMES = ("read", "load", "iter", "scan", "tail", "stream", "parse", "replay")


def _jsonl_literals(node: ast.AST) -> set[str]:
    """Every '*.jsonl' string constant appearing anywhere under `node`."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and _JSONL_RE.match(n.value):
            out.add(n.value)
    return out


def _classify_scope(node: ast.AST) -> tuple[bool, bool]:
    """(writes, reads) — does this scope contain a write / read operation?

    Scope-level rather than line-level on purpose: the filename literal and the
    operation are normally on different lines, and requiring them to share one
    is what made 27 of 32 basenames invisible.
    """
    writes = reads = False
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            fn = n.func
            attr = fn.attr if isinstance(fn, ast.Attribute) else None
            name = attr or (fn.id if isinstance(fn, ast.Name) else "")
            low = (name or "").lower()
            if attr in _WRITE_ATTRS or any(h in low for h in _WRITE_NAMES):
                writes = True
            if attr in _READ_ATTRS or any(h in low for h in _READ_NAMES):
                reads = True
            if attr == "open":
                mode = ""
                for a in n.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        mode = a.value
                        break
                for kw in n.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = str(kw.value.value)
                if mode in _WRITE_MODES:
                    writes = True
                elif mode in _READ_MODES or not mode:
                    reads = True
    return writes, reads


def find_write_only_jsonl(root: Path | None = None) -> list[str]:
    """jsonl basenames written under `root` but never read there.

    ADVISORY — a hit means 'look here', not 'proven dead reader'.

    Resolves module-level constants (LOG = "x.jsonl") so a scope referencing the
    constant counts as touching the file. Anything it cannot classify is NOT
    dropped — see find_unclassified_jsonl. Silent omission was the original
    defect: a 46 MB write-only log went unreported for months because neither
    flag was ever set on it.
    """
    return sorted(n for n, s in _scan_jsonl(root).items() if s["write"] and not s["read"])


def find_unclassified_jsonl(root: Path | None = None) -> list[str]:
    """jsonl basenames the heuristic could not classify either way.

    Reported rather than dropped: 'no evidence' is a prompt to look by hand, and
    silently discarding it is how the class hid in the first place.
    """
    return sorted(n for n, s in _scan_jsonl(root).items() if not s["write"] and not s["read"])


def _scan_jsonl(root: Path | None = None) -> dict[str, dict[str, bool]]:
    files = (
        [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]
        if root is not None
        else _brain_py_files()
    )
    seen: dict[str, dict[str, bool]] = {}

    def bump(name: str, writes: bool, reads: bool) -> None:
        s = seen.setdefault(name, {"write": False, "read": False})
        s["write"] = s["write"] or writes
        s["read"] = s["read"] or reads

    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        # Module-level constants: LOG = "thing.jsonl" -> {"LOG": "thing.jsonl"}
        const_names: dict[str, str] = {}
        for n in tree.body:
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant):
                v = n.value.value
                if isinstance(v, str) and _JSONL_RE.match(v):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            const_names[t.id] = v

        # Function bodies are the scopes. The module is NOT one: ast.walk on the
        # module descends into every function, so treating it as a scope makes a
        # file's names inherit every operation anywhere in that file — which
        # marked a genuinely write-only log as read and hid it again.
        scopes: list[ast.AST] = [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for scope in scopes:
            names = _jsonl_literals(scope)
            for ref in ast.walk(scope):
                if isinstance(ref, ast.Name) and ref.id in const_names:
                    names.add(const_names[ref.id])
            if not names:
                continue
            writes, reads = _classify_scope(scope)
            for name in names:
                bump(name, writes, reads)

        # A constant declared but never referenced still deserves reporting.
        for v in const_names.values():
            bump(v, False, False)

    return seen


def main() -> int:
    strict = "--strict" in sys.argv
    emo = find_emotionless_memory_creates()
    woj = find_write_only_jsonl()
    unc = find_unclassified_jsonl()

    print("== organ wire-back audit ==\n")
    print(f"1. Memory.create_new without emotions= : {len(emo)} site(s)")
    for path, lineno in emo:
        print(f"     {path.relative_to(ROOT)}:{lineno}")
    print(f"\n2. jsonl written-but-not-read (advisory) : {len(woj)} file(s)")
    for name in woj:
        print(f"     {name}")
    print(f"\n3. jsonl the heuristic could not classify : {len(unc)} file(s)")
    for name in unc:
        print(f"     {name}")
    if unc:
        print(
            "     (reported, not dropped — 'no evidence' still needs a human look. "
            "Silent omission is why a 46 MB write-only log went unreported.)"
        )
    print(
        "\nAll three are seeds for the docs/maturity-manifest.md refresh — review "
        "each hit: is the memory intentionally inert, is the writer's reader really "
        "dead, or is a new organ silently half-wired?"
    )
    if strict and (emo or woj):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
