"""Guard for the rollback invariant (#286 §7, brain/state_compat.py): a dataclass
built from a persisted record must go through from_known_fields, never
`Cls(**record)`, which rejects fields a newer brain added. Scans brain/ with ast
so multi-line calls are caught.

Limits: this is a name heuristic, not a type check. A record passed under a
name in `_KWARG_NAMES` (e.g. `BridgeState(**kwargs)`) is not caught. The real
protection is the per-reader canaries in
tests/unit/test_persisted_state_tolerance.py — a new persisted-state reader
needs its own canary there, not just a clean run of this guard."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# `**` of these names is keyword-argument plumbing, not a persisted record.
_KWARG_NAMES = {"kwargs", "kw", "opts", "options", "params", "overrides", "defaults",
                "extra", "engine_kwargs"}
# Reviewed exceptions (file, callee): not persisted records, or already filtered.
# NOTE: an entry exempts that callee name for the WHOLE file, not one call site —
# a second, unfiltered `cls(**record)` elsewhere in the same file would also pass.
_ALLOWED = {
    ("brain/memory/judge_full_ft.py", "CrossEncoder"),  # model keyword options
    ("brain/pronouns.py", "PronounSet"),  # filters to known fields itself
    ("brain/state_compat.py", "cls"),  # the helper itself — filters to declared fields first
}


def _strict_constructions() -> list[str]:
    found = []
    for path in sorted((REPO / "brain").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = ast.unparse(node.func).split(".")[-1]
            if not (callee[:1].isupper() or callee == "cls") or (rel, callee) in _ALLOWED:
                continue
            for kw in node.keywords:
                if kw.arg is None and ast.unparse(kw.value) not in _KWARG_NAMES:
                    found.append(f"{rel}:{node.lineno}: {ast.unparse(node.func)}(**{ast.unparse(kw.value)})")
    return found


def test_no_dataclass_is_built_from_a_record_with_double_star():
    found = _strict_constructions()
    assert not found, (
        "Build persisted-state dataclasses with brain.state_compat.from_known_fields — "
        "`Cls(**record)` breaks the rollback invariant. If a hit is keyword plumbing, "
        "not a record, add it to _ALLOWED with a reason:\n" + "\n".join(found)
    )
