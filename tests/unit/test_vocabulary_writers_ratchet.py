"""Ratchet: only the shared module writes her vocabulary file (#741, #801 section 4, criterion 3).

`brain/emotion/vocabulary_file.py` is the one writer of `emotion_vocabulary.json` (json_io's own corrupt-file recovery adds no
name and mentions no vocabulary path). This test scans every call in `brain/` that writes a file (`write_json`,
`write_text`, `os.replace`, `shutil.copy`, `open(..., "w")` and the like) and flags one whose arguments, keywords or receiver
mention `vocab` (a `vocab_path`, a `vocabulary_path(...)`, `emotion_vocabulary.json`). A hit outside the
shared module is a failure unless the file is in `KNOWN_EXCEPTIONS`: the writer that #801 moves in a later step. Each exception
must still hold a hit, so the list cannot go stale: when step 8 moves the migrator, the test fails until its line is removed here.

The scan is by name: a write whose arguments never mention `vocab` (a bare `path` parameter) is not seen. The spy on `write_json` for
the rebuild step and the migrator (spec criterion 10) is the complement, in step 8.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping

import pytest

from tests.source_tree import tree_asts

SHARED_MODULE = "brain/emotion/vocabulary_file.py"

# file -> why it may still write her vocabulary file (removed by the step named).
KNOWN_EXCEPTIONS: dict[str, str] = {
    "brain/migrator/cli.py": "the OG migrator writes the first vocabulary file itself; moves onto the shared function in step 8 (spec section 9)",
}

# Calls that write a file or put one in place.
_WRITE_NAMES = frozenset(
    {"write_json", "write_jsonl", "save_bytes_atomic", "write_text", "write_bytes", "replace", "rename", "copy", "copy2", "copyfile", "move"}
)
_WRITE_MODES = ("w", "a", "x")


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _is_write_open(call: ast.Call) -> bool:
    if _call_name(call) != "open":
        return False
    modes = [a.value for a in call.args[1:2] if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    modes += [k.value.value for k in call.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str)]
    return any(m[:1] in _WRITE_MODES for m in modes)


def vocabulary_write_hits(trees: Mapping[str, ast.Module]) -> dict[str, list[int]]:
    """{file: [line, ...]} of write calls whose arguments, keywords or receiver mention `vocab` (case-insensitive)."""
    hits: dict[str, list[int]] = {}
    for rel, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not (_call_name(node) in _WRITE_NAMES or _is_write_open(node)):
                continue
            parts = [ast.unparse(a) for a in node.args] + [ast.unparse(k.value) for k in node.keywords] + [ast.unparse(node.func)]
            if any("vocab" in p.lower() for p in parts):
                hits.setdefault(rel, []).append(node.lineno)
    return hits


@pytest.mark.no_base_failure(reason="a ratchet over the whole tree: on the old code the writers it flags are all moved or listed, so it is a guard against new writers, not a fix")
def test_only_the_shared_module_writes_her_vocabulary_file() -> None:
    hits = vocabulary_write_hits(tree_asts("brain"))
    unexpected = {rel: lines for rel, lines in hits.items() if rel != SHARED_MODULE and rel not in KNOWN_EXCEPTIONS}
    assert not unexpected, (
        "Code other than brain/emotion/vocabulary_file.py writes her vocabulary file: "
        + "; ".join(f"{rel}:{lines}" for rel, lines in sorted(unexpected.items()))
        + ". Use vocabulary_file.add / replace_fields / merge / remove."
    )
    stale = sorted(rel for rel in KNOWN_EXCEPTIONS if rel not in hits)
    assert not stale, f"these known exceptions no longer write her vocabulary file, remove them from KNOWN_EXCEPTIONS: {stale}"


@pytest.mark.no_base_failure(reason="a self-test of the scanner function on synthetic trees; it does not read the product tree, so it passes on any base that has this file")
def test_the_scanner_finds_a_synthetic_offender_and_ignores_a_state_file_write() -> None:
    offender = ast.parse("from x import write_json\ndef f(vocab_path, data):\n    write_json(vocab_path, data, backups=3)\n")
    keyword_only = ast.parse("def g(path, data):\n    write_json(path, data, backups=_VOCABULARY_BACKUPS)\n")
    receiver = ast.parse("def h(p):\n    vocabulary_path(p).write_text('x')\n")
    opened = ast.parse("def o(vocab_path):\n    open(vocab_path, 'w')\n")
    unrelated = ast.parse("def s(path, state):\n    write_json(path, state, indent=2)\n")
    reads = ast.parse("def r(vocab_path):\n    return open(vocab_path).read()\n")
    trees = {"a.py": offender, "b.py": keyword_only, "c.py": receiver, "d.py": opened, "e.py": unrelated, "f.py": reads}
    assert set(vocabulary_write_hits(trees)) == {"a.py", "b.py", "c.py", "d.py"}
