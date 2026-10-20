"""Ratchet: no new hand-rolled temp-plus-replace in brain/ outside the shared writer (#465 C2).

Every state file is written through ``brain.utils.atomic_write`` (a unique temp per save, a bounded retry of a Windows
PermissionError on the replace, fsync, cleanup). This test fails when ``brain/`` gains a call of one of these forms
that is not on a line carrying an explicit exemption marker with a reason::

    os.replace(a, b)  # atomic-write: ok -- <why this is not a hand-rolled atomic write>

Flagged forms (RUNNER PROPOSAL, see 1-spec.md section 4): ``os.replace/rename/renames`` (and ``from os import`` of them),
a method call ``X.replace(arg)`` / ``X.rename(arg)`` with exactly one positional argument and no keywords (the ``Path``
forms), ``tempfile.mkstemp``, ``shutil.move``, and the retry helpers ``_replace_with_retry`` / ``replace_retrying`` /
``_retry_perm`` (so a helper cannot be a way round the check). A bare ``replace(obj, a=1)`` that came from
``dataclasses`` is not a file replace and is not flagged: from-imports are resolved by their origin module.

The marker must sit on one of the source lines of the call itself. A marker on lines with no flagged call is a stale
marker and fails, as does a marker with no reason. The exact number of markers in ``brain/`` is pinned, so a drive-by
marker is a visible diff. The module that holds the shared writer, ``brain/utils/atomic_write.py``, is the one file
skipped. Markers on pure-move sites (a plain rename of a finished file) are main's ruling, not the owner's original
wording (the owner named the exclusive-create cases).

Not seen (known limits): a ``replace`` fetched with ``getattr``, ``functools.partial(os.replace, ...)``, an aliased
``import os as _os``, ``os.link`` plus unlink, and a writer that writes straight to the final name.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHARED_WRITER = "brain/utils/atomic_write.py"
MARKER = re.compile(r"#\s*atomic-write:\s*ok\b(?:\s*--\s*(\S.*?)?)?\s*$")
RETRY_HELPERS = {"_replace_with_retry", "replace_retrying", "_retry_perm"}
OS_FUNCS = {"replace", "rename", "renames"}

# The exact number of markers in brain/ (1-spec.md section 3): 15 from the base scan, the staging publish, the two
# shutil.move directory moves, and the three heal moves in attempt_heal.py.
EXPECTED_MARKERS = 21


def flagged_calls(source: str) -> list[tuple[int, int, str]]:
    """(first line, last line, kind) of each flagged call in ``source``."""
    tree = ast.parse(source)
    bound: dict[
        str, str
    ] = {}  # local name -> flagged kind, for from-imports of os / tempfile / shutil
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                local = a.asname or a.name
                if node.module == "os" and a.name in OS_FUNCS:
                    bound[local] = f"os.{a.name}"
                elif node.module == "tempfile" and a.name == "mkstemp":
                    bound[local] = "tempfile.mkstemp"
                elif node.module == "shutil" and a.name == "move":
                    bound[local] = "shutil.move"
    out: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        kind: str | None = None
        if isinstance(f, ast.Name):
            if f.id in bound:
                kind = bound[f.id]
            elif f.id in RETRY_HELPERS:
                kind = f"retry-helper:{f.id}"
        elif isinstance(f, ast.Attribute):
            base = f.value.id if isinstance(f.value, ast.Name) else None
            if base == "os" and f.attr in OS_FUNCS:
                kind = f"os.{f.attr}"
            elif base == "tempfile" and f.attr == "mkstemp":
                kind = "tempfile.mkstemp"
            elif base == "shutil" and f.attr == "move":
                kind = "shutil.move"
            elif f.attr in RETRY_HELPERS:
                kind = f"retry-helper:{f.attr}"
            elif (
                f.attr in ("replace", "rename")
                and len(node.args) == 1
                and not node.keywords
                and base != "dataclasses"
            ):
                kind = f"path.{f.attr}"
        if kind:
            out.append((node.lineno, node.end_lineno or node.lineno, kind))
    return out


def check_source(source: str, rel: str) -> tuple[list[str], int]:
    """(errors, number of valid markers) for one file."""
    errors: list[str] = []
    lines = source.splitlines()
    marker_lines: dict[int, str | None] = {}
    for i, text in enumerate(lines, start=1):
        m = MARKER.search(text)
        if m:
            marker_lines[i] = m.group(1)
    for ln, reason in marker_lines.items():
        if not reason:
            errors.append(f"{rel}:{ln}: atomic-write marker without a reason")
    calls = flagged_calls(source)
    covered: set[int] = set()
    for first, last, kind in calls:
        own = [ln for ln in marker_lines if first <= ln <= last]
        if own and marker_lines[own[0]]:
            covered.update(own)
        else:
            errors.append(
                f"{rel}:{first}: {kind} outside the shared writer needs an atomic-write marker"
            )
    for ln, reason in marker_lines.items():
        if reason and ln not in covered:
            errors.append(f"{rel}:{ln}: stale atomic-write marker (no flagged call on its lines)")
    return errors, len(covered)


def scan_tree(root: Path = REPO) -> tuple[list[str], int]:
    errors: list[str] = []
    markers = 0
    for path in sorted((root / "brain").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel == SHARED_WRITER:
            continue
        errs, n = check_source(path.read_text(encoding="utf-8"), rel)
        errors += errs
        markers += n
    return errors, markers


def test_brain_has_no_unmarked_hand_rolled_replace_and_the_marker_count_is_pinned() -> None:
    errors, markers = scan_tree()
    assert errors == []
    assert markers == EXPECTED_MARKERS, f"{markers} markers; the pin is {EXPECTED_MARKERS}"


def test_the_two_owner_named_exclusive_create_cases_carry_their_why() -> None:
    for rel in ("brain/memory/cost_series.py", "brain/memory/embeddings.py"):
        text = (REPO / rel).read_text(encoding="utf-8")
        reasons = [m.group(1) for line in text.splitlines() if (m := MARKER.search(line))]
        assert reasons and all(r for r in reasons), rel


# --- the oracle can fail: a table of synthetic snippets (ST1.5f) -------------------------------------------------

FLAGGED = {
    "os.replace": "import os\nos.replace(a, b)\n",
    "os.rename": "import os\nos.rename(a, b)\n",
    "os.renames": "import os\nos.renames(a, b)\n",
    "from os import replace": "from os import replace\nreplace(a, b)\n",
    "from os import rename as mv": "from os import rename as mv\nmv(a, b)\n",
    "path.replace": "def f(tmp, dst):\n    tmp.replace(dst)\n",
    "path.rename": "def f(tmp, dst):\n    tmp.rename(dst)\n",
    "mkstemp": "import tempfile\nfd, name = tempfile.mkstemp()\n",
    "from tempfile import mkstemp": "from tempfile import mkstemp\nmkstemp()\n",
    "shutil.move": "import shutil\nshutil.move(a, b)\n",
    "retry helper by name": "_replace_with_retry(a, b)\n",
    "retry helper by attribute": "heal._retry_perm(lambda: 1)\n",
    "replace_retrying": "overlay.replace_retrying(a, b)\n",
}
LOOK_ALIKES = {
    "str.replace": "x = s.replace('a', 'b')\n",
    "datetime.replace": "x = dt.replace(tzinfo=None)\n",
    "dataclasses.replace attr": "import dataclasses\ndataclasses.replace(o)\n",
    "from dataclasses import replace (module level)": "from dataclasses import replace\ny = replace(o, a=1)\n",
    "from dataclasses import replace (function local)": "def f(o):\n    from dataclasses import replace\n    return replace(o, a=1)\n",
    "dc_replace alias": "def f(o):\n    from dataclasses import replace as dc_replace\n    return dc_replace(o)\n",
    "os.path.join": "import os\nos.path.join(a, b)\n",
    "os.link": "import os\nos.link(a, b)\n",
}


def test_every_flagged_form_is_flagged() -> None:
    for name, src in FLAGGED.items():
        assert flagged_calls(src), name


def test_look_alikes_are_not_flagged() -> None:
    for name, src in LOOK_ALIKES.items():
        assert flagged_calls(src) == [], name


def test_a_marker_with_a_reason_on_the_calls_own_lines_exempts_it() -> None:
    ok = "import os\nos.replace(a, b)  # atomic-write: ok -- moving a finished file, not writing one\n"
    assert check_source(ok, "x.py") == ([], 1)
    multi = "import os\nos.replace(\n    a,  # atomic-write: ok -- the marker may sit on any line of the call\n    b,\n)\n"
    assert check_source(multi, "x.py") == ([], 1)


def test_a_marker_without_a_reason_fails() -> None:
    errors, _ = check_source("import os\nos.replace(a, b)  # atomic-write: ok\n", "x.py")
    assert any("without a reason" in e for e in errors)
    errors, _ = check_source("import os\nos.replace(a, b)  # atomic-write: ok --\n", "x.py")
    assert any("without a reason" in e for e in errors)


def test_a_marker_on_lines_with_no_flagged_call_is_stale_and_fails() -> None:
    errors, _ = check_source("x = 1  # atomic-write: ok -- nothing here is flagged\n", "x.py")
    assert any("stale" in e for e in errors)
    # a neighbouring call's marker cannot exempt a different call
    src = "import os\nos.replace(a, b)  # atomic-write: ok -- first\n\nos.rename(c, d)\n"
    errors, _ = check_source(src, "x.py")
    assert len(errors) == 1 and "os.rename" in errors[0]


def test_an_unmarked_flagged_call_fails() -> None:
    errors, _ = check_source("import os\nos.replace(a, b)\n", "x.py")
    assert errors and "needs an atomic-write marker" in errors[0]
