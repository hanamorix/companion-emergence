"""Ratchet: no new direct JSON file I/O in brain/ outside ``brain.health.json_io`` (#438).

The app is moving every JSON and JSONL file read and write onto ``brain.health.json_io``. This test fails when
``brain/`` gains a direct file read or write of JSON, or a new caller of a legacy helper, that is not in
``tests/unit/json_io_allowlist.txt``. The allowlist is today's unmigrated sites; each migration PR deletes the
sites it moves in the same diff.

Allowlist lines: ``path::kind::count   # tag``. ``kind`` is ``read``, ``write``, ``not-file-io`` (a json call that is not
file I/O: a model reply, an HTTP body, a sqlite column, a prompt string) or ``legacy:<helper>`` (a call of
``attempt_heal`` and the other legacy helpers json_io replaces). The count is a ceiling that must also be met exactly
unless the entry is tagged ``incoming: #N`` (a pending PR changes that file: slack either way is allowed, and a
``warnings.warn`` says to lower the ceiling and drop the tag once the PR has landed). An untagged entry above the
observed count FAILS with a message to lower it, so freed slack can never be reused silently (N-3). ``permanent: why``
marks a site that never migrates (it is replaced by an in-site ``# json-io: ok -- reason`` marker in the last
migration PR; the central file exists only because this phase could not edit call sites).

Reconciliation: every ``json.load/loads/dump/dumps`` call is put in exactly one category (read, write, not-file-io,
or ``indirect`` for a json function used as a value or fetched by ``__import__``), and each category has a ceiling, so a
new call cannot be unaccounted for. This is a ceiling check, not an oracle independent of the scanner.

Known limits (blind spots): names are followed within one function only; ``json.loads(self._slurp(p))`` (a one-hop
helper) is not followed; readers that parse through a domain constructor with no ``json`` call in the file
(``AuditRow.from_jsonl(line)``) are invisible and are listed in the call-site inventory instead; ``getattr(json,
"loads")``, ``from json import *``, ``json.JSONDecoder().decode(...)`` and other JSON libraries are not seen. Import
aliases are tracked per scope (a function-local ``import json as _json`` counts); a json function passed around as a
value (``map(json.loads, lines)``, ``loads = json.loads``) or fetched with ``__import__("json")`` /
``importlib.import_module("json")`` is counted as ``indirect``. Run ``python tests/unit/test_no_direct_json_io.py
--print`` to regenerate the observed counts in allowlist format.
"""

from __future__ import annotations

import ast
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ALLOWLIST = Path(__file__).with_name("json_io_allowlist.txt")

JSON_OPS = {"load", "loads", "dump", "dumps"}
READ_ATTRS = {"read_text", "read_bytes", "read", "readline", "readlines"}
WRITE_ATTRS = {"write_text", "write", "write_bytes", "writelines"}
WRITE_FUNCS = {"save_with_backup_text"}  # a legacy helper that writes the text it is given
OPEN_NAMES = {"open", "fdopen"}
PASS_THROUGH = {"enumerate", "reversed", "iter", "list", "tuple", "sorted"}
LEGACY = {
    "attempt_heal",
    "save_with_backup",
    "read_jsonl_skipping_corrupt",
    "iter_jsonl_skipping_corrupt",
    "iter_jsonl_streaming",
    "read_last_n_jsonl_lines",
}
# The module itself and the legacy helpers' own files are not call sites of what they define.
SKIP_FILES = {"brain/health/json_io.py"}
SKIP_LEGACY_IN = {
    "brain/health/attempt_heal.py",
    "brain/health/jsonl_reader.py",
    "brain/health/json_io.py",
}
MARKER = re.compile(r"#\s*json-io:\s*ok\s*--\s*(\S.*)$")
NOT_FILE_CLASSES = (
    "model-reply",
    "prompt-or-log",
    "sqlite-or-wire",
    "string-codec",
    "subprocess",
)
TAG_OK = re.compile(
    r"^(G[1-9][\w,/\- ]*|permanent:\s*\S|held:\s*\S|incoming:\s*#\d+|legacy|not-file:\s*("
    + "|".join(NOT_FILE_CLASSES)
    + "))"
)


# ---------------------------------------------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------------------------------------------


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _has_read_call(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in READ_ATTRS
        for n in ast.walk(node)
    )


def _roots(node: ast.AST) -> set[str]:
    """The names a value is *derived from by method chain*: ``text.splitlines()`` -> {text}; ``f(x)`` -> {f}.

    An argument of a call does not make the result derived from it (``provider.complete(prompt)`` is not a file read
    just because ``prompt`` embeds a file's text).
    """
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _roots(node.value)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in PASS_THROUGH and node.args:
            return _roots(
                node.args[0]
            )  # enumerate(fh), reversed(lines), list(fh) iterate their argument
        return _roots(node.func)
    return set()


def _derived(node: ast.AST, file_vars: set[str]) -> bool:
    return _has_read_call(node) or _is_open_call(node) or bool(_roots(node) & file_vars)


def _is_open_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id in OPEN_NAMES) or (
        isinstance(f, ast.Attribute) and f.attr in OPEN_NAMES
    )


class _FileScan(ast.NodeVisitor):
    def __init__(self, lines: list[str], count_legacy: bool) -> None:
        self.lines = lines
        self.count_legacy = count_legacy
        self.counts: Counter[str] = Counter()
        self.module_aliases: list[set[str]] = [set()]  # per-scope names bound to the json module
        self.func_aliases: list[dict[str, str]] = [
            {}
        ]  # per-scope name -> op for `from json import loads as x`
        self.fn_stack: list[ast.AST] = []
        self.bad_markers: list[int] = []
        self._call_funcs: set[int] = set()

    # -- import aliases are tracked per scope -------------------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.name == "json":
                self.module_aliases[-1].add(a.asname or "json")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "json":
            for a in node.names:
                if a.name in JSON_OPS:
                    self.func_aliases[-1][a.asname or a.name] = a.name

    def _scoped(self, node: ast.AST) -> None:
        self.module_aliases.append(set())
        self.func_aliases.append({})
        self.fn_stack.append(node)
        saved = (getattr(self, "_flow", None), getattr(self, "_fn_node", None))
        self._analyse_function(node)
        self.generic_visit(node)
        self._flow, self._fn_node = (
            saved  # a nested function must not replace the enclosing function's flow data
        )
        self.fn_stack.pop()
        self.module_aliases.pop()
        self.func_aliases.pop()

    def visit_FunctionDef(self, node: ast.AST) -> None:  # noqa: N802 - ast.NodeVisitor protocol
        self._scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AST) -> None:  # noqa: N802
        self._scoped(node)

    # -- per-function data flow (names only) -----------------------------------------------------------------
    def _analyse_function(self, fn: ast.AST) -> None:
        file_vars: set[str] = set()
        assigned: dict[str, list[ast.AST]] = defaultdict(list)
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    for name in _names(t):
                        assigned[name].append(n.value)
            elif isinstance(n, ast.With):
                for item in n.items:
                    if item.optional_vars is not None and (
                        _is_open_call(item.context_expr) or _has_read_call(item.context_expr)
                    ):
                        file_vars |= _names(item.optional_vars)
        changed = True
        while changed:
            changed = False
            for n in ast.walk(fn):
                targets: list[str] = []
                source: ast.AST | None = None
                if isinstance(n, ast.Assign):
                    targets, source = [x for t in n.targets for x in _names(t)], n.value
                elif isinstance(n, (ast.For, ast.comprehension)):
                    targets, source = list(_names(n.target)), n.iter
                if source is not None and _derived(source, file_vars):
                    for t in targets:
                        if t not in file_vars:
                            file_vars.add(t)
                            changed = True
        write_arg_names: set[str] = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and self._is_write_call(n):
                for a in [*n.args, *[k.value for k in n.keywords]]:
                    write_arg_names |= _names(a)
        self._flow = (file_vars, assigned, write_arg_names)
        self._fn_node = fn

    @staticmethod
    def _is_write_call(call: ast.Call) -> bool:
        f = call.func
        return (isinstance(f, ast.Attribute) and f.attr in WRITE_ATTRS) or (
            isinstance(f, ast.Name) and f.id in WRITE_FUNCS
        )

    # -- calls ------------------------------------------------------------------------------------------------
    def _json_op(self, call: ast.Call) -> str | None:
        f = call.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.attr in JSON_OPS:
            if any(f.value.id in s for s in self.module_aliases):
                return f.attr
        if isinstance(f, ast.Name):
            for scope in reversed(self.func_aliases):
                if f.id in scope:
                    return scope[f.id]
        return None

    def _is_json_value(self, node: ast.AST) -> bool:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.attr in JSON_OPS
        ):
            return any(node.value.id in s for s in self.module_aliases)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            return any(node.id in scope for scope in self.func_aliases)
        return False

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802 - ast.NodeVisitor protocol
        if id(node) not in self._call_funcs and self._is_json_value(node):
            self.counts["indirect"] += (
                1  # json.loads used as a value: map(json.loads, ...), loads = json.loads
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if id(node) not in self._call_funcs and self._is_json_value(node):
            self.counts["indirect"] += 1
        self.generic_visit(node)

    def _dynamic_import_of_json(self, node: ast.Call) -> bool:
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        return (
            name in ("__import__", "import_module")
            and bool(node.args)
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "json"
        )

    def visit_Call(self, node: ast.Call) -> None:
        self._call_funcs.add(id(node.func))
        if self._dynamic_import_of_json(node):
            self.counts["indirect"] += 1
        op = self._json_op(node)
        if op:
            self._classify(node, op)
        if self.count_legacy:
            f = node.func
            name = (
                f.id
                if isinstance(f, ast.Name)
                else f.attr
                if isinstance(f, ast.Attribute)
                else None
            )
            if name in LEGACY:
                self.counts[f"legacy:{name}"] += 1
        self.generic_visit(node)

    def _marked(self, node: ast.Call) -> bool:
        span = {node.lineno, node.end_lineno or node.lineno}
        for ln in span:
            m = MARKER.search(self.lines[ln - 1])
            if m:
                return True
        for ln in span:
            if re.search(r"#\s*json-io:\s*ok\b", self.lines[ln - 1]):
                self.bad_markers.append(ln)  # a marker with no reason does not exempt
        return False

    def _classify(self, node: ast.Call, op: str) -> None:
        if self._marked(node):
            return
        file_vars, assigned, write_arg_names = getattr(self, "_flow", (set(), {}, set()))
        in_function = bool(self.fn_stack)
        kind = "not-file-io"
        if op == "load":
            kind = "read"
        elif op == "dump":
            kind = "write"
        elif op == "loads" and node.args:
            arg = node.args[0]
            if _has_read_call(arg) or (in_function and _roots(arg) & file_vars):
                kind = "read"
        elif op == "dumps":
            if self._dumps_flows_to_write(node, assigned, write_arg_names, in_function):
                kind = "write"
        self.counts[kind] += 1

    def _dumps_flows_to_write(
        self, node: ast.Call, assigned, write_arg_names, in_function: bool
    ) -> bool:
        fn = self._fn_node if in_function else None
        root = fn if fn is not None else None
        if root is None:
            return False
        # directly inside the arguments of a write call
        for n in ast.walk(root):
            if isinstance(n, ast.Call) and self._is_write_call(n):
                for a in [*n.args, *[k.value for k in n.keywords]]:
                    if any(c is node for c in ast.walk(a)):
                        return True
        # assigned to a name that is later passed to a write call
        for name, values in assigned.items():
            if name in write_arg_names and any(any(c is node for c in ast.walk(v)) for v in values):
                return True
        return False


def scan_source(source: str, rel: str) -> Counter[str]:
    tree = ast.parse(source)
    scanner = _FileScan(source.splitlines(), rel not in SKIP_LEGACY_IN)
    scanner.visit(tree)
    return scanner.counts


def scan(root: Path) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = {}
    for path in sorted((root / "brain").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in SKIP_FILES or "__pycache__" in path.parts:
            continue
        counts = scan_source(path.read_text(encoding="utf-8"), rel)
        if counts:
            out[rel] = counts
    return out


# ---------------------------------------------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    count: int
    tag: str
    incoming: str | None
    permanent: bool


def parse_allowlist(text: str) -> dict[tuple[str, str], Entry]:
    out: dict[tuple[str, str], Entry] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        body, _, comment = line.partition("#")
        parts = [p.strip() for p in body.split("::")]
        if len(parts) != 3 or not parts[2].isdigit():
            raise ValueError(
                f"allowlist line {n}: expected `path::kind::count  # tag`, got {raw!r}"
            )
        tag = comment.strip()
        if not TAG_OK.match(tag):
            raise ValueError(
                f"allowlist line {n}: needs a tag (G1-G9, permanent: why, incoming: #N), got {tag!r}"
            )
        key = (parts[0], parts[1])
        if key in out:
            raise ValueError(f"allowlist line {n}: duplicate entry {key}")
        m = re.search(r"incoming:\s*(#\d+)", tag)
        out[key] = Entry(int(parts[2]), tag, m.group(1) if m else None, tag.startswith("permanent"))
    return out


def compare(
    root: Path, observed: dict[str, Counter[str]], allow: dict[tuple[str, str], Entry]
) -> tuple[list[str], list[str]]:
    """(errors, warnings) of the observed counts against the allowlist; also the call-node oracle."""
    errors: list[str] = []
    warns: list[str] = []
    for (rel, kind), entry in sorted(allow.items()):
        obs = observed.get(rel, Counter()).get(kind, 0)
        exists = (root / rel).exists()
        if obs > entry.count:
            errors.append(
                f"{rel}: {obs} `{kind}` site(s), allowlist says {entry.count}: route it through brain.health.json_io"
            )
        elif obs < entry.count:
            if entry.incoming:
                if exists:
                    warns.append(
                        f"{rel}::{kind}: {obs} < ceiling {entry.count}; if {entry.incoming} has merged, lower the "
                        f"ceiling to {obs} and drop the tag"
                    )
            elif not exists or obs == 0:
                errors.append(f"{rel}::{kind}: no longer present; remove this allowlist entry")
            else:
                errors.append(
                    f"{rel}::{kind}: observed {obs}, ceiling {entry.count}: lower the ceiling to {obs}"
                )
    for rel, counts in observed.items():
        for kind, n in counts.items():
            if n > 0 and (rel, kind) not in allow:
                errors.append(
                    f"{rel}: {n} `{kind}` site(s) not in the allowlist: use brain.health.json_io (or `# json-io: ok -- reason`)"
                )
    return errors, warns


def format_observed(observed: dict[str, Counter[str]]) -> str:
    lines = []
    for rel in sorted(observed):
        for kind in sorted(observed[rel]):
            lines.append(f"{rel}::{kind}::{observed[rel][kind]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------


def test_cs2_a_the_tree_matches_the_allowlist() -> None:
    allow = parse_allowlist(ALLOWLIST.read_text(encoding="utf-8"))
    errors, warns = compare(REPO, scan(REPO), allow)
    _emit(warns)
    assert not errors, "\n".join(errors)


def _kinds(source: str, rel: str = "brain/x.py") -> dict[str, int]:
    return dict(scan_source(source, rel))


@pytest.mark.parametrize(
    "source,expected",
    [
        ("import json\ndef f(fh):\n    return json.load(fh)\n", {"read": 1}),
        (
            "import json\nfrom pathlib import Path\ndef f(p):\n    return json.loads(Path(p).read_text())\n",
            {"read": 1},
        ),
        (
            "import json\ndef f(p):\n    text = p.read_text()\n    return json.loads(text)\n",
            {"read": 1},
        ),
        (
            "import json\ndef f(p):\n    raw = p.read_bytes()\n    return json.loads(raw.decode('utf-8'))\n",
            {"read": 1},
        ),
        (
            "import json\ndef f(p):\n    with open(p) as fh:\n        for line in fh:\n            json.loads(line)\n",
            {"read": 1},
        ),
        (
            "import json\ndef f(p):\n    for n, line in enumerate(p.read_text().splitlines()):\n        json.loads(line)\n",
            {"read": 1},
        ),
        ("import json\ndef f(fh, d):\n    json.dump(d, fh)\n", {"write": 1}),
        ("import json\ndef f(p, d):\n    p.write_text(json.dumps(d))\n", {"write": 1}),
        ("import json\ndef f(fh, d):\n    fh.write(json.dumps(d) + '\\n')\n", {"write": 1}),
        (
            "import json\ndef f(fh, d):\n    line = json.dumps(d)\n    fh.write(line)\n",
            {"write": 1},
        ),
        ("import json as j\ndef f(p):\n    return j.loads(p.read_text())\n", {"read": 1}),
        ("from json import loads as l\ndef f(p):\n    return l(p.read_text())\n", {"read": 1}),
        (
            "def f(p):\n    import json as _json\n    return _json.loads(p.read_text())\n",
            {"read": 1},
        ),
        ("import json\ndef f(reply):\n    return json.loads(reply)\n", {"not-file-io": 1}),
        (
            "import json\ndef f(p, provider):\n    prompt = p.read_text()\n    return json.loads(provider.complete(prompt))\n",
            {"not-file-io": 1},
        ),
        ("import json\ndef f(d):\n    return json.dumps(d)\n", {"not-file-io": 1}),
        (
            "from brain.health.attempt_heal import attempt_heal\ndef f(p):\n    return attempt_heal(p, dict)\n",
            {"legacy:attempt_heal": 1},
        ),
        (
            "from brain.health import jsonl_reader as r\ndef f(p):\n    return r.read_jsonl_skipping_corrupt(p)\n",
            {"legacy:read_jsonl_skipping_corrupt": 1},
        ),
        (
            "from brain.health.attempt_heal import save_with_backup\ndef f(p, d):\n    save_with_backup(p, d)\n",
            {"legacy:save_with_backup": 1},
        ),
        (
            "import json\ndef outer(p, d):\n    def inner(x):\n        return x\n    p.write_text(json.dumps(d))\n",
            {"write": 1},
        ),
        # F8 (code review): the first-class and dynamic-import forms, and open() as a direct iteration source
        (
            "import json\ndef f(p):\n    return list(map(json.loads, p.read_text().splitlines()))\n",
            {"indirect": 1},
        ),
        (
            "import json\nloads = json.loads\ndef f(p):\n    return loads(p.read_text())\n",
            {"indirect": 1},
        ),
        (
            "from json import loads as l\ndef f(p):\n    return list(map(l, p.read_text().splitlines()))\n",
            {"indirect": 1},
        ),
        (
            "def f(p):\n    j = __import__('json')\n    return j.loads(p.read_text())\n",
            {"indirect": 1},
        ),
        (
            "import importlib\ndef f():\n    return importlib.import_module('json')\n",
            {"indirect": 1},
        ),
        ("import json\ndef f(p):\n    for l in open(p):\n        json.loads(l)\n", {"read": 1}),
        ("import json\ndef f(p):\n    return [json.loads(l) for l in open(p)]\n", {"read": 1}),
        ("import json\ndef f(p):\n    return [json.loads(l) for l in p.open()]\n", {"read": 1}),
    ],
)
def test_cs2_b_each_pattern_family_is_classified(source: str, expected: dict[str, int]) -> None:
    assert _kinds(source) == expected


def test_cs2_b_the_documented_blind_spots_are_named_in_the_module_doc() -> None:
    doc = __doc__ or ""
    for spot in ("getattr(json", "from json import *", "JSONDecoder", "one-hop", "from_jsonl"):
        assert spot in doc, spot


def test_cs2_b_the_legacy_helper_files_do_not_count_their_own_definitions() -> None:
    src = "def attempt_heal(p, f):\n    return attempt_heal(p, f)\n"
    assert scan_source(src, "brain/health/attempt_heal.py") == Counter()


def _allow(text: str) -> dict:
    return parse_allowlist(text)


def _compare(tmp_path: Path, files: dict[str, str], allow_text: str):
    for rel, src in files.items():
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(
            src, encoding="utf-8"
        )  # a real brain/ file holds non-ASCII; the platform codec is not UTF-8
    return compare(tmp_path, scan(tmp_path), _allow(allow_text))


_ONE_READ = "import json\ndef f(p):\n    return json.loads(p.read_text())\n"


def test_cs2_c_a_site_above_its_ceiling_fails(tmp_path: Path) -> None:
    two = _ONE_READ + "def g(p):\n    return json.loads(p.read_text())\n"
    errors, _ = _compare(tmp_path, {"brain/x.py": two}, "brain/x.py::read::1   # G3\n")
    assert errors and "2" in errors[0] and "json_io" in errors[0]


def test_cs2_c_a_site_in_no_allowlist_entry_fails(tmp_path: Path) -> None:
    errors, _ = _compare(tmp_path, {"brain/x.py": _ONE_READ}, "")
    assert errors and "not in the allowlist" in errors[0]


def test_cs2_c_a_new_caller_of_a_legacy_helper_fails(tmp_path: Path) -> None:
    src = "from brain.health.attempt_heal import attempt_heal\ndef f(p):\n    return attempt_heal(p, dict)\n"
    errors, _ = _compare(tmp_path, {"brain/x.py": src}, "")
    assert errors and "legacy:attempt_heal" in errors[0]


def test_cs2_d_an_untagged_drop_below_the_ceiling_fails_with_a_message_to_lower_it(
    tmp_path: Path,
) -> None:
    errors, _ = _compare(tmp_path, {"brain/x.py": _ONE_READ}, "brain/x.py::read::3   # G3\n")
    assert len(errors) == 1 and "lower the ceiling to 1" in errors[0]


def test_cs2_d_a_tagged_drop_passes_and_emits_a_warning_that_the_test_run_shows(
    tmp_path: Path,
) -> None:
    errors, warns = _compare(
        tmp_path, {"brain/x.py": _ONE_READ}, "brain/x.py::read::3   # G3; incoming: #429\n"
    )
    assert (
        errors == []
        and len(warns) == 1
        and "#429" in warns[0]
        and "lower the ceiling to 1" in warns[0]
    )
    with pytest.warns(UserWarning, match="lower the ceiling to 1"):
        _emit(warns)


def _emit(warns: list[str]) -> None:
    for w in warns:
        warnings.warn(w, stacklevel=1)


def test_cs2_d_an_incoming_entry_for_a_file_that_does_not_exist_yet_passes_silently(
    tmp_path: Path,
) -> None:
    errors, warns = _compare(tmp_path, {}, "brain/health/new.py::write::1   # G6; incoming: #427\n")
    assert errors == [] and warns == []


def test_cs2_d_an_untagged_entry_for_a_file_that_is_gone_fails(tmp_path: Path) -> None:
    errors, _ = _compare(tmp_path, {}, "brain/gone.py::read::1   # G3\n")
    assert errors and "remove this allowlist entry" in errors[0]


def test_cs2_e_a_marker_with_a_reason_exempts_one_call_and_one_without_does_not() -> None:
    ok = "import json\ndef f(p):\n    return json.loads(p.read_text())  # json-io: ok -- bootstrap, must not import brain\n"
    assert _kinds(ok) == {}
    bare = "import json\ndef f(p):\n    return json.loads(p.read_text())  # json-io: ok\n"
    assert _kinds(bare) == {"read": 1}


def test_cs2_e_allowlist_lines_need_a_tag_and_a_count() -> None:
    with pytest.raises(ValueError):
        parse_allowlist("brain/x.py::read::1\n")
    with pytest.raises(ValueError):
        parse_allowlist("brain/x.py::read::one   # G3\n")
    with pytest.raises(ValueError):
        parse_allowlist("brain/x.py::read::1   # G3\nbrain/x.py::read::2   # G3\n")


def test_cs2_b_a_non_ascii_source_is_scanned_as_utf8(tmp_path: Path) -> None:
    source = (
        "# caf\u00e9 \u65e5\u672c\nimport json\ndef f(p):\n    return json.loads(p.read_text())\n"
    )
    errors, _ = _compare(tmp_path, {"brain/x.py": source}, "brain/x.py::read::1   # G3\n")
    assert errors == []


def test_cs2_g_the_call_node_oracle_one_over_fails_and_at_the_ceiling_passes(
    tmp_path: Path,
) -> None:
    """A copy of a real file with one more json call than its combined ceiling fails; at the ceiling it passes."""
    real = (REPO / "brain" / "self_model" / "state.py").read_text(encoding="utf-8")
    observed_real = scan_source(real, "brain/self_model/state.py")
    total = sum(v for k, v in observed_real.items() if k in ("read", "write", "not-file-io"))
    allow_text = "".join(
        f"brain/self_model/state.py::{k}::{v}   # G3\n" for k, v in observed_real.items()
    )
    errors, _ = _compare(tmp_path, {"brain/self_model/state.py": real}, allow_text)
    assert errors == [] and total >= 1
    bumped = real + "\n\ndef _extra(blob):\n    import json\n    return json.loads(blob)\n"
    errors, _ = _compare(tmp_path, {"brain/self_model/state.py": bumped}, allow_text)
    assert errors and "not-file-io" in errors[0]


def test_cs4_json_io_has_no_second_atomic_writer_or_backup_rotation() -> None:
    """One API: json_io.py writes through attempt_heal's save engine; the only deliberate duplicate is the heal (plan P11)."""
    src = (REPO / "brain" / "health" / "json_io.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = {
        "os.replace",
        "os.rename",
        "shutil.move",
        "shutil.copy2",
        "shutil.copyfile",
        "tempfile.mkstemp",
    }
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                dotted = ast.unparse(node.func)
                assert dotted not in forbidden, f"{fn.name}: {dotted}"
                if dotted == "os.fsync":  # only the append path syncs a file itself
                    assert fn.name == "append_jsonl", f"{fn.name}: a second atomic writer"
    assert ".bak{i + 1}" not in src, "a backup rotation loop"
    assert "save_with_backup_text(" in src


def test_cs6_importing_json_io_first_pulls_in_nothing_from_the_bridge() -> None:
    import subprocess

    code = (
        "import sys\n"
        "import brain.health.json_io\n"
        "bad = sorted(m for m in sys.modules if m == 'brain.bridge' or m.startswith('brain.bridge.'))\n"
        "print(bad)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


if __name__ == "__main__":  # pragma: no cover
    if "--print" in sys.argv:
        print(
            format_observed(
                scan(Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else REPO)
            )
        )
