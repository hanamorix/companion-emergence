"""The model guard (#529, D8/D11): no Claude model is chosen outside the tunables accessor (G1-G5, C16-C19).

AST checks, strictly stronger than the two regex checks they replace and covering ``provider.py``, which the old
checks skipped. The checkers are small pure functions over a source string; the sanctioned accessors and the
allow-lists are test-file constants, so the general guard planned in #531 can lift them. Each rule has a
violating fixture its checker flags and a conforming fixture that passes (``test_fixture_*``).

Rules over every ``brain/**/*.py``:

G1  no model literal: a string token equal to an alias word, or matching a full id with a version digit group
    (``claude-sonnet-5``, ``claude-3-5-sonnet-x``; NOT ``claude-cli`` or ``claude-config``), or embedding
    ``claude-(sonnet|opus|haiku)-<digit>`` in prose. Allow-list entries are ``(file, literal, shape)``.
G2  no model at an argv sink without the accessor (``--model``, ``--model=``, ``--fallback-model``, the
    ``ANTHROPIC_*MODEL`` environment variables).
G3  no copy of an accessor result that outlives the call (module or class scope, parameter default, attribute or
    subscript target, global/nonlocal, setattr/append/add/setdefault/partial, cache decorators).
G4  no constructor bypass (``model=`` on the CLI provider, ``probe_model=`` outside the one probe, the retired
    identifiers).
G5  key ownership (no ``model.`` key literal outside ``model_tier.py``; only ``set_job_models`` calls
    ``tunables.set_values``).
"""

from __future__ import annotations

import ast
import functools
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.source_tree import brain_sources

_REPO_ROOT = Path(__file__).resolve().parents[4]
_APP_SRC = _REPO_ROOT / "app" / "src"
_APP_RUST = _REPO_ROOT / "app" / "src-tauri" / "src"


# ---------------------------------------------------------------------------
# Constants the general guard (#531) can lift
# ---------------------------------------------------------------------------

ALIAS_WORDS = frozenset({"sonnet", "opus", "haiku"})
ID_FULL = re.compile(r"claude-[a-z0-9.-]*\d[a-z0-9.-]*")
ID_PROSE = re.compile(r"claude-(sonnet|opus|haiku)-\d")
# The sanctioned accessors: every way production code may obtain a model id at call time.
ACCESSORS = frozenset({"model_for_tier", "_cli_model", "configured_job_models"})
RETIRED_IDENTIFIERS = frozenset(
    {"model_override", "TIER_MODEL", "MODEL_LITTLE", "MODEL_MEDIUM", "DEFAULT_MODEL"}
)
ENV_MODEL_NAMES = re.compile(r"ANTHROPIC_MODEL|ANTHROPIC_DEFAULT_[A-Z_]*MODEL")
CACHE_DECORATORS = frozenset({"cache", "lru_cache"})
SINK_CALL_NAMES = frozenset({"setattr", "partial", "append", "add", "setdefault"})
PROBE_SITE = ("bridge/server.py", "_check_model_blocking")
# The one function outside set_job_models that may call set_values: the old-file fold, which names no key (G5).
MIGRATION_FOLD_SITE = ("tunables_migration.py", "_fold")

# (file relative to brain/, literal, shape): each entry must still match something at that shape (C19).
G1_ALLOW: tuple[tuple[str, str, str], ...] = (
    # the job table is the DEFINITION of the code defaults, not a copy of them
    ("bridge/model_tier.py", "claude-sonnet-5", "register_default"),
    ("bridge/model_tier.py", "haiku", "register_default"),
    ("bridge/model_tier.py", "claude-haiku-4-5", "register_default"),
    ("bridge/model_tier.py", "claude-opus-5", "opus_choice"),
    # the model grammar's three alias words
    ("bridge/model_tier.py", "sonnet", "known_models"),
    ("bridge/model_tier.py", "opus", "known_models"),
    ("bridge/model_tier.py", "haiku", "known_models"),
    # the picker's vocabulary: the three words resolve_choice / choice_for map to ids
    ("bridge/model_tier.py", "sonnet", "choice_word"),
    ("bridge/model_tier.py", "opus", "choice_word"),
    ("bridge/model_tier.py", "haiku", "choice_word"),
    # the per-turn budget is looked up by model FAMILY (substring of the id), not by a chosen model
    ("bridge/provider.py", "haiku", "budget_family_key"),
    ("bridge/provider.py", "sonnet", "budget_family_key"),
    ("bridge/provider.py", "opus", "budget_family_key"),
    # audit labels: the word records which attempt a row describes, it selects nothing
    ("initiate/adaptive.py", "haiku", "literal_subscript"),
    ("initiate/adaptive.py", "sonnet", "literal_subscript"),
    ("initiate/d_call_schema.py", "haiku", "literal_subscript"),
    ("initiate/d_call_schema.py", "sonnet", "literal_subscript"),
    ("initiate/audit.py", "haiku", "dget_default"),
    ("initiate/reflection.py", "haiku", "label_keyword"),
    ("initiate/reflection.py", "sonnet", "label_keyword"),
    ("initiate/reflection.py", "haiku", "label_param_default"),
)

# Alias-word vocabulary the app may hold (the picker's three words); ids are never allowed.
APP_ALIAS_ALLOW = ("lib/modelId.ts", "components/panels/ModelPicker.tsx")


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


@dataclass
class Parsed:
    tree: ast.AST
    parents: dict[int, ast.AST]
    aliases: dict[str, str]


@functools.lru_cache(maxsize=1)  # the five checks of one file run back to back: parse each file once (#540)
def _parse(source: str) -> Parsed:
    tree = ast.parse(source)
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                aliases[a.asname or a.name] = a.name
    return Parsed(tree, parents, aliases)


def _ancestors(p: Parsed, node: ast.AST):
    cur = p.parents.get(id(node))
    while cur is not None:
        yield cur
        cur = p.parents.get(id(cur))


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = n.body
            if body and isinstance(body[0], ast.Expr):
                v = body[0].value
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    ids.add(id(v))
    return ids


def _call_name(p: Parsed, call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return p.aliases.get(f.id, f.id)
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _is_accessor_call(p: Parsed, node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_name(p, node) in ACCESSORS


def _enclosing_function(p: Parsed, node: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for a in _ancestors(p, node):
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return a
        if isinstance(a, ast.ClassDef):
            return None
    return None


def _scope_kind(p: Parsed, node: ast.AST) -> str:
    """'function' when the nearest enclosing scope is a function or lambda, else 'module' / 'class'."""
    for a in _ancestors(p, node):
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return "function"
        if isinstance(a, ast.ClassDef):
            return "class"
    return "module"


# ---------------------------------------------------------------------------
# G1: literal shapes
# ---------------------------------------------------------------------------


def _shape_of(p: Parsed, node: ast.Constant) -> str | None:
    parent = p.parents.get(id(node))
    grand = p.parents.get(id(parent)) if parent is not None else None
    # tunables.register("key", <default>)
    if isinstance(parent, ast.Call) and _call_name(p, parent) == "register" and node in parent.args[1:2]:
        return "register_default"
    if isinstance(parent, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OPUS_CHOICE" for t in parent.targets):
        return "opus_choice"
    for a in _ancestors(p, node):
        if isinstance(a, (ast.Assign, ast.AnnAssign)):
            targets = a.targets if isinstance(a, ast.Assign) else [a.target]
            if any(isinstance(t, ast.Name) and t.id == "KNOWN_MODELS" for t in targets):
                return "known_models"
            break
    # ("haiku", tunables.register(...)) inside a tuple of pairs
    if isinstance(parent, ast.Tuple) and parent.elts and parent.elts[0] is node and len(parent.elts) == 2:
        second = parent.elts[1]
        if isinstance(second, ast.Call) and _call_name(p, second) == "register":
            return "budget_family_key"
    # Literal["haiku", "sonnet"] subscripts
    sub = parent if isinstance(parent, ast.Subscript) else (grand if isinstance(grand, ast.Subscript) else None)
    if sub is not None and isinstance(sub.value, ast.Name) and sub.value.id == "Literal":
        return "literal_subscript"
    # .get("d_model_tier", "haiku")
    if (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Attribute)
        and parent.func.attr == "get"
        and len(parent.args) == 2
        and parent.args[1] is node
        and isinstance(parent.args[0], ast.Constant)
        and str(parent.args[0].value).startswith("d_model_tier")
    ):
        return "dget_default"
    # model_tier="sonnet" / model_tier_used="haiku" keywords
    if isinstance(parent, ast.keyword) and parent.arg in ("model_tier", "model_tier_used"):
        return "label_keyword"
    # the picker vocabulary inside resolve_choice / choice_for
    fn = _enclosing_function(p, node)
    if fn is not None and fn.name in ("resolve_choice", "choice_for", "needs_probe"):
        return "choice_word"
    # def f(*, model_tier: str = "haiku")
    if isinstance(parent, ast.arguments):
        args = parent.args + parent.kwonlyargs
        defaults = list(parent.defaults) + [d for d in parent.kw_defaults if d is not None]
        names = [a.arg for a in args]
        if node in defaults and any(n.startswith("model_tier") for n in names):
            return "label_param_default"
    return None


def _is_assembly_piece(p: Parsed, node: ast.Constant) -> bool:
    """A string starting ``claude-`` that is one piece of a larger string: an f-string part, a ``+`` or ``%`` operand,
    or the template of ``.format(...)``. Such a piece completes to a model id at run time, which G1's whole-literal
    match cannot see."""
    if not node.value.startswith("claude-") or node.value.startswith("claude-cli"):  # claude-cli is the provider name
        return False
    parent = p.parents.get(id(node))
    if isinstance(parent, ast.JoinedStr):
        return True
    if isinstance(parent, ast.BinOp) and isinstance(parent.op, (ast.Add, ast.Mod)):
        return True
    return isinstance(parent, ast.Attribute) and parent.attr == "format" and isinstance(p.parents.get(id(parent)), ast.Call)


def _is_model_literal(value: str) -> bool:
    return value in ALIAS_WORDS or bool(ID_FULL.fullmatch(value)) or bool(ID_PROSE.search(value))


def check_g1(source: str, relpath: str, allow=G1_ALLOW, used: set | None = None) -> list[str]:
    p = _parse(source)
    docs = _docstring_ids(p.tree)
    out: list[str] = []
    for node in ast.walk(p.tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)) or id(node) in docs:
            continue
        if _is_assembly_piece(p, node):
            out.append(f"G1 {relpath}:{node.lineno}: model id assembled from pieces ({node.value[:60]!r})")
            continue
        if not _is_model_literal(node.value):
            continue
        shape = _shape_of(p, node)
        entry = (relpath, node.value, shape)
        if shape is not None and entry in allow:
            if used is not None:
                used.add(entry)
            continue
        out.append(f"G1 {relpath}:{node.lineno}: model literal {node.value[:60]!r} (shape {shape})")
    return out


# ---------------------------------------------------------------------------
# G2: argv sinks
# ---------------------------------------------------------------------------


def _assignments_to(func: ast.AST, name: str) -> list[ast.AST]:
    """Every value assigned to the local ``name`` in ``func``; None entries stand for an unanalysable binding."""
    values: list[ast.AST] = []
    for n in ast.walk(func):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    values.append(n.value)
                elif isinstance(t, (ast.Tuple, ast.List)) and any(isinstance(e, ast.Name) and e.id == name for e in t.elts):
                    values.append(ast.Constant(value=None))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == name and n.value is not None:
            values.append(n.value)
        elif isinstance(n, (ast.AugAssign, ast.NamedExpr)):
            t = n.target
            if isinstance(t, ast.Name) and t.id == name:
                values.append(ast.Constant(value=None))
        elif isinstance(n, (ast.For, ast.comprehension)):
            if isinstance(n.target, ast.Name) and n.target.id == name:
                values.append(ast.Constant(value=None))
        elif isinstance(n, ast.withitem) and isinstance(n.optional_vars, ast.Name) and n.optional_vars.id == name:
            values.append(ast.Constant(value=None))
    return values


def _is_sanctioned_value(p: Parsed, value: ast.AST, scope_node: ast.AST) -> bool:
    if _is_accessor_call(p, value):
        return True
    if isinstance(value, ast.Name):
        func = _enclosing_function(p, scope_node)
        if func is None:
            return False
        params = {a.arg for a in func.args.args + func.args.kwonlyargs + func.args.posonlyargs}
        if func.args.vararg:
            params.add(func.args.vararg.arg)
        if func.args.kwarg:
            params.add(func.args.kwarg.arg)
        if value.id in params:
            return False
        vals = _assignments_to(func, value.id)
        return bool(vals) and all(_is_accessor_call(p, v) for v in vals)
    return False


def check_g2(source: str, relpath: str) -> list[str]:
    p = _parse(source)
    out: list[str] = []
    for node in ast.walk(p.tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            parent = p.parents.get(id(node))
            if v == "--model":
                if isinstance(parent, (ast.List, ast.Tuple)):
                    idx = parent.elts.index(node)
                    if idx + 1 >= len(parent.elts) or not _is_sanctioned_value(p, parent.elts[idx + 1], node):
                        out.append(f"G2 {relpath}:{node.lineno}: --model value is not an accessor result")
                elif (
                    isinstance(parent, ast.Call)
                    and isinstance(parent.func, ast.Attribute)
                    and parent.func.attr in ("append", "extend", "insert")
                ):
                    out.append(f"G2 {relpath}:{node.lineno}: --model appended on its own (unverifiable form)")
                # anything else (``cmd.index("--model")``, an argparse option, a comparison) reads or declares the flag
            elif v.startswith("--model="):
                out.append(f"G2 {relpath}:{node.lineno}: single-token --model=<x> form")
            elif v == "--fallback-model":
                out.append(f"G2 {relpath}:{node.lineno}: --fallback-model")
            elif ENV_MODEL_NAMES.fullmatch(v):
                out.append(f"G2 {relpath}:{node.lineno}: model environment variable {v}")
    return out


# ---------------------------------------------------------------------------
# G3: no copy that outlives the call
# ---------------------------------------------------------------------------


def _returns_accessor(p: Parsed, func: ast.FunctionDef) -> bool:
    for n in ast.walk(func):
        if isinstance(n, ast.Return) and n.value is not None:
            if _is_accessor_call(p, n.value):
                return True
            if isinstance(n.value, ast.Name):
                vals = _assignments_to(func, n.value.id)
                if vals and any(_is_accessor_call(p, v) for v in vals):
                    return True
    return False


def _decorated_with_cache(func: ast.AST) -> bool:
    for d in getattr(func, "decorator_list", []):
        target = d.func if isinstance(d, ast.Call) else d
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
        if name in CACHE_DECORATORS:
            return True
    return False


def check_g3(source: str, relpath: str) -> list[str]:
    p = _parse(source)
    out: list[str] = []
    returners = {
        n.name
        for n in ast.walk(p.tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _returns_accessor(p, n)
    }

    def flag(node: ast.AST, why: str) -> None:
        out.append(f"G3 {relpath}:{getattr(node, 'lineno', '?')}: {why}")

    for node in ast.walk(p.tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(p, node)
        if name in returners and name not in ACCESSORS and _scope_kind(p, node) != "function":
            flag(node, f"module/class-scope call of {name}(), which returns an accessor result")
        if name not in ACCESSORS:
            continue
        scope = _scope_kind(p, node)
        if scope != "function":
            flag(node, f"accessor call at {scope} scope")
        for a in _ancestors(p, node):
            if isinstance(a, ast.arguments):
                flag(node, "accessor call in a parameter default")
                break
            if isinstance(a, ast.Call) and a is not node:
                cn = _call_name(p, a)
                if cn in SINK_CALL_NAMES and (node in a.args or any(k.value is node for k in a.keywords)):
                    flag(node, f"accessor result passed to {cn}()")
                if cn == "field":
                    flag(node, "accessor call inside dataclass field()")
            if isinstance(a, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = a.targets if isinstance(a, ast.Assign) else [a.target]
                for t in targets:
                    if isinstance(t, (ast.Attribute, ast.Subscript)):
                        flag(node, "accessor result stored in an attribute or subscript target")
                    if isinstance(t, ast.Name):
                        fn = _enclosing_function(p, a)
                        if fn is not None:
                            for s in ast.walk(fn):
                                if isinstance(s, (ast.Global, ast.Nonlocal)) and t.id in s.names:
                                    flag(node, "accessor result stored in a global/nonlocal name")
                break
            if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)) and _decorated_with_cache(a):
                flag(node, "accessor call inside a cached function")
    return out


# ---------------------------------------------------------------------------
# G4 / G5
# ---------------------------------------------------------------------------


def check_g4(source: str, relpath: str) -> list[str]:
    p = _parse(source)
    out: list[str] = []
    for node in ast.walk(p.tree):
        idents: list[str] = []
        if isinstance(node, ast.Name):
            idents.append(node.id)
        elif isinstance(node, ast.Attribute):
            idents.append(node.attr)
        elif isinstance(node, ast.arg):
            idents.append(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            idents.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            idents.append(node.name)
        elif isinstance(node, ast.alias):
            idents.extend([node.name, node.asname or ""])
        for ident in idents:
            if ident in RETIRED_IDENTIFIERS:
                out.append(f"G4 {relpath}:{getattr(node, 'lineno', '?')}: retired identifier {ident}")
        if isinstance(node, ast.Call) and _call_name(p, node) == "ClaudeCliProvider":
            for kw in node.keywords:
                if kw.arg == "model":
                    out.append(f"G4 {relpath}:{node.lineno}: ClaudeCliProvider(model=...)")
                if kw.arg == "probe_model":
                    fn = _enclosing_function(p, node)
                    if (relpath, getattr(fn, "name", None)) != PROBE_SITE:
                        out.append(f"G4 {relpath}:{node.lineno}: probe_model= outside the one probe site")
        if isinstance(node, ast.FunctionDef) and node.name == "get_provider":
            names = {a.arg for a in node.args.args + node.args.kwonlyargs}
            for bad in ("model_override", "persona_dir"):
                if bad in names:
                    out.append(f"G4 {relpath}:{node.lineno}: get_provider has a {bad} parameter")
    return out


def check_g5(source: str, relpath: str) -> list[str]:
    p = _parse(source)
    docs = _docstring_ids(p.tree)
    out: list[str] = []
    owner = relpath == "bridge/model_tier.py"
    for node in ast.walk(p.tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("model.")
            and not owner
            and id(node) not in docs
        ):
            out.append(f"G5 {relpath}:{node.lineno}: model. key literal {node.value!r} outside model_tier.py")
        if isinstance(node, ast.Call) and _call_name(p, node) == "set_values" and relpath != "tunables.py":
            fn = _enclosing_function(p, node)
            # The legacy-file fold writes whatever keys an old JSON settings file held, naming none of them: it is key-agnostic,
            # so it cannot know a model key (nothing to leak). The exemption is that ONE function, not its file: a second
            # caller there is flagged. Every model-specific writer goes through set_job_models.
            fn_name = getattr(fn, "name", None)
            if (relpath, fn_name) == MIGRATION_FOLD_SITE:
                continue
            if not (owner and fn_name == "set_job_models"):
                out.append(f"G5 {relpath}:{node.lineno}: set_values called outside set_job_models")
    return out


def check_all(source: str, relpath: str) -> list[str]:
    return (
        check_g1(source, relpath)
        + check_g2(source, relpath)
        + check_g3(source, relpath)
        + check_g4(source, relpath)
        + check_g5(source, relpath)
    )


# ---------------------------------------------------------------------------
# The guard over the real tree
# ---------------------------------------------------------------------------


@functools.cache
def _scan_brain() -> tuple[tuple[str, ...], frozenset[tuple[str, str, str]]]:
    """G1-G5 over every file of brain/ (paths relative to brain/): the offenders, and the allow-list entries that matched.

    Both tests below read it, so the tree is scanned once per process."""
    offenders: list[str] = []
    used: set[tuple[str, str, str]] = set()
    for full, source in brain_sources().items():
        rel = full.removeprefix("brain/")
        offenders += check_g1(source, rel, used=used)
        for check in (check_g2, check_g3, check_g4, check_g5):
            offenders += check(source, rel)
    return tuple(offenders), frozenset(used)


@pytest.mark.slow(reason='6.0 s on macos in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
def test_no_model_choice_outside_the_accessor_anywhere_in_brain() -> None:
    """C16: G1-G5 pass on the final tree, provider.py included."""
    offenders, _used = _scan_brain()
    assert not offenders, "model choice outside the tunables accessor:\n" + "\n".join(offenders)


def test_every_allow_list_entry_still_matches_something_at_its_shape() -> None:
    """C19: allow-lists cannot rot. An entry that matches nothing at its shape fails (delete it)."""
    _offenders, used = _scan_brain()
    unused = [e for e in G1_ALLOW if e not in used]
    assert unused == [], f"stale allow-list entries (no match at their shape): {unused}"


# ---------------------------------------------------------------------------
# C17: each rule flags a violating fixture and passes a conforming one
# ---------------------------------------------------------------------------

_CONFORMING = {
    "d11_local_copy": (
        "from brain.bridge.model_tier import model_for_tier\n"
        "def f(t):\n"
        "    model = model_for_tier(t)\n"
        "    argv = ['claude', '--model', model]\n"
        "    return argv\n"
    ),
    "direct_call_in_argv": (
        "from brain.bridge.model_tier import model_for_tier\n"
        "def f(t):\n"
        "    return ['claude', '--model', model_for_tier(t)]\n"
    ),
    "cli_model_method": (
        "class P:\n"
        "    def g(self):\n"
        "        model = self._cli_model()\n"
        "        cmd = ['claude']\n"
        "        cmd.extend(['--model', model])\n"
        "        return cmd\n"
    ),
    "prose_and_claude_cli": (
        "X = 'claude-cli'\nY = 'claude-config'\nZ = 'claude-work'\nW = 'claude-tool'\n"
        "def f():\n    '''docstring naming claude-sonnet-5 and sonnet is exempt'''\n    return 'a sonnet is a poem'\n"
    ),
}

_VIOLATING = {
    "module_constant": "from brain.bridge.model_tier import model_for_tier\nM = model_for_tier('x')\n",
    "class_constant": "from brain.bridge.model_tier import model_for_tier\nclass C:\n    M = model_for_tier('x')\n",
    "parameter_default": "from brain.bridge.model_tier import model_for_tier\ndef f(m=model_for_tier('x')):\n    return m\n",
    "dataclass_default_factory": (
        "from dataclasses import dataclass, field\n"
        "from brain.bridge.model_tier import model_for_tier\n"
        "@dataclass\nclass C:\n    m: str = field(default_factory=lambda: model_for_tier('x'))\n"
    ),
    "self_attribute": (
        "from brain.bridge.model_tier import model_for_tier\n"
        "class C:\n    def __init__(self):\n        self._m = model_for_tier('x')\n"
    ),
    "subscript_cache": (
        "from brain.bridge.model_tier import model_for_tier\n"
        "def f(cache, k):\n    cache[k] = model_for_tier('x')\n"
    ),
    "nonlocal": (
        "from brain.bridge.model_tier import model_for_tier\n"
        "def outer():\n    m = ''\n    def inner():\n        nonlocal m\n        m = model_for_tier('x')\n    return inner\n"
    ),
    "global": (
        "from brain.bridge.model_tier import model_for_tier\n"
        "M = ''\ndef f():\n    global M\n    M = model_for_tier('x')\n"
    ),
    "setattr": (
        "from brain.bridge.model_tier import model_for_tier\n"
        "def f(o):\n    setattr(o, 'm', model_for_tier('x'))\n"
    ),
    "append": (
        "from brain.bridge.model_tier import model_for_tier\n"
        "def f(acc):\n    acc.append(model_for_tier('x'))\n"
    ),
    "setdefault": (
        "from brain.bridge.model_tier import model_for_tier\n"
        "def f(acc):\n    acc.setdefault('m', model_for_tier('x'))\n"
    ),
    "partial": (
        "import functools\nfrom brain.bridge.model_tier import model_for_tier\n"
        "def f(g):\n    return functools.partial(g, model_for_tier('x'))\n"
    ),
    "lru_cache": (
        "import functools\nfrom brain.bridge.model_tier import model_for_tier\n"
        "@functools.lru_cache\ndef f():\n    return model_for_tier('x')\n"
    ),
    "module_call_of_returning_function": (
        "from brain.bridge.model_tier import model_for_tier\n"
        "def current():\n    return model_for_tier('x')\n"
        "M = current()\n"
    ),
    "aliased_import": "from brain.bridge.model_tier import model_for_tier as m\nM = m('x')\n",
    "argv_literal_alias": "def f():\n    return ['claude', '--model', 'sonnet']\n",
    "argv_parameter": "def f(some_param):\n    return ['claude', '--model', some_param]\n",
    "argv_appended_alone": "def f(cmd, m):\n    cmd.append('--model')\n    cmd.append(m)\n",
    "argv_single_token": "def f():\n    return ['claude', '--model=claude-sonnet-5']\n",
    "fallback_model": "def f(x):\n    return ['claude', '--fallback-model', x]\n",
    "env_model": "def f(env):\n    env['ANTHROPIC_MODEL'] = 'x'\n",
    "env_default_model": "def f(env):\n    env['ANTHROPIC_DEFAULT_SONNET_MODEL'] = 'x'\n",
    "model_key_literal": "KEY = 'model.interactive_chat'\n",
    "set_values_elsewhere": "from brain import tunables\ndef f():\n    tunables.set_values({'a': 1})\n",
    "constructor_model": "from brain.bridge.provider import ClaudeCliProvider\np = ClaudeCliProvider(model='x')\n",
    "probe_model_elsewhere": "from brain.bridge.provider import ClaudeCliProvider\ndef f(x):\n    return ClaudeCliProvider(probe_model=x)\n",
    "retired_identifier": "from x import TIER_MODEL\n",
    "get_provider_override_param": "def get_provider(name, model_override=None):\n    return name\n",
    "full_id_literal": "X = 'claude-sonnet-5'\n",
    "full_id_with_version_digits": "X = 'claude-3-5-sonnet-20241022'\n",
    "full_id_in_prose": "X = 'use a full id such as claude-opus-5 here'\n",
    "id_assembled_in_an_fstring": "def f(fam):\n    return f'claude-{fam}-5'\n",
    "id_assembled_by_concatenation": "def f(fam):\n    return 'claude-' + fam + '-5'\n",
    "id_assembled_by_percent_format": "def f(fam):\n    return 'claude-%s-5' % fam\n",
    "id_assembled_by_format_call": "def f(fam):\n    return 'claude-{}-5'.format(fam)\n",
}


@pytest.mark.parametrize("name", sorted(_CONFORMING))
def test_fixture_conforming_source_passes_every_rule(name: str) -> None:
    assert check_all(_CONFORMING[name], "some/module.py") == []


@pytest.mark.parametrize("name", sorted(_VIOLATING))
def test_fixture_violating_source_is_flagged(name: str) -> None:
    flagged = check_all(_VIOLATING[name], "some/module.py")
    assert flagged, f"fixture {name!r} was not flagged by any rule"


def test_fixture_g5_exempts_the_fold_function_only_not_the_file() -> None:
    """Rev-4 review S2: a second ``set_values`` caller inside tunables_migration.py is flagged; the fold itself passes."""
    fold = "from brain import tunables\ndef _fold(legacy, complete):\n    tunables.set_values({'a': 1}, only_if_default=True)\n"
    second = "from brain import tunables\ndef _another(u):\n    tunables.set_values(u)\n"
    nested = "from brain import tunables\ndef _fold(l, c):\n    def inner(u):\n        tunables.set_values(u)\n    return inner\n"
    module_level = "from brain import tunables\ntunables.set_values({})\n"
    assert check_g5(fold, "tunables_migration.py") == []
    assert check_g5(second, "tunables_migration.py") != []
    assert check_g5(nested, "tunables_migration.py") != []
    assert check_g5(module_level, "tunables_migration.py") != []
    assert check_g5(fold, "other.py") != []  # the same function name elsewhere is not exempt


def test_fixture_the_one_probe_site_may_use_probe_model() -> None:
    src = "from brain.bridge.provider import ClaudeCliProvider\ndef _check_model_blocking(x):\n    return ClaudeCliProvider(probe_model=x)\n"
    assert check_g4(src, "bridge/server.py") == []


def test_fixture_g1_allow_entry_is_shape_specific() -> None:
    """C19: a real literal in an allow-listed file in a non-label shape fails; in the label shape it passes."""
    label = "def f(deps):\n    return DCallRow(model_tier_used='sonnet')\n"
    other = "X = 'claude-sonnet-5'\n"
    assert check_g1(label, "initiate/reflection.py") == []
    assert check_g1(other, "initiate/reflection.py") != []
    # a labelled site that disappears leaves its allow-list entry unmatched
    used: set = set()
    check_g1("X = 1\n", "initiate/reflection.py", used=used)
    assert ("initiate/reflection.py", "sonnet", "label_keyword") not in used


def test_fixture_g1_does_not_flag_unrelated_claude_words() -> None:
    src = "A = 'claude-cli'\nB = 'claude-config'\nC = 'claude-work'\nD = 'claude-tool'\nE = 'the claude CLI'\n"
    assert check_g1(src, "x.py") == []
    # a "claude-" prefix TEST is not id assembly; a prefix piece that a variable completes is
    assert check_g1("def f(v):\n    return v.startswith('claude-')\n", "x.py") == []
    assert check_g1("def f(n):\n    return f'claude-{n}'\n", "x.py") != []  # a prefix a variable completes can build any id
    assert check_g1("def f(n):\n    return f'the claude CLI {n}'\n", "x.py") == []
    assert check_g1("def f(m):\n    return 'claude-cli:' + m\n", "x.py") == []  # the provider label is not a model id


# ---------------------------------------------------------------------------
# C18: the app guard (string tokens only)
# ---------------------------------------------------------------------------


def _strip_ts_tokens(source: str) -> list[tuple[int, str]]:
    """The quoted string tokens of TypeScript/TSX source, comments removed: ``(line, content)``."""
    out: list[tuple[int, str]] = []
    i, n, line = 0, len(source), 1
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if ch == "\n":
            line += 1
            i += 1
        elif ch == "/" and nxt == "/":
            while i < n and source[i] != "\n":
                i += 1
        elif ch == "/" and nxt == "*":
            end = source.find("*/", i + 2)
            end = n if end == -1 else end + 2
            line += source.count("\n", i, end)
            i = end
        elif ch in "'\"`":
            quote, j, buf = ch, i + 1, []
            start_line = line
            while j < n and source[j] != quote:
                if source[j] == "\\" and j + 1 < n:
                    buf.append(source[j + 1])
                    j += 2
                    continue
                if source[j] == "\n":
                    line += 1
                    if quote != "`":
                        break
                buf.append(source[j])
                j += 1
            out.append((start_line, "".join(buf)))
            i = j + 1
        else:
            i += 1
    return out


def _rust_tokens(source: str) -> list[tuple[int, str]]:
    """String literals of Rust source before the first ``#[cfg(test)]``, comments removed."""
    cut = source.find("#[cfg(test)]")
    body = source if cut == -1 else source[:cut]
    out: list[tuple[int, str]] = []
    i, n, line = 0, len(body), 1
    while i < n:
        ch = body[i]
        nxt = body[i + 1] if i + 1 < n else ""
        if ch == "\n":
            line += 1
            i += 1
        elif ch == "/" and nxt == "/":
            while i < n and body[i] != "\n":
                i += 1
        elif ch == "/" and nxt == "*":
            end = body.find("*/", i + 2)
            end = n if end == -1 else end + 2
            line += body.count("\n", i, end)
            i = end
        elif ch == '"':
            j, buf, start_line = i + 1, [], line
            while j < n and body[j] != '"':
                if body[j] == "\\" and j + 1 < n:
                    buf.append(body[j + 1])
                    j += 2
                    continue
                if body[j] == "\n":
                    line += 1
                buf.append(body[j])
                j += 1
            out.append((start_line, "".join(buf)))
            i = j + 1
        elif ch == "'" and re.match(r"'(\\.|[^'\\])'", body[i : i + 4]):
            i += 3 if body[i + 1] != "\\" else 4
        else:
            i += 1
    return out


def check_app_tokens(tokens: list[tuple[int, str]], relpath: str, alias_allowed: bool) -> list[str]:
    out: list[str] = []
    for line, text in tokens:
        if ID_FULL.fullmatch(text) or ID_PROSE.search(text):
            out.append(f"{relpath}:{line}: model id in a string token {text[:60]!r}")
        elif text in ALIAS_WORDS and not alias_allowed:
            out.append(f"{relpath}:{line}: model alias word {text!r} outside the picker vocabulary")
    return out


def _app_files():
    for path in sorted(_APP_SRC.rglob("*")):
        if path.suffix in (".ts", ".tsx") and ".test." not in path.name and "node_modules" not in path.parts:
            yield path, _strip_ts_tokens, path.relative_to(_APP_SRC).as_posix()
    for path in sorted(_APP_RUST.glob("*.rs")):
        yield path, _rust_tokens, "src-tauri/" + path.name


def test_the_app_holds_no_model_id_and_alias_words_only_in_the_picker_vocabulary() -> None:
    offenders: list[str] = []
    for path, tokenize, rel in _app_files():
        offenders += check_app_tokens(tokenize(path.read_text(encoding="utf-8")), rel, rel in APP_ALIAS_ALLOW)
    assert offenders == [], "model choice in the app:\n" + "\n".join(offenders)


def test_the_connection_panel_no_longer_defaults_the_picker_to_sonnet() -> None:
    text = (_APP_SRC / "components" / "panels" / "ConnectionPanel.tsx").read_text(encoding="utf-8")
    assert '?? "sonnet"' not in text


def test_app_allow_list_entries_still_hold_alias_vocabulary() -> None:
    """C19 for the app: an allow-listed file that no longer holds an alias word must leave the list."""
    for rel in APP_ALIAS_ALLOW:
        path = _APP_SRC / rel
        assert path.exists(), rel
        words = {t for _l, t in _strip_ts_tokens(path.read_text(encoding="utf-8")) if t in ALIAS_WORDS}
        assert words, f"{rel} holds no alias-word token any more: remove it from APP_ALIAS_ALLOW"


def test_fixture_app_tokenizer_flags_ids_and_ignores_comments_prose_and_claude_cli() -> None:
    src = (
        "// const A = 'claude-sonnet-5'\n"
        "/* const B = \"sonnet\" */\n"
        "const C = 'claude-cli';\n"
        "const D = \"claude-sonnet-5\";\n"
        "const E = `Leave blank for the default`;\n"
        "const F = 'sonnet';\n"
    )
    flagged = check_app_tokens(_strip_ts_tokens(src), "x.ts", alias_allowed=False)
    assert len(flagged) == 2 and "claude-sonnet-5" in flagged[0] and "'sonnet'" in flagged[1]
    assert check_app_tokens(_strip_ts_tokens(src), "x.ts", alias_allowed=True) == [flagged[0]]
    rust = 'const A: &str = "claude-opus-5";\n// "claude-sonnet-5"\n#[cfg(test)]\nmod t { const B: &str = "claude-sonnet-5"; }\n'
    assert len(check_app_tokens(_rust_tokens(rust), "lib.rs", alias_allowed=False)) == 1
