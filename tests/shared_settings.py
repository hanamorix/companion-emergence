"""The checkers of the shared-settings guard (#531); the tests that run them are tests/unit/test_shared_settings_guard.py.

A *setting* is a value meant to live in one place: a user tunable (``tunables.register``, read through
``tunables.get_tunable``) or a developer constant (``brain/dev_constants.py``). The guard finds code that copies,
reassigns or caches one. Nothing is typed twice: ``derive_settings`` reads the settings out of the tree itself.

What is detected (``brain/*.py``; findings are one per site, identity ``(rule, file, site, settings)``, no line numbers):

R1  write: an attribute store, augmented store or delete on a setting's constant name; ``setattr``/``delattr`` with such
    a literal name; a rebind of a name imported from ``brain.dev_constants`` or ``brain.tunables``; ``set_values`` called
    outside the sanctioned writers (``set_job_models``, the legacy-file ``_fold``, tunables.py).
R2  name: a module or class constant that is given a value other than ``None``, or a parameter that has a default other
    than ``None``, whose name equals a setting's name once leading underscores, case and unit words are ignored (exact
    set of words). A bare annotation (``schema_version: str`` in a dataclass) holds no copy and is not looked at. A dev
    constant also answers to its name without a group prefix (see below).
R3  value, in three tiers; one finding per site, naming every setting that shares the value; the first tier that reports a
    site wins:
      distinctive: a numeric literal or constant arithmetic expression (``+ - * **``) equal to a setting's value that has
        at least four significant digits, or a string equal to a scalar string setting's value when it contains a
        character that is not a letter (file names, ids, versions);
      collection: a list, tuple, set or ``frozenset(...)`` literal of strings whose element set (at least two) equals the
        element set of a collection setting; elements are never matched one by one;
      loose: any numeric value equal to a setting's value, bound to a name (assignment target, parameter, keyword) that
        contains every non-unit word of ONE of the setting's name forms, that form having at least two such words.
R4  outlives the call: a result of ``get_tunable`` or of a model accessor kept at module or class scope, in a parameter
    default, in an attribute or subscript target, in a global or nonlocal, handed to setattr/append/add/setdefault/partial,
    in a dataclass ``field``, under a cache decorator, or returned by a function that is itself called at module or class
    scope. A local variable is fine.
R4b a literal where the registered default constant belongs: ``get_tunable(key, <number or string literal>)``.
R5  alias and default: (a) a module or class-level ``X = dev_constants.Y`` (a second name, a re-export alias, a frozen
    dataclass field default); (b) a parameter default that is exactly a dev constant read, which is frozen when the function
    is defined. A ``from brain.dev_constants import NAME`` binding is not flagged.
G1 G2 G4 G5  the model case (#529): the model grammar is the recognizer of a copy of a model setting, because ids change
    per release and a list of values cannot be kept. G1 model literals (``MODEL_LITERAL_ALLOW`` holds the setting's own
    vocabulary), G2 ``--model`` argv sinks and the ``ANTHROPIC_*MODEL`` environment names, G4 constructor bypass and retired
    identifiers, G5 a ``model.`` key literal outside model_tier.py.
APP-R3, APP-MODEL  the app (TypeScript and Rust, tests excluded): the distinctive tier over numeric expressions and string
    tokens, plus the model tokens. Names are not compared across languages.

Thresholds. Each is derived from a measurement of the tree and says which:

* Four significant digits. A literal's purpose is not in the AST, and round figures recur for unrelated reasons. The
  counts of numeric literals equal to a setting's value over ``brain/*.py`` (folded exactly as this module folds, ``+ - *
  **`` only; int and float equal; the definitions included), measured when this guard landed: 1.0: 1019, 2.0: 226,
  5.0: 102, 3: 157, 30.0: 33, 200: 48, 8: 61, 20.0: 42. Every such figure has one or two significant digits. The shared
  value that separates three from four is ``86400`` ("864", three digits): the seconds in a day, written in 15 places of
  ``brain/`` for unrelated purposes and also the value of ``COST_SERIES_WINDOW_S``. Three would make it a source of false
  positives; four excludes it, and the next shared figure up (``0.14227``, five digits) is a measured result. Values with
  four or more digits occurred at their own definitions only, except the 20 MiB byte figure (the default of three upload
  tunables, also written in ``bridge/server.py`` and ``app/src/components/ChatPanel.tsx``).
* Two words. One word names a kind of thing, not a purpose (``OPUS``, ``FALLBACK``, ``REPROBE`` would match anything), so
  a name form (exact or loose) needs at least two non-unit words; the loose tier needs all of them in the bound name.
* Unit words (``S SEC SECS SECOND SECONDS MS MINUTES HOURS DAYS BYTES``) are spellings of the unit of the number, not its
  purpose: ``_S`` and ``_SECONDS`` are one name. ``DEFAULT`` is a qualifier and is not a unit word.
* Group prefix: the moves into dev_constants gave constants a prefix (``ATTUNEMENT_``, ``KINDLED_LINK_``,
  ``JUDGE_TUNE_GRADE_``), so a copy under the natural unprefixed name would not equal the name. A leading run of name
  words shared by at least three dev constants is a group prefix; the name without it is also a form of the setting when
  at least three non-unit words remain. Neither number is a tuned target. ``MIN_GROUP_SIZE = 3`` is the smallest size
  that is more than a pair (a prefix shared by two constants may be part of two different names); it has no measurement
  behind it, and on today's tree any size from 1 to 5 gives the same findings. ``MIN_STRIPPED_WORDS = 3`` is the smallest
  remainder with no unrelated match on today's tree: with two words the rule flagged five unrelated constants,
  ``SCHEMA_VERSION`` (the attunement schema against the other schema versions in ``ingest/`` and ``works/``) and
  ``CADENCE_FILE`` (against two unrelated cadence files in ``kindled_link/``). The cost: a copy under these natural
  names is not caught by name, the eight prefixed dev constants whose remainder is one or two words
  (``SCHEMA_VERSION``, ``ADDRESS_COOLDOWN``, ``SESSION_COOLDOWN``, ``BUDGET_MAX``, ``CADENCE_FILE``, ``INTERVAL``,
  ``GRADE_LORA``, ``GRADE_ORDER``); a copy of their value with four or more digits, or bound to the setting's words, still
  is.

NOT detected: a copy of a round value under an unrelated name (``delay = 60`` for a 60 s tick) or under a related name
that lacks one of the setting's words (``scan_cap = 500`` for ``DEFAULT_SCAN_CAP``); a copy computed at run time; a value
spelled differently across a call (``datetime.timedelta(minutes=15)`` for 900 s) or in another unit or inside a string
(``PRAGMA busy_timeout = 5000`` for the 5.0 s ``SQLITE_STORE_BUSY_TIMEOUT_S``, in brain/initiate/resonance.py); a default
computed from a dev constant (``def f(x=dev_constants.Y * 3600.0)``); constant folding other than ``+ - * **`` (``/``,
``//``, ``<<`` are not folded); a collection that is not a literal of strings (a dict, a comprehension, a list built by a
call); a tunable registered through ``from brain.tunables import register`` or ``brain.tunables.register(...)`` (the tree
has the one spelling ``tunables.register``); a result of ``tunables.get_raw`` (the raw-file reader of the migration and the
model writer) kept somewhere that outlives the call; a function that returns an accessor result and is called at module or
class scope from ANOTHER file (the returner set is per file); and anything in a language the check does not parse. In the
app also: a collection of strings (the accepted image kinds in ChatPanel.tsx), a value scaled to another unit
(``HEARTBEAT_INTERVAL_MINUTES = 15`` against a 900 s setting, ``600_000`` ms against 600.0 s), a value spelled in prose
("max 20 MB") or inside a template literal, hex, shifts, ``**`` and exponents, and Rust integer suffixes (``1024u64``); a
product is read as written, so ``n * 20 * 1024 * 1024`` reads its trailing run ``20 * 1024 * 1024``.

The checkers are small pure functions over a source string (or a ``Parsed``); a checker takes the ``Settings`` it should
guard, so fixtures run over a small synthetic one and the real tree over ``derive_settings``. Parent maps are built per
file and dropped; the shared parsed trees of ``tests/source_tree.py`` are never annotated.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType, ModuleType

from tests.source_tree import tree_asts, tree_sources

DEV_OWNER = "brain/dev_constants.py"

UNIT_WORDS = frozenset({"S", "SEC", "SECS", "SECOND", "SECONDS", "MS", "MINUTES", "HOURS", "DAYS", "BYTES"})
MIN_SIGNIFICANT_DIGITS = 4
MIN_FORM_WORDS = 2
MIN_STRIPPED_WORDS = MIN_FORM_WORDS + 1
MIN_GROUP_SIZE = 3

# The model case (#529, commit 44af2447): the grammar that recognizes a copy of a model setting.
ALIAS_WORDS = frozenset({"sonnet", "opus", "haiku"})
ID_FULL = re.compile(r"claude-[a-z0-9.-]*\d[a-z0-9.-]*")
ID_PROSE = re.compile(r"claude-(sonnet|opus|haiku)-\d")
# The sanctioned accessors: every way production code may obtain a model id at call time. ``get_tunable`` joins them for R4.
MODEL_ACCESSORS = frozenset({"model_for_tier", "_cli_model", "configured_job_models"})
ACCESSORS = MODEL_ACCESSORS | {"get_tunable"}
RETIRED_IDENTIFIERS = frozenset(
    {"model_override", "TIER_MODEL", "MODEL_LITTLE", "MODEL_MEDIUM", "DEFAULT_MODEL"}
)
ENV_MODEL_NAMES = re.compile(r"ANTHROPIC_MODEL|ANTHROPIC_DEFAULT_[A-Z_]*MODEL")
CACHE_DECORATORS = frozenset({"cache", "lru_cache"})
SINK_CALL_NAMES = frozenset({"setattr", "partial", "append", "add", "setdefault"})
PROBE_SITE = ("bridge/server.py", "_check_model_blocking")
# The sanctioned callers of ``set_values`` (R1d): the model writer, and the old-file fold, which names no key.
SET_VALUES_SITES = (("bridge/model_tier.py", "set_job_models"), ("tunables_migration.py", "_fold"))

# (file relative to brain/, literal, shape): each entry must still match something at that shape (C19). The job table is
# the DEFINITION of the code defaults, not a copy of them; the rest are the setting's own vocabulary and labels.
MODEL_LITERAL_ALLOW: tuple[tuple[str, str, str], ...] = (
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

# Alias-word vocabulary the app may hold (the picker's three words), repository-relative; ids are never allowed.
APP_ALIAS_ALLOW = ("app/src/lib/modelId.ts", "app/src/components/panels/ModelPicker.tsx")


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One violation at one site. Identity is ``key`` (no line number: it survives edits); the line is for humans."""

    rule: str
    file: str
    site: str
    settings: tuple[str, ...]
    line: int
    message: str

    @property
    def key(self) -> tuple[str, str, str, tuple[str, ...]]:
        return (self.rule, self.file, self.site, self.settings)


def _finding(rule: str, file: str, line: int, site: str, text: str, settings: Iterable[str] = ()) -> Finding:
    labels = tuple(sorted(settings))
    suffix = f" [settings: {', '.join(labels)}]" if labels else ""
    return Finding(rule, file, site, labels, line, f"{rule} {file}:{line}: {text} (site: {site}){suffix}")


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _raw_words(name: str) -> list[str]:
    return [w.upper() for w in re.split(r"[_.]+", _CAMEL.sub("_", name)) if w]


def name_words(name: str) -> frozenset[str]:
    """The set of words of a name: leading underscores, case and unit words ignored; split on ``_``, ``.`` and camel case."""
    return frozenset(w for w in _raw_words(name) if w not in UNIT_WORDS)


def dev_name_forms(names: Iterable[str]) -> dict[str, tuple[frozenset[str], ...]]:
    """The forms every dev constant answers to: its own word set, and the word set without each group prefix.

    A group prefix is a leading run of words (units counted as words) shared by at least ``MIN_GROUP_SIZE`` of the
    constants; the remainder is a form when at least ``MIN_STRIPPED_WORDS`` non-unit words are left.
    """
    words = {n: _raw_words(n) for n in names}
    prefixes: Counter[tuple[str, ...]] = Counter()
    for w in words.values():
        for k in range(1, len(w)):
            prefixes[tuple(w[:k])] += 1
    out: dict[str, tuple[frozenset[str], ...]] = {}
    for n, w in words.items():
        forms = [name_words(n)]
        for k in range(1, len(w)):
            if prefixes[tuple(w[:k])] >= MIN_GROUP_SIZE:
                rest = frozenset(x for x in w[k:] if x not in UNIT_WORDS)
                if len(rest) >= MIN_STRIPPED_WORDS and rest not in forms:
                    forms.append(rest)
        out[n] = tuple(forms)
    return out


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def significant_digits(value: float) -> int:
    """Digits of the shortest decimal repr without leading and trailing zeros (``86400`` -> 3, ``20971520`` -> 7)."""
    text = repr(abs(value)).lower().split("e")[0]
    return len(text.replace(".", "").strip("0"))


def _is_number(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


def fold_number(node: ast.AST) -> int | float | None:
    """A numeric literal, or constant arithmetic over numeric literals with ``+ - * **`` and unary signs; else None."""
    if _is_number(node):
        return node.value  # type: ignore[attr-defined]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = fold_number(node.operand)
        if v is None:
            return None
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Pow)):
        a, b = fold_number(node.left), fold_number(node.right)
        if a is None or b is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if abs(b) > 1000:
                return None
            result = a**b
        except (OverflowError, ZeroDivisionError, ValueError):
            return None
        return result if isinstance(result, (int, float)) else None
    return None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TunableInfo:
    key: str
    default: object | None  # None when the default does not fold (a call, an ambiguous name)
    bound_name: str | None  # the constant the ``register`` call is assigned to, if any
    owner_file: str  # repository-relative file holding the ``register`` call


@dataclass(frozen=True)
class _Setting:
    label: str
    value: object | None
    exact: tuple[frozenset[str], ...]  # R2 forms
    loose: tuple[frozenset[str], ...]  # R3 loose forms (at least two words each)
    owner: str
    names: tuple[str, ...]  # constant names an outside write would use (R1)


class Settings:
    """The settings the checkers guard: dev constants (name -> value) and tunables (key -> TunableInfo)."""

    def __init__(
        self,
        dev: Mapping[str, object],
        tunables: Mapping[str, TunableInfo],
        definition_sites: frozenset[tuple[str, int]] = frozenset(),
        dev_owner: str = DEV_OWNER,
    ) -> None:
        self.dev = dict(dev)
        self.tunables = dict(tunables)
        self.definition_sites = definition_sites
        self.dev_owner = dev_owner
        self.dev_forms = dev_name_forms(self.dev)
        settings: list[_Setting] = []
        for name, value in self.dev.items():
            forms = self.dev_forms[name]
            wide = tuple(f for f in forms if len(f) >= MIN_FORM_WORDS)
            settings.append(_Setting(f"dev_constants.{name}", value, wide, wide, dev_owner, (name,)))
        for key, info in self.tunables.items():
            bound = name_words(info.bound_name) if info.bound_name else frozenset()
            bound = bound if len(bound) >= MIN_FORM_WORDS else frozenset()  # one word names a kind of thing
            segments = key.split(".")
            loose = [bound] if len(bound) >= MIN_FORM_WORDS else []
            loose += [f for f in (name_words(".".join(segments[i:])) for i in range(len(segments))) if len(f) >= MIN_FORM_WORDS]
            value = None if key.startswith("model.") else info.default  # a model id is the model case's (G1)
            settings.append(
                _Setting(f"tunables:{key}", value, (bound,) if bound else (), tuple(dict.fromkeys(loose)), info.owner_file,
                         (info.bound_name,) if info.bound_name else ())
            )
        self.constant_names = frozenset(n for st in settings for n in st.names)
        self.by_exact: dict[frozenset[str], list[_Setting]] = {}
        self.by_number: dict[float, list[_Setting]] = {}
        self.by_string: dict[str, list[_Setting]] = {}
        self.by_collection: dict[frozenset[str], list[_Setting]] = {}
        for st in settings:
            for form in st.exact:
                if form:
                    self.by_exact.setdefault(form, []).append(st)
            v = st.value
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                self.by_number.setdefault(v, []).append(st)
            elif isinstance(v, str):
                if not v.isalpha():  # a plain word is not distinctive
                    self.by_string.setdefault(v, []).append(st)
            elif isinstance(v, (list, tuple, set, frozenset)) and all(isinstance(e, str) for e in v) and len(set(v)) >= 2:
                self.by_collection.setdefault(frozenset(v), []).append(st)


def make_settings(dev: Mapping[str, object] | None = None, tunables: Iterable[TunableInfo] = ()) -> Settings:
    """A ``Settings`` from literal pieces (fixtures)."""
    return Settings(dev or {}, {t.key: t for t in tunables})


def _dev_values(path: Path) -> dict[str, object]:
    """Public scalar and string-collection constants of the dev-constants file, from a private copy of the module."""
    spec = importlib.util.spec_from_file_location("_shared_settings_dev_constants", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out: dict[str, object] = {}
    for name, value in vars(module).items():
        if name.startswith("_") or isinstance(value, (ModuleType, FunctionType, type, bool)):
            continue
        if isinstance(value, (int, float, str)) or (
            isinstance(value, (tuple, list, set, frozenset)) and all(isinstance(e, str) for e in value)
        ):
            out[name] = value
    return out


def _import_aliases(nodes: Iterable[ast.AST]) -> tuple[set[str], set[str]]:
    """Names bound to the ``brain.tunables`` module and to the ``brain.dev_constants`` module in one file."""
    tun, dev = {"tunables"}, set()
    for n in nodes:
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name == "tunables":
                    tun.add(a.asname or a.name)
                elif a.name == "dev_constants":
                    dev.add(a.asname or a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.asname and a.name.endswith("brain.tunables"):
                    tun.add(a.asname)
                elif a.asname and a.name.endswith("brain.dev_constants"):
                    dev.add(a.asname)
    return tun, dev


def _is_register_call(node: ast.AST, tun_mods: set[str], rel: str) -> bool:
    if not isinstance(node, ast.Call) or not node.args:
        return False
    first, f = node.args[0], node.func
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        return False
    if isinstance(f, ast.Attribute):
        return f.attr == "register" and isinstance(f.value, ast.Name) and f.value.id in tun_mods
    return isinstance(f, ast.Name) and f.id == "register" and rel == "brain/tunables.py"


def _fold_default(node: ast.AST, resolve) -> object | None:
    n = fold_number(node)
    if n is not None:
        return n
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
        values = [e.value for e in node.elts]  # type: ignore[attr-defined]
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.Name):
        return resolve(node.id)
    return None


def derive_settings(trees: Mapping[str, ast.Module], dev_path: Path) -> Settings:
    """The settings of a tree: dev constants from the file at ``dev_path``, tunables from every ``register`` call.

    A default that is a Name is resolved one hop to the module-level definition of that name when it is unique in the
    tree (the definition is then the tunable's own default and is not a copy of it); a default that does not fold (a
    call) is ``None`` and the tunable still counts for R1, R2 and the loose names.
    """
    definitions: dict[str, list[tuple[str, int, ast.AST]]] = {}
    for rel, tree in trees.items():
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and stmt.value is not None:
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        definitions.setdefault(t.id, []).append((rel, stmt.lineno, stmt.value))
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                definitions.setdefault(stmt.target.id, []).append((rel, stmt.lineno, stmt.value))
    sites: set[tuple[str, int]] = set()
    tunables: dict[str, TunableInfo] = {}
    for rel, tree in trees.items():
        nodes = list(ast.walk(tree))
        tun_mods, _dev_mods = _import_aliases(nodes)
        bound: dict[int, str] = {}
        for n in nodes:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                bound[id(n.value)] = n.targets[0].id
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                bound[id(n.value)] = n.target.id

        def resolve(name: str):
            found = definitions.get(name, [])
            if len(found) != 1:
                return None
            where, line, value = found[0]
            default = _fold_default(value, lambda _n: None)
            if default is not None:
                sites.add((where, line))
            return default

        for n in nodes:
            if _is_register_call(n, tun_mods, rel):
                key = n.args[0].value  # type: ignore[attr-defined]
                default = _fold_default(n.args[1], resolve) if len(n.args) > 1 else None
                tunables[key] = TunableInfo(key, default, bound.get(id(n)), rel)
    return Settings(_dev_values(dev_path), tunables, frozenset(sites))


# ---------------------------------------------------------------------------
# One parsed file
# ---------------------------------------------------------------------------


class Parsed:
    """A file's tree with a parent map and the import aliases the rules need. Built per file, never stored on the tree."""

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree
        self.nodes = list(ast.walk(tree))
        self.parents: dict[int, ast.AST] = {}
        for n in self.nodes:
            for c in ast.iter_child_nodes(n):
                self.parents[id(c)] = n
        self.aliases: dict[str, str] = {}
        self.from_dev: dict[str, str] = {}  # local name -> dev constant name, for ``from brain.dev_constants import X``
        self.from_tunables: set[str] = set()  # local names imported from brain.tunables
        for n in self.nodes:
            if isinstance(n, ast.ImportFrom):
                last = (n.module or "").split(".")[-1]
                for a in n.names:
                    self.aliases[a.asname or a.name] = a.name
                    if last == "dev_constants":
                        self.from_dev[a.asname or a.name] = a.name
                    elif last == "tunables":
                        self.from_tunables.add(a.asname or a.name)
        self.tun_mods, self.dev_mods = _import_aliases(self.nodes)
        self.docstrings: set[int] = set()
        for n in self.nodes:
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
                first = n.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                    self.docstrings.add(id(first.value))

    def parent(self, node: ast.AST) -> ast.AST | None:
        return self.parents.get(id(node))

    def ancestors(self, node: ast.AST):
        cur = self.parents.get(id(node))
        while cur is not None:
            yield cur
            cur = self.parents.get(id(cur))


def parse(source: str | Parsed) -> Parsed:
    return source if isinstance(source, Parsed) else Parsed(ast.parse(source))


def _call_name(p: Parsed, call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return p.aliases.get(f.id, f.id)
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _is_accessor_call(p: Parsed, node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_name(p, node) in ACCESSORS


def _is_model_accessor_call(p: Parsed, node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_name(p, node) in MODEL_ACCESSORS


def _enclosing_function(p: Parsed, node: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for a in p.ancestors(node):
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return a
        if isinstance(a, ast.ClassDef):
            return None
    return None


def _scope_kind(p: Parsed, node: ast.AST) -> str:
    """'function' when the nearest enclosing scope is a function or lambda, else 'module' / 'class'."""
    for a in p.ancestors(node):
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return "function"
        if isinstance(a, ast.ClassDef):
            return "class"
    return "module"


def _scope_site(p: Parsed, node: ast.AST) -> str:
    for a in p.ancestors(node):
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return f"in {a.name}"
        if isinstance(a, ast.Lambda):
            return "in lambda"
        if isinstance(a, ast.ClassDef):
            return f"in class {a.name}"
    return "in module"


def _in_register_call(p: Parsed, node: ast.AST, rel: str) -> bool:
    return any(_is_register_call(a, p.tun_mods, rel) for a in p.ancestors(node))


def _statement_line(p: Parsed, node: ast.AST) -> int:
    cur: ast.AST | None = node
    while cur is not None and not isinstance(cur, ast.stmt):
        cur = p.parent(cur)
    return getattr(cur, "lineno", 0)


def _target_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _binding(p: Parsed, node: ast.AST) -> tuple[str | None, str]:
    """(the name a value is bound to, the site): an assignment value, a keyword value or a parameter default."""
    parent = p.parent(node)
    if isinstance(parent, ast.Assign) and parent.value is node:
        for t in parent.targets:
            name = _target_name(t)
            if name:
                return name, f"assign {name}"
    elif isinstance(parent, ast.AnnAssign) and parent.value is node:
        name = _target_name(parent.target)
        if name:
            return name, f"assign {name}"
    elif isinstance(parent, ast.keyword) and parent.value is node and parent.arg:
        call = p.parent(parent)
        callee = _call_name(p, call) if isinstance(call, ast.Call) else None
        return parent.arg, f"kw {callee or '?'}.{parent.arg}"
    elif isinstance(parent, ast.arguments):
        for func_name, arg in _defaults_with_params(p, parent):
            if arg[1] is node:
                return arg[0], f"param {func_name}.{arg[0]}"
    return None, _scope_site(p, node)


def _defaults_with_params(p: Parsed, arguments: ast.arguments) -> list[tuple[str, tuple[str, ast.AST]]]:
    """(function name, (parameter name, default node)) for every parameter of a signature that has a default."""
    owner = p.parent(arguments)
    func_name = getattr(owner, "name", "lambda") if owner is not None else "?"
    positional = arguments.posonlyargs + arguments.args
    out: list[tuple[str, tuple[str, ast.AST]]] = []
    for arg, default in zip(positional[len(positional) - len(arguments.defaults):], arguments.defaults, strict=True):
        out.append((func_name, (arg.arg, default)))
    for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        if default is not None:
            out.append((func_name, (arg.arg, default)))
    return out


# ---------------------------------------------------------------------------
# R1 write
# ---------------------------------------------------------------------------


def check_r1(source: str | Parsed, relpath: str, settings: Settings) -> list[Finding]:
    p = parse(source)
    rel = relpath.removeprefix("brain/")
    imported = set(p.from_dev) | p.from_tunables
    out: list[Finding] = []
    for node in p.nodes:
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.attr in settings.constant_names:
            out.append(_finding("R1", relpath, node.lineno, f"attr {node.attr}", f"write to the setting constant {node.attr} from outside its owner"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("setattr", "delattr"):
            name = node.args[1] if len(node.args) > 1 else None
            if isinstance(name, ast.Constant) and name.value in settings.constant_names:
                out.append(_finding("R1", relpath, node.lineno, f"{node.func.id} {name.value}", f"{node.func.id} on the setting constant {name.value}"))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in imported:
            fn = _enclosing_function(p, node)
            rebinding = _scope_kind(p, node) != "function" or (
                fn is not None and any(isinstance(s, ast.Global) and node.id in s.names for s in ast.walk(fn))
            )
            if rebinding:
                out.append(_finding("R1", relpath, node.lineno, f"name {node.id}", f"rebinding of {node.id}, imported from a settings module"))
        elif isinstance(node, ast.Call) and _call_name(p, node) == "set_values" and relpath != "brain/tunables.py":
            fn = _enclosing_function(p, node)
            fn_name = getattr(fn, "name", None)
            if (rel, fn_name) not in SET_VALUES_SITES:
                out.append(_finding("R1", relpath, node.lineno, f"call set_values {_scope_site(p, node)}", "set_values called outside the sanctioned writers"))
    return out


# ---------------------------------------------------------------------------
# R2 name
# ---------------------------------------------------------------------------


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def check_r2(source: str | Parsed, relpath: str, settings: Settings) -> list[Finding]:
    p = parse(source)
    out: list[Finding] = []

    def check(name: str, line: int, site: str) -> None:
        words = name_words(name)
        found = [st for st in settings.by_exact.get(words, []) if st.owner != relpath] if words else []
        if found:
            out.append(_finding("R2", relpath, line, site, f"{name} has the name of a shared setting", (st.label for st in found)))

    for node in p.nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and _scope_kind(p, node) != "function":
            if node.value is None or _is_none(node.value):
                continue  # a bare annotation or a None holds no copy
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    check(t.id, node.lineno, f"assign {t.id}")
        elif isinstance(node, ast.arguments):
            for func_name, (arg, default) in _defaults_with_params(p, node):
                if not _is_none(default):
                    check(arg, getattr(default, "lineno", 0), f"param {func_name}.{arg}")
    return out


# ---------------------------------------------------------------------------
# R3 value
# ---------------------------------------------------------------------------


def _is_str_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _string_collection(node: ast.AST) -> frozenset[str] | None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and node.elts and all(_is_str_constant(e) for e in node.elts):
        elements = frozenset(e.value for e in node.elts)  # type: ignore[attr-defined]
        return elements if len(elements) >= 2 else None
    return None


_WRAPPERS = frozenset({"frozenset", "set", "tuple", "list"})


def check_r3(source: str | Parsed, relpath: str, settings: Settings) -> list[Finding]:
    if relpath == settings.dev_owner:
        return []
    p = parse(source)
    out: list[Finding] = []
    for node in p.nodes:
        if isinstance(node, (ast.Constant, ast.BinOp, ast.UnaryOp)):
            value = fold_number(node)
            if value is not None:
                parent = p.parent(node)
                if isinstance(parent, (ast.BinOp, ast.UnaryOp)) and fold_number(parent) is not None:
                    continue  # a part of a larger constant expression: the maximal one is judged
                _number(p, node, value, relpath, settings, out)
                continue
            if _is_str_constant(node):
                _string(p, node, relpath, settings, out)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Call)):
            _collection(p, node, relpath, settings, out)
    return out


def _exempt(p: Parsed, node: ast.AST, relpath: str, settings: Settings) -> bool:
    return _in_register_call(p, node, relpath) or (relpath, _statement_line(p, node)) in settings.definition_sites


def _number(p: Parsed, node: ast.AST, value: float, relpath: str, settings: Settings, out: list[Finding]) -> None:
    sharing = settings.by_number.get(value)
    if not sharing or _exempt(p, node, relpath, settings):
        return
    bound, site = _binding(p, node)
    if significant_digits(value) >= MIN_SIGNIFICANT_DIGITS:
        out.append(_finding("R3", relpath, node.lineno, site, f"the value {value!r} is also a shared setting's value", (st.label for st in sharing)))
        return
    if bound is None:
        return
    words = name_words(bound)
    matched = [st for st in sharing if st.owner != relpath and any(form <= words for form in st.loose)]
    if matched:
        out.append(_finding("R3-loose", relpath, node.lineno, site, f"{bound} = {value!r} carries a shared setting's value and words", (st.label for st in matched)))


def _string(p: Parsed, node: ast.Constant, relpath: str, settings: Settings, out: list[Finding]) -> None:
    sharing = settings.by_string.get(node.value)
    if not sharing or id(node) in p.docstrings or isinstance(p.parent(node), ast.JoinedStr) or _exempt(p, node, relpath, settings):
        return
    _bound, site = _binding(p, node)
    out.append(_finding("R3", relpath, node.lineno, site, f"the string {node.value[:60]!r} is also a shared setting's value", (st.label for st in sharing)))


def _collection(p: Parsed, node: ast.AST, relpath: str, settings: Settings, out: list[Finding]) -> None:
    if isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id in _WRAPPERS and len(node.args) == 1 and not node.keywords):
            return
        elements = _string_collection(node.args[0])
    else:
        parent = p.parent(node)
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id in _WRAPPERS and parent.args == [node]:
            return  # the wrapping call is the collection
        elements = _string_collection(node)
    sharing = settings.by_collection.get(elements) if elements else None
    if not sharing or _exempt(p, node, relpath, settings):
        return
    _bound, site = _binding(p, node)
    out.append(_finding("R3-collection", relpath, node.lineno, site, "a collection with the same elements as a shared setting", (st.label for st in sharing)))


# ---------------------------------------------------------------------------
# R4 outlives the call, R4b literal default
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


def _returns_accessor(p: Parsed, func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
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


def _accessor_settings(node: ast.Call, name: str) -> tuple[str, ...]:
    if name == "get_tunable" and node.args and _is_str_constant(node.args[0]):
        return (f"tunables:{node.args[0].value}",)  # type: ignore[attr-defined]
    return (f"accessor:{name}",)


def check_r4(source: str | Parsed, relpath: str, settings: Settings) -> list[Finding]:
    p = parse(source)
    out: list[Finding] = []
    returners = {
        n.name
        for n in p.nodes
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _returns_accessor(p, n)
    }

    def flag(node: ast.Call, name: str, why: str) -> None:
        out.append(_finding("R4", relpath, node.lineno, f"{why} {_scope_site(p, node)}", f"{why}: a copy of {name}() that outlives the call", _accessor_settings(node, name)))

    for node in p.nodes:
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(p, node)
        if name in returners and name not in ACCESSORS and _scope_kind(p, node) != "function":
            flag(node, name, f"module or class scope call of {name}, which returns an accessor result")
        if name not in ACCESSORS:
            continue
        scope = _scope_kind(p, node)
        if scope != "function":
            flag(node, name, f"call at {scope} scope")
        for a in p.ancestors(node):
            if isinstance(a, ast.arguments):
                flag(node, name, "call in a parameter default")
                break
            if isinstance(a, ast.Call) and a is not node:
                cn = _call_name(p, a)
                if cn in SINK_CALL_NAMES and (node in a.args or any(k.value is node for k in a.keywords)):
                    flag(node, name, f"result passed to {cn}()")
                if cn == "field":
                    flag(node, name, "call inside dataclass field()")
            if isinstance(a, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = a.targets if isinstance(a, ast.Assign) else [a.target]
                for t in targets:
                    if isinstance(t, (ast.Attribute, ast.Subscript)):
                        flag(node, name, "result stored in an attribute or subscript target")
                    if isinstance(t, ast.Name):
                        fn = _enclosing_function(p, a)
                        if fn is not None:
                            for s in ast.walk(fn):
                                if isinstance(s, (ast.Global, ast.Nonlocal)) and t.id in s.names:
                                    flag(node, name, "result stored in a global or nonlocal name")
                break
            if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)) and _decorated_with_cache(a):
                flag(node, name, "call inside a cached function")
    return out


def check_r4b(source: str | Parsed, relpath: str, settings: Settings) -> list[Finding]:
    p = parse(source)
    out: list[Finding] = []
    for node in p.nodes:
        if not (isinstance(node, ast.Call) and _call_name(p, node) == "get_tunable"):
            continue
        keywords = {k.arg: k.value for k in node.keywords if k.arg}
        default = node.args[1] if len(node.args) > 1 else keywords.get("default")
        if default is None:
            continue
        if fold_number(default) is not None or _is_str_constant(default):
            out.append(_finding("R4b", relpath, node.lineno, f"get_tunable default {_scope_site(p, node)}",
                                "a literal where the registered default constant belongs", _accessor_settings(node, "get_tunable")))
    return out


# ---------------------------------------------------------------------------
# R5 alias and default
# ---------------------------------------------------------------------------


def _dev_ref(p: Parsed, node: ast.AST | None, settings: Settings) -> str | None:
    """The dev constant a node reads, when the node IS exactly that read (``dev_constants.NAME`` or an imported name)."""
    if isinstance(node, ast.Attribute) and node.attr in settings.dev:
        base = node.value
        if isinstance(base, ast.Name) and base.id in p.dev_mods:
            return node.attr
        if isinstance(base, ast.Attribute) and base.attr == "dev_constants":
            return node.attr
    if isinstance(node, ast.Name) and p.from_dev.get(node.id) in settings.dev:
        return p.from_dev[node.id]
    return None


def check_r5(source: str | Parsed, relpath: str, settings: Settings) -> list[Finding]:
    if relpath == settings.dev_owner:
        return []
    p = parse(source)
    out: list[Finding] = []
    for node in p.nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None and _scope_kind(p, node) != "function":
            ref = _dev_ref(p, node.value, settings)
            if ref:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                name = next((n for n in (_target_name(t) for t in targets) if n), "?")
                out.append(_finding("R5a", relpath, node.lineno, f"assign {name}", f"{name} is a second name for dev_constants.{ref}", (f"dev_constants.{ref}",)))
        elif isinstance(node, ast.arguments):
            for func_name, (arg, default) in _defaults_with_params(p, node):
                ref = _dev_ref(p, default, settings)
                if ref:
                    out.append(_finding("R5b", relpath, getattr(default, "lineno", 0), f"param {func_name}.{arg}", f"{arg} defaults to dev_constants.{ref}, frozen when the function is defined", (f"dev_constants.{ref}",)))
    return out


# ---------------------------------------------------------------------------
# The model case (#529): G1, G2, G4, G5
# ---------------------------------------------------------------------------


def _shape_of(p: Parsed, node: ast.Constant) -> str | None:
    parent = p.parent(node)
    grand = p.parent(parent) if parent is not None else None
    # tunables.register("key", <default>)
    if isinstance(parent, ast.Call) and _call_name(p, parent) == "register" and node in parent.args[1:2]:
        return "register_default"
    if isinstance(parent, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OPUS_CHOICE" for t in parent.targets):
        return "opus_choice"
    for a in p.ancestors(node):
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
    parent = p.parent(node)
    if isinstance(parent, ast.JoinedStr):
        return True
    if isinstance(parent, ast.BinOp) and isinstance(parent.op, (ast.Add, ast.Mod)):
        return True
    return isinstance(parent, ast.Attribute) and parent.attr == "format" and isinstance(p.parent(parent), ast.Call)


def _is_model_literal(value: str) -> bool:
    return value in ALIAS_WORDS or bool(ID_FULL.fullmatch(value)) or bool(ID_PROSE.search(value))


def check_g1(
    source: str | Parsed, relpath: str, allow: Sequence[tuple[str, str, str]] = MODEL_LITERAL_ALLOW, used: set | None = None
) -> list[Finding]:
    """G1: no model literal. Allow-list entries are ``(file relative to brain/, literal, shape)``."""
    p = parse(source)
    rel = relpath.removeprefix("brain/")
    out: list[Finding] = []
    for node in p.nodes:
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)) or id(node) in p.docstrings:
            continue
        if _is_assembly_piece(p, node):
            out.append(_finding("G1", relpath, node.lineno, f"id assembled from pieces {node.value[:60]!r}", "model id assembled from pieces"))
            continue
        if not _is_model_literal(node.value):
            continue
        shape = _shape_of(p, node)
        entry = (rel, node.value, shape)
        if shape is not None and entry in allow:
            if used is not None:
                used.add(entry)
            continue
        out.append(_finding("G1", relpath, node.lineno, f"model literal {node.value[:60]!r} shape {shape}", f"model literal {node.value[:60]!r}"))
    return out


def _is_sanctioned_value(p: Parsed, value: ast.AST, scope_node: ast.AST) -> bool:
    """A value that may follow ``--model``: a result of a MODEL accessor (not any ``get_tunable`` result), directly or
    through a local that is only ever assigned one."""
    if _is_model_accessor_call(p, value):
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
        return bool(vals) and all(_is_model_accessor_call(p, v) for v in vals)
    return False


def check_g2(source: str | Parsed, relpath: str) -> list[Finding]:
    """G2: no model at an argv sink without the accessor."""
    p = parse(source)
    out: list[Finding] = []

    def add(node: ast.AST, site: str, text: str) -> None:
        out.append(_finding("G2", relpath, node.lineno, site, text))

    for node in p.nodes:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            parent = p.parent(node)
            if v == "--model":
                if isinstance(parent, (ast.List, ast.Tuple)):
                    idx = parent.elts.index(node)
                    if idx + 1 >= len(parent.elts) or not _is_sanctioned_value(p, parent.elts[idx + 1], node):
                        add(node, f"--model value {_scope_site(p, node)}", "--model value is not an accessor result")
                elif (
                    isinstance(parent, ast.Call)
                    and isinstance(parent.func, ast.Attribute)
                    and parent.func.attr in ("append", "extend", "insert")
                ):
                    add(node, f"--model appended alone {_scope_site(p, node)}", "--model appended on its own (unverifiable form)")
                # anything else (``cmd.index("--model")``, an argparse option, a comparison) reads or declares the flag
            elif v.startswith("--model="):
                add(node, f"--model= {_scope_site(p, node)}", "single-token --model=<x> form")
            elif v == "--fallback-model":
                add(node, f"--fallback-model {_scope_site(p, node)}", "--fallback-model")
            elif ENV_MODEL_NAMES.fullmatch(v):
                add(node, f"environment variable {v}", f"model environment variable {v}")
    return out


def check_g4(source: str | Parsed, relpath: str) -> list[Finding]:
    """G4: no constructor bypass (``model=`` on the CLI provider, ``probe_model=`` outside the one probe, retired names)."""
    p = parse(source)
    rel = relpath.removeprefix("brain/")
    out: list[Finding] = []

    def add(node: ast.AST, site: str, text: str) -> None:
        out.append(_finding("G4", relpath, getattr(node, "lineno", 0), site, text))

    for node in p.nodes:
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
                add(node, f"retired identifier {ident}", f"retired identifier {ident}")
        if isinstance(node, ast.Call) and _call_name(p, node) == "ClaudeCliProvider":
            for kw in node.keywords:
                if kw.arg == "model":
                    add(node, "ClaudeCliProvider(model=...)", "ClaudeCliProvider(model=...)")
                if kw.arg == "probe_model":
                    fn = _enclosing_function(p, node)
                    if (rel, getattr(fn, "name", None)) != PROBE_SITE:
                        add(node, f"probe_model= {_scope_site(p, node)}", "probe_model= outside the one probe site")
        if isinstance(node, ast.FunctionDef) and node.name == "get_provider":
            names = {a.arg for a in node.args.args + node.args.kwonlyargs}
            for bad in ("model_override", "persona_dir"):
                if bad in names:
                    add(node, f"get_provider parameter {bad}", f"get_provider has a {bad} parameter")
    return out


def check_g5(source: str | Parsed, relpath: str) -> list[Finding]:
    """G5: key ownership, no ``model.`` key literal outside model_tier.py (``set_values`` ownership is R1d)."""
    p = parse(source)
    out: list[Finding] = []
    if relpath.removeprefix("brain/") == "bridge/model_tier.py":
        return out
    for node in p.nodes:
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("model.") and id(node) not in p.docstrings:
            out.append(_finding("G5", relpath, node.lineno, f"model key literal {node.value!r}", f"model. key literal {node.value!r} outside model_tier.py"))
    return out


def check_all(source: str | Parsed, relpath: str, settings: Settings, used: set | None = None) -> list[Finding]:
    p = parse(source)
    return [
        *check_g1(p, relpath, used=used),
        *check_g2(p, relpath),
        *check_g4(p, relpath),
        *check_g5(p, relpath),
        *check_r1(p, relpath, settings),
        *check_r2(p, relpath, settings),
        *check_r3(p, relpath, settings),
        *check_r4(p, relpath, settings),
        *check_r4b(p, relpath, settings),
        *check_r5(p, relpath, settings),
    ]


# ---------------------------------------------------------------------------
# The app: TypeScript and Rust
# ---------------------------------------------------------------------------


def _lex(source: str, language: str) -> tuple[str, list[tuple[int, str]]]:
    """(the source with comments and string contents blanked, newlines kept; the quoted string tokens ``(line, text)``).

    Rust is cut at the first ``#[cfg(test)]``; its strings are double-quoted only, and a char literal is skipped.
    """
    rust = language == "rust"
    if rust:
        cut = source.find("#[cfg(test)]")
        source = source if cut == -1 else source[:cut]
    quotes = '"' if rust else "'\"`"
    code: list[str] = []
    tokens: list[tuple[int, str]] = []
    i, n, line = 0, len(source), 1

    def blank(chunk: str) -> None:
        code.extend("\n" if c == "\n" else " " for c in chunk)

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if ch == "\n":
            line += 1
            code.append("\n")
            i += 1
        elif ch == "/" and nxt == "/":
            j = i
            while j < n and source[j] != "\n":
                j += 1
            blank(source[i:j])
            i = j
        elif ch == "/" and nxt == "*":
            end = source.find("*/", i + 2)
            end = n if end == -1 else end + 2
            blank(source[i:end])
            line += source.count("\n", i, end)
            i = end
        elif ch in quotes:
            quote, j, buf, start_line = ch, i + 1, [], line
            while j < n and source[j] != quote:
                if source[j] == "\\" and j + 1 < n:
                    buf.append(source[j + 1])
                    j += 2
                    continue
                if source[j] == "\n":
                    line += 1
                    if quote != "`" and not rust:
                        break
                buf.append(source[j])
                j += 1
            tokens.append((start_line, "".join(buf)))
            blank(source[i : j + 1])
            i = j + 1
        elif rust and ch == "'" and re.match(r"'(\\.|[^'\\])'", source[i : i + 4]):
            width = 3 if source[i + 1] != "\\" else 4
            blank(source[i : i + width])
            i += width
        else:
            code.append(ch)
            i += 1
    return "".join(code), tokens


def ts_tokens(source: str) -> list[tuple[int, str]]:
    """The quoted string tokens of TypeScript/TSX source, comments removed: ``(line, content)``."""
    return _lex(source, "ts")[1]


def rust_tokens(source: str) -> list[tuple[int, str]]:
    """String literals of Rust source before the first ``#[cfg(test)]``, comments removed."""
    return _lex(source, "rust")[1]


_APP_NUMBER = re.compile(r"(?<![\w.])(\d[\d_]*(?:\.\d+)?(?:\s*\*\s*\d[\d_]*(?:\.\d+)?)*)(?![\w.])")


def app_numbers(source: str, language: str) -> list[tuple[int, int | float]]:
    """``(line, value)`` of every maximal numeric expression of app source: digits, ``_`` separators, ``*`` products.

    Comments and string contents are not read (a number in prose is not a copy); Rust is cut at ``#[cfg(test)]``. Hex,
    shifts, ``**``, exponents and integer suffixes do not match (see the module docstring).
    """
    code, _tokens = _lex(source, language)
    out: list[tuple[int, int | float]] = []
    for m in _APP_NUMBER.finditer(code):
        value: int | float = 1
        for part in m.group(1).split("*"):
            text = part.strip().replace("_", "")
            value *= float(text) if "." in text else int(text)
        out.append((code.count("\n", 0, m.start()) + 1, value))
    return out


def app_findings(source: str, relpath: str, language: str, settings: Settings, alias_allowed: bool) -> list[Finding]:
    """APP-R3 (a distinctive number or string equal to a setting's value) and APP-MODEL (model tokens) in one app file."""
    out: list[Finding] = []
    for line, value in app_numbers(source, language):
        sharing = settings.by_number.get(value)
        if sharing and significant_digits(value) >= MIN_SIGNIFICANT_DIGITS:
            shown = repr(int(value)) if float(value).is_integer() else repr(value)
            out.append(_finding("APP-R3", relpath, line, f"number {shown}", f"the number {shown} is also a shared setting's value", (st.label for st in sharing)))
    for line, text in _lex(source, language)[1]:
        sharing = settings.by_string.get(text)
        if sharing:
            out.append(_finding("APP-R3", relpath, line, f"string {text[:60]!r}", f"the string {text[:60]!r} is also a shared setting's value", (st.label for st in sharing)))
        if ID_FULL.fullmatch(text) or ID_PROSE.search(text):
            out.append(_finding("APP-MODEL", relpath, line, f"model id {text[:60]!r}", f"model id in a string token {text[:60]!r}"))
        elif text in ALIAS_WORDS and not alias_allowed:
            out.append(_finding("APP-MODEL", relpath, line, f"alias word {text!r}", f"model alias word {text!r} outside the picker vocabulary"))
    return out


def _app_files() -> Iterable[tuple[str, str, str]]:
    for rel, text in tree_sources("app/src", (".ts", ".tsx"), ("node_modules",)).items():
        if ".test." not in rel.rsplit("/", 1)[-1]:
            yield rel, text, "ts"
    for rel, text in tree_sources("app/src-tauri/src", (".rs",)).items():
        if "/" not in rel.removeprefix("app/src-tauri/src/"):
            yield rel, text, "rust"


# ---------------------------------------------------------------------------
# The real tree and the ratchet
# ---------------------------------------------------------------------------


def scan_real_tree(settings: Settings) -> tuple[list[Finding], frozenset[tuple[str, str, str]]]:
    """Every rule over ``brain/*.py`` and the app: the findings, and the model vocabulary entries that matched."""
    findings: list[Finding] = []
    used: set[tuple[str, str, str]] = set()
    for rel, tree in tree_asts("brain").items():
        findings += check_all(Parsed(tree), rel, settings, used=used)
    for rel, text, language in _app_files():
        findings += app_findings(text, rel, language, settings, alias_allowed=rel in APP_ALIAS_ALLOW)
    return findings, frozenset(used)


def ratchet_problems(
    findings: Sequence[Finding], allow: Sequence[tuple[str, str, str, tuple[str, ...], str]], allow_size: int
) -> list[str]:
    """What is wrong between the findings and the shrink-only list ``allow`` (entries ``(rule, file, site, settings, reason)``).

    The findings must equal the entries as a multiset; every entry carries a reason; ``allow_size`` equals ``len(allow)``.
    """
    problems = [f"ALLOW entry has no reason: {e[:4]}" for e in allow if not str(e[4]).strip()]
    wanted = Counter(tuple(e[:4]) for e in allow)
    seen = Counter(f.key for f in findings)
    unlisted = seen - wanted
    for f in findings:
        if unlisted.get(f.key, 0) > 0:
            unlisted[f.key] -= 1
            problems.append(
                f"{f.message}\n    rule {f.rule}, file {f.file}, site {f.site!r}, settings {list(f.settings)}: this finding is not in ALLOW. "
                "Remove the copy (read the setting through its accessor); if it is a different thing with an equal value, add an "
                "ALLOW entry with the reason and raise ALLOW_SIZE, which needs review."
            )
    for key, count in (wanted - seen).items():
        problems += [f"stale ALLOW entry {key}: nothing is found there any more, delete it (and lower ALLOW_SIZE)"] * count
    if len(allow) != allow_size:
        if len(allow) > allow_size:
            problems.append(
                f"ALLOW grew: it has {len(allow)} entries and ALLOW_SIZE is {allow_size}. The list may only shrink; growth needs review"
            )
        else:
            problems.append(f"ALLOW shrank: it has {len(allow)} entries and ALLOW_SIZE is {allow_size}; lower ALLOW_SIZE to {len(allow)}")
    return problems
