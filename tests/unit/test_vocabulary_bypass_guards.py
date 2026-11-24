"""Guards that stop a new reader or writer bypassing the shared ones for her vocabulary (#803 step 17).

The owner's standing rule: "All file access go through one of the existing shared accessors, no exceptions." The shared pieces are
`brain/emotion/vocabulary_file.py` (the one writer, the one path function `vocabulary_path`), `vocabulary.py` (the registry, the
one tidier `canonical_name`, the one name `VOCABULARY_FILENAME`) and `MemoryStore`'s name check. Each guard below fails when the
bypass it names is added, and each has a self-test on synthetic source that shows it can fail:

* A. the file's name is spelled once in `brain/` (`vocabulary.VOCABULARY_FILENAME`).
* B. every use of the path function or the name constant outside the two vocabulary modules is pinned to its exact stripped line
  with a reason (how the site gets the path), so a new reader or writer shows up by where its path comes from, whatever it calls.
* D. the in-process registry and the copy-following state are used only by the modules that own them.
* E. a second tidier of feeling names (a new hit fails; today's hits are pinned with the reason each is not a feeling name).

Pins are `(file relative to brain/, stripped line) -> (count, reason)`. A use that is not pinned fails, a line that appears more
often than its count fails (a copy of a pinned line is a new use), an entry with fewer hits than its count fails (stale), and an
entry with no reason fails.

What these cannot catch is stated beside each scanner and in step17-decisions.md: a name built from pieces, a module passed around
under another name, `importlib`, and anything outside `brain/` (tests and scripts are not scanned).
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping

import pytest

from tests.source_tree import tree_asts, tree_sources

VOCABULARY = "brain/emotion/vocabulary.py"
VOCABULARY_FILE = "brain/emotion/vocabulary_file.py"
PERSONA_LOADER = "brain/emotion/persona_loader.py"
FILENAME = "emotion_vocabulary.json"

Pins = Mapping[tuple[str, str], tuple[int, str]]


# ------------------------------------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------------------------------------


def _line(sources: Mapping[str, str], rel: str, lineno: int) -> str:
    return sources[rel].splitlines()[lineno - 1].strip()


def _keyed(sources: Mapping[str, str], hits: Iterable[tuple[str, int]]) -> Counter[tuple[str, str]]:
    """Count hits by (file relative to brain/, stripped source line of the hit)."""
    return Counter((rel.removeprefix("brain/"), _line(sources, rel, lineno)) for rel, lineno in hits)


def pin_problems(found: Mapping[tuple[str, str], int], pins: Pins) -> list[str]:
    """What is wrong between the hits found and the pins: an unpinned hit, a count above the pin, a stale pin, a pin without a reason."""
    problems: list[str] = []
    for key, n in sorted(found.items()):
        if key not in pins:
            problems.append(f"not pinned: {key[0]}: {key[1]}")
        elif n > pins[key][0]:
            problems.append(f"{n} hits but pinned for {pins[key][0]}: {key[0]}: {key[1]}")
    for key, (count, reason) in sorted(pins.items()):
        if found.get(key, 0) < count:
            problems.append(f"stale pin (found {found.get(key, 0)}, pinned {count}): {key[0]}: {key[1]}")
        if not reason.strip():
            problems.append(f"pin without a reason: {key[0]}: {key[1]}")
    return problems


def _parse(sources: Mapping[str, str]) -> dict[str, ast.Module]:
    return {rel: ast.parse(text) for rel, text in sources.items()}


def _docstring_nodes(tree: ast.Module) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


# ------------------------------------------------------------------------------------------------
# A. one spelling of the file's name
# ------------------------------------------------------------------------------------------------


_NAME_TOKEN = re.compile(r"(?:^|[/\\])" + re.escape(FILENAME))


def literal_sites(trees: Mapping[str, ast.Module], defining: frozenset[str]) -> list[tuple[str, int]]:
    """Every string constant that is the file's name as a path token (it starts with the name, or the name follows `/` or `\\`, an
    f-string's literal part; a suffix such as `.bak` or `.tmp` still counts), outside `defining` and outside docstrings. Log text that
    mentions the name has spaces and is not a path token; `nell_emotion_vocabulary.json` (a legacy OG file) is another name.

    Not caught: the name built with `+` or a format from pieces that are not themselves path tokens, `Path.glob("*vocabulary*")`, the
    stem `"emotion_vocabulary"` alone, a name read from config. Two literals written side by side are one constant to the parser, so
    that shape is caught.
    """
    hits: list[tuple[str, int]] = []
    for rel, tree in trees.items():
        if rel in defining:
            continue
        docs = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                if _NAME_TOKEN.search(node.value) and not any(c.isspace() for c in node.value):
                    hits.append((rel, node.lineno))
    return hits


@pytest.mark.no_base_failure(reason="a ratchet over the whole tree: the literals it flags were moved to the one constant in the same change, so it guards against a new one and is not a fix")
def test_the_file_name_is_spelled_once_in_brain() -> None:
    hits = literal_sites(tree_asts("brain"), frozenset({VOCABULARY}))
    assert not hits, (
        f'the literal "{FILENAME}" is spelled outside brain/emotion/vocabulary.py: '
        + "; ".join(f"{rel}:{n}" for rel, n in hits)
        + ". Use vocabulary_file.vocabulary_path(persona_dir) for a path or vocabulary.VOCABULARY_FILENAME for the name."
    )


@pytest.mark.no_base_failure(reason="a self-test of the scanner on synthetic source; it does not read the product tree")
def test_a_scanner_flags_the_literal_in_each_shape_and_ignores_prose() -> None:
    trees = _parse(
        {
            "brain/emotion/vocabulary.py": f'NAME = "{FILENAME}"\n',
            "brain/a.py": f'def f(d):\n    return d / "{FILENAME}"\n',
            "brain/b.py": f'def f(d):\n    return f"{{d}}/{FILENAME}"\n',
            "brain/c.py": 'def f(d):\n    return d / "emotion_" "vocabulary.json"\n',
            "brain/d.py": f'def f(d):\n    return {{"{FILENAME}": 1}}\n',
            "brain/d2.py": f'def f(d):\n    return d / "{FILENAME}.bak"\n',
            "brain/e.py": f'def f():\n    """About {FILENAME}."""\n    log("{FILENAME} was missing")\n    return "nell_{FILENAME}"\n',
        }
    )
    flagged = {rel for rel, _n in literal_sites(trees, frozenset({VOCABULARY}))}
    assert flagged == {"brain/a.py", "brain/b.py", "brain/c.py", "brain/d.py", "brain/d2.py"}


# ------------------------------------------------------------------------------------------------
# B. every use of the path function or the constant is pinned
# ------------------------------------------------------------------------------------------------

_USE_NAMES = frozenset({"vocabulary_path", "VOCABULARY_FILENAME"})


def path_use_sites(trees: Mapping[str, ast.Module], exempt: frozenset[str]) -> list[tuple[str, int]]:
    """Every use of `vocabulary_path` or `VOCABULARY_FILENAME` outside `exempt`: a name, an attribute, an import alias, or the name as
    the string of a `getattr`. A listed site is where a path to her file is made, so a new reader or writer shows up here.

    Not caught: a path made from `persona_dir / name` where `name` comes from a loop over the persona's files (A catches a literal;
    a scan of the folder opens every file), and a path handed in as a parameter by a pinned caller (the pin names what the caller does).
    """
    hits: list[tuple[str, int]] = []
    for rel, tree in trees.items():
        if rel in exempt:
            continue
        for node in ast.walk(tree):
            if (
                (isinstance(node, ast.Name) and node.id in _USE_NAMES)
                or (isinstance(node, ast.Attribute) and node.attr in _USE_NAMES)
                or (isinstance(node, ast.alias) and node.name in _USE_NAMES)
                or (isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in _USE_NAMES)
            ):
                hits.append((rel, node.lineno))
    return hits


_LOAD = "hands the path to load_persona_vocabulary (the loader), which reads her file through vocabulary_file / json_io"
_IMPORT = "imports the name that the pinned use in the same file needs"
_NAME_LIST = "the file's name in a list of persona files (names only, nothing is opened through it here)"

# (file relative to brain/, stripped line) -> (count, reason)
PINNED_USES: dict[tuple[str, str], tuple[int, str]] = {
    ("bridge/server.py", "path = vocabulary_file.vocabulary_path(persona_dir)"): (
        1,
        "the bridge's start-up: the path goes to load_persona_vocabulary_with_anomaly (the loader, the owner's load)",
    ),
    ("bridge/supervisor.py", "from brain.emotion.vocabulary_file import vocabulary_path"): (2, _IMPORT),
    ("bridge/supervisor.py", "load_persona_vocabulary(vocabulary_path(persona_dir), store=store)"): (2, "heartbeat and soul-review: " + _LOAD),
    ("cli.py", "VOCABULARY_FILENAME,"): (1, "the walker's file list of `nell health check` (" + _NAME_LIST + ")"),
    ("cli.py", "from brain.emotion.vocabulary import VOCABULARY_FILENAME"): (1, _IMPORT),
    ("cli.py", "load_persona_vocabulary(vocabulary_file.vocabulary_path(persona_dir), read_only=True)"): (
        1,
        "`nell interest list`: " + _LOAD + " (read-only: no repair, no write)",
    ),
    ("cli.py", "vocabulary_file.remove_core_entries(vocabulary_file.vocabulary_path(persona_dir))"): (
        1,
        "the rebuild step's removal of core entries: the shared writer's own function (spec section 9)",
    ),
    ("emotion/engine.py", "refresh(vocabulary_file.vocabulary_path(self.persona_dir))"): (
        1,
        "builds the path for `persona_loader.refresh_if_changed`, which does not exist, so `_refresh_vocabulary` is a no-op that reads nothing "
        "(BYCATCH medium, step 15: the dead hook can go)",
    ),
    ("emotion/vocabulary_access.py", "persona_loader.adopt_copy(vocabulary_file.vocabulary_path(persona_dir))"): (
        1,
        "the no-bridge read-only path: the loader adopts the copy of her file",
    ),
    ("emotion/vocabulary_access.py", "written = vocabulary_file.add(vocabulary_file.vocabulary_path(persona_dir), [kept[i] for i in valid_at])"): (
        1,
        "the owner's add: the shared writer's own function",
    ),
    ("growth/scheduler.py", "vocab_path = vocabulary_file.vocabulary_path(persona_dir)"): (
        1,
        "the growth tick's health read of her file: `vocabulary_file.read` (the shared reader), owner only",
    ),
    ("health/alarm.py", "VOCABULARY_FILENAME,"): (1, "the set of identity files whose reset raises an alarm (" + _NAME_LIST + ")"),
    ("health/alarm.py", "from brain.emotion.vocabulary import VOCABULARY_FILENAME"): (1, _IMPORT),
    ("health/vocab_repair.py", "vocab_path = vocabulary_file.vocabulary_path(persona_dir)"): (
        1,
        "the vocabulary repair's read: json_io `read_json` with `Recovery.CONTENT` (no repair, no write); every change goes through vocabulary_file",
    ),
    ("health/vocab_repair.py", "vocabulary_file.vocabulary_path(persona_dir),"): (
        1,
        "the placeholder check's read: json_io `read_json` with `Recovery.CONTENT` (no repair, no write)",
    ),
    ("health/walker.py", 'VOCABULARY_FILENAME: {"version": 1, "emotions": []},'): (1, "the walker's table of default file contents (" + _NAME_LIST + ")"),
    ("health/walker.py", "from brain.emotion.vocabulary import VOCABULARY_FILENAME"): (1, _IMPORT),
    ("health/walker.py", "if name == VOCABULARY_FILENAME:"): (
        1,
        "picks her file out of the walker's list: it goes to `_read_vocabulary`, which repairs only in the owner, through vocabulary_file",
    ),
    ("mcp_server/__init__.py", "from brain.emotion.vocabulary_file import vocabulary_path"): (1, _IMPORT),
    ("mcp_server/__init__.py", "load_persona_vocabulary(vocabulary_path(persona_dir), read_only=True)"): (
        1,
        "the MCP server's start: " + _LOAD + " (read-only: no repair, no write)",
    ),
    ("migrator/cli.py", "vocab_target = vocabulary_file.vocabulary_path(work_dir)"): (
        1,
        "the migrator's first vocabulary file: written by `vocabulary_file.add` (the shared writer) before any memory is created",
    ),
    ("recovery/engine.py", 'for name in ("memories.db", "hebbian.db", VOCABULARY_FILENAME):'): (
        1,
        "the recover backup copies her file byte for byte with shutil.copy2: this site READS THE FILE ITSELF, not through vocabulary_file or "
        "json_io (known, behaviour unchanged here; recorded in step17-decisions.md for Planning)",
    ),
    ("recovery/engine.py", "from brain.emotion.vocabulary import VOCABULARY_FILENAME"): (1, _IMPORT),
}


def _brain_sources() -> Mapping[str, str]:
    return tree_sources("brain")


@pytest.mark.no_base_failure(reason="a ratchet over the whole tree: on the old code every site it pins already existed, so it is a guard against new ones and is not a fix")
def test_every_use_of_the_path_function_and_the_name_is_pinned() -> None:
    sources = _brain_sources()
    hits = path_use_sites(tree_asts("brain"), frozenset({VOCABULARY, VOCABULARY_FILE}))
    problems = pin_problems(_keyed(sources, hits), PINNED_USES)
    assert not problems, (
        "a use of vocabulary_path / VOCABULARY_FILENAME is new or gone. A new one is a reader or writer of her file (or a list of persona "
        "files): go through vocabulary_file / vocabulary_access, or pin it with a reason in PINNED_USES:\n  " + "\n  ".join(problems)
    )


@pytest.mark.no_base_failure(reason="a self-test of the scanner and the pin check on synthetic source; it does not read the product tree")
def test_b_scanner_finds_each_way_to_make_the_path_and_the_pin_check_fails_on_each_drift() -> None:
    sources = {
        "brain/emotion/vocabulary_file.py": "def vocabulary_path(d):\n    return d\n",
        "brain/a.py": "from brain.emotion import vocabulary_file\ndef f(d):\n    return vocabulary_file.vocabulary_path(d)\n",
        "brain/b.py": "from brain.emotion.vocabulary_file import vocabulary_path as vp\n",
        "brain/c.py": 'def f(m):\n    return getattr(m, "vocabulary_path")\n',
        "brain/d.py": "from brain.emotion.vocabulary import VOCABULARY_FILENAME\nNAMES = [VOCABULARY_FILENAME]\n",
        "brain/e.py": "def f(d):\n    return d / 'other.json'\n",
    }
    hits = path_use_sites(_parse(sources), frozenset({VOCABULARY_FILE}))
    assert {rel for rel, _n in hits} == {"brain/a.py", "brain/b.py", "brain/c.py", "brain/d.py"}
    found = _keyed(sources, hits)
    pinned = {key: (n, "reason") for key, n in found.items()}
    assert pin_problems(found, pinned) == []
    # a use that is not pinned, a second copy of a pinned line, a stale pin, a pin without a reason: each is a problem
    unpinned = dict(pinned)
    unpinned.pop(("a.py", "return vocabulary_file.vocabulary_path(d)"))
    assert any(p.startswith("not pinned") for p in pin_problems(found, unpinned))
    doubled = Counter(found)
    doubled[("a.py", "return vocabulary_file.vocabulary_path(d)")] += 1
    assert any("hits but pinned for" in p for p in pin_problems(doubled, pinned))
    assert any(p.startswith("stale pin") for p in pin_problems({}, pinned))
    blank = dict(pinned)
    blank[("a.py", "return vocabulary_file.vocabulary_path(d)")] = (1, " ")
    assert any(p.startswith("pin without a reason") for p in pin_problems(found, blank))


# ------------------------------------------------------------------------------------------------
# D. the registry and the copy-following state belong to the modules that own them
# ------------------------------------------------------------------------------------------------

# Public names of vocabulary.py that only some modules may use: name -> the files that may. (Every underscore name of vocabulary.py
# and of vocabulary_file.py is private to its own file and the vocabulary modules, computed from the files, not listed here.)
_COPY_STATE_FILES = frozenset({VOCABULARY, VOCABULARY_FILE, PERSONA_LOADER})
_REGISTRY_FILES = frozenset({VOCABULARY, VOCABULARY_FILE})
VOCABULARY_RESTRICTED_PUBLIC: dict[str, frozenset[str]] = {
    "register": _REGISTRY_FILES,
    "replace_entry": _REGISTRY_FILES,
    "set_file_copy": _COPY_STATE_FILES,
    "clear_file_copy": _COPY_STATE_FILES,
    "note_copy_key": _COPY_STATE_FILES,
    "copy_matches": _COPY_STATE_FILES,
    "followed_path": _COPY_STATE_FILES,
    "suspended": _COPY_STATE_FILES,
    "set_bridge_owner": frozenset({VOCABULARY, "brain/bridge/server.py", "brain/migrator/cli.py"}),
}
# Public names anyone may use: reads, the tidier and the gate, the owner test, the follow-the-file read hook.
VOCABULARY_OPEN_PUBLIC = frozenset(
    {
        "Emotion",
        "EmotionCategory",
        "VOCABULARY_FILENAME",
        "logger",
        "canonical_name",
        "gate_name",
        "pool_by_canonical_name",
        "is_bridge_owner",
        "follow_file",
        "get",
        "list_all",
        "by_category",
        "is_baseline",
    }
)


def top_level_names(tree: ast.Module) -> set[str]:
    """The names a module defines at its top level: functions, classes and assignment targets (not imports)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _private(names: Iterable[str]) -> set[str]:
    return {n for n in names if n.startswith("_") and not (n.startswith("__") and n.endswith("__"))}


def module_misuse(
    trees: Mapping[str, ast.Module], module: str, allowed_for: Callable[[str], frozenset[str] | None]
) -> list[tuple[str, int, str]]:
    """Every use of a restricted name of the module `module` (its last dotted component, `vocabulary` or `vocabulary_file`) by a
    file that may not use it: `(file, line, name)`. `allowed_for(name)` gives the files that may use the name, or None when it is open.

    The module is recognised wherever it is named: `from brain.emotion import vocabulary [as v]`, `import brain.emotion.vocabulary as
    v`, a relative import, any attribute called `vocabulary` (`emotion.vocabulary.x`), and the names imported from it, with `getattr`
    and a literal name as well. A `*` import from it always fails.

    Not caught: the module bound to another name by assignment (`v2 = vocabulary`), `importlib` / `__import__`, a module passed in as
    a parameter, and the tests (not scanned).
    """
    hits: list[tuple[str, int, str]] = []

    def check(rel: str, lineno: int, name: str) -> None:
        allowed = allowed_for(name)
        if allowed is not None and rel not in allowed:
            hits.append((rel, lineno, name))

    for rel, tree in trees.items():
        aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                aliases.update(a.asname for a in node.names if a.asname and a.name.split(".")[-1] == module)
            elif isinstance(node, ast.ImportFrom):
                aliases.update((a.asname or a.name) for a in node.names if a.name == module)
                if (node.module or "").split(".")[-1] == module:
                    for a in node.names:
                        if a.name == "*":
                            hits.append((rel, node.lineno, "*"))
                        else:
                            check(rel, node.lineno, a.name)

        def is_module(base: ast.expr, aliases: set[str] = aliases) -> bool:
            return (isinstance(base, ast.Name) and base.id in aliases) or (isinstance(base, ast.Attribute) and base.attr == module)

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and is_module(node.value):
                check(rel, node.lineno, node.attr)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and is_module(node.args[0])
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                check(rel, node.lineno, node.args[1].value)
    return hits


def _vocabulary_allowed(private: set[str]) -> Callable[[str], frozenset[str] | None]:
    def allowed_for(name: str) -> frozenset[str] | None:
        if name in VOCABULARY_RESTRICTED_PUBLIC:
            return VOCABULARY_RESTRICTED_PUBLIC[name]
        if name in private:
            return _REGISTRY_FILES
        return None

    return allowed_for


def _vocabulary_file_allowed(private: set[str]) -> Callable[[str], frozenset[str] | None]:
    return lambda name: frozenset({VOCABULARY_FILE}) if name in private else None


# (file relative to brain/, stripped line) -> (count, reason): a restricted name used where it is not owned.
PINNED_RESTRICTED_USES: dict[tuple[str, str], tuple[int, str]] = {
    ("health/reconstruct.py", "baseline_names: set[str] = {e.name for e in _vocabulary._BASELINE}"): (
        1,
        "reads the names of the framework's own 30 entries (an immutable tuple) to keep them out of the rebuilt file: no registry change",
    ),
    ("migrator/cli.py", "framework_baseline_names = {e.name for e in _vocabulary._BASELINE}"): (
        1,
        "reads the names of the framework's own 30 entries (an immutable tuple) to keep them out of the first vocabulary file: no registry change",
    ),
}


@pytest.mark.no_base_failure(reason="a ratchet over the whole tree: on the old code every use it pins already existed, so it is a guard against new ones and is not a fix")
def test_the_registry_and_the_copy_state_are_used_only_by_their_owners() -> None:
    trees = tree_asts("brain")
    sources = _brain_sources()
    vocab_private = _private(top_level_names(trees[VOCABULARY]))
    file_private = _private(top_level_names(trees[VOCABULARY_FILE]))
    hits = [(rel, n) for rel, n, _name in module_misuse(trees, "vocabulary", _vocabulary_allowed(vocab_private))]
    hits += [(rel, n) for rel, n, _name in module_misuse(trees, "vocabulary_file", _vocabulary_file_allowed(file_private))]
    problems = pin_problems(_keyed(sources, hits), PINNED_RESTRICTED_USES)
    assert not problems, (
        "the registry (_REGISTRY, register, replace_entry, _unregister), the copy-following state or a private name of the vocabulary "
        "modules is used outside the modules that own it. Change her vocabulary through vocabulary_file (add / replace_fields / merge / "
        "remove) so the file and the registry stay one thing:\n  " + "\n  ".join(problems)
    )


@pytest.mark.no_base_failure(reason="a completeness check over vocabulary.py's own top-level names; on the old code every name was already classified")
def test_every_public_name_of_the_vocabulary_module_is_classified() -> None:
    public = top_level_names(tree_asts("brain")[VOCABULARY]) - _private(top_level_names(tree_asts("brain")[VOCABULARY]))
    classified = set(VOCABULARY_RESTRICTED_PUBLIC) | VOCABULARY_OPEN_PUBLIC
    assert public == classified, (
        f"classify each new public name of vocabulary.py as restricted (VOCABULARY_RESTRICTED_PUBLIC) or open (VOCABULARY_OPEN_PUBLIC); "
        f"unclassified: {sorted(public - classified)}; listed but gone: {sorted(classified - public)}"
    )


@pytest.mark.no_base_failure(reason="a self-test of the scanner on synthetic source; it does not read the product tree")
def test_d_scanner_flags_each_way_to_reach_the_registry() -> None:
    private = {"_REGISTRY", "_unregister"}
    allowed_for = _vocabulary_allowed(private)
    bypasses = {
        "brain/a.py": "from brain.emotion import vocabulary\ndef f(e):\n    vocabulary.register(e)\n",
        "brain/b.py": "from brain.emotion import vocabulary as v\ndef f():\n    return v._REGISTRY\n",
        "brain/c.py": "import brain.emotion.vocabulary as v\ndef f(n):\n    v._unregister(n)\n",
        "brain/d.py": "from brain import emotion\ndef f(e):\n    emotion.vocabulary.replace_entry(e)\n",
        "brain/e.py": "from brain.emotion.vocabulary import register\n",
        "brain/f.py": "from brain.emotion.vocabulary import *\n",
        "brain/g.py": "from . import vocabulary\ndef f(e):\n    vocabulary.register(e)\n",
        "brain/h.py": 'from brain.emotion import vocabulary\ndef f():\n    return getattr(vocabulary, "_REGISTRY")\n',
        "brain/i.py": "from brain.emotion import vocabulary\ndef f():\n    vocabulary.set_file_copy(1, 2)\n",
        "brain/j.py": "from brain.emotion import vocabulary\ndef f():\n    vocabulary.set_bridge_owner(True)\n",
    }
    clean = {
        "brain/emotion/persona_loader.py": "from brain.emotion import vocabulary\ndef f():\n    vocabulary.set_file_copy(1, 2)\n",
        "brain/emotion/vocabulary_file.py": "from brain.emotion import vocabulary\ndef f(e):\n    vocabulary.register(e)\n",
        "brain/bridge/server.py": "from brain.emotion import vocabulary\ndef f():\n    vocabulary.set_bridge_owner(True)\n",
        "brain/k.py": "from brain.emotion import vocabulary\ndef f(n):\n    return vocabulary.get(n), vocabulary.canonical_name(n)\n",
        "brain/l.py": "def f(x):\n    return x.register()\n",
    }
    flagged = {rel for rel, _n, _name in module_misuse(_parse(bypasses | clean), "vocabulary", allowed_for)}
    assert flagged == set(bypasses)
    file_hits = module_misuse(
        _parse({"brain/m.py": "from brain.emotion import vocabulary_file\ndef f():\n    return vocabulary_file._REGISTRY_LOCK\n"}),
        "vocabulary_file",
        _vocabulary_file_allowed({"_REGISTRY_LOCK"}),
    )
    assert [name for _rel, _n, name in file_hits] == ["_REGISTRY_LOCK"]


# ------------------------------------------------------------------------------------------------
# E. a second tidier of names
# ------------------------------------------------------------------------------------------------

_REGEX_CALLS = frozenset({"compile", "sub", "subn", "split", "match", "search", "fullmatch", "findall", "finditer"})
_NEGATED_ALNUM_CLASS = re.compile(r"\[\^[^\]]*(?:a-z|A-Z|0-9|\\w|\\d)")
_NON_WORD = re.compile(r"\\W")
_SPACE_JOINER_CLASS = re.compile(r"\[[^\]]*\\s[^\]]*[_-][^\]]*\]|\[[^\]]*[_-][^\]]*\\s[^\]]*\]")  # a class of whitespace and `_` or `-`
_STRIP_CALLS = frozenset({"strip", "lstrip", "rstrip"})


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            out.update({t.id: node.value.value for t in node.targets if isinstance(t, ast.Name)})
    return out


def _receiver_methods(call: ast.Call) -> set[str]:
    """The names of the methods called along the receiver chain of `call` (`s.lower().replace(...)` -> {"lower"})."""
    names: set[str] = set()
    node: ast.expr = call.func.value if isinstance(call.func, ast.Attribute) else call.func
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
            node = node.func.value
        elif isinstance(node, ast.Attribute):
            node = node.value
        else:
            return names


def tidier_hits(trees: Mapping[str, ast.Module], exempt: frozenset[str]) -> list[tuple[str, int]]:
    """Code shaped like a name tidier, outside `exempt` (vocabulary.py, the one tidier): a strip of underscores; a regex whose pattern
    holds a negated alphanumeric class, `\\W`, or a class of whitespace with `_` or `-` (a string, or a module-level string constant of
    the same file); `.replace` joined with
    `.lower()` / `.casefold()`; `.replace(x, "_")` or `.replace(x, "-")`; any `.isalnum`.

    Not caught: a tidier written with `str.translate`, a character loop on another predicate, split-and-join, a pattern built at run
    time or imported from another file, a replacement other than `_` or `-`, a name tidied by a prompt. It finds the usual shapes, not
    intent: a hit says look, and today's hits are each pinned with the reason they are not feeling names.
    """
    hits: list[tuple[str, int]] = []
    for rel, tree in trees.items():
        if rel in exempt:
            continue
        consts = _module_string_constants(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "isalnum":
                hits.append((rel, node.lineno))
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            attr = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
            first = node.args[0] if node.args else None
            if attr in _STRIP_CALLS and isinstance(first, ast.Constant) and isinstance(first.value, str):
                chars = set(first.value)
                if "_" in chars and chars <= set("_- \t"):
                    hits.append((rel, node.lineno))
            if attr in _REGEX_CALLS and first is not None:
                pattern = first.value if isinstance(first, ast.Constant) else consts.get(first.id) if isinstance(first, ast.Name) else None
                if isinstance(pattern, str) and (
                    _NEGATED_ALNUM_CLASS.search(pattern) or _NON_WORD.search(pattern) or _SPACE_JOINER_CLASS.search(pattern)
                ):
                    hits.append((rel, node.lineno))
            if attr == "replace":
                receiver = _receiver_methods(node)
                second = node.args[1] if len(node.args) == 2 else None
                if receiver & {"lower", "casefold"} or (
                    isinstance(second, ast.Constant) and isinstance(second.value, str) and second.value in ("_", "-")
                ):
                    hits.append((rel, node.lineno))
            if attr in ("lower", "casefold") and "replace" in _receiver_methods(node):
                hits.append((rel, node.lineno))
    return hits


# (file relative to brain/, stripped line) -> (count, reason): today's hits, none of them a tidier of a feeling name.
TIDIER_HITS: dict[tuple[str, str], tuple[int, str]] = {
    ("cli.py", 'name = value.replace("_", "-")'): (1, "not a feeling name: the job name of a model-tier argument (`background_generative` or `background-generative`)"),
    ("engines/research_notes.py", '_SAFE = re.compile(r"[^A-Za-z0-9._-]")'): (1, "not a feeling name: makes an interest id safe for a file name"),
    ("health/attempt_heal.py", 'return iso_utc(now).replace(":", "-")'): (1, "not a feeling name: a timestamp made safe for a quarantine file name"),
    ("memory/known_names.py", "if ch.isalnum():"): (1, "not a feeling name: the text as the full-text index's tokenizer sees it (words of letters and digits), for the known-names search"),
    ("memory/store.py", 'pieces = [p for token in query for p in re.split(r"[^A-Za-z0-9]+", token)]'): (1, "not a feeling name: splits a recall search query into words"),
    ("memory/store.py", 'pieces = re.split(r"[^A-Za-z0-9]+", query)'): (1, "not a feeling name: splits a recall search query into words"),
    ("notes/write.py", 's = re.sub(r"[^\\w\\s-]", "", s).strip()'): (1, "not a feeling name: the slug of a note's title for its file name"),
    ("soul/review.py", 'safe_source_id = re.sub(r"[^A-Za-z0-9_.:-]+", "-", source_id)[:120]'): (1, "not a feeling name: makes a memory id safe inside a crystallization id"),
}


@pytest.mark.no_base_failure(reason="a ratchet over the whole tree: on the old code every hit it pins already existed, so it is a guard against a new tidier and is not a fix")
def test_no_second_tidier_of_feeling_names() -> None:
    sources = _brain_sources()
    hits = tidier_hits(tree_asts("brain"), frozenset({VOCABULARY}))
    problems = pin_problems(_keyed(sources, hits), TIDIER_HITS)
    assert not problems, (
        "code shaped like a name tidier is new (or a pinned one is gone). A feeling name is tidied by vocabulary.canonical_name and "
        "gated by vocabulary.gate_name, nowhere else. If this is not a feeling name, pin it in TIDIER_HITS with the reason:\n  "
        + "\n  ".join(problems)
    )


@pytest.mark.no_base_failure(reason="a self-test of the scanner on synthetic source; it does not read the product tree")
def test_e_scanner_flags_each_tidier_shape_and_ignores_plain_text_work() -> None:
    tidiers = {
        "brain/a.py": 'def f(n):\n    return n.strip("_")\n',
        "brain/b.py": 'import re\ndef f(n):\n    return re.sub(r"[^a-z0-9]+", "_", n)\n',
        "brain/c.py": 'def f(n):\n    return n.lower().replace(" ", "")\n',
        "brain/d.py": 'def f(n):\n    return n.replace(" ", "_")\n',
        "brain/e.py": 'def f(n):\n    return "".join(c for c in n if c.isalnum())\n',
        "brain/f.py": 'import re\n_P = re.compile(r"[\\W_]+")\n',
        "brain/g.py": 'import re\n_PAT = r"[^A-Za-z0-9]"\ndef f(n):\n    return re.sub(_PAT, "", n)\n',
        "brain/h.py": 'def f(n):\n    return n.replace("-", " ").lower()\n',
        "brain/i.py": 'def f(n):\n    return n.replace(" ", "-")\n',
        "brain/i2.py": 'import re\n_SEP = re.compile(r"[\\s_-]+")\n',
    }
    plain = {
        "brain/emotion/vocabulary.py": 'def f(n):\n    return n.strip("_").lower().replace(" ", "_")\n',
        "brain/j.py": 'def f(n):\n    return n.strip().lower()\n',
        "brain/k.py": 'import re\ndef f(n):\n    return re.sub(r"\\s+", " ", n).replace("a", "b")\n',
        "brain/l.py": 'def f(n):\n    return n.strip("/ ")\n',
    }
    flagged = {rel for rel, _n in tidier_hits(_parse(tidiers | plain), frozenset({VOCABULARY}))}
    assert flagged == set(tidiers)
