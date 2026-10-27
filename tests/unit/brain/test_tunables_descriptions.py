"""Every registered tunable has a one-line description (#529, owner ruling E8: "yes description"; criteria C57, C46, C56).

The description is what a person reads above the setting in ``tunables.toml``, so it must exist, fit on one line and be
plain English. Enforcement so a later tunable cannot be added without one: a fresh-interpreter registry check, an AST scan
of every ``tunables.register(...)`` call (and the bare ``register(...)`` in ``tunables.py``), and the text rules a machine
can check. Whether a description is ACCURATE against its reading site is a stage-6 reviewer rubric, not tested here.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.tunables_file import register_calls, register_calls_in_source

REPO = Path(__file__).resolve().parents[3]
BRAIN = REPO / "brain"

LLM_TELLS = ("delve", "tapestry", "testament", "seamless", "leverage", "utilize", "pivotal", "realm", "embark", "elevate", "unlock", "not just")
MODEL_ID = re.compile(r"claude-[a-z0-9.-]*\d")

# The module-scoped `probe` fixture starts a fresh interpreter that imports every brain module: run the module on one
# xdist worker (--dist loadgroup) so that happens once, not once per worker (#540).
pytestmark = pytest.mark.xdist_group("tunables-probe")


def _register_calls() -> list[tuple[str, int, ast.Call]]:
    """Every call of ``brain.tunables.register`` under brain/, however the module or the function was imported (rev-4 S7)."""
    return register_calls(REPO)


def _has_description(call: ast.Call) -> bool:
    return any(k.arg == "description" for k in call.keywords) or len(call.args) >= 3


# ---------------------------------------------------------------------------------------------------------------
# (b) the AST scan
# ---------------------------------------------------------------------------------------------------------------


def test_the_scan_finds_the_registrations_at_all() -> None:
    calls = _register_calls()
    assert len(calls) >= 40  # the scan is not silently empty
    assert any(rel == "brain/tunables.py" for rel, _l, _c in calls)  # the bare register() of the lull key is covered


def test_every_register_call_passes_a_description() -> None:
    missing = [f"{rel}:{line}" for rel, line, call in _register_calls() if not _has_description(call)]
    assert not missing, "tunables.register(...) without description=:\n  " + "\n  ".join(missing)


_ALIASED = {
    "module imported under another name": "from brain import tunables as t\nX = t.register('zz.a', 1, description='A.')\n",
    "dotted module import with an alias": "import brain.tunables as bt\nX = bt.register('zz.a', 1, description='A.')\n",
    "dotted module import, full path call": "import brain.tunables\nX = brain.tunables.register('zz.a', 1, description='A.')\n",
    "function imported by name": "from brain.tunables import register\nX = register('zz.a', 1, description='A.')\n",
    "function imported under another name": "from brain.tunables import register as reg\nX = reg('zz.a', 1, description='A.')\n",
    "relative module import": "from .. import tunables\nX = tunables.register('zz.a', 1, description='A.')\n",
    "relative function import": "from ..tunables import register\nX = register('zz.a', 1, description='A.')\n",
    "star import": "from brain.tunables import *\nX = register('zz.a', 1, description='A.')\n",
    "function-local import": "def f():\n    from brain import tunables as t\n    return t.register('zz.a', 1, description='A.')\n",
    "the usual shape": "from brain import tunables\nX = tunables.register('zz.a', 1, description='A.')\n",
}

_NOT_REGISTERS = {
    "another module's register": "from brain import prompt_strings\nX = prompt_strings.register('zz.a')\n",
    "a local function called register": "def register(k):\n    return k\nX = register('zz.a')\n",
    "an unrelated object": "import atexit\natexit.register(print)\n",
    "tunables.get_tunable": "from brain import tunables\nX = tunables.get_tunable('zz.a', 1)\n",
}


@pytest.mark.parametrize("name", sorted(_ALIASED))
def test_the_scan_finds_a_register_call_however_it_was_imported(name: str) -> None:
    found = register_calls_in_source(_ALIASED[name], "brain/memory/zz_mod.py")
    assert len(found) == 1 and found[0].args[0].value == "zz.a", name


@pytest.mark.parametrize("name", sorted(_NOT_REGISTERS))
def test_the_scan_ignores_a_register_that_is_not_the_tunables_one(name: str) -> None:
    assert register_calls_in_source(_NOT_REGISTERS[name], "brain/memory/zz_mod.py") == [], name


def test_the_scan_reads_the_bare_register_inside_tunables_py_itself() -> None:
    found = register_calls_in_source("def register(k, d):\n    return d\nX = register('zz.a', 1)\n", "brain/tunables.py")
    assert len(found) == 1


def test_a_planted_aliased_registration_in_brain_is_found_by_the_real_scan(tmp_path: Path) -> None:
    """The mutation: a module that registers through an alias used to be invisible to every register scan."""
    (tmp_path / "brain" / "memory").mkdir(parents=True)
    (tmp_path / "brain" / "memory" / "zz_planted.py").write_text(
        "from brain import tunables as t\nZ = t.register('zz.planted', 1)\n", encoding="utf-8"
    )
    found = register_calls(tmp_path)
    assert [(rel, c.args[0].value, _has_description(c)) for rel, _l, c in found] == [("brain/memory/zz_planted.py", "zz.planted", False)]


def test_a_register_call_without_a_description_would_be_caught(tmp_path: Path) -> None:
    tree = ast.parse("from brain import tunables\nX = tunables.register('a.b', 1)\nY = tunables.register('a.c', 1, description='Fine.')\n")
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert [_has_description(c) for c in calls] == [False, True]


# ---------------------------------------------------------------------------------------------------------------
# (a) the fresh-interpreter registry check, (e) the generated file, C56 no drift
# ---------------------------------------------------------------------------------------------------------------

_PROBE = r"""
import importlib, json, sys
sys.path.insert(0, {root!r})
modules = {modules!r}
import brain.bridge.server  # the real boot imports this
for name in modules:
    importlib.import_module(name)
from brain import tunables
from brain.utils import toml_io
tunables.sync_file()
reg = dict(tunables._registry)
desc = dict(tunables._descriptions)
path = tunables._file_path()
doc = toml_io.TomlDocument.from_bytes(path.read_bytes(), path)
layout, drift = {{}}, {{}}
for key, default in reg.items():
    run = doc.comment_above(key)
    layout[key] = run[-2:]
    got = tunables.get_tunable(key, default)
    if got != default or type(got) is not type(default):
        drift[key] = [repr(got), repr(default)]
print(json.dumps({{
    "keys": sorted(reg), "described": sorted(k for k, v in desc.items() if v),
    "descriptions": desc, "defaults": {{k: repr(v) for k, v in reg.items()}},
    "layout": layout, "drift": drift, "values_in_file": sorted(doc.values()),
}}))
"""


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> dict:
    modules = sorted({
        rel[:-3].replace("/", ".")
        for rel, _l, _c in _register_calls()
        if rel != "brain/tunables.py"
    })
    home = tmp_path_factory.mktemp("probe-home")
    env = {**os.environ, "KINDLED_HOME": str(home)}
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(root=str(REPO), modules=modules)],
        capture_output=True, text=True, env=env, timeout=300, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_every_registered_key_has_a_non_empty_description_in_a_fresh_interpreter(probe: dict) -> None:
    missing = sorted(set(probe["keys"]) - set(probe["described"]))
    assert not missing, f"registered without a description: {missing}"


def test_the_registry_count_equals_the_ast_count_so_neither_check_can_be_escaped(probe: dict) -> None:
    literal_keys = [
        c.args[0].value
        for _rel, _line, c in _register_calls()
        if c.args and isinstance(c.args[0], ast.Constant) and isinstance(c.args[0].value, str)
    ]
    assert len(literal_keys) == len(_register_calls()), "a registration with a non-literal key escapes the key check"
    assert len(set(literal_keys)) == len(literal_keys), "a key is registered twice"
    assert sorted(literal_keys) == probe["keys"]


def test_the_generated_file_shows_each_description_directly_above_its_default_line(probe: dict) -> None:
    for key, last_two in probe["layout"].items():
        assert len(last_two) == 2, key
        assert last_two[1].startswith("# default: "), key
        assert last_two[0] == f"# {probe['descriptions'][key]}", key


def test_sync_at_boot_writes_every_key_and_each_reads_back_equal_in_value_and_type(probe: dict) -> None:
    assert sorted(probe["values_in_file"]) == probe["keys"]
    assert probe["drift"] == {}, probe["drift"]  # C56: an untouched key reads as the code default object, type and all


# ---------------------------------------------------------------------------------------------------------------
# (c) the text rules
# ---------------------------------------------------------------------------------------------------------------


def _whole_word(needle: str, haystack: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9_.])" + re.escape(needle) + r"(?![A-Za-z0-9_.])", haystack, re.IGNORECASE) is not None


def _problems(key: str, text: str, default_repr: str) -> list[str]:
    out: list[str] = []
    if "\n" in text or "\r" in text:
        out.append("not one line")
    if not 1 <= len(text) <= 100:
        out.append(f"length {len(text)} not in 1..100")
    if text != text.strip():
        out.append("leading or trailing space")
    if not text[:1].isupper():
        out.append("does not start with a capital letter")
    if not text.endswith("."):
        out.append("does not end with a period")
    if "—" in text or "–" in text:
        out.append("em-dash or en-dash")
    if MODEL_ID.search(text):
        out.append("holds a model id")
    for bad in ("http", "#", "/", "\\", "~"):
        if bad in text:
            out.append(f"contains {bad!r}")
    shown = default_repr.strip("'\"")
    if len(shown) >= 4 and _whole_word(shown, text):
        out.append("repeats the default's text")
    low = text.lower()
    for word in LLM_TELLS:
        if word in low:
            out.append(f"LLM-tell word {word!r}")
    return out


def test_every_description_passes_the_text_rules(probe: dict) -> None:
    bad = {
        key: problems
        for key, text in probe["descriptions"].items()
        if (problems := _problems(key, text, probe["defaults"][key]))
    }
    assert not bad, bad


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        ("Seconds the app waits for the first reply.", []),
        ("two\nlines.", ["not one line"]),
        ("no capital here.", ["does not start with a capital letter"]),
        ("No period at the end", ["does not end with a period"]),
        ("Has an em—dash.", ["em-dash or en-dash"]),
        ("Uses claude-sonnet-5 by name.", ["holds a model id"]),
        ("See https://example.test.", ["contains 'http'"]),
        ("x" * 101 + ".", ["length 102 not in 1..100"]),
        (" Leading space.", ["leading or trailing space", "does not start with a capital letter"]),
        ("We delve into it.", ["LLM-tell word 'delve'"]),
        ("Repeats haiku in its text.", ["repeats the default's text"]),
    ],
)
def test_each_text_rule_flags_its_violation(text: str, expect: list[str]) -> None:
    got = _problems("k", text, "'haiku'")
    assert got == [] if not expect else all(e in got for e in expect)


def test_a_short_default_never_causes_a_false_repeat_failure() -> None:
    assert _problems("k", "How many at once, from 1 up.", "1") == []
    assert _problems("k", "Dollars, such as 5.0, per turn.", "5.0") == []
