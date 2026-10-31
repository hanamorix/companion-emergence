"""The shared-settings guard (#531): a value meant to live in one place is not copied or reassigned elsewhere.

A *setting* is a user tunable (``tunables.register``, read through ``tunables.get_tunable``) or a developer constant
(``brain/dev_constants.py``). The checkers live in ``tests/shared_settings.py``; this file holds the planted-violation
fixtures (each rule flags a violating source and passes a conforming one, over a small synthetic ``Settings`` so that
renaming a real setting cannot break a fixture), the ratchet, and the scan of the real tree against ``ALLOW``.

``ALLOW`` is the list of violations that exist today. It is meant to shrink: the scan must equal it as a multiset, so a new
copy fails (remove the copy; add an entry only for a different thing with an equal value, with the reason), a listed entry
with no finding fails (delete it), and ``ALLOW_SIZE`` pins the length so that growth needs a second visible edit and
shrinkage tells you to lower it.

The model case of #529 (no Claude model chosen outside the tunables accessor; rules G1, G2, G4, G5, and R4 for the
accessors) is one case of this guard; its own vocabulary list ``MODEL_LITERAL_ALLOW`` (definitions and labels that are
the model setting's own words, not copies) lives in ``tests/shared_settings.py`` with its own stale-entry test and is not
part of the ALLOW count.
"""

from __future__ import annotations

import textwrap

import pytest

from tests.shared_settings import (
    APP_ALIAS_ALLOW,
    MODEL_LITERAL_ALLOW,
    Finding,
    Settings,
    TunableInfo,
    app_findings,
    app_numbers,
    check_all,
    check_g1,
    check_g4,
    check_g5,
    check_r1,
    derive_settings,
    make_settings,
    ratchet_problems,
    rust_tokens,
    scan_real_tree,
    ts_tokens,
)
from tests.source_tree import REPO, brain_trees

# Only tests that pass on the pre-change code by design are exempt from tests/no_base_failure's "must fail on the base"
# rule (scripts/new_tests_fail_on_base.py), each with the reason that applies to it. The real-tree test
# (`test_real_tree_findings_equal_the_allow_list`) carries no mark: it fails on any tree that holds a copy the list does not name.
_NBF_SYNTHETIC = pytest.mark.no_base_failure(
    reason="#531 guard: a checker (tests/shared_settings.py) or the ratchet logic run over a source or a list this test builds; "
    "the product code is not an input, so the test passes on the pre-change code and on any other"
)
_NBF_TREE_SNAPSHOT = pytest.mark.no_base_failure(
    reason="#531 guard: asserts a property of the tree that the pre-change tree already has (model vocabulary entries still match, "
    "settings are derived, no model-rule finding, the app files' alias words, the connection panel); it records that state, "
    "it does not test a change"
)

# ---------------------------------------------------------------------------
# The synthetic settings every fixture runs against
# ---------------------------------------------------------------------------

_MIB20 = 20 * 1024 * 1024
SETTINGS = make_settings(
    dev={
        "WIDGET_TIMEOUT_S": 45.0,  # round value, two name words
        "ORBIT": 77,  # one name word
        "UPLOAD_CEILING_BYTES": _MIB20,  # distinctive number (7 significant digits)
        "STORE_FILE": "widget_store.db",  # distinctive string (has a '.' and an '_')
        "MODE_WORD": "strict",  # a plain word: not distinctive
        "ROUND_TICK_S": 60,
        "DAY_LENGTH_S": 86400,  # three significant digits ("864"): not distinctive
        "ZETA_DAILY_BUDGET_DEFAULT": 150,  # a group of four constants share the prefix ZETA
        "ZETA_MIN_AGE_HOURS": 6.0,  # two words remain after ZETA: too generic to be a form of its own
        "ZETA_ROW_LIMIT_CAP": 40,
        "ZETA_SPAN_HOURS": 3.0,  # one word remains after ZETA
        "PAIR_FADE_FLOOR": 0.2,  # a pair shares PAIR: a pair is not a group
        "PAIR_FADE_STEP": 0.1,
        "TRIO_GAUGE_WIDTH_LIMIT": 13,  # three constants are the smallest group
        "TRIO_PANEL_HEIGHT_LIMIT": 14,
        "TRIO_ROTOR_SPEED_LIMIT": 15,
        "TWIN_ALPHA_BETA_GAMMA": 11,  # a pair whose remainders are long enough: still a pair, not a group
        "TWIN_DELTA_EPSILON_ETA": 12,
        "IMAGE_KINDS": ("alpha_kind", "beta_kind"),  # a collection of strings
    },
    tunables=[
        TunableInfo("area.retry_wait_seconds", 12.0, "RETRY_WAIT_SECONDS", "brain/area.py"),
        TunableInfo("area.upload_cap", _MIB20, None, "brain/area.py"),
        TunableInfo("area.read_cap", 524288, None, "brain/area.py"),
        TunableInfo("area.cap", 777, "_CAP", "brain/area.py"),  # a bound name of one word
        TunableInfo("other.sample_rows", 100, "SAMPLE_ROWS", "brain/other.py"),
        TunableInfo("model.thing", "claude-test-9", None, "brain/bridge/model_tier.py"),
    ],
)

_MODULE = "brain/some/module.py"


def _findings(source: str, path: str = _MODULE) -> list[Finding]:
    return check_all(textwrap.dedent(source), path, SETTINGS)


def _rules(source: str, path: str = _MODULE) -> set[str]:
    return {f.rule for f in _findings(source, path)}


# ---------------------------------------------------------------------------
# The model case of #529 (rules G1, G2, G4, G5 and the accessor rule R4): the old fixtures, same names, same sources
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

# The same outliving shapes for the general accessor: ``tunables.get_tunable`` (R4 is the #529 G3 rule over every accessor).
_OUTLIVING_SHAPES = (
    "module_constant", "class_constant", "parameter_default", "dataclass_default_factory", "self_attribute",
    "subscript_cache", "nonlocal", "global", "setattr", "append", "setdefault", "partial", "lru_cache",
    "module_call_of_returning_function",
)


def _as_get_tunable(source: str) -> str:
    return source.replace(
        "from brain.bridge.model_tier import model_for_tier\n", "from brain import tunables\n_D = 'a'\n"
    ).replace("model_for_tier('x')", "tunables.get_tunable('a.b', _D)")


_VIOLATING_GET_TUNABLE = {f"get_tunable_{n}": _as_get_tunable(_VIOLATING[n]) for n in _OUTLIVING_SHAPES}
_CONFORMING_GET_TUNABLE = {
    "get_tunable_local_variable": (
        "from brain import tunables\n"
        "_LIMIT_D = tunables.register('a.b', 5, description='d')\n"
        "def f():\n    v = tunables.get_tunable('a.b', _LIMIT_D)\n    return v + 1\n"
    ),
}


_VIOLATING_G2_GET_TUNABLE = {
    "argv_from_get_tunable": "from brain import tunables\ndef f():\n    return ['claude', '--model', tunables.get_tunable('a.b', 'x')]\n",
    "argv_from_a_local_holding_get_tunable": (
        "from brain import tunables\ndef f():\n    m = tunables.get_tunable('a.b', 'x')\n    return ['claude', '--model', m]\n"
    ),
}


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_VIOLATING_G2_GET_TUNABLE))
def test_fixture_g2_a_get_tunable_result_is_not_a_model_accessor_result(name: str) -> None:
    """Only the MODEL accessors may feed ``--model``: a tunable of any other key is not the model setting."""
    assert "G2" in {f.rule for f in check_all(_VIOLATING_G2_GET_TUNABLE[name], _MODULE, SETTINGS)}, name


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_CONFORMING) + sorted(_CONFORMING_GET_TUNABLE))
def test_fixture_conforming_model_and_accessor_source_passes_every_rule(name: str) -> None:
    source = _CONFORMING.get(name) or _CONFORMING_GET_TUNABLE[name]
    assert check_all(source, _MODULE, SETTINGS) == []


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_VIOLATING))
def test_fixture_violating_model_source_is_flagged(name: str) -> None:
    flagged = check_all(_VIOLATING[name], _MODULE, SETTINGS)
    assert flagged, f"fixture {name!r} was not flagged by any rule"


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_VIOLATING_GET_TUNABLE))
def test_fixture_get_tunable_copy_that_outlives_the_call_is_flagged_by_r4(name: str) -> None:
    assert "R4" in {f.rule for f in check_all(_VIOLATING_GET_TUNABLE[name], _MODULE, SETTINGS)}, name


@_NBF_SYNTHETIC
def test_fixture_g5_model_key_is_the_model_cases_and_set_values_is_r1d() -> None:
    """The model key literal is G5; ``set_values`` ownership is R1(d): the fold function only, not its file."""
    fold = "from brain import tunables\ndef _fold(legacy, complete):\n    tunables.set_values({'a': 1}, only_if_default=True)\n"
    second = "from brain import tunables\ndef _another(u):\n    tunables.set_values(u)\n"
    nested = "from brain import tunables\ndef _fold(l, c):\n    def inner(u):\n        tunables.set_values(u)\n    return inner\n"
    module_level = "from brain import tunables\ntunables.set_values({})\n"
    assert check_r1(fold, "brain/tunables_migration.py", SETTINGS) == []
    assert check_r1(second, "brain/tunables_migration.py", SETTINGS) != []
    assert check_r1(nested, "brain/tunables_migration.py", SETTINGS) != []
    assert check_r1(module_level, "brain/tunables_migration.py", SETTINGS) != []
    assert check_r1(fold, "brain/other.py", SETTINGS) != []  # the same function name elsewhere is not exempt
    assert [f.rule for f in check_g5("KEY = 'model.interactive_chat'\n", "brain/some/module.py")] == ["G5"]
    assert check_g5("KEY = 'model.interactive_chat'\n", "brain/bridge/model_tier.py") == []


@_NBF_SYNTHETIC
def test_fixture_the_one_probe_site_may_use_probe_model() -> None:
    src = "from brain.bridge.provider import ClaudeCliProvider\ndef _check_model_blocking(x):\n    return ClaudeCliProvider(probe_model=x)\n"
    assert check_g4(src, "brain/bridge/server.py") == []


@_NBF_SYNTHETIC
def test_fixture_g1_allow_entry_is_shape_specific() -> None:
    """C19: a real literal in an allow-listed file in a non-label shape fails; in the label shape it passes."""
    label = "def f(deps):\n    return DCallRow(model_tier_used='sonnet')\n"
    other = "X = 'claude-sonnet-5'\n"
    assert check_g1(label, "brain/initiate/reflection.py") == []
    assert check_g1(other, "brain/initiate/reflection.py") != []
    # a labelled site that disappears leaves its allow-list entry unmatched
    used: set = set()
    check_g1("X = 1\n", "brain/initiate/reflection.py", used=used)
    assert ("initiate/reflection.py", "sonnet", "label_keyword") not in used


@_NBF_SYNTHETIC
def test_fixture_g1_does_not_flag_unrelated_claude_words() -> None:
    src = "A = 'claude-cli'\nB = 'claude-config'\nC = 'claude-work'\nD = 'claude-tool'\nE = 'the claude CLI'\n"
    assert check_g1(src, "brain/x.py") == []
    # a "claude-" prefix TEST is not id assembly; a prefix piece that a variable completes is
    assert check_g1("def f(v):\n    return v.startswith('claude-')\n", "brain/x.py") == []
    assert check_g1("def f(n):\n    return f'claude-{n}'\n", "brain/x.py") != []  # a prefix a variable completes can build any id
    assert check_g1("def f(n):\n    return f'the claude CLI {n}'\n", "brain/x.py") == []
    assert check_g1("def f(m):\n    return 'claude-cli:' + m\n", "brain/x.py") == []  # the provider label is not a model id


# ---------------------------------------------------------------------------
# R1 write
# ---------------------------------------------------------------------------

_DC = "from brain import dev_constants\n"
_R1_VIOLATING = {
    "attribute_store_on_a_dev_constant": _DC + "def f():\n    dev_constants.WIDGET_TIMEOUT_S = 1.0\n",
    "augmented_store_on_a_dev_constant": _DC + "def f():\n    dev_constants.WIDGET_TIMEOUT_S += 1.0\n",
    "delete_of_a_dev_constant": _DC + "def f():\n    del dev_constants.WIDGET_TIMEOUT_S\n",
    "attribute_store_on_a_tunable_constant": "from brain import area\ndef f():\n    area.RETRY_WAIT_SECONDS = 3.0\n",
    "setattr_with_a_setting_name": "def f(m):\n    setattr(m, 'WIDGET_TIMEOUT_S', 1.0)\n",
    "delattr_with_a_setting_name": "def f(m):\n    delattr(m, 'RETRY_WAIT_SECONDS')\n",
    "rebind_of_an_imported_dev_constant": "from brain.dev_constants import WIDGET_TIMEOUT_S\nWIDGET_TIMEOUT_S = 1.0\n",
    "rebind_through_global": (
        "from brain.dev_constants import WIDGET_TIMEOUT_S\ndef f():\n    global WIDGET_TIMEOUT_S\n    WIDGET_TIMEOUT_S = 2.0\n"
    ),
    "rebind_of_a_name_imported_from_tunables": "from brain.tunables import get_tunable\nget_tunable = None\n",
    "set_values_outside_the_writers": "from brain import tunables\ndef f():\n    tunables.set_values({'a': 1})\n",
}
_R1_CONFORMING = {
    "reading_a_dev_constant": (_DC + "def f():\n    return dev_constants.WIDGET_TIMEOUT_S + 1\n", _MODULE),
    "set_values_in_set_job_models": (
        "from brain import tunables\ndef set_job_models(u):\n    tunables.set_values(u)\n", "brain/bridge/model_tier.py"
    ),
    "set_values_in_the_fold": (
        "from brain import tunables\ndef _fold(a, b):\n    tunables.set_values({}, only_if_default=True)\n", "brain/tunables_migration.py"
    ),
    "set_values_in_tunables_itself": ("def set_values(u):\n    return u\ndef g():\n    set_values({})\n", "brain/tunables.py"),
    "a_local_that_shadows_an_imported_name": (
        "from brain.dev_constants import WIDGET_TIMEOUT_S\ndef f():\n    WIDGET_TIMEOUT_S = 1.0\n    return WIDGET_TIMEOUT_S\n", _MODULE
    ),
    "a_local_that_shares_a_setting_name_but_was_not_imported": ("def f():\n    WIDGET_TIMEOUT_S = 1.0\n    return WIDGET_TIMEOUT_S\n", _MODULE),
}


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_R1_VIOLATING))
def test_fixture_r1_write_is_flagged(name: str) -> None:
    assert "R1" in _rules(_R1_VIOLATING[name]), name


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_R1_CONFORMING))
def test_fixture_r1_conforming_source_passes_every_rule(name: str) -> None:
    source, path = _R1_CONFORMING[name]
    assert _findings(source, path) == [], name


# ---------------------------------------------------------------------------
# R2 name
# ---------------------------------------------------------------------------

_R2_VIOLATING = {
    "module_constant_same_name_other_value": "WIDGET_TIMEOUT = 99.0\n",
    "private_lowercase_unit_spelling": "_widget_timeout_seconds = 3\n",
    "class_constant": "class C:\n    WIDGET_TIMEOUT_S = 1\n",
    "parameter_with_a_default": "def f(widget_timeout=3.5):\n    return widget_timeout\n",
    "tunable_bound_constant_name": "RETRY_WAIT = 4.5\n",
    "unprefixed_form_of_a_group_constant": "DAILY_BUDGET_DEFAULT = 200\n",
    "unprefixed_form_as_a_parameter": "def f(daily_budget_default=3.5):\n    return daily_budget_default\n",
    "unprefixed_form_in_a_class": "class C:\n    ROW_LIMIT_CAP = 9\n",
    "unprefixed_form_of_the_smallest_group": "GAUGE_WIDTH_LIMIT = 9.5\n",
    "drifted_copy_named_after_a_tunable_key": "UPLOAD_CAP = 1234\n",  # #601: area.upload_cap, value drifted
}
_R2_CONFORMING = {
    "owner_file_of_a_dev_constant": ("WIDGET_TIMEOUT_S: float = 45.0\n", "brain/dev_constants.py"),
    "owner_file_of_a_tunable": (
        "from brain import tunables\nRETRY_WAIT_SECONDS = tunables.register('area.retry_wait_seconds', 12.0, description='d')\n",
        "brain/area.py",
    ),
    "parameter_with_a_none_default": ("def f(widget_timeout=None):\n    return widget_timeout\n", _MODULE),
    "parameter_without_a_default": ("def f(widget_timeout):\n    return widget_timeout\n", _MODULE),
    "a_pair_prefix_is_not_a_group": ("FADE_FLOOR = 9.5\n", _MODULE),
    "a_remainder_of_one_word_is_not_matched": ("SPAN = 8.5\n", _MODULE),
    "a_remainder_of_two_words_is_not_matched": ("MIN_AGE = 8.5\nDAILY_BUDGET = 9.5\n", _MODULE),
    "a_pair_is_not_a_group_even_with_long_remainders": ("ALPHA_BETA_GAMMA = 8.5\n", _MODULE),
    "a_setting_name_of_one_word_is_not_matched": ("ORBIT = 5.5\nCAP = 3.5\ndef f(cap=2.5):\n    return cap\n", _MODULE),
    "a_bare_annotation_holds_no_copy": ("class C:\n    widget_timeout: float\n    widget_timeout_s: float | None = None\n", _MODULE),
    "a_local_variable": ("def f():\n    widget_timeout = 3.5\n    return widget_timeout\n", _MODULE),
    "a_different_name": ("WIDGET_RETRY = 3.5\nCACHE_TIMEOUT = 4.5\n", _MODULE),
}


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_R2_VIOLATING))
def test_fixture_r2_same_name_is_flagged(name: str) -> None:
    assert "R2" in _rules(_R2_VIOLATING[name]), name


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_R2_CONFORMING))
def test_fixture_r2_conforming_source_passes_every_rule(name: str) -> None:
    source, path = _R2_CONFORMING[name]
    assert _findings(source, path) == [], name


@_NBF_SYNTHETIC
def test_fixture_group_prefix_is_removed_only_from_a_group_and_only_when_two_words_remain() -> None:
    """The prefixes the move to dev_constants added (ATTUNEMENT_, KINDLED_LINK_, JUDGE_TUNE_GRADE_) hide the natural name from
    an exact name match. A leading run of name words shared by at least three dev constants is a group prefix; the name
    without it is also a form of the setting, when at least three words remain."""
    forms = SETTINGS.dev_forms
    assert frozenset({"DAILY", "BUDGET", "DEFAULT"}) in forms["ZETA_DAILY_BUDGET_DEFAULT"]
    assert frozenset({"ZETA", "DAILY", "BUDGET", "DEFAULT"}) in forms["ZETA_DAILY_BUDGET_DEFAULT"]
    assert frozenset({"ROW", "LIMIT", "CAP"}) in forms["ZETA_ROW_LIMIT_CAP"]
    assert forms["ZETA_MIN_AGE_HOURS"] == (frozenset({"ZETA", "MIN", "AGE"}),)  # two words remain: too generic
    assert forms["ZETA_SPAN_HOURS"] == (frozenset({"ZETA", "SPAN"}),)  # one word remains: not a form
    assert forms["PAIR_FADE_FLOOR"] == (frozenset({"PAIR", "FADE", "FLOOR"}),)  # a pair is not a group
    assert frozenset({"GAUGE", "WIDTH", "LIMIT"}) in forms["TRIO_GAUGE_WIDTH_LIMIT"]  # three constants are a group
    assert forms["TWIN_ALPHA_BETA_GAMMA"] == (frozenset({"TWIN", "ALPHA", "BETA", "GAMMA"}),)  # a pair of long names neither
    assert forms["WIDGET_TIMEOUT_S"] == (frozenset({"WIDGET", "TIMEOUT"}),)  # no group at all


# ---------------------------------------------------------------------------
# R3 distinctive value, loose value, collection
# ---------------------------------------------------------------------------

_R3_VIOLATING = {
    "number_written_out": "def f():\n    return 20971520\n",
    "number_as_a_product": "def f():\n    return 20 * 1024 * 1024\n",
    "number_as_a_float_product": "def f():\n    return 20.0 * 1024 * 1024\n",
    "number_in_a_parameter_default": "def f(limit=20 * 1024 * 1024):\n    return limit\n",
    "string_with_a_dot_and_an_underscore": "def f():\n    return 'widget_store.db'\n",
    "number_equal_to_a_tunable_default": "def f():\n    return 524288\n",
}
_R3_CONFORMING = {
    "round_values": ("def f():\n    time.sleep(60)\n    return [86400, 30, 200]\n", _MODULE),
    "a_plain_word_string": ("def f():\n    return 'strict'\n", _MODULE),
    "the_register_call_own_default": (
        "from brain import tunables\nCOPY = tunables.register('area.copy_cap', 20 * 1024 * 1024, description='d')\n", _MODULE
    ),
    "the_register_call_own_string_default": (
        "from brain import tunables\nNAME = tunables.register('area.copy_name', 'widget_store.db', description='d')\n", _MODULE
    ),
    "a_docstring_that_is_the_value": ("def f():\n    '''widget_store.db'''\n    return 1\n", _MODULE),
    "a_part_of_an_fstring_that_is_the_value": ("def f(x):\n    return f'widget_store.db{x}'\n", _MODULE),
    "a_product_that_is_not_the_value": ("def f():\n    return 20 * 1024\n", _MODULE),
    "owner_file_of_the_dev_constants": ("UPLOAD_CEILING_BYTES = 20 * 1024 * 1024\nSTORE_FILE = 'widget_store.db'\n", "brain/dev_constants.py"),
}
_R3_LOOSE_VIOLATING = {
    "local_with_every_word_of_a_dev_name": "def f():\n    widget_timeout_seconds = 45\n    return widget_timeout_seconds\n",
    "keyword_argument": "def f(w):\n    return w.wait(widget_timeout=45)\n",
    "renamed_copy_of_a_tunable_default": "def f():\n    retry_wait = 12\n    return retry_wait\n",
    "parameter_default": "def f(retry_wait_s=12.0):\n    return retry_wait_s\n",
    "unprefixed_form_of_a_group_constant": "def f():\n    daily_budget_default = 150\n    return daily_budget_default\n",
    "full_key_path_of_a_tunable": "def f():\n    area_retry_wait = 12.0\n    return area_retry_wait\n",
    "more_words_than_the_form": "def f():\n    big_widget_timeout_in_use = 45.0\n    return big_widget_timeout_in_use\n",
}
_R3_LOOSE_CONFORMING = {
    "same_value_unrelated_name": ("def f():\n    delay = 45.0\n    return delay\n", _MODULE),
    "a_form_of_one_word_names_a_kind_of_thing": ("def f():\n    orbit = 77\n    return orbit\n", _MODULE),
    "only_some_of_the_words": ("def f():\n    widget_limit = 45.0\n    return widget_limit\n", _MODULE),
    "owner_file_of_the_tunable": ("def f():\n    retry_wait = 12\n    return retry_wait\n", "brain/area.py"),
    "the_value_inside_a_register_call": (
        "from brain import tunables\nWIDGET_TIMEOUT_COPY = tunables.register('area.x', 45.0, description='d')\n", _MODULE
    ),
    "different_value_same_words": ("def f():\n    widget_timeout = 46.0\n    return widget_timeout\n", _MODULE),
}
_R3_COLLECTION_VIOLATING = {
    "list_of_the_same_set": "def f():\n    return ['alpha_kind', 'beta_kind']\n",
    "tuple_in_another_order": "KINDS = ('beta_kind', 'alpha_kind')\n",
    "set_literal": "def f():\n    return {'alpha_kind', 'beta_kind'}\n",
    "frozenset_call": "def f():\n    return frozenset({'alpha_kind', 'beta_kind'})\n",
    "default_argument": "def f(kinds=('alpha_kind', 'beta_kind')):\n    return kinds\n",
}
_R3_COLLECTION_CONFORMING = {
    "a_different_set": ("def f():\n    return ['alpha_kind', 'gamma_kind']\n", _MODULE),
    "one_element": ("def f():\n    return ['alpha_kind']\n", _MODULE),
    "a_superset": ("def f():\n    return ['alpha_kind', 'beta_kind', 'gamma_kind']\n", _MODULE),
    "one_element_is_not_matched_alone": ("def f():\n    return 'alpha_kind'\n", _MODULE),
    "the_same_set_in_a_register_call": (
        "from brain import tunables\nK = tunables.register('area.kinds', ['alpha_kind', 'beta_kind'], description='d')\n", _MODULE
    ),
}


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_R3_VIOLATING))
def test_fixture_r3_distinctive_value_is_flagged(name: str) -> None:
    assert "R3" in _rules(_R3_VIOLATING[name]), name


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_R3_LOOSE_VIOLATING))
def test_fixture_r3_loose_value_with_the_settings_words_is_flagged(name: str) -> None:
    assert "R3-loose" in _rules(_R3_LOOSE_VIOLATING[name]), name


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_R3_COLLECTION_VIOLATING))
def test_fixture_r3_collection_of_the_same_elements_is_flagged_once(name: str) -> None:
    flagged = [f for f in _findings(_R3_COLLECTION_VIOLATING[name]) if f.rule == "R3-collection"]
    assert len(flagged) == 1 and flagged[0].settings == ("dev_constants.IMAGE_KINDS",), name


@_NBF_SYNTHETIC
@pytest.mark.parametrize(
    "name",
    [f"value:{n}" for n in sorted(_R3_CONFORMING)]
    + [f"loose:{n}" for n in sorted(_R3_LOOSE_CONFORMING)]
    + [f"collection:{n}" for n in sorted(_R3_COLLECTION_CONFORMING)],
)
def test_fixture_r3_conforming_source_passes_every_rule(name: str) -> None:
    kind, key = name.split(":", 1)
    source, path = {"value": _R3_CONFORMING, "loose": _R3_LOOSE_CONFORMING, "collection": _R3_COLLECTION_CONFORMING}[kind][key]
    assert _findings(source, path) == [], name


@_NBF_SYNTHETIC
def test_fixture_r3_reports_one_finding_per_site_naming_every_setting_that_shares_the_value() -> None:
    flagged = _findings("def f():\n    return 20 * 1024 * 1024\n")
    assert [f.rule for f in flagged] == ["R3"]
    assert flagged[0].settings == ("dev_constants.UPLOAD_CEILING_BYTES", "tunables:area.upload_cap")
    assert flagged[0].site == "in f" and flagged[0].line == 2


@_NBF_SYNTHETIC
def test_fixture_r3_distinctive_wins_over_loose_at_one_site() -> None:
    """B6: a literal that is distinctive AND bound to the setting's words is reported once, as the precise rule."""
    flagged = _findings("def f():\n    upload_ceiling = 20971520\n    return upload_ceiling\n")
    assert [f.rule for f in flagged] == ["R3"]


@_NBF_SYNTHETIC
def test_fixture_r3_a_model_tunables_string_is_the_model_cases_not_r3() -> None:
    assert {f.rule for f in _findings("X = 'claude-test-9'\n")} == {"G1"}


@_NBF_SYNTHETIC
def test_fixture_r3_loose_names_only_the_settings_whose_words_match() -> None:
    """The loose tier lists the settings whose value AND words match, not every setting that shares the value."""
    flagged = _findings("def f():\n    retry_wait = 12\n    return retry_wait\n")
    assert [f.settings for f in flagged if f.rule == "R3-loose"] == [("tunables:area.retry_wait_seconds",)]


# ---------------------------------------------------------------------------
# R4 outlives the call (the accessors), R4b literal default, R5 alias and default
# ---------------------------------------------------------------------------

_R4B_VIOLATING = {
    "numeric_literal": "from brain import tunables\ndef f():\n    return tunables.get_tunable('area.k', 5)\n",
    "string_literal": "from brain import tunables\ndef f():\n    return tunables.get_tunable('area.k', 'text here')\n",
    "folded_numeric_expression": "from brain import tunables\ndef f():\n    return tunables.get_tunable('area.k', 2 * 3)\n",
    "keyword_form": "from brain import tunables\ndef f():\n    return tunables.get_tunable(key='area.k', default=5)\n",
}
_R4B_CONFORMING = {
    "constant_default": (
        "from brain import tunables\n_K_D = tunables.register('area.k', 5, description='d')\n"
        "def f():\n    return tunables.get_tunable('area.k', _K_D)\n"
    ),
    "none_default": ("from brain import tunables\ndef f():\n    return tunables.get_tunable('area.k', None)\n"),
}
_R5_VIOLATING = {
    "alias_at_module_level": ("R5a", _DC + "X = dev_constants.WIDGET_TIMEOUT_S\n"),
    "alias_with_an_annotation": ("R5a", _DC + "X: float = dev_constants.WIDGET_TIMEOUT_S\n"),
    "alias_at_class_level": ("R5a", _DC + "class C:\n    X = dev_constants.WIDGET_TIMEOUT_S\n"),
    "alias_of_a_from_import": ("R5a", "from brain.dev_constants import WIDGET_TIMEOUT_S\nY = WIDGET_TIMEOUT_S\n"),
    "alias_through_a_module_alias": ("R5a", "from brain import dev_constants as dc\nX = dc.WIDGET_TIMEOUT_S\n"),
    "frozen_dataclass_field_default": (
        "R5a",
        _DC + "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass C:\n    t: float = dev_constants.WIDGET_TIMEOUT_S\n",
    ),
    "parameter_default": ("R5b", _DC + "def f(x=dev_constants.WIDGET_TIMEOUT_S):\n    return x\n"),
    "keyword_only_parameter_default": ("R5b", _DC + "def f(*, x=dev_constants.WIDGET_TIMEOUT_S):\n    return x\n"),
}
_R5_CONFORMING = {
    "read_inside_a_body": (_DC + "def f():\n    return dev_constants.WIDGET_TIMEOUT_S\n", _MODULE),
    "a_local_variable_holding_the_read": (_DC + "def f():\n    x = dev_constants.WIDGET_TIMEOUT_S\n    return x\n", _MODULE),
    "module_import_alone": (_DC, _MODULE),
    "a_default_computed_from_a_constant_is_not_detected": (
        _DC + "def f(x=dev_constants.WIDGET_TIMEOUT_S * 3600.0):\n    return x\n", _MODULE
    ),
    "a_class_attribute_that_is_not_a_constant": (_DC + "class C:\n    X = dev_constants.helper()\n", _MODULE),
    "a_different_module_attribute": ("import os\nX = os.sep\n", _MODULE),
}


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_R4B_VIOLATING))
def test_fixture_r4b_literal_default_of_get_tunable_is_flagged(name: str) -> None:
    assert "R4b" in _rules(_R4B_VIOLATING[name]), name


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_R4B_CONFORMING) + sorted(_R5_CONFORMING))
def test_fixture_r4b_and_r5_conforming_source_passes_every_rule(name: str) -> None:
    source, path = (_R4B_CONFORMING[name], _MODULE) if name in _R4B_CONFORMING else _R5_CONFORMING[name]
    assert _findings(source, path) == [], name


@_NBF_SYNTHETIC
@pytest.mark.parametrize("name", sorted(_R5_VIOLATING))
def test_fixture_r5_alias_and_default_are_flagged(name: str) -> None:
    rule, source = _R5_VIOLATING[name]
    assert rule in _rules(source), name


# ---------------------------------------------------------------------------
# Settings derived from source (nothing typed twice)
# ---------------------------------------------------------------------------


@_NBF_SYNTHETIC
def test_fixture_derive_settings_reads_dev_constants_and_every_register_call(tmp_path) -> None:
    import ast

    dev = tmp_path / "dev_constants.py"
    dev.write_text(
        "from __future__ import annotations\nimport os\n_PRIVATE = 5\nA_COUNT: int = 3\nB_NAME = 'x.db'\nC_FLAG = True\n"
        "D_SIZE = 20 * 1024 * 1024\nE_KINDS = frozenset({'p', 'q'})\n\ndef helper():\n    return 1\n",
        encoding="utf-8",
    )
    trees = {
        "brain/area.py": ast.parse(
            "from brain import tunables\n"
            "_BASE_DAYS = 14.0\n"
            "RETRY: float = tunables.register('area.retry', 20 * 1024 * 1024, description='d')\n"
            "KINDS = tunables.register('area.kinds', ['a/b', 'c/d'], description='d')\n"
            "WINDOW = tunables.register('area.window', _BASE_DAYS, description='d')\n"
            "COMPUTED = tunables.register('area.computed', _compute(1), description='d')\n"
            "AMBIG = tunables.register('area.ambig', _TWICE, description='d')\n"
            "_TWICE = 1\n"
            "TABLE = [('x', tunables.register('area.in_table', 7.5, description='d'))]\n"
            "def helper(register):\n    return register(stop)\n"
        ),
        "brain/other.py": ast.parse("_TWICE = 2\nfrom brain import tunables\nz = tunables.get_tunable('area.retry', 1)\n"),
        "brain/tunables.py": ast.parse("def register(key, default, description=''):\n    return default\nA = register('chat.lull', 600.0, description='d')\n"),
    }
    settings = derive_settings(trees, dev)
    assert settings.dev == {
        "A_COUNT": 3, "B_NAME": "x.db", "D_SIZE": 20971520, "E_KINDS": frozenset({"p", "q"}),
    }
    keys = {k: (t.default, t.bound_name, t.owner_file) for k, t in settings.tunables.items()}
    assert keys == {
        "area.retry": (20971520, "RETRY", "brain/area.py"),
        "area.kinds": (["a/b", "c/d"], "KINDS", "brain/area.py"),
        "area.window": (14.0, "WINDOW", "brain/area.py"),  # a name resolved one hop to its unique definition
        "area.computed": (None, "COMPUTED", "brain/area.py"),  # a call does not fold: key and name still count
        "area.ambig": (None, "AMBIG", "brain/area.py"),  # defined in two files: not unique, not resolved
        "area.in_table": (7.5, None, "brain/area.py"),
        "chat.lull": (600.0, "A", "brain/tunables.py"),
    }
    assert ("brain/area.py", 2) in settings.definition_sites  # the one-hop definition is the tunable's own default


# ---------------------------------------------------------------------------
# The app: numbers and strings of TypeScript and Rust that mirror a Python setting
# ---------------------------------------------------------------------------


@_NBF_SYNTHETIC
def test_fixture_app_numbers_folds_products_and_skips_comments_strings_and_test_modules() -> None:
    ts = "const A = 20 * 1024 * 1024;\n// 99999\nconst B = '12345' + 1_000 * 20;\n/* 77777\n 88888 */ const C = 0.5 * 4;\n"
    assert app_numbers(ts, "ts") == [(1, 20971520), (3, 20000), (5, 2.0)]
    rust = "const A: u64 = 20 * 1024 * 1024;\n// 31337\n#[cfg(test)]\nmod t { const B: u64 = 424242; }\n"
    assert app_numbers(rust, "rust") == [(1, 20971520)]
    # a number inside an identifier or a longer token is not read: hex, exponents and integer suffixes are not detected
    assert app_numbers("x20 = 0x1F + 1e5 + 1024u64 + 3.14\n", "rust") == [(1, 3.14)]


@_NBF_SYNTHETIC
def test_fixture_app_flags_a_numeric_mirror_of_a_setting_in_typescript_and_rust() -> None:
    ts = app_findings("const MAX_BYTES = 20 * 1024 * 1024;\n", "app/src/a.ts", "ts", SETTINGS, alias_allowed=False)
    rs = app_findings("const MAX: u64 = 20 * 1024 * 1024;\n", "app/src-tauri/src/a.rs", "rust", SETTINGS, alias_allowed=False)
    for flagged in (ts, rs):
        assert [f.rule for f in flagged] == ["APP-R3"]
        assert flagged[0].settings == ("dev_constants.UPLOAD_CEILING_BYTES", "tunables:area.upload_cap")


@_NBF_SYNTHETIC
def test_fixture_app_flags_a_distinctive_string_and_passes_round_numbers_and_plain_words() -> None:
    assert [f.rule for f in app_findings("const F = 'widget_store.db';\n", "app/src/a.ts", "ts", SETTINGS, alias_allowed=False)] == ["APP-R3"]
    assert app_findings("const T = 60; const S = 'strict'; const D = 86400; const M = 20 * 1024;\n", "app/src/a.ts", "ts", SETTINGS, alias_allowed=False) == []


@_NBF_SYNTHETIC
def test_fixture_app_tokenizer_flags_model_ids_and_ignores_comments_prose_and_claude_cli() -> None:
    src = (
        "// const A = 'claude-sonnet-5'\n"
        "/* const B = \"sonnet\" */\n"
        "const C = 'claude-cli';\n"
        "const D = \"claude-sonnet-5\";\n"
        "const E = `Leave blank for the default`;\n"
        "const F = 'sonnet';\n"
    )
    flagged = app_findings(src, "app/src/x.ts", "ts", SETTINGS, alias_allowed=False)
    assert [f.rule for f in flagged] == ["APP-MODEL", "APP-MODEL"]
    assert "claude-sonnet-5" in flagged[0].message and "'sonnet'" in flagged[1].message
    assert [f.message for f in app_findings(src, "app/src/x.ts", "ts", SETTINGS, alias_allowed=True)] == [flagged[0].message]
    rust = 'const A: &str = "claude-opus-5";\n// "claude-sonnet-5"\n#[cfg(test)]\nmod t { const B: &str = "claude-sonnet-5"; }\n'
    assert len(app_findings(rust, "app/src-tauri/src/lib.rs", "rust", SETTINGS, alias_allowed=False)) == 1
    assert [t for _l, t in ts_tokens(src)] == ["claude-cli", "claude-sonnet-5", "Leave blank for the default", "sonnet"]
    assert [t for _l, t in rust_tokens(rust)] == ["claude-opus-5"]


# ---------------------------------------------------------------------------
# The ratchet (over synthetic lists, never the real one)
# ---------------------------------------------------------------------------


def _finding(rule="R3", file="brain/a.py", site="in f", settings=("dev_constants.A",), line=7) -> Finding:
    return Finding(rule, file, site, tuple(settings), line, f"{file}:{line}: {rule} at {site}")


def _entry(rule="R3", file="brain/a.py", site="in f", settings=("dev_constants.A",), reason="a stated reason"):
    return (rule, file, site, tuple(settings), reason)


@_NBF_SYNTHETIC
def test_ratchet_a_finding_that_is_not_listed_fails_and_names_rule_file_site_and_settings() -> None:
    problems = ratchet_problems([_finding(), _finding(file="brain/b.py", site="assign X", settings=("dev_constants.A", "tunables:k"))], [_entry()], 1)
    assert len(problems) == 1
    text = problems[0]
    assert "R3" in text and "brain/b.py" in text and "assign X" in text and "dev_constants.A" in text and "tunables:k" in text
    assert "not in ALLOW" in text


@_NBF_SYNTHETIC
def test_ratchet_a_listed_entry_without_a_finding_fails_and_says_delete_it() -> None:
    problems = ratchet_problems([], [_entry(), _entry(file="brain/gone.py")], 2)
    assert len(problems) == 2 and all("delete" in p for p in problems) and any("brain/gone.py" in p for p in problems)


@_NBF_SYNTHETIC
def test_ratchet_is_a_multiset_a_second_copy_at_a_listed_site_is_new() -> None:
    assert len(ratchet_problems([_finding(), _finding()], [_entry()], 1)) == 1
    assert ratchet_problems([_finding(), _finding()], [_entry(), _entry()], 2) == []


@_NBF_SYNTHETIC
def test_ratchet_size_must_equal_the_list_and_says_whether_it_grew_or_shrank() -> None:
    grown = ratchet_problems([_finding()], [_entry()], 0)
    shrunk = ratchet_problems([_finding()], [_entry()], 3)
    assert len(grown) == 1 and "grew" in grown[0] and "review" in grown[0]
    assert len(shrunk) == 1 and "shrank" in shrunk[0] and "lower ALLOW_SIZE" in shrunk[0]
    assert ratchet_problems([_finding()], [_entry()], 1) == []


@_NBF_SYNTHETIC
def test_ratchet_an_entry_without_a_reason_fails() -> None:
    assert len(ratchet_problems([_finding()], [_entry(reason="")], 1)) == 1
    assert len(ratchet_problems([_finding()], [_entry(reason="   ")], 1)) == 1


@_NBF_SYNTHETIC
def test_ratchet_counts_findings_by_identity_without_line_numbers() -> None:
    """An edit that moves a copy to another line does not break the list; the line is only in the message."""
    assert ratchet_problems([_finding(line=10)], [_entry()], 1) == []


# ---------------------------------------------------------------------------
# The guard over the real tree (brain/ Python plus the app's TypeScript and Rust)
# ---------------------------------------------------------------------------

# Violations that exist today: (rule, file, site, settings, reason). It is meant to shrink (see the module docstring).
_IMAGE_TYPES = ("tunables:files.image_types",)
_UPLOAD_CAPS = ("tunables:files.image_max_bytes", "tunables:files.pdf_max_bytes", "tunables:files.upload_max_bytes")
ALLOW: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    # The four kinds of image written out beside files.image_types. Different purpose, decided from the code in #531: the
    # setting says which images the read_file tool opens and can only narrow the set (the byte sniff recognises exactly these
    # four and only they have a file extension), while these constants say which images chat, upload and the image store
    # accept. Sharing one setting would let a user who drops a kind from the viewing list break storing it.
    ("R3-collection", "brain/bridge/chat.py", "assign _ALLOWED_MEDIA_TYPES", _IMAGE_TYPES,
     "different purpose: the four kinds the byte sniff can recognise, which ImageBlock accepts; files.image_types is the "
     "read_file viewing list and can only narrow it"),
    ("R3-collection", "brain/bridge/server.py", "assign _ALLOWED_UPLOAD_MEDIA_TYPES", _IMAGE_TYPES,
     "different purpose: the kinds an upload's declared type is cross-checked against the sniffed type for; "
     "files.image_types is the read_file viewing list"),
    ("R3-collection", "brain/images.py", "assign _ALLOWED_MEDIA_TYPES", _IMAGE_TYPES,
     "different purpose: the kinds the image store can save (beside _MEDIA_TYPE_TO_EXT); files.image_types is the "
     "read_file viewing list"),
    # The app mirrors the upload cap.
    ("APP-R3", "app/src/components/ChatPanel.tsx", "number 20971520", _UPLOAD_CAPS,
     "real copy of the 20 MiB upload cap (MAX_BYTES, applied to every staged file) that cannot read the setting: the "
     "TypeScript cannot import the Python tunable; tracked in #560"),
)
ALLOW_SIZE = 4


@pytest.fixture(scope="module")
def scan() -> tuple[Settings, list[Finding], frozenset[tuple[str, str, str]]]:
    """The scan of the real tree, done in setup (it costs seconds of CPU, which belongs to no test): the settings derived
    from the tree, the findings of every rule, and the model vocabulary entries that matched."""
    settings = derive_settings(brain_trees(), REPO / "brain" / "dev_constants.py")
    findings, used = scan_real_tree(settings)
    return settings, findings, used


def test_real_tree_findings_equal_the_allow_list(scan) -> None:
    _settings, findings, _used = scan
    problems = ratchet_problems(findings, ALLOW, ALLOW_SIZE)
    assert not problems, "the shared-settings guard (#531):\n" + "\n".join(problems)


@_NBF_TREE_SNAPSHOT
def test_every_model_vocabulary_entry_still_matches_something_at_its_shape(scan) -> None:
    """C19: the model case's vocabulary list cannot rot. An entry that matches nothing at its shape fails (delete it)."""
    _settings, _findings_, used = scan
    unused = [e for e in MODEL_LITERAL_ALLOW if e not in used]
    assert unused == [], f"stale model vocabulary entries (no match at their shape): {unused}"


@_NBF_TREE_SNAPSHOT
def test_the_scan_derives_both_kinds_of_setting_from_the_tree(scan) -> None:
    """The scan is not vacuous: settings of both kinds were derived. (Without it a derivation that finds nothing, and an
    ALLOW list shrunk to nothing, would pass the real-tree test with no setting guarded.)"""
    settings, _findings_, _used = scan
    assert settings.dev, "no dev constant was derived from brain/dev_constants.py"
    assert settings.tunables, "no tunables.register call was derived from brain/"


@_NBF_TREE_SNAPSHOT
def test_no_model_rule_finding_is_ever_allow_listed(scan) -> None:
    """The #529 model rules are not ratcheted: no model id or alias word is chosen outside the accessor (G1, G2, G4, G5), held
    by the app outside the picker vocabulary (APP-MODEL), or kept from a model accessor beyond the call (R4), and no
    ``set_values`` call sits outside the sanctioned writers (R1d); none of them may be put in ALLOW."""
    _settings, findings, _used = scan
    def is_model_finding(rule: str, site: str, settings: tuple[str, ...]) -> bool:
        copy_of_a_model_accessor = rule == "R4" and any(s.startswith("accessor:") for s in settings)
        return rule in {"G1", "G2", "G4", "G5", "APP-MODEL"} or copy_of_a_model_accessor or (rule == "R1" and "set_values" in site)

    assert [f.message for f in findings if is_model_finding(f.rule, f.site, f.settings)] == []
    assert [e for e in ALLOW if is_model_finding(e[0], e[2], e[3])] == []


@_NBF_TREE_SNAPSHOT
def test_app_alias_vocabulary_entries_still_hold_an_alias_word() -> None:
    """C19 for the app: a file allowed to hold the picker's alias words that holds none any more must leave the list."""
    for rel in APP_ALIAS_ALLOW:
        path = REPO / rel
        assert path.exists(), rel
        words = {t for _l, t in ts_tokens(path.read_text(encoding="utf-8")) if t in {"sonnet", "opus", "haiku"}}
        assert words, f"{rel} holds no alias-word token any more: remove it from APP_ALIAS_ALLOW"


@_NBF_TREE_SNAPSHOT
def test_the_connection_panel_no_longer_defaults_the_picker_to_sonnet() -> None:
    text = (REPO / "app" / "src" / "components" / "panels" / "ConnectionPanel.tsx").read_text(encoding="utf-8")
    assert '?? "sonnet"' not in text
