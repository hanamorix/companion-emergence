"""N2 (name-recall fix): the add-name tool, item 3 of "Names are added three ways".

Criteria C6 (tool half, S69, S70, S87). These tests reach the tool through ``ADD_NAME_TOOL_NAME``
so the owner's rename is a one-constant change; the approved wording is held by
tests/unit/test_persona_wording_snapshot.py. Naming: synthetic user = Bob, persona = Canary.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest

from brain import prompt_strings
from brain.chat.salience import SalienceSignal, assess_salience
from brain.chat.tool_recruit import REFLEXIVE_CORE, select_tools, tools_for_capability
from brain.memory import known_names as kn
from brain.memory.recall_stopwords import RECALL_STOPWORDS
from brain.tools import NELL_TOOL_NAMES
from brain.tools.dispatch import _DISPATCH, ToolDispatchError, dispatch
from brain.tools.impls import add_known_name as impl_mod
from brain.tools.impls.add_known_name import add_known_name
from brain.tools.schemas import ADD_NAME_TOOL_NAME, SCHEMAS, build_schemas


def _rows(persona: Path) -> list[tuple[str, str, str]]:
    path = kn.known_names_path(persona)
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(
            "SELECT name_lower, display, source FROM known_names ORDER BY name_lower"
        ).fetchall()
    finally:
        conn.close()


def _ctx(tmp_path: Path) -> dict:
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    return {
        "store": MemoryStore(":memory:"),
        "hebbian": HebbianMatrix(":memory:"),
        "persona_dir": tmp_path,
    }


# ---------------------------------------------------------------- the tool itself


def test_adds_a_name_with_source_tool_and_intact_display(tmp_path):
    assert add_known_name("Wren", persona_dir=tmp_path) == {"added": True, "name": "Wren"}
    assert _rows(tmp_path) == [("wren", "Wren", "tool")]
    assert "wren" in kn.load_known_names(tmp_path).names


def test_missing_list_file_is_created_on_first_write(tmp_path):
    assert not kn.known_names_path(tmp_path).exists()
    add_known_name("Wren", persona_dir=tmp_path)
    assert kn.known_names_path(tmp_path).exists()


@pytest.mark.parametrize("name", ["Grace", "Will Smith", "  Wren  "])
def test_every_non_stopword_entry_is_admitted_as_given(tmp_path, name):
    got = add_known_name(name, persona_dir=tmp_path)
    assert got == {"added": True, "name": name.strip()}
    assert len(_rows(tmp_path)) == 1
    assert _rows(tmp_path)[0][1] == name.strip()  # display form intact (S87)


@pytest.mark.parametrize("name", ["will", "THE"])
def test_a_one_word_stopword_entry_is_added_as_a_stopword_name(tmp_path, name):
    """S104 (names increment; the base refused these, S70): the kindled may add a name
    whose whole form is one recall stopword; it is flagged and matches only written
    capitalized (S151: in any position)."""
    assert name.lower() in RECALL_STOPWORDS
    got = add_known_name(name, persona_dir=tmp_path)
    assert got == {"added": True, "name": name}
    assert _rows(tmp_path) == [(name.lower(), name, "tool")]


@pytest.mark.parametrize("name", ["of the"])
def test_an_all_stopword_entry_is_admitted_as_given(tmp_path, name):
    """S144: any string the kindled adds is accepted; S106's all-stopword refusal is gone."""
    got = add_known_name(name, persona_dir=tmp_path)
    assert got == {"added": True, "name": name}
    assert _rows(tmp_path) == [(name.lower(), name, "tool")]


@pytest.mark.parametrize("name", ["Dr. Smith", "C++"])
def test_periods_and_other_characters_are_admitted(tmp_path, name):
    """S144, S153: no character rule; the name is listed with its display as given."""
    got = add_known_name(name, persona_dir=tmp_path)
    assert got == {"added": True, "name": name}
    assert [r[1] for r in _rows(tmp_path)] == [name]


def test_a_name_of_exactly_100_characters_is_added_and_101_is_not(tmp_path):
    """S153: the one limit, measured on the stripped text."""
    ok = "W" * 100
    assert add_known_name("  " + ok + "  ", persona_dir=tmp_path) == {"added": True, "name": ok}
    too_long = "W" * 100 + "x"
    got = add_known_name(too_long, persona_dir=tmp_path)
    assert got == {"added": False, "name": too_long, "message": TOO_LONG.format(name=too_long)}
    assert [r[1] for r in _rows(tmp_path)] == [ok]


# The approved reply (wording file section 9, S168): the registered text, whatever its wording is now (the wording itself
# is held by tests/unit/test_persona_wording_snapshot.py).
TOO_LONG = prompt_strings.get("tools.add_name.too_long_reply")


def test_the_over_100_reply_is_the_approved_text_for_the_stripped_name(tmp_path):
    name = "Wren " * 20 + "x"  # 101 characters once stripped
    got = add_known_name("   " + name + "   ", persona_dir=tmp_path)
    assert got == {
        "added": False,
        "name": name,
        "message": TOO_LONG.format(name=name),
    }
    assert _rows(tmp_path) == [] and not kn.known_names_path(tmp_path).exists(), "nothing written"


def test_the_length_is_measured_after_strip(tmp_path):
    ok = "Wren " * 19 + "Wren"  # 99 characters
    assert add_known_name(ok + " " * 50, persona_dir=tmp_path)["added"] is True
    assert add_known_name("W" * 100 + " " * 50, persona_dir=tmp_path)["added"] is True
    assert "message" in add_known_name(" " * 5 + "W" * 101, persona_dir=tmp_path)


@pytest.mark.parametrize("name", ["{name}" * 20])
def test_a_name_with_braces_or_percent_signs_is_echoed_unchanged(tmp_path, name):
    got = add_known_name(name, persona_dir=tmp_path)
    assert len(name) > 100
    assert got["message"] == TOO_LONG.format(name=name)
    assert got["name"] == name and got["added"] is False


def test_a_failed_write_stays_a_bare_added_false(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(kn, "_transact", boom)
    got = add_known_name("Wren", persona_dir=tmp_path)
    assert got == {"added": False, "name": "Wren"}


def test_the_over_100_reply_reaches_the_caller_through_dispatch(tmp_path):
    name = "W" * 101
    assert dispatch(ADD_NAME_TOOL_NAME, {"name": name}, **_ctx(tmp_path)) == {
        "added": False,
        "name": name,
        "message": TOO_LONG.format(name=name),
    }
    assert not kn.known_names_path(tmp_path).exists()


def test_the_100_character_limit_stated_in_the_texts_is_the_constant():
    """B2f: the number in the description and in the refusal reply is MAX_NAME_LENGTH."""
    n = kn.MAX_NAME_LENGTH
    assert re.findall(r"\d+", SCHEMAS["add_name"]["description"]) == [str(n)]
    assert re.findall(r"\d+", prompt_strings.get("tools.add_name.too_long_reply")) == [str(n)]


@pytest.mark.parametrize("bad", [None, 7, ["Wren"], {"n": "Wren"}, "", "   "])
def test_bad_argument_returns_added_false_and_never_raises(tmp_path, bad):
    got = add_known_name(bad, persona_dir=tmp_path)
    assert got["added"] is False
    assert set(got) == {"added", "name"}
    assert _rows(tmp_path) == []


def test_result_is_structured_fields_only_and_json_serialisable(tmp_path):
    got = add_known_name("Wren", persona_dir=tmp_path)
    assert set(got) == {"added", "name"}
    assert isinstance(got["added"], bool) and isinstance(got["name"], str)
    json.dumps(got)


def test_a_name_already_listed_reports_added_true_and_becomes_hers(tmp_path):
    # Names increment I2 (S105 "never one the kindled added"; the approved parameter
    # text "spelled the way you want it kept"): her add makes the row hers.
    kn.admit_names(tmp_path, ["Wren"], "gate")
    assert add_known_name("wren", persona_dir=tmp_path)["added"] is True
    assert _rows(tmp_path) == [("wren", "wren", "tool")]


def test_write_failure_reports_added_false_not_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(impl_mod, "confirm_readd", lambda *a, **k: "refused")
    assert add_known_name("Wren", persona_dir=tmp_path) == {"added": False, "name": "Wren"}


def test_the_writer_is_confirm_readd(tmp_path, monkeypatch):
    # Names increment (R3-4): the tool's one writer.
    seen = []

    def _spy(persona_dir, name, **kw):
        seen.append((Path(persona_dir), name))
        return "added"

    monkeypatch.setattr(impl_mod, "confirm_readd", _spy)
    add_known_name("Wren", persona_dir=tmp_path)
    assert seen == [(tmp_path, "Wren")]


# ---------------------------------------------------------------- registry and dispatch


def test_tool_is_registered_dispatchable_and_has_a_schema():
    assert ADD_NAME_TOOL_NAME in NELL_TOOL_NAMES
    assert ADD_NAME_TOOL_NAME in _DISPATCH and _DISPATCH[ADD_NAME_TOOL_NAME] is add_known_name
    schema = SCHEMAS[ADD_NAME_TOOL_NAME]
    assert schema["name"] == ADD_NAME_TOOL_NAME
    assert schema["parameters"]["required"] == ["name"]
    assert set(schema["parameters"]["properties"]) == {"name"}
    assert build_schemas("Canary")[ADD_NAME_TOOL_NAME]["parameters"]["required"] == ["name"]


def test_dispatch_routes_to_the_impl_in_process(tmp_path):
    got = dispatch(ADD_NAME_TOOL_NAME, {"name": "Wren"}, **_ctx(tmp_path))
    assert got == {"added": True, "name": "Wren"}
    assert _rows(tmp_path) == [("wren", "Wren", "tool")]


def test_dispatch_all_stopword_entry_is_added(tmp_path):
    got = dispatch(ADD_NAME_TOOL_NAME, {"name": "I am"}, **_ctx(tmp_path))
    assert got == {"added": True, "name": "I am"}
    assert _rows(tmp_path) == [("i am", "I am", "tool")]


def test_dispatch_requires_the_name_argument(tmp_path):
    with pytest.raises(ToolDispatchError, match="missing required argument"):
        dispatch(ADD_NAME_TOOL_NAME, {}, **_ctx(tmp_path))


def test_tool_is_dispatched_in_the_mcp_process_not_routed_to_the_bridge():
    from brain.mcp_server.tools import _BRIDGE_ROUTED_TOOLS

    assert ADD_NAME_TOOL_NAME not in _BRIDGE_ROUTED_TOOLS


# ---------------------------------------------------------------- always-available tier (S69)


def test_tool_is_in_the_reflexive_core():
    assert ADD_NAME_TOOL_NAME in REFLEXIVE_CORE


def test_non_maximal_turn_with_no_memory_flag_recruits_it():
    """S69: present on an ordinary turn, not only when memory salience fires."""
    signal = assess_salience("ok")
    assert signal.score < 0.999
    assert not (signal.references_past or signal.mentions_entity_or_date or signal.topic_shift)
    assert ADD_NAME_TOOL_NAME in select_tools(signal)


def test_maximal_signal_and_capability_expansion_still_include_it():
    assert ADD_NAME_TOOL_NAME in select_tools(SalienceSignal.maximal())
    for capability in ("files", "memory", "works"):
        assert ADD_NAME_TOOL_NAME in tools_for_capability(capability)


def test_core_membership_is_exact_no_other_tool_moved():
    """The existing eleven stay; the two name tools are the additions (S69; I4)."""
    from brain.tools.schemas import REMOVE_NAME_TOOL_NAME

    assert REFLEXIVE_CORE[-1] == REMOVE_NAME_TOOL_NAME
    assert REFLEXIVE_CORE[:-2] == (
        "record_monologue",
        "recall_monologue",
        "reach_for_capability",
        "get_emotional_state",
        "get_body_state",
        "get_soul",
        "get_personality",
        "felt_time_now",
        "pressure_since",
        "search_memories",
        "read_full_memory",
    )
    assert REFLEXIVE_CORE[-2] == ADD_NAME_TOOL_NAME
