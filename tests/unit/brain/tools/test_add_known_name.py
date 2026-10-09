"""N2 (name-recall fix): the add-name tool, item 3 of "Names are added three ways".

Criteria C6 (tool half, S69, S70, S87). The tool's identifier and text are PLACEHOLDERS
(I10); these tests reach the tool through ``ADD_NAME_TOOL_NAME`` so the owner's rename is
a one-constant change. Naming: synthetic user = Bob, persona = Canary.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("name", ["Grace", "Will Smith", "NY", "the Hague", "  Wren  "])
def test_every_non_stopword_entry_is_admitted_as_given(tmp_path, name):
    got = add_known_name(name, persona_dir=tmp_path)
    assert got == {"added": True, "name": name.strip()}
    assert len(_rows(tmp_path)) == 1
    assert _rows(tmp_path)[0][1] == name.strip()  # display form intact (S87)


@pytest.mark.parametrize("name", ["will", "Will", "the", "THE", "may"])
def test_stopword_entry_is_rejected_with_nothing_written(tmp_path, name):
    """S70: an entry whose lower-cased form is a recall stopword is rejected."""
    assert name.lower() in RECALL_STOPWORDS
    got = add_known_name(name, persona_dir=tmp_path)
    assert got["added"] is False
    assert _rows(tmp_path) == []
    assert not kn.known_names_path(tmp_path).exists()


@pytest.mark.parametrize("bad", [None, 7, ["Wren"], {"n": "Wren"}, "", "   ", "!!!"])
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


def test_a_name_already_listed_reports_added_true_and_keeps_the_first_row(tmp_path):
    kn.admit_names(tmp_path, ["Wren"], "gate")
    assert add_known_name("wren", persona_dir=tmp_path)["added"] is True
    assert _rows(tmp_path) == [("wren", "Wren", "gate")]  # INSERT OR IGNORE: first row kept


def test_write_failure_reports_added_false_not_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(impl_mod, "admit_names", lambda *a, **k: [])
    assert add_known_name("Wren", persona_dir=tmp_path) == {"added": False, "name": "Wren"}


def test_the_writer_goes_through_admit_names_with_source_tool(tmp_path, monkeypatch):
    seen = []

    def _spy(persona_dir, names, source, **kw):
        seen.append((Path(persona_dir), list(names), source))
        return ["wren"]

    monkeypatch.setattr(impl_mod, "admit_names", _spy)
    add_known_name("Wren", persona_dir=tmp_path)
    assert seen == [(tmp_path, ["Wren"], "tool")]


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


def test_dispatch_stopword_returns_added_false(tmp_path):
    got = dispatch(ADD_NAME_TOOL_NAME, {"name": "will"}, **_ctx(tmp_path))
    assert got["added"] is False
    assert _rows(tmp_path) == []


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
    """The existing eleven stay; the add-name tool is the one addition."""
    assert REFLEXIVE_CORE[:-1] == (
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
    assert REFLEXIVE_CORE[-1] == ADD_NAME_TOOL_NAME
