"""Tests for brain/tools/schemas.py — SCHEMAS dict + LOVE_TYPES constant."""

from __future__ import annotations

from brain.tools import NELL_TOOL_NAMES
from brain.tools.schemas import LOVE_TYPES, SCHEMAS


def test_schemas_has_all_dispatched_tools() -> None:
    """SCHEMAS contains a schema entry for every name in NELL_TOOL_NAMES."""
    assert set(SCHEMAS.keys()) == set(NELL_TOOL_NAMES)


def test_each_schema_has_name_description_parameters() -> None:
    """Every schema has a name, description, and parameters block."""
    for tool_name, schema in SCHEMAS.items():
        assert "name" in schema, f"{tool_name} missing 'name'"
        assert "description" in schema, f"{tool_name} missing 'description'"
        assert "parameters" in schema, f"{tool_name} missing 'parameters'"
        assert isinstance(schema["description"], str), f"{tool_name} description not a string"
        assert len(schema["description"]) > 0, f"{tool_name} description is empty"


def test_each_schema_name_matches_key() -> None:
    """Each schema's 'name' field matches the dict key."""
    for key, schema in SCHEMAS.items():
        assert schema["name"] == key, f"key={key!r} but schema name={schema['name']!r}"


def test_parameters_have_required_field() -> None:
    """Each schema's parameters block has a 'required' key (list)."""
    for tool_name, schema in SCHEMAS.items():
        params = schema["parameters"]
        assert "required" in params, f"{tool_name} parameters missing 'required'"
        assert isinstance(params["required"], list), f"{tool_name} 'required' is not a list"


def test_required_fields_present_in_properties_for_gated_tools() -> None:
    """Tools with non-empty required lists have matching property definitions."""
    for tool_name, schema in SCHEMAS.items():
        params = schema["parameters"]
        required = params.get("required", [])
        properties = params.get("properties", {})
        for field in required:
            assert field in properties, f"{tool_name} required field {field!r} not in properties"


def test_love_types_is_populated() -> None:
    """LOVE_TYPES dict is non-empty with string keys and values."""
    assert len(LOVE_TYPES) > 0
    for k, v in LOVE_TYPES.items():
        assert isinstance(k, str)
        assert isinstance(v, str)
        assert len(k) > 0
        assert len(v) > 0


def test_love_types_has_canonical_entries() -> None:
    """LOVE_TYPES contains expected canonical entries from OG nell_brain.py."""
    assert "romantic" in LOVE_TYPES
    assert "identity" in LOVE_TYPES
    assert "craft" in LOVE_TYPES
    assert "defiant" in LOVE_TYPES
    assert "eternal" in LOVE_TYPES


def test_crystallize_soul_schema_references_love_types() -> None:
    """crystallize_soul description mentions at least one LOVE_TYPES key."""
    schema = SCHEMAS["crystallize_soul"]
    desc = schema["description"]
    # The description should list at least one love type
    assert any(k in desc for k in LOVE_TYPES), (
        "crystallize_soul description should reference LOVE_TYPES values"
    )


def test_add_memory_required_fields() -> None:
    """add_memory requires content, memory_type, domain, emotions."""
    schema = SCHEMAS["add_memory"]
    required = schema["parameters"]["required"]
    assert "content" in required
    assert "memory_type" in required
    assert "domain" in required
    assert "emotions" in required


def test_search_memories_required_fields() -> None:
    """search_memories requires only 'query'; emotion and limit are optional."""
    schema = SCHEMAS["search_memories"]
    required = schema["parameters"]["required"]
    assert "query" in required
    assert "emotion" not in required
    assert "limit" not in required


def test_no_arg_tools_have_empty_required() -> None:
    """Tools that take no args have empty required lists and empty properties."""
    # get_body_state now takes an optional session_hours arg — excluded from this set.
    no_arg_tools = {"get_emotional_state", "get_personality", "boot", "get_soul"}
    for tool_name in no_arg_tools:
        schema = SCHEMAS[tool_name]
        assert schema["parameters"]["required"] == [], (
            f"{tool_name} should have empty required list"
        )
        assert schema["parameters"]["properties"] == {}, f"{tool_name} should have empty properties"


def test_get_body_state_schema_has_optional_session_hours() -> None:
    """get_body_state schema declares session_hours as an optional number."""
    schema = SCHEMAS["get_body_state"]
    assert schema["parameters"]["required"] == []
    props = schema["parameters"]["properties"]
    assert "session_hours" in props
    assert props["session_hours"]["type"] == "number"


def test_build_schemas_replaces_nell_with_companion_name() -> None:
    """build_schemas('Mira') must return schemas with 'Mira' instead of 'Nell'."""
    from brain.tools.schemas import build_schemas

    schemas = build_schemas("Mira")
    for name, schema in schemas.items():
        desc = schema.get("description", "")
        assert "Nell" not in desc, (
            f"Schema '{name}' still contains 'Nell' after build_schemas('Mira'): {desc!r}"
        )


def test_build_schemas_does_not_mutate_original_schemas() -> None:
    """build_schemas must not modify the module-level SCHEMAS constant."""
    from brain.tools.schemas import SCHEMAS, build_schemas

    original_desc = SCHEMAS["get_soul"]["description"]
    build_schemas("Mira")
    assert SCHEMAS["get_soul"]["description"] == original_desc


def test_build_schemas_with_nell_is_identical_to_schemas() -> None:
    """build_schemas('Nell') must be a semantically equivalent copy of SCHEMAS."""
    from brain.tools.naming import fill_tool_prefix
    from brain.tools.schemas import SCHEMAS, build_schemas

    built = build_schemas("Nell")
    for name in SCHEMAS:
        assert built[name]["description"] == fill_tool_prefix(SCHEMAS[name]["description"], ""), (
            f"build_schemas('Nell') description differs for '{name}'"
        )


def test_build_tools_list_uses_companion_name() -> None:
    """build_tools_list('Mira') must produce tool descriptions naming 'Mira', not 'Nell'."""
    from brain.chat.tool_loop import build_tools_list

    tools = build_tools_list("Mira")
    for tool in tools:
        desc = tool["function"].get("description", "")
        assert "Nell" not in desc, (
            f"Tool {tool['function']['name']!r} still contains 'Nell': {desc!r}"
        )


# The owner-ratified wording for #371 (decisions.md rulings 1-2), pinned EXACTLY. This text is in the cached
# prompt prefix: any edit to it is a deliberate, reviewed change, not drift.
_SOURCE_MEMORY_ID_PROPERTY = {
    "type": "string",
    "description": (
        "Optional. The id of an existing memory this moment is about; it is kept from being forgotten for as long "
        "as this crystallisation stands. Pass an id only if it is already in this conversation: a recalled memory "
        "line, a search or read result, or the id {tool_prefix}add_memory just returned. Never search just to find one. If the "
        "moment is not a memory yet and you want it kept, save it with {tool_prefix}add_memory first and pass the id it returns."
    ),
}


def test_crystallize_soul_source_memory_id_property_is_exactly_the_ratified_one() -> None:
    """#371 C6(b)/(e): one optional string property, last, with the exact ratified wording, and no
    persona name in the raw static text (build_schemas() templates 'Nell' away, which would hide it)."""
    schema = SCHEMAS["crystallize_soul"]
    props = schema["parameters"]["properties"]

    assert list(props) == ["moment", "love_type", "why_it_matters", "who_or_what", "resonance", "source_memory_id"]
    assert props["source_memory_id"] == _SOURCE_MEMORY_ID_PROPERTY
    assert "source_memory_id" not in schema["parameters"]["required"]
    assert "Nell" not in props["source_memory_id"]["description"]
    assert "Hana" not in props["source_memory_id"]["description"]


def test_crystallize_soul_schema_rejects_non_string_ids_the_way_the_mcp_sdk_validates() -> None:
    """#371 C3 (MCP path) / m1: the MCP SDK runs jsonschema.validate(arguments, inputSchema) BEFORE the tool
    handler, so null / int / list never reach the impl over MCP; blank and real ids do. This pins the plain
    ``string`` type (a ["string","null"] union would let null through and change the cached schema)."""
    import jsonschema
    import pytest

    from brain.tools.schemas import build_schemas

    params = build_schemas("Companion")["crystallize_soul"]["parameters"]
    base = {"moment": "m", "love_type": "craft", "why_it_matters": "w"}

    for accepted in ("", "   ", "m-123"):
        jsonschema.validate({**base, "source_memory_id": accepted}, params)
    for rejected in (None, 5, ["m-1"]):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**base, "source_memory_id": rejected}, params)
