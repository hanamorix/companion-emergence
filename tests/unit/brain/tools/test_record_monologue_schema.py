"""Schema tests for the record_monologue tool added in v0.0.26."""
from __future__ import annotations


def test_record_monologue_in_nell_tool_names():
    from brain.tools import NELL_TOOL_NAMES
    assert "record_monologue" in NELL_TOOL_NAMES


def test_record_monologue_schema_built():
    from brain.tools.schemas import build_schemas
    schemas = build_schemas("Nell")
    assert "record_monologue" in schemas
    s = schemas["record_monologue"]
    assert s["name"] == "record_monologue"
    params = s["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) == {"monologue", "feed_digest"}
    assert params["properties"]["monologue"]["type"] == "string"
    assert params["properties"]["monologue"]["maxLength"] == 3000
    assert params["properties"]["feed_digest"]["maxLength"] == 400


def test_record_monologue_description_mentions_tool_purpose():
    from brain.tools.schemas import build_schemas
    s = build_schemas("Nell")["record_monologue"]
    desc = s["description"].lower()
    assert any(token in desc for token in ("monologue", "thought", "drift"))


def test_record_monologue_description_includes_persona_name():
    from brain.tools.schemas import build_schemas
    s = build_schemas("Iris")["record_monologue"]
    assert "Iris" in s["description"]
