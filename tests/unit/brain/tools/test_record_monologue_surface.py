from brain.tools.schemas import build_schemas


def test_record_monologue_schema_has_surface_flag():
    schemas = build_schemas("Nell")
    props = schemas["record_monologue"]["parameters"]["properties"]
    assert "surface" in props
    assert props["surface"]["type"] == "boolean"
    # surface is optional (defaults true at the handler) — not in `required`
    assert "surface" not in schemas["record_monologue"]["parameters"]["required"]
