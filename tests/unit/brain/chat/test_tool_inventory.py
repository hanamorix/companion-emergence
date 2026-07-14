"""Tests for brain.chat.tool_inventory — the derived toolset block."""

from __future__ import annotations

from brain.chat.tool_inventory import build_tool_inventory
from brain.tools import NELL_TOOL_NAMES


def test_inventory_names_every_registered_tool() -> None:
    """The whole point: the model's picture of its faculties is DERIVED from the
    registry, so it can never drift from reality (#69 — the hand-written list
    lost 14 of 27 tools over ~28 versions)."""
    out = build_tool_inventory("Nell")
    for name in NELL_TOOL_NAMES:
        assert f"`{name}`" in out, f"{name!r} missing from the generated inventory"


def test_inventory_is_byte_stable() -> None:
    """CACHE INVARIANT: this block sits in the frozen system prefix. Two calls
    must be byte-identical, or every turn re-pays cache-creation instead of
    cache-read. A future edit that folds in per-turn state fails here."""
    assert build_tool_inventory("Nell") == build_tool_inventory("Nell")


def test_inventory_renders_full_registry_not_a_subset() -> None:
    """Recruitment gives the model schemas for THIS turn's tools only. The
    inventory's whole job is telling her what exists when it is not in hand, so
    it must never render the recruited subset."""
    out = build_tool_inventory("Nell")
    assert len(NELL_TOOL_NAMES) == 27
    assert sum(1 for n in NELL_TOOL_NAMES if f"`{n}`" in out) == 27


def test_inventory_carries_the_reach_valve() -> None:
    """The behavioural payload: 'I don't have that tool' is sometimes literally
    true (the file tools are not in REFLEXIVE_CORE), so she must be told to
    reach rather than to declare incapacity."""
    out = build_tool_inventory("Nell")
    assert "`reach_for_capability`" in out
    assert "never" in out.lower()


def test_inventory_uses_the_companion_name() -> None:
    out = build_tool_inventory("Phoebe")
    assert "Phoebe" in out


def test_inventory_falls_soft_on_a_tool_with_no_schema(monkeypatch) -> None:
    """The prompt must never be the thing that breaks a turn. A registered tool
    with no schema renders name-only rather than raising."""
    import brain.chat.tool_inventory as ti

    monkeypatch.setattr(ti, "NELL_TOOL_NAMES", ("get_soul", "ghost_tool"))
    out = build_tool_inventory("Nell")
    assert "`ghost_tool`" in out
    assert "`get_soul`" in out


def test_falls_soft_tool_renders_name_only_no_dangling_gloss(monkeypatch) -> None:
    """Tightens test_inventory_falls_soft_on_a_tool_with_no_schema: that test only
    checked the schema-less tool's name APPEARED, which would still pass if a
    stray `None` or gloss got appended. The implementation promises a bare
    `- `name`` line with no trailing `— gloss` — pin that exactly, and pin that
    a schema-bearing tool alongside it is unaffected and still gets its gloss."""
    import brain.chat.tool_inventory as ti

    monkeypatch.setattr(ti, "NELL_TOOL_NAMES", ("get_soul", "ghost_tool"))
    out = build_tool_inventory("Nell")
    lines = out.splitlines()
    assert "- `ghost_tool`" in lines
    assert any(line.startswith("- `get_soul` — ") for line in lines)


def test_inventory_gloss_is_one_sentence_not_the_whole_description() -> None:
    """Full descriptions are ~2194 tokens and duplicate what recruited tools
    already carry in their schemas; the first sentence is ~619."""
    from brain.tools.schemas import build_schemas

    out = build_tool_inventory("Nell")
    full = build_schemas("Nell")["compact_history"]["description"]
    assert len(full) > 200  # guard the premise: this description IS long
    assert full.strip() not in out


def test_gloss_does_not_truncate_at_an_abbreviation() -> None:
    """`. ` is not a sentence boundary when it sits inside `e.g. ` — splitting
    there ships a dangling fragment into the frozen prefix, which the model then
    reads every turn as its description of a real faculty."""
    out = build_tool_inventory("Nell")
    assert "— e.g\n" not in out
    assert "e.g. ~/Desktop" in out


def test_schema_prose_is_british_but_identifiers_are_not() -> None:
    """Prose is free; identifiers are frozen. `crystallize_soul` is the exact
    string the model must call — anglicising it invents a tool that isn't there.
    The .py files never see the britfix hook, so this is done by hand."""
    import re

    from brain.tools.schemas import build_schemas

    schemas = build_schemas("Nell")

    american = {
        name: re.findall(r"\b\w*(?:iz|yz)(?:e|es|ed|ing|ation|ations)\b", v["description"])
        for name, v in schemas.items()
    }
    offenders = {n: w for n, w in american.items() if w}
    assert not offenders, f"American prose left in schema descriptions: {offenders}"

    assert schemas["crystallize_soul"]["name"] == "crystallize_soul"
