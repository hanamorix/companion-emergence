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


def test_gloss_is_bounded_so_one_sentence_means_one_line() -> None:
    """A "sentence" is only short by convention. felt_time_now's first sentence
    is 398 chars — it uses colons and parentheses, never a `. ` boundary — so
    the one-line economy the gloss exists for missed its single worst offender
    while every test reported clean. The bound is what makes the promise true.
    """
    from brain.chat.tool_inventory import _MAX_GLOSS_CHARS, _gloss
    from brain.tools.schemas import build_schemas

    # Pin the policy against a literal, not against itself. Asserting only
    # "no gloss exceeds _MAX_GLOSS_CHARS" passes at any value — set the
    # constant to 9999 and the bound is gone while the test still reports
    # clean. That is the failure this whole block exists to prevent.
    assert _MAX_GLOSS_CHARS <= 160, "the gloss bound was raised past one line"

    schemas = build_schemas("Nell")
    over = {
        name: len(_gloss(schemas[name]["description"]))
        for name in NELL_TOOL_NAMES
        if len(_gloss(schemas[name]["description"])) > _MAX_GLOSS_CHARS + 1
    }
    assert not over, f"glosses over the {_MAX_GLOSS_CHARS}-char bound: {over}"


def test_gloss_truncates_on_a_word_and_says_that_it_did() -> None:
    """Cut mid-word and the model reads a typo; cut silently and it reads a
    complete thought that isn't one. The ellipsis is the honesty, and the
    result must stay a real prefix of the source — a gloss that rewords the
    schema is a second place for the truth to drift."""
    from brain.chat.tool_inventory import _MAX_GLOSS_CHARS, _gloss

    source = "Return your felt sense of time: " + "lived_age_hours and pressure " * 20
    out = _gloss(source)

    assert len(out) <= _MAX_GLOSS_CHARS + 1  # +1 for the ellipsis itself
    assert out.endswith("…")
    assert not out[:-1].endswith(" ")
    assert source.startswith(out[:-1]), "the gloss must be a prefix, not a rewrite"


def test_schema_prose_is_british_but_identifiers_are_not() -> None:
    """Prose is free; identifiers are frozen. `crystallize_soul` is the exact
    string the model must call — anglicising it invents a tool that isn't there.
    The .py files never see the britfix hook, so this is done by hand."""
    import re

    from brain.tools.schemas import build_schemas

    schemas = build_schemas("Nell")
    # Ceiling: this pattern also matches words with no -ise form — size, seize,
    # prize, capsize, maize. None appear in a description today. If one ever
    # does, this test will demand a misspelling to pass: add an allow-list then
    # rather than respelling the word. Don't respell the word.
    american = re.compile(r"\b\w*(?:iz|yz)(?:e|es|ed|ing|ation|ations)\b")

    # Every description the model can see, not just the top-level one: a tool's
    # parameter descriptions ride along in its schema when it is recruited, so
    # they are her voice too. Scanning only the top level let two American
    # strings sit inside crystallize_soul's own parameters while this test
    # reported clean.
    offenders: dict[str, list[str]] = {}
    for name, schema in schemas.items():
        described = {name: schema.get("description", "")}
        props = (schema.get("parameters") or {}).get("properties") or {}
        for param, spec in props.items():
            described[f"{name}.{param}"] = spec.get("description", "") or ""
        for where, prose in described.items():
            hits = american.findall(prose)
            if hits:
                offenders[where] = hits
    assert not offenders, f"American prose left in schema descriptions: {offenders}"

    # Identifiers are frozen. Checked separately from the prose sweep above,
    # which cannot see them: the regex needs a word boundary after the -ize, and
    # the underscore in crystallize_soul denies it one.
    assert "crystallize_soul" in schemas, "the tool's dict key was renamed"
    assert schemas["crystallize_soul"]["name"] == "crystallize_soul"
