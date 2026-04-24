"""Tests for brain.migrator.og_reflex — extracting OG reflex arcs via AST."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from brain.migrator.og_reflex import extract_arcs_from_og


def test_extract_arcs_from_og_simple(tmp_path: Path):
    src = textwrap.dedent("""\
        REFLEX_ARCS = {
            "creative_pitch": {
                "trigger": {"creative_hunger": 9},
                "days_since_min": 0,
                "action": "generate_story_pitch",
                "output": "gifts",
                "cooldown_hours": 48,
                "description": "desc",
                "prompt_template": "You are Nell. {creative_hunger}/10."
            },
            "loneliness_journal": {
                "trigger": {"loneliness": 7},
                "days_since_min": 2,
                "action": "write_journal",
                "output": "journal",
                "cooldown_hours": 24,
                "description": "desc",
                "prompt_template": "You are Nell."
            }
        }
    """)
    path = tmp_path / "reflex_engine.py"
    path.write_text(src, encoding="utf-8")

    arcs = extract_arcs_from_og(path)
    assert len(arcs) == 2
    names = {a["name"] for a in arcs}
    assert names == {"creative_pitch", "loneliness_journal"}
    cp = next(a for a in arcs if a["name"] == "creative_pitch")
    assert cp["days_since_human_min"] == 0
    assert cp["output_memory_type"] == "reflex_gift"
    assert cp["action"] == "generate_pitch"
    assert cp["prompt_template"] == "You are Nell. {creative_hunger}/10."


def test_extract_arcs_no_dict_raises(tmp_path: Path):
    path = tmp_path / "empty.py"
    path.write_text("# no arcs here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_arcs_from_og(path)
