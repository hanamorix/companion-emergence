"""Names increment (name-recall fix): the persona-facing strings are the approved
wording, byte-exact, and no placeholder is left (S107; criteria N18, N23-N25).

This file held N2's PLACEHOLDER-marker test; the owner's wording now replaces
every placeholder. The texts below are copied from
``~/.claude/plans/name-recall-fix-wording.md`` sections 1, 4 and 7; the stage-8
harness additionally compares them with that file. The two consolidation prompts
are pinned in ``tests/unit/brain/engines/test_names_increment_consolidation.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from brain import prompt_strings
from brain.tools import schemas

_REPO = Path(__file__).resolve().parents[3]

ADD_DESCRIPTION = (
    "Add a name to your list of known names: a person, a pet, a place, or anything else with a "
    "proper name. When a known name comes up in conversation, even in passing, memories that "
    "mention it are found. If the name is also an everyday word (like Will), it is only matched "
    "when written with a capital letter partway through a sentence."
)
ADD_PARAM = "The name, spelled the way you want it kept."
REMOVE_DESCRIPTION = (
    "Remove a name from your list of known names, for example an ordinary word that was added "
    "by mistake. Removing a name does not change any memories."
)
REMOVE_PARAM = "The name to remove, as it appears on your list."
REJECTED_REPLY = (
    "{name} was previously removed from your list of known names, if you want to add it back, "
    "call add_name again with the same name as your next tool call."
)

_TELLS = ("—", "–", "delve", "tapestry", "it's not just", "not just a", "testament")


def test_the_tool_identifiers_and_texts_are_the_approved_wording():
    assert schemas.ADD_NAME_TOOL_NAME == "add_name"
    assert schemas.REMOVE_NAME_TOOL_NAME == "remove_name"
    assert schemas.ADD_NAME_TOOL_DESCRIPTION == ADD_DESCRIPTION
    assert schemas.ADD_NAME_PARAM_DESCRIPTION == ADD_PARAM
    assert schemas.REMOVE_NAME_TOOL_DESCRIPTION == REMOVE_DESCRIPTION
    assert schemas.REMOVE_NAME_PARAM_DESCRIPTION == REMOVE_PARAM


@pytest.mark.parametrize(
    ("tool", "description", "param"),
    [("add_name", ADD_DESCRIPTION, ADD_PARAM), ("remove_name", REMOVE_DESCRIPTION, REMOVE_PARAM)],
)
def test_the_schema_entries_carry_the_texts(tool, description, param):
    for table in (schemas.SCHEMAS, schemas.build_schemas("Canary")):
        entry = table[tool]
        assert entry["name"] == tool
        assert entry["description"] == description
        assert entry["parameters"]["properties"]["name"]["description"] == param
        assert entry["parameters"]["required"] == ["name"]


def test_the_rejected_reply_is_the_approved_text():
    assert prompt_strings.get("tools.add_name.rejected_reply") == REJECTED_REPLY


@pytest.mark.parametrize("text", [ADD_DESCRIPTION, ADD_PARAM, REMOVE_DESCRIPTION, REMOVE_PARAM, REJECTED_REPLY])
def test_no_em_dash_or_llm_tell(text):
    for tell in _TELLS:
        assert tell not in text.lower(), tell


def _consolidation_toml_section() -> str:
    text = (_REPO / "brain/prompt_strings.toml").read_text(encoding="utf-8")
    start = text.index("[engines.consolidation]")
    return text[start : text.index("\n[", start + 1)]


@pytest.mark.parametrize(
    "path",
    [
        "brain/tools/schemas.py",
        "brain/tools/impls/add_known_name.py",
        "brain/tools/impls/remove_known_name.py",
        "brain/chat/tool_recruit.py",
    ],
)
def test_no_placeholder_left_in_the_name_tool_files(path):
    assert "placeholder" not in (_REPO / path).read_text(encoding="utf-8").lower()


def test_no_placeholder_left_in_the_consolidation_prompts():
    assert "placeholder" not in _consolidation_toml_section().lower()


def test_app_job_descriptors_carry_the_two_labels_and_the_marked_placeholder():
    lines = (_REPO / "app/src/App.tsx").read_text(encoding="utf-8").splitlines()
    stripped = [line.strip() for line in lines]
    assert 'cosine_floor_bootstrap: "similarity pass mark setup",' in stripped
    assert 'rerank_floor_bootstrap: "re-scoring pass mark setup",' in stripped
    i = stripped.index('rerank_width_remeasure: "re-scoring cost check",')
    assert "PLACEHOLDER" in stripped[i - 1]
    # the only PLACEHOLDER in App.tsx is that marked comment
    marked = [n for n, line in enumerate(lines) if re.search("placeholder", line, re.I) and "PLACEHOLDER wording" in line]
    assert marked == [i - 1]
