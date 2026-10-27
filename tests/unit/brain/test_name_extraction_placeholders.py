"""Names increment (name-recall fix): the persona-facing strings are the approved
wording, byte-exact (S107; criteria N18, N23-N25).

The owner's wording replaced every placeholder (the last one, the `rerank_width_remeasure` label, is final: wording
file section 8); that no placeholder comes back is listed in tests/unit/test_removed_names_stay_removed.py. The texts below are copied from
``~/.claude/plans/name-recall-fix-wording.md`` sections 1, 4, 7 and 8; the stage-8
harness additionally compares them with that file. The two consolidation prompts
are pinned in ``tests/unit/brain/engines/test_names_increment_consolidation.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain import prompt_strings
from brain.tools import schemas

_REPO = Path(__file__).resolve().parents[3]

ADD_DESCRIPTION = (
    "Add a name to your list of known names: a person, a pet, a place, or anything else with a "
    "proper name. When a known name comes up in conversation, even in passing, memories that "
    "mention it are found. If the name is also an everyday word (like Will), it is only matched "
    "when written with a capital letter. Names can be up to 100 characters long."
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

TOO_LONG_REPLY = "The name {name} is over the 100 character limit and was not added."

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


def test_the_over_100_reply_is_the_approved_text():
    assert prompt_strings.get("tools.add_name.too_long_reply") == TOO_LONG_REPLY


@pytest.mark.parametrize(
    "text", [ADD_DESCRIPTION, ADD_PARAM, REMOVE_DESCRIPTION, REMOVE_PARAM, REJECTED_REPLY, TOO_LONG_REPLY]
)
def test_no_em_dash_or_llm_tell(text):
    for tell in _TELLS:
        assert tell not in text.lower(), tell


def test_app_job_descriptors_carry_the_three_final_labels():
    """Wording file section 4 (two labels) and section 8 (`rerank_width_remeasure`, final):
    each label is present at its site."""
    lines = (_REPO / "app/src/App.tsx").read_text(encoding="utf-8").splitlines()
    stripped = [line.strip() for line in lines]
    assert 'cosine_floor_bootstrap: "similarity pass mark setup",' in stripped
    assert 'rerank_floor_bootstrap: "re-scoring pass mark setup",' in stripped
    assert 'rerank_width_remeasure: "re-scoring cost check",' in stripped
