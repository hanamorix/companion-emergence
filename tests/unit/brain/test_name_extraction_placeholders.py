"""N2 (name-recall fix): INV-I10, automated half. The five strings the build introduces.

Plan section 8: the tool's identifier, its description, its `name` parameter description,
the gate-judge prompt and the re-appraiser prompt. Each is registered in one place
(``brain/prompt_strings.toml`` or ``brain/tools/schemas.py``), carries the PLACEHOLDER
marker, and holds no em-dash and no other LLM-tell phrasing. The human half (the owner
supplies or approves each string byte-exact) is not tested here and is not done: Roy's
wording is set at the end of the build (I10, S51).
"""

from __future__ import annotations

import pytest

from brain import prompt_strings
from brain.engines import consolidation as cons
from brain.tools import schemas

_TOML_KEYS = {
    "gate judge prompt": "engines.consolidation.classifier_prompt",
    "re-appraiser prompt": "engines.consolidation.reappraiser_prompt",
}

_TELLS = ("—", "–", "delve", "tapestry", "it's not just", "not just a", "testament")


def _strings() -> dict[str, str]:
    return {
        "tool identifier": schemas.ADD_NAME_TOOL_NAME,
        "tool description": schemas.ADD_NAME_TOOL_DESCRIPTION,
        "tool parameter description": schemas.ADD_NAME_PARAM_DESCRIPTION,
        "gate judge prompt": prompt_strings.get(_TOML_KEYS["gate judge prompt"]),
        "re-appraiser prompt": prompt_strings.get(_TOML_KEYS["re-appraiser prompt"]),
    }


@pytest.mark.parametrize("label", list(_strings()))
def test_each_string_carries_the_placeholder_marker(label):
    assert "placeholder" in _strings()[label].lower(), label


@pytest.mark.parametrize("label", list(_strings()))
def test_each_string_has_no_em_dash_or_llm_tell(label):
    text = _strings()[label].lower()
    for tell in _TELLS:
        assert tell not in text, (label, tell)


def test_the_prompts_are_registered_in_the_toml_and_used_by_the_consolidation_module():
    # `register` fails closed on a missing key, so importing the module already proved
    # both keys exist; this pins that the module's constants ARE the toml values.
    assert cons._CLASSIFIER_PROMPT == prompt_strings.get(_TOML_KEYS["gate judge prompt"])
    assert cons._REAPPRAISER_PROMPT == prompt_strings.get(_TOML_KEYS["re-appraiser prompt"])


def test_the_tool_strings_are_one_constant_each_and_the_schema_reads_them():
    entry = schemas.SCHEMAS[schemas.ADD_NAME_TOOL_NAME]
    assert entry["name"] == schemas.ADD_NAME_TOOL_NAME
    assert entry["description"] == schemas.ADD_NAME_TOOL_DESCRIPTION
    assert (
        entry["parameters"]["properties"]["name"]["description"]
        == schemas.ADD_NAME_PARAM_DESCRIPTION
    )


def test_the_gate_judge_prompt_asks_for_names_on_the_whole_candidate():
    text = prompt_strings.get(_TOML_KEYS["gate judge prompt"])
    assert '"names"' in text and "whole candidate" in text
    assert "duplicate" in text and "merge" in text


def test_the_reappraiser_prompt_asks_for_the_number_and_names_as_json():
    text = prompt_strings.get(_TOML_KEYS["re-appraiser prompt"])
    assert '"importance"' in text and '"names"' in text
