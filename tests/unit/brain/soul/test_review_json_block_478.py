"""#478: the soul review's JSON block finder is string-aware, and keeps looking when the first `{...}` is not the answer.

`_extract_json_block` counted `{` and `}` without noticing string literals, so a decision whose reasoning contained a lone brace was
cut in the wrong place (or never closed) and the candidate was deferred with a parse error instead of getting the model's answer; a
preamble such as `Here is {the} verdict: {...}` made it stop at `{the}`. A valid decision was lost on that pass.
"""

from __future__ import annotations

import json

import pytest

from brain.soul.review import _extract_json_block, parse_decision


def _reply(reasoning: str, decision: str = "reject") -> str:
    return json.dumps({"decision": decision, "confidence": 8, "reasoning": reasoning})


@pytest.mark.parametrize(
    "reasoning",
    [
        "she wrote ':}' at the end",  # a closing brace inside the string
        "quoting {x which never closes",  # a lone opening brace
        'she said "}" and then {"',  # braces next to an escaped quote
        "a template {name} and an emoticon :} and a stray {",  # all of them
        "ends with a backslash \\",  # an escape right before the closing quote
    ],
)
def test_a_decision_whose_text_has_braces_is_parsed(reasoning):
    d = parse_decision(_reply(reasoning), "c1")

    assert d.decision == "reject" and d.parse_error == "" and d.reasoning == reasoning
    assert d.confidence == 8


def test_a_preamble_with_braces_does_not_hide_the_real_object():
    raw = "Here is {the} verdict: " + _reply("the real answer")

    d = parse_decision(raw, "c1")

    assert d.decision == "reject" and d.parse_error == "" and d.reasoning == "the real answer"


def test_a_reply_wrapped_in_a_fence_with_prose_still_parses():
    # PIN
    raw = "Sure!\n```json\n" + _reply("fenced") + "\n```\nHope that helps {really}"

    d = parse_decision(raw, "c1")

    assert d.decision == "reject" and d.reasoning == "fenced"


def test_the_block_finder_returns_the_object_text_itself():
    obj = _reply("x ':}' y")

    assert _extract_json_block("pre {a} " + obj + " post") == obj


def test_a_reply_with_no_braces_is_still_no_json_block():
    # PIN
    d = parse_decision("I could not decide.", "c1")

    assert d.decision == "defer" and "no JSON block" in d.parse_error


def test_a_block_that_does_not_decode_is_still_reported_as_a_decode_failure():
    # PIN: the failure reporting is unchanged when NO block decodes.
    d = parse_decision('{"decision": "accept", "confidence": }', "c1")

    assert d.decision == "defer" and "json decode failed" in d.parse_error


def test_an_earlier_object_without_a_decision_is_not_taken_for_the_answer():
    # A preamble `{}` decodes (it is an object) but carries no decision: the object that CARRIES one is the answer.
    d = parse_decision("note: {} then " + _reply("the answer"), "c1")

    assert d.decision == "reject" and d.reasoning == "the answer"


def test_when_no_object_carries_a_decision_the_first_object_is_reported_as_before():
    d = parse_decision('{"hello": "world"}', "c1")

    assert d.decision == "defer" and "invalid decision value" in d.parse_error
