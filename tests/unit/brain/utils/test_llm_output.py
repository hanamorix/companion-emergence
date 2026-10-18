"""Tests for brain.utils.llm_output."""

from __future__ import annotations

import pytest

from brain.utils.llm_output import extract_json_object


def test_extract_json_object_pure_json():
    raw = '{"score": 0.7}'
    assert extract_json_object(raw) == '{"score": 0.7}'


def test_extract_json_object_with_prose():
    raw = 'Here you go:\n{"score": 0.7}\nThanks.'
    assert extract_json_object(raw) == '{"score": 0.7}'


def test_extract_json_object_with_fence():
    raw = 'Result:\n```json\n{"score": 0.7}\n```\nDone.'
    assert extract_json_object(raw) == '{"score": 0.7}'


def test_extract_json_object_with_fence_no_lang():
    raw = '```\n{"score": 0.7}\n```'
    assert extract_json_object(raw) == '{"score": 0.7}'


def test_extract_json_object_raises_when_no_json():
    with pytest.raises(ValueError):
        extract_json_object("no JSON here at all")


def test_loads_json_object_prefers_the_whole_reply_over_a_fenced_fragment():
    """#364 review: a valid reply whose string value holds a fenced {} is the reply itself."""
    from brain.utils.llm_output import loads_json_object

    raw = '{"decision": "send_quiet", "reasoning": "```\\n{}\\n```"}'
    assert loads_json_object(raw)["decision"] == "send_quiet"


def test_loads_json_object_falls_back_to_the_tolerant_extract():
    from brain.utils.llm_output import loads_json_object

    assert loads_json_object('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}


def test_loads_json_object_rejects_a_non_object():
    import pytest

    from brain.utils.llm_output import loads_json_object

    with pytest.raises(ValueError):
        loads_json_object("[1, 2]")
