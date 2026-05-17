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
