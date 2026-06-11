"""Tests for build_detector_system_prompt only_categories restriction."""
from __future__ import annotations

from brain.attunement.prompts import build_detector_system_prompt


def test_restriction_phrase_present_when_only_categories_given():
    prompt = build_detector_system_prompt(frozenset({"relational"}))
    assert "relational" in prompt
    assert "FOR THIS PASS ONLY" in prompt


def test_no_restriction_phrase_when_only_categories_is_none():
    prompt = build_detector_system_prompt(None)
    assert "FOR THIS PASS ONLY" not in prompt


def test_run_detector_threads_only_categories_into_system_prompt():
    """run_detector(..., only_categories=...) must pass it into the prompt."""
    from unittest.mock import patch

    from brain.attunement.detector import run_detector
    from brain.attunement.store import BufferTurn

    turns = [BufferTurn(id="t1", content="hello world")]
    cats = frozenset({"topic_affinity", "response_shape"})

    with patch("brain.attunement.detector.build_detector_system_prompt") as mock_build, \
         patch("brain.attunement.detector._call_haiku", return_value=""):
        mock_build.return_value = "mocked prompt"
        run_detector(buffer_slice=turns, reply_text="", only_categories=cats)

    mock_build.assert_called_once_with(only_categories=cats, companion_name="")
