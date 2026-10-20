"""#479: the chat extractor and the attunement detector accept a JSON reply that has prose around its fence.

Both stripped a code fence only when the reply STARTED with one and ended with one, so "Here is the JSON:" before the fence, or a
sentence after the closing fence, made `json.loads` fail and the turn's extraction (or attunement read) was thrown away. They now
read the reply with the shared tolerant helper `brain.utils.llm_output.loads_json_object`, keeping their own validation and the
parse-failure log (#364).
"""

from __future__ import annotations

import json

import pytest

from brain.attunement.detector import _parse_output
from brain.chat.extractor import ExtractorOutput, extract_from_thinking
from brain.health.parse_failures import PARSE_FAILURE_LOG

_EX = {"memory_writes": [{"episode": "noticed the gap", "salience": 0.3}]}
_DET = {"current_read": {"tone_label": "warm", "mood_valence": 0.5}, "pattern_candidates": []}

_WRAPPINGS = {
    "prose-before-fence": lambda body: "Here is the JSON:\n```json\n" + body + "\n```",
    "prose-after-fence": lambda body: "```json\n" + body + "\n```\nHope that helps!",
    "prose-both-sides": lambda body: "Sure thing.\n```\n" + body + "\n```\nLet me know if you want changes.",
    "bare-object-with-prose": lambda body: "My answer: " + body + " (done)",
    "fence-only": lambda body: "```json\n" + body + "\n```",  # PIN
    "bare-json": lambda body: body,  # PIN
}


class _Provider:
    def __init__(self, text):
        self.text = text

    def generate(self, prompt, *, system=None):
        return self.text


def _rows(persona_dir):
    path = persona_dir / PARSE_FAILURE_LOG
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


@pytest.mark.parametrize("wrap", sorted(_WRAPPINGS))
def test_the_extractor_reads_the_object_whatever_surrounds_it(tmp_path, wrap):
    reply = _WRAPPINGS[wrap](json.dumps(_EX))

    out = extract_from_thinking(
        provider=_Provider(reply), monologue_blocks=("m",), visible_reply="hi", recent_turn_context=(), persona_dir=tmp_path
    )

    assert [w.episode for w in out.memory_writes] == ["noticed the gap"]
    assert _rows(tmp_path) == []  # no parse failure recorded


@pytest.mark.parametrize("wrap", sorted(_WRAPPINGS))
def test_the_detector_reads_the_object_whatever_surrounds_it(tmp_path, wrap):
    reply = _WRAPPINGS[wrap](json.dumps(_DET))

    out = _parse_output(reply, "turn-1", tmp_path)

    assert out.current_read.tone_label == "warm"
    assert _rows(tmp_path) == []


def test_an_extractor_reply_with_no_object_is_still_recorded_and_empty(tmp_path):
    # PIN (#364)
    reply = "I could not produce JSON this time."

    out = extract_from_thinking(
        provider=_Provider(reply), monologue_blocks=("m",), visible_reply="hi", recent_turn_context=(), persona_dir=tmp_path
    )

    assert out == ExtractorOutput()
    (row,) = _rows(tmp_path)
    assert row["site"] == "extractor" and row["raw"] == reply and row["error"].startswith("non-JSON")


def test_a_detector_reply_with_no_object_is_still_recorded_and_declines(tmp_path):
    # PIN (#364)
    reply = "no json here"

    out = _parse_output(reply, "turn-1", tmp_path)

    assert out.current_read.tone_label == "unknown"
    (row,) = _rows(tmp_path)
    assert row["site"] == "attunement" and row["raw"] == reply and row["error"].startswith("non-JSON")


def test_a_reply_whose_object_breaks_the_schema_is_still_a_schema_failure(tmp_path):
    # PIN: validation stays each caller's own.
    reply = "Here you go: " + json.dumps({"memory_writes": [{"episode": "x", "salience": 99.0}]})

    out = extract_from_thinking(
        provider=_Provider(reply), monologue_blocks=("m",), visible_reply="hi", recent_turn_context=(), persona_dir=tmp_path
    )

    assert out == ExtractorOutput()
    (row,) = _rows(tmp_path)
    assert row["site"] == "extractor" and row["error"].startswith("schema")


# --- code review 3b, S5: an unrelated object in prose is not an answer ---------------------------------------------------


_UNRELATED = json.dumps({"note": "this is not the answer", "n": 3})


@pytest.mark.parametrize("wrap", ["prose-before-fence", "prose-after-fence", "bare-object-with-prose"])
def test_an_unrelated_object_in_prose_is_recorded_as_a_parse_failure_by_the_extractor(tmp_path, wrap):
    reply = _WRAPPINGS[wrap](_UNRELATED)

    out = extract_from_thinking(
        provider=_Provider(reply), monologue_blocks=("m",), visible_reply="hi", recent_turn_context=(), persona_dir=tmp_path
    )

    assert out == ExtractorOutput()
    (row,) = _rows(tmp_path)
    assert row["site"] == "extractor" and row["raw"] == reply and row["error"].startswith("non-JSON")


@pytest.mark.parametrize("wrap", ["prose-before-fence", "prose-after-fence", "bare-object-with-prose"])
def test_an_unrelated_object_in_prose_is_recorded_as_a_parse_failure_by_the_detector(tmp_path, wrap):
    reply = _WRAPPINGS[wrap](_UNRELATED)

    out = _parse_output(reply, "turn-1", tmp_path)

    assert out.current_read.tone_label == "unknown"
    (row,) = _rows(tmp_path)
    assert row["site"] == "attunement" and row["raw"] == reply and row["error"].startswith("non-JSON")


@pytest.mark.parametrize("wrap", ["fence-only", "bare-json"])
def test_an_empty_object_that_is_the_whole_reply_is_still_a_valid_empty_answer(tmp_path, wrap):
    # PIN: `{}` as the entire reply (bare or fenced) is a legitimate "nothing to extract"; no row, as before.
    reply = _WRAPPINGS[wrap]("{}")
    out = extract_from_thinking(
        provider=_Provider(reply), monologue_blocks=("m",), visible_reply="hi", recent_turn_context=(), persona_dir=tmp_path
    )
    assert out == ExtractorOutput()
    assert _rows(tmp_path) == []


# --- confirm review S5: `{}` with chatter around it is still the valid empty answer ---------------------------------------


@pytest.mark.parametrize("wrap", ["prose-before-fence", "prose-after-fence", "prose-both-sides", "bare-object-with-prose"])
def test_an_empty_object_with_chatter_is_a_valid_empty_answer_for_the_extractor(tmp_path, wrap):
    reply = _WRAPPINGS[wrap]("{}")
    out = extract_from_thinking(
        provider=_Provider(reply), monologue_blocks=("m",), visible_reply="hi", recent_turn_context=(), persona_dir=tmp_path
    )
    assert out == ExtractorOutput()
    assert _rows(tmp_path) == []  # nothing failed: `{}` is what a model says when there is nothing to extract


@pytest.mark.parametrize("wrap", ["prose-before-fence", "bare-object-with-prose"])
def test_an_empty_object_with_chatter_is_not_a_parse_failure_for_the_detector(tmp_path, wrap):
    out = _parse_output(_WRAPPINGS[wrap]("{}"), "turn-1", tmp_path)
    assert out.current_read.tone_label == "unknown"
    assert _rows(tmp_path) == []
