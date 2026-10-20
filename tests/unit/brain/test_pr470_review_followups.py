"""Review follow-ups on #470 (the other maintainer's review of the PR head 584ec8a1): each test names the defect it pins.

Written first and observed failing on 584ec8a1 (the classifier recursion test passes there and guards the fix: the moved scan
must keep catching RecursionError). Synthetic names only; every persona is a tmp dir.
"""

from __future__ import annotations

import pytest

from brain.engines.consolidation import _make_haiku_classifier
from brain.memory.store import Memory


class _Reply:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, *_a, **_k) -> str:
        return self.reply


def _cand() -> Memory:
    return Memory.create_new(content="a synthetic candidate", memory_type="dream", domain="us", emotions={})


# --- item 1: the classifier reads a verdict out of prose with braces ---------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        '{"verdict": "new"}\nNote: the candidate quotes {braces}.',
        'Verdict: {"verdict": "new", "names": []} -- I considered {x} first',
        'Here is {the} verdict: {"verdict": "new"}',
        '{"note": "context"} then the answer {"verdict": "new"}',
    ],
)
def test_classifier_reads_a_verdict_with_braces_in_the_surrounding_prose(reply):
    # Fails on 584ec8a1: the greedy `\{.*\}` spans the stray braces, json.loads fails, UnusableClassifierReply -> the
    # candidate is quarantined for good although the model gave a valid verdict.
    decision = _make_haiku_classifier(_Reply(reply))(_cand(), [])
    assert decision.verdict == "new"


def test_classifier_treats_a_reply_nested_past_the_recursion_limit_as_unusable():
    # The scan must catch RecursionError as `json.loads` callers do (#406's tuple): escaping the classifier it would read as
    # a FAILED CALL (ServiceCallFailed -> requeue and end the pass) on every tick, not one quarantine with evidence.
    from brain.engines.consolidation import UnusableClassifierReply

    reply = '{"verdict": "new", "x": ' + "[" * 200_000 + "]" * 200_000 + "}"
    with pytest.raises(UnusableClassifierReply):
        _make_haiku_classifier(_Reply(reply))(_cand(), [])


def test_find_json_object_block_prefers_the_given_key_and_catches_recursion():
    from brain.utils.llm_output import find_json_object_block

    assert find_json_object_block('{"a": 1} {"verdict": "new"}', prefer_key="verdict") == '{"verdict": "new"}'
    assert find_json_object_block('{"a": 1} {"verdict": "new"}') == '{"a": 1}'
    assert find_json_object_block("no braces at all") is None
    deep = '{"x": ' + "[" * 200_000 + "]" * 200_000 + "}"
    assert find_json_object_block(deep, prefer_key="verdict") == deep  # the balanced fallback; the caller's loads reports it


# --- item 1b: soul review's parse_decision always returns a Decision ----------------------------------------------------


def test_soul_parse_decision_returns_a_decision_for_a_reply_nested_past_the_recursion_limit():
    # On 584ec8a1 the #478 scan catches only ValueError, so RecursionError escapes `parse_decision` (documented as "Always
    # returns a Decision") and ends the whole soul-review pass at its unguarded call.
    from brain.soul.review import parse_decision

    raw = '{"decision": "accept", "x": ' + "[" * 200_000 + "]" * 200_000 + "}"
    decision = parse_decision(raw, "cand-1")
    assert decision.decision == "defer"
    assert decision.parse_error
