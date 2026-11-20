"""The `record_monologue` digest is written in first person (emotion rework B1 PR 7).

Once the engine is live, the digest is the stored note itself: it comes back in her interior-continuity block under
"your recent thoughts, as you noted them" (#688) and in `recall_monologue`. A third-person note reads as someone else's.
This changes the voice of the inner-life Feed, which shows the same digest.
"""

from __future__ import annotations

from brain.tools.schemas import SCHEMAS


def _props() -> dict:
    return SCHEMAS["record_monologue"]["parameters"]["properties"]


def test_digest_and_surface_descriptions_are_first_person() -> None:
    digest = _props()["feed_digest"]["description"]
    surface = _props()["surface"]["description"]
    for text in (digest, surface):
        assert "third-person" not in text and "third person" not in text
    assert "first person" in digest
    assert "'I " in digest  # its example is a first-person note
    assert "she " not in digest
