"""Test that the D-reflection task-frame prompt does not contain false promises.

draft_space has no reader and no automatic re-ingestion path. The prompt must
not tell the model (or the persona) that a demoted draft will resurface in a
soul review or be seen again — that mechanism doesn't exist.
"""

from __future__ import annotations

from brain.initiate.reflection import _TASK_FRAME_TEMPLATE


def test_task_frame_does_not_promise_re_ingestion():
    """_TASK_FRAME_TEMPLATE must not claim demoted drafts resurface automatically."""
    text = _TASK_FRAME_TEMPLATE.lower()
    assert "see it again" not in text, (
        "_TASK_FRAME_TEMPLATE promises re-ingestion via 'see it again' — "
        "draft_space has no reader; this is a false promise"
    )
    assert "next soul review" not in text, (
        "_TASK_FRAME_TEMPLATE promises draft resurfaces in 'next soul review' — "
        "that re-ingestion path does not exist"
    )
