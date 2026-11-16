"""The recall_monologue description must say only what is true (#688)."""
from __future__ import annotations

from brain.tools.schemas import SCHEMAS

NEW = (
    "Search your own recent interior for an earlier thought. Each match comes back as you kept it, "
    "with when you had it: the short note you made at the time, or the whole thought if it is from "
    "before you kept notes."
)


def test_recall_monologue_description_is_true_in_both_states():
    desc = SCHEMAS["recall_monologue"]["description"]
    assert desc == NEW
    for word in ("verbatim", "blurred", "vivid", "—"):
        assert word not in desc
