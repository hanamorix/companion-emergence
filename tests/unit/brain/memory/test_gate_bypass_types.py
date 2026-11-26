"""Pin the exact membership of the consolidation gate's bypass list.

Memory types in ``GATE_BYPASS_TYPES`` skip the consolidation gate and land in recall at once. The emotion rework
(Engine 7) dropped two of its four members, the emotion-only row types that nothing writes any more; an addition
and a removal must both fail here, so neither happens by accident.
"""

from __future__ import annotations


def test_gate_bypass_types_are_exactly_these():
    from brain.memory.pending import GATE_BYPASS_TYPES

    assert GATE_BYPASS_TYPES == frozenset({"journal_entry", "initiate_outbound"}), (
        "GATE_BYPASS_TYPES changed: the plan pins exactly journal_entry and initiate_outbound "
        "(an addition and a removal both fail here)"
    )
