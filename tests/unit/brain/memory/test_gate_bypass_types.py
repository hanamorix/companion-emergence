"""Pin the exact membership of the consolidation gate's bypass list.

Memory types in ``GATE_BYPASS_TYPES`` skip the consolidation gate and land in recall at once. The list is
changed on purpose by the emotion rework (Engine 7 drops two of its four members); until then an addition
and a removal must both fail here, so neither happens by accident.
"""

from __future__ import annotations

import pytest


@pytest.mark.no_base_failure(
    reason="pins the exact membership of the gate-bypass set; passes on the current code by design"
)
def test_gate_bypass_types_are_exactly_these():
    from brain.memory.pending import GATE_BYPASS_TYPES

    assert GATE_BYPASS_TYPES == frozenset(
        {"journal_entry", "initiate_outbound", "monologue_emotion", "self_model_reconcile"}
    ), (
        "GATE_BYPASS_TYPES changed: the plan pins exactly journal_entry, initiate_outbound, "
        "monologue_emotion and self_model_reconcile (an addition and a removal both fail here)"
    )
