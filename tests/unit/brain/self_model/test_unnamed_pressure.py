"""`compute_unnamed_pressure`: the body's residual signal with no emotion channel to land on."""

from __future__ import annotations

import pytest

from brain.self_model.derived import compute_unnamed_pressure


@pytest.mark.no_base_failure(
    reason=(
        "moved unchanged from tests/unit/brain/self_model/test_derived.py (deleted with the windows it tested); "
        "compute_unnamed_pressure exists on the base, so the test passes there by design"
    )
)
def test_compute_unnamed_pressure_is_public_and_fires_only_for_an_extreme_body_with_no_low_arousal_channel():
    """The live tick calls it with the engine's short averages."""
    assert compute_unnamed_pressure({"joy": 4.0}, body_energy=8, body_exhaustion=0) == 0.0
    assert compute_unnamed_pressure({"joy": 4.0}, body_energy=1, body_exhaustion=8) > 0.0
