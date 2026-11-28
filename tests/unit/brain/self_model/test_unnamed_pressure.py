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
    assert compute_unnamed_pressure({"joy": 4.0}, body_energy=8) == 0.0
    assert compute_unnamed_pressure({"joy": 4.0}, body_energy=1) > 0.0


@pytest.mark.no_base_failure(
    reason=(
        "the function compares whatever number it is given, on the base commit as well; what #650 changed is that the "
        "live tick now passes the unrounded saved level, which tests/unit/brain/test_energy_readers_one_level.py proves "
        "through the real tick and which does fail on the base"
    )
)
def test_unnamed_pressure_compares_the_unrounded_energy_so_it_is_on_only_at_the_floor():
    """Spec 4: on at 1.0, off at 1.4 and 1.5 (it was on below 1.5 when the level was rounded first)."""
    assert compute_unnamed_pressure({"joy": 4.0}, body_energy=1.0) == pytest.approx(0.4)
    assert compute_unnamed_pressure({"joy": 4.0}, body_energy=1.4) == 0.0
    assert compute_unnamed_pressure({"joy": 4.0}, body_energy=1.5) == 0.0
    # a low-arousal channel in the read still takes the signal home
    assert compute_unnamed_pressure({"grief": 2.0}, body_energy=1.0) == 0.0


def test_unnamed_pressure_is_driven_by_energy_alone_and_no_exhaustion_input_exists():
    """Spec section 3 and A12: energy <= 1.0 still fires; below the floor the deficit grows; the exhaustion input is gone."""
    assert compute_unnamed_pressure({"joy": 4.0}, body_energy=1.0) == pytest.approx(0.4)
    assert compute_unnamed_pressure({"joy": 4.0}, body_energy=0.5) == pytest.approx(0.6)
    assert compute_unnamed_pressure({"joy": 4.0}, body_energy=1.1) == 0.0
    assert compute_unnamed_pressure({"grief": 4.0}, body_energy=1.0) == 0.0
    with pytest.raises(TypeError):
        compute_unnamed_pressure({"joy": 4.0}, body_energy=8, body_exhaustion=8)  # type: ignore[call-arg]
