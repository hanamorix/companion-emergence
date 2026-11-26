"""The memory area's shared importance functions.

A memory's importance (0 to 10) slows how fast it is forgotten and how fast the feelings stored with it fade. Both read
their stretch from `importance_stretch`, so the formula and its number live here and nowhere else: the forgetting pass
(`brain.forgetting.salience`) and the emotion fade (`brain.emotion.engine_utils.fade`) cannot drift apart. The clamp of the
importance to 0 to 10 is written once, in `importance_ratio`, and the forgetting pass's fade gate (`brain.forgetting.policy`)
reads it from there too.
"""

from __future__ import annotations

from brain import dev_constants


def importance_ratio(importance: float) -> float:
    """The importance as a fraction from 0 to 1: 0 at or below importance 0, 1 at or above 10.

    A NaN importance reads as 0.0 (`max(0.0, nan)` keeps 0.0), so it fails toward the baseline and never raises.
    """
    return min(1.0, max(0.0, importance / 10.0))


def importance_stretch(importance: float) -> float:
    """The factor an importance multiplies a time scale by: 1.0 at importance 0 (it never shrinks), `1 + gain` at 10.

    The gain is `dev_constants.HORIZON_IMPORTANCE_GAIN`, read at call time.
    """
    return 1.0 + dev_constants.HORIZON_IMPORTANCE_GAIN * importance_ratio(importance)
