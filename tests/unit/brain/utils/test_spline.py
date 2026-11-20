"""brain.utils.spline.monotone_spline: exact at its points, monotone without overshoot, the research curve's values."""

from __future__ import annotations

import pytest

from brain.utils.spline import monotone_spline

# The research gate's opposing-feelings curve (spec table), as plain points so this test does not depend on the
# constant that Step 2 adds. The helper is general; these are just seven monotone points to run it on.
CURVE = [(0, 0), (1, 4), (2, 5), (3, 5.3), (5, 5.8), (7, 6.5), (10, 8)]


def test_exact_at_every_given_point():
    f = monotone_spline(CURVE)
    for x, y in CURVE:
        assert f(x) == pytest.approx(y, abs=1e-12)


def test_rises_monotonically_on_a_fine_grid_without_overshoot():
    f = monotone_spline(CURVE)
    grid = [i / 1000 for i in range(0, 10001)]
    values = [f(x) for x in grid]
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))
    # No overshoot: between two neighbouring points the curve stays inside the two y values.
    for (x0, y0), (x1, y1) in zip(CURVE, CURVE[1:], strict=False):
        for x in (x0 + (x1 - x0) * t / 50 for t in range(51)):
            assert y0 - 1e-12 <= f(x) <= y1 + 1e-12


def test_falling_data_is_monotone_too():
    f = monotone_spline([(0, 10), (1, 4), (2, 3), (5, 0)])
    values = [f(i / 100) for i in range(0, 501)]
    assert all(b <= a for a, b in zip(values, values[1:], strict=False))


def test_flat_stretch_and_turning_point_do_not_overshoot():
    f = monotone_spline([(0, 0), (1, 5), (2, 5), (3, 5), (4, 0)])
    values = [f(i / 100) for i in range(0, 401)]
    assert max(values) <= 5.0 + 1e-12
    assert min(values) >= 0.0 - 1e-12
    for x in (1.0, 1.5, 2.0, 2.5, 3.0):
        assert f(x) == pytest.approx(5.0, abs=1e-12)


# The research curve's stated values (spec 1). The spec states them rounded (2.5, 4.6, 5.55, 6.1, 6.9, 7.4); the curve's
# real values are 2.4875, 4.6423, 5.5460, 6.1212, 6.9387, 7.4393, so each differs from its rounded statement by at
# most 0.043. The tolerance of 0.05 is the spec's own (A3: 0.5 -> 2.5 +/- 0.05) and covers the rounding and no more.
@pytest.mark.parametrize(
    ("x", "stated"),
    [(0.5, 2.5), (1.5, 4.6), (4, 5.55), (6, 6.1), (8, 6.9), (9, 7.4)],
)
def test_the_research_curves_stated_values(x, stated):
    assert monotone_spline(CURVE)(x) == pytest.approx(stated, abs=0.05)


# The exact values of the one PCHIP curve through these points; the same numbers scipy's PchipInterpolator gives
# (cross-checked once by hand, never imported here). They pin the slope rules, which the rounded values above cannot.
@pytest.mark.parametrize(
    ("x", "exact"),
    [
        (0.5, 2.4875),
        (1.5, 4.642307692307693),
        (4, 5.5459608843537405),
        (6, 6.121172480620155),
        (8, 6.938656330749354),
        (9, 7.439328165374677),
    ],
)
def test_the_research_curves_exact_values(x, exact):
    assert monotone_spline(CURVE)(x) == pytest.approx(exact, abs=1e-9)


# The end-point slope has two special branches that the research curve never reaches (it uses the plain three-point
# slope at both ends), so these three-point sets reach each one at the first end and, mirrored, at the last end. The
# expected values are the ones scipy's PchipInterpolator gives for the same points (cross-checked once by hand, never
# imported here). Each value changes if its branch is dropped: without the zero rule the end slope is the three-point
# slope (-1 or the mirror), and without the 3*d0 cap it is 4 instead of 3.
@pytest.mark.parametrize(
    ("points", "expected"),
    [
        # first end: three-point slope s = -1 points against d0 = 1, so the end slope is 0
        ([(0, 0), (1, 1), (2, 6)], {0.5: 0.2916666666666667, 1.5: 2.833333333333333}),
        # first end: the first two stretches change direction and |s| = 4 > 3|d0| = 3, so the end slope is 3*d0
        ([(0, 0), (1, 1), (2, -4)], {0.5: 0.875, 1.5: -0.5}),
        # last end, the mirror of each case above
        ([(0, 6), (1, 1), (2, 0)], {0.5: 2.8333333333333335, 1.5: 0.2916666666666667}),
        ([(0, -4), (1, 1), (2, 0)], {0.5: -0.5, 1.5: 0.875}),
        # uneven stretches, cap branch
        ([(0, 0), (1, 1), (4, -2)], {0.5: 0.6875, 2.5: 0.4375}),
    ],
)
def test_end_slope_special_branches(points, expected):
    f = monotone_spline(points)
    for x, y in expected.items():
        assert f(x) == pytest.approx(y, abs=1e-12)


@pytest.mark.parametrize("xs", [[0, 1, 1, 2], [0, 2, 1, 3], [3, 2, 1, 0]])
def test_x_not_strictly_increasing_raises(xs):
    with pytest.raises(ValueError, match="strictly increasing"):
        monotone_spline([(x, float(i)) for i, x in enumerate(xs)])


@pytest.mark.parametrize("points", [[], [(0, 1)]])
def test_fewer_than_two_points_raises(points):
    with pytest.raises(ValueError, match="at least 2 points"):
        monotone_spline(points)


def test_two_points_make_a_straight_line():
    f = monotone_spline([(1, 2), (3, 10)])
    assert f(1) == pytest.approx(2)
    assert f(2) == pytest.approx(6)
    assert f(2.5) == pytest.approx(8)
    assert f(3) == pytest.approx(10)


def test_outside_the_range_holds_the_end_value_flat():
    f = monotone_spline(CURVE)
    assert f(-5) == 0.0
    assert f(-1e-9) == 0.0
    assert f(10.0001) == 8.0
    assert f(1e9) == 8.0
