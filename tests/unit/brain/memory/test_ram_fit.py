"""S220: the reranker's memory model (`brain.memory.ram_fit`). Pure arithmetic; the numbers
in the arm-B table are measurements of the RAM grid (synthetic text, the production
fp16 re-scorer with the memory pool and memory patterns off; growth = the call's peak
resident growth in MB)."""

from __future__ import annotations

import random

import pytest

from brain.memory import ram_fit
from brain.memory.ram_fit import ShapeGroup

# (pairs, length, peak growth MB): arm B of the RAM grid, 22 shapes, one call each.
ARM_B = [
    (1, 128, 0.2), (1, 256, 0.2), (1, 512, 8.4), (1, 1024, 191.9), (2, 512, 68.1),
    (3, 512, 143.8), (4, 512, 216.2), (2, 1024, 359.9), (3, 1024, 643.3), (1, 64, 0.2),
    (1, 192, 0.2), (1, 384, 0.2), (1, 640, 37.3), (1, 768, 83.2), (1, 896, 146.9),
    (4, 256, 44.1), (5, 512, 348.1), (6, 512, 425.3), (4, 1024, 880.3), (5, 1024, 1141.2),
    (6, 1024, 1390.4), (7, 1024, 1624.6),
]  # fmt: skip


def _groups(rows, *, full_at=(1, 1024)):
    return [ShapeGroup(b, ln, (b, ln) == full_at, 1, y) for b, ln, y in rows]


# --------------------------------------------------------------------------- packing


def test_a_shape_round_trips_through_its_packed_integer() -> None:
    for pairs, length, full in [(1, 1, False), (7, 1024, True), (58, 512, False), (3, 300, True)]:
        x = ram_fit.encode_shape(pairs, length, full=full)
        assert x is not None and ram_fit.decode_shape(x) == (pairs, length, full)


def test_an_unpackable_shape_is_refused() -> None:
    assert ram_fit.encode_shape(0, 10) is None
    assert ram_fit.encode_shape(2, 0) is None
    assert ram_fit.encode_shape(2, ram_fit.MAX_LENGTH + 1) is None
    assert ram_fit.encode_shape(2, ram_fit.MAX_LENGTH) is not None


def test_the_marker_does_not_change_the_length_or_the_pairs() -> None:
    plain, marked = ram_fit.encode_shape(5, 1024), ram_fit.encode_shape(5, 1024, full=True)
    assert plain != marked
    assert ram_fit.decode_shape(plain)[:2] == ram_fit.decode_shape(marked)[:2] == (5, 1024)


def test_rows_of_one_shape_with_and_without_the_marker_merge() -> None:
    plain, marked = ram_fit.encode_shape(1, 1024), ram_fit.encode_shape(1, 1024, full=True)
    other = ram_fit.encode_shape(2, 512)
    groups = ram_fit.groups_from_rows([(plain, 2, 10.0), (marked, 1, 5.0), (other, 3, 9.0)])
    assert groups == [
        ShapeGroup(1, 1024, True, 3, 15.0),
        ShapeGroup(2, 512, False, 3, 9.0),
    ]


# --------------------------------------------------------------------------- enough


def test_three_shapes_are_not_enough_however_many_samples() -> None:
    groups = [ShapeGroup(1, ln, ln == 1024, 1000, 5.0 * ln) for ln in (1024, 512, 256)]
    assert ram_fit.distinct_shapes(groups) == {(1, 1024), (1, 512), (1, 256)}
    assert ram_fit.enough_shapes(groups) is False
    assert ram_fit.fit_ram(groups) is None


def test_four_collinear_shapes_are_not_enough() -> None:
    """Equal lengths: S = L * T, so the columns (1, T, S) have rank 2."""
    groups = [ShapeGroup(b, 100, b == 1, 1, 3.0 * b) for b in (1, 2, 3, 4)]
    assert ram_fit.full_rank({(g.pairs, g.length) for g in groups}) is False
    assert ram_fit.enough_shapes(groups) is False


def test_four_independent_shapes_without_the_full_length_sample_are_not_enough() -> None:
    """The reviewer's case: short shapes only say nothing about a long first batch."""
    groups = [
        ShapeGroup(b, ln, False, 1, 1.0) for b, ln in [(1, 128), (1, 256), (1, 384), (4, 256)]
    ]
    assert ram_fit.full_rank({(g.pairs, g.length) for g in groups}) is True
    assert ram_fit.enough_shapes(groups) is False


def test_four_independent_shapes_with_the_full_length_sample_are_enough() -> None:
    groups = [ShapeGroup(1, ln, ln == 1024, 1, 1.0) for ln in (1024, 512, 256, 128)]
    assert ram_fit.enough_shapes(groups) is True
    assert ram_fit.fit_ram(groups) is not None


def test_duplicates_of_one_shape_count_once() -> None:
    groups = [ShapeGroup(1, 1024, True, 500, 0.0)] + [
        ShapeGroup(1, 512, False, 500, 0.0),
        ShapeGroup(1, 256, False, 500, 0.0),
        ShapeGroup(1, 256, False, 500, 0.0),  # the same shape again
    ]
    assert len(ram_fit.distinct_shapes(groups)) == 3
    assert ram_fit.enough_shapes(groups) is False


def test_dominated_needs_both_pairs_and_length() -> None:
    shapes = {(1, 1024), (4, 256)}
    assert ram_fit.dominated(1, 512, shapes) is True
    assert ram_fit.dominated(3, 200, shapes) is True
    assert ram_fit.dominated(2, 512, shapes) is False
    assert ram_fit.dominated(5, 100, shapes) is False


# --------------------------------------------------------------------------- the fit


def test_a_noiseless_set_recovers_its_coefficients() -> None:
    a, b, c = 7.0, 0.05, 2.0e-4
    shapes = [(1, 1024), (1, 512), (2, 256), (3, 700), (5, 100), (4, 900)]
    groups = [
        ShapeGroup(p, ln, (p, ln) == (1, 1024), 2, 2 * (a + b * p * ln + c * p * ln * ln))
        for p, ln in shapes
    ]
    fit = ram_fit.fit_ram(groups)
    assert fit is not None
    assert (fit.a, fit.b, fit.c) == pytest.approx((a, b, c), rel=1e-9)
    assert (fit.shapes, fit.samples, fit.max_pairs, fit.max_length) == (6, 12, 5, 1024)
    assert fit.predict(2, 777) == pytest.approx(a + b * 2 * 777 + c * 2 * 777 * 777)


def test_the_production_fit_of_the_arm_b_table() -> None:
    """All 22 rows, flat zone included, exactly as the window would hold them."""
    fit = ram_fit.fit_ram(_groups(ARM_B))
    assert fit is not None
    assert (fit.a, fit.b, fit.c) == pytest.approx((-50.117, 0.060322, 1.67096e-4), rel=2e-3)
    big = [(b, ln, y) for b, ln, y in ARM_B if y > 1.0]
    assert len(big) == 17
    rmse = (sum((fit.predict(b, ln) - y) ** 2 for b, ln, y in big) / len(big)) ** 0.5
    assert rmse <= 26.0
    assert abs(fit.predict(7, 1024) - 1624.6) / 1624.6 <= 0.03


def test_the_ladder_only_set_uses_the_non_negative_refit() -> None:
    """Four B=1 lengths of the same table: the free fit has a negative T slope."""
    rows = [(1, 1024, 191.9), (1, 512, 8.4), (1, 256, 0.2), (1, 128, 0.2)]
    fit = ram_fit.fit_ram(_groups(rows))
    assert fit is not None
    assert fit.b == 0.0
    assert (fit.a, fit.c) == pytest.approx((-17.42, 1.9415e-4), rel=1e-2)
    for pairs in range(1, 12):
        for length in (64, 200, 512, 1024):
            assert fit.predict(pairs + 1, length) >= fit.predict(pairs, length)
            assert fit.predict(pairs, length + 1) >= fit.predict(pairs, length)


def test_predictions_are_never_negative_and_never_fall_with_size() -> None:
    fit = ram_fit.fit_ram(_groups(ARM_B))
    assert fit is not None and fit.a < 0
    rng = random.Random(7)
    for _ in range(500):
        b, ln = rng.randint(1, 60), rng.randint(1, 1024)
        assert fit.predict(b, ln) >= 0.0
        assert fit.predict(b + 1, ln) >= fit.predict(b, ln)
        assert fit.predict(b, ln + 1) >= fit.predict(b, ln)


def test_a_fit_from_weighted_groups_equals_the_fit_from_the_expanded_samples() -> None:
    rows = [(1, 1024, 190.0), (1, 512, 9.0), (1, 256, 1.0), (1, 128, 0.5), (2, 512, 70.0)]
    grouped, expanded = [], []
    for i, (b, ln, y) in enumerate(rows):
        n = i + 2
        grouped.append(ShapeGroup(b, ln, (b, ln) == (1, 1024), n, n * y))
        expanded.extend(ShapeGroup(b, ln, (b, ln) == (1, 1024), 1, y) for _ in range(n))
    # the expanded form is one group per sample: fold them like the store would
    folded: dict = {}
    for g in expanded:
        key = (g.pairs, g.length)
        old = folded.get(key)
        folded[key] = ShapeGroup(
            g.pairs,
            g.length,
            g.full,
            (old.n if old else 0) + 1,
            (old.sum_y if old else 0.0) + g.sum_y,
        )
    fit_a, fit_b = ram_fit.fit_ram(grouped), ram_fit.fit_ram(list(folded.values()))
    assert fit_a == fit_b


def test_repeated_small_shapes_do_not_dilute_the_large_shape_prediction() -> None:
    """S74's bound: calls that add no memory must not drag the bound towards zero."""
    base = ram_fit.fit_ram(_groups(ARM_B))
    diluted_rows = _groups(ARM_B) + [ShapeGroup(4, 256, False, 1000, 1000 * 44.1)]
    diluted = ram_fit.fit_ram(diluted_rows)
    assert base is not None and diluted is not None
    assert abs(diluted.predict(7, 1024) - base.predict(7, 1024)) / base.predict(7, 1024) < 0.02


def test_the_intercept_only_model_is_used_when_nothing_grows() -> None:
    groups = [
        ShapeGroup(b, ln, (b, ln) == (1, 1024), 1, 3.0)
        for b, ln in [(1, 1024), (1, 512), (2, 256), (3, 100)]
    ]
    fit = ram_fit.fit_ram(groups)
    assert fit is not None and (fit.b, fit.c) == (0.0, 0.0) and fit.a == pytest.approx(3.0)


def test_a_window_with_no_positive_growth_is_no_model() -> None:
    """S204 in its new form: an always-zero peak is a failed measurement, not a model that
    predicts no memory (that would switch the RAM bound off silently)."""
    groups = [ShapeGroup(1, ln, ln == 1024, 3, 0.0) for ln in (1024, 512, 256, 128)]
    assert ram_fit.enough_shapes(groups) is True
    assert ram_fit.fit_ram(groups) is None


def test_the_completion_marker_row_is_not_a_shape() -> None:
    rows = [
        (ram_fit.encode_shape(1, 1024, full=True), 2, 380.0),
        (ram_fit.COMPLETE_MARKER, 1, 0.0),
        (ram_fit.encode_shape(1, 512), 1, 70.0),
    ]
    assert ram_fit.ladder_complete(rows) is True
    assert ram_fit.ladder_complete(rows[:1] + rows[2:]) is False
    groups = ram_fit.groups_from_rows(rows)
    assert [(g.pairs, g.length) for g in groups] == [(1, 512), (1, 1024)]
    assert ram_fit.encode_shape(1, 1) != ram_fit.COMPLETE_MARKER


# --------------------------------------------------------------------------- S242: the per-dimension cap


def _fit(max_pairs: int, max_length: int) -> ram_fit.RamFit:
    return ram_fit.RamFit(a=0.0, b=1.0, c=0.0, shapes=4, samples=4, max_pairs=max_pairs, max_length=max_length)


@pytest.mark.parametrize(
    ("pairs", "length", "expected"),
    [
        (1, 1024, True),  # the largest shape measured
        (1, 100, True),  # shorter
        (1, 1025, False),  # longer than any measured
        (2, 100, False),  # more pairs than any measured
        (2, 1025, False),
        (58, 1024, False),
    ],
)
def test_a_fit_with_single_pairs_only_covers_no_more_pairs_and_no_longer_a_padded_length(
    pairs: int, length: int, expected: bool
) -> None:
    assert _fit(1, 1024).within_measured(pairs, length) is expected


def test_the_cap_lifts_once_a_batch_of_more_than_one_pair_is_measured() -> None:
    """The order's condition: 'until the fit holds a measured batch of more than one pair'. The residual (a
    larger batch is then extrapolated) is the disclosed reading R6."""
    assert _fit(2, 128).within_measured(50, 5000) is True
    assert _fit(2, 128).within_measured(2, 128) is True


def test_the_dimensions_are_the_largest_measured_ones_of_the_window() -> None:
    """`max_pairs` / `max_length` of a real fit are the per-dimension maxima over all its groups, so a window
    of single pairs gives the cap and one 2-pair shape lifts it."""
    singles = _groups([(1, 1024, 191.9), (1, 512, 8.4), (1, 256, 0.2), (1, 128, 0.2)])
    capped = ram_fit.fit_ram(singles)
    assert capped is not None and (capped.max_pairs, capped.max_length) == (1, 1024)
    assert capped.within_measured(1, 1024) and not capped.within_measured(2, 1024)
    lifted = ram_fit.fit_ram(singles + _groups([(2, 128, 1.0)], full_at=(0, 0)))
    assert lifted is not None and lifted.max_pairs == 2 and lifted.within_measured(50, 1024)



# --- S244: the lift needs a batch of more than one pair AND S220's usable-fit conditions ---


def _shapes_of(groups):
    return {(g.pairs, g.length) for g in groups}


def test_a_two_pair_batch_with_only_three_distinct_shapes_gives_no_fit_and_so_no_lift() -> None:
    groups = _groups([(1, 1024, 191.9), (1, 512, 8.4), (2, 256, 10.0)])
    assert len(_shapes_of(groups)) == 3 and any(g.full for g in groups)
    assert not ram_fit.enough_shapes(groups)
    assert ram_fit.fit_ram(groups) is None


def test_a_two_pair_batch_with_four_full_rank_shapes_but_no_full_length_sample_gives_no_fit() -> None:
    groups = _groups([(1, 512, 8.4), (1, 256, 0.2), (1, 128, 0.2), (2, 128, 1.0)], full_at=(0, 0))
    assert len(_shapes_of(groups)) == 4 and ram_fit.full_rank(_shapes_of(groups))
    assert not any(g.full for g in groups)
    assert not ram_fit.enough_shapes(groups)
    assert ram_fit.fit_ram(groups) is None


def test_a_two_pair_batch_on_a_rank_deficient_set_of_shapes_gives_no_fit() -> None:
    """Four shapes of one length: the S column is the T column times the length, so the rank is 2."""
    groups = _groups([(1, 100, 5.0), (2, 100, 10.0), (3, 100, 15.0), (4, 100, 20.0)], full_at=(1, 100))
    assert len(_shapes_of(groups)) == 4 and any(g.full for g in groups)
    assert not ram_fit.full_rank(_shapes_of(groups))
    assert ram_fit.fit_ram(groups) is None


def test_the_lift_comes_with_all_the_usable_fit_conditions() -> None:
    """Four full-rank shapes, a full-length sample and a 2-pair batch: a fit exists and its cap is lifted."""
    groups = _groups([(1, 1024, 191.9), (1, 512, 8.4), (1, 256, 0.2), (2, 128, 1.0)])
    assert ram_fit.enough_shapes(groups)
    fit = ram_fit.fit_ram(groups)
    assert fit is not None and fit.max_pairs == 2 and fit.within_measured(50, 5000)


def test_the_docstrings_state_the_conditions_of_the_lift_and_the_marker_rule() -> None:
    module_doc = " ".join((ram_fit.__doc__ or "").split())
    method_doc = " ".join((ram_fit.RamFit.within_measured.__doc__ or "").split())
    for text in (module_doc, method_doc):
        assert "more than three distinct shapes" in text and "full rank" in text and "full-length sample" in text
    assert "lifts only once a batch of more than one pair is measured AND the fit is usable" in module_doc
    assert "records no marker" in module_doc and "MORE THAN ONE PAIR" in module_doc
    assert "headroom still records its completion marker" not in module_doc
