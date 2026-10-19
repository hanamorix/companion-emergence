"""brain.utils.finite.clamp_nonfinite: the clamp and its one-warning-per-call evidence."""

from __future__ import annotations

import logging
import math

from brain.utils.finite import clamp_nonfinite


class _F(float):
    pass


def test_finite_values_are_returned_unchanged_and_nothing_is_logged(caplog) -> None:
    data = {"a": 1.5, "b": [1, 2.0, ("x", 3.0)], "c": {"d": None, "e": "nan"}, "f": True}
    with caplog.at_level(logging.WARNING, logger="brain.utils.finite"):
        out = clamp_nonfinite(data)
    assert out == {"a": 1.5, "b": [1, 2.0, ["x", 3.0]], "c": {"d": None, "e": "nan"}, "f": True}
    assert not caplog.records


def test_nan_and_both_infinities_are_replaced_in_nested_dicts_lists_and_tuples(caplog) -> None:
    data = {"a": float("nan"), "b": [1.0, float("inf"), (float("-inf"), 2.0)], "c": {"d": {"e": float("nan")}}}
    with caplog.at_level(logging.WARNING, logger="brain.utils.finite"):
        out = clamp_nonfinite(data, where="unit")
    assert out == {"a": 0.0, "b": [1.0, 0.0, [0.0, 2.0]], "c": {"d": {"e": 0.0}}}
    assert math.isnan(data["a"])  # the input is not modified
    (rec,) = caplog.records  # one warning per call, however many values were replaced
    msg = rec.getMessage()
    assert rec.levelno == logging.WARNING and "unit" in msg
    for path in ("a=nan", "b[1]=inf", "b[2][0]=-inf", "c.d.e=nan"):
        assert path in msg


def test_a_float_subclass_and_a_none_replacement_are_handled(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="brain.utils.finite"):
        assert clamp_nonfinite(_F("nan")) == 0.0
        assert clamp_nonfinite(float("nan"), None) is None
    assert len(caplog.records) == 2 and "<value>" in caplog.records[0].getMessage()


def test_a_long_list_of_replacements_is_summarised_in_one_line(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="brain.utils.finite"):
        clamp_nonfinite([float("nan")] * 25)
    (rec,) = caplog.records
    assert "+15 more" in rec.getMessage()
