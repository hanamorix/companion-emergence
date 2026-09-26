"""brain.state_compat.from_known_fields — tolerant dataclass construction (#286 §7)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from brain.state_compat import from_known_fields


@dataclass(frozen=True)
class _Rec:
    a: int
    b: str = "x"


def test_ignores_fields_the_class_does_not_declare():
    assert from_known_fields(_Rec, {"a": 1, "b": "y", "_future_field": 9}) == _Rec(a=1, b="y")


def test_missing_required_field_still_raises():
    with pytest.raises(TypeError):
        from_known_fields(_Rec, {"b": "y"})


def test_rejects_a_non_dataclass():
    with pytest.raises(TypeError):
        from_known_fields(dict, {"a": 1})
