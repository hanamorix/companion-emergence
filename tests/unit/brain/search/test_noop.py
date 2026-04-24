"""Tests for brain.search.base.NoopWebSearcher."""

from __future__ import annotations

from brain.search.base import NoopWebSearcher


def test_noop_returns_empty_list():
    s = NoopWebSearcher()
    assert s.search("any query") == []


def test_noop_returns_empty_with_limit_arg():
    s = NoopWebSearcher()
    assert s.search("any query", limit=10) == []


def test_noop_name():
    assert NoopWebSearcher().name() == "noop"
