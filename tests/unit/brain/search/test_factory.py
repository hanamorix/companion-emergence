"""Tests for brain.search.factory.get_searcher()."""

from __future__ import annotations

import pytest

from brain.search.base import NoopWebSearcher
from brain.search.factory import get_searcher


def test_get_searcher_ddgs():
    s = get_searcher("ddgs")
    assert s.name() == "ddgs"


def test_get_searcher_noop():
    s = get_searcher("noop")
    assert isinstance(s, NoopWebSearcher)


def test_get_searcher_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown searcher"):
        get_searcher("not_a_searcher")


def test_get_searcher_claude_tool_raises_user_friendly_error():
    """claude-tool is a Phase 1 stub — factory should give user a clear
    message instead of returning an instance that crashes on first use."""
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        get_searcher("claude-tool")
