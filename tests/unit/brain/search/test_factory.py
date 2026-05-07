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


def test_get_searcher_claude_tool_now_unknown():
    """claude-tool was a Phase 1 stub; removed from the public surface in
    the 2026-05-07 audit P2 round. It now reads as any other unknown name
    so legacy hand-edited config doesn't surface NotImplementedError to
    users — PersonaConfig allowlists heal it to the default first."""
    with pytest.raises(ValueError, match="Unknown searcher"):
        get_searcher("claude-tool")
