"""Searcher factory — resolve a name to an instance."""

from __future__ import annotations

from brain.search.base import NoopWebSearcher, WebSearcher
from brain.search.claude_tool_searcher import ClaudeToolWebSearcher
from brain.search.ddgs_searcher import DdgsWebSearcher


def get_searcher(name: str) -> WebSearcher:
    """Resolve a searcher identifier to an instance. Raises ValueError on unknown."""
    if name == "ddgs":
        return DdgsWebSearcher()
    if name == "noop":
        return NoopWebSearcher()
    if name == "claude-tool":
        return ClaudeToolWebSearcher()
    raise ValueError(f"Unknown searcher: {name!r}")
