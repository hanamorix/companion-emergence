"""Searcher factory — resolve a name to an instance."""

from __future__ import annotations

from brain.search.base import NoopWebSearcher, WebSearcher
from brain.search.ddgs_searcher import DdgsWebSearcher


def get_searcher(name: str) -> WebSearcher:
    """Resolve a searcher identifier to an instance.

    Raises ValueError on unknown name. Raises NotImplementedError for
    Phase 1 stubs (claude-tool) with a user-friendly message pointing
    to the working alternatives.
    """
    if name == "ddgs":
        return DdgsWebSearcher()
    if name == "noop":
        return NoopWebSearcher()
    if name == "claude-tool":
        raise NotImplementedError(
            "The 'claude-tool' searcher is not yet implemented (Phase 1 stub). "
            "Use 'ddgs' (default, free, no API key) or 'noop' (for tests)."
        )
    raise ValueError(f"Unknown searcher: {name!r}")
