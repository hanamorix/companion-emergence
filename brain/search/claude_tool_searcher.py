"""Stub for `claude -p --allowed-tools WebSearch` based searcher.

Phase 1 ships this as NotImplementedError. If a Claude-CLI user wants
to route web search through Claude's tool loop instead of DDG, this is
where that lives. Mirrors how OllamaProvider ships as a stub.
"""

from __future__ import annotations

from brain.search.base import SearchResult, WebSearcher


class ClaudeToolWebSearcher(WebSearcher):
    """Not implemented in Phase 1 — see docstring."""

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        raise NotImplementedError(
            "ClaudeToolWebSearcher is a Phase 1 stub. Use DdgsWebSearcher "
            "(default) or NoopWebSearcher instead, or implement this if you "
            "want to route search through `claude -p --allowed-tools WebSearch`."
        )

    def name(self) -> str:
        return "claude-tool"
