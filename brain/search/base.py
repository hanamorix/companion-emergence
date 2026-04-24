"""WebSearcher ABC + shared types + NoopWebSearcher."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    """One web search hit."""

    title: str
    url: str
    snippet: str


class WebSearcher(ABC):
    """Abstract searcher. Subclasses implement `search` and `name`."""

    @abstractmethod
    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        """Return up to `limit` results for `query`. Empty list on any
        transient failure — research engine falls back to memory-only
        synthesis rather than crashing.
        """

    @abstractmethod
    def name(self) -> str:
        """Short identifier: 'ddgs', 'noop', 'claude-tool'."""


class NoopWebSearcher(WebSearcher):
    """Returns no results. Used in tests and CI to keep them zero-network."""

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        return []

    def name(self) -> str:
        return "noop"
