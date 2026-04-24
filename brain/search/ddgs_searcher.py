"""DuckDuckGo web search via the `ddgs` library. No API key, free, zero-cost."""

from __future__ import annotations

import logging

from ddgs import DDGS

from brain.search.base import SearchResult, WebSearcher

logger = logging.getLogger(__name__)


class DdgsWebSearcher(WebSearcher):
    """DuckDuckGo search through the `ddgs` Python library.

    Default searcher for the framework. Works with any LLM backend —
    no dependency on `claude` CLI or any specific provider. Transient
    errors (network, rate-limit, parser failures) return an empty list
    plus a warning log so the research engine can gracefully fall back
    to memory-only synthesis.
    """

    def __init__(self, region: str = "wt-wt", timeout_seconds: int = 15) -> None:
        self._region = region
        self._timeout = timeout_seconds

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        try:
            with DDGS(timeout=self._timeout) as ddgs:
                raw = list(ddgs.text(query, region=self._region, max_results=limit))
        except Exception as exc:
            logger.warning("ddgs search failed for %r: %s", query[:80], exc)
            return []

        return [
            SearchResult(
                title=str(r.get("title", "")),
                url=str(r.get("href") or r.get("url", "")),
                snippet=str(r.get("body", "")),
            )
            for r in raw
        ]

    def name(self) -> str:
        return "ddgs"
