"""Searcher factory — resolve a name to an instance."""

from __future__ import annotations

from brain.search.base import NoopWebSearcher, WebSearcher
from brain.search.ddgs_searcher import DdgsWebSearcher


def get_searcher(name: str) -> WebSearcher:
    """Resolve a searcher identifier to an instance.

    Raises ValueError on unknown name. The 'claude-tool' Phase-1 stub
    has been removed from the public surface (audit 2026-05-07 P2);
    PersonaConfig allowlists guard against legacy values surviving in
    hand-edited or migrated config files.
    """
    if name == "ddgs":
        return DdgsWebSearcher()
    if name == "noop":
        return NoopWebSearcher()
    raise ValueError(f"Unknown searcher: {name!r}")
