"""Searcher factory — resolve a name to an instance."""

from __future__ import annotations

from brain.search.base import NoopWebSearcher, WebSearcher
from brain.search.ddgs_searcher import DdgsWebSearcher


def get_searcher(name: str) -> WebSearcher:
    """Resolve a searcher identifier to an instance.

    Raises ValueError on unknown name. A 'claude-tool' Phase-1 stub
    once lived here; it was removed from the public surface in the
    2026-05-07 audit P2 round (F-004). PersonaConfig allowlists heal
    legacy values in hand-edited or migrated config files to the
    default before they ever reach this factory.
    """
    if name == "ddgs":
        return DdgsWebSearcher()
    if name == "noop":
        return NoopWebSearcher()
    raise ValueError(f"Unknown searcher: {name!r}")
