"""Tests for brain.search.ddgs_searcher.DdgsWebSearcher."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from brain.search.base import SearchResult
from brain.search.ddgs_searcher import DdgsWebSearcher


def test_ddgs_happy_path():
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_ctx
    fake_ctx.__exit__.return_value = None
    fake_ctx.text.return_value = [
        {"title": "T1", "href": "https://example.com/1", "body": "s1"},
        {"title": "T2", "href": "https://example.com/2", "body": "s2"},
    ]

    with patch("brain.search.ddgs_searcher.DDGS", return_value=fake_ctx):
        out = DdgsWebSearcher().search("quantum mechanics", limit=2)

    assert len(out) == 2
    assert out[0] == SearchResult(title="T1", url="https://example.com/1", snippet="s1")
    assert out[1].url == "https://example.com/2"


def test_ddgs_uses_href_or_url_field():
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_ctx
    fake_ctx.__exit__.return_value = None
    fake_ctx.text.return_value = [{"title": "T", "url": "https://x.com", "body": "s"}]

    with patch("brain.search.ddgs_searcher.DDGS", return_value=fake_ctx):
        out = DdgsWebSearcher().search("q", limit=1)

    assert out[0].url == "https://x.com"


def test_ddgs_transient_failure_returns_empty(caplog):
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_ctx
    fake_ctx.__exit__.return_value = None
    fake_ctx.text.side_effect = RuntimeError("network down")

    with patch("brain.search.ddgs_searcher.DDGS", return_value=fake_ctx):
        with caplog.at_level(logging.WARNING, logger="brain.search.ddgs_searcher"):
            out = DdgsWebSearcher().search("q")

    assert out == []
    assert any("ddgs search failed" in r.message for r in caplog.records)


def test_ddgs_name():
    assert DdgsWebSearcher().name() == "ddgs"
