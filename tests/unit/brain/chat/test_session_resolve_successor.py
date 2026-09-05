"""resolve_successor never raises on a corrupt pointer (C3b, walk-level)."""

from __future__ import annotations

import logging
from pathlib import Path

from brain.chat.session import _resolve_successor, resolve_successor


def test_invalid_successor_pointer_returns_none_and_warns(tmp_path: Path, caplog) -> None:
    d = tmp_path / "active_conversations"
    d.mkdir()
    (d / "s_a.rolled_to").write_text('{"successor": "../x"}', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="brain.chat.session"):
        assert resolve_successor(tmp_path, "s_a") is None
    assert any(rec.name == "brain.chat.session" for rec in caplog.records)


def test_underscore_alias_kept() -> None:
    assert _resolve_successor is resolve_successor
