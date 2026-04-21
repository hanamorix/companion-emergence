"""Shared pytest fixtures and configuration for companion-emergence tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove companion-emergence-relevant env vars for isolation."""
    for key in [
        "NELLBRAIN_HOME",
        "NELL_IPC_JID",
        "BRIDGE_BIND",
        "PROVIDER",
        "MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    yield
