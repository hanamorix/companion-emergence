"""Shared pytest fixtures and configuration for companion-emergence tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Walk upward from this file to find the repo root (pyproject.toml).

    Replaces brittle `Path(__file__).parents[N]` patterns in tests that
    need absolute paths to checked-in resources.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not find pyproject.toml above {here}")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove companion-emergence-relevant env vars for isolation.

    Used by tests for brain.paths and brain.config (Tasks 2+).
    Each key here corresponds to an env var the framework reads at runtime.
    """
    for key in [
        "NELLBRAIN_HOME",
        "NELL_IPC_JID",
        "BRIDGE_BIND",
        "PROVIDER",
        "MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    yield
