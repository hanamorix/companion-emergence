"""Regression test: extended-thinking dead-code artefacts must stay removed.

Per spec docs/superpowers/specs/2026-05-30-inner-monologue-tool-call-design.md §6,
the v0.0.25 extended-thinking plumbing was removed because the underlying CLI
capability doesn't exist (Claude Code 2.1.157 does not surface thinking blocks
in any output format). This test grep-walks the source tree and fails if any
of the dead tokens reappear in production code.
"""
from __future__ import annotations

from pathlib import Path

import pytest

FORBIDDEN_TOKENS = (
    "thinking_budget_tokens",
    "--thinking",
    "--budget-tokens",
    "_write_thinking_log",
    "thinking_log.jsonl",
    "thinking_blocks",
    "setPersonaThinking",
    "/persona/config/thinking",
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

INCLUDE_DIRS = ("brain", "app/src")

EXCLUDE_SUBSTRINGS = (
    "node_modules",
    ".pyc",
    "__pycache__",
    "app/src-tauri/target",
    "app/src-tauri/python-runtime",
    "docs/",
    "CHANGELOG",
    ".public-sync",
    "tests/unit/brain/cleanup",
)

SOURCE_EXTENSIONS = (".py", ".ts", ".tsx", ".rs", ".json")


def _walk_source_files():
    for root_name in INCLUDE_DIRS:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            path_str = str(path)
            if any(s in path_str for s in EXCLUDE_SUBSTRINGS):
                continue
            if path.suffix not in SOURCE_EXTENSIONS:
                continue
            yield path


@pytest.mark.parametrize("token", FORBIDDEN_TOKENS)
def test_no_forbidden_token_in_source(token: str):
    """Each forbidden token must not appear in tracked source files under brain/ or app/src/."""
    hits: list[str] = []
    for path in _walk_source_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if token in text:
            hits.append(str(path.relative_to(PROJECT_ROOT)))
    assert not hits, (
        f"Forbidden v0.0.25-era token {token!r} found in:\n  "
        + "\n  ".join(hits)
        + "\nThese paths must not reintroduce extended-thinking plumbing."
    )
