"""Regression test: extended-thinking dead-code artefacts must stay removed.

Per spec docs/superpowers/specs/2026-05-30-inner-monologue-tool-call-design.md §6,
the v0.0.25 extended-thinking plumbing was removed because the underlying CLI
capability doesn't exist (Claude Code 2.1.157 does not surface thinking blocks
in any output format). This test grep-walks the source tree and fails if any
of the dead tokens reappear in production code.
"""
from __future__ import annotations

from tests.source_tree import tree_sources

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

INCLUDE_DIRS = ("brain", "app/src")

# Matched against each file's path relative to the repository.
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


def test_no_forbidden_token_in_source():
    """No forbidden token appears in the source files under brain/ or app/src/ (one pass over the tree for all tokens)."""
    hits: dict[str, list[str]] = {}
    for root_name in INCLUDE_DIRS:
        for rel, text in tree_sources(root_name, SOURCE_EXTENSIONS, EXCLUDE_SUBSTRINGS).items():
            for token in FORBIDDEN_TOKENS:
                if token in text:
                    hits.setdefault(token, []).append(rel)
    assert not hits, "Forbidden v0.0.25-era tokens found (these paths must not reintroduce extended-thinking plumbing):\n" + "\n".join(
        f"  {token!r} in {', '.join(paths)}" for token, paths in hits.items()
    )
