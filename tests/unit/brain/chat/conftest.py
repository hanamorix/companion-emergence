"""Shared fixtures for tests/unit/brain/chat/.

#154: pass-2 monologue extraction (`brain.chat.tool_loop._spawn_pass2`) now
builds its own `TIER_BACKGROUND_CLASSIFIER` provider internally, via
`build_tier_provider(persona_dir, ...)`, instead of using the `provider`
argument callers pass to `run_tool_loop`/`_spawn_pass2` (see that module's
docstring — kept as a real, backward-compatible parameter specifically so
existing callers/tests didn't need updating, but no longer read for this
call). Several tests in this directory drive `run_tool_loop` end-to-end and
then synchronously drain `pass2_queue` (so the extraction call actually
executes inside the test), using a `tmp_path`-based `persona_dir` with no
`persona_config.json` — `build_tier_provider` would default to a REAL
`ClaudeCliProvider` there and attempt a genuine `claude` CLI subprocess call,
hanging/timing out in CI.

This autouse fixture patches `build_tier_provider` at its source module for
every test in this directory, returning a cheap, deterministic stand-in
instead. Tests that specifically assert on the classifier-tier provider's
identity (e.g. test_tool_loop_pass2_spawn.py's
`test_pass2_fires_after_record_monologue`) re-patch it themselves within the
test body — that per-test patch simply takes over from here, since both use
the same `monkeypatch` fixture instance.
"""
from __future__ import annotations

import json

import pytest


class _DefaultClassifierTierProvider:
    """A safe, generic stand-in — never shells out to a real subprocess.

    Returns a syntactically-valid empty extraction so pass-2's parse step
    doesn't itself raise; tests that care about the extraction's CONTENT
    override this fixture's patch with their own.
    """

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return json.dumps(
            {
                "memory_writes": [],
                "emotion_delta": {},
                "crystallisation": [],
                "reflex_audit": [],
            }
        )


@pytest.fixture(autouse=True)
def _safe_classifier_tier_provider(monkeypatch: pytest.MonkeyPatch):
    """Prevent pass-2 extraction from constructing a real ClaudeCliProvider."""
    monkeypatch.setattr(
        "brain.bridge.model_tier.build_tier_provider",
        lambda _persona_dir, _tier: _DefaultClassifierTierProvider(),
    )
