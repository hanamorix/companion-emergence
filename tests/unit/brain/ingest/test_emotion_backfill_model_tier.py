"""Tests for #154's model-tier fix to brain.ingest.emotion_backfill — C3/C5/C7.

C3: the usage-log label for emotion_backfill's tagger must derive from the real
    model the constructed provider used, not a disconnected hardcoded literal.
C5: `_make_default_tagger`'s `if provider is None` fallback branch — confirmed
    dead in production (the live caller, supervisor.py, always supplies a
    provider) — must source its model from `model_for_tier(TIER_BACKGROUND_
    CLASSIFIER)`, the SAME single source of truth the live path now uses,
    rather than a separately-hardcoded, potentially-divergent literal.
C7: the oracle must be shown able to fail — this file's
    `test_supervisor_call_site_names_the_classifier_tier` fails against the
    PRE-#154 source (read via `git show`) and passes against the post-#154
    source, proving it is not vacuously true.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from brain.bridge.model_tier import TIER_BACKGROUND_CLASSIFIER, model_for_tier
from brain.bridge.provider import ClaudeCliProvider
from brain.ingest.emotion_backfill import _make_default_tagger

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SUPERVISOR_PATH = _REPO_ROOT / "brain" / "bridge" / "supervisor.py"


class _RecordingProvider:
    """A minimal stand-in exposing `.generate()` and a `._model` attribute, so
    the wiring can be verified without shelling out to a live `claude` CLI."""

    def __init__(self, model: str) -> None:
        self._model = model
        self.generate_calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None, persona_dir=None) -> str:
        self.generate_calls.append((prompt, system))
        return "{}"


def test_tagger_uses_exactly_the_provider_it_was_given(tmp_path):
    """Wiring check: whatever provider `run_emotion_backfill`'s caller supplies
    is the ONE the tagger actually calls .generate() on — no silent fallback to
    a different object. Combined with `build_tier_provider`'s own test (that it
    constructs a real Haiku ClaudeCliProvider), this proves the live path is
    honest end-to-end without needing a real subprocess spawn here."""
    provider = _RecordingProvider(model="haiku")
    tagger = _make_default_tagger(provider)

    class _Mem:
        content = "a memory about the ocean"

    tagger(_Mem())
    assert len(provider.generate_calls) == 1


def test_make_default_tagger_dead_branch_sources_from_model_for_tier(monkeypatch):
    """C5: the `provider is None` fallback branch (confirmed dead in production —
    the live caller always supplies a provider) constructs its ClaudeCliProvider
    with the SAME model `model_for_tier(TIER_BACKGROUND_CLASSIFIER)` returns —
    no separately-hardcoded literal to drift out of sync."""
    captured: dict[str, object] = {}
    real_init = ClaudeCliProvider.__init__

    def _capturing_init(self, model="sonnet", timeout_seconds=60):
        captured["model"] = model
        real_init(self, model=model, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(ClaudeCliProvider, "__init__", _capturing_init)

    _make_default_tagger(None)

    assert captured["model"] == model_for_tier(TIER_BACKGROUND_CLASSIFIER) == "haiku"


def _emotion_backfill_call_site_tier(supervisor_src: str) -> str | None:
    m = re.search(
        r"_emotion_backfill_run\(\s*\n"
        r"\s*persona_dir,\s*provider=build_tier_provider\(persona_dir,\s*([A-Z_]+)\)",
        supervisor_src,
    )
    return m.group(1) if m else None


def test_supervisor_call_site_names_the_classifier_tier():
    """The live (fixed) source names TIER_BACKGROUND_CLASSIFIER."""
    tier = _emotion_backfill_call_site_tier(_SUPERVISOR_PATH.read_text(encoding="utf-8"))
    assert tier == "TIER_BACKGROUND_CLASSIFIER"


def test_oracle_fails_against_the_pre_fix_source(monkeypatch):
    """C7 — fail-capable proof: the SAME check, run against the base commit's
    `supervisor.py` (via `git show`), does NOT find a build_tier_provider(...)
    call at all (the pre-#154 source passes the bare shared `provider` straight
    through) — confirming the oracle can fail, not merely pass vacuously.
    Skips (does not fail the suite) if the recorded base sha is unavailable in
    this checkout, per this project's git-show-only C7 policy (no hand-copied
    fallback fixture — see the guarded-change decisions log)."""
    base_sha = _find_pre_154_base_sha()
    if base_sha is None:
        pytest.skip("base sha for the pre-#154 tree not resolvable in this checkout")

    result = subprocess.run(
        ["git", "show", f"{base_sha}:brain/bridge/supervisor.py"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"git show failed for {base_sha}: {result.stderr.strip()[:200]}")

    pre_fix_src = result.stdout
    assert "_emotion_backfill_run(persona_dir, provider=provider)" in pre_fix_src
    assert _emotion_backfill_call_site_tier(pre_fix_src) is None


def _find_pre_154_base_sha() -> str | None:
    """The commit this branch's #154 work is based on top of — the merge of
    #166 per this project's handoff notes. Resolved dynamically (not
    hardcoded) via the merge-base with origin/main, falling back to None
    (skip) if that ref isn't available in this checkout (e.g. a shallow
    clone or CI worker without the remote configured)."""
    result = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/main"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip()
