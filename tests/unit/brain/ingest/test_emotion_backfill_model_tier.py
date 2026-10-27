"""Tests for #154's model-tier routing of brain.ingest.emotion_backfill.

The live call site (`brain/bridge/supervisor.py`) must build the backfill's provider for the
background-classifier tier. The behaviour check is
`tests/unit/brain/bridge/test_central_cadence.py::test_the_emotion_backfill_job_builds_its_provider_for_the_classifier_tier`;
this file keeps the source-text check of the same call site and its oracle:
`test_supervisor_call_site_names_the_classifier_tier` fails against the PRE-#154 source (read via `git show`) and
passes against the post-#154 source, proving it is not vacuously true.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SUPERVISOR_PATH = _REPO_ROOT / "brain" / "bridge" / "supervisor.py"


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


# The last main commit BEFORE #194 completed the #154 tier routing (first
# parent of its merge commit 5589f096). Pinned, not computed: the previous
# ``git merge-base HEAD origin/main`` was only "pre-fix" while #154 lived on a
# feature branch — once merged, every main checkout's merge-base was post-fix
# and the oracle failed on main (#205).
_PRE_154_BASE_SHA = "44b4d5227b791ccf0756294f08f82223e726a979"


def _find_pre_154_base_sha() -> str | None:
    """Return the pinned pre-#154 sha if this checkout has it, else None (skip)
    — e.g. a shallow clone on a CI worker."""
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{_PRE_154_BASE_SHA}^{{commit}}"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return _PRE_154_BASE_SHA if result.returncode == 0 else None
