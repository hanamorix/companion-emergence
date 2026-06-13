"""Tests for brain/self_model/articulate.py — budgeted+throttled Haiku gap articulation.

R-D1: daily budget exhausted → None, no provider call.
Fail-soft: provider raises → None, no exception propagated.
Threshold gate: gap below _GAP_THRESHOLD → None, no provider call.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from brain.self_model.articulate import _DAILY_ARTICULATE_BUDGET, _GAP_THRESHOLD, articulate
from brain.self_model.gap import Gap


def _gap(magnitude: float) -> Gap:
    return Gap(
        per_channel={"grief": magnitude},
        magnitude=magnitude,
        unnamed_pressure=0.0,
    )


class _CountingProvider:
    """Minimal provider stub that counts generate() calls."""

    def __init__(self, *, raises: bool = False, response: str = "I notice a weight I haven't named.") -> None:
        self.calls: list[tuple[str, str | None]] = []
        self._raises = raises
        self._response = response

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        if self._raises:
            raise RuntimeError("provider failure")
        return self._response

    def name(self) -> str:
        return "counting-fake"


def test_below_threshold_returns_none_no_provider_call(tmp_path):
    """Gap below _GAP_THRESHOLD → None, provider is NOT called."""
    provider = _CountingProvider()
    gap = _gap(magnitude=_GAP_THRESHOLD - 0.01)
    result = articulate(gap, provider=provider, persona_dir=tmp_path)
    assert result is None
    assert len(provider.calls) == 0


def test_above_threshold_calls_provider_returns_note(tmp_path):
    """Gap above _GAP_THRESHOLD → exactly one provider call, returns the note string."""
    provider = _CountingProvider(response="I notice a weight I haven't named.")
    gap = _gap(magnitude=_GAP_THRESHOLD + 0.1)
    result = articulate(gap, provider=provider, persona_dir=tmp_path)
    assert result == "I notice a weight I haven't named."
    assert len(provider.calls) == 1


def test_provider_raises_returns_none_fail_soft(tmp_path):
    """Provider failure → None returned, no exception propagated (fail-soft)."""
    provider = _CountingProvider(raises=True)
    gap = _gap(magnitude=_GAP_THRESHOLD + 0.5)
    result = articulate(gap, provider=provider, persona_dir=tmp_path)
    assert result is None
    # The provider was reached (one call attempted before it raised)
    assert len(provider.calls) == 1


def _exhaust_budget(persona_dir: Path, *, cap: int) -> None:
    """Pre-fill the budget file so zero calls remain today."""
    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    path = persona_dir / "self_model" / "daily_articulate_budget.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": today, "count": cap}))


def test_daily_budget_exhausted_returns_none_no_call(tmp_path):
    """R-D1: budget at cap → None, provider NOT called."""
    _exhaust_budget(tmp_path, cap=_DAILY_ARTICULATE_BUDGET)
    provider = _CountingProvider()
    gap = _gap(magnitude=_GAP_THRESHOLD + 0.5)
    result = articulate(gap, provider=provider, persona_dir=tmp_path)
    assert result is None
    assert len(provider.calls) == 0


def test_corrupt_budget_file_still_allows_call(tmp_path):
    """Corrupt budget file → fail-safe-permissive: call is allowed (infra error ≠ deny)."""
    budget_path = tmp_path / "self_model" / "daily_articulate_budget.json"
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    budget_path.write_text("NOT VALID JSON {{{{")
    provider = _CountingProvider(response="Something shifts.")
    gap = _gap(magnitude=_GAP_THRESHOLD + 0.5)
    result = articulate(gap, provider=provider, persona_dir=tmp_path)
    assert result == "Something shifts."
    assert len(provider.calls) == 1
