"""Tests for brain/self_model/articulate.py — budgeted+throttled Haiku gap articulation.

R-D1: daily budget exhausted → None, no provider call.
Fail-soft: provider raises → None, no exception propagated.
Threshold gate: gap below _GAP_THRESHOLD → None, no provider call.
Routing: build_self_model_provider forces SELF_MODEL_MODEL (haiku) regardless of
  the persona's configured chat model, mirroring brain/chat/compaction.py's
  build_compaction_provider. The usage log's model= must reflect the model the
  provider actually used, not a disconnected hardcoded label (was: model="haiku"
  hardcoded at articulate.py:162 even when a mis-routed provider ran something
  else entirely).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from brain.bridge.provider import ClaudeCliProvider
from brain.self_model.articulate import (
    _DAILY_ARTICULATE_BUDGET,
    _GAP_THRESHOLD,
    SELF_MODEL_MODEL,
    articulate,
    build_self_model_provider,
)
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


# ---------------------------------------------------------------------------
# Routing regression: the articulate note must run on SELF_MODEL_MODEL
# (haiku), not whatever provider/model the caller happens to be threading
# through (the persona chat provider) — and the usage log must record the
# model that actually ran.
# ---------------------------------------------------------------------------


def test_build_self_model_provider_forces_haiku_regardless_of_chat_model(tmp_path):
    """build_self_model_provider must carry model_override=SELF_MODEL_MODEL
    ("haiku") even when the persona's configured chat model is something
    else (e.g. sonnet) — this is a cheap housekeeping call, not a chat reply,
    and must not inherit the persona chat model. Mirrors
    test_provider_factory_model.test_factory_override_beats_config."""
    (tmp_path / "persona_config.json").write_text(
        json.dumps({"provider": "claude-cli", "searcher": "noop", "model": "sonnet"})
    )
    provider = build_self_model_provider(tmp_path)
    assert isinstance(provider, ClaudeCliProvider)
    assert SELF_MODEL_MODEL == "haiku"
    assert provider._model == "haiku"  # noqa: SLF001 — NOT "sonnet", the chat model


class _ModelledProvider:
    """Provider stub that, unlike _CountingProvider, exposes ._model — the
    attribute real providers (ClaudeCliProvider/OllamaProvider) set from
    model_override, so the usage-log label can be derived from it."""

    def __init__(self, model: str, *, response: str = "Something shifts.") -> None:
        self._model = model
        self._response = response
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return self._response

    def name(self) -> str:
        return f"claude-cli:{self._model}"


def _read_usage_rows(persona_dir: Path) -> list[dict]:
    path = persona_dir / "chat_usage.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_usage_log_derives_model_from_provider_not_hardcoded(tmp_path):
    """The usage log's model= must reflect the model the provider actually
    used. Proven by passing a provider whose ._model is deliberately NOT
    "haiku" and confirming the log records THAT value — a hardcoded
    model="haiku" label (the pre-fix bug) would fail this."""
    provider = _ModelledProvider("sonnet")  # deliberately not haiku
    gap = _gap(magnitude=_GAP_THRESHOLD + 0.5)
    result = articulate(gap, provider=provider, persona_dir=tmp_path)
    assert result == "Something shifts."

    rows = _read_usage_rows(tmp_path)
    assert rows, "expected a usage row to be logged"
    assert rows[-1]["call_type"] == "self_model_articulate"
    assert rows[-1]["model"] == "sonnet"


def test_usage_log_records_haiku_when_correctly_routed(tmp_path):
    """End-to-end shape of the fix: a provider built via
    build_self_model_provider (._model == "haiku") logs "haiku"."""
    provider = _ModelledProvider(SELF_MODEL_MODEL)
    gap = _gap(magnitude=_GAP_THRESHOLD + 0.5)
    articulate(gap, provider=provider, persona_dir=tmp_path)

    rows = _read_usage_rows(tmp_path)
    assert rows[-1]["model"] == "haiku"


def test_usage_log_falls_back_to_self_model_constant_without_model_attr(tmp_path):
    """A provider with no ._model (e.g. FakeProvider in tests) falls back to
    SELF_MODEL_MODEL rather than an unrelated hardcoded string."""
    provider = _CountingProvider(response="Something shifts.")
    gap = _gap(magnitude=_GAP_THRESHOLD + 0.5)
    articulate(gap, provider=provider, persona_dir=tmp_path)

    rows = _read_usage_rows(tmp_path)
    assert rows[-1]["model"] == SELF_MODEL_MODEL
