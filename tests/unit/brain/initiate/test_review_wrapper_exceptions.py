"""Tests for #452: the D-reflection LLM-call wrapper maps provider failures to
the right D* exception, and the tick reacts to each in the intended way.

Before the fix, `_make_haiku_call` caught `(TimeoutError, TimeoutError.__class__)`;
`TimeoutError.__class__` is `type`, so any non-TimeoutError failure raised
`TypeError: catching classes that do not inherit from BaseException`. The same
change makes the rate-limit text test and the auth-deferral path live, so those
are pinned here too. Synthetic personas only; no real CLI.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.bridge.provider import ProviderError
from brain.bridge.provider_auth import ProviderAuthDeferred
from brain.initiate.audit import read_recent_d_calls
from brain.initiate.emit import emit_initiate_candidate, read_candidates
from brain.initiate.reflection import DProviderError, DRateLimitError, DTimeoutError
from brain.initiate.review import (
    _make_haiku_call,
    _make_sonnet_call,
    run_initiate_review_tick,
)
from brain.initiate.schemas import SemanticContext


class _FakeProvider:
    """Stands in for the gate provider: generate() raises `exc` or returns `text`."""

    def __init__(self, exc: BaseException | None = None, text: str = "{}") -> None:
        self._exc = exc
        self._text = text
        self.calls = 0

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._text


def _call(provider: _FakeProvider):
    return _make_haiku_call(provider)(system="s", user="u")


_LIMIT_FRAME = (
    "ClaudeCliProvider failed (exit 1): api_error_status=429; is_error=True; "
    "You've hit your session limit"
)

# ---------------------------------------------------------------------------
# Unit: the wrapper's exception mapping
# ---------------------------------------------------------------------------


def test_subprocess_timeout_becomes_dtimeout() -> None:
    original = subprocess.TimeoutExpired("claude", 5)
    with pytest.raises(DTimeoutError) as info:
        _call(_FakeProvider(original))
    assert str(info.value) == str(original)
    assert info.value.__cause__ is original


def test_plain_timeout_still_becomes_dtimeout() -> None:
    original = TimeoutError("ClaudeCliProvider: subprocess timed out after 30s")
    with pytest.raises(DTimeoutError) as info:
        _call(_FakeProvider(original))
    assert str(info.value) == str(original)
    assert info.value.__cause__ is original


def test_rate_limit_frame_becomes_dratelimit() -> None:
    original = RuntimeError(_LIMIT_FRAME)
    with pytest.raises(DRateLimitError) as info:
        _call(_FakeProvider(original))
    assert str(info.value) == str(original)
    assert info.value.__cause__ is original


@pytest.mark.parametrize(
    "original",
    [
        RuntimeError("ClaudeCliProvider failed (exit 1): boom"),
        ProviderError("claude_cli_error", "boom"),
    ],
)
def test_other_failures_become_dprovider(original: Exception) -> None:
    with pytest.raises(DProviderError) as info:
        _call(_FakeProvider(original))
    assert str(info.value) == str(original)
    assert info.value.__cause__ is original


@pytest.mark.parametrize(
    "text",
    [
        "failed to generate a response",
        "could not separate the frames",
        "ClaudeCliProvider: unexpected output format: 'accurate and moderate'",
        "ClaudeCliProvider failed (exit 1): wrote 429 bytes, pid 14290",
        "unable to operate; will iterate and migrate",
    ],
)
def test_ordinary_words_and_numbers_are_not_rate_limits(text: str) -> None:
    with pytest.raises(DProviderError):
        _call(_FakeProvider(RuntimeError(text)))


@pytest.mark.parametrize(
    "text",
    [
        _LIMIT_FRAME,
        "HTTP 429 Too Many Requests",
        "429 Too Many Requests",
        "Rate limit exceeded",
        "rate_limit_error",
        "You've hit your usage limit",
    ],
)
def test_real_rate_limits_are_classified(text: str) -> None:
    with pytest.raises(DRateLimitError):
        _call(_FakeProvider(RuntimeError(text)))


def test_auth_deferral_propagates_unchanged() -> None:
    original = ProviderAuthDeferred("provider auth expired — deferred")
    with pytest.raises(ProviderAuthDeferred) as info:
        _call(_FakeProvider(original))
    assert info.value is original


def test_success_returns_text_and_latency() -> None:
    text, latency_ms, tin, tout = _call(_FakeProvider(text="hello"))
    assert (text, tin, tout) == ("hello", 0, 0)
    assert latency_ms >= 0


def test_sonnet_wrapper_has_the_same_mapping() -> None:
    with pytest.raises(DRateLimitError):
        _make_sonnet_call(_FakeProvider(RuntimeError(_LIMIT_FRAME)))(system="s", user="u")
    with pytest.raises(DProviderError):
        _make_sonnet_call(_FakeProvider(RuntimeError("boom")))(system="s", user="u")


# ---------------------------------------------------------------------------
# Tick level: what happens to the queued candidates in each case
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)


def _tick_setup(tmp_path: Path, monkeypatch, exc: BaseException):
    """Synthetic persona with one queued candidate; the gate provider always raises `exc`.
    Composition is stubbed to record which candidates would be composed."""
    persona = tmp_path / "persona"
    emit_initiate_candidate(
        persona,
        kind="message",
        source="dream",
        source_id="d_wrapper",
        semantic_context=SemanticContext(),
        now=_BASE,
    )
    gate = _FakeProvider(exc)
    monkeypatch.setattr("brain.bridge.model_tier.build_tier_provider", lambda *_a, **_k: gate)
    composed: list[str] = []

    def _record(persona_dir, candidate, **_kw):
        composed.append(candidate.candidate_id)

    monkeypatch.setattr("brain.initiate.review._process_one_candidate", _record)
    return persona, composed


def _tick(persona: Path, minutes: int) -> None:
    run_initiate_review_tick(
        persona,
        provider=MagicMock(),
        voice_template="x",
        cap_per_tick=3,
        now=_BASE + timedelta(minutes=minutes),
    )


def _failure_types(persona: Path) -> list[str | None]:
    rows = read_recent_d_calls(persona, window_hours=24, now=_BASE + timedelta(hours=1))
    return [r.failure_type for r in rows]


def test_tick_rate_limit_demotes_candidate_to_draft_space(tmp_path: Path, monkeypatch) -> None:
    persona, composed = _tick_setup(tmp_path, monkeypatch, RuntimeError(_LIMIT_FRAME))
    _tick(persona, 1)
    assert read_candidates(persona) == []
    assert (persona / "draft_space.md").exists()
    assert composed == []
    assert _failure_types(persona) == ["rate_limit"]


def test_tick_generate_wording_is_a_provider_error_not_a_rate_limit(
    tmp_path: Path, monkeypatch
) -> None:
    persona, composed = _tick_setup(
        tmp_path, monkeypatch, RuntimeError("ClaudeCliProvider failed (exit 1): failed to generate")
    )
    _tick(persona, 1)
    assert len(read_candidates(persona)) == 1
    assert not (persona / "draft_space.md").exists()
    assert composed == []
    assert _failure_types(persona) == ["provider_error"]
    # Three consecutive provider errors: the stranded-candidate escape hatch promotes all.
    _tick(persona, 2)
    _tick(persona, 3)
    assert len(composed) == 1


def test_tick_subprocess_timeout_keeps_candidate_queued(tmp_path: Path, monkeypatch) -> None:
    persona, composed = _tick_setup(tmp_path, monkeypatch, subprocess.TimeoutExpired("claude", 5))
    _tick(persona, 1)
    assert len(read_candidates(persona)) == 1
    assert composed == []
    assert _failure_types(persona) == ["timeout"]


def test_tick_auth_deferral_is_a_quiet_no_op_never_a_failure(tmp_path: Path, monkeypatch) -> None:
    persona, composed = _tick_setup(
        tmp_path, monkeypatch, ProviderAuthDeferred("provider auth expired — deferred")
    )
    for minute in (1, 2, 3):
        _tick(persona, minute)  # must not raise
    assert len(read_candidates(persona)) == 1
    assert not (persona / "draft_space.md").exists()
    assert composed == []  # three deferrals never reach the 3-failure promote-all
    assert _failure_types(persona) == []  # no dcall row at all
