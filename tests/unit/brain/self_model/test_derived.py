"""Tests for brain/self_model/derived.py — Task 1 + Task 1b.

TDD order:
  Task 1 — recency-weighted mean + body adjustment, fail-open (4 tests)
  Task 1b — conservative unnamed_pressure residual (2 tests)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from brain.memory.store import Memory
from brain.self_model.derived import DerivedRead, compute_derived


def _mem(emotions: dict, age_days: float) -> Memory:
    m = Memory.create_new(
        content="x",
        memory_type="episodic",
        domain="chat",
        emotions=emotions,
    )
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=age_days))
    return m


def test_derived_returns_recency_weighted_not_maxpool():
    """OLD high-intensity joy + RECENT low-intensity grief.

    max-pool (declared) would surface joy; recency-mean (derived) leans
    grief because the recent memory carries far more weight.
    """
    mems = [_mem({"joy": 9.0}, age_days=20), _mem({"grief": 3.0}, age_days=0)]
    out = compute_derived(mems, body_energy=5, body_exhaustion=2)
    assert isinstance(out, DerivedRead)
    # derived grief should be >= joy (recency wins), unlike max-pool which would say joy=9
    assert out.channels.get("grief", 0) >= out.channels.get("joy", 0)


def test_identical_recent_and_peak_signals_no_divergence():
    """One memory: recency-mean == max-pool == same vector → derived ~ declared."""
    mems = [_mem({"loneliness": 4.0}, age_days=1)]
    out = compute_derived(mems, body_energy=5, body_exhaustion=2)
    assert abs(out.channels.get("loneliness", 0) - 4.0) < 1.5  # close to the single value


def test_empty_memories_fails_open_to_no_gap():
    out = compute_derived([], body_energy=5, body_exhaustion=2)
    assert out.channels == {} or all(v == 0 for v in out.channels.values())
    assert out.unnamed_pressure == 0.0


def test_compute_derived_never_raises_on_bad_input():
    """A memory with None emotions must not crash (fail-open)."""
    bad = _mem({}, age_days=1)
    object.__setattr__(bad, "emotions", None)
    out = compute_derived([bad], body_energy=5, body_exhaustion=2)
    assert isinstance(out, DerivedRead)


# ─── Task 1b ───────────────────────────────────────────────────────────────


def test_ordinary_state_zero_unnamed_pressure():
    """R-E5: ordinary body state (energy=5, exhaustion=2) yields exactly 0.0."""
    mems = [_mem({"joy": 3.0}, age_days=1), _mem({"loneliness": 2.0}, age_days=2)]
    out = compute_derived(mems, body_energy=5, body_exhaustion=2)
    assert out.unnamed_pressure == 0.0


def test_strong_bodily_signal_with_no_channel_home_flags_pressure():
    """High exhaustion + depleted energy but NO low-arousal emotion memories.

    exhaustion=9, energy=1 → extreme low-arousal body state.
    Only memory is faint, stale joy — not a low-arousal channel.
    Residual body signal that cannot be absorbed → unnamed_pressure > 0.
    """
    mems = [_mem({"joy": 1.0}, age_days=30)]  # faint, stale, joy only
    out = compute_derived(mems, body_energy=1, body_exhaustion=9)
    assert out.unnamed_pressure > 0.0
