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


def test_derived_is_recent_window_peak_excluding_old_peaks():
    """Windowed peak: a high-intensity OLD memory beyond the recent window does
    NOT contribute; only the recent window's peaks do.

    (Replaces the old recency-MEAN contract — the derived read is now a peak
    over the most-recent _RECENT_WINDOW_COUNT memories, commensurable with the
    declared lifetime peak. The two diverge only when a channel's peak is older
    than the window — an honest "I claim this but haven't felt it lately" gap.)
    """
    # 31 recent grief memories + 1 old joy peak. With a 30-memory window the
    # oldest (joy) falls outside the window → derived sees grief, not joy.
    mems = [_mem({"grief": 3.0}, age_days=0) for _ in range(31)]
    mems.append(_mem({"joy": 9.0}, age_days=60))
    out = compute_derived(mems, body_energy=5, body_exhaustion=2)
    assert isinstance(out, DerivedRead)
    assert out.channels.get("grief", 0) == 3.0
    assert out.channels.get("joy", 0) == 0.0  # old peak excluded by the recent window


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


# ─── Windowed-peak regression (self_model_state.json magnitude-354 artifact) ──


def test_recent_diverse_channels_no_dilution_artifact():
    """Regression for the live magnitude-354 bug.

    With many memories EACH carrying a different channel at peak (the normal
    shape of a real persona — any one channel appears in only a fraction of
    memories), the old total-mass-normalised derived diluted every channel to
    a small fraction of its peak, so the gap vs declared (max-pool peak) was a
    large uniform-negative offset (~N_channels × peak). The windowed peak reads
    each recently-felt channel at its actual peak → derived ≈ declared → the
    gap is small.
    """
    from brain.emotion.aggregate import aggregate_state
    from brain.self_model.gap import compute_gap

    chans = ["joy", "grief", "curiosity", "love"]
    mems = [_mem({chans[i % len(chans)]: 7.0}, age_days=i % 3) for i in range(40)]
    declared = aggregate_state(mems)
    derived = compute_derived(mems, body_energy=5, body_exhaustion=2)
    gap = compute_gap(declared, derived)
    assert gap.magnitude < 3.0, f"dilution artifact present: magnitude={gap.magnitude:.1f}"
