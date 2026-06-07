"""Tests for brain.chat.reflection_gate — debounce for background pass-2 spawns."""
from brain.chat.reflection_gate import _MIN_TURNS_BETWEEN, should_reflect
from brain.chat.salience import SalienceSignal, assess_salience


def test_threshold_is_lowered_to_030():
    from brain.chat.reflection_gate import _SALIENCE_THRESHOLD
    assert _SALIENCE_THRESHOLD == 0.30


def test_trivial_turn_does_not_reflect(tmp_path):
    assert should_reflect(assess_salience("ok"), tmp_path, kind="attunement", turn_index=10) is False


def test_significant_turn_reflects_when_window_elapsed(tmp_path):
    sig = SalienceSignal.maximal()
    assert should_reflect(sig, tmp_path, kind="attunement", turn_index=10) is True
    # immediately after, the cursor blocks a second fire within the window
    assert should_reflect(sig, tmp_path, kind="attunement", turn_index=10 + _MIN_TURNS_BETWEEN - 1) is False
    # far enough later, fires again
    assert should_reflect(sig, tmp_path, kind="attunement", turn_index=10 + _MIN_TURNS_BETWEEN) is True


def test_corrupt_state_fails_open(tmp_path):
    (tmp_path / "reflection_state.json").write_text("{not json")
    assert should_reflect(SalienceSignal.maximal(), tmp_path, kind="attunement", turn_index=1) is True


def test_quiet_emotional_turn_reflects(tmp_path):
    from brain.chat.salience import assess_salience
    s = assess_salience("Hey love. I'm back — long day of editing, my eyes are sandpaper.")
    assert should_reflect(s, tmp_path, kind="attunement", turn_index=1) is True


def test_per_kind_cursors_are_independent(tmp_path):
    sig = SalienceSignal.maximal()
    assert should_reflect(sig, tmp_path, kind="attunement", turn_index=5) is True
    # a different kind has its own cursor — fires even though attunement just fired
    assert should_reflect(sig, tmp_path, kind="monologue", turn_index=5) is True
