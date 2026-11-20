"""The shared importance stretch (`brain.memory.memory_utils`), used by the emotion fade and the forgetting pass.

One function holds the formula (`importance_stretch`) and one holds the clamp (`importance_ratio`). These tests pin the values,
that both callers read the one function (and the one gain) at call time, that the forgetting results did not change, and that
the modules import in every order without a cycle.

The module imports neither `memory_utils` nor `engine_utils` at the top: both are absent on the base, and test (c), which passes
on the base by design, has to be collectable there. Each test that needs one imports it inside.

Naming: synthetic persona = Canary. No real persona or user names.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

from brain import dev_constants
from brain.felt_time.state import FeltTimeState
from brain.forgetting import policy, salience
from brain.memory.store import Memory
from tests.source_tree import REPO

_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
_HORIZON = salience._FRESHNESS_LIVED_HOURS_HORIZON


class _Frozen(datetime):
    """A `datetime` whose `now()` is fixed, so a freshness value is a function of the memory's age alone."""

    @classmethod
    def now(cls, tz=None):
        return _NOW if tz is None else _NOW.astimezone(tz)


def _memory(importance: float, *, state: str = "active", lived_hours: float = 0.0) -> Memory:
    """A synthetic memory `lived_hours` old at the frozen clock (`felt` below makes lived hours equal wall hours)."""
    memory = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions={}, importance=0.0)
    object.__setattr__(memory, "importance", importance)
    object.__setattr__(memory, "state", state)
    object.__setattr__(memory, "created_at", _NOW - timedelta(hours=lived_hours))
    return memory


_FELT = FeltTimeState(lived_age_hours=1.0)  # no first tick: the lived rate is 1.0, lived hours = wall hours


def _freshness(importance: float, lived_hours: float) -> float:
    return salience._freshness_input(_memory(importance, lived_hours=lived_hours), _FELT)


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(salience, "datetime", _Frozen)


# --- (a) values -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("importance", "ratio"),
    [(-3.0, 0.0), (0.0, 0.0), (5.0, 0.5), (10.0, 1.0), (12.0, 1.0)],
)
def test_importance_ratio_is_the_importance_over_ten_held_to_zero_and_one(importance: float, ratio: float) -> None:
    from brain.memory import memory_utils

    assert memory_utils.importance_ratio(importance) == ratio


@pytest.mark.parametrize(
    ("importance", "stretch"),
    [(-3.0, 1.0), (0.0, 1.0), (5.0, 4.5), (10.0, 8.0), (12.0, 8.0)],
)
def test_importance_stretch_is_one_plus_gain_times_the_ratio(
    monkeypatch: pytest.MonkeyPatch, importance: float, stretch: float
) -> None:
    from brain.memory import memory_utils

    monkeypatch.setattr(dev_constants, "HORIZON_IMPORTANCE_GAIN", 7.0)
    assert memory_utils.importance_stretch(importance) == stretch


def test_a_non_finite_importance_fails_toward_the_baseline_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # `min(1.0, max(0.0, nan / 10.0))` is 0.0: `max(0.0, nan)` keeps 0.0. The order of min and max matters here.
    from brain.memory import memory_utils

    monkeypatch.setattr(dev_constants, "HORIZON_IMPORTANCE_GAIN", 7.0)
    assert memory_utils.importance_ratio(float("nan")) == 0.0
    assert memory_utils.importance_stretch(float("nan")) == 1.0
    assert memory_utils.importance_ratio(float("inf")) == 1.0
    assert memory_utils.importance_stretch(float("inf")) == 8.0
    assert memory_utils.importance_ratio(float("-inf")) == 0.0
    assert memory_utils.importance_stretch(float("-inf")) == 1.0


# --- (b) one function, one number: both callers read them at call time ---------------------------------------------------


def test_b1_the_stretch_reads_the_gain_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # Literal numbers, not built from `importance_stretch`: a hard-coded 52.0 or a gain bound at import fails here.
    from brain.memory import memory_utils

    monkeypatch.setattr(dev_constants, "HORIZON_IMPORTANCE_GAIN", 7.0)
    assert memory_utils.importance_stretch(0.0) == 1.0
    assert memory_utils.importance_stretch(5.0) == 4.5
    assert memory_utils.importance_stretch(10.0) == 8.0


def test_b2_the_stretch_reads_the_ratio_from_the_one_function(monkeypatch: pytest.MonkeyPatch) -> None:
    # A stretch that clamps the importance itself ignores the patched ratio and fails.
    from brain.memory import memory_utils

    monkeypatch.setattr(dev_constants, "HORIZON_IMPORTANCE_GAIN", 7.0)
    monkeypatch.setattr(memory_utils, "importance_ratio", lambda importance: 0.25)
    assert memory_utils.importance_stretch(3.0) == 2.75
    assert memory_utils.importance_stretch(9.0) == 2.75


def test_b3_both_callers_follow_the_gain_so_neither_holds_a_copy_of_the_number(
    monkeypatch: pytest.MonkeyPatch, frozen_clock: None
) -> None:
    from brain.emotion import engine_utils

    monkeypatch.setattr(dev_constants, "HORIZON_IMPORTANCE_GAIN", 7.0)
    # Importance 5 at gain 7.0 stretches by 4.5: a half-life of 14 days becomes 63, a horizon of 2160 hours becomes 9720.
    assert engine_utils.fade(8.0, 63.0, 14.0, 5.0) == 4.0
    assert _freshness(5.0, _HORIZON * 4.5 / 2.0) == pytest.approx(0.5)


def test_b4_both_callers_follow_the_function_so_neither_holds_a_copy_of_the_formula(
    monkeypatch: pytest.MonkeyPatch, frozen_clock: None
) -> None:
    from brain.emotion import engine_utils
    from brain.memory import memory_utils

    # A stretch that is no longer the real formula: a caller that wrote the formula itself, or bound the name at import, ignores it.
    monkeypatch.setattr(memory_utils, "importance_stretch", lambda importance: 3.0 + importance)
    assert engine_utils.fade(8.0, 112.0, 14.0, 5.0) == 4.0  # 14 x 8.0
    assert _freshness(5.0, _HORIZON * 8.0 / 2.0) == pytest.approx(0.5)


def test_b5_the_fade_gate_follows_the_one_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.memory import memory_utils

    memory = _memory(0.0)
    # Importance 0, no arc: the gate is FADE_THRESHOLD (0.25), so salience 0.2 fades.
    assert policy.next_state(memory, salience=0.2, consecutive_low_passes=0, narrative_weight=0.0) == policy.Transition.FADE
    # The ratio read as 1.0 lowers the gate to 0.25 / (1 + FADE_IMPORTANCE_GAIN), well under 0.2: nothing happens. A `next_state`
    # that clamps the importance itself would still fade.
    monkeypatch.setattr(memory_utils, "importance_ratio", lambda importance: 1.0)
    assert policy.next_state(memory, salience=0.2, consecutive_low_passes=0, narrative_weight=0.0) == policy.Transition.NONE


# --- (c) the forgetting results are those of origin/main, within float rounding -----------------------------------------
# Values taken on origin/main before any source edit by the scratch script in the build records (base_values.py): the
# unchanged `_freshness_input` and `next_state`. The shared stretch groups `gain x (importance / 10)` where the old code wrote
# `gain x importance / 10`, so a last-bit difference is possible and allowed (owner ruling 2026-11-03: not bit-exact).

_FRESH_HOURS = (0.0, 100.0, 2160.0, 5000.0, 20000.0, 60000.0)
_FRESH = {
    -4.0: (1.0, 0.9537037037037037, 0.0, 0.0, 0.0, 0.0),
    0.0: (1.0, 0.9537037037037037, 0.0, 0.0, 0.0, 0.0),
    3.0: (1.0, 0.9972110664881749, 0.9397590361445783, 0.8605533244087461, 0.4422132976349844, 0.0),
    5.0: (1.0, 0.9982853223593965, 0.962962962962963, 0.9142661179698217, 0.6570644718792866, 0.0),
    7.5: (1.0, 0.9988425925925926, 0.975, 0.9421296296296297, 0.7685185185185185, 0.3055555555555556),
    10.0: (1.0, 0.9991264849755416, 0.9811320754716981, 0.956324248777079, 0.8252969951083159, 0.47589098532494756),
    14.0: (1.0, 0.9991264849755416, 0.9811320754716981, 0.956324248777079, 0.8252969951083159, 0.47589098532494756),
}

_SALIENCES = (0.05, 0.07, 0.08, 0.12, 0.2, 0.25, 0.3)
# (state, narrative weight, importance) -> the transition at each salience above, two low passes
_STATES = {
    ("active", 0.0, -2.0): ("fade", "fade", "fade", "fade", "fade", "none", "none"),
    ("active", 0.0, 0.0): ("fade", "fade", "fade", "fade", "fade", "none", "none"),
    ("active", 0.0, 4.0): ("fade", "fade", "fade", "fade", "none", "none", "none"),
    ("active", 0.0, 10.0): ("fade", "fade", "fade", "none", "none", "none", "none"),
    ("active", 0.0, 15.0): ("fade", "fade", "fade", "none", "none", "none", "none"),
    ("active", 0.5, -2.0): ("fade", "fade", "fade", "fade", "none", "none", "none"),
    ("active", 0.5, 0.0): ("fade", "fade", "fade", "fade", "none", "none", "none"),
    ("active", 0.5, 4.0): ("fade", "fade", "fade", "none", "none", "none", "none"),
    ("active", 0.5, 10.0): ("fade", "fade", "none", "none", "none", "none", "none"),
    ("active", 0.5, 15.0): ("fade", "fade", "none", "none", "none", "none", "none"),
    ("fading", 0.0, -2.0): ("lose", "lose", "lose", "none", "none", "unfade", "unfade"),
    ("fading", 0.0, 0.0): ("lose", "lose", "lose", "none", "none", "unfade", "unfade"),
    ("fading", 0.0, 4.0): ("lose", "lose", "lose", "none", "unfade", "unfade", "unfade"),
    ("fading", 0.0, 10.0): ("lose", "lose", "lose", "unfade", "unfade", "unfade", "unfade"),
    ("fading", 0.0, 15.0): ("lose", "lose", "lose", "unfade", "unfade", "unfade", "unfade"),
    ("fading", 0.5, -2.0): ("lose", "lose", "lose", "none", "unfade", "unfade", "unfade"),
    ("fading", 0.5, 0.0): ("lose", "lose", "lose", "none", "unfade", "unfade", "unfade"),
    ("fading", 0.5, 4.0): ("lose", "lose", "lose", "unfade", "unfade", "unfade", "unfade"),
    ("fading", 0.5, 10.0): ("lose", "lose", "unfade", "unfade", "unfade", "unfade", "unfade"),
    ("fading", 0.5, 15.0): ("lose", "lose", "unfade", "unfade", "unfade", "unfade", "unfade"),
}


@pytest.mark.no_base_failure(
    reason="a refactor guard: the freshness values are those of origin/main, so it passes there by design (the shared stretch changes no result)"
)
@pytest.mark.parametrize("importance", sorted(_FRESH))
def test_c_freshness_gives_the_values_origin_main_gave(importance: float, frozen_clock: None) -> None:
    for hours, expected in zip(_FRESH_HOURS, _FRESH[importance], strict=True):
        assert _freshness(importance, hours) == pytest.approx(expected, rel=1e-12, abs=1e-12), (importance, hours)


@pytest.mark.no_base_failure(
    reason="a refactor guard: the transitions are those of origin/main, so it passes there by design (the shared ratio changes no result)"
)
@pytest.mark.parametrize(("state", "narrative", "importance"), sorted(_STATES))
def test_c_next_state_gives_the_transitions_origin_main_gave(state: str, narrative: float, importance: float) -> None:
    got = tuple(
        str(
            policy.next_state(
                _memory(importance, state=state), salience=s, consecutive_low_passes=2, narrative_weight=narrative
            ).value
        )
        for s in _SALIENCES
    )
    assert got == _STATES[(state, narrative, importance)]


# --- (d) no import cycle, in any order --------------------------------------------------------------------------------

_MODULES = ("brain.emotion.engine_utils", "brain.forgetting.salience", "brain.memory.memory_utils")


@pytest.mark.parametrize("order", list(itertools.permutations(_MODULES)), ids=lambda order: "-".join(m.rsplit(".", 1)[1] for m in order))
def test_d_the_three_modules_import_in_any_order(order: tuple[str, ...]) -> None:
    code = f"import sys, importlib\nsys.path.insert(0, {str(REPO)!r})\nfor name in {order!r}:\n    importlib.import_module(name)\n"
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
