"""#648: `contentment`, `peace`, `calm` and `groundedness` are registered feelings.

`brain/body/state.py` raises body energy when `peace` is high in a fresh session, and no vocabulary registered `peace`, so the
engine's `read()` left it out, `feels_high("peace")` was always False and the rule could never fire. The four feelings are baseline
entries now (`contentment` core; `peace`, `calm`, `groundedness` complex, flavours of contentment), their half-lives are written once
in `dev_constants.EMOTION_HALF_LIFE_DAYS`, and the four descriptions are the owner's words. Personas here are tmp directories with
synthetic state; nothing touches a real persona.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from brain import dev_constants
from brain.body.state import body_state_from_short
from brain.emotion import engine as engine_module
from brain.emotion import engine_files, vocabulary
from brain.emotion.engine_utils import Average, EmotionState, EngineState, SnapshotDay
from tests.unit.brain.emotion._engine_support import install, make_persona

# (name, category, description, half-life in days): the owner's definitions (#648).
FOUR = (
    ("contentment", "core", "the quiet pleasure of what is, safe, wanting nothing more", 1.0),
    ("peace", "complex", "settled and unhurried, nothing pressing", 4.0 / 24.0),
    ("calm", "complex", "steady and unruffled, nothing stirred up", 4.0 / 24.0),
    ("groundedness", "complex", "rooted and steady, sure of where you stand", 2.0),
)


@pytest.mark.parametrize(("name", "category", "description", "days"), FOUR, ids=[f[0] for f in FOUR])
def test_the_feeling_is_registered_with_the_owners_definition(name, category, description, days):
    emotion = vocabulary.get(name)
    assert emotion is not None
    assert (emotion.category, emotion.description, emotion.decay_half_life_days) == (category, description, days)


@pytest.mark.parametrize(("name", "category", "description", "days"), FOUR, ids=[f[0] for f in FOUR])
def test_the_half_life_is_in_the_engines_table_and_the_baseline_reads_it_from_there(name, category, description, days):
    assert dev_constants.EMOTION_HALF_LIFE_DAYS[name] == days
    baseline = {e.name: e for e in vocabulary._BASELINE}[name]
    assert baseline.decay_half_life_days == dev_constants.EMOTION_HALF_LIFE_DAYS[name]


def test_the_four_baseline_entries_take_their_half_life_from_the_table_and_hold_no_literal_of_their_own():
    """One source: each of the four `Emotion(...)` calls in the baseline passes `EMOTION_HALF_LIFE_DAYS[<name>]`, not a number."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(vocabulary))
    found: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Emotion"
            and len(node.args) >= 4
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in {f[0] for f in FOUR}
        ):
            found[node.args[0].value] = node.args[3]
    assert set(found) == {f[0] for f in FOUR}
    for name, argument in found.items():
        assert isinstance(argument, ast.Subscript), name
        assert ast.unparse(argument) == f"EMOTION_HALF_LIFE_DAYS['{name}']"


def test_the_engine_knows_the_half_life_of_each_without_a_persona_file(tmp_path):
    engine = engine_module.EmotionEngine(tmp_path)
    for name, _category, _description, days in FOUR:
        assert engine.half_life(name) == days


def test_the_extractor_keeps_the_four_names_instead_of_dropping_them():
    from brain.chat.extractor import _filter_to_registered

    asked = {"contentment": 3.0, "peace": 2.0, "calm": 4.0, "groundedness": 5.0, "not_a_feeling_zz": 1.0}
    assert _filter_to_registered(asked) == {"contentment": 3.0, "peace": 2.0, "calm": 4.0, "groundedness": 5.0}


def _days(*shorts: float) -> list[SnapshotDay]:
    return [SnapshotDay(f"2026-10-{i + 1:02d}", s, 0.0, True, None) for i, s in enumerate(shorts)]


def test_the_body_energy_rule_fires_in_a_fresh_session_when_peace_is_high_through_the_real_engine(tmp_path):
    """No patched `read()`: a synthetic persona's state file holds `peace` in the upper half of its own 30-day range.

    The persona's vocabulary file has no `peace` entry, so only the baseline can register it. On the old code the engine's `read()`
    left the unregistered name out, `feels_high("peace")` was False, and the energy was the same as with no peace at all.
    """
    persona = make_persona(tmp_path / "persona")
    saved_at = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
    state = EngineState(saved_at=saved_at, last_update=saved_at, active_day="2026-11-04")
    state.emotions["peace"] = EmotionState(short=Average(5.0, 1.0), long=Average(3.0, 1.0), snapshot=_days(2.0, 4.0, 6.0, 3.0))
    engine_files.save_state(persona, state)
    install(persona)
    live = engine_module.engine_for(persona)

    assert live.feels_high("peace") is True

    def energy(feels_high):
        return body_state_from_short(
            {"peace": 5.0}, feels_high=feels_high, session_hours=0.2, words_written=0, days_since_contact=0.0, now=saved_at
        ).energy

    assert energy(live.feels_high) == energy(lambda name: False) + 1
