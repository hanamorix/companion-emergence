"""The check on a feeling name (#801 section 6): `vocabulary_access.names_used`, `names_known`, `check_feelings`, and the modes of
`engine_utils.resolve_emotions` that run it.

Synthetic persona folders in tmp; the bridge is faked at `post_to_bridge`. The store-side checks are in
tests/unit/brain/memory/test_store_name_check.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.emotion import engine_utils, vocabulary, vocabulary_access
from brain.memory.store import Memory

FILE = "emotion_vocabulary.json"


def _entry(name: str) -> dict:
    return {"name": name, "description": "a real description", "category": "persona_extension", "decay_half_life_days": 45.0, "intensity_clamp": 10.0}


def _write(persona: Path, entries: list) -> Path:
    path = persona / FILE
    path.write_text(json.dumps({"version": 1, "emotions": entries}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _retry_window_clean():
    vocabulary_access._ADD_RETRY_AFTER.clear()
    yield
    vocabulary_access._ADD_RETRY_AFTER.clear()


class _Bridge(list):
    """The calls made to the bridge, as (route, body); `reply` is what the bridge answers (a function of the call)."""

    reply = staticmethod(lambda route, body: {"error": "bridge unreachable"})


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> _Bridge:
    calls = _Bridge()

    def post(persona_dir: Path, route: str, body: dict, *, timeout: float):
        calls.append((route, body))
        return calls.reply(route, body)

    monkeypatch.setattr(vocabulary_access, "post_to_bridge", post)
    return calls


def _answer_add(outcomes: dict[str, tuple[str | None, str]]):
    """A bridge reply to `/vocabulary/add`: each raw name maps to (spelling, outcome)."""

    def reply(route: str, body: dict):
        assert route == vocabulary_access.ADD_ROUTE
        return {"results": [{"name": i["name"], "spelling": outcomes[i["name"]][0], "outcome": outcomes[i["name"]][1], "changed": False} for i in body["items"]]}

    return reply


# ---------------------------------------------------------------------------------------------
# names_used
# ---------------------------------------------------------------------------------------------


def test_a_framework_name_is_answered_from_the_registry_and_nothing_is_asked(tmp_path: Path, bridge: _Bridge) -> None:
    assert vocabulary_access.names_used(tmp_path, ["Joy", "LOVE"]) == {"Joy": "joy", "LOVE": "love"}
    assert bridge == []


def test_with_no_persona_folder_only_this_process_registry_answers(bridge: _Bridge) -> None:
    vocabulary.register(vocabulary.Emotion("Quiet Dread", "dread, still", "persona_extension", 6.0))
    assert vocabulary_access.names_used(None, ["quiet-dread", "unknown_word", "joy"]) == {"quiet-dread": "Quiet Dread", "unknown_word": None, "joy": "joy"}
    assert bridge == []


def test_another_process_makes_one_check_for_every_name_outside_the_framework(tmp_path: Path, bridge: _Bridge) -> None:
    bridge.reply = _answer_add({"Quiet Dread": ("Quiet Dread", "used"), "new word": ("new_word", "used"), "---": (None, "refused")})
    got = vocabulary_access.names_used(tmp_path, ["Quiet Dread", "joy", "new word", "---", "Quiet Dread"])
    assert got == {"Quiet Dread": "Quiet Dread", "joy": "joy", "new word": "new_word", "---": None}
    # "---" is refused by the gate before anything is asked; the other two names go in one call, each once
    assert len(bridge) == 1 and bridge[0][0] == vocabulary_access.ADD_ROUTE
    assert [i["name"] for i in bridge[0][1]["items"]] == ["Quiet Dread", "new word"]


def test_with_no_bridge_a_name_in_her_file_is_used_under_the_files_spelling_and_the_file_is_not_touched(tmp_path: Path, bridge: _Bridge) -> None:
    path = _write(tmp_path, [_entry("Quiet Dread")])
    before = path.read_bytes()
    got = vocabulary_access.names_used(tmp_path, ["quiet-dread", "never_seen"])
    assert got == {"quiet-dread": "Quiet Dread", "never_seen": None}
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [FILE]  # no backup, lock or heal file either


def test_a_discard_for_want_of_a_bridge_leaves_one_warning(tmp_path: Path, bridge: _Bridge, caplog: pytest.LogCaptureFixture) -> None:
    from brain.utils import log_throttle

    log_throttle._last.clear()
    with caplog.at_level(logging.WARNING, logger="brain.emotion.vocabulary_access"):
        vocabulary_access.names_used(tmp_path, ["never_seen_zz"])
    assert any("discarded" in r.getMessage() and "never_seen_zz" in r.getMessage() for r in caplog.records)


def test_the_owner_uses_a_name_it_holds_with_no_file_work_and_adds_a_new_one_that_passes_the_gate(tmp_path: Path, bridge: _Bridge) -> None:
    vocabulary.register(vocabulary.Emotion("Quiet Dread", "dread, still", "persona_extension", 6.0))
    vocabulary.set_bridge_owner(True)  # tests/conftest.py puts the flag back
    assert vocabulary_access.names_used(tmp_path, ["quiet-dread"]) == {"quiet-dread": "Quiet Dread"}
    assert not (tmp_path / FILE).exists()  # a name it holds costs no file write
    got = vocabulary_access.names_used(tmp_path, ["Brand New Word", "---"])
    assert got == {"Brand New Word": "brand_new_word", "---": None}
    names = [e["name"] for e in json.loads((tmp_path / FILE).read_text(encoding="utf-8"))["emotions"]]
    assert names == ["brand_new_word"]
    assert vocabulary.get("Brand-New-Word") is not None
    assert bridge == []  # the owner asks nobody


def test_a_name_the_owner_could_not_write_is_kept_tidied_and_not_asked_again_inside_the_window(
    tmp_path: Path, bridge: _Bridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(vocabulary_access, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    bridge.reply = _answer_add({"Slow Word": (None, "cannot_write")})
    assert vocabulary_access.names_used(tmp_path, ["Slow Word"]) == {"Slow Word": "slow_word"}  # kept tidied
    assert len(bridge) == 1
    clock[0] += 5.0
    assert vocabulary_access.names_used(tmp_path, ["Slow Word"]) == {"Slow Word": "slow_word"}
    assert len(bridge) == 1, "inside the window the name is not asked about again"
    clock[0] += 10_000.0
    vocabulary_access.names_used(tmp_path, ["Slow Word"])
    assert len(bridge) == 2, "after the window the next read tries again"


def test_a_name_that_is_added_later_clears_its_retry_window(tmp_path: Path, bridge: _Bridge, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(vocabulary_access, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    bridge.reply = _answer_add({"Slow Word": (None, "cannot_write")})
    vocabulary_access.names_used(tmp_path, ["Slow Word"])
    clock[0] += 10_000.0
    bridge.reply = _answer_add({"Slow Word": ("slow_word", "used")})
    assert vocabulary_access.names_used(tmp_path, ["Slow Word"]) == {"Slow Word": "slow_word"}
    assert vocabulary_access._ADD_RETRY_AFTER == {}


# ---------------------------------------------------------------------------------------------
# names_known: a read of another file's names never adds
# ---------------------------------------------------------------------------------------------


def test_names_known_never_adds_a_name_even_in_the_owner(tmp_path: Path, bridge: _Bridge) -> None:
    vocabulary.register(vocabulary.Emotion("Quiet Dread", "dread, still", "persona_extension", 6.0))
    vocabulary.set_bridge_owner(True)
    assert vocabulary_access.names_known(tmp_path, ["quiet-dread", "Brand New Word", "Joy"]) == {"quiet-dread": "Quiet Dread", "Brand New Word": None, "Joy": "joy"}
    assert not (tmp_path / FILE).exists()


def test_names_known_asks_the_lookup_route_once_in_another_process(tmp_path: Path, bridge: _Bridge) -> None:
    bridge.reply = lambda route, body: {
        "entries": [
            {"name": "a b", "spelling": "A B", "category": "persona_extension", "description": "d", "half_life_days": None},
            {"name": "zz", "unknown": True},
        ]
    }
    assert vocabulary_access.names_known(tmp_path, ["a b", "zz", "joy"]) == {"a b": "A B", "zz": None, "joy": "joy"}
    assert [c[0] for c in bridge] == [vocabulary_access.LOOKUP_ROUTE]


# ---------------------------------------------------------------------------------------------
# check_feelings: the one helper for a write
# ---------------------------------------------------------------------------------------------


def test_check_feelings_uses_the_registered_spelling_and_averages_two_spellings(tmp_path: Path, bridge: _Bridge) -> None:
    """Spec criterion 6, the write half: `Joy` with `joy` registered is `joy`; two spellings on one memory carry their mean."""
    assert vocabulary_access.check_feelings(tmp_path, {"Joy": 6.0}) == {"joy": 6.0}
    assert vocabulary_access.check_feelings(tmp_path, {"Joy": 4.0, "joy": 8.0, "love": 2.0}) == {"joy": 6.0, "love": 2.0}


def test_check_feelings_drops_a_refused_name_and_a_key_that_is_not_text(tmp_path: Path, bridge: _Bridge) -> None:
    got = vocabulary_access.check_feelings(tmp_path, {"---": 3.0, 7: 2.0, "joy": 1.0})  # type: ignore[dict-item]
    assert got == {"joy": 1.0}


def test_a_value_too_large_for_a_float_does_not_raise_in_the_check(tmp_path: Path, bridge: _Bridge) -> None:
    assert vocabulary_access.finite_number(10**400) is False
    got = vocabulary_access.check_feelings(tmp_path, {"Joy": 10**400, "joy": 4.0})
    assert got == {"joy": 4.0}


def test_check_feelings_keeps_a_lone_value_that_is_not_a_number_as_given(tmp_path: Path, bridge: _Bridge) -> None:
    assert vocabulary_access.check_feelings(tmp_path, {"joy": "high"}) == {"joy": "high"}


# ---------------------------------------------------------------------------------------------
# resolve_emotions and the readers
# ---------------------------------------------------------------------------------------------


def test_resolve_emotions_tidy_makes_no_check(bridge: _Bridge) -> None:
    got = engine_utils.resolve_emotions({"Unknown Word": 3.0, "Joy": 2.0}, check="tidy")
    assert got == {"unknown_word": 3.0, "joy": 2.0}
    assert bridge == []


def test_resolve_emotions_registry_drops_a_name_this_process_does_not_hold(bridge: _Bridge) -> None:
    got = engine_utils.resolve_emotions({"Unknown Word": 3.0, "Joy": 2.0}, check="registry")
    assert got == {"joy": 2.0}
    assert bridge == []


def test_resolve_emotions_with_a_persona_folder_asks_once_and_keys_by_the_tidied_name(tmp_path: Path, bridge: _Bridge) -> None:
    bridge.reply = _answer_add({"Quiet Dread": ("Quiet Dread", "used"), "quiet-dread": ("Quiet Dread", "used"), "gone": (None, "missing")})
    got = engine_utils.resolve_emotions({"Quiet Dread": 2.0, "quiet-dread": 6.0, "gone": 9.0, "Joy": 5.0}, check=tmp_path)
    assert got == {"quiet_dread": 4.0, "joy": 5.0}
    assert len(bridge) == 1


def test_a_row_stored_under_two_spellings_reads_as_one_name_with_their_mean() -> None:
    """Spec criterion 6, the read half."""
    memory = Memory.from_stored_row(
        id="m", content="x", memory_type="conversation", domain="us", created_at=Memory.create_new("x", "conversation", "us").created_at,
        emotions={"Joy": 4.0, "joy": 8.0},
    )
    assert engine_utils.memory_feelings(memory, None, None, None).feelings == {"joy": 6.0}


def test_a_row_with_a_name_this_process_does_not_hold_reads_without_it_and_is_not_changed() -> None:
    memory = Memory.create_new("x", "conversation", "us")
    memory._stored_emotions = {"joy": 3.0, "stray_word": 7.0}
    assert engine_utils.memory_feelings(memory, None, None, None).feelings == {"joy": 3.0}
    assert memory._stored_emotions == {"joy": 3.0, "stray_word": 7.0}  # a read never rewrites the row


def test_faded_emotion_finds_a_stored_value_by_tidied_name() -> None:
    memory = Memory.create_new("x", "conversation", "us")
    memory._stored_emotions = {"Joy": 6.0}
    assert engine_utils.faded_emotion(memory, "joy", None, None, None) == 6.0
    memory._stored_emotions = {"Joy": 4.0, "JOY": 8.0}
    assert engine_utils.faded_emotion(memory, "joy", None, None, None) == 6.0
    assert engine_utils.faded_emotion(memory, "Joy", None, None, None) == 4.0  # the exact stored key first
    with pytest.raises(KeyError):
        engine_utils.faded_emotion(memory, "love", None, None, None)
