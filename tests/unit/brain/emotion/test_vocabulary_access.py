"""Tests for brain.emotion.vocabulary_access: how a process asks about and adds to her vocabulary (#801 section 5).

The bridge is the one owner. Another process asks it over the local web API and, with no bridge, reads her file read-only.
The two-process case with a real bridge is in tests/bridge/test_vocabulary_two_process.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import vocabulary, vocabulary_access, vocabulary_file
from brain.emotion.vocabulary_access import Found
from brain.emotion.vocabulary_file import NameResult

FILE = "emotion_vocabulary.json"


def _entry(name: str, half: float | None = 45.0) -> dict:
    return {"name": name, "description": "a real description", "category": "persona_extension", "decay_half_life_days": half, "intensity_clamp": 10.0}


def _write(persona: Path, entries: list) -> Path:
    path = persona / FILE
    path.write_text(json.dumps({"version": 1, "emotions": entries}), encoding="utf-8")
    return path


class _Calls(list):
    """The calls made to the bridge, as (persona_dir, route, body, timeout); `answer` is what the bridge says next."""

    answer: object = {"error": "bridge unreachable"}


@pytest.fixture
def bridge_calls(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    calls = _Calls()

    def post(persona_dir: Path, route: str, body: dict, *, timeout: float):
        calls.append((persona_dir, route, body, timeout))
        return calls.answer

    monkeypatch.setattr(vocabulary_access, "post_to_bridge", post)
    return calls


# ---------------------------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------------------------


def test_the_owner_answers_from_its_registry_with_no_call(tmp_path: Path, bridge_calls: list) -> None:
    vocabulary.register(vocabulary.Emotion("Quiet Dread", "dread, still", "persona_extension", 6.0))
    vocabulary.set_bridge_owner(True)  # tests/conftest.py puts the flag back
    found = vocabulary_access.lookup(tmp_path, ["quiet_dread", "joy", "no_such_word"])
    assert found == {
        "quiet_dread": Found("Quiet Dread", "persona_extension", "dread, still", 6.0),
        "joy": Found("joy", "core", "happiness, delight, warmth", 3.0),
        "no_such_word": None,
    }
    assert bridge_calls == []


def test_all_is_every_entry_keyed_by_its_tidied_name(tmp_path: Path) -> None:
    vocabulary.register(vocabulary.Emotion("Quiet Dread", "dread, still", "persona_extension", 6.0))
    vocabulary.set_bridge_owner(True)
    everything = vocabulary_access.lookup(tmp_path, vocabulary_access.ALL)
    assert everything["quiet_dread"].spelling == "Quiet Dread"
    assert everything["joy"].category == "core"
    assert all(vocabulary.canonical_name(k) == k for k in everything)


def test_another_process_asks_the_bridge_once_for_the_whole_batch(tmp_path: Path, bridge_calls: list) -> None:
    bridge_calls.answer = {
        "entries": [
            {"name": "Far Word", "spelling": "far_word", "category": "persona_extension", "description": "d", "half_life_days": None},
            {"name": "nope", "unknown": True},
        ]
    }
    found = vocabulary_access.lookup(tmp_path, ["Far Word", "nope"])
    assert found == {"Far Word": Found("far_word", "persona_extension", "d", None), "nope": None}
    assert bridge_calls == [(tmp_path, "/vocabulary/lookup", {"names": ["Far Word", "nope"]}, dev_constants.VOCABULARY_BRIDGE_TIMEOUT_S)]
    assert vocabulary.get("far_word") is None  # no copy of her vocabulary is kept in this process


def test_an_empty_batch_makes_no_call(tmp_path: Path, bridge_calls: list) -> None:
    assert vocabulary_access.lookup(tmp_path, []) == {}
    assert bridge_calls == []


@pytest.mark.parametrize(
    "answer",
    [
        {"error": "bridge unreachable"},
        {"error": "bridge timeout"},
        {"error": "bridge error 500"},
        {"entries": "not a list"},
        {"entries": [{"name": "x"}]},
        ["not", "a", "dict"],
    ],
)
def test_no_bridge_or_a_bridge_that_does_not_answer_a_lookup_reads_her_file_read_only(
    tmp_path: Path, bridge_calls: list, answer: object
) -> None:
    path = _write(tmp_path, [_entry("Quiet Dread", 6.0)])
    before = path.read_bytes()
    bridge_calls.answer = answer
    found = vocabulary_access.lookup(tmp_path, ["quiet_dread", "no_such_word"])
    assert found["quiet_dread"] == Found("Quiet Dread", "persona_extension", "a real description", 6.0)
    assert found["no_such_word"] is None
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [FILE]


def test_a_body_that_is_not_json_is_handled_like_no_bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def post(*_a, **_k):
        raise json.JSONDecodeError("bad", "", 0)

    monkeypatch.setattr(vocabulary_access, "post_to_bridge", post)
    assert vocabulary_access.lookup(tmp_path, ["joy"])["joy"] is not None


def test_no_bridge_and_a_file_that_cannot_be_read_leaves_only_the_framework_names(tmp_path: Path, bridge_calls: list) -> None:
    (tmp_path / FILE).write_text("{not json", encoding="utf-8")
    found = vocabulary_access.lookup(tmp_path, ["joy", "quiet_dread"])
    assert found["joy"] is not None and found["quiet_dread"] is None
    assert (tmp_path / FILE).read_text(encoding="utf-8") == "{not json"  # nothing quarantined, reset or rewritten


# ---------------------------------------------------------------------------------------------
# check (the lookup outcome of section 3, and the add)
# ---------------------------------------------------------------------------------------------


def test_the_owner_adds_through_the_shared_writer_with_no_call(tmp_path: Path, bridge_calls: list) -> None:
    vocabulary.set_bridge_owner(True)
    results = vocabulary_access.check(tmp_path, [("Fresh Word", {"description": "new"}), ("---", None)])
    assert results == [NameResult("Fresh Word", "fresh_word", "used", True), NameResult("---", None, "refused")]
    assert bridge_calls == []
    assert [e["name"] for e in json.loads((tmp_path / FILE).read_text(encoding="utf-8"))["emotions"]] == ["fresh_word"]
    assert vocabulary.get("fresh_word") is not None


def test_another_process_sends_the_whole_batch_to_the_bridge_and_writes_nothing(tmp_path: Path, bridge_calls: list) -> None:
    bridge_calls.answer = {
        "results": [
            {"name": "Fresh Word", "spelling": "fresh_word", "outcome": "used", "changed": True},
            {"name": "---", "spelling": None, "outcome": "refused", "changed": False},
        ]
    }
    results = vocabulary_access.check(tmp_path, [("Fresh Word", {"description": "new"}), ("---", None)])
    assert results == [NameResult("Fresh Word", "fresh_word", "used", True), NameResult("---", None, "refused")]
    assert bridge_calls == [
        (
            tmp_path,
            "/vocabulary/add",
            {"items": [{"name": "Fresh Word", "fields": {"description": "new"}}, {"name": "---", "fields": None}]},
            dev_constants.VOCABULARY_BRIDGE_TIMEOUT_S,
        )
    ]
    assert not (tmp_path / FILE).exists()
    assert vocabulary.get("fresh_word") is None


def test_with_no_bridge_a_name_in_her_file_is_used_under_the_file_spelling_and_any_other_is_discarded(
    tmp_path: Path, bridge_calls: list
) -> None:
    """Criterion 5, at the level of the check: nothing new is added, nothing is written, a new name is not used."""
    path = _write(tmp_path, [_entry("Quiet Dread")])
    before = path.read_bytes()
    results = vocabulary_access.check(tmp_path, [("quiet_dread", None), ("Brand New Feeling", None), ("", None)])
    assert results == [
        NameResult("quiet_dread", "Quiet Dread", "used"),
        NameResult("Brand New Feeling", None, "missing"),
        NameResult("", None, "refused"),
    ]
    assert path.read_bytes() == before
    assert vocabulary.get("brand_new_feeling") is None
    assert sorted(p.name for p in tmp_path.iterdir()) == [FILE]


def test_with_no_bridge_and_no_file_no_file_is_made(tmp_path: Path, bridge_calls: list) -> None:
    results = vocabulary_access.check(tmp_path, [("brand_new", None), ("joy", None)])
    assert [r.outcome for r in results] == ["missing", "used"]
    assert list(tmp_path.iterdir()) == []


def test_an_answer_with_the_wrong_number_of_results_is_not_trusted(tmp_path: Path, bridge_calls: list) -> None:
    bridge_calls.answer = {"results": []}
    assert [r.outcome for r in vocabulary_access.check(tmp_path, [("brand_new", None)])] == ["missing"]


def test_an_empty_batch_adds_nothing_and_makes_no_call(tmp_path: Path, bridge_calls: list) -> None:
    assert vocabulary_access.check(tmp_path, []) == []
    assert bridge_calls == []


def test_the_bridge_timeout_is_a_named_dev_constant() -> None:
    assert dev_constants.VOCABULARY_BRIDGE_TIMEOUT_S > dev_constants.EMOTION_VOCABULARY_LOCK_WAIT_S


def test_the_entry_fields_a_caller_may_give_are_the_files_fields_but_the_name() -> None:
    assert set(vocabulary_access.ENTRY_FIELDS) == {"description", "category", "decay_half_life_days", "intensity_clamp"}
    assert vocabulary_file.vocabulary_path(Path("/x")) == Path("/x") / FILE
