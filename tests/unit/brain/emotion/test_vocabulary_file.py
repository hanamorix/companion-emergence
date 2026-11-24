"""Tests for brain.emotion.vocabulary_file: the one writer of her vocabulary file (#741, #801 section 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import vocabulary, vocabulary_file
from brain.emotion.vocabulary_file import NameResult
from brain.health.json_io import JsonFileError, Recovery
from brain.health.reconstruct import PLACEHOLDER_DECAY_DAYS, PLACEHOLDER_DESCRIPTION
from brain.utils import file_lock


def _entry(name: str, desc: str = "a real description", half: float | None = 45.0) -> dict:
    return {
        "name": name,
        "description": desc,
        "category": "persona_extension",
        "decay_half_life_days": half,
        "intensity_clamp": 10.0,
    }


def _stub(name: str) -> dict:
    return _entry(name, PLACEHOLDER_DESCRIPTION, PLACEHOLDER_DECAY_DAYS)


def _write(tmp_path: Path, entries: list) -> Path:
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(json.dumps({"version": 1, "emotions": entries}), encoding="utf-8")
    return path


def _read(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))["emotions"]


@pytest.fixture
def writes(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Every `write_json` the shared module makes, with its keyword arguments; the real write still happens."""
    calls: list[dict] = []
    real = vocabulary_file.write_json

    def spy(path, data, **kwargs):
        calls.append({"path": path, **kwargs})
        return real(path, data, **kwargs)

    monkeypatch.setattr(vocabulary_file, "write_json", spy)
    return calls


# ---------------------------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------------------------


def test_add_appends_a_new_name_under_its_tidied_form_with_default_fields_and_registers_it(tmp_path: Path, writes) -> None:
    path = _write(tmp_path, [_entry("existing one")])
    [res] = vocabulary_file.add(path, [("Fresh Feeling", None)])
    assert res == NameResult("Fresh Feeling", "fresh_feeling", "used", True)
    assert _read(path)[-1] == {
        "name": "fresh_feeling",
        "description": PLACEHOLDER_DESCRIPTION,
        "category": "persona_extension",
        "decay_half_life_days": PLACEHOLDER_DECAY_DAYS,
        "intensity_clamp": 10.0,
    }
    assert vocabulary.get("fresh_feeling") is not None
    assert len(writes) == 1


def test_add_takes_the_given_fields_over_the_defaults(tmp_path: Path) -> None:
    path = _write(tmp_path, [])
    vocabulary_file.add(path, [("wistful", {"description": "a soft longing", "decay_half_life_days": 7.0, "name": "ignored"})])
    entry = _read(path)[0]
    assert (entry["name"], entry["description"], entry["decay_half_life_days"]) == ("wistful", "a soft longing", 7.0)
    assert vocabulary.get("wistful").description == "a soft longing"


def test_add_of_a_name_already_in_the_file_is_used_under_the_file_spelling_and_writes_nothing(tmp_path: Path, writes) -> None:
    path = _write(tmp_path, [_entry("Quiet Dread Add")])
    before = path.read_bytes()
    [res] = vocabulary_file.add(path, [("quiet_dread_add", None)])
    assert res == NameResult("quiet_dread_add", "Quiet Dread Add", "used", False)
    assert path.read_bytes() == before
    assert writes == []


def test_the_registry_is_changed_after_the_file_is_written_and_inside_the_file_lock(tmp_path: Path, monkeypatch) -> None:
    path = _write(tmp_path, [])
    seen: dict = {}
    real = vocabulary_file._register_new

    def probe(entry: dict) -> None:
        seen["file_had_it"] = [e["name"] for e in _read(path)] == [entry["name"]]
        with file_lock.file_lock(path, blocking=False) as acquired:  # still held by this very transaction
            seen["lock_held"] = not acquired
        real(entry)

    monkeypatch.setattr(vocabulary_file, "_register_new", probe)
    vocabulary_file.add(path, [("ordered one", None)])
    assert seen == {"file_had_it": True, "lock_held": True}
    assert vocabulary.get("ordered_one") is not None


def test_an_error_in_the_registry_step_is_raised_and_is_not_reported_as_a_failed_write(tmp_path: Path, monkeypatch) -> None:
    path = _write(tmp_path, [])

    def broken(emotion):
        raise RuntimeError("registry bug")

    monkeypatch.setattr(vocabulary, "register", broken)
    with pytest.raises(RuntimeError, match="registry bug"):
        vocabulary_file.add(path, [("registry broken", None)])
    assert [e["name"] for e in _read(path)] == ["registry_broken"]  # the file is written
    with file_lock.file_lock(path, blocking=False) as acquired:  # and its lock was released
        assert acquired


def test_add_dedupes_against_the_file_not_the_registry(tmp_path: Path) -> None:
    """A framework name that her file does not hold is added (core refusal is a later step); the registry keeps its entry."""
    path = _write(tmp_path, [])
    [res] = vocabulary_file.add(path, [("joy", None)])
    assert res.changed and [e["name"] for e in _read(path)] == ["joy"]
    assert vocabulary.get("joy").description == "happiness, delight, warmth"


@pytest.mark.parametrize("bad", ["", "---", "x" * 101])
def test_add_refuses_a_name_the_gate_refuses_and_writes_nothing(tmp_path: Path, bad: str, writes) -> None:
    path = _write(tmp_path, [_entry("keep me")])
    before = path.read_bytes()
    assert vocabulary_file.add(path, [(bad, None)]) == [NameResult(bad, None, "refused")]
    assert path.read_bytes() == before and writes == []


def test_add_accepts_a_100_character_name(tmp_path: Path) -> None:
    path = _write(tmp_path, [])
    [res] = vocabulary_file.add(path, [("y" * 100, None)])
    assert res.outcome == "used" and res.changed


def test_add_of_a_batch_is_one_write_and_keeps_the_order_of_the_results(tmp_path: Path, writes) -> None:
    path = _write(tmp_path, [])
    results = vocabulary_file.add(path, [("one add", None), ("---", None), ("two add", None), ("one_add", None)])
    assert [(r.outcome, r.spelling, r.changed) for r in results] == [
        ("used", "one_add", True),
        ("refused", None, False),
        ("used", "two_add", True),
        ("used", "one_add", False),
    ]
    assert [e["name"] for e in _read(path)] == ["one_add", "two_add"]
    assert len(writes) == 1


def test_add_creates_the_file_when_it_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "emotion_vocabulary.json"
    vocabulary_file.add(path, [("brand new", None)])
    assert [e["name"] for e in _read(path)] == ["brand_new"]


def test_add_to_a_corrupt_file_is_cannot_write_for_every_name_and_leaves_the_file(tmp_path: Path) -> None:
    path = tmp_path / "emotion_vocabulary.json"
    path.write_bytes(b"\xff{torn")
    results = vocabulary_file.add(path, [("a corrupt", None), ("---", None)])
    assert [r.outcome for r in results] == ["cannot_write", "refused"]
    assert path.read_bytes() == b"\xff{torn"
    assert vocabulary.get("a_corrupt") is None


def test_add_to_a_file_without_an_emotions_list_is_cannot_write(tmp_path: Path) -> None:
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(json.dumps({"version": 1, "emotions": "no"}), encoding="utf-8")
    assert vocabulary_file.add(path, [("nolist", None)])[0].outcome == "cannot_write"


def test_a_write_goes_through_write_json_verified_with_the_shared_backup_count(tmp_path: Path, writes, monkeypatch) -> None:
    monkeypatch.setattr(vocabulary_file, "backup_count", lambda path: 5)
    path = _write(tmp_path, [])
    vocabulary_file.add(path, [("counted", None)])
    assert writes[0]["verify"] is True and writes[0]["backups"] == 5


def test_backup_count_is_the_adaptive_count_and_falls_back_to_the_default(tmp_path: Path, monkeypatch) -> None:
    from brain.health import adaptive

    path = tmp_path / "emotion_vocabulary.json"
    monkeypatch.setattr(
        vocabulary_file, "compute_treatment", lambda persona_dir, name: adaptive.FileTreatment(backup_count=6, verify_after_write=True)
    )
    assert vocabulary_file.backup_count(path) == 6

    def unreadable(persona_dir, name):
        raise JsonFileError(path, "io", "no")

    monkeypatch.setattr(vocabulary_file, "compute_treatment", unreadable)
    assert vocabulary_file.backup_count(path) == adaptive.DEFAULT_BACKUP_COUNT


def test_a_lock_held_past_the_wait_is_cannot_write_and_writes_nothing(tmp_path: Path, monkeypatch, writes) -> None:
    monkeypatch.setattr(dev_constants, "EMOTION_VOCABULARY_LOCK_WAIT_S", 0.2)
    path = _write(tmp_path, [])
    before = path.read_bytes()
    with file_lock.file_lock(path):
        res = vocabulary_file.add(path, [("locked out", None)])[0]
    assert res.outcome == "cannot_write"
    assert path.read_bytes() == before and writes == []
    assert vocabulary.get("locked_out") is None


def test_a_failed_write_is_cannot_write_and_the_registry_is_not_updated(tmp_path: Path, monkeypatch) -> None:
    path = _write(tmp_path, [])

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(vocabulary_file, "write_json", boom)
    assert vocabulary_file.add(path, [("no room", None)])[0].outcome == "cannot_write"
    assert vocabulary.get("no_room") is None


def test_a_write_that_did_not_verify_is_cannot_write_and_the_registry_is_not_updated(tmp_path: Path, monkeypatch) -> None:
    path = _write(tmp_path, [])
    monkeypatch.setattr(vocabulary_file, "write_json", lambda *a, **k: False)
    assert vocabulary_file.add(path, [("unverified", None)])[0].outcome == "cannot_write"
    assert vocabulary.get("unverified") is None


# ---------------------------------------------------------------------------------------------
# replace_fields
# ---------------------------------------------------------------------------------------------


def test_replace_fields_changes_the_file_entry_and_the_registry_and_keeps_other_keys(tmp_path: Path, writes) -> None:
    entry = _stub("Mend Me")
    entry["my own key"] = "kept"
    path = _write(tmp_path, [entry])
    vocabulary_file.load(path)
    results = vocabulary_file.replace_fields(path, {"mend_me": {"description": "mended", "decay_half_life_days": 9.0}})
    assert results == [NameResult("mend_me", "Mend Me", "used", True)]
    stored = _read(path)[0]
    assert (stored["name"], stored["description"], stored["decay_half_life_days"], stored["my own key"]) == ("Mend Me", "mended", 9.0, "kept")
    assert vocabulary.get("mend me").description == "mended" and vocabulary.get("mend me").name == "Mend Me"
    assert len(writes) == 1


def test_replace_fields_is_one_write_for_several_names(tmp_path: Path, writes) -> None:
    path = _write(tmp_path, [_stub("rf one"), _stub("rf two")])
    vocabulary_file.replace_fields(path, {"rf one": {"description": "1"}, "rf two": {"description": "2"}})
    assert [e["description"] for e in _read(path)] == ["1", "2"] and len(writes) == 1


def test_replace_fields_only_if_false_leaves_that_entry_alone(tmp_path: Path, writes) -> None:
    path = _write(tmp_path, [_entry("already described")])
    before = path.read_bytes()
    res = vocabulary_file.replace_fields(
        path, {"already described": {"description": "x"}}, only_if=lambda e: e["description"] == PLACEHOLDER_DESCRIPTION
    )
    assert res == [NameResult("already described", "already described", "used", False)]
    assert path.read_bytes() == before and writes == []


def test_replace_fields_with_values_the_entry_already_has_writes_nothing(tmp_path: Path, writes) -> None:
    path = _write(tmp_path, [_entry("same already")])
    res = vocabulary_file.replace_fields(path, {"same already": {"description": "a real description"}})
    assert res[0].changed is False and writes == []


def test_replace_fields_reports_a_refused_name_and_a_missing_one(tmp_path: Path) -> None:
    path = _write(tmp_path, [_entry("present")])
    results = vocabulary_file.replace_fields(path, {"---": {"description": "x"}, "absent": {"description": "x"}})
    assert [r.outcome for r in results] == ["refused", "missing"]


def test_replace_fields_never_replaces_a_framework_entry_in_the_registry(tmp_path: Path) -> None:
    path = _write(tmp_path, [_entry("joy", "my own joy")])
    vocabulary_file.replace_fields(path, {"joy": {"description": "changed in the file"}})
    assert _read(path)[0]["description"] == "changed in the file"
    assert vocabulary.get("joy").description == "happiness, delight, warmth"


def test_replace_fields_on_a_corrupt_file_is_cannot_write(tmp_path: Path) -> None:
    path = tmp_path / "emotion_vocabulary.json"
    path.write_bytes(b"\xff{torn")
    assert vocabulary_file.replace_fields(path, {"x one": {"description": "d"}})[0].outcome == "cannot_write"


# ---------------------------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------------------------


def test_merge_of_a_three_twin_file_succeeds_without_waiting_on_its_own_lock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dev_constants, "EMOTION_VOCABULARY_LOCK_WAIT_S", 0.5)
    path = _write(tmp_path, [_stub("Twin Three"), _stub("twin_three"), _entry("TWIN-THREE", "held")])
    result = vocabulary_file.merge(path)
    assert result.cannot_write is False
    assert result.pairs == [("Twin Three", "Twin Three"), ("twin_three", "Twin Three")]
    assert [(e["name"], e["description"]) for e in _read(path)] == [("Twin Three", "held")]


def test_merge_with_no_twin_writes_nothing(tmp_path: Path, writes) -> None:
    path = _write(tmp_path, [_entry("lone one"), _entry("_lone_blend")])
    before = path.read_bytes()
    result = vocabulary_file.merge(path)
    assert result.pairs == [] and not result.cannot_write
    assert path.read_bytes() == before and writes == []


def test_merge_names_limits_it_to_those_names(tmp_path: Path) -> None:
    path = _write(
        tmp_path, [_entry("ma one", "1"), _entry("ma_one", "2"), _entry("mb one", "3"), _entry("mb_one", "4")]
    )
    result = vocabulary_file.merge(path, ["MB-One"])
    assert result.pairs == [("mb_one", "mb one")]
    assert [e["name"] for e in _read(path)] == ["ma one", "ma_one", "mb one"]


def test_merge_makes_the_registered_entry_the_merged_entry(tmp_path: Path) -> None:
    path = _write(tmp_path, [_stub("reg twin"), _entry("reg_twin", "real one")])
    vocabulary_file.load(path)
    assert vocabulary.get("reg twin").description == PLACEHOLDER_DESCRIPTION
    vocabulary_file.merge(path)
    assert vocabulary.get("reg_twin").description == "real one" and vocabulary.get("reg_twin").name == "reg twin"


def test_merge_on_a_corrupt_file_is_cannot_write(tmp_path: Path) -> None:
    path = tmp_path / "emotion_vocabulary.json"
    path.write_bytes(b"\xff{torn")
    assert vocabulary_file.merge(path).cannot_write is True


# The twin merge rule (moved here from the vocabulary repair, #174, #801 section 3).


def _names(data: dict) -> list:
    return [e["name"] if isinstance(e, dict) and "name" in e else e for e in data["emotions"]]


def test_merge_keeps_the_first_in_file_spelling_with_the_later_real_description() -> None:
    data = {"emotions": [_stub("comfort seeking"), _entry("comfort_seeking", "held still")]}
    pairs = vocabulary_file.merge_twin_entries(data)
    assert pairs == [("comfort seeking", "comfort seeking")]  # the dropped placeholder, and the spelling that stays
    assert [e["name"] for e in data["emotions"]] == ["comfort seeking"]
    assert data["emotions"][0]["description"] == "held still"
    assert data["emotions"][0]["decay_half_life_days"] == 45.0


def test_merge_of_two_real_twins_keeps_the_first_entry_whole() -> None:
    first, second = _entry("comfort seeking", "one"), _entry("comfort_seeking", "two")
    data = {"emotions": [first, second]}
    assert vocabulary_file.merge_twin_entries(data) == [("comfort_seeking", "comfort seeking")]
    assert data["emotions"] == [_entry("comfort seeking", "one")]


def test_merge_of_three_twins_remembers_the_first_spelling_not_the_donors() -> None:
    data = {"emotions": [_stub("Comfort Seeking"), _stub("comfort_seeking"), _entry("COMFORT-SEEKING", "held")]}
    pairs = vocabulary_file.merge_twin_entries(data)
    assert [e["name"] for e in data["emotions"]] == ["Comfort Seeking"]
    assert data["emotions"][0]["description"] == "held"
    assert pairs == [("Comfort Seeking", "Comfort Seeking"), ("comfort_seeking", "Comfort Seeking")]


def test_merge_leaves_a_lone_entry_alone_and_reports_nothing() -> None:
    entries = [_entry("comfort seeking"), _entry("_grief_love_blend"), _entry("Quiet Dread")]
    data = {"emotions": [dict(e) for e in entries]}
    assert vocabulary_file.merge_twin_entries(data) == []
    assert data["emotions"] == entries


def test_merge_keeps_non_dict_nameless_and_empty_tidied_entries_in_place() -> None:
    nameless = {"description": "no name"}
    data = {
        "emotions": [
            "not an entry",
            _entry("comfort seeking", "one"),
            nameless,
            _entry("---"),
            7,
            _entry("comfort_seeking", "two"),
            _entry("___"),
        ]
    }
    assert vocabulary_file.merge_twin_entries(data) == [("comfort_seeking", "comfort seeking")]
    assert _names(data) == ["not an entry", "comfort seeking", nameless, "---", 7, "___"]
    assert data["emotions"][1] == _entry("comfort seeking", "one")
    assert data["emotions"][2] is nameless


# ---------------------------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------------------------


def test_remove_takes_the_entry_out_of_the_file_and_the_registry_with_a_verified_write(tmp_path: Path, writes) -> None:
    path = _write(tmp_path, [_entry("Doomed One"), _entry("kept one")])
    vocabulary_file.load(path)
    assert vocabulary.get("doomed one") is not None
    res = vocabulary_file.remove(path, "doomed_one")
    assert res == NameResult("doomed_one", "doomed_one", "used", True)
    assert [e["name"] for e in _read(path)] == ["kept one"]
    assert vocabulary.get("doomed one") is None and vocabulary.get("kept one") is not None
    assert writes[0]["verify"] is True


def test_remove_of_a_framework_name_leaves_it_registered(tmp_path: Path) -> None:
    path = _write(tmp_path, [_entry("joy", "my joy"), _entry("other keep")])
    res = vocabulary_file.remove(path, "Joy")
    assert res.changed and [e["name"] for e in _read(path)] == ["other keep"]
    assert vocabulary.get("joy") is not None


def test_remove_of_an_absent_name_writes_nothing(tmp_path: Path, writes) -> None:
    path = _write(tmp_path, [_entry("only one")])
    assert vocabulary_file.remove(path, "absent") == NameResult("absent", None, "missing")
    assert writes == []


def test_remove_refuses_a_name_that_tidies_to_nothing(tmp_path: Path, writes) -> None:
    path = _write(tmp_path, [{"description": "nameless"}])
    assert vocabulary_file.remove(path, "---").outcome == "refused"
    assert writes == []


# ---------------------------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------------------------


def test_load_registers_the_entries_under_the_file_spelling_and_writes_nothing(tmp_path: Path, writes) -> None:
    path = _write(tmp_path, [_entry("Load Spelled")])
    count, anomaly = vocabulary_file.load(path)
    assert (count, anomaly) == (1, None)
    assert vocabulary.get("load_spelled").name == "Load Spelled"
    assert vocabulary_file.load(path)[0] == 0  # idempotent
    assert writes == []


def test_load_skips_an_entry_the_gate_refuses_and_registers_nothing_under_the_empty_key(tmp_path: Path) -> None:
    path = _write(tmp_path, [_entry("---"), _entry("z" * 101), _entry("fine entry")])
    count, _ = vocabulary_file.load(path)
    assert count == 1
    assert "" not in vocabulary._REGISTRY and vocabulary.get("---") is None
    assert len(_read(path)) == 3  # a refused entry stays in her file


def test_load_with_raise_raises_for_a_corrupt_file_and_with_heal_recovers(tmp_path: Path) -> None:
    path = tmp_path / "emotion_vocabulary.json"
    path.write_bytes(b"\xff{torn")
    with pytest.raises(JsonFileError):
        vocabulary_file.load(path, recover=Recovery.RAISE)
    count, anomaly = vocabulary_file.load(path)
    assert count == 0 and anomaly is not None


def test_the_schema_check_rejects_a_file_without_an_emotions_list() -> None:
    with pytest.raises(ValueError):
        vocabulary_file.validate_schema({"emotions": "nope"})
    with pytest.raises(ValueError):
        vocabulary_file.validate_schema([])
    vocabulary_file.validate_schema({"emotions": []})
