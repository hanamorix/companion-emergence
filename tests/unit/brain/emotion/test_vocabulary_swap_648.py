"""#648: the one-time swap of a persona's own entries for `contentment`, `calm`, `peace` and `groundedness`.

A persona that already holds an entry for one of the four names (a placeholder healed from memories, a repaired description, or its
own words with its own half-life) gets that entry rewritten to the baseline definition once, so every persona starts from the owner's
fixed values. The run is recorded in `vocabulary_swap_648.json` in the persona directory. Personas here are tmp directories with
synthetic vocabulary files; nothing touches a real persona.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from brain.emotion import vocabulary, vocabulary_swap
from brain.emotion.vocabulary_swap import run_vocabulary_swap_648
from brain.growth.crystallizers import reflex
from brain.health import json_io

VOCAB = "emotion_vocabulary.json"
MARKER = "vocabulary_swap_648.json"
FOUR = ("contentment", "calm", "peace", "groundedness")
PLACEHOLDER = "(reconstructed from memory)"


def _entry(name: str, description: str, days: float | None, category: str = "persona_extension", **extra: object) -> dict:
    return {"name": name, "description": description, "category": category, "decay_half_life_days": days, "intensity_clamp": 10.0, **extra}


def _write_vocab(persona: Path, entries: list) -> bytes:
    persona.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"version": 1, "emotions": entries}, indent=2) + "\n"
    (persona / VOCAB).write_text(text, encoding="utf-8")
    return text.encode("utf-8")


def _entries(persona: Path) -> list:
    return json.loads((persona / VOCAB).read_text(encoding="utf-8"))["emotions"]


def _by_name(persona: Path) -> dict[str, dict]:
    return {e["name"]: e for e in _entries(persona)}


def _definition(name: str) -> dict:
    emotion = vocabulary.get(name)
    assert emotion is not None
    return {"category": emotion.category, "description": emotion.description, "decay_half_life_days": emotion.decay_half_life_days}


def _definition_of(entry: dict) -> dict:
    return {k: entry[k] for k in ("category", "description", "decay_half_life_days")}


@pytest.fixture
def persona(tmp_path: Path) -> Path:
    p = tmp_path / "persona"
    p.mkdir()
    return p


def _mixed_entries() -> list:
    """A placeholder, a repaired description, and two persona-own entries (Phoebe's shapes: 6 and 14 days), among others."""
    return [
        _entry("love", "deep caring", None, "core"),
        _entry("contentment", "her own sense of a long quiet evening", 6.0, "core", origin="her own"),
        _entry("calm", "a still sea, written by the repair", 14.0),
        _entry("peace", PLACEHOLDER, 14.0),
        _entry("joy", "her own joy", 9.0, "core"),
        _entry("groundedness", "feet down, roots down", 14.0, "complex"),
        _entry("zz_custom", "something of hers", 3.0, extra_note="keep me"),
    ]


# --- what the swap does ---------------------------------------------------------------------------------------------


def test_the_four_entries_are_swapped_to_the_baseline_definition_and_everything_else_is_kept(persona):
    before = _mixed_entries()
    _write_vocab(persona, before)
    run_vocabulary_swap_648(persona)
    after = _entries(persona)
    assert [e["name"] for e in after] == [e["name"] for e in before]  # same entries, same order, same spelling
    for old, new in zip(before, after, strict=True):
        if old["name"] in FOUR:
            assert _definition_of(new) == _definition(old["name"])
            assert new["intensity_clamp"] == old["intensity_clamp"]
            assert {k: v for k, v in new.items() if k not in _definition(old["name"])} == {
                k: v for k, v in old.items() if k not in _definition(old["name"])
            }  # every other key of the entry is kept
        else:
            assert new == old


def test_the_two_persona_own_entries_that_the_owner_named_are_replaced(persona):
    """Phoebe's shape: contentment at 6 days and groundedness at 14 days become 1 day and 2 days."""
    _write_vocab(persona, _mixed_entries())
    run_vocabulary_swap_648(persona)
    by_name = _by_name(persona)
    assert by_name["contentment"]["decay_half_life_days"] == 1.0
    assert by_name["groundedness"]["decay_half_life_days"] == 2.0
    assert by_name["peace"]["description"] == vocabulary.get("peace").description  # a placeholder is replaced too
    assert by_name["calm"]["description"] == vocabulary.get("calm").description


def test_other_words_keep_their_own_half_lives_baseline_names_included(persona):
    _write_vocab(persona, _mixed_entries())
    run_vocabulary_swap_648(persona)
    by_name = _by_name(persona)
    assert by_name["joy"]["decay_half_life_days"] == 9.0 and by_name["joy"]["description"] == "her own joy"
    assert by_name["zz_custom"]["decay_half_life_days"] == 3.0
    assert by_name["love"]["decay_half_life_days"] is None


def test_the_run_is_recorded_in_a_marker_and_a_second_run_changes_nothing(persona):
    _write_vocab(persona, _mixed_entries())
    run_vocabulary_swap_648(persona)
    marker = json.loads((persona / MARKER).read_text(encoding="utf-8"))
    assert marker["swapped"] == 4
    # the persona later gives contentment a half-life of its own: afterwards its own words win, as today
    entries = _entries(persona)
    next(e for e in entries if e["name"] == "contentment")["decay_half_life_days"] = 7.0
    after_edit = _write_vocab(persona, entries)
    run_vocabulary_swap_648(persona)
    assert (persona / VOCAB).read_bytes() == after_edit
    assert _by_name(persona)["contentment"]["decay_half_life_days"] == 7.0


def test_a_persona_with_none_of_the_four_still_gets_its_marker_and_a_later_calm_is_never_swapped(persona):
    before = [_entry("love", "deep caring", None, "core"), _entry("zz_custom", "mine", 3.0)]
    original = _write_vocab(persona, before)
    run_vocabulary_swap_648(persona)
    assert (persona / MARKER).is_file()
    assert (persona / VOCAB).read_bytes() == original  # nothing to swap, nothing written
    entries = _entries(persona) + [_entry("calm", "the persona's own calm", 30.0)]
    coined = _write_vocab(persona, entries)
    run_vocabulary_swap_648(persona)
    assert (persona / VOCAB).read_bytes() == coined
    assert _by_name(persona)["calm"]["decay_half_life_days"] == 30.0


def test_a_missing_vocabulary_file_is_a_completed_run_and_no_file_is_created(persona):
    run_vocabulary_swap_648(persona)
    assert (persona / MARKER).is_file()
    assert not (persona / VOCAB).exists()


def test_an_entry_already_on_the_baseline_definition_is_not_rewritten(persona):
    entries = [_entry(n, vocabulary.get(n).description, vocabulary.get(n).decay_half_life_days, vocabulary.get(n).category) for n in FOUR]
    original = _write_vocab(persona, entries)
    run_vocabulary_swap_648(persona)
    assert (persona / VOCAB).read_bytes() == original
    assert (persona / MARKER).is_file()


def test_a_variant_spelling_is_swapped_and_keeps_its_spelling_and_a_non_dict_entry_is_kept(persona):
    _write_vocab(persona, [_entry("Peace", "mine", 30.0), "not an entry", _entry("CALM", "mine too", 30.0), 7, {"description": "no name"}])
    run_vocabulary_swap_648(persona)
    entries = _entries(persona)
    assert [e["name"] if isinstance(e, dict) and "name" in e else e for e in entries[:4]] == ["Peace", "not an entry", "CALM", 7]
    assert _definition_of(entries[0]) == _definition("peace")
    assert _definition_of(entries[2]) == _definition("calm")
    assert entries[4] == {"description": "no name"}


def test_twin_entries_of_one_name_are_both_rewritten_and_left_to_the_repair_to_merge(persona):
    _write_vocab(persona, [_entry("peace", "one", 5.0), _entry("peace", "two", 6.0)])
    run_vocabulary_swap_648(persona)
    assert [_definition_of(e) for e in _entries(persona)] == [_definition("peace")] * 2


def test_the_reflex_crystallizer_still_accepts_triggers_on_the_four_names(persona):
    """The trigger check reads the name set of the vocabulary file; the swap rewrites entries and keeps every name."""
    _write_vocab(persona, _mixed_entries())
    names_before = reflex._load_emotion_vocabulary(persona)
    run_vocabulary_swap_648(persona)
    names_after = reflex._load_emotion_vocabulary(persona)
    assert names_after == names_before and set(FOUR) <= names_after
    proposal = {"name": "quiet_arc", "description": "d", "trigger": {"calm": 7.0, "peace": 7.0}, "prompt_template": "x"}
    accepted, reason = reflex._validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names=set(),
        removed_arc_names=set(),
        emotion_vocabulary=names_after,
        existing_trigger_keysets=[],
        active_arc_count=0,
        total_cap=10,
    )
    assert "gate 4" not in reason  # the four names are known; any later gate is about something else


# --- the write path: verified, with backups ----------------------------------------------------------------------------


def test_the_replacement_is_written_with_verify_and_a_backup_chain_and_the_old_file_is_kept_as_bak1(persona, monkeypatch):
    original = _write_vocab(persona, _mixed_entries())
    seen: dict = {}
    real = json_io.write_json

    def recording(path, data, **kwargs):
        if Path(path).name == VOCAB:
            seen.update(kwargs)
        return real(path, data, **kwargs)

    monkeypatch.setattr(vocabulary_swap, "write_json", recording)
    run_vocabulary_swap_648(persona)
    assert seen["verify"] is True and seen["backups"] == 3
    assert (persona / (VOCAB + ".bak1")).read_bytes() == original


def test_a_replacement_that_fails_its_read_back_counts_as_a_failed_write_and_no_marker_is_written(persona, monkeypatch, caplog):
    _write_vocab(persona, _mixed_entries())
    monkeypatch.setattr(json_io, "_verify", lambda *a, **k: False)  # the read-back says the new file is not what was written
    with caplog.at_level(logging.WARNING, logger="brain.emotion.vocabulary_swap"):
        run_vocabulary_swap_648(persona)
    assert not (persona / MARKER).exists()
    assert any("vocabulary" in r.getMessage() for r in caplog.records)


def test_a_write_that_raises_leaves_the_file_as_it_was_writes_no_marker_and_warns(persona, monkeypatch, caplog):
    original = _write_vocab(persona, _mixed_entries())
    real = json_io.write_json

    def failing(path, data, **kwargs):
        if Path(path).name == VOCAB:
            raise OSError("disk full")
        return real(path, data, **kwargs)

    monkeypatch.setattr(vocabulary_swap, "write_json", failing)
    with caplog.at_level(logging.WARNING, logger="brain.emotion.vocabulary_swap"):
        run_vocabulary_swap_648(persona)  # does not raise
    assert (persona / VOCAB).read_bytes() == original
    assert not (persona / MARKER).exists()
    assert any("disk full" in r.getMessage() for r in caplog.records)


def test_a_marker_that_cannot_be_written_warns_and_the_swap_stands(persona, monkeypatch, caplog):
    _write_vocab(persona, _mixed_entries())
    real = json_io.write_json

    def failing(path, data, **kwargs):
        if Path(path).name == MARKER:
            raise OSError("read-only folder")
        return real(path, data, **kwargs)

    monkeypatch.setattr(vocabulary_swap, "write_json", failing)
    with caplog.at_level(logging.WARNING, logger="brain.emotion.vocabulary_swap"):
        run_vocabulary_swap_648(persona)
    assert _by_name(persona)["contentment"]["decay_half_life_days"] == 1.0
    assert not (persona / MARKER).exists()
    assert any("read-only folder" in r.getMessage() for r in caplog.records)


def test_another_writer_changing_the_file_between_the_read_and_the_write_aborts_with_no_marker(persona, monkeypatch, caplog):
    _write_vocab(persona, _mixed_entries())
    real_read = vocabulary_swap.read_json_result
    theirs: list[bytes] = []

    def read_then_someone_else_writes(path, **kwargs):
        result = real_read(path, **kwargs)
        if Path(path).name == VOCAB:
            theirs.append(_write_vocab(persona, [*_mixed_entries(), _entry("zz_new", "added by the growth scheduler", 5.0)]))
        return result

    monkeypatch.setattr(vocabulary_swap, "read_json_result", read_then_someone_else_writes)
    with caplog.at_level(logging.WARNING, logger="brain.emotion.vocabulary_swap"):
        run_vocabulary_swap_648(persona)
    assert (persona / VOCAB).read_bytes() == theirs[0]  # their write is not overwritten by our older read
    assert not (persona / MARKER).exists()  # so the next start tries again
    assert any("changed" in r.getMessage() for r in caplog.records)


# --- the read path: a file that cannot be trusted is never written over -----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["{ not json", "", '{"version": 1, "emotions": "nope"}', '{"version": 1}', "[1, 2]"],
    ids=["malformed", "empty", "emotions-not-a-list", "no-emotions", "not-an-object"],
)
def test_a_vocabulary_file_that_cannot_be_read_is_left_alone_with_no_marker_and_a_warning(persona, text, caplog):
    (persona / VOCAB).write_text(text, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="brain.emotion.vocabulary_swap"):
        run_vocabulary_swap_648(persona)  # does not raise
    assert (persona / VOCAB).read_text(encoding="utf-8") == text
    assert not (persona / MARKER).exists()
    assert not (persona / (VOCAB + ".bak1")).exists()
    assert any("vocabulary" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("text", ["{ not json", "", "null"], ids=["malformed", "empty", "null"])
def test_a_marker_that_cannot_be_read_stops_the_swap_and_the_vocabulary_file_is_left_alone(persona, text, caplog):
    original = _write_vocab(persona, _mixed_entries())
    (persona / MARKER).write_text(text, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="brain.emotion.vocabulary_swap"):
        run_vocabulary_swap_648(persona)
    assert (persona / VOCAB).read_bytes() == original
    assert (persona / MARKER).read_text(encoding="utf-8") == text
    assert any("marker" in r.getMessage() for r in caplog.records)


def test_a_bug_inside_the_swap_is_logged_and_does_not_stop_start_up(persona, monkeypatch, caplog):
    original = _write_vocab(persona, _mixed_entries())

    def boom(*_a, **_k):
        raise RuntimeError("a bug")

    monkeypatch.setattr(vocabulary_swap, "read_json_result", boom)
    with caplog.at_level(logging.WARNING, logger="brain.emotion.vocabulary_swap"):
        run_vocabulary_swap_648(persona)  # does not raise
    assert (persona / VOCAB).read_bytes() == original
    assert not (persona / MARKER).exists()
    assert any("a bug" in r.getMessage() or "RuntimeError" in r.getMessage() for r in caplog.records)


def test_a_persona_folder_that_does_not_exist_gets_no_folder_and_no_marker(tmp_path):
    ghost = tmp_path / "no-such-persona"
    run_vocabulary_swap_648(ghost)
    assert not ghost.exists()
