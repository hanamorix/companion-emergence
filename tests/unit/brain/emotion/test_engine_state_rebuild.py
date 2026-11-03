"""The emotion engine's `half_life` lookup (Engine 0, #575).

`half_life(emotion)` answers, in order: the persona's own vocabulary file
(`<persona_dir>/emotion_vocabulary.json`, an entry with a numeric half-life), the
built-in table `dev_constants.EMOTION_HALF_LIFE_DAYS`, then
`dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS`. Names are canonical on both sides
(`vocabulary.canonical_name`). The file is read through `read_json_result` under
`Recovery.DEFAULT` (bad content takes the default, nothing is quarantined, the file
is never written) and re-read only when its stat key changes.

Later engine slices extend this file.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine_files, vocabulary
from brain.emotion.engine import EmotionEngine
from brain.health.json_io import JsonRead

VOCAB = "emotion_vocabulary.json"
FILES_LOGGER = "brain.emotion.engine_files"


def _entry(name: object, days: object = 5.0, **extra: object) -> dict:
    return {
        "name": name,
        "description": "d",
        "category": "core",
        "decay_half_life_days": days,
        **extra,
    }


def _write_vocab(persona_dir: Path, entries: list) -> Path:
    path = persona_dir / VOCAB
    path.write_text(json.dumps({"version": 1, "emotions": entries}), encoding="utf-8")
    return path


def _engine(persona_dir: Path) -> EmotionEngine:
    return EmotionEngine(persona_dir)


def _file_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == FILES_LOGGER and r.levelno >= logging.WARNING]


@pytest.fixture
def read_log(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every path `engine_files` reads, in order (the real read still happens)."""
    calls: list[Path] = []
    real = engine_files.read_json_result

    def spy(path: Path, **kwargs: object) -> JsonRead:
        calls.append(path)
        return real(path, **kwargs)

    monkeypatch.setattr(engine_files, "read_json_result", spy)
    return calls


# --- order: persona file, table, default ------------------------------------------------------


def test_persona_entry_wins_over_the_table(tmp_path: Path) -> None:
    # joy is in the table (3.0) and in the file (9.0): a reversed order would answer 3.0.
    assert dev_constants.EMOTION_HALF_LIFE_DAYS["joy"] == 3.0
    _write_vocab(tmp_path, [_entry("joy", 9.0)])
    assert _engine(tmp_path).half_life("joy") == 9.0


def test_table_wins_over_the_default_and_unknown_gets_the_default(tmp_path: Path) -> None:
    _write_vocab(tmp_path, [_entry("something_else", 4.0)])
    engine = _engine(tmp_path)
    assert engine.half_life("grief") == 60.0
    assert engine.half_life("not_in_table_or_file") == 14.0
    assert engine.half_life("not_in_table_or_file") == dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS


def test_a_persona_only_name_returns_the_persona_value(tmp_path: Path) -> None:
    _write_vocab(tmp_path, [_entry("persona_only_word", 4.5)])
    assert _engine(tmp_path).half_life("persona_only_word") == 4.5


def test_constants_are_read_at_call_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)  # built first: a constructor-time capture would not see the patch
    monkeypatch.setattr(dev_constants, "EMOTION_HALF_LIFE_DAYS", {"joy": 1.25})
    monkeypatch.setattr(dev_constants, "EMOTION_DEFAULT_HALF_LIFE_DAYS", 3.5)
    assert engine.half_life("joy") == 1.25
    assert engine.half_life("grief") == 3.5


# --- table values -----------------------------------------------------------------------------


def test_love_and_belonging_come_from_the_table_with_no_persona_file(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    assert engine.half_life("love") == 730.0
    assert engine.half_life("belonging") == 365.0
    assert dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS == 14.0


def test_every_table_key_is_a_canonical_name() -> None:
    for key in dev_constants.EMOTION_HALF_LIFE_DAYS:
        assert key == vocabulary.canonical_name(key)


# --- null means "never fades": the lookup goes on to the table, then 14 days ------------------


def test_a_null_persona_entry_falls_to_the_table_then_the_default(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_vocab(
        tmp_path, [_entry("love", None), _entry("belonging", None), _entry("neither_word", None)]
    )
    engine = _engine(tmp_path)
    with caplog.at_level(logging.DEBUG):
        assert engine.half_life("love") == 730.0
        assert engine.half_life("belonging") == 365.0
        assert engine.half_life("neither_word") == 14.0
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


# --- names are canonical on both sides --------------------------------------------------------


def test_file_entry_and_argument_match_across_spellings(tmp_path: Path) -> None:
    # The table's comfort_seeking is 1.0; the persona value differs, so only the file can answer 6.0.
    assert dev_constants.EMOTION_HALF_LIFE_DAYS["comfort_seeking"] == 1.0
    _write_vocab(tmp_path, [_entry("comfort-seeking", 6.0)])
    assert _engine(tmp_path).half_life("comfort_seeking") == 6.0

    other = tmp_path / "other"
    other.mkdir()
    _write_vocab(other, [_entry("comfort_seeking", 6.0)])
    assert _engine(other).half_life("Comfort-Seeking") == 6.0


def test_table_and_default_answer_under_a_variant_spelling(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    assert engine.half_life(" Joy ") == 3.0
    assert engine.half_life("Touch-Hunger") == 1.5
    assert engine.half_life("Totally-Unknown Word") == 14.0


# --- duplicates and odd names -----------------------------------------------------------------


def test_first_valid_entry_wins_and_a_later_duplicate_is_silent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_vocab(
        tmp_path,
        [_entry("joy", 9.0), _entry("Joy", 2.0), _entry("love", None), _entry("love", 5.0)],
    )
    engine = _engine(tmp_path)
    with caplog.at_level(logging.DEBUG):
        assert engine.half_life("joy") == 9.0
        assert (
            engine.half_life("love") == 730.0
        )  # the null decided it; the later numeric entry does not override
    assert _file_warnings(caplog) == []


def test_a_faulted_first_entry_does_not_count_as_the_first(tmp_path: Path) -> None:
    _write_vocab(tmp_path, [_entry("joy", "not a number"), _entry("joy", 9.0)])
    assert _engine(tmp_path).half_life("joy") == 9.0


def test_a_present_non_text_name_is_coerced_not_a_fault(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_vocab(tmp_path, [_entry(7, 8.0)])
    engine = _engine(tmp_path)
    with caplog.at_level(logging.DEBUG):
        assert engine.half_life("7") == 8.0
    assert _file_warnings(caplog) == []


# --- the cache follows the file ---------------------------------------------------------------


def test_the_file_is_re_read_when_its_size_changes(tmp_path: Path) -> None:
    path = _write_vocab(tmp_path, [_entry("joy", 9.0)])
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 9.0
    _write_vocab(tmp_path, [_entry("joy", 10.25)])
    assert path.stat().st_size > 0
    assert engine.half_life("joy") == 10.25


def test_a_same_size_replacement_with_a_new_inode_is_seen(tmp_path: Path) -> None:
    path = _write_vocab(tmp_path, [_entry("joy", 5.0)])
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 5.0
    size = path.stat().st_size
    replacement = tmp_path / "replacement.tmp"
    _write_vocab_to(replacement, [_entry("joy", 6.0)])
    assert replacement.stat().st_size == size
    os.replace(replacement, path)
    assert engine.half_life("joy") == 6.0


def _write_vocab_to(path: Path, entries: list) -> None:
    path.write_text(json.dumps({"version": 1, "emotions": entries}), encoding="utf-8")


_T1 = 1_700_000_000_000_000_000
_T2 = _T1 + 5_000_000_000


def test_a_change_of_modification_time_alone_is_seen(tmp_path: Path) -> None:
    path = _write_vocab(tmp_path, [_entry("joy", 5.0)])
    os.utime(path, ns=(_T1, _T1))
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 5.0
    _write_vocab(tmp_path, [_entry("joy", 6.0)])  # same size, same inode, in place
    os.utime(path, ns=(_T2, _T2))
    assert engine.half_life("joy") == 6.0


def test_a_change_of_size_alone_is_seen(tmp_path: Path) -> None:
    path = _write_vocab(tmp_path, [_entry("joy", 5.0)])
    os.utime(path, ns=(_T1, _T1))
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 5.0
    _write_vocab(tmp_path, [_entry("joy", 5.25)])  # one byte longer, same inode
    os.utime(path, ns=(_T1, _T1))  # modification time put back
    assert engine.half_life("joy") == 5.25


def test_a_change_of_inode_alone_is_seen(tmp_path: Path) -> None:
    path = _write_vocab(tmp_path, [_entry("joy", 5.0)])
    os.utime(path, ns=(_T1, _T1))
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 5.0
    replacement = tmp_path / "replacement.tmp"
    _write_vocab_to(replacement, [_entry("joy", 6.0)])
    os.utime(replacement, ns=(_T1, _T1))
    os.replace(replacement, path)
    assert engine.half_life("joy") == 6.0


def test_the_file_is_not_re_read_while_its_key_is_unchanged(
    tmp_path: Path, read_log: list[Path]
) -> None:
    _write_vocab(tmp_path, [_entry("joy", 9.0)])
    engine = _engine(tmp_path)
    for _ in range(3):
        assert engine.half_life("joy") == 9.0
        assert engine.half_life("grief") == 60.0
    assert len(read_log) == 1


def test_a_write_between_the_stat_and_the_read_is_seen_by_the_next_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_vocab(tmp_path, [_entry("joy", 5.0)])
    real = engine_files.read_json_result
    state = {"armed": True}

    def read_then_replace(p: Path, **kwargs: object) -> JsonRead:
        result = real(p, **kwargs)
        if state["armed"]:
            state["armed"] = False
            replacement = tmp_path / "replacement.tmp"
            _write_vocab_to(replacement, [_entry("joy", 7.125)])  # a different size and value
            os.replace(replacement, path)
        return result

    monkeypatch.setattr(engine_files, "read_json_result", read_then_replace)
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 5.0  # answered from the bytes it read
    assert (
        engine.half_life("joy") == 7.125
    )  # the stored key was the pre-read stat, so this re-reads


def test_a_bad_file_that_is_then_fixed_gives_the_persona_value(tmp_path: Path) -> None:
    path = tmp_path / VOCAB
    path.write_text("{not json", encoding="utf-8")
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 3.0
    _write_vocab(tmp_path, [_entry("joy", 9.0)])
    assert engine.half_life("joy") == 9.0


def test_a_file_that_is_then_deleted_drops_the_persona_value_quietly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_vocab(tmp_path, [_entry("joy", 9.0)])
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 9.0
    path.unlink()
    with caplog.at_level(logging.DEBUG):
        assert engine.half_life("joy") == 3.0
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_a_file_that_appears_after_the_first_call_is_seen(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 3.0  # no file yet
    _write_vocab(tmp_path, [_entry("joy", 9.0)])
    assert engine.half_life("joy") == 9.0


# --- a failure that says nothing about the content is not cached (#530) -----------------------


def _fail_next_read_with_io(monkeypatch: pytest.MonkeyPatch) -> None:
    real = engine_files.read_json_result
    state = {"armed": True}

    def flaky(path: Path, **kwargs: object) -> JsonRead:
        if state["armed"]:
            state["armed"] = False
            return JsonRead(data={}, outcome="defaulted", kind="io")
        return real(path, **kwargs)

    monkeypatch.setattr(engine_files, "read_json_result", flaky)


def test_an_io_failure_on_the_first_read_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_vocab(tmp_path, [_entry("joy", 9.0)])
    _fail_next_read_with_io(monkeypatch)
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 3.0  # the built-in value, for this call only
    assert engine.half_life("joy") == 9.0  # same stat key, read again


def test_an_io_failure_after_a_good_load_keeps_the_last_good_table_and_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_vocab(tmp_path, [_entry("joy", 5.0)])
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 5.0
    replacement = tmp_path / "replacement.tmp"
    _write_vocab_to(replacement, [_entry("joy", 7.5)])
    os.replace(replacement, path)
    _fail_next_read_with_io(monkeypatch)
    assert engine.half_life("joy") == 5.0  # last known good stays in force
    assert engine.half_life("joy") == 7.5  # the failure was not cached: the same key is read again


def _fail_next_stat(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    real = pathlib.Path.stat
    state = {"armed": True}

    def flaky(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if state["armed"] and self == target:
            state["armed"] = False
            raise PermissionError("denied (test)")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "stat", flaky)


def test_a_stat_error_before_the_first_load_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_vocab(tmp_path, [_entry("joy", 9.0)])
    _fail_next_stat(monkeypatch, path)
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 3.0
    assert engine.half_life("joy") == 9.0


def test_a_stat_error_after_a_good_load_keeps_the_table_and_reads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_log: list[Path]
) -> None:
    path = _write_vocab(tmp_path, [_entry("joy", 9.0)])
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 9.0
    _fail_next_stat(monkeypatch, path)
    assert engine.half_life("joy") == 9.0  # not read as "no file"
    assert engine.half_life("joy") == 9.0
    assert len(read_log) == 1  # the key never changed, so nothing was re-read


def test_a_content_fault_is_cached_for_an_unchanged_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, read_log: list[Path]
) -> None:
    (tmp_path / VOCAB).write_text("{not json", encoding="utf-8")
    engine = _engine(tmp_path)
    with caplog.at_level(logging.WARNING):
        assert engine.half_life("joy") == 3.0
        first = len(caplog.records)
        assert engine.half_life("joy") == 3.0
        assert len(caplog.records) == first
    assert first >= 1
    assert len(read_log) == 1


# --- a bad file is reported with the specific fault; the built-in half-life answers -----------


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("{not json", "malformed"),
        ('["a", "list"]', "shape"),
        ('{"version": 1, "emotions": "joy"}', "schema"),
    ],
)
def test_a_file_read_json_reports_names_the_path_and_the_built_in_answers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, text: str, kind: str
) -> None:
    path = tmp_path / VOCAB
    path.write_text(text, encoding="utf-8")
    before = path.read_bytes()
    engine = _engine(tmp_path)
    with caplog.at_level(logging.WARNING):
        assert engine.half_life("joy") == 3.0
        assert engine.half_life("unknown_word") == 14.0
    reports = [
        r for r in caplog.records if r.levelno >= logging.WARNING and str(path) in r.getMessage()
    ]
    assert len(reports) == 1
    assert kind in reports[0].getMessage()
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [VOCAB]


_BIG_INT = "1" + "0" * 399


@pytest.mark.parametrize(
    ("entry_json", "label", "field"),
    [
        ("5", "emotions[1]", "entry"),
        (
            '{"description": "d", "category": "core", "decay_half_life_days": 5.0}',
            "emotions[1]",
            "name",
        ),
        (
            '{"name": "joy", "description": "d", "category": "core"}',
            "'joy'",
            "decay_half_life_days",
        ),
        ('{"name": "joy", "decay_half_life_days": "7"}', "'joy'", "decay_half_life_days"),
        ('{"name": "joy", "decay_half_life_days": true}', "'joy'", "decay_half_life_days"),
        ('{"name": "joy", "decay_half_life_days": NaN}', "'joy'", "decay_half_life_days"),
        ('{"name": "joy", "decay_half_life_days": 1e999}', "'joy'", "decay_half_life_days"),
        (
            '{"name": "joy", "decay_half_life_days": ' + _BIG_INT + "}",
            "'joy'",
            "decay_half_life_days",
        ),
        ('{"name": "joy", "decay_half_life_days": 0}', "'joy'", "decay_half_life_days"),
        ('{"name": "joy", "decay_half_life_days": -3}', "'joy'", "decay_half_life_days"),
    ],
)
def test_an_entry_fault_is_one_warning_naming_file_entry_and_field(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, entry_json: str, label: str, field: str
) -> None:
    path = tmp_path / VOCAB
    good = json.dumps(_entry("grief", 9.0))
    path.write_text(
        '{"version": 1, "emotions": [' + good + ", " + entry_json + "]}", encoding="utf-8"
    )
    before = path.read_bytes()
    engine = _engine(tmp_path)
    with caplog.at_level(logging.WARNING):
        assert engine.half_life("grief") == 9.0  # the good entry beside the bad one still answers
        assert engine.half_life("joy") == 3.0  # the faulted emotion takes the built-in value
        assert engine.half_life("grief") == 9.0
    warnings = _file_warnings(caplog)
    assert len(warnings) == 1  # once per change of the file, not once per call
    message = warnings[0].getMessage()
    assert str(path) in message
    assert label in message
    assert field in message
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [VOCAB]


def test_half_life_never_raises_on_a_pile_of_faults(tmp_path: Path) -> None:
    path = tmp_path / VOCAB
    path.write_text(
        '{"emotions": [null, 3, "x", {}, {"name": []}, {"name": "joy", "decay_half_life_days": [1]},'
        ' {"name": "love", "decay_half_life_days": {"a": 1}}]}',
        encoding="utf-8",
    )
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 3.0
    assert engine.half_life("love") == 730.0


# --- no file ----------------------------------------------------------------------------------


def test_no_vocabulary_file_answers_from_the_table_and_default_without_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = _engine(tmp_path)
    with caplog.at_level(logging.DEBUG):
        assert engine.half_life("grief") == 60.0
        assert engine.half_life("never_heard_of_it") == 14.0
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
