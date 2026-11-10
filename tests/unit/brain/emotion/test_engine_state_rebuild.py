"""The emotion engine's `half_life` lookup (Engine 0, #575).

`half_life(emotion)` answers, in order: the persona's own vocabulary file
(`<persona_dir>/emotion_vocabulary.json`, an entry with a numeric half-life), the
built-in table `dev_constants.EMOTION_HALF_LIFE_DAYS`, then
`dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS`. Names are canonical on both sides
(`vocabulary.canonical_name`). The file is read through `read_json_result` under
`Recovery.DEFAULT` (bad content takes the default, nothing is quarantined, the file
is never written) and re-read only when its stat key changes.

The file also holds the cases of the engine's files (Engine 2, #577): the no-heal walk
`json_io.read_json_walk`, the state file's validator and JSON form, the stat-key cache, the switch
file and `is_live()`. The queue and the change log are in `test_engine_queue.py`. Later engine
slices extend this file.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import logging
import os
import pathlib
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain import dev_constants, paths
from brain.emotion import engine_files, engine_utils, vocabulary
from brain.emotion.engine import EmotionEngine
from brain.emotion.engine_utils import (
    Average,
    EmotionState,
    EngineState,
    RepeatEntry,
    RunRecord,
    SnapshotDay,
)
from brain.health import json_io
from brain.health.json_io import JsonRead, JsonWalk, read_json_walk
from brain.utils import file_lock, file_ops
from brain.utils.file_lock import LockTimeoutError

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


def _write_vocab_to(path: Path, entries: list) -> None:
    path.write_text(json.dumps({"version": 1, "emotions": entries}), encoding="utf-8")


def _write_vocab(persona_dir: Path, entries: list) -> Path:
    path = persona_dir / VOCAB
    _write_vocab_to(path, entries)
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
    _write_vocab(tmp_path, [_entry("joy", 9.0)])
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 9.0
    _write_vocab(tmp_path, [_entry("joy", 10.25)])
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


def _real_key(path: Path) -> tuple[int, int, int]:
    st = os.stat(path)  # not `Path.stat`: the test below records those calls
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def test_the_cache_key_is_the_one_file_ops_stat_path_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_log: list[Path]
) -> None:
    """The cache re-reads exactly when `file_ops.stat_path(<vocabulary file>).key` changes.

    The accessor is replaced by a fake that returns a settable `PathStat` on every call, so the
    real file's own stat does not decide anything.
    """
    path = _write_vocab(tmp_path, [_entry("joy", 5.0)])
    fake = {"stat": file_ops.PathStat(10, 20, 30, True, False)}
    asked: list[Path] = []

    def fake_stat_path(p: str | Path) -> file_ops.PathStat | None:
        asked.append(Path(p))
        return fake["stat"]

    path_stat_calls: list[Path] = []
    real_path_stat = pathlib.Path.stat

    def spy(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        path_stat_calls.append(self)
        return real_path_stat(self, *args, **kwargs)

    monkeypatch.setattr(file_ops, "stat_path", fake_stat_path)
    monkeypatch.setattr(pathlib.Path, "stat", spy)

    def rewrite(days: float) -> None:
        before = _real_key(path)
        _write_vocab(tmp_path, [_entry("joy", days)])
        assert _real_key(path) != before  # the real key moved (the size changed)

    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 5.0
    assert len(read_log) == 1

    # Same accessor key, different real file: no re-read, the old answer stands.
    rewrite(5.25)
    assert engine.half_life("joy") == 5.0
    assert len(read_log) == 1

    # Only the file/folder flags change: the key is `.key`, not the whole `PathStat`.
    fake["stat"] = fake["stat"]._replace(is_file=False, is_dir=True)
    assert engine.half_life("joy") == 5.0
    assert len(read_log) == 1

    # Each component of the key alone, differing from the stored key in that one field: a re-read.
    new_values = iter([6.125, 7.0625, 8.03125])
    for field in ("inode", "mtime_ns", "size"):
        stored = fake["stat"]
        fake["stat"] = stored._replace(**{field: getattr(stored, field) + 1})
        days = next(new_values)
        rewrite(days)
        reads_before = len(read_log)
        assert engine.half_life("joy") == days, field
        assert len(read_log) == reads_before + 1, field

    assert asked
    assert set(asked) == {path}  # the vocabulary file, not the persona folder
    assert [p for p in path_stat_calls if p == path] == []  # no raw `Path.stat` on it


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


def _fail_next_stat(monkeypatch: pytest.MonkeyPatch, target: Path) -> dict[str, bool]:
    """Make the next `file_ops.stat_path(target)` raise; the returned state shows whether it fired."""
    real = file_ops.stat_path
    state = {"armed": True}

    def flaky(path: str | Path) -> file_ops.PathStat | None:
        if state["armed"] and Path(path) == target:
            state["armed"] = False
            raise PermissionError("denied (test)")
        return real(path)

    monkeypatch.setattr(file_ops, "stat_path", flaky)
    return state


def test_a_stat_error_before_the_first_load_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_vocab(tmp_path, [_entry("joy", 9.0)])
    state = _fail_next_stat(monkeypatch, path)
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 3.0
    assert state["armed"] is False  # the injected error really was raised
    assert engine.half_life("joy") == 9.0


def test_a_stat_error_after_a_good_load_keeps_the_table_and_reads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_log: list[Path]
) -> None:
    path = _write_vocab(tmp_path, [_entry("joy", 9.0)])
    engine = _engine(tmp_path)
    assert engine.half_life("joy") == 9.0
    state = _fail_next_stat(monkeypatch, path)
    assert engine.half_life("joy") == 9.0  # not read as "no file"
    assert state["armed"] is False  # the injected error really was raised
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
    assert f"(field {field}:" in message
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
    tmp_path: Path, caplog: pytest.LogCaptureFixture, read_log: list[Path]
) -> None:
    engine = _engine(tmp_path)
    with caplog.at_level(logging.DEBUG):
        assert engine.half_life("grief") == 60.0
        assert engine.half_life("never_heard_of_it") == 14.0
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert read_log == []  # a missing file has no key and is never read


# ================================================================================================
# Engine 2 (#577): the no-heal walk, the state file, its cache, the switch and is_live()
# ================================================================================================

NOW = datetime(2026, 6, 10, 15, 30, tzinfo=UTC)
WALK_LOGGER = "brain.health.json_io"


def _snapshot_of(root: Path) -> dict[str, bytes | None]:
    """Every entry under `root` (recursively) with its bytes (None for a directory)."""
    out: dict[str, bytes | None] = {}
    for p in sorted(root.rglob("*")):
        out[str(p.relative_to(root))] = p.read_bytes() if p.is_file() else None
    return out


def _full_state() -> EngineState:
    return EngineState(
        saved_at=NOW,
        reset_at=NOW - timedelta(days=2),
        last_update=NOW - timedelta(hours=3),
        active_day="2026-06-10",
        persona_start_at=NOW - timedelta(days=40),
        repeats={"mem-1": RepeatEntry(NOW - timedelta(hours=5), 2)},
        emotions={
            "joy": EmotionState(
                short=Average(4.25, 1.5),
                long=Average(3.5, 12.75),
                last_entry_day="2026-06-10",
                snapshot=[
                    SnapshotDay("2026-06-08", 4.0, 3.5, True, RunRecord("2026-06-05", 1, "2026-06-08")),
                    SnapshotDay("2026-06-09", 4.5, 3.75, False, RunRecord("2026-06-05", 1, "2026-06-08")),
                    SnapshotDay("2026-06-10", 4.25, 3.5, True, None),
                ],
            ),
            "grief": EmotionState(
                short=Average(0.0, 0.5),
                long=Average(1.25, 3.0),
                last_entry_day=None,
                snapshot=[],
            ),
        },
    )


def _valid_json() -> dict:
    return engine_files.state_to_json(_full_state())


def _write_state_dict(persona_dir: Path, data: object) -> Path:
    path = engine_files.state_path(persona_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- paths: every one through paths.state_dir ----------------------------------------------------


def test_every_engine_path_is_built_through_paths_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moved = tmp_path / "moved"
    monkeypatch.setattr(paths, "state_dir", lambda persona_dir: moved)
    assert engine_files.state_path(tmp_path) == moved / dev_constants.EMOTION_STATE_FILENAME
    assert engine_files.queue_path(tmp_path) == moved / dev_constants.EMOTION_PENDING_FILENAME
    assert engine_files.switch_path(tmp_path) == moved / dev_constants.EMOTION_SWITCH_FILENAME
    assert engine_files.changes_dir(tmp_path) == moved / dev_constants.EMOTION_CHANGES_DIRNAME


def test_the_engine_paths_sit_under_the_persona_state_folder(tmp_path: Path) -> None:
    state = tmp_path / "state"
    assert engine_files.state_path(tmp_path) == state / "emotion_state.json"
    assert engine_files.queue_path(tmp_path) == state / "emotion_pending.jsonl"
    assert engine_files.switch_path(tmp_path) == state / "emotion_switch.json"
    assert engine_files.changes_dir(tmp_path) == state / "emotion_changes"


def test_file_names_are_read_from_dev_constants_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dev_constants, "EMOTION_STATE_FILENAME", "s.json")
    monkeypatch.setattr(dev_constants, "EMOTION_PENDING_FILENAME", "q.json")
    monkeypatch.setattr(dev_constants, "EMOTION_SWITCH_FILENAME", "w.json")
    monkeypatch.setattr(dev_constants, "EMOTION_CHANGES_DIRNAME", "c")
    assert engine_files.state_path(tmp_path).name == "s.json"
    assert engine_files.queue_path(tmp_path).name == "q.json"
    assert engine_files.switch_path(tmp_path).name == "w.json"
    assert engine_files.changes_dir(tmp_path).name == "c"


def test_the_lock_is_the_state_file_sidecar(tmp_path: Path) -> None:
    with engine_files.state_lock(tmp_path):
        pass
    assert (tmp_path / "state" / "emotion_state.json.lock").exists()


def _recording_locked_within(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, float]]:
    """Replace `file_lock.locked_within` with a recorder that holds no lock; the list holds each (path, wait) it was given."""
    seen: list[tuple[Path, float]] = []

    @contextlib.contextmanager
    def recorder(path: Path, seconds: float, **kwargs: object):
        seen.append((Path(path), seconds))
        yield

    monkeypatch.setattr(file_lock, "locked_within", recorder)
    return seen


def test_the_lock_is_taken_through_locked_within_with_the_state_path_and_the_dev_constant_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _recording_locked_within(monkeypatch)
    with engine_files.state_lock(tmp_path):
        pass
    assert seen == [(engine_files.state_path(tmp_path), dev_constants.EMOTION_LOCK_WAIT_S)]  # the state file, not the sidecar


def test_the_lock_wait_is_read_from_the_dev_constant_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _recording_locked_within(monkeypatch)
    monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 0.37)
    with engine_files.state_lock(tmp_path):
        pass
    assert [wait for _, wait in seen] == [0.37]


def test_a_wedged_holder_gives_a_lock_timeout_after_the_wait_and_the_body_never_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held_lock
) -> None:
    """A real second holder (a thread of this process holding the blocking lock). Red on an unbounded lock: the join times out."""
    monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 0.2)
    outcome: dict[str, object] = {}

    def attempt() -> None:
        try:
            with engine_files.state_lock(tmp_path):
                outcome["ran"] = True
        except BaseException as exc:  # reported below, not lost on the worker thread
            outcome["error"] = exc

    with held_lock(engine_files.state_path(tmp_path)):
        worker = threading.Thread(target=attempt, name="state-lock-attempt", daemon=True)
        worker.start()
        worker.join(5)
        gave_up_in_time = not worker.is_alive()
    worker.join(10)  # an unbounded wait ends once the holder lets go, so no thread outlives the test
    assert gave_up_in_time, "no bound on the wait: the attempt was still waiting 5 s after a 0.2 s wait was set"
    assert isinstance(outcome.get("error"), LockTimeoutError)
    assert "ran" not in outcome


def test_the_timeout_is_an_oserror_so_a_caller_must_catch_it_first() -> None:
    assert issubclass(LockTimeoutError, OSError)


def test_reading_a_persona_without_a_state_folder_creates_none(tmp_path: Path) -> None:
    assert engine_files.read_state(tmp_path).problem == "missing"
    assert engine_files.StateCache(tmp_path).get().problem == "missing"
    assert engine_files.switch_exists(tmp_path) is False
    assert engine_files.read_switch(tmp_path) is None
    assert engine_files.read_queue(tmp_path, from_line=0) == engine_files.QueueRead([], 0)
    assert engine_files.read_changes(tmp_path, NOW) == []
    assert engine_files.list_change_files(tmp_path) == []
    assert not EmotionEngine(tmp_path).is_live()
    assert list(tmp_path.iterdir()) == []


# --- read_json_walk ---------------------------------------------------------------------------


def _chain(tmp_path: Path, **contents: str | None) -> Path:
    """Write main/bak1/bak2/bak3 under tmp_path/f.json from keyword args (None: leave absent)."""
    base = tmp_path / "f.json"
    names = {"main": "f.json", "bak1": "f.json.bak1", "bak2": "f.json.bak2", "bak3": "f.json.bak3"}
    for key, text in contents.items():
        if text is not None:
            (tmp_path / names[key]).write_text(text, encoding="utf-8")
    return base


def test_walk_main_wins_over_valid_backups(tmp_path: Path) -> None:
    base = _chain(tmp_path, main='{"v": 0}', bak1='{"v": 1}', bak2='{"v": 2}', bak3='{"v": 3}')
    walk = read_json_walk(base, expect=dict)
    assert isinstance(walk, JsonWalk)
    assert (walk.data, walk.source, walk.problem) == ({"v": 0}, "main", None)


@pytest.mark.parametrize(
    ("contents", "source", "value"),
    [
        ({"main": "{bad", "bak1": '{"v": 1}', "bak2": '{"v": 2}', "bak3": '{"v": 3}'}, "bak1", 1),
        ({"main": None, "bak1": "[]", "bak2": '{"v": 2}', "bak3": '{"v": 3}'}, "bak2", 2),
        ({"main": "", "bak1": "{bad", "bak2": "7", "bak3": '{"v": 3}'}, "bak3", 3),
    ],
)
def test_walk_falls_through_the_backups_in_order(
    tmp_path: Path, contents: dict, source: str, value: int
) -> None:
    base = _chain(tmp_path, **contents)
    walk = read_json_walk(base, expect=dict)
    assert (walk.source, walk.problem, walk.data) == (source, None, {"v": value})


def test_walk_skips_a_copy_the_validator_rejects(tmp_path: Path) -> None:
    def validate(data: dict) -> None:
        if data.get("ok") is not True:
            raise ValueError("not ok")

    base = _chain(tmp_path, main='{"ok": false}', bak1='{"ok": true, "from": "bak1"}')
    walk = read_json_walk(base, expect=dict, validate=validate)
    assert (walk.source, walk.data["from"]) == ("bak1", "bak1")


def test_a_validator_bug_is_not_swallowed(tmp_path: Path) -> None:
    base = _chain(tmp_path, main='{"a": 1}')

    def broken(data: dict) -> None:
        raise KeyError("bug")

    with pytest.raises(KeyError):
        read_json_walk(base, expect=dict, validate=broken)


def test_walk_with_nothing_at_all_is_missing(tmp_path: Path) -> None:
    walk = read_json_walk(tmp_path / "f.json", expect=dict)
    assert (walk.data, walk.source, walk.problem, walk.signature) == (None, None, "missing", None)


def test_walk_with_files_but_none_valid_is_unreadable(tmp_path: Path) -> None:
    base = _chain(tmp_path, main="{bad", bak2="[1]")
    walk = read_json_walk(base, expect=dict)
    assert (walk.data, walk.source, walk.problem, walk.signature) == (None, None, "unreadable", None)


def test_a_directory_in_place_of_a_copy_is_a_skip_not_an_exception(tmp_path: Path) -> None:
    (tmp_path / "f.json").mkdir()
    (tmp_path / "f.json.bak1").write_text('{"v": 1}', encoding="utf-8")
    walk = read_json_walk(tmp_path / "f.json", expect=dict)
    assert (walk.source, walk.data) == ("bak1", {"v": 1})
    only_dir = tmp_path / "g"
    only_dir.mkdir()
    assert read_json_walk(tmp_path / "g", expect=dict).problem == "unreadable"


@pytest.mark.parametrize(
    "contents",
    [
        {"main": "{bad", "bak1": '{"v": 1}'},
        {"main": "{bad", "bak1": "{worse", "bak2": "[]"},
        {},
        {"main": '{"v": 0}', "bak1": '{"v": 1}'},
    ],
)
def test_walk_writes_nothing_quarantines_nothing_rotates_nothing(
    tmp_path: Path, contents: dict
) -> None:
    base = _chain(tmp_path, **contents)
    before = _snapshot_of(tmp_path)
    mtimes = {p: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    read_json_walk(base, expect=dict)
    assert _snapshot_of(tmp_path) == before
    assert {p: p.stat().st_mtime_ns for p in tmp_path.iterdir()} == mtimes


def test_walk_reports_each_skipped_copy_and_not_an_absent_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    base = _chain(tmp_path, main="{bad", bak2='{"v": 2}')  # bak1 absent
    with caplog.at_level(logging.DEBUG, logger=WALK_LOGGER):
        read_json_walk(base, expect=dict)
    text = " | ".join(r.getMessage() for r in caplog.records)
    assert str(tmp_path / "f.json") in text
    assert "malformed" in text
    assert "bak1" not in text


def test_walk_signature_is_that_of_the_bytes_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "f.json"
    path.write_text('{"v": 1}', encoding="utf-8")
    real = json_io.read_bytes_stat
    taken: list[tuple[int, int, int]] = []

    def read_then_replace(p: Path):
        raw, sig = real(p)
        taken.append(sig)
        p.write_text('{"v": 2222222}', encoding="utf-8")  # a newer file lands after the read
        return raw, sig

    monkeypatch.setattr(json_io, "read_bytes_stat", read_then_replace)
    walk = read_json_walk(path, expect=dict)
    monkeypatch.undo()
    assert walk.data == {"v": 1}
    assert walk.signature == taken[0]
    st = path.stat()
    assert walk.signature != (st.st_ino, st.st_mtime_ns, st.st_size)


def test_walk_takes_a_text_path_too(tmp_path: Path) -> None:
    (tmp_path / "f.json").write_text('{"v": 1}', encoding="utf-8")
    (tmp_path / "f.json.bak1").write_text('{"v": 2}', encoding="utf-8")
    (tmp_path / "f.json").write_text("{bad", encoding="utf-8")
    assert read_json_walk(str(tmp_path / "f.json"), expect=dict).source == "bak1"


def test_walk_signature_is_the_ino_mtime_size_triple(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text('{"v": 1}', encoding="utf-8")
    st = path.stat()
    assert read_json_walk(path, expect=dict).signature == (st.st_ino, st.st_mtime_ns, st.st_size)


# --- the state file: validator ---------------------------------------------------------------


def _set(path: tuple, value: object):
    def mutate(data: dict) -> None:
        node = data
        for step in path[:-1]:
            node = node[step]
        node[path[-1]] = value

    return mutate


def _drop(path: tuple):
    def mutate(data: dict) -> None:
        node = data
        for step in path[:-1]:
            node = node[step]
        del node[path[-1]]

    return mutate


_REJECTIONS = {
    "saved_at missing": _drop(("saved_at",)),
    "saved_at null": _set(("saved_at",), None),
    "saved_at not text": _set(("saved_at",), 12),
    "saved_at not a time": _set(("saved_at",), "yesterday"),
    "reset_at not a time": _set(("reset_at",), "x"),
    "reset_at a number": _set(("reset_at",), 5),
    "last_update not a time": _set(("last_update",), "x"),
    "persona_start_at present but unparseable": _set(("persona_start_at",), "not a time"),
    "persona_start_at a number": _set(("persona_start_at",), 7),
    "active_day not a date": _set(("active_day",), "banana"),
    "active_day compact form": _set(("active_day",), "20260610"),
    "active_day a number": _set(("active_day",), 20260610),
    "active_day unset while last_update is set": _set(("active_day",), None),
    "emotions missing": _drop(("emotions",)),
    "emotions a list": _set(("emotions",), []),
    "repeats a list": _set(("repeats",), []),
    "emotion not an object": _set(("emotions", "joy"), 3),
    "emotion missing short": _drop(("emotions", "joy", "short")),
    "emotion missing long": _drop(("emotions", "joy", "long")),
    "emotion missing last_entry_day": _drop(("emotions", "joy", "last_entry_day")),
    "short not an object": _set(("emotions", "joy", "short"), 3),
    "short avg missing": _drop(("emotions", "joy", "short", "avg")),
    "short weight missing": _drop(("emotions", "joy", "short", "weight")),
    "long weight missing": _drop(("emotions", "joy", "long", "weight")),
    "negative short weight": _set(("emotions", "joy", "short", "weight"), -0.25),
    "negative long weight": _set(("emotions", "joy", "long", "weight"), -1),
    "nan weight": _set(("emotions", "joy", "long", "weight"), float("nan")),
    "infinite average": _set(("emotions", "joy", "short", "avg"), float("inf")),
    "average a bool": _set(("emotions", "joy", "short", "avg"), True),
    "weight a string": _set(("emotions", "joy", "long", "weight"), "2"),
    "weight too large for a float": _set(("emotions", "joy", "long", "weight"), 10**400),
    "last_entry_day later than active_day": _set(("emotions", "joy", "last_entry_day"), "2026-06-11"),
    "last_entry_day not a date": _set(("emotions", "joy", "last_entry_day"), "2026-13-45"),
    "last_entry_day compact form": _set(("emotions", "joy", "last_entry_day"), "20260101"),
}


@pytest.mark.parametrize("label", sorted(_REJECTIONS))
def test_the_validator_rejects(label: str) -> None:
    engine_files.validate_state(_valid_json())  # the baseline itself is valid
    data = _valid_json()
    _REJECTIONS[label](data)
    with pytest.raises(ValueError):
        engine_files.validate_state(data)


@pytest.mark.parametrize("error", [OverflowError("year 1"), OSError("pre-1970 on Windows")])
@pytest.mark.parametrize("field", ["saved_at", "reset_at", "last_update", "persona_start_at"])
def test_a_time_that_cannot_be_turned_into_a_local_day_is_rejected(
    field: str, error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A time `day_of` cannot convert would fail every close for ever; it must never look valid."""
    data = _valid_json()
    data[field] = "0001-01-01T00:00:00Z"
    real = engine_utils.to_local

    def failing(dt: datetime) -> datetime:
        if dt.year == 1:
            raise error
        return real(dt)

    monkeypatch.setattr(engine_utils, "to_local", failing)
    with pytest.raises(ValueError):
        engine_files.validate_state(data)


def test_the_validator_rejects_a_state_that_is_not_an_object() -> None:
    for bad in ([], None, 3, "x"):
        with pytest.raises(ValueError):
            engine_files.validate_state(bad)


def test_the_validator_accepts_a_null_or_missing_persona_start_at() -> None:
    data = _valid_json()
    data["persona_start_at"] = None
    engine_files.validate_state(data)
    del data["persona_start_at"]
    engine_files.validate_state(data)
    assert engine_files.state_from_json(data).persona_start_at is None


def test_an_average_outside_zero_to_ten_is_held_and_the_state_kept() -> None:
    data = _valid_json()
    data["emotions"]["joy"]["short"]["avg"] = 11
    data["emotions"]["joy"]["long"]["avg"] = -0.5
    engine_files.validate_state(data)
    state = engine_files.state_from_json(data)
    assert state.emotions["joy"].short.avg == dev_constants.EMOTION_VALUE_CAP
    assert state.emotions["joy"].long.avg == 0.0
    assert state.emotions["joy"].short.weight == 1.5  # the weight is untouched


def test_a_missing_or_malformed_snapshot_list_is_an_empty_snapshot_with_a_line(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    for index, how in enumerate((_drop(("emotions", "joy", "snapshot")), _set(("emotions", "joy", "snapshot"), "x"))):
        data = _valid_json()
        how(data)
        engine_files.validate_state(data)
        with caplog.at_level(logging.ERROR, logger=FILES_LOGGER):
            state = engine_files.state_from_json(data, label=f"sparse {tmp_path} {index}")
        assert state.emotions["joy"].snapshot == []
        assert any("joy" in r.getMessage() for r in caplog.records)
        caplog.clear()


def test_active_day_unset_with_a_last_entry_day_is_accepted() -> None:
    data = _valid_json()
    data["active_day"] = None
    data["last_update"] = None
    engine_files.validate_state(data)


def test_a_fresh_empty_state_with_every_nullable_field_null_is_accepted(tmp_path: Path) -> None:
    fresh = EngineState(saved_at=NOW)
    engine_files.save_state(tmp_path, fresh)
    loaded = engine_files.read_state(tmp_path)
    assert loaded.problem is None and loaded.source == "main"
    assert loaded.state == fresh


def test_states_that_are_valid_but_sparse_are_accepted(tmp_path: Path) -> None:
    state = _full_state()
    state.emotions["joy"].snapshot = []
    state.emotions["grief"].short = Average(0.0, 0.0)  # weight 0
    state.emotions["grief"].long = Average(10.0, 0.0)
    state.repeats = {}
    engine_files.save_state(tmp_path, state)
    path = engine_files.state_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["some_future_key"] = {"nan_here": float("nan")}
    data["emotions"]["joy"]["extra"] = [1, 2]
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = engine_files.read_state(tmp_path)
    assert loaded.source == "main"
    assert loaded.state.emotions["grief"].long.avg == 10.0
    assert loaded.state.emotions["joy"].snapshot == []


# --- the state file: bad snapshot days and bad repeats ------------------------------------------


def _day(day: str, short=4.0, long=3.5, felt=1, run=None) -> list:
    return [day, short, long, felt, run]


_BAD_DAYS = {
    "non-finite short": _day("2026-06-09", short=float("nan")),
    "non-finite long": _day("2026-06-09", long=float("inf")),
    "short a string": _day("2026-06-09", short="4"),
    "felt 2": _day("2026-06-09", felt=2),
    "felt true": _day("2026-06-09", felt=True),
    "felt null": _day("2026-06-09", felt=None),
    "day not a date": _day("soon"),
    "day compact": _day("20260609"),
    "direction 0": _day("2026-06-09", run=["2026-06-05", 0, "2026-06-09"]),
    "direction 2": _day("2026-06-09", run=["2026-06-05", 2, "2026-06-09"]),
    "direction true": _day("2026-06-09", run=["2026-06-05", True, "2026-06-09"]),
    "start not a date": _day("2026-06-09", run=["x", 1, "2026-06-09"]),
    "last-with not a date": _day("2026-06-09", run=["2026-06-05", 1, "x"]),
    "start later than the day": _day("2026-06-09", run=["2026-06-10", 1, "2026-06-09"]),
    "last-with later than the day": _day("2026-06-09", run=["2026-06-05", 1, "2026-06-10"]),
    "last-with before start": _day("2026-06-09", run=["2026-06-09", 1, "2026-06-08"]),
    "last-with two days before the day": _day("2026-06-09", run=["2026-06-05", 1, "2026-06-07"]),
    "run of the wrong length": _day("2026-06-09", run=["2026-06-05", 1]),
    "run not a list": _day("2026-06-09", run="x"),
    "day of the wrong length": ["2026-06-09", 4.0, 3.5, 1],
    "day with an extra field and a bad felt": ["2026-06-09", 4.0, 3.5, 2, None, "later"],
    "run with an extra field and a bad direction": _day("2026-06-09", run=["2026-06-05", 0, "2026-06-09", "later"]),
    "day not a list": "2026-06-09",
}


@pytest.mark.parametrize("label", sorted(_BAD_DAYS))
def test_a_bad_snapshot_day_is_dropped_and_logged_and_the_state_stays_valid(
    label: str, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    data = _valid_json()
    good_before = _day("2026-06-08", 4.0, 3.5, 1, ["2026-06-05", 1, "2026-06-08"])
    good_after = _day("2026-06-10", 4.25, 3.5, 1, None)
    data["emotions"]["joy"]["snapshot"] = [good_before, _BAD_DAYS[label], good_after]
    engine_files.validate_state(data)
    with caplog.at_level(logging.ERROR, logger=FILES_LOGGER):
        state = engine_files.state_from_json(data, label=f"case {label} {tmp_path}")
    assert [d.day for d in state.emotions["joy"].snapshot] == ["2026-06-08", "2026-06-10"]
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(messages) == 1 and "joy" in messages[0]
    assert state.emotions["grief"].snapshot == []


def test_snapshot_days_out_of_order_or_repeated_are_dropped(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    data = _valid_json()
    data["emotions"]["joy"]["snapshot"] = [
        _day("2026-06-08"),
        _day("2026-06-08"),  # same label as the last valid day
        _day("2026-06-07"),  # earlier
        _day("2026-06-09"),
    ]
    with caplog.at_level(logging.ERROR, logger=FILES_LOGGER):
        state = engine_files.state_from_json(data, label=f"order {tmp_path}")
    assert [d.day for d in state.emotions["joy"].snapshot] == ["2026-06-08", "2026-06-09"]
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 2


def test_a_snapshot_average_outside_the_scale_is_held_and_the_day_kept(tmp_path: Path) -> None:
    data = _valid_json()
    data["emotions"]["joy"]["snapshot"] = [_day("2026-06-09", short=11, long=-2)]
    state = engine_files.state_from_json(data, label=f"hold {tmp_path}")
    day = state.emotions["joy"].snapshot[0]
    assert (day.day, day.short, day.long) == ("2026-06-09", 10.0, 0.0)


def test_a_snapshot_average_is_held_to_the_dev_constant_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dev_constants, "EMOTION_VALUE_CAP", 5.0)
    data = _valid_json()
    data["emotions"]["joy"]["short"]["avg"] = 7
    data["emotions"]["joy"]["snapshot"] = [_day("2026-06-09", short=8, long=6)]
    state = engine_files.state_from_json(data, label=f"cap {tmp_path}")
    assert state.emotions["joy"].short.avg == 5.0
    day = state.emotions["joy"].snapshot[0]
    assert (day.short, day.long) == (5.0, 5.0)


def test_a_snapshot_day_with_more_fields_than_today_keeps_the_known_ones(tmp_path: Path) -> None:
    """Additive-only: a newer brain's six-field day is read by its first five fields, not dropped after a rollback."""
    data = _valid_json()
    data["emotions"]["joy"]["snapshot"] = [
        ["2026-06-08", 4.0, 3.5, 1, ["2026-06-05", 1, "2026-06-08"], "a newer field"],
        ["2026-06-09", 4.5, 3.5, 0, None, {"x": 1}, 7],
    ]
    engine_files.validate_state(data)
    state = engine_files.state_from_json(data, label=f"six {tmp_path}")
    assert state.emotions["joy"].snapshot == [
        SnapshotDay("2026-06-08", 4.0, 3.5, True, RunRecord("2026-06-05", 1, "2026-06-08")),
        SnapshotDay("2026-06-09", 4.5, 3.5, False, None),
    ]


def test_a_run_record_with_more_fields_than_today_keeps_the_known_ones(tmp_path: Path) -> None:
    data = _valid_json()
    data["emotions"]["joy"]["snapshot"] = [_day("2026-06-09", run=["2026-06-05", -1, "2026-06-08", "a newer field"])]
    state = engine_files.state_from_json(data, label=f"four {tmp_path}")
    assert state.emotions["joy"].snapshot[0].run == RunRecord("2026-06-05", -1, "2026-06-08")


def test_a_state_still_holding_the_dropped_oldest_entry_field_is_valid_and_it_is_not_written_back(tmp_path: Path) -> None:
    """A state written by a development build carries `oldest_entry_at`; whatever it holds, the state stays usable and the key goes at the next save."""
    for value in ("not a time", [], 5, None, "2026-06-01T00:00:00Z"):
        data = _valid_json()
        data["oldest_entry_at"] = value
        engine_files.validate_state(data)
        state = engine_files.state_from_json(data, label=f"leftover {tmp_path}")
        assert "oldest_entry_at" not in engine_files.state_to_json(state)
        assert not hasattr(state, "oldest_entry_at")


def test_a_run_record_one_day_before_the_day_is_valid(tmp_path: Path) -> None:
    data = _valid_json()
    data["emotions"]["joy"]["snapshot"] = [_day("2026-06-09", run=["2026-06-05", -1, "2026-06-08"])]
    state = engine_files.state_from_json(data, label=f"pending {tmp_path}")
    assert state.emotions["joy"].snapshot[0].run == RunRecord("2026-06-05", -1, "2026-06-08")


@pytest.mark.parametrize(
    "bad",
    [
        {"first_at": "x", "count": 1},
        {"first_at": "2026-06-10T10:00:00Z", "count": -1},
        {"first_at": "2026-06-10T10:00:00Z", "count": 1.5},
        {"first_at": "2026-06-10T10:00:00Z", "count": True},
        {"count": 1},
        5,
    ],
)
def test_a_bad_repeat_record_is_dropped_with_a_line_and_the_state_stays_valid(
    bad: object, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    data = _valid_json()
    data["repeats"]["mem-bad"] = bad
    engine_files.validate_state(data)
    with caplog.at_level(logging.ERROR, logger=FILES_LOGGER):
        state = engine_files.state_from_json(data, label=f"repeat {tmp_path}")
    assert set(state.repeats) == {"mem-1"}
    assert any("mem-bad" in r.getMessage() for r in caplog.records)


# --- the mutation sweep: nothing but ValueError/TypeError, and no false rejection ---------------

_WRONG = [None, "x", [], {}, True, 10**400, -1, float("nan"), float("inf")]


def _paths_in(node: object, prefix: tuple = ()):
    yield prefix
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _paths_in(value, (*prefix, key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _paths_in(value, (*prefix, index))


def _replace_at(data: dict, path: tuple, value: object) -> dict:
    clone = copy.deepcopy(data)
    node = clone
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value
    return clone


def test_no_mutation_of_a_valid_state_makes_the_validator_or_the_conversion_raise_oddly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.CRITICAL)  # thousands of dropped days and records are logged, by design
    base = _valid_json()
    checked = 0
    for path in _paths_in(base):
        if not path:
            continue
        for wrong in _WRONG:
            data = _replace_at(base, path, wrong)
            try:
                engine_files.validate_state(data)
            except (ValueError, TypeError):
                checked += 1
                continue
            engine_files.state_from_json(data, label=f"sweep {tmp_path} {path} {wrong!r}")
            checked += 1
    assert checked > 400


def test_the_walk_never_raises_on_a_mutated_state_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.CRITICAL)
    base = _valid_json()
    path = engine_files.state_path(tmp_path)
    path.parent.mkdir(parents=True)
    for target in _paths_in(base):
        if len(target) < 2:
            continue
        for wrong in (None, "x", 10**400, [], True):
            text = json.dumps(_replace_at(base, target, wrong))
            path.write_text(text, encoding="utf-8")
            walk = read_json_walk(path, expect=dict, validate=engine_files.validate_state)
            assert walk.problem in (None, "unreadable")


# --- the state file: JSON form -----------------------------------------------------------------


def test_the_json_form_round_trips_a_full_state() -> None:
    state = _full_state()
    assert engine_files.state_from_json(engine_files.state_to_json(state)) == state


# --- the queue position kept in the state: `queue_line` (S14, S21) ----------------------------


def test_a_new_state_starts_at_queue_position_zero() -> None:
    assert EngineState().queue_line == 0


@pytest.mark.parametrize("line", [0, 1, 7, 10**400])
def test_the_queue_position_is_written_and_round_trips_as_given(line: int) -> None:
    state = dataclasses.replace(_full_state(), queue_line=line)
    data = engine_files.state_to_json(state)
    assert data["queue_line"] == line
    restored = engine_files.state_from_json(json.loads(json.dumps(data)))
    assert restored.queue_line == line and restored == state


def test_the_queue_position_survives_a_save_and_a_walk(tmp_path: Path) -> None:
    engine_files.save_state(tmp_path, dataclasses.replace(_full_state(), queue_line=5))
    loaded = engine_files.read_state(tmp_path)
    assert loaded.state is not None and loaded.state.queue_line == 5


_BAD_QUEUE_LINES = {
    "missing": None,
    "a string": "5",
    "a float": 5.0,
    "a bool": True,
    "a negative int": -3,
    "a list": [1],
    "a dict": {"a": 1},
    "nan": float("nan"),
    "null": None,
}


@pytest.mark.parametrize("case", sorted(_BAD_QUEUE_LINES))
def test_a_queue_position_that_is_not_a_whole_number_of_zero_or_more_reads_as_zero_with_a_log_line(
    case: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """design doc section 2.9. `count_throttled` keeps process-global state and the suite runs in random order: the label is unique to the case."""
    data = _valid_json()
    if case == "missing":
        del data["queue_line"]
    else:
        data["queue_line"] = _BAD_QUEUE_LINES[case]
    label = f"{case}-{tmp_path.name}"
    assert engine_files.validate_state(data) is None  # a pin: a state is never refused for it
    with caplog.at_level(logging.ERROR, logger=FILES_LOGGER):
        state = engine_files.state_from_json(data, label=label)
    assert state.queue_line == 0
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR and label in r.getMessage()]
    assert len(lines) == 1 and "queue_line" in lines[0]


def test_the_queue_position_log_key_holds_the_label_and_the_type_only_and_the_text_caps_the_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    keys: list[str] = []
    real = engine_files.count_throttled

    def spy(logger: logging.Logger, key: str, *args: object, **kwargs: object):
        keys.append(key)
        return real(logger, key, *args, **kwargs)

    monkeypatch.setattr(engine_files, "count_throttled", spy)
    label = f"keys-{tmp_path.name}"
    for value in ("SECRET-" + "x" * 5000, [list(range(5000))], {str(i): i for i in range(5000)}, float("nan")):
        data = _valid_json()
        data["queue_line"] = value
        with caplog.at_level(logging.ERROR, logger=FILES_LOGGER):
            assert engine_files.state_from_json(data, label=label).queue_line == 0
    assert [key for key in keys if "queue_line" in key] == [
        f"engine_files:queue_line:{label}:{name}" for name in ("str", "list", "dict", "float")
    ]
    assert not any("SECRET" in key for key in keys)
    assert all(len(r.getMessage()) < 400 for r in caplog.records if label in r.getMessage())


def test_a_huge_whole_queue_position_is_kept_without_a_log_line(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    data = _valid_json()
    data["queue_line"] = 10**400
    with caplog.at_level(logging.DEBUG, logger=FILES_LOGGER):
        assert engine_files.state_from_json(data, label=f"huge-{tmp_path.name}").queue_line == 10**400
    assert [r for r in caplog.records if r.name == FILES_LOGGER] == []


def test_the_saved_file_has_the_documented_shape(tmp_path: Path) -> None:
    engine_files.save_state(tmp_path, _full_state())
    engine_files.save_state(tmp_path, _full_state())
    path = engine_files.state_path(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") == 1  # compact: one line and the final newline
    data = json.loads(text)
    assert set(data) == {
        "saved_at",
        "reset_at",
        "last_update",
        "active_day",
        "persona_start_at",
        "queue_line",
        "repeats",
        "emotions",
    }
    assert data["queue_line"] == 0
    for key in ("saved_at", "reset_at", "last_update", "persona_start_at"):
        assert data[key].endswith("Z"), key
    assert data["saved_at"] == "2026-06-10T15:30:00Z"
    assert data["repeats"] == {"mem-1": {"first_at": "2026-06-10T10:30:00Z", "count": 2}}
    joy = data["emotions"]["joy"]
    assert set(joy) == {"short", "long", "last_entry_day", "snapshot"}
    assert joy["short"] == {"avg": 4.25, "weight": 1.5}
    first = joy["snapshot"][0]
    assert first == ["2026-06-08", 4.0, 3.5, 1, ["2026-06-05", 1, "2026-06-08"]]
    assert type(first[3]) is int
    assert joy["snapshot"][1][3] == 0 and type(joy["snapshot"][1][3]) is int
    assert joy["snapshot"][2][4] is None
    assert data["emotions"]["grief"]["last_entry_day"] is None
    assert (tmp_path / "state" / "emotion_state.json.bak1").exists()


def test_save_state_keeps_three_backups(tmp_path: Path) -> None:
    state = _full_state()
    for i in range(5):
        state.saved_at = NOW + timedelta(minutes=i)
        engine_files.save_state(tmp_path, state)
    names = sorted(
        p.name
        for p in (tmp_path / "state").iterdir()
        if p.name.startswith("emotion_state.json") and not p.name.endswith(".lock")
    )
    assert names == [
        "emotion_state.json",
        "emotion_state.json.bak1",
        "emotion_state.json.bak2",
        "emotion_state.json.bak3",
    ]


def test_save_state_does_not_change_its_argument_and_needs_saved_at(tmp_path: Path) -> None:
    state = _full_state()
    before = copy.deepcopy(state)
    engine_files.save_state(tmp_path, state)
    assert state == before
    with pytest.raises(ValueError):
        engine_files.save_state(tmp_path, EngineState())
    assert not (tmp_path / "state" / "emotion_state.json.tmp").exists()


def test_a_non_finite_value_fails_before_any_file_changes(tmp_path: Path) -> None:
    engine_files.save_state(tmp_path, _full_state())
    path = engine_files.state_path(tmp_path)
    before = _snapshot_of(tmp_path)
    bad = _full_state()
    bad.emotions["joy"].long.avg = float("nan")
    with pytest.raises(ValueError):
        engine_files.save_state(tmp_path, bad)
    assert _snapshot_of(tmp_path) == before
    assert path.exists()


def test_a_failed_write_leaves_the_old_file_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine_files.save_state(tmp_path, _full_state())
    before = _snapshot_of(tmp_path)

    def refuse(*args: object, **kwargs: object):
        raise PermissionError("read-only")

    monkeypatch.setattr(json_io, "save_with_backup_text", refuse)
    with pytest.raises(OSError):
        engine_files.save_state(tmp_path, _full_state())
    monkeypatch.undo()
    assert {k: v for k, v in _snapshot_of(tmp_path).items() if not k.endswith(".lock")} == {
        k: v for k, v in before.items() if not k.endswith(".lock")
    }


def test_naive_times_are_taken_as_utc_when_saved(tmp_path: Path) -> None:
    state = _full_state()
    state.saved_at = datetime(2026, 6, 10, 15, 30)  # naive: UTC by the repo's rule
    engine_files.save_state(tmp_path, state)
    data = json.loads(engine_files.state_path(tmp_path).read_text(encoding="utf-8"))
    assert data["saved_at"] == "2026-06-10T15:30:00Z"


def test_read_state_gives_independent_objects_each_call(tmp_path: Path) -> None:
    engine_files.save_state(tmp_path, _full_state())
    first = engine_files.read_state(tmp_path)
    first.state.emotions["joy"].short.avg = 9.9
    first.state.repeats.clear()
    second = engine_files.read_state(tmp_path)
    assert second.state == _full_state()


# --- read_state: which copy, and what it says --------------------------------------------------


def test_read_state_from_main_is_caught_up(tmp_path: Path) -> None:
    engine_files.save_state(tmp_path, _full_state())
    loaded = engine_files.read_state(tmp_path)
    assert (loaded.source, loaded.problem, loaded.caught_up) == ("main", None, True)
    assert loaded.shifts is None
    st = engine_files.state_path(tmp_path).stat()
    assert loaded.key == (st.st_ino, st.st_mtime_ns, st.st_size)


def test_a_state_from_a_backup_is_flagged_not_caught_up_and_main_is_not_touched(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    older = _full_state()
    older.saved_at = NOW - timedelta(hours=1)
    engine_files.save_state(tmp_path, older)
    engine_files.save_state(tmp_path, _full_state())  # main now newer; bak1 is `older`
    main = engine_files.state_path(tmp_path)
    main.write_text("{torn", encoding="utf-8")
    before = _snapshot_of(tmp_path)
    with caplog.at_level(logging.WARNING):
        loaded = engine_files.read_state(tmp_path)
    assert (loaded.source, loaded.caught_up) == ("bak1", False)
    assert loaded.state.saved_at == older.saved_at
    assert _snapshot_of(tmp_path) == before
    assert any("bak1" in r.getMessage() for r in caplog.records)


def test_read_state_reports_unreadable_but_not_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        assert engine_files.read_state(tmp_path).problem == "missing"
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    _write_state_dict(tmp_path, {"saved_at": "nope", "emotions": {}})
    with caplog.at_level(logging.ERROR, logger=FILES_LOGGER):
        loaded = engine_files.read_state(tmp_path)
    assert (loaded.state, loaded.source, loaded.problem) == (None, None, "unreadable")
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


# --- the stat-key cache ------------------------------------------------------------------------


@pytest.fixture
def walks(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []
    real = engine_files.read_json_walk

    def spy(path: Path, **kwargs: object) -> JsonWalk:
        calls.append(path)
        return real(path, **kwargs)

    monkeypatch.setattr(engine_files, "read_json_walk", spy)
    return calls


def test_the_cache_does_not_read_again_while_the_key_is_unchanged(
    tmp_path: Path, walks: list[Path]
) -> None:
    engine_files.save_state(tmp_path, _full_state())
    cache = engine_files.StateCache(tmp_path)
    first = cache.get()
    for _ in range(4):
        assert cache.get() is first
    assert len(walks) == 1
    assert first.key == file_ops.stat_path(engine_files.state_path(tmp_path)).key


def test_the_cache_hit_check_asks_file_ops_stat_path_for_the_state_file_and_no_raw_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walks: list[Path]
) -> None:
    engine_files.save_state(tmp_path, _full_state())
    cache = engine_files.StateCache(tmp_path)
    first = cache.get()
    asked: list[Path] = []
    real_stat_path = file_ops.stat_path

    def spy(path: str | Path):
        asked.append(Path(path))
        return real_stat_path(path)

    def no_raw_stat(self: Path, *args: object, **kwargs: object):
        raise AssertionError(f"a raw Path.stat of {self}")

    monkeypatch.setattr(file_ops, "stat_path", spy)
    monkeypatch.setattr(pathlib.Path, "stat", no_raw_stat)
    assert cache.get() is first
    assert asked == [engine_files.state_path(tmp_path)]
    assert len(walks) == 1


def test_switch_exists_asks_file_ops_stat_path_for_the_switch_file_and_no_raw_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written = tmp_path / "written"
    engine_files.write_switch(written, NOW)
    asked: list[Path] = []
    real_stat_path = file_ops.stat_path

    def spy(path: str | Path):
        asked.append(Path(path))
        return real_stat_path(path)

    def no_raw_stat(self: Path, *args: object, **kwargs: object):
        raise AssertionError(f"a raw Path.stat of {self}")

    monkeypatch.setattr(file_ops, "stat_path", spy)
    monkeypatch.setattr(pathlib.Path, "stat", no_raw_stat)
    assert engine_files.switch_exists(tmp_path / "absent") is False
    assert engine_files.switch_exists(written) is True
    assert asked == [engine_files.switch_path(tmp_path / "absent"), engine_files.switch_path(written)]


def test_the_cache_reloads_when_the_size_changes(tmp_path: Path, walks: list[Path]) -> None:
    engine_files.save_state(tmp_path, _full_state())
    cache = engine_files.StateCache(tmp_path)
    first = cache.get()
    newer = _full_state()
    newer.emotions["joy"].short.avg = 4.123456789  # longer text
    engine_files.save_state(tmp_path, newer)
    second = cache.get()
    assert second is not first
    assert second.state.emotions["joy"].short.avg == 4.123456789
    assert len(walks) == 2


def test_the_cache_reloads_on_a_mtime_change_alone(tmp_path: Path) -> None:
    engine_files.save_state(tmp_path, _full_state())
    path = engine_files.state_path(tmp_path)
    os.utime(path, ns=(_T1, _T1))
    cache = engine_files.StateCache(tmp_path)
    first = cache.get()
    path.write_text(path.read_text(encoding="utf-8").replace("4.25", "4.75"), encoding="utf-8")  # same size, same inode
    os.utime(path, ns=(_T2, _T2))
    second = cache.get()
    assert second is not first
    assert second.state.emotions["joy"].short.avg == 4.75


def test_the_cache_reloads_on_an_inode_change_alone(tmp_path: Path) -> None:
    engine_files.save_state(tmp_path, _full_state())
    path = engine_files.state_path(tmp_path)
    os.utime(path, ns=(_T1, _T1))
    cache = engine_files.StateCache(tmp_path)
    first = cache.get()
    replacement = path.with_name("replacement.tmp")
    replacement.write_text(path.read_text(encoding="utf-8").replace("4.25", "4.75"), encoding="utf-8")
    os.utime(replacement, ns=(_T1, _T1))
    os.replace(replacement, path)
    second = cache.get()
    assert second is not first and second.state.emotions["joy"].short.avg == 4.75


def test_a_replace_swaps_the_whole_tuple_and_leaves_the_old_one_alone(tmp_path: Path) -> None:
    engine_files.save_state(tmp_path, _full_state())
    cache = engine_files.StateCache(tmp_path)
    first = cache.get()
    snapshot_of_first = copy.deepcopy(first.state)
    newer = _full_state()
    newer.emotions["joy"].long.avg = 8.5
    engine_files.save_state(tmp_path, newer)
    second = cache.get()
    assert first.state == snapshot_of_first
    assert first.state.emotions["joy"].long.avg == 3.5
    assert second.state.emotions["joy"].long.avg == 8.5
    assert cache.get() is second


def test_a_state_from_a_backup_and_an_unreadable_one_are_re_walked_every_call(
    tmp_path: Path, walks: list[Path]
) -> None:
    engine_files.save_state(tmp_path, _full_state())
    engine_files.save_state(tmp_path, _full_state())
    engine_files.state_path(tmp_path).write_text("{torn", encoding="utf-8")
    cache = engine_files.StateCache(tmp_path)
    one, two = cache.get(), cache.get()
    assert one.source == two.source == "bak1" and one.caught_up is False
    assert len(walks) == 2
    for backup in ("bak1", "bak2", "bak3"):
        (tmp_path / "state" / f"emotion_state.json.{backup}").unlink(missing_ok=True)
    bad = cache.get()
    assert (bad.state, bad.problem) == (None, "unreadable")
    cache.get()
    assert len(walks) == 4


def test_a_main_file_that_comes_back_after_a_backup_load_is_seen(tmp_path: Path) -> None:
    older = _full_state()
    older.saved_at = NOW - timedelta(hours=1)
    engine_files.save_state(tmp_path, older)
    engine_files.save_state(tmp_path, _full_state())
    main = engine_files.state_path(tmp_path)
    main.write_text("{torn", encoding="utf-8")
    cache = engine_files.StateCache(tmp_path)
    assert cache.get().source == "bak1"
    engine_files.save_state(tmp_path, _full_state())
    assert cache.get().source == "main" and cache.get().caught_up is True


def test_a_stat_error_on_the_hit_check_falls_through_to_a_re_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walks: list[Path]
) -> None:
    engine_files.save_state(tmp_path, _full_state())
    cache = engine_files.StateCache(tmp_path)
    cache.get()

    def refuse(path: Path):
        raise PermissionError("no stat")

    monkeypatch.setattr(file_ops, "stat_path", refuse)
    assert cache.get().source == "main"
    assert len(walks) == 2


def test_shifts_are_stored_only_against_the_tuple_they_were_worked_out_from(tmp_path: Path) -> None:
    engine_files.save_state(tmp_path, _full_state())
    cache = engine_files.StateCache(tmp_path)
    first = cache.get()
    assert cache.store_shifts(first, {"joy": 1}) is True
    stored = cache.get()
    assert stored is not first and stored.shifts == {"joy": 1} and stored.state is first.state
    newer = _full_state()
    newer.emotions["joy"].long.avg = 7.0
    engine_files.save_state(tmp_path, newer)
    fresh = cache.get()
    assert fresh.shifts is None
    assert cache.store_shifts(first, {"joy": 2}) is False  # computed from a state that is gone
    assert cache.get() is fresh


# --- the switch --------------------------------------------------------------------------------


def test_the_switch_is_written_once_and_holds_only_the_time(tmp_path: Path) -> None:
    first = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
    assert engine_files.write_switch(tmp_path, first) is True
    path = engine_files.switch_path(tmp_path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"switch_over_at": "2026-07-01T08:00:00Z"}
    before = path.read_bytes()
    assert engine_files.write_switch(tmp_path, first + timedelta(days=3)) is False
    assert path.read_bytes() == before
    assert engine_files.read_switch(tmp_path) == first


def test_a_corrupt_switch_main_reads_from_its_backup_and_still_exists(tmp_path: Path) -> None:
    when = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
    engine_files.write_switch(tmp_path, when)
    path = engine_files.switch_path(tmp_path)
    path.with_name(path.name + ".bak1").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text("{torn", encoding="utf-8")
    assert engine_files.switch_exists(tmp_path) is True
    assert engine_files.read_switch(tmp_path) == when


def test_a_switch_main_with_no_usable_time_falls_to_a_backup_that_has_one(tmp_path: Path) -> None:
    when = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
    path = engine_files.switch_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"switch_over_at": "never"}', encoding="utf-8")
    path.with_name(path.name + ".bak1").write_text('{"switch_over_at": "2026-07-01T08:00:00Z"}', encoding="utf-8")
    assert engine_files.read_switch(tmp_path) == when
    path.write_text('["not", "an", "object"]', encoding="utf-8")
    assert engine_files.read_switch(tmp_path) == when


def test_the_files_are_written_with_the_modes_the_design_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """State: three backups, compact. Switch: three backups, once. (The queue is appended, not written: its test is in test_engine_queue.py.)"""
    calls: list[tuple[str, dict]] = []
    real = engine_files.write_json

    def spy(path: Path, data: object, **kwargs: object):
        calls.append((path.name, kwargs))
        return real(path, data, **kwargs)

    monkeypatch.setattr(engine_files, "write_json", spy)
    engine_files.save_state(tmp_path, _full_state())
    engine_files.write_switch(tmp_path, NOW)
    assert calls == [
        ("emotion_state.json", {"backups": 3, "indent": None, "verify": False}),
        ("emotion_switch.json", {"backups": 3, "if_missing": True}),
    ]


def test_read_switch_is_none_for_no_file_and_for_a_switch_with_no_usable_time(tmp_path: Path) -> None:
    assert engine_files.read_switch(tmp_path) is None
    path = engine_files.switch_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"switch_over_at": "never"}', encoding="utf-8")
    assert engine_files.read_switch(tmp_path) is None
    assert engine_files.switch_exists(tmp_path) is True


# --- is_live() ---------------------------------------------------------------------------------


def test_is_live_is_false_with_no_switch_and_true_on_the_same_instance_once_it_is_written(
    tmp_path: Path,
) -> None:
    engine = EmotionEngine(tmp_path)
    assert engine.is_live() is False
    assert engine.is_live() is False
    engine_files.write_switch(tmp_path, NOW)
    assert engine.is_live() is True


def test_a_true_answer_is_cached_for_the_life_of_the_instance(tmp_path: Path) -> None:
    engine_files.write_switch(tmp_path, NOW)
    engine = EmotionEngine(tmp_path)
    assert engine.is_live() is True
    engine_files.switch_path(tmp_path).unlink()
    assert engine.is_live() is True
    assert EmotionEngine(tmp_path).is_live() is False  # a new instance asks the file again


def test_a_false_answer_is_re_stated_each_time_and_a_true_one_never(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    real = engine_files.switch_state

    def counting(persona_dir: Path) -> bool | None:
        calls.append(persona_dir)
        return real(persona_dir)

    monkeypatch.setattr(engine_files, "switch_state", counting)
    engine = EmotionEngine(tmp_path)
    engine.is_live()
    engine.is_live()
    assert len(calls) == 2
    engine_files.write_switch(tmp_path, NOW)
    engine.is_live()
    engine.is_live()
    engine.is_live()
    assert len(calls) == 3


def _refuse_the_switch_stat(monkeypatch: pytest.MonkeyPatch) -> None:
    real_stat = file_ops.stat_path

    def refuse(path: str | Path):
        if Path(path).name == "emotion_switch.json":
            raise PermissionError("denied")
        return real_stat(path)

    monkeypatch.setattr(file_ops, "stat_path", refuse)


def test_switch_state_tells_a_seen_file_from_a_missing_one_from_an_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """V3: True seen, False missing, None on a stat error other than "no such file" (logged)."""
    engine_files.write_switch(tmp_path / "seen", NOW)
    assert engine_files.switch_state(tmp_path / "seen") is True
    assert engine_files.switch_state(tmp_path / "missing") is False
    _refuse_the_switch_stat(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=FILES_LOGGER):
        assert engine_files.switch_state(tmp_path / "seen") is None
    assert any("emotion_switch.json" in r.getMessage() for r in caplog.records)


def test_a_stat_error_other_than_not_found_answers_not_switched_and_switch_exists_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Owner 2026-11-05: "If the system can't tell if its switched, assume its not." A switch file that is on disk but
    cannot be inspected is still answered False, with a warning that names the error."""
    engine_files.write_switch(tmp_path, NOW)
    _refuse_the_switch_stat(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=FILES_LOGGER):
        assert engine_files.switch_exists(tmp_path) is False
    assert any("emotion_switch.json" in r.getMessage() and "denied" in r.getMessage() for r in caplog.records)


def test_is_live_is_false_on_a_stat_error_and_the_unknown_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    real = engine_files.switch_state

    def counting(persona_dir: Path) -> bool | None:
        calls.append(persona_dir)
        return real(persona_dir)

    monkeypatch.setattr(engine_files, "switch_state", counting)
    engine_files.write_switch(tmp_path, NOW)  # switched, but the stat of the file fails
    engine = EmotionEngine(tmp_path)
    with monkeypatch.context() as fault:
        _refuse_the_switch_stat(fault)
        assert engine.is_live() is False
        assert engine.is_live() is False
        assert len(calls) == 2  # asked again on the next call: an unknown is never remembered
    assert engine.is_live() is True  # the fault cleared; the file is seen now
    assert len(calls) == 3


def test_only_a_switch_file_that_was_seen_is_cached_as_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine_files.write_switch(tmp_path, NOW)
    engine = EmotionEngine(tmp_path)
    assert engine.is_live() is True
    _refuse_the_switch_stat(monkeypatch)
    assert engine.is_live() is True  # a seen file is remembered for the life of the instance, even under a later stat error
    assert EmotionEngine(tmp_path).is_live() is False  # a new instance meeting the fault cannot tell, so not switched
    monkeypatch.undo()
    engine_files.switch_path(tmp_path).unlink()
    assert engine.is_live() is True


def test_fold_change_queues_its_event_and_read_is_still_empty(tmp_path: Path) -> None:
    engine = EmotionEngine(tmp_path, now=lambda: NOW)
    engine.fold_change({"joy": 1.0}, None, "test")
    assert [(e.kind, e.at, e.source, dict(e.emotions)) for e in engine_files.read_queue(tmp_path, from_line=0).events] == [
        ("change", NOW, "test", {"joy": 1.0})
    ]
    assert not hasattr(engine, "recorded_calls")  # the record lives on the test fixture only
    assert engine.read() == {}
    engine_files.write_switch(tmp_path, NOW)
    assert engine.is_live() is True
    assert engine.read() == {}  # no state file yet: nothing to read
