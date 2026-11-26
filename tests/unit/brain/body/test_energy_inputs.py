"""Gathering the inputs of the body energy step (#650 spec 2.4, 2.3 speaker averages): synthetic persona dirs only."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain import dev_constants
from brain.body import energy_inputs
from brain.body.energy_inputs import gather_energy_inputs
from brain.emotion.engine import Reading
from brain.ingest.buffer import append_archive, ingest_turn

NOW = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
START = NOW - timedelta(seconds=900)


def _say(persona_dir: Path, speaker: str, words: int, at: datetime, session: str = "sess_a", text: str | None = None):
    body = text if text is not None else " ".join(f"w{i}" for i in range(words))
    ingest_turn(persona_dir, {"session_id": session, "speaker": speaker, "text": body, "ts": at.isoformat()})


def _archive(persona_dir: Path, rows: list[dict], session: str = "sess_old") -> None:
    append_archive(persona_dir, session, rows)


def _row(speaker: str, words: int, at: datetime, session: str = "sess_old") -> dict:
    return {"session_id": session, "speaker": speaker, "text": " ".join(f"a{i}" for i in range(words)), "ts": at.isoformat()}


def _gather(persona_dir: Path):
    return gather_energy_inputs(persona_dir, interval_start=START, now=NOW)


def _lf(inputs):
    return [(m.weight, round(m.length_factor, 6)) for m in inputs.messages]


def _ago(**kw):
    return NOW - timedelta(**kw)


# ── the feelings ────────────────────────────────────────────────────────────────────────────────────────────────────


def test_feelings_come_from_one_read_and_high_from_feels_high(tmp_path, emotion_engine_stub):
    emotion_engine_stub.readings = {
        "joy": Reading(short=6.0, long=3.0, difference=3.0, shift=0),
        "peace": Reading(short=4.0, long=4.0, difference=0.0, shift=0),
    }
    emotion_engine_stub.high = {"joy", "peace", "anger"}
    reads = []
    real_read = emotion_engine_stub.read
    emotion_engine_stub.read = lambda: (reads.append(1), real_read())[1]
    inputs = _gather(tmp_path)
    assert len(reads) == 1
    assert inputs.short == {"joy": 6.0, "peace": 4.0}
    assert inputs.long == {"joy": 3.0, "peace": 4.0}
    assert inputs.high == frozenset({"joy"})  # only lift and drain names are asked; peace and anger are not


def test_nothing_readable_gives_empty_feelings_and_no_high(tmp_path, emotion_engine_stub):
    emotion_engine_stub.high = {"joy"}
    inputs = _gather(tmp_path)
    assert inputs.short == {} and inputs.long == {} and inputs.high == frozenset()


def test_the_persona_start_is_read_from_the_state_file_and_none_closes_the_gate(tmp_path, emotion_engine_stub, monkeypatch):
    assert _gather(tmp_path).persona_start_at is None
    began = NOW - timedelta(days=40)
    monkeypatch.setattr(
        energy_inputs.engine_files,
        "read_state",
        lambda persona_dir: SimpleNamespace(state=SimpleNamespace(persona_start_at=began)),
    )
    assert _gather(tmp_path).persona_start_at == began
    assert _gather(tmp_path).interval_start == START


# ── the interval's messages ─────────────────────────────────────────────────────────────────────────────────────────


def test_only_user_and_assistant_rows_after_the_start_up_to_now_count(tmp_path, emotion_engine_stub):
    _say(tmp_path, "assistant", 10, _ago(days=2))  # an earlier message: for the average only
    _say(tmp_path, "user", 10, START - timedelta(seconds=1))  # before the interval
    _say(tmp_path, "user", 10, START)  # not after the start
    _say(tmp_path, "user", 10, START + timedelta(seconds=1))
    _say(tmp_path, "assistant", 10, NOW - timedelta(seconds=1))
    _say(tmp_path, "image", 10, NOW - timedelta(seconds=2))
    _say(tmp_path, "summary", 10, NOW - timedelta(seconds=3))
    _say(tmp_path, "user", 10, NOW + timedelta(seconds=5))  # after now: the next heartbeat's
    inputs = _gather(tmp_path)
    assert sorted(m.weight for m in inputs.messages) == [0.5, 1.0]


def test_the_speaker_weights_are_hers_and_the_users(tmp_path, emotion_engine_stub):
    _say(tmp_path, "user", 5, START + timedelta(seconds=10))
    _say(tmp_path, "assistant", 5, START + timedelta(seconds=20))
    inputs = _gather(tmp_path)
    assert [m.weight for m in inputs.messages] == [
        dev_constants.BODY_ENERGY_USER_MESSAGE_WEIGHT,
        dev_constants.BODY_ENERGY_HER_MESSAGE_WEIGHT,
    ]


def test_a_first_message_counts_one(tmp_path, emotion_engine_stub):
    _say(tmp_path, "user", 500, START + timedelta(seconds=10))
    assert _lf(_gather(tmp_path)) == [(0.5, 1.0)]


def test_length_is_words_split_on_whitespace(tmp_path, emotion_engine_stub):
    _say(tmp_path, "user", 0, _ago(days=1), text="one  two\tthree\nfour")  # 4 words
    _say(tmp_path, "user", 0, START + timedelta(seconds=10), text="a b c d e f g h")  # 8 words: ratio 2 -> 1.5
    assert _lf(_gather(tmp_path)) == [(0.5, 1.5)]


def test_each_message_uses_the_average_of_its_own_speakers_earlier_messages(tmp_path, emotion_engine_stub):
    _say(tmp_path, "user", 10, _ago(days=3))
    _say(tmp_path, "user", 30, _ago(days=2))  # user average 20
    _say(tmp_path, "assistant", 100, _ago(days=2))  # her average 100
    _say(tmp_path, "user", 40, START + timedelta(seconds=10))  # 40 / 20 = 2 -> 1.5
    _say(tmp_path, "assistant", 50, START + timedelta(seconds=20))  # 50 / 100 = 0.5
    assert _lf(_gather(tmp_path)) == [(0.5, 1.5), (1.0, 0.5)]


def test_earlier_messages_of_the_same_heartbeat_count_toward_the_average(tmp_path, emotion_engine_stub):
    _say(tmp_path, "user", 10, START + timedelta(seconds=10))  # first: 1
    _say(tmp_path, "user", 30, START + timedelta(seconds=20))  # average of the earlier = 10: 3 -> 2.0
    _say(tmp_path, "user", 20, START + timedelta(seconds=30))  # average of 10 and 30 = 20: 1
    assert _lf(_gather(tmp_path)) == [(0.5, 1.0), (0.5, 2.0), (0.5, 1.0)]


def test_a_speaker_whose_average_is_zero_counts_one(tmp_path, emotion_engine_stub):
    ingest_turn(tmp_path, {"session_id": "sess_a", "speaker": "assistant", "text": "", "ts": _ago(days=1).isoformat()})
    _say(tmp_path, "assistant", 30, START + timedelta(seconds=10))
    assert _lf(_gather(tmp_path)) == [(1.0, 1.0)]


def test_archived_rows_count_for_the_average_once_each_and_old_rows_do_not(tmp_path, emotion_engine_stub):
    same = _row("user", 10, _ago(days=5))
    _archive(tmp_path, [same, same, _row("user", 30, _ago(days=4))])  # two byte-identical turns count once: average 20
    _archive(tmp_path, [_row("user", 1000, _ago(days=40))], session="sess_ancient")  # outside the 30 days
    _say(tmp_path, "user", 40, START + timedelta(seconds=10))
    assert _lf(_gather(tmp_path)) == [(0.5, 1.5)]  # 40 / 20 = 2; a duplicate counted twice would give 1.7


def test_a_turn_in_the_buffer_and_the_archive_is_counted_once(tmp_path, emotion_engine_stub):
    at = _ago(days=2)
    turn = _row("user", 10, at, session="sess_a")
    ingest_turn(tmp_path, turn)
    _archive(tmp_path, [turn], session="sess_a")
    _say(tmp_path, "user", 30, _ago(days=1))  # the earlier messages are 10 and 30 words: average 20
    _say(tmp_path, "user", 40, START + timedelta(seconds=10))  # 40 / 20 = 2 -> 1.5 (a duplicate counted twice would give 1.7)
    assert _lf(_gather(tmp_path)) == [(0.5, 1.5)]


def test_the_thirty_day_read_runs_only_when_the_interval_has_messages(tmp_path, emotion_engine_stub, monkeypatch):
    _say(tmp_path, "user", 10, _ago(days=2))  # not in the interval
    calls = []
    monkeypatch.setattr(energy_inputs, "_window_rows", lambda *a, **k: calls.append(1) or {})
    assert _gather(tmp_path).messages == ()
    assert calls == []
    _say(tmp_path, "user", 10, START + timedelta(seconds=10))
    monkeypatch.undo()
    assert len(_gather(tmp_path).messages) == 1


def test_an_unreadable_buffer_costs_only_its_own_rows(tmp_path, emotion_engine_stub, caplog, monkeypatch):
    from brain.ingest import buffer

    _say(tmp_path, "user", 10, START + timedelta(seconds=10), session="sess_good")
    _say(tmp_path, "user", 10, START + timedelta(seconds=11), session="sess_bad")
    real = buffer.read_session_after

    def flaky(persona_dir, session_id, after_ts):
        if session_id == "sess_bad":
            raise OSError("sharing violation")
        return real(persona_dir, session_id, after_ts)

    monkeypatch.setattr(buffer, "read_session_after", flaky)
    with caplog.at_level(logging.WARNING):
        inputs = _gather(tmp_path)
    assert len(inputs.messages) == 1
    assert any("sess_bad" in r.getMessage() for r in caplog.records)


def test_a_row_with_text_that_is_not_a_string_or_a_time_that_does_not_parse_is_skipped(tmp_path, emotion_engine_stub):
    path = tmp_path / "active_conversations"
    path.mkdir()
    first = (START + timedelta(seconds=5)).isoformat()
    second = (START + timedelta(seconds=6)).isoformat()
    (path / "sess_odd.jsonl").write_text(
        f'{{"speaker": "user", "text": null, "ts": "{first}"}}\n'
        '{"speaker": "user", "text": "fine words here", "ts": "not a time"}\n'
        f'{{"speaker": "user", "text": "fine words here", "ts": "{second}"}}\n'
    )
    assert len(_gather(tmp_path).messages) == 1


def test_gathering_writes_nothing(tmp_path, emotion_engine_stub):
    _say(tmp_path, "user", 10, START + timedelta(seconds=10))
    before = sorted((p, p.stat().st_mtime_ns, p.stat().st_size) for p in tmp_path.rglob("*") if p.is_file())
    _gather(tmp_path)
    after = sorted((p, p.stat().st_mtime_ns, p.stat().st_size) for p in tmp_path.rglob("*") if p.is_file())
    assert before == after


@pytest.mark.parametrize("folder", ["active_conversations", "archived_conversations"])
def test_a_missing_folder_gives_no_messages(tmp_path, emotion_engine_stub, folder):
    assert not (tmp_path / folder).exists()
    assert _gather(tmp_path).messages == ()
