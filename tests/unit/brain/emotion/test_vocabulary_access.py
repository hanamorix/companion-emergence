"""Tests for brain.emotion.vocabulary_access: how a process asks about and adds to her vocabulary (#801 section 5).

The bridge is the one owner. Another process asks it over the local web API and, with no bridge, reads her file read-only.
The two-process case with a real bridge is in tests/bridge/test_vocabulary_two_process.py.
"""

from __future__ import annotations

import json
import logging
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


# ---------------------------------------------------------------------------------------------
# a failure is reported unless the bridge is simply not running
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def no_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The report is throttled per kind for the whole process; let each test see its own."""
    monkeypatch.setattr(vocabulary_access, "warn_throttled", lambda logger, key, msg, *a, **k: logger.warning(msg, *a))


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == vocabulary_access.logger.name and r.levelno >= logging.WARNING]


def test_a_bridge_that_is_not_running_is_the_quiet_case(tmp_path: Path, bridge_calls: _Calls, no_throttle: None, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        vocabulary_access.lookup(tmp_path, ["joy"])
        vocabulary_access.check(tmp_path, [("brand_new", None)])
    assert _warnings(caplog) == []


@pytest.mark.parametrize(
    "answer",
    [{"error": "bridge error 401"}, {"error": "bridge error 500"}, {"error": "bridge timeout"}, {"entries": "wrong shape", "results": 5}],
)
def test_any_other_failure_is_reported_once_with_its_kind(tmp_path: Path, bridge_calls: _Calls, no_throttle: None, caplog: pytest.LogCaptureFixture, answer: dict) -> None:
    bridge_calls.answer = answer
    with caplog.at_level(logging.WARNING):
        vocabulary_access.lookup(tmp_path, ["joy"])
        vocabulary_access.start_turn(tmp_path)  # a timeout skips the rest of the turn (below); this test is about each call's report
        vocabulary_access.check(tmp_path, [("brand_new", None)])
    messages = _warnings(caplog)
    assert len(messages) == 2 and all("read-only" in m for m in messages)
    assert any(answer.get("error", "not a vocabulary reply") in m for m in messages) or "entries" in answer


def test_a_timeout_on_an_add_says_the_bridge_may_still_have_added_the_names(tmp_path: Path, bridge_calls: _Calls, no_throttle: None, caplog: pytest.LogCaptureFixture) -> None:
    bridge_calls.answer = {"error": "bridge timeout"}
    with caplog.at_level(logging.WARNING):
        results = vocabulary_access.check(tmp_path, [("brand_new", None)])
    assert [r.outcome for r in results] == ["missing"]
    assert "may still have added" in _warnings(caplog)[0]


def test_a_body_that_is_not_json_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_throttle: None, caplog: pytest.LogCaptureFixture) -> None:
    def post(*_a, **_k):
        raise json.JSONDecodeError("bad", "", 0)

    monkeypatch.setattr(vocabulary_access, "post_to_bridge", post)
    with caplog.at_level(logging.WARNING):
        vocabulary_access.lookup(tmp_path, ["joy"])
    assert "not JSON" in _warnings(caplog)[0]


# ---------------------------------------------------------------------------------------------
# the read-only file is read once per version of the file
# ---------------------------------------------------------------------------------------------


def test_with_no_bridge_the_file_is_read_once_while_it_is_unchanged_and_again_when_it_changes(
    tmp_path: Path, bridge_calls: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brain.emotion import persona_loader

    path = _write(tmp_path, [_entry("Quiet Dread")])
    loads: list[Path] = []
    real = persona_loader.load_persona_vocabulary
    monkeypatch.setattr(persona_loader, "load_persona_vocabulary", lambda p, **k: (loads.append(p), real(p, **k))[1])
    for _ in range(5):
        assert vocabulary_access.lookup(tmp_path, ["no_such_word"])["no_such_word"] is None  # misses: the file is still read once
    assert loads == [path]
    _write(tmp_path, [_entry("Quiet Dread"), _entry("A Longer Name Now")])
    assert vocabulary_access.lookup(tmp_path, ["a_longer_name_now"])["a_longer_name_now"] is not None
    assert loads == [path, path]


def test_with_no_bridge_a_file_that_cannot_be_read_is_reported_once_and_read_once(
    tmp_path: Path, bridge_calls: _Calls, monkeypatch: pytest.MonkeyPatch, no_throttle: None, caplog: pytest.LogCaptureFixture
) -> None:
    from brain.emotion import persona_loader

    (tmp_path / FILE).write_text("{not json", encoding="utf-8")
    loads: list[Path] = []
    real = persona_loader.load_persona_vocabulary
    monkeypatch.setattr(persona_loader, "load_persona_vocabulary", lambda p, **k: (loads.append(p), real(p, **k))[1])
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            vocabulary_access.lookup(tmp_path, ["no_such_word"])
    assert len(loads) == 1
    assert len([m for m in _warnings(caplog) if "cannot be read" in m]) == 1


# ---------------------------------------------------------------------------------------------
# the owner's add takes entry fields only, and only valid ones
# ---------------------------------------------------------------------------------------------


def test_the_owner_add_ignores_any_key_that_is_not_an_entry_field(tmp_path: Path) -> None:
    vocabulary.set_bridge_owner(True)
    results = vocabulary_access.check(tmp_path, [("word_a", {"name": "other", "description": "d", "extra": 1})])
    assert results == [NameResult("word_a", "word_a", "used", True)]
    entry = json.loads((tmp_path / FILE).read_text(encoding="utf-8"))["emotions"][0]
    assert entry["name"] == "word_a" and entry["description"] == "d" and "extra" not in entry


@pytest.mark.parametrize(
    "fields",
    [
        {"description": 5},
        {"category": None},
        {"decay_half_life_days": "7"},
        {"decay_half_life_days": True},
        {"decay_half_life_days": 0},
        {"decay_half_life_days": -1.0},
        {"decay_half_life_days": float("nan")},
        {"decay_half_life_days": 10**400},
        {"intensity_clamp": None},
        {"intensity_clamp": float("inf")},
    ],
)
def test_the_owner_add_refuses_an_item_with_an_invalid_field_and_writes_nothing_for_it(tmp_path: Path, fields: dict) -> None:
    vocabulary.set_bridge_owner(True)
    results = vocabulary_access.check(tmp_path, [("bad_one", fields), ("good_one", {"decay_half_life_days": None, "intensity_clamp": 8})])
    assert [(r.raw, r.outcome) for r in results] == [("bad_one", "refused"), ("good_one", "used")]
    assert [e["name"] for e in json.loads((tmp_path / FILE).read_text(encoding="utf-8"))["emotions"]] == ["good_one"]
    assert vocabulary.get("bad_one") is None


def test_the_owner_add_with_only_invalid_items_makes_no_file(tmp_path: Path) -> None:
    vocabulary.set_bridge_owner(True)
    assert [r.outcome for r in vocabulary_access.check(tmp_path, [("bad_one", {"description": 5})])] == ["refused"]
    assert list(tmp_path.iterdir()) == []


def test_the_bridge_timeout_is_a_named_dev_constant() -> None:
    assert dev_constants.VOCABULARY_BRIDGE_TIMEOUT_S > dev_constants.EMOTION_VOCABULARY_LOCK_WAIT_S


def test_the_entry_fields_a_caller_may_give_are_the_files_fields_but_the_name() -> None:
    assert set(vocabulary_access.ENTRY_FIELDS) == {"description", "category", "decay_half_life_days", "intensity_clamp"}
    assert vocabulary_file.vocabulary_path(Path("/x")) == Path("/x") / FILE


# ---------------------------------------------------------------------------------------------
# a bridge that is up but never answers: one wait per turn (#803 step 9)
# ---------------------------------------------------------------------------------------------

TIMEOUT = {"error": "bridge timeout"}


def test_after_one_timeout_later_lookups_and_adds_make_no_call_and_use_the_file(tmp_path: Path, bridge_calls: _Calls) -> None:
    path = _write(tmp_path, [_entry("Quiet Dread", 6.0)])
    before = path.read_bytes()
    bridge_calls.answer = TIMEOUT
    first = vocabulary_access.lookup(tmp_path, ["quiet_dread", "no_such_word"])
    assert len(bridge_calls) == 1
    for _ in range(3):
        assert vocabulary_access.lookup(tmp_path, ["quiet_dread", "no_such_word"]) == first
    adds = vocabulary_access.check(tmp_path, [("quiet_dread", None), ("Brand New Feeling", None), ("---", None)])
    assert [(r.outcome, r.spelling) for r in adds] == [("used", "Quiet Dread"), ("missing", None), ("refused", None)]
    assert len(bridge_calls) == 1  # one wait, then nothing was sent
    assert path.read_bytes() == before and vocabulary.get("brand_new_feeling") is None


def test_a_skipped_call_returns_what_a_timed_out_call_returns(tmp_path: Path, bridge_calls: _Calls) -> None:
    _write(tmp_path, [_entry("Quiet Dread", 6.0)])
    bridge_calls.answer = TIMEOUT
    items = [("quiet_dread", None), ("Brand New Feeling", None), ("---", None)]
    timed_out_add = vocabulary_access.check(tmp_path, items)
    skipped_add = vocabulary_access.check(tmp_path, items)
    vocabulary_access.start_turn(tmp_path)
    timed_out_lookup = vocabulary_access.lookup(tmp_path, ["quiet_dread", "nope"])
    skipped_lookup = vocabulary_access.lookup(tmp_path, ["quiet_dread", "nope"])
    assert skipped_add == timed_out_add and skipped_lookup == timed_out_lookup
    assert len(bridge_calls) == 2


def test_an_add_timeout_skips_later_lookups_and_a_lookup_timeout_skips_later_adds(tmp_path: Path, bridge_calls: _Calls) -> None:
    _write(tmp_path, [_entry("Quiet Dread", 6.0)])
    bridge_calls.answer = TIMEOUT
    vocabulary_access.check(tmp_path, [("brand_new", None)])
    vocabulary_access.lookup(tmp_path, ["quiet_dread"])
    assert len(bridge_calls) == 1
    vocabulary_access.start_turn(tmp_path)
    vocabulary_access.lookup(tmp_path, ["quiet_dread"])
    vocabulary_access.check(tmp_path, [("brand_new", None)])
    assert len(bridge_calls) == 2


def test_the_next_turn_asks_the_bridge_again_and_only_for_the_persona_that_timed_out(tmp_path: Path, bridge_calls: _Calls) -> None:
    other = tmp_path / "other"
    other.mkdir()
    bridge_calls.answer = TIMEOUT
    vocabulary_access.lookup(tmp_path, ["joy"])
    vocabulary_access.lookup(tmp_path, ["joy"])
    assert [c[0] for c in bridge_calls] == [tmp_path]
    vocabulary_access.lookup(other, ["joy"])  # another persona's bridge was not the one that timed out
    assert [c[0] for c in bridge_calls] == [tmp_path, other]
    vocabulary_access.start_turn(tmp_path)
    bridge_calls.answer = {"entries": [{"name": "joy", "unknown": True}]}
    assert vocabulary_access.lookup(tmp_path, ["joy"]) == {"joy": None}  # the bridge's answer is used again
    assert [c[0] for c in bridge_calls] == [tmp_path, other, tmp_path]


@pytest.mark.parametrize("answer", [{"error": "bridge unreachable"}, {"error": "bridge error 500"}, {"entries": "wrong"}, ["no"]])
def test_only_a_timeout_stops_the_calls(tmp_path: Path, bridge_calls: _Calls, answer: object) -> None:
    bridge_calls.answer = answer
    vocabulary_access.lookup(tmp_path, ["joy"])
    vocabulary_access.lookup(tmp_path, ["joy"])
    assert len(bridge_calls) == 2


def test_the_owner_never_reads_the_flag(tmp_path: Path, bridge_calls: _Calls, monkeypatch: pytest.MonkeyPatch) -> None:
    reads: list[str] = []

    class Recording(dict):
        def get(self, *a, **k):
            reads.append("get")
            return super().get(*a, **k)

        def __getitem__(self, key):
            reads.append("getitem")
            return super().__getitem__(key)

        def __contains__(self, key):
            reads.append("contains")
            return super().__contains__(key)

    monkeypatch.setattr(vocabulary_access, "_BRIDGE_TIMED_OUT", Recording({tmp_path: True}))
    vocabulary.set_bridge_owner(True)
    assert vocabulary_access.lookup(tmp_path, ["joy"])["joy"] is not None
    assert vocabulary_access.lookup(tmp_path, vocabulary_access.ALL)
    assert vocabulary_access.check(tmp_path, [("Fresh Word", None)])[0].outcome == "used"
    assert vocabulary_access.names_used(tmp_path, ["fresh_word"]) == {"fresh_word": "fresh_word"}
    assert vocabulary_access.names_known(tmp_path, ["fresh_word"]) == {"fresh_word": "fresh_word"}
    assert reads == [] and bridge_calls == []


def test_the_timeout_report_says_later_calls_go_to_the_file_until_the_next_turn_and_a_skipped_call_says_nothing(
    tmp_path: Path, bridge_calls: _Calls, no_throttle: None, caplog: pytest.LogCaptureFixture
) -> None:
    bridge_calls.answer = TIMEOUT
    with caplog.at_level(logging.WARNING):
        vocabulary_access.lookup(tmp_path, ["joy"])
        first = _warnings(caplog)
        vocabulary_access.lookup(tmp_path, ["joy"])
        vocabulary_access.check(tmp_path, [("brand_new", None)])
    assert first == [
        "vocabulary: the bridge did not answer /vocabulary/lookup (bridge timeout); "
        "her file is read read-only for this call and for later vocabulary calls until the next turn"
    ]
    assert _warnings(caplog) == first


def test_the_add_timeout_report_keeps_its_may_have_added_words(tmp_path: Path, bridge_calls: _Calls, no_throttle: None, caplog: pytest.LogCaptureFixture) -> None:
    bridge_calls.answer = TIMEOUT
    with caplog.at_level(logging.WARNING):
        vocabulary_access.check(tmp_path, [("brand_new", None)])
    assert _warnings(caplog) == [
        "vocabulary: the bridge did not answer /vocabulary/add (bridge timeout; the bridge may still have added the names); "
        "her file is read read-only for this call and for later vocabulary calls until the next turn"
    ]


def test_a_real_bridge_that_accepts_and_never_answers_costs_one_wait_per_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole path with a real socket: the stub accepts a connection and never replies; `post_to_bridge` is not replaced."""
    import socket
    import threading
    import time

    from brain.bridge import state_file

    wait = 0.4
    monkeypatch.setattr(dev_constants, "VOCABULARY_BRIDGE_TIMEOUT_S", wait)
    _write(tmp_path, [_entry("Quiet Dread", 6.0)])
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    accepted: list[socket.socket] = []

    def accept_forever() -> None:
        while True:
            try:
                accepted.append(server.accept()[0])
            except OSError:
                return

    thread = threading.Thread(target=accept_forever, daemon=True)
    thread.start()
    state_file.write(
        tmp_path,
        state_file.BridgeState(
            persona="p", pid=None, port=server.getsockname()[1], started_at="2026-11-24T10:00:00Z", stopped_at=None,
            shutdown_clean=False, client_origin="tests", auth_token="t",
        ),
    )
    try:
        begin = time.monotonic()
        first = vocabulary_access.lookup(tmp_path, ["quiet_dread", "no_such_word"])
        first_wait = time.monotonic() - begin
        begin = time.monotonic()
        for _ in range(3):
            assert vocabulary_access.lookup(tmp_path, ["quiet_dread", "no_such_word"]) == first
        adds = vocabulary_access.check(tmp_path, [("quiet_dread", None), ("Brand New Feeling", None)])
        later = time.monotonic() - begin
        assert first["quiet_dread"] is not None and first["no_such_word"] is None
        assert [r.outcome for r in adds] == ["used", "missing"]
        assert first_wait >= wait * 0.9 and later < wait * 0.5  # one wait, then no wait at all
        assert len(accepted) == 1  # one connection was ever made
    finally:
        server.close()
        for conn in accepted:
            conn.close()
        thread.join(timeout=2)
