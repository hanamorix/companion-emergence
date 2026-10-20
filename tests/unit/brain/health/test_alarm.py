"""Tests for brain.health.alarm."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.alarm import compute_pending_alarms


def _audit_entry(
    file: str, when: datetime, action: str = "restored_from_bak1", kind: str = "json_parse_error"
) -> dict:
    return {
        "timestamp": when.isoformat().replace("+00:00", "Z"),
        "anomalies": [
            {
                "timestamp": when.isoformat().replace("+00:00", "Z"),
                "file": file,
                "kind": kind,
                "action": action,
                "quarantine_path": None,
                "likely_cause": "unknown",
                "detail": "",
            }
        ],
    }


def _seed(persona_dir: Path, entries: list[dict]) -> None:
    persona_dir.mkdir(parents=True, exist_ok=True)
    p = persona_dir / "heartbeats.log.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def test_no_anomalies_no_alarms(tmp_path: Path) -> None:
    assert compute_pending_alarms(tmp_path) == []


def test_reset_to_default_on_identity_file_alarms(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        [_audit_entry("emotion_vocabulary.json", datetime.now(UTC), action="reset_to_default")],
    )
    alarms = compute_pending_alarms(tmp_path)
    assert len(alarms) == 1
    assert alarms[0].file == "emotion_vocabulary.json"


def test_sqlite_integrity_fail_alarms(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        [
            _audit_entry(
                "memories.db",
                datetime.now(UTC),
                kind="sqlite_integrity_fail",
                action="alarmed_unrecoverable",
            )
        ],
    )
    alarms = compute_pending_alarms(tmp_path)
    assert len(alarms) == 1
    assert alarms[0].kind == "sqlite_integrity_fail"


def test_acknowledged_alarm_suppressed(tmp_path: Path) -> None:
    when = datetime.now(UTC)
    entries = [
        _audit_entry("emotion_vocabulary.json", when, action="reset_to_default"),
        # User acknowledged after the reset
        {
            "timestamp": (when + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
            "user_acknowledged": ["emotion_vocabulary.json"],
        },
    ]
    _seed(tmp_path, entries)
    alarms = compute_pending_alarms(tmp_path)
    assert alarms == []


# --- the gate's quarantine and damaged files are read (#470 review item 5) ---------------------------------------------

import pytest  # noqa: E402

from brain.health import alarm as alarm_mod  # noqa: E402
from brain.health.alarm import RECURRING_THRESHOLD  # noqa: E402

_QUARANTINE = "consolidation_quarantine.jsonl"
_DAMAGED = "pending_candidates.damaged.jsonl"
_FIELD = {_QUARANTINE: "quarantined_at", _DAMAGED: "kept_at"}
_KIND = {_QUARANTINE: "quarantined_candidates", _DAMAGED: "damaged_queue_lines"}


def _rows(persona_dir: Path, name: str, stamps: list[datetime], *, extra: list[str] | None = None) -> None:
    """Write one row per stamp (oldest first) to the kept-aside file ``name``, then any raw extra lines."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({_FIELD[name]: s.isoformat(), "entry": {"id": f"c{i}"}}) for i, s in enumerate(stamps)]
    (persona_dir / name).write_text("\n".join([*lines, *(extra or [])]) + "\n", encoding="utf-8")


def _recent(n: int, *, days_ago: float = 0.0) -> list[datetime]:
    base = datetime.now(UTC) - timedelta(days=days_ago)
    return [base - timedelta(minutes=n - i) for i in range(n)]  # oldest first


@pytest.mark.parametrize("name", [_QUARANTINE, _DAMAGED])
def test_fewer_rows_than_the_recurring_threshold_raise_no_alarm(tmp_path: Path, name: str) -> None:
    # One quarantined item (with its evidence) is a handled event, not an alarm.
    _rows(tmp_path, name, _recent(RECURRING_THRESHOLD - 1))
    assert compute_pending_alarms(tmp_path) == []


@pytest.mark.parametrize("name", [_QUARANTINE, _DAMAGED])
def test_rows_at_the_recurring_threshold_raise_one_alarm_with_the_count(tmp_path: Path, name: str) -> None:
    stamps = _recent(RECURRING_THRESHOLD + 2)
    _rows(tmp_path, name, stamps)
    alarms = compute_pending_alarms(tmp_path)
    assert [(a.file, a.kind, a.occurrences_in_window) for a in alarms] == [(name, _KIND[name], len(stamps))]
    assert abs((alarms[0].first_seen_at - stamps[0]).total_seconds()) < 1  # the oldest counted row


def test_the_threshold_is_the_one_the_anomaly_alarm_already_used() -> None:
    # No new magic number: the named constant is the `>= 6` that was inline for recurring anomalies.
    assert RECURRING_THRESHOLD == 6
    assert alarm_mod.WINDOW_DAYS == 7


def test_the_anomaly_alarm_still_fires_at_the_same_count(tmp_path: Path) -> None:
    when = datetime.now(UTC)
    _seed(tmp_path, [_audit_entry("user_preferences.json", when - timedelta(minutes=i)) for i in range(RECURRING_THRESHOLD)])
    assert [a.file for a in compute_pending_alarms(tmp_path)] == ["user_preferences.json"]
    _seed(tmp_path, [_audit_entry("user_preferences.json", when - timedelta(minutes=i)) for i in range(RECURRING_THRESHOLD - 1)])
    assert compute_pending_alarms(tmp_path) == []


def test_rows_older_than_the_window_are_not_counted(tmp_path: Path) -> None:
    old = _recent(RECURRING_THRESHOLD + 4, days_ago=alarm_mod.WINDOW_DAYS + 1)
    _rows(tmp_path, _QUARANTINE, [*old, *_recent(RECURRING_THRESHOLD - 1)])
    assert compute_pending_alarms(tmp_path) == []


def test_unreadable_rows_and_a_missing_file_do_not_count_or_raise(tmp_path: Path) -> None:
    junk = ["not json at all", '{"quarantined_at": "yesterday-ish"}', '{"entry": {}}', "[1, 2]", ""]
    _rows(tmp_path, _QUARANTINE, _recent(RECURRING_THRESHOLD - 1), extra=junk)
    assert compute_pending_alarms(tmp_path) == []  # 5 good rows only
    _rows(tmp_path, _QUARANTINE, _recent(RECURRING_THRESHOLD), extra=junk)
    assert [a.occurrences_in_window for a in compute_pending_alarms(tmp_path)] == [RECURRING_THRESHOLD]
    assert compute_pending_alarms(tmp_path / "no-such-persona") == []


def test_a_damaged_line_inside_the_tail_does_not_hide_the_older_rows(tmp_path: Path) -> None:
    # A line that does not parse costs one line of the tail, not the rows behind it.
    stamps = _recent(RECURRING_THRESHOLD * 3)
    _rows(tmp_path, _QUARANTINE, stamps)
    path = tmp_path / _QUARANTINE
    lines = path.read_text(encoding="utf-8").splitlines()
    lines.insert(len(lines) - 2, "{torn")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert [a.occurrences_in_window for a in compute_pending_alarms(tmp_path)] == [len(stamps)]


def test_the_read_is_bounded_by_the_window_not_the_history_of_the_file(tmp_path: Path, monkeypatch) -> None:
    # The quarantine file has no rotation and /health reads this on its restart poll: never parse the whole history.
    history = _recent(20_000, days_ago=alarm_mod.WINDOW_DAYS + 30)
    _rows(tmp_path, _QUARANTINE, [*history, *_recent(RECURRING_THRESHOLD)])
    asked: list[int] = []
    real = alarm_mod.read_jsonl_tail

    def spy(path, n, **kw):
        asked.append(n)
        return real(path, n, **kw)

    monkeypatch.setattr(alarm_mod, "read_jsonl_tail", spy)
    alarms = compute_pending_alarms(tmp_path)
    assert [a.occurrences_in_window for a in alarms] == [RECURRING_THRESHOLD]
    assert asked and max(asked) <= 4 * RECURRING_THRESHOLD


def test_an_acknowledgement_counts_only_the_rows_after_it(tmp_path: Path) -> None:
    ack_at = datetime.now(UTC) - timedelta(hours=1)
    _rows(tmp_path, _QUARANTINE, [*_recent(RECURRING_THRESHOLD + 3, days_ago=1)])
    _seed(tmp_path, [{"timestamp": ack_at.isoformat().replace("+00:00", "Z"), "user_acknowledged": [_QUARANTINE]}])
    assert compute_pending_alarms(tmp_path) == []  # acknowledged: the old rows are not counted again

    # a gate bug that keeps quarantining after the acknowledgement raises it again once the threshold is reached
    after = [datetime.now(UTC) - timedelta(minutes=RECURRING_THRESHOLD - i) for i in range(RECURRING_THRESHOLD - 1)]
    _rows(tmp_path, _QUARANTINE, [*_recent(RECURRING_THRESHOLD + 3, days_ago=1), *after])
    assert compute_pending_alarms(tmp_path) == []  # 5 rows since the acknowledgement
    more = [*after, datetime.now(UTC)]
    _rows(tmp_path, _QUARANTINE, [*_recent(RECURRING_THRESHOLD + 3, days_ago=1), *more])
    assert [a.occurrences_in_window for a in compute_pending_alarms(tmp_path)] == [RECURRING_THRESHOLD]


def test_acknowledging_one_file_leaves_the_other_alarm(tmp_path: Path) -> None:
    ack_at = datetime.now(UTC)
    _rows(tmp_path, _QUARANTINE, _recent(RECURRING_THRESHOLD, days_ago=1))
    _rows(tmp_path, _DAMAGED, _recent(RECURRING_THRESHOLD, days_ago=1))
    _seed(tmp_path, [{"timestamp": ack_at.isoformat().replace("+00:00", "Z"), "user_acknowledged": [_QUARANTINE]}])
    assert [a.file for a in compute_pending_alarms(tmp_path)] == [_DAMAGED]


def test_file_names_are_the_producers_constants() -> None:
    # Renaming a producer's file without the alarm's name fails here instead of silently unwiring the alarm.
    from brain.engines import consolidation
    from brain.memory import pending

    assert alarm_mod.QUARANTINE_FILENAME == consolidation._QUARANTINE_FILENAME == _QUARANTINE
    assert alarm_mod.DAMAGED_FILENAME == pending._DAMAGED_FILENAME == _DAMAGED


def test_a_gate_that_quarantines_everything_raises_the_alarm(tmp_path: Path) -> None:
    # The end-to-end case of the review: a gate bug (here, a classifier that never yields a verdict) quarantines every
    # candidate, with evidence, and memories quietly stop being promoted. The alarm is what makes that visible.
    from brain.engines.consolidation import UnusableClassifierReply, run_consolidation
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.pending import PendingQueue
    from brain.memory.store import Memory, MemoryStore

    store = MemoryStore(tmp_path / "memories.db")
    heb = HebbianMatrix(tmp_path / "hebbian.db")
    try:
        queue = PendingQueue(tmp_path)
        for i in range(RECURRING_THRESHOLD):
            queue.enqueue(
                Memory.create_new(content=f"candidate number {i}", memory_type="dream", domain="us", emotions={}),
                source="dream",
            )

        def broken(_cand, _ctx):
            raise UnusableClassifierReply("no usable verdict", "a reply")

        result = run_consolidation(store, persona_dir=tmp_path, hebbian=heb, classifier=broken)
        assert result.quarantined == RECURRING_THRESHOLD
    finally:
        store.close()
        heb.close()
    assert [(a.file, a.occurrences_in_window) for a in compute_pending_alarms(tmp_path)] == [
        (_QUARANTINE, RECURRING_THRESHOLD)
    ]


def test_damaged_queue_lines_kept_aside_by_a_claim_raise_the_alarm(tmp_path: Path) -> None:
    from brain.memory.pending import PendingQueue

    queue = PendingQueue(tmp_path)
    queue.path.write_bytes(b"".join(b"{torn %d\n" % i for i in range(RECURRING_THRESHOLD)))
    queue.claim()
    assert [(a.file, a.kind, a.occurrences_in_window) for a in compute_pending_alarms(tmp_path)] == [
        (_DAMAGED, "damaged_queue_lines", RECURRING_THRESHOLD)
    ]
