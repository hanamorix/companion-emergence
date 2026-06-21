from datetime import UTC, datetime, timedelta

from brain.kindled_link.relationship import (
    load_reflection_cadence,
    reflection_is_due,
    save_reflection_cadence,
)


def test_due_when_never_run(tmp_path):
    assert reflection_is_due(tmp_path, datetime(2026, 6, 20, 12, 0, tzinfo=UTC)) is True


def test_not_due_right_after_run(tmp_path):
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    save_reflection_cadence(tmp_path, now)
    assert reflection_is_due(tmp_path, now + timedelta(hours=1)) is False


def test_due_after_interval(tmp_path):
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    save_reflection_cadence(tmp_path, now)
    assert reflection_is_due(tmp_path, now + timedelta(hours=25)) is True


def test_load_roundtrips(tmp_path):
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    save_reflection_cadence(tmp_path, now)
    assert load_reflection_cadence(tmp_path)["last_run"] == now.isoformat()
