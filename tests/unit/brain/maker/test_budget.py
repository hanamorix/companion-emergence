from datetime import UTC, datetime

from brain.maker.budget import consume_budget


def test_consume_until_cap_then_false(tmp_path):
    now = datetime(2026, 6, 14, 12, 0, 0, tzinfo=UTC)
    assert consume_budget(tmp_path, now=now, cap=2) is True
    assert consume_budget(tmp_path, now=now, cap=2) is True
    assert consume_budget(tmp_path, now=now, cap=2) is False  # exhausted


def test_resets_next_day(tmp_path):
    d1 = datetime(2026, 6, 14, 23, 0, 0, tzinfo=UTC)
    d2 = datetime(2026, 6, 15, 1, 0, 0, tzinfo=UTC)
    assert consume_budget(tmp_path, now=d1, cap=1) is True
    assert consume_budget(tmp_path, now=d1, cap=1) is False
    assert consume_budget(tmp_path, now=d2, cap=1) is True  # new day


def test_corrupt_budget_resets(tmp_path):
    (tmp_path / "maker_budget.json").write_text("{bad", encoding="utf-8")
    assert consume_budget(tmp_path, now=datetime(2026, 6, 14, tzinfo=UTC), cap=1) is True
