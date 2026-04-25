"""Unit tests for brain.engines._interests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.engines._interests import Interest, InterestSet

DEFAULT_INTERESTS_PATH = Path(__file__).parents[4] / "brain" / "engines" / "default_interests.json"


def _sample_dict(**overrides) -> dict:
    base = {
        "id": "abc-123",
        "topic": "Test topic",
        "pull_score": 6.5,
        "scope": "either",
        "related_keywords": ["test", "topic"],
        "notes": "",
        "first_seen": "2026-04-01T10:00:00Z",
        "last_fed": "2026-04-15T10:00:00Z",
        "last_researched_at": None,
        "feed_count": 3,
        "source_types": ["manual"],
    }
    base.update(overrides)
    return base


# ---- Interest ----


def test_interest_from_dict_valid():
    i = Interest.from_dict(_sample_dict())
    assert i.id == "abc-123"
    assert i.pull_score == 6.5
    assert i.scope == "either"
    assert i.related_keywords == ("test", "topic")
    assert i.last_researched_at is None


def test_interest_from_dict_with_researched_timestamp():
    data = _sample_dict(last_researched_at="2026-04-20T10:00:00Z")
    i = Interest.from_dict(data)
    assert i.last_researched_at is not None
    assert i.last_researched_at.tzinfo is not None


def test_interest_from_dict_invalid_scope_raises():
    with pytest.raises(ValueError):
        Interest.from_dict(_sample_dict(scope="whatever"))


def test_interest_to_dict_roundtrip():
    original = Interest.from_dict(_sample_dict())
    restored = Interest.from_dict(original.to_dict())
    assert restored == original


# ---- InterestSet: load / save ----


def test_interestset_load_missing_falls_back_to_defaults(tmp_path: Path):
    missing = tmp_path / "nope.json"
    loaded = InterestSet.load(missing, default_path=DEFAULT_INTERESTS_PATH)
    assert loaded.interests == ()  # default is empty


# ---- Health T10: attempt_heal wiring ----


def test_interestset_load_corrupt_quarantines_restores_from_bak(tmp_path: Path):
    """Corrupt live file + valid .bak1 → restore .bak1, return its data, anomaly set."""
    path = tmp_path / "interests.json"
    bak1 = tmp_path / "interests.json.bak1"

    # .bak1 has a valid interest
    bak1.write_text(json.dumps({"version": 1, "interests": [_sample_dict()]}), encoding="utf-8")
    path.write_text("{corrupt json{{", encoding="utf-8")

    result, anomaly = InterestSet.load_with_anomaly(path, default_path=DEFAULT_INTERESTS_PATH)

    assert anomaly is not None
    assert "bak1" in anomaly.action
    assert len(result.interests) == 1
    assert result.interests[0].topic == "Test topic"
    # original quarantined
    corrupt_files = list(tmp_path.glob("interests.json.corrupt-*"))
    assert len(corrupt_files) == 1


def test_interestset_load_corrupt_no_bak_resets_to_default(tmp_path: Path):
    """Corrupt live file + no .bak → reset to default (empty), anomaly with reset_to_default."""
    path = tmp_path / "interests.json"
    path.write_text("{corrupt json{{", encoding="utf-8")

    result, anomaly = InterestSet.load_with_anomaly(path, default_path=DEFAULT_INTERESTS_PATH)

    assert anomaly is not None
    assert anomaly.action == "reset_to_default"
    assert result.interests == ()


def test_interestset_load_corrupt_falls_back(tmp_path: Path):
    bad = tmp_path / "interests.json"
    bad.write_text("not valid{", encoding="utf-8")
    loaded = InterestSet.load(bad, default_path=DEFAULT_INTERESTS_PATH)
    assert loaded.interests == ()


def test_interestset_load_valid_file(tmp_path: Path):
    path = tmp_path / "interests.json"
    path.write_text(json.dumps({"version": 1, "interests": [_sample_dict()]}), encoding="utf-8")
    loaded = InterestSet.load(path, default_path=DEFAULT_INTERESTS_PATH)
    assert len(loaded.interests) == 1
    assert loaded.interests[0].topic == "Test topic"


def test_interestset_save_atomic(tmp_path: Path):
    path = tmp_path / "interests.json"
    s = InterestSet(interests=(Interest.from_dict(_sample_dict()),))
    s.save(path)
    # File exists + valid JSON + .new tempfile cleaned up
    assert path.exists()
    assert not (path.with_suffix(path.suffix + ".new")).exists()
    reloaded = InterestSet.load(path, default_path=DEFAULT_INTERESTS_PATH)
    assert reloaded.interests[0].id == "abc-123"


def test_interestset_load_bad_interest_skipped_good_kept(tmp_path: Path):
    path = tmp_path / "interests.json"
    payload = {
        "version": 1,
        "interests": [_sample_dict(), {"id": "broken"}],  # second missing required keys
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = InterestSet.load(path, default_path=DEFAULT_INTERESTS_PATH)
    topics = {i.topic for i in loaded.interests}
    assert "Test topic" in topics
    assert len(loaded.interests) == 1


# ---- InterestSet: helpers ----


def test_interestset_find_by_topic():
    s = InterestSet(interests=(Interest.from_dict(_sample_dict(topic="Rebecca")),))
    assert s.find_by_topic("Rebecca") is not None
    assert s.find_by_topic("rebecca") is not None  # case-insensitive
    assert s.find_by_topic("Unknown") is None


def test_interestset_bump_existing_topic():
    now = datetime.now(UTC)
    s = InterestSet(interests=(Interest.from_dict(_sample_dict(pull_score=6.0, feed_count=3)),))
    new_s = s.bump("Test topic", amount=0.5, now=now)
    bumped = new_s.find_by_topic("Test topic")
    assert bumped is not None
    assert bumped.pull_score == 6.5
    assert bumped.feed_count == 4
    assert bumped.last_fed == now


def test_interestset_bump_unknown_topic_returns_unchanged():
    s = InterestSet(interests=(Interest.from_dict(_sample_dict()),))
    new_s = s.bump("Unknown", amount=1.0, now=datetime.now(UTC))
    assert new_s == s


def test_interestset_list_eligible_respects_pull_threshold():
    now = datetime.now(UTC)
    low = Interest.from_dict(_sample_dict(id="low", topic="A", pull_score=5.0))
    high = Interest.from_dict(_sample_dict(id="high", topic="B", pull_score=7.0))
    s = InterestSet(interests=(low, high))
    eligible = s.list_eligible(pull_threshold=6.0, cooldown_hours=24.0, now=now)
    assert [i.id for i in eligible] == ["high"]


def test_interestset_list_eligible_respects_cooldown():
    now = datetime.now(UTC)
    recent = Interest.from_dict(
        _sample_dict(
            id="recent",
            topic="A",
            pull_score=7.0,
            last_researched_at=(now - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        )
    )
    old = Interest.from_dict(
        _sample_dict(
            id="old",
            topic="B",
            pull_score=7.0,
            last_researched_at=(now - timedelta(hours=30)).isoformat().replace("+00:00", "Z"),
        )
    )
    never = Interest.from_dict(_sample_dict(id="never", topic="C", pull_score=7.0))
    s = InterestSet(interests=(recent, old, never))
    eligible = s.list_eligible(pull_threshold=6.0, cooldown_hours=24.0, now=now)
    ids = [i.id for i in eligible]
    assert "recent" not in ids
    assert "old" in ids
    assert "never" in ids


def test_interestset_list_eligible_sorted_pull_desc_then_oldest_research():
    now = datetime.now(UTC)
    a = Interest.from_dict(
        _sample_dict(
            id="a",
            topic="A",
            pull_score=7.0,
            last_researched_at=(now - timedelta(hours=50)).isoformat().replace("+00:00", "Z"),
        )
    )
    b = Interest.from_dict(_sample_dict(id="b", topic="B", pull_score=8.0))
    c = Interest.from_dict(
        _sample_dict(
            id="c",
            topic="C",
            pull_score=7.0,
            last_researched_at=(now - timedelta(hours=100)).isoformat().replace("+00:00", "Z"),
        )
    )
    s = InterestSet(interests=(a, b, c))
    eligible = s.list_eligible(pull_threshold=6.0, cooldown_hours=24.0, now=now)
    # b first (highest pull), then c (older research than a), then a
    assert [i.id for i in eligible] == ["b", "c", "a"]
