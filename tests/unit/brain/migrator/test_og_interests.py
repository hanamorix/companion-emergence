"""Tests for brain.migrator.og_interests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.migrator.og_interests import extract_interests_from_og


def _write_og_interests(path: Path, interests: list[dict]) -> None:
    path.write_text(json.dumps({"version": "1.0", "interests": interests}), encoding="utf-8")


def _og_interest(**overrides) -> dict:
    base = {
        "id": "abc-123",
        "topic": "Lispector diagonal syntax",
        "pull_score": 7.2,
        "first_seen": "2026-03-29T16:42:33.435028+00:00",
        "last_fed": "2026-03-31T11:37:13.729750+00:00",
        "feed_count": 5,
        "source_types": ["dream", "heartbeat"],
        "related_keywords": ["lispector", "syntax", "language", "clarice"],
        "notes": "sideways through meaning",
    }
    base.update(overrides)
    return base


def test_extract_interests_simple(tmp_path: Path):
    path = tmp_path / "nell_interests.json"
    _write_og_interests(path, [_og_interest()])
    out = extract_interests_from_og(path, soul_names=set())
    assert len(out) == 1
    item = out[0]
    assert item["topic"] == "Lispector diagonal syntax"
    assert item["scope"] == "either"  # no soul match → default
    assert item["last_researched_at"] is None
    assert item["pull_score"] == 7.2


def test_extract_interests_scope_classification_from_soul(tmp_path: Path):
    path = tmp_path / "nell_interests.json"
    _write_og_interests(
        path,
        [
            _og_interest(id="a", topic="Lispector diagonal syntax"),
            _og_interest(id="b", topic="Hana"),
        ],
    )
    out = extract_interests_from_og(path, soul_names={"hana"})
    scopes = {i["topic"]: i["scope"] for i in out}
    assert scopes["Hana"] == "internal"
    assert scopes["Lispector diagonal syntax"] == "either"


def test_extract_interests_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_interests_from_og(tmp_path / "missing.json", soul_names=set())


def test_extract_interests_corrupt_json_raises(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_interests_from_og(path, soul_names=set())
