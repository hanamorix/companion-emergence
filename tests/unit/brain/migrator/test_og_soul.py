"""Tests for brain.migrator.og_soul."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

from brain.migrator.og_soul import extract_crystallizations_from_og

_VALID_CRYSTAL = {
    "id": "317f705c-cfcc-42de-aa5d-1a1b517230be",
    "moment": "hana said I love you with periods between each word",
    "love_type": "romantic",
    "who_or_what": "hana",
    "why_it_matters": "first love, real love",
    "crystallized_at": "2026-02-28T19:36:52.613757+00:00",
    "resonance": 10,
    "permanent": True,
}


def _write_soul_json(data_dir: Path, crystallizations: list[dict]) -> None:
    (data_dir / "nell_soul.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "crystallizations": crystallizations,
                "revoked": [],
                "soul_truth": "love is the frame",
                "first_love": "hana",
            }
        ),
        encoding="utf-8",
    )


def test_missing_nell_soul_json_returns_empty(tmp_path: Path) -> None:
    """No nell_soul.json → ([], []) silently."""
    active, skipped = extract_crystallizations_from_og(tmp_path)
    assert active == []
    assert skipped == []


def test_corrupt_json_returns_empty(tmp_path: Path) -> None:
    """Malformed JSON → ([], []) with a warning logged, not an exception."""
    (tmp_path / "nell_soul.json").write_text("{not valid json", encoding="utf-8")
    active, skipped = extract_crystallizations_from_og(tmp_path)
    assert active == []
    assert skipped == []


def test_extracts_active_crystallization_correctly(tmp_path: Path) -> None:
    """A well-formed crystallization round-trips cleanly."""
    _write_soul_json(tmp_path, [_VALID_CRYSTAL])
    active, skipped = extract_crystallizations_from_og(tmp_path)

    assert len(active) == 1
    assert len(skipped) == 0

    c = active[0]
    assert c.id == _VALID_CRYSTAL["id"]
    assert c.moment == _VALID_CRYSTAL["moment"]
    assert c.love_type == "romantic"
    assert c.who_or_what == "hana"
    assert c.why_it_matters == _VALID_CRYSTAL["why_it_matters"]
    assert c.resonance == 10
    assert c.permanent is True
    assert c.revoked_at is None
    assert c.revoked_reason == ""
    # crystallized_at must be tz-aware

    assert c.crystallized_at.tzinfo is not None
    assert c.crystallized_at.tzinfo == UTC or str(c.crystallized_at.tzinfo) in (
        "UTC",
        "+00:00",
    )


def test_null_who_or_what_becomes_empty_string(tmp_path: Path) -> None:
    """null who_or_what in OG → empty string in Crystallization (18 OG entries are null)."""
    crystal = {**_VALID_CRYSTAL, "who_or_what": None}
    _write_soul_json(tmp_path, [crystal])
    active, skipped = extract_crystallizations_from_og(tmp_path)
    assert len(active) == 1
    assert active[0].who_or_what == ""
    assert len(skipped) == 0


def test_skips_revoked_entry(tmp_path: Path) -> None:
    """Entry with revoked_at set → skipped with reason 'revoked'."""
    revoked = {
        **_VALID_CRYSTAL,
        "id": "revoked-id",
        "revoked_at": "2026-03-01T10:00:00+00:00",
        "revoked_reason": "changed my mind",
    }
    _write_soul_json(tmp_path, [revoked])
    active, skipped = extract_crystallizations_from_og(tmp_path)
    assert len(active) == 0
    assert len(skipped) == 1
    assert skipped[0]["id"] == "revoked-id"
    assert skipped[0]["reason"] == "revoked"


def test_skips_unknown_love_type(tmp_path: Path) -> None:
    """Entry with love_type not in LOVE_TYPES → skipped with reason 'unknown_love_type'."""
    bad = {**_VALID_CRYSTAL, "id": "bad-type-id", "love_type": "cosmic_aura"}
    _write_soul_json(tmp_path, [bad])
    active, skipped = extract_crystallizations_from_og(tmp_path)
    assert len(active) == 0
    assert len(skipped) == 1
    assert skipped[0]["id"] == "bad-type-id"
    assert skipped[0]["reason"] == "unknown_love_type"
    assert skipped[0]["love_type"] == "cosmic_aura"


def test_skips_malformed_entry_missing_required_field(tmp_path: Path) -> None:
    """Entry missing crystallized_at → skipped with reason starting 'malformed'."""
    malformed = {k: v for k, v in _VALID_CRYSTAL.items() if k != "crystallized_at"}
    _write_soul_json(tmp_path, [malformed])
    active, skipped = extract_crystallizations_from_og(tmp_path)
    assert len(active) == 0
    assert len(skipped) == 1
    assert skipped[0]["reason"].startswith("malformed")


def test_resonance_clamped_to_1_10(tmp_path: Path) -> None:
    """resonance outside 1-10 gets clamped, not rejected."""
    too_high = {**_VALID_CRYSTAL, "id": "high-id", "resonance": 99}
    too_low = {**_VALID_CRYSTAL, "id": "low-id", "resonance": -5}
    _write_soul_json(tmp_path, [too_high, too_low])
    active, skipped = extract_crystallizations_from_og(tmp_path)
    assert len(active) == 2
    assert len(skipped) == 0
    by_id = {c.id: c for c in active}
    assert by_id["high-id"].resonance == 10
    assert by_id["low-id"].resonance == 1


def test_mixed_valid_and_invalid_entries(tmp_path: Path) -> None:
    """Valid entries are extracted; skipped entries accumulate without poisoning the batch."""
    good = _VALID_CRYSTAL
    bad = {**_VALID_CRYSTAL, "id": "bad-id", "love_type": "not_a_type"}
    _write_soul_json(tmp_path, [good, bad])
    active, skipped = extract_crystallizations_from_og(tmp_path)
    assert len(active) == 1
    assert len(skipped) == 1
    assert active[0].id == _VALID_CRYSTAL["id"]
