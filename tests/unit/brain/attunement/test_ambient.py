"""Tests for attunement ambient block rendering."""
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.attunement.ambient import build_attunement_block
from brain.attunement.schemas import SCHEMA_VERSION, CurrentRead, LearnedPattern
from brain.attunement.store import _append_pattern, write_current_read


def _now_iso():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_read():
    return CurrentRead(
        ts=_now_iso(),
        source_turn_id="t1",
        tone_label="warm",
        tone_justification="soft phrasing",
        cadence_label="measured",
        cadence_justification="full sentences",
        mood_valence=0.3,
        mood_intensity=0.5,
        predicted_arc_shape="settling in for a long evening",
        schema_version=SCHEMA_VERSION,
    )


def _make_pattern(id_: str, maturity: str, last_addressed_at=None, description="some pattern"):
    return LearnedPattern(
        id=id_,
        category="tone",
        canonical_key=f"key-{id_}",
        description=description,
        evidence_count=12 if maturity == "known" else 5,
        maturity=maturity,
        first_seen_at="2026-04-01T00:00:00Z",
        last_confirmed_at="2026-05-31T12:00:00Z",
        last_addressed_at=last_addressed_at,
        crystallised_at="2026-05-15T00:00:00Z" if maturity == "known" else None,
        falsified_at=None,
        examples=[],
        schema_version=SCHEMA_VERSION,
    )


def test_block_includes_current_read_paragraph(tmp_path: Path):
    write_current_read(tmp_path, _make_read())
    block = build_attunement_block(tmp_path)
    assert "warm" in block
    assert "measured" in block
    assert "settling in for a long evening" in block


def test_block_renders_known_patterns_confidently(tmp_path: Path):
    write_current_read(tmp_path, _make_read())
    _append_pattern(tmp_path, _make_pattern("p1", "known", description="she softens about the dog"))
    block = build_attunement_block(tmp_path)
    assert "she softens about the dog" in block
    # No hedging
    assert "you seem to" not in block.lower()


def test_block_renders_forming_patterns_hedged(tmp_path: Path):
    write_current_read(tmp_path, _make_read())
    _append_pattern(tmp_path, _make_pattern("p1", "forming", description="goes terse when tired"))
    block = build_attunement_block(tmp_path)
    assert "goes terse when tired" in block
    assert "you seem to" in block.lower() or "it feels like" in block.lower()


def test_block_hides_recently_addressed_pattern(tmp_path: Path):
    write_current_read(tmp_path, _make_read())
    # Addressed 2 hours ago — within 6h cooldown
    recent = (datetime.now(UTC) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _append_pattern(tmp_path, _make_pattern("p1", "known", last_addressed_at=recent, description="DOG_PATTERN"))
    block = build_attunement_block(tmp_path)
    assert "DOG_PATTERN" not in block


def test_block_includes_addressed_pattern_after_cooldown(tmp_path: Path):
    write_current_read(tmp_path, _make_read())
    # Addressed 7 hours ago — past 6h cooldown
    old = (datetime.now(UTC) - timedelta(hours=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _append_pattern(tmp_path, _make_pattern("p1", "known", last_addressed_at=old, description="DOG_PATTERN"))
    block = build_attunement_block(tmp_path)
    assert "DOG_PATTERN" in block


def test_block_skips_immature_and_falsified_patterns(tmp_path: Path):
    write_current_read(tmp_path, _make_read())
    _append_pattern(tmp_path, _make_pattern("p1", "immature", description="IMMATURE_DESC"))
    _append_pattern(tmp_path, _make_pattern("p2", "falsified", description="FALSIFIED_DESC"))
    block = build_attunement_block(tmp_path)
    assert "IMMATURE_DESC" not in block
    assert "FALSIFIED_DESC" not in block


def test_block_empty_when_no_state(tmp_path: Path):
    block = build_attunement_block(tmp_path)
    assert block.strip() == ""


def test_block_alpha_1_does_not_include_addressability_directive(tmp_path: Path):
    """Addressability directive ships in v0.0.28 final, not alpha.1."""
    write_current_read(tmp_path, _make_read())
    block = build_attunement_block(tmp_path)
    assert "Don't force it" not in block
    assert "load-bearing" not in block.lower()
