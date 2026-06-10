"""Test that build_attunement_block caps rendered patterns at _ATTUNEMENT_RENDER_CAP."""
from pathlib import Path

from brain.attunement.ambient import _ATTUNEMENT_RENDER_CAP, build_attunement_block
from brain.attunement.schemas import SCHEMA_VERSION, LearnedPattern
from brain.attunement.store import _append_pattern


def _make_pattern(idx: int, maturity: str, evidence_count: int) -> LearnedPattern:
    """Build a LearnedPattern with the given maturity and evidence count."""
    return LearnedPattern(
        id=f"id-{maturity}-{idx:02d}",
        category="tone",
        canonical_key=f"key-{maturity}-{idx:02d}",
        description=f"{maturity} pattern {idx:02d}",
        evidence_count=evidence_count,
        maturity=maturity,
        first_seen_at="2026-04-01T00:00:00Z",
        last_confirmed_at="2026-05-31T12:00:00Z",
        last_addressed_at=None,
        crystallised_at="2026-05-15T00:00:00Z" if maturity == "known" else None,
        falsified_at=None,
        examples=[],
        schema_version=SCHEMA_VERSION,
    )


def _count_bullet_lines(block: str) -> int:
    """Count lines that start with '- ' (rendered pattern bullets)."""
    return sum(1 for line in block.splitlines() if line.startswith("- "))


def test_render_cap_limits_to_top_n(tmp_path: Path):
    """More than _ATTUNEMENT_RENDER_CAP patterns → only _ATTUNEMENT_RENDER_CAP rendered."""
    # Seed 5 known + 6 forming = 11 surfaceable patterns (> cap of 8)
    for i in range(5):
        _append_pattern(tmp_path, _make_pattern(i, "known", evidence_count=12))
    for i in range(6):
        _append_pattern(tmp_path, _make_pattern(i, "forming", evidence_count=4))

    block = build_attunement_block(tmp_path)
    assert _count_bullet_lines(block) == _ATTUNEMENT_RENDER_CAP


def test_known_patterns_appear_before_forming(tmp_path: Path):
    """known patterns must be ranked above forming patterns in the rendered output."""
    # Seed exactly cap worth: 4 known + 4 forming (exactly 8)
    for i in range(4):
        _append_pattern(tmp_path, _make_pattern(i, "known", evidence_count=10))
    for i in range(4):
        _append_pattern(tmp_path, _make_pattern(i, "forming", evidence_count=3))

    block = build_attunement_block(tmp_path)
    bullet_lines = [line for line in block.splitlines() if line.startswith("- ")]

    # known bullets are rendered WITHOUT "she seems to"; forming WITH "she seems to"
    known_indices = [j for j, ln in enumerate(bullet_lines) if "she seems to" not in ln.lower()]
    forming_indices = [j for j, ln in enumerate(bullet_lines) if "she seems to" in ln.lower()]

    assert known_indices, "no known-pattern bullets found"
    assert forming_indices, "no forming-pattern bullets found"
    assert max(known_indices) < min(forming_indices), (
        "all known bullets must precede all forming bullets"
    )


def test_fewer_than_cap_all_rendered(tmp_path: Path):
    """When surfaceable count <= cap, all patterns are rendered."""
    for i in range(3):
        _append_pattern(tmp_path, _make_pattern(i, "known", evidence_count=10))
    for i in range(2):
        _append_pattern(tmp_path, _make_pattern(i, "forming", evidence_count=3))

    block = build_attunement_block(tmp_path)
    assert _count_bullet_lines(block) == 5
