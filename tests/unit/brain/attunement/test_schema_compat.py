"""Forward/backward schema-compat for learned_patterns across versions."""
import json
from pathlib import Path

from brain.attunement.store import read_learned_patterns


def test_reads_alpha1_tone_cadence_file(tmp_path: Path):
    """A v0.0.29 reader reads an alpha.1 (tone/cadence) learned_patterns file cleanly."""
    d = tmp_path / "attunement"
    d.mkdir(parents=True)
    (d / "learned_patterns.jsonl").write_text(json.dumps({
        "id": "abc", "category": "tone", "canonical_key": "warm", "description": "warm",
        "evidence_count": 4, "maturity": "forming", "first_seen_at": "x", "last_confirmed_at": "x",
        "last_addressed_at": None, "crystallised_at": None, "falsified_at": None,
        "examples": ["hi"], "schema_version": "0.0.28-alpha.1",
    }) + "\n")
    patterns = read_learned_patterns(tmp_path)
    assert len(patterns) == 1 and patterns[0].category == "tone"


def test_unknown_category_row_skipped_not_raised(tmp_path: Path):
    """A row with a category this reader doesn't know is skipped, file stays valid."""
    d = tmp_path / "attunement"
    d.mkdir(parents=True)
    (d / "learned_patterns.jsonl").write_text(json.dumps({
        "id": "z", "category": "future_category", "canonical_key": "k", "description": "d",
        "evidence_count": 1, "maturity": "immature", "first_seen_at": "x", "last_confirmed_at": "x",
        "last_addressed_at": None, "crystallised_at": None, "falsified_at": None,
        "examples": [], "schema_version": "9.9.9",
    }) + "\n")
    # read_learned_patterns swallows the ValueError from LearnedPattern.__post_init__
    assert read_learned_patterns(tmp_path) == []
