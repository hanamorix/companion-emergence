"""Tests for brain.growth.scheduler — orchestrator + atomic apply."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from brain.growth.proposal import EmotionProposal
from brain.growth.scheduler import GrowthTickResult, run_growth_tick
from brain.memory.store import MemoryStore


def _seed_vocab(persona_dir: Path, names: list[str] | None = None) -> None:
    """Seed an emotion_vocabulary.json with the given names as core emotions."""
    if names is None:
        names = ["love", "joy"]
    entries = [
        {
            "name": n,
            "description": f"the feeling of {n}",
            "category": "core",
            "decay_half_life_days": 7.0,
        }
        for n in names
    ]
    (persona_dir / "emotion_vocabulary.json").write_text(
        json.dumps({"version": 1, "emotions": entries}, indent=2),
        encoding="utf-8",
    )


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    pdir = tmp_path / "persona"
    pdir.mkdir()
    return pdir


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


def test_run_growth_tick_no_proposals_returns_zero(persona_dir: Path, store: MemoryStore) -> None:
    """Phase 2a's real crystallizer returns [] — scheduler returns count=0."""
    _seed_vocab(persona_dir)
    result = run_growth_tick(persona_dir, store, datetime.now(UTC))
    assert isinstance(result, GrowthTickResult)
    assert result.emotions_added == 0
    assert result.proposals_seen == 0
    # No log file created when nothing happened
    assert not (persona_dir / "emotion_growth.log.jsonl").exists()


def test_run_growth_tick_applies_proposal_atomically(
    persona_dir: Path, store: MemoryStore
) -> None:
    """When a crystallizer returns a proposal, vocabulary + log update together."""
    _seed_vocab(persona_dir)
    proposal = EmotionProposal(
        name="lingering",
        description="the slow trail of warmth after a loved person leaves the room",
        decay_half_life_days=7.0,
        evidence_memory_ids=("mem_a", "mem_b"),
        score=0.78,
        relational_context="recurred during Hana's tender messages",
    )

    with patch(
        "brain.growth.scheduler.crystallize_vocabulary",
        return_value=[proposal],
    ):
        result = run_growth_tick(persona_dir, store, datetime.now(UTC))

    assert result.emotions_added == 1
    assert result.proposals_seen == 1
    assert result.proposals_rejected == 0

    # Vocabulary file updated
    vocab = json.loads((persona_dir / "emotion_vocabulary.json").read_text(encoding="utf-8"))
    names = [e["name"] for e in vocab["emotions"]]
    assert "lingering" in names
    new_entry = next(e for e in vocab["emotions"] if e["name"] == "lingering")
    assert new_entry["category"] == "persona_extension"
    assert new_entry["decay_half_life_days"] == 7.0

    # Growth log updated
    log_path = persona_dir / "emotion_growth.log.jsonl"
    assert log_path.exists()
    [line] = log_path.read_text(encoding="utf-8").splitlines()
    parsed = json.loads(line)
    assert parsed["type"] == "emotion_added"
    assert parsed["name"] == "lingering"
    assert parsed["relational_context"] == "recurred during Hana's tender messages"


def test_run_growth_tick_skips_proposal_with_existing_name(
    persona_dir: Path, store: MemoryStore
) -> None:
    """Idempotent: a proposal whose name already exists in the vocabulary is skipped silently."""
    _seed_vocab(persona_dir, names=["love", "joy"])
    proposal = EmotionProposal(
        name="love",  # already in vocab
        description="dup",
        decay_half_life_days=None,
        evidence_memory_ids=(),
        score=0.5,
        relational_context=None,
    )
    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[proposal]):
        result = run_growth_tick(persona_dir, store, datetime.now(UTC))
    assert result.emotions_added == 0
    assert result.proposals_seen == 1
    # Skipped duplicates aren't counted as rejections — re-proposing is normal.
    assert result.proposals_rejected == 0
    assert not (persona_dir / "emotion_growth.log.jsonl").exists()


def test_run_growth_tick_rejects_proposal_with_invalid_chars(
    persona_dir: Path, store: MemoryStore
) -> None:
    """Names containing path-traversal chars or curly braces are rejected as schema-invalid."""
    _seed_vocab(persona_dir)
    bad_names = ["bad/name", "bad\\name", "bad{name}", ""]
    for bn in bad_names:
        proposal = EmotionProposal(
            name=bn,
            description="x",
            decay_half_life_days=None,
            evidence_memory_ids=(),
            score=0.5,
            relational_context=None,
        )
        with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[proposal]):
            result = run_growth_tick(persona_dir, store, datetime.now(UTC))
        assert result.emotions_added == 0
        assert result.proposals_rejected == 1


def test_run_growth_tick_dry_run_does_not_write(
    persona_dir: Path, store: MemoryStore
) -> None:
    """dry_run=True calls crystallizer but writes neither vocabulary nor log."""
    _seed_vocab(persona_dir)
    proposal = EmotionProposal(
        name="lingering",
        description="x",
        decay_half_life_days=None,
        evidence_memory_ids=(),
        score=0.5,
        relational_context=None,
    )
    vocab_before = (persona_dir / "emotion_vocabulary.json").read_text(encoding="utf-8")
    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[proposal]):
        result = run_growth_tick(persona_dir, store, datetime.now(UTC), dry_run=True)
    assert result.emotions_added == 1  # would-have-added semantics
    # Files unchanged
    assert (persona_dir / "emotion_vocabulary.json").read_text(encoding="utf-8") == vocab_before
    assert not (persona_dir / "emotion_growth.log.jsonl").exists()


def test_run_growth_tick_handles_multiple_proposals(
    persona_dir: Path, store: MemoryStore
) -> None:
    _seed_vocab(persona_dir)
    proposals = [
        EmotionProposal(
            name=f"p{i}",
            description=f"desc {i}",
            decay_half_life_days=float(i),
            evidence_memory_ids=(),
            score=0.5,
            relational_context=None,
        )
        for i in range(3)
    ]
    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=proposals):
        result = run_growth_tick(persona_dir, store, datetime.now(UTC))
    assert result.emotions_added == 3
    log_path = persona_dir / "emotion_growth.log.jsonl"
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 3
