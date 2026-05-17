"""brain.growth.crystallizers.creative_dna — tests for evolution mechanism."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.chat import ChatResponse
from brain.bridge.provider import LLMProvider
from brain.creative.dna import load_creative_dna, save_creative_dna
from brain.growth.crystallizers.creative_dna import (
    CreativeDnaCrystallizationResult,
    _gather_recent_fiction,
    crystallize_creative_dna,
)
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


class _FakeProvider(LLMProvider):
    def __init__(self, response: str):
        self._response = response

    def name(self) -> str:
        return "fake-creative-dna"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self._response

    def chat(self, messages, *, tools=None, options=None):
        return ChatResponse(content=self._response, tool_calls=[])


@pytest.fixture
def persona_dir(tmp_path: Path):
    p = tmp_path / "p"
    p.mkdir()
    (p / "active_conversations").mkdir()
    return p


@pytest.fixture
def store(persona_dir):
    s = MemoryStore(persona_dir / "memories.db")
    yield s
    s.close()


@pytest.fixture
def hebbian(persona_dir):
    h = HebbianMatrix(persona_dir / "hebbian.db")
    yield h
    h.close()


def test_happy_path_emerging_addition(persona_dir, store, hebbian):
    """LLM proposes one emerging addition; passes all gates; persists."""
    response = json.dumps(
        {
            "emerging_additions": [
                {
                    "name": "intentional sentence fragments",
                    "reasoning": "appeared in 3 recent fiction sessions distinct from previous patterns",
                    "evidence_memory_ids": ["mem_a", "mem_b", "mem_c"],
                }
            ],
            "emerging_promotions": [],
            "active_demotions": [],
        }
    )
    provider = _FakeProvider(response)
    result = crystallize_creative_dna(
        store=store,
        persona_dir=persona_dir,
        provider=provider,
        persona_name="testpersona",
        now=datetime.now(UTC),
    )
    assert isinstance(result, CreativeDnaCrystallizationResult)
    assert len(result.emerging_additions) == 1
    assert result.emerging_additions[0]["name"] == "intentional sentence fragments"

    # File updated
    dna = load_creative_dna(persona_dir)
    emerging_names = [t["name"] for t in dna["tendencies"]["emerging"]]
    assert "intentional sentence fragments" in emerging_names


def test_gather_recent_fiction_includes_production_conversation_prose(persona_dir, store):
    """Creative DNA scans extracted chat memories, not legacy conversation type only."""
    prose = (
        '"Come closer," she said. '
        + "The room answered with a hush. " * 80
        + "\n\nThe sentence kept unfolding into another sentence."
    )
    memory = Memory.create_new(
        content=prose,
        memory_type="note",
        domain="brain",
        tags=["auto_ingest", "conversation", "note"],
        metadata={"source_summary": "conversation:sess-fiction"},
    )
    store.create(memory)

    corpus = _gather_recent_fiction(
        store,
        cutoff=datetime.now(UTC) - timedelta(days=1),
    )

    assert any(
        item["memory_id"] == memory.id and item["type"] == "conversation_prose" for item in corpus
    )


def test_returns_empty_on_provider_error(persona_dir, store, hebbian):
    class _Boom(LLMProvider):
        def name(self):
            return "boom"

        def generate(self, prompt, *, system=None):
            from brain.bridge.provider import ProviderError

            raise ProviderError("test", "simulated")

        def chat(self, messages, *, tools=None, options=None):
            from brain.bridge.provider import ProviderError

            raise ProviderError("test", "simulated")

    result = crystallize_creative_dna(
        store=store,
        persona_dir=persona_dir,
        provider=_Boom(),
        persona_name="t",
        now=datetime.now(UTC),
    )
    assert result == CreativeDnaCrystallizationResult([], [], [])


def test_returns_empty_on_malformed_json(persona_dir, store, hebbian):
    provider = _FakeProvider("not valid json prose response")
    result = crystallize_creative_dna(
        store=store,
        persona_dir=persona_dir,
        provider=provider,
        persona_name="t",
        now=datetime.now(UTC),
    )
    assert result == CreativeDnaCrystallizationResult([], [], [])


def test_gate_1_invalid_name_rejected(persona_dir, store, hebbian):
    response = json.dumps(
        {
            "emerging_additions": [
                {
                    "name": "../../etc/passwd",
                    "reasoning": "valid reasoning but invalid name",
                    "evidence_memory_ids": [],
                }
            ],
            "emerging_promotions": [],
            "active_demotions": [],
        }
    )
    result = crystallize_creative_dna(
        store=store,
        persona_dir=persona_dir,
        provider=_FakeProvider(response),
        persona_name="t",
        now=datetime.now(UTC),
    )
    assert result.emerging_additions == []


def test_gate_4_short_reasoning_rejected(persona_dir, store, hebbian):
    """Reasoning < 20 chars after strip → rejected."""
    response = json.dumps(
        {
            "emerging_additions": [
                {
                    "name": "valid pattern",
                    "reasoning": "too short",
                    "evidence_memory_ids": [],
                }
            ],
            "emerging_promotions": [],
            "active_demotions": [],
        }
    )
    result = crystallize_creative_dna(
        store=store,
        persona_dir=persona_dir,
        provider=_FakeProvider(response),
        persona_name="t",
        now=datetime.now(UTC),
    )
    assert result.emerging_additions == []


def test_gate_6_total_cap_3(persona_dir, store, hebbian):
    """LLM proposes 5 changes; only first 3 accepted."""
    response = json.dumps(
        {
            "emerging_additions": [
                {
                    "name": f"pattern {i}",
                    "reasoning": f"reasoning long enough for gate 4 here {i}",
                    "evidence_memory_ids": [],
                }
                for i in range(5)
            ],
            "emerging_promotions": [],
            "active_demotions": [],
        }
    )
    result = crystallize_creative_dna(
        store=store,
        persona_dir=persona_dir,
        provider=_FakeProvider(response),
        persona_name="t",
        now=datetime.now(UTC),
    )
    total = (
        len(result.emerging_additions)
        + len(result.emerging_promotions)
        + len(result.active_demotions)
    )
    assert total <= 3


def test_gate_5_emerging_promotion_must_exist(persona_dir, store, hebbian):
    """Promote a name not in current emerging → rejected."""
    save_creative_dna(
        persona_dir,
        {
            "version": 1,
            "core_voice": "v",
            "strengths": [],
            "tendencies": {"active": [], "emerging": [], "fading": []},
            "influences": [],
            "avoid": [],
        },
    )
    response = json.dumps(
        {
            "emerging_additions": [],
            "emerging_promotions": [
                {
                    "name": "nonexistent",
                    "reasoning": "reasoning long enough for the gate-4 length check",
                }
            ],
            "active_demotions": [],
        }
    )
    result = crystallize_creative_dna(
        store=store,
        persona_dir=persona_dir,
        provider=_FakeProvider(response),
        persona_name="t",
        now=datetime.now(UTC),
    )
    assert result.emerging_promotions == []


def test_behavioral_log_entry_written_on_acceptance(persona_dir, store, hebbian):
    from brain.behavioral.log import read_behavioral_log

    response = json.dumps(
        {
            "emerging_additions": [
                {
                    "name": "valid emerging name",
                    "reasoning": "this reasoning is definitely longer than twenty chars",
                    "evidence_memory_ids": ["mem_a"],
                }
            ],
            "emerging_promotions": [],
            "active_demotions": [],
        }
    )
    crystallize_creative_dna(
        store=store,
        persona_dir=persona_dir,
        provider=_FakeProvider(response),
        persona_name="t",
        now=datetime.now(UTC),
    )
    log = read_behavioral_log(persona_dir / "behavioral_log.jsonl")
    assert len(log) == 1
    assert log[0]["kind"] == "creative_dna_emerging_added"
    assert log[0]["name"] == "valid emerging name"


def test_creative_dna_crystallization_emits_initiate_candidate(
    persona_dir,
    store,
    hebbian,
):
    """After an accepted creative_dna change commits, emit a candidate."""
    from brain.initiate.emit import read_candidates

    response = json.dumps(
        {
            "emerging_additions": [
                {
                    "name": "intentional sentence fragments",
                    "reasoning": "appeared in 3 recent fiction sessions distinct from previous patterns",
                    "evidence_memory_ids": ["mem_a", "mem_b", "mem_c"],
                }
            ],
            "emerging_promotions": [],
            "active_demotions": [],
        }
    )
    crystallize_creative_dna(
        store=store,
        persona_dir=persona_dir,
        provider=_FakeProvider(response),
        persona_name="t",
        now=datetime.now(UTC),
    )

    candidates = read_candidates(persona_dir)
    assert any(
        c.source == "crystallization"
        and c.source_id == "creative_dna_addition:intentional sentence fragments"
        for c in candidates
    )
