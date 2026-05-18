"""Tests for brain/forgetting/tool.py — recall_forgotten MCP tool."""

import pytest

from brain.forgetting import graveyard
from brain.forgetting.salience import SalienceInputs
from brain.forgetting.tool import recall_forgotten
from brain.memory.store import Memory


def _make_memory(content) -> Memory:
    return Memory.create_new(content=content, memory_type="episodic", domain="chat", emotions={})


def test_recall_forgotten_returns_graveyard_hits(tmp_path):
    graveyard.append(
        tmp_path,
        memory=_make_memory("apple banana"),
        salience_at_drop=0.05,
        inputs=SalienceInputs(emotion=0, hebbian=0, recall=0, soul=0, freshness=0),
        lived_age_hours=100.0,
        reason="x",
    )
    result = recall_forgotten(arguments={"query": "apple"}, persona_dir=tmp_path)
    assert "hits" in result
    assert len(result["hits"]) == 1
    assert result["hits"][0]["summary"] == "apple banana"


def test_recall_forgotten_empty_when_no_match(tmp_path):
    result = recall_forgotten(arguments={"query": "nothing"}, persona_dir=tmp_path)
    assert result["hits"] == []


def test_recall_forgotten_rejects_missing_query(tmp_path):
    with pytest.raises(ValueError):
        recall_forgotten(arguments={}, persona_dir=tmp_path)
