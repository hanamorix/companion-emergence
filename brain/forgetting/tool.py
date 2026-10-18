"""recall_forgotten MCP tool impl per spec §5.

Read-only. Searches the graveyard JSONL only (not active memory).
Used when Nell wants to consciously reach back: 'what did I used to
know about X?'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from brain.forgetting import graveyard
from brain.memory.relevance import names_in


def recall_forgotten(*, arguments: dict[str, Any], persona_dir: Path) -> dict[str, Any]:
    query = arguments.get("query")
    if not query or not isinstance(query, str):
        raise ValueError("recall_forgotten requires a non-empty 'query' string")
    # The listed names in the query are searched too, and their hits lead (S119, S120).
    hits = graveyard.search(persona_dir, query, limit=5, names=names_in(persona_dir, query))
    return {"hits": hits}
