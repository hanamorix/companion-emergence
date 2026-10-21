"""#523: the extraction backfill (`nell memory backfill-extraction`) re-extracts conversation text the old tail cut never
showed the extractor. Synthetic persona only (persona Kestrel, user Bob, made-up words). Criteria B1-B16 of the #523 run.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from brain.engines.consolidation import Decision, run_consolidation
from brain.ingest.commit import commit_item
from brain.ingest.types import ExtractedItem
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

_PROMOTE_ALL = lambda _cand, _ctx: Decision("new")  # noqa: E731


@pytest.fixture
def stores(tmp_path):
    persona = tmp_path / "Kestrel"
    persona.mkdir()
    store = MemoryStore(persona / "memories.db")
    hebbian = HebbianMatrix(persona / "hebbian.db")
    yield persona, store, hebbian
    store.close()
    hebbian.close()


# ---------------------------------------------------------------------------
# commit_item: the back-date and the marker survive the pending queue and the gate (B9/B15 plumbing)
# ---------------------------------------------------------------------------


def test_commit_item_back_dates_and_marks_through_the_gate(stores) -> None:
    persona, store, hebbian = stores
    when = datetime(2026, 7, 3, 21, 14, 5, tzinfo=UTC)
    marker = {"extraction_backfill": {"from": "2026-07-03T21:00:00+00:00", "to": "2026-07-03T21:14:05+00:00"}}
    item = ExtractedItem(text="Bob keeps a jar of snorvel buttons by the door", label="fact", importance=6)
    mem_id = commit_item(item, session_id="sess_old", store=store, hebbian=hebbian, extra_metadata=marker, created_at=when)
    assert mem_id is not None
    run_consolidation(store, persona_dir=persona, classifier=_PROMOTE_ALL)
    mem = store.get(mem_id)
    assert mem is not None
    assert mem.created_at == when
    assert mem.metadata["extraction_backfill"] == marker["extraction_backfill"]
    assert mem.metadata["source_summary"] == "conversation:sess_old"


def test_commit_item_without_the_new_arguments_dates_now(stores) -> None:
    persona, store, hebbian = stores
    before = datetime.now(UTC)
    item = ExtractedItem(text="Bob named the kettle Plonk", label="fact", importance=6)
    mem_id = commit_item(item, session_id="sess_new", store=store, hebbian=hebbian)
    run_consolidation(store, persona_dir=persona, classifier=_PROMOTE_ALL)
    mem = store.get(mem_id)
    assert mem is not None and mem.created_at >= before.replace(microsecond=0)
    assert "extraction_backfill" not in mem.metadata
