"""record_monologue → Tier 2 trace memory + Tier 3 gated digest, in one call,
via the dispatch entry point (the path the tool loop uses)."""
import json

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools.dispatch import dispatch


def test_record_monologue_writes_trace_and_gated_digest(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    out = dispatch(
        "record_monologue",
        {"monologue": "the raw first-person drift", "feed_digest": "she drifted", "surface": False},
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
    )
    assert out["ok"] is True
    assert out["monologue_text"] == "the raw first-person drift"
    # Tier 2:
    traces = store.list_by_type("monologue_trace")
    assert len(traces) == 1 and traces[0].content == "the raw first-person drift"
    # Tier 3 (withheld):
    line = json.loads((tmp_path / "monologue_digest.jsonl").read_text().splitlines()[0])
    assert line["surfaced"] is False
