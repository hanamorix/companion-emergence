"""#456 C51: a stop during a heartbeat's consolidation stage, on the merged base (#470 in).

The stop arrives while the consolidation pass is classifying (the provider sees the process-wide stop flag and refuses
the next call with ShutdownDeferred, as the real provider does). Then: no candidate is promoted unclassified, none is
lost (every drained candidate is settled or still in #470's processing file), and the next run classifies the
unsettled remainder before anything queued later. Synthetic content only; a tmp persona with a real MemoryStore and
PendingQueue.
"""

from __future__ import annotations

import pytest

from brain.bridge import cli_throttle
from brain.bridge.cli_throttle import ShutdownDeferred
from brain.engines.consolidation import Reappraisal, run_consolidation
from brain.health.json_io import read_jsonl
from brain.memory.hebbian import HebbianMatrix
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore

_PROCESSING = "pending_candidates.processing.jsonl"
_SETTLED = "pending_candidates.settled.jsonl"

A, B, C, D = "alpha memory", "bravo memory", "charlie memory", "delta memory"


@pytest.fixture
def persona(tmp_path):
    cli_throttle.reset()
    store = MemoryStore(tmp_path / "memories.db")
    heb = HebbianMatrix(tmp_path / "hebbian.db")
    yield tmp_path, store, heb, PendingQueue(tmp_path)
    store.close()
    heb.close()
    cli_throttle.reset()


class _StopAwareProvider:
    """Classifies "new"; the stop begins during call number ``stop_during``; every call after it is refused."""

    def __init__(self, stop_during: int | None):
        self.stop_during = stop_during
        self.calls: list[str] = []

    def generate(self, prompt, *, system=None):
        if cli_throttle.shutdown_requested():
            raise ShutdownDeferred("the bridge is stopping")
        self.calls.append(prompt)
        if self.stop_during is not None and len(self.calls) == self.stop_during:
            cli_throttle.request_shutdown()
        return '{"verdict": "new"}'


def _reappraiser(_m):
    return Reappraisal(importance=9.0)


def _enqueue(queue, *contents) -> None:
    for c in contents:
        queue.enqueue(Memory.create_new(content=c, memory_type="dream", domain="us", emotions={}), source="dream")


def _contents(store) -> list[str]:
    return sorted(r[0] for r in store._conn.execute("SELECT content FROM memories"))


def _queued(queue) -> list[str]:
    return [r["content"] for r in read_jsonl(queue.path)] if queue.path.exists() else []


def _unsettled(tmp) -> list[str]:
    p, s = tmp / _PROCESSING, tmp / _SETTLED
    if not p.exists():
        return []
    done = {r["id"] for r in read_jsonl(s)} if s.exists() else set()
    return [e.get("content") for e in read_jsonl(p) if e["_claim_id"] not in done]


def _which(prompt: str) -> str:
    """The candidate a classifier prompt asks about (its <candidate> fence; the context memories follow it)."""
    head = prompt[prompt.index("<candidate>"): prompt.index("</candidate>")]
    return next(c for c in (A, B, C, D) if c in head)


def test_a_stop_mid_consolidation_promotes_nothing_unclassified_loses_nothing_and_resumes_leftovers_first(persona):
    tmp, store, heb, queue = persona
    _enqueue(queue, A, B, C)
    provider = _StopAwareProvider(stop_during=1)

    run_consolidation(store, persona_dir=tmp, provider=provider, hebbian=heb, reappraiser=_reappraiser)

    assert [_which(p) for p in provider.calls] == [A]
    assert _contents(store) == [A]  # B and C were never classified: not promoted
    assert sorted(_unsettled(tmp) + _queued(queue)) == sorted([B, C])  # nothing lost

    # A later candidate arrives; the next start (flag clear) runs consolidation again.
    _enqueue(queue, D)
    cli_throttle.reset()
    after = _StopAwareProvider(stop_during=None)
    run_consolidation(store, persona_dir=tmp, provider=after, hebbian=heb, reappraiser=_reappraiser)

    order = [_which(p) for p in after.calls]
    assert sorted(order[:2]) == sorted([B, C]) and order[2:] == [D]  # the unsettled remainder first
    assert _contents(store) == sorted([A, B, C, D])
    assert _unsettled(tmp) == [] and _queued(queue) == []
