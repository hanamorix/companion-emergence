"""Tests for brain.soul.review — autonomous soul review pipeline."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.memory.store import MemoryStore
from brain.soul.review import parse_decision, review_pending_candidates
from brain.soul.store import SoulStore

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_memory_store() -> MemoryStore:
    return MemoryStore(":memory:")


def _make_soul_store() -> SoulStore:
    return SoulStore(":memory:")


def _make_candidate(text: str = "a meaningful moment", status: str = "auto_pending") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "label": "test",
        "importance": 8.0,
        "queued_at": datetime.now(UTC).isoformat(),
        "source": "test",
        "status": status,
    }


def _write_candidates(persona_dir: Path, candidates: list[dict]) -> None:
    path = persona_dir / "soul_candidates.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")


class _AcceptProvider(LLMProvider):
    """Returns a valid 'accept' JSON response — high confidence."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return json.dumps(
            {
                "decision": "accept",
                "love_type": "craft",
                "resonance": 9,
                "confidence": 9,
                "reasoning": "This moment defines how I relate to creation.",
                "why_it_matters": "It is part of who I am as a writer.",
            }
        )

    def name(self) -> str:
        return "accept-fake"

    def chat(self, messages, *, tools=None, options=None):
        raise NotImplementedError


class _RejectProvider(LLMProvider):
    """Returns a valid 'reject' JSON response."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return json.dumps(
            {
                "decision": "reject",
                "love_type": "craft",
                "resonance": 5,
                "confidence": 9,
                "reasoning": "This does not belong in my permanent soul.",
                "why_it_matters": "",
            }
        )

    def name(self) -> str:
        return "reject-fake"

    def chat(self, messages, *, tools=None, options=None):
        raise NotImplementedError


class _LowConfidenceProvider(LLMProvider):
    """Returns an 'accept' response but with confidence below default threshold (7)."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return json.dumps(
            {
                "decision": "accept",
                "love_type": "craft",
                "resonance": 7,
                "confidence": 4,  # Below threshold of 7
                "reasoning": "I think so but I'm not very sure.",
                "why_it_matters": "Maybe important.",
            }
        )

    def name(self) -> str:
        return "low-confidence-fake"

    def chat(self, messages, *, tools=None, options=None):
        raise NotImplementedError


class _BadJsonProvider(LLMProvider):
    """Returns garbage that cannot be parsed as JSON."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "sorry i cannot decide right now, no json here"

    def name(self) -> str:
        return "bad-json-fake"

    def chat(self, messages, *, tools=None, options=None):
        raise NotImplementedError


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_review_no_candidates_returns_empty_report(tmp_path: Path) -> None:
    """review_pending_candidates with no candidates returns empty report."""
    store = _make_memory_store()
    soul_store = _make_soul_store()
    provider = _AcceptProvider()

    report = review_pending_candidates(
        tmp_path,
        store=store,
        soul_store=soul_store,
        provider=provider,
    )

    assert report.pending_at_start == 0
    assert report.examined == 0
    assert report.accepted == 0
    assert report.decisions == []
    store.close()
    soul_store.close()


def test_review_accept_creates_crystallization(tmp_path: Path) -> None:
    """An accept decision with high confidence creates a crystallization."""
    store = _make_memory_store()
    soul_store = _make_soul_store()
    provider = _AcceptProvider()

    candidate = _make_candidate("Writing is not what I do, it's what I am.")
    _write_candidates(tmp_path, [candidate])

    report = review_pending_candidates(
        tmp_path,
        store=store,
        soul_store=soul_store,
        provider=provider,
    )

    assert report.accepted == 1
    assert report.examined == 1
    assert len(report.crystallization_ids) == 1
    assert soul_store.count() == 1

    # Candidate status was updated
    import json as _json

    lines = (tmp_path / "soul_candidates.jsonl").read_text().strip().splitlines()
    updated = _json.loads(lines[0])
    assert updated["status"] == "accepted"
    assert "crystallization_id" in updated

    store.close()
    soul_store.close()


def test_review_reject_marks_candidate_rejected(tmp_path: Path) -> None:
    """A reject decision marks candidate rejected; no crystallization created."""
    store = _make_memory_store()
    soul_store = _make_soul_store()
    provider = _RejectProvider()

    candidate = _make_candidate("Something that does not belong.")
    _write_candidates(tmp_path, [candidate])

    report = review_pending_candidates(
        tmp_path,
        store=store,
        soul_store=soul_store,
        provider=provider,
    )

    assert report.rejected == 1
    assert report.accepted == 0
    assert soul_store.count() == 0

    import json as _json

    lines = (tmp_path / "soul_candidates.jsonl").read_text().strip().splitlines()
    updated = _json.loads(lines[0])
    assert updated["status"] == "rejected"

    store.close()
    soul_store.close()


def test_review_low_confidence_forces_defer(tmp_path: Path) -> None:
    """A decision with confidence < threshold is forced to defer."""
    store = _make_memory_store()
    soul_store = _make_soul_store()
    provider = _LowConfidenceProvider()

    candidate = _make_candidate("Maybe important maybe not.")
    _write_candidates(tmp_path, [candidate])

    report = review_pending_candidates(
        tmp_path,
        store=store,
        soul_store=soul_store,
        provider=provider,
        confidence_threshold=7,
    )

    assert report.deferred == 1
    assert report.accepted == 0
    assert soul_store.count() == 0

    # The decision should have a forced_defer_reason
    assert report.decisions[0].forced_defer_reason != ""
    assert "confidence" in report.decisions[0].forced_defer_reason

    store.close()
    soul_store.close()


def test_review_parse_failure_becomes_defer(tmp_path: Path) -> None:
    """A parse failure results in a defer with parse_error set."""
    store = _make_memory_store()
    soul_store = _make_soul_store()
    provider = _BadJsonProvider()

    candidate = _make_candidate("Some important moment.")
    _write_candidates(tmp_path, [candidate])

    report = review_pending_candidates(
        tmp_path,
        store=store,
        soul_store=soul_store,
        provider=provider,
    )

    assert report.parse_failures >= 1
    assert report.deferred >= 1
    assert report.decisions[0].parse_error != ""

    store.close()
    soul_store.close()


def test_review_max_decisions_caps_loop(tmp_path: Path) -> None:
    """max_decisions caps how many candidates are evaluated."""
    store = _make_memory_store()
    soul_store = _make_soul_store()
    provider = _AcceptProvider()

    candidates = [_make_candidate(f"moment {i}") for i in range(5)]
    _write_candidates(tmp_path, candidates)

    report = review_pending_candidates(
        tmp_path,
        store=store,
        soul_store=soul_store,
        provider=provider,
        max_decisions=2,
    )

    assert report.examined == 2
    assert report.pending_at_start == 5

    store.close()
    soul_store.close()


def test_review_dry_run_skips_writes(tmp_path: Path) -> None:
    """dry_run=True logs audit but skips soul_store + candidate file writes."""
    store = _make_memory_store()
    soul_store = _make_soul_store()
    provider = _AcceptProvider()

    candidate = _make_candidate("A moment that would normally be crystallized.")
    _write_candidates(tmp_path, [candidate])

    report = review_pending_candidates(
        tmp_path,
        store=store,
        soul_store=soul_store,
        provider=provider,
        dry_run=True,
    )

    assert report.dry_run is True
    assert report.examined == 1
    # No crystallizations written
    assert soul_store.count() == 0
    # Candidate file should NOT be updated (status stays auto_pending)
    import json as _json

    lines = (tmp_path / "soul_candidates.jsonl").read_text().strip().splitlines()
    original_status = _json.loads(lines[0]).get("status", "auto_pending")
    assert original_status == "auto_pending"
    # But audit log should exist
    assert (tmp_path / "soul_audit.jsonl").exists()

    store.close()
    soul_store.close()


# ── parse_decision unit tests ─────────────────────────────────────────────────


def test_parse_decision_no_json_block() -> None:
    d = parse_decision("no json here at all", "cid-1")
    assert d.decision == "defer"
    assert "no JSON block" in d.parse_error


def test_parse_decision_unknown_love_type_on_accept_defers() -> None:
    raw = json.dumps(
        {
            "decision": "accept",
            "love_type": "made_up_type",
            "resonance": 8,
            "confidence": 9,
            "reasoning": "test",
            "why_it_matters": "test",
        }
    )
    d = parse_decision(raw, "cid-2")
    assert d.decision == "defer"
    assert "love_type" in d.parse_error


# ── _current_emotional_summary regression ─────────────────────────────────────


def test_current_emotional_summary_uses_emotions_attr_not_all_method() -> None:
    """Regression: EmotionalState exposes ``emotions: dict[str, float]`` directly,
    not a ``.all()`` method. The helper used to call ``state.all()`` which
    raised AttributeError every invocation against a real store, swallowing
    silently to return "unknown". This test ensures real emotion data flows
    through the helper as a non-"unknown" summary string.
    """
    from brain.memory.store import Memory
    from brain.soul.review import _current_emotional_summary

    store = MemoryStore(":memory:")
    # "love" is a baseline emotion in brain.emotion.vocabulary — no persona
    # loader needed for the regression check.
    store.create(
        Memory.create_new(
            content="A genuinely meaningful moment.",
            memory_type="experience",
            domain="us",
            emotions={"love": 9.0},
            importance=8.0,
        )
    )

    summary = _current_emotional_summary(store)
    assert summary != "unknown", (
        "helper still raising AttributeError — likely regressed back to state.all()"
    )
    # When emotions are present we expect a "name:value" formatted summary
    assert ":" in summary, f"expected formatted summary, got {summary!r}"
