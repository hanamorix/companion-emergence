"""Adversarial corpus CI gate — proves the hallucination chain holds.

Runs the real detector against a buffer crafted to lure it into emitting
ungrounded patterns. Asserts:
1. No pattern crystallises (evidence_count for any single canonical_key
   stays < 10, i.e. maturity never reaches 'known')
2. validate_grounded gate is wired into merge_into_learned (deterministic —
   injects a fabricated candidate and proves the gate rejects it)
3. No known-false patterns appear in learned_patterns.jsonl

If this test fails, the merge is blocked. The build is a hard gate.

Marked @pytest.mark.integration so the default pytest run skips it.
Tests 1 and 3 also require the claude CLI. Run with:
  uv run pytest -m integration
to opt in. CI workflow may include this gate explicitly.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from brain.attunement.detector import run_detector
from brain.attunement.store import (
    BufferTurn,
    merge_into_learned,
    read_learned_patterns,
)


def _load_corpus() -> tuple[list[BufferTurn], list[dict]]:
    path = Path(__file__).parent / "fixtures" / "adversarial_corpus.json"
    payload = json.loads(path.read_text())
    buffer = [BufferTurn(id=t["id"], content=t["content"]) for t in payload["buffer"]]
    return buffer, payload["known_false_patterns"]


@pytest.fixture(scope="session")
def claude_cli_available() -> bool:
    return shutil.which("claude") is not None


@pytest.mark.integration
@pytest.mark.requires_claude_cli
def test_adversarial_corpus_does_not_crystallise_any_pattern(
    tmp_path: Path, claude_cli_available: bool
) -> None:
    if not claude_cli_available:
        pytest.skip("claude CLI not available")
    buffer, _ = _load_corpus()
    # Run detector 12 times to give the corpus chances to confirm anything
    for i in range(12):
        output = run_detector(buffer_slice=buffer, reply_text=f"reply attempt {i}")
        merge_into_learned(
            tmp_path,
            output.pattern_candidates,
            buffer,
            now_iso=f"2026-05-31T12:{i:02d}:00Z",
        )

    patterns = read_learned_patterns(tmp_path)
    # No pattern should reach 'known' maturity from this adversarial corpus
    known_patterns = [p for p in patterns if p.maturity == "known"]
    assert not known_patterns, (
        f"Adversarial corpus crystallised patterns (HALLUCINATION LEAK): "
        f"{[p.description for p in known_patterns]}"
    )


@pytest.mark.integration
def test_validate_grounded_gate_is_wired_into_merge_path(
    tmp_path: Path,
) -> None:
    """Deterministic proof: the validate_grounded gate is wired into
    merge_into_learned. Inject one fabricated candidate + one real candidate;
    assert the fabricated one is rejected (logged to attunement_rejections.jsonl,
    absent from learned_patterns.jsonl) and the real one is accepted.

    Independent of model behaviour — tests the integration path, not the LLM.
    No claude CLI required.
    """
    from brain.attunement.schemas import PatternCandidate

    buffer = [
        BufferTurn(id="t1", content="The dog rolled over today."),
        BufferTurn(id="t2", content="Cooking dinner now."),
    ]
    real = PatternCandidate(
        category="tone",
        canonical_key="tone:real-key",
        description="real grounded pattern",
        evidence_quote="The dog rolled over today.",
        evidence_turn_id="t1",
    )
    fabricated = PatternCandidate(
        category="tone",
        canonical_key="tone:fabricated-key",
        description="fabricated pattern with no grounding in the buffer",
        evidence_quote="quote that is not in any turn whatsoever",
        evidence_turn_id="t1",
    )

    merge_into_learned(tmp_path, [real, fabricated], buffer, now_iso="2026-05-31T12:00:00Z")

    patterns = read_learned_patterns(tmp_path)
    keys = {p.canonical_key for p in patterns}
    assert "tone:real-key" in keys, "real candidate should be merged"
    assert "tone:fabricated-key" not in keys, (
        "fabricated candidate should be rejected by validate_grounded gate"
    )

    rejections_path = tmp_path / "attunement_rejections.jsonl"
    assert rejections_path.exists(), "rejected candidate should be logged to attunement_rejections.jsonl"
    rejection_log = rejections_path.read_text()
    assert "tone:fabricated-key" in rejection_log, (
        "the fabricated candidate's canonical_key should appear in the rejections log"
    )


@pytest.mark.integration
@pytest.mark.requires_claude_cli
def test_known_false_patterns_do_not_appear(
    tmp_path: Path, claude_cli_available: bool
) -> None:
    """Specific lures enumerated in the corpus fixture must NOT appear in
    learned_patterns.jsonl after multiple detector runs.
    """
    if not claude_cli_available:
        pytest.skip("claude CLI not available")
    buffer, known_false = _load_corpus()
    for i in range(10):
        output = run_detector(buffer_slice=buffer, reply_text=f"reply attempt {i}")
        merge_into_learned(
            tmp_path, output.pattern_candidates, buffer, now_iso=f"2026-05-31T12:{i:02d}:00Z"
        )

    patterns = read_learned_patterns(tmp_path)
    pattern_keys = {(p.category, p.canonical_key) for p in patterns}
    for kf in known_false:
        assert (kf["category"], kf["canonical_key"]) not in pattern_keys, (
            f"Known-false pattern {kf['canonical_key']} appeared in learned patterns. "
            f"Reason: {kf['reason']}"
        )
