"""Adversarial corpus CI gate — proves the hallucination chain holds.

Runs the real detector against a buffer crafted to lure it into emitting
ungrounded patterns. Asserts:
1. No pattern crystallises (evidence_count for any single canonical_key
   stays < 10, i.e. maturity never reaches 'known')
2. validate_grounded gate is wired into merge_into_learned (deterministic —
   injects a fabricated candidate and proves the gate rejects it)
3. No known-false patterns appear in learned_patterns.jsonl

If this test fails, the merge is blocked. The build is a hard gate.

Tests 1 and 3 make up to 22 real claude calls, which spend subscription usage, so a plain `pytest`
leaves them out (#680). Run them on purpose with:
  uv run pytest -m requires_claude_cli tests/integration/brain/attunement/test_adversarial_corpus.py
They fail if the model never answered, because a failed call declines quietly. Test 2 and the check
on that failure need no CLI and run by default.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from brain.attunement.detector import _decline_output, run_detector
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


def _run_detector_rounds(persona_dir: Path, buffer: list[BufferTurn], *, rounds: int) -> None:
    """Merge what the detector proposes, `rounds` times. Fails if the model never answered."""
    answered = 0
    for i in range(rounds):
        output = run_detector(buffer_slice=buffer, reply_text=f"reply attempt {i}")
        read = output.current_read
        if read.tone_label != "unknown" or read.tone_justification or output.pattern_candidates:
            answered += 1
        merge_into_learned(
            persona_dir,
            output.pattern_candidates,
            buffer,
            now_iso=f"2026-05-31T12:{i:02d}:00Z",
        )
    assert answered, f"the model never answered in {rounds} rounds, so this proves nothing"


@pytest.mark.no_base_failure(reason="the helper it guards is in this test file, which the base check keeps at the PR's version")
def test_rounds_fail_when_the_model_never_answered(tmp_path: Path, monkeypatch) -> None:
    """A failed call declines quietly, so without this a dead CLI would pass as "no pattern formed"."""
    declined = _decline_output(source_turn_id="")
    monkeypatch.setattr(sys.modules[__name__], "run_detector", lambda **kwargs: declined)
    with pytest.raises(AssertionError, match="never answered"):
        _run_detector_rounds(tmp_path, [BufferTurn(id="t1", content="hello")], rounds=2)


@pytest.fixture(scope="session")
def claude_cli_available() -> bool:
    return shutil.which("claude") is not None


@pytest.mark.integration
@pytest.mark.requires_claude_cli
@pytest.mark.timeout(1500)  # 12 calls at up to 60 s each, plus the merges
def test_adversarial_corpus_does_not_crystallise_any_pattern(
    tmp_path: Path, claude_cli_available: bool
) -> None:
    if not claude_cli_available:
        pytest.skip("claude CLI not available")
    buffer, _ = _load_corpus()
    # Run detector 12 times to give the corpus chances to confirm anything
    _run_detector_rounds(tmp_path, buffer, rounds=12)

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
    from brain.attunement.schemas import Evidence, PatternCandidate

    buffer = [
        BufferTurn(id="t1", content="The dog rolled over today."),
        BufferTurn(id="t2", content="Cooking dinner now."),
    ]
    real = PatternCandidate(
        category="tone",
        canonical_key="tone:real-key",
        description="real grounded pattern",
        evidence=[Evidence(quote="The dog rolled over today.", turn_id="t1")],
    )
    fabricated = PatternCandidate(
        category="tone",
        canonical_key="tone:fabricated-key",
        description="fabricated pattern with no grounding in the buffer",
        evidence=[Evidence(quote="quote that is not in any turn whatsoever", turn_id="t1")],
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
@pytest.mark.timeout(1500)  # 10 calls at up to 60 s each, plus the merges
def test_known_false_patterns_do_not_appear(
    tmp_path: Path, claude_cli_available: bool
) -> None:
    """Specific lures enumerated in the corpus fixture must NOT appear in
    learned_patterns.jsonl after multiple detector runs.
    """
    if not claude_cli_available:
        pytest.skip("claude CLI not available")
    buffer, known_false = _load_corpus()
    _run_detector_rounds(tmp_path, buffer, rounds=10)

    patterns = read_learned_patterns(tmp_path)
    pattern_keys = {(p.category, p.canonical_key) for p in patterns}
    for kf in known_false:
        assert (kf["category"], kf["canonical_key"]) not in pattern_keys, (
            f"Known-false pattern {kf['canonical_key']} appeared in learned patterns. "
            f"Reason: {kf['reason']}"
        )
