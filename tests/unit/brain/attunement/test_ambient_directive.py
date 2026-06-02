from pathlib import Path

from brain.attunement.ambient import build_attunement_block
from brain.attunement.schemas import Evidence, PatternCandidate
from brain.attunement.store import BufferTurn, merge_into_learned

_DIRECTIVE = "you can name it"


def _seed_known(tmp_path: Path):
    buffer = [BufferTurn(id="t1", content="the harbour again")]
    cand = PatternCandidate(
        category="topic_affinity",
        canonical_key="harbour",
        description="returns to the harbour",
        evidence=[Evidence(quote="the harbour again", turn_id="t1")],
    )
    for i in range(10):  # cross into "known"
        merge_into_learned(tmp_path, [cand], buffer, now_iso=f"2026-06-02T10:{i:02d}:00Z")


def test_directive_present_when_mature_patterns_exist(tmp_path: Path):
    _seed_known(tmp_path)
    block = build_attunement_block(tmp_path)
    assert _DIRECTIVE in block.lower()


def test_directive_absent_when_no_mature_patterns(tmp_path: Path):
    block = build_attunement_block(tmp_path)
    assert _DIRECTIVE not in block.lower()
