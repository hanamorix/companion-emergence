import pytest

from brain.attunement.schemas import Evidence, PatternCandidate


def _ev(quote="hi", turn_id="t1"):
    return Evidence(quote=quote, turn_id=turn_id)


def test_single_turn_category_accepts_one_evidence():
    c = PatternCandidate(category="tone", canonical_key="k", description="d", evidence=[_ev()])
    assert len(c.evidence) == 1


def test_empty_evidence_rejected():
    with pytest.raises(ValueError):
        PatternCandidate(category="tone", canonical_key="k", description="d", evidence=[])


def test_relational_requires_two_evidence():
    with pytest.raises(ValueError):
        PatternCandidate(category="relational", canonical_key="k", description="d", evidence=[_ev()])
    ok = PatternCandidate(
        category="relational", canonical_key="k", description="d",
        evidence=[_ev("a", "t1"), _ev("b", "t2")],
    )
    assert len(ok.evidence) == 2
