"""Membership test — hebbian path + embedding path + boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain.narrative_memory.arc import Arc, ArcMember
from brain.narrative_memory.membership import (
    MEMBER_EMBEDDING_THRESHOLD,
    MEMBER_HEBBIAN_THRESHOLD,
    EmbeddingsView,
    HebbianView,
    centroid_for,
    is_candidate,
)


@dataclass
class FakeHebbian:
    """HebbianView fake — pairwise weight lookup."""

    weights: dict[tuple[str, str], float]

    def weight(self, a: str, b: str) -> float:
        if (a, b) in self.weights:
            return self.weights[(a, b)]
        if (b, a) in self.weights:
            return self.weights[(b, a)]
        return 0.0


@dataclass
class FakeEmbeddings:
    """EmbeddingsView fake — memory_id -> vector lookup."""

    vectors: dict[str, np.ndarray]

    def get(self, memory_id: str) -> np.ndarray | None:
        return self.vectors.get(memory_id)


def _arc_with_members(member_ids: tuple[str, ...]) -> Arc:
    members = tuple(
        ArcMember(
            memory_id=mid,
            joined_at_iso="2026-05-19T10:00:00+00:00",
            lived_age_at_join=412.0,
            salience_at_join=0.7,
        )
        for mid in member_ids
    )
    return Arc(
        id="arc_test",
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref="dream_evt_1",
        seed_memory_ids=(member_ids[0],),
        title="test arc",
        opened_at_iso="2026-05-19T10:00:00+00:00",
        lived_age_at_open=412.0,
        last_extended_at_iso="2026-05-19T10:00:00+00:00",
        closed_at_iso=None,
        lived_age_at_close=None,
        members=members,
    )


@dataclass
class FakeMemory:
    id: str


def test_module_exports_protocols_and_thresholds():
    # Public-surface contract — protocol classes and the embedding threshold
    # are imported above; this test pins them so the import isn't
    # flagged as unused and the contract stays stable.
    assert MEMBER_EMBEDDING_THRESHOLD == 0.6
    assert HebbianView is not None
    assert EmbeddingsView is not None


def test_is_candidate_hebbian_path_above_threshold():
    arc = _arc_with_members(("mem_a", "mem_b"))
    candidate = FakeMemory(id="mem_c")
    hebbian = FakeHebbian(weights={("mem_c", "mem_b"): MEMBER_HEBBIAN_THRESHOLD})
    embeddings = FakeEmbeddings(vectors={})
    cache: dict[str, np.ndarray] = {}
    result, via = is_candidate(
        candidate, arc, hebbian=hebbian, embeddings=embeddings, centroid_cache=cache
    )
    assert result is True
    assert via == "hebbian"


def test_is_candidate_hebbian_path_below_threshold():
    arc = _arc_with_members(("mem_a",))
    candidate = FakeMemory(id="mem_c")
    hebbian = FakeHebbian(weights={("mem_c", "mem_a"): MEMBER_HEBBIAN_THRESHOLD - 0.5})
    embeddings = FakeEmbeddings(vectors={})
    cache: dict[str, np.ndarray] = {}
    result, via = is_candidate(
        candidate, arc, hebbian=hebbian, embeddings=embeddings, centroid_cache=cache
    )
    assert result is False
    assert via is None


def test_is_candidate_embedding_path_above_threshold():
    arc = _arc_with_members(("mem_a",))
    candidate = FakeMemory(id="mem_c")
    hebbian = FakeHebbian(weights={})
    # Cosine = 1.0 for parallel vectors > MEMBER_EMBEDDING_THRESHOLD
    vec = np.array([1.0, 0.0, 0.0])
    embeddings = FakeEmbeddings(vectors={"mem_a": vec, "mem_c": vec})
    cache: dict[str, np.ndarray] = {}
    result, via = is_candidate(
        candidate, arc, hebbian=hebbian, embeddings=embeddings, centroid_cache=cache
    )
    assert result is True
    assert via == "embedding"


def test_is_candidate_embedding_path_below_threshold():
    arc = _arc_with_members(("mem_a",))
    candidate = FakeMemory(id="mem_c")
    hebbian = FakeHebbian(weights={})
    # Orthogonal -> cosine 0.0 < MEMBER_EMBEDDING_THRESHOLD
    embeddings = FakeEmbeddings(
        vectors={"mem_a": np.array([1.0, 0.0]), "mem_c": np.array([0.0, 1.0])}
    )
    cache: dict[str, np.ndarray] = {}
    result, via = is_candidate(
        candidate, arc, hebbian=hebbian, embeddings=embeddings, centroid_cache=cache
    )
    assert result is False
    assert via is None


def test_is_candidate_both_paths_fail():
    arc = _arc_with_members(("mem_a",))
    candidate = FakeMemory(id="mem_c")
    hebbian = FakeHebbian(weights={})
    embeddings = FakeEmbeddings(vectors={})
    cache: dict[str, np.ndarray] = {}
    result, via = is_candidate(
        candidate, arc, hebbian=hebbian, embeddings=embeddings, centroid_cache=cache
    )
    assert result is False
    assert via is None


def test_centroid_for_with_multiple_members_is_mean():
    arc = _arc_with_members(("mem_a", "mem_b"))
    embeddings = FakeEmbeddings(
        vectors={
            "mem_a": np.array([1.0, 0.0]),
            "mem_b": np.array([0.0, 1.0]),
        }
    )
    cache: dict[str, np.ndarray] = {}
    centroid = centroid_for(arc, embeddings=embeddings, cache=cache)
    assert centroid is not None
    np.testing.assert_allclose(centroid, np.array([0.5, 0.5]))
    # Second call hits cache (same object identity).
    assert centroid_for(arc, embeddings=embeddings, cache=cache) is centroid


def test_centroid_for_single_member_falls_back_to_seed():
    arc = _arc_with_members(("mem_a",))
    vec = np.array([1.0, 0.0])
    embeddings = FakeEmbeddings(vectors={"mem_a": vec})
    cache: dict[str, np.ndarray] = {}
    centroid = centroid_for(arc, embeddings=embeddings, cache=cache)
    assert centroid is not None
    np.testing.assert_allclose(centroid, vec)


def test_centroid_for_all_members_missing_returns_none():
    arc = _arc_with_members(("mem_a", "mem_b"))
    embeddings = FakeEmbeddings(vectors={})
    cache: dict[str, np.ndarray] = {}
    assert centroid_for(arc, embeddings=embeddings, cache=cache) is None
