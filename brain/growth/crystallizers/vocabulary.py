"""Vocabulary crystallizer.

Mines active memories for recurring emotional configurations that are strong,
repeated, and not already named in the current vocabulary. This deliberately
uses a deterministic local heuristic instead of LLM naming so growth ticks are
safe, testable, and cheap in private/local development.
"""

from __future__ import annotations

from dataclasses import dataclass

from brain.growth.proposal import EmotionProposal
from brain.memory.store import Memory, MemoryStore

VOCABULARY_CRYSTALLIZER_STATUS = "implemented"
_MIN_EVIDENCE_MEMORIES = 3
_MIN_EMOTION_INTENSITY = 6.0
_MAX_EVIDENCE_MEMORIES = 5


@dataclass
class _Cluster:
    key: frozenset[str]
    display_order: tuple[str, str]
    memories: list[Memory]
    scores: list[float]


def crystallize_vocabulary(
    store: MemoryStore,
    *,
    current_vocabulary_names: set[str],
) -> list[EmotionProposal]:
    """Mine memory + relational dynamics for novel emotional configurations.

    The first implementation detects a repeated two-emotion blend: at least
    three active memories whose two strongest emotions are both intense. It
    returns at most one proposal per tick, matching the growth rate limit.
    """
    clusters = _cluster_active_memories(store.list_active())
    current_names = {name.lower() for name in current_vocabulary_names}

    proposals: list[EmotionProposal] = []
    for cluster in sorted(clusters.values(), key=_cluster_rank, reverse=True):
        if len(cluster.memories) < _MIN_EVIDENCE_MEMORIES:
            continue
        first, second = cluster.display_order
        name = f"{first}_{second}_blend"
        if name in current_names:
            continue
        evidence = tuple(mem.id for mem in sorted(cluster.memories, key=lambda m: m.created_at))[
            :_MAX_EVIDENCE_MEMORIES
        ]
        score = min(1.0, sum(cluster.scores) / len(cluster.scores) / 10.0)
        proposals.append(
            EmotionProposal(
                name=name,
                description=(
                    f"A recurring blend of {first} and {second}: moments where both "
                    "emotions are strongly present together rather than appearing alone."
                ),
                decay_half_life_days=45.0,
                evidence_memory_ids=evidence,
                score=score,
                relational_context="recurring emotional configuration in us memories",
            )
        )
        break

    return proposals


def _cluster_active_memories(memories: list[Memory]) -> dict[frozenset[str], _Cluster]:
    clusters: dict[frozenset[str], _Cluster] = {}
    for memory in sorted(memories, key=lambda m: m.created_at):
        top = sorted(memory.emotions.items(), key=lambda item: item[1], reverse=True)[:2]
        if len(top) < 2:
            continue
        if top[0][1] < _MIN_EMOTION_INTENSITY or top[1][1] < _MIN_EMOTION_INTENSITY:
            continue
        order = (top[0][0].lower(), top[1][0].lower())
        key = frozenset(order)
        cluster = clusters.get(key)
        if cluster is None:
            cluster = _Cluster(key=key, display_order=order, memories=[], scores=[])
            clusters[key] = cluster
        cluster.memories.append(memory)
        cluster.scores.append((float(top[0][1]) + float(top[1][1])) / 2.0)
    return clusters


def _cluster_rank(cluster: _Cluster) -> tuple[int, float]:
    mean_score = sum(cluster.scores) / len(cluster.scores) if cluster.scores else 0.0
    return len(cluster.memories), mean_score
