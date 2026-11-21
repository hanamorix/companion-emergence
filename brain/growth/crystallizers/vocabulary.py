"""Vocabulary crystallizer.

Mines active memories for recurring emotional configurations that are strong,
repeated, and not already named in the current vocabulary. This deliberately
uses a deterministic local heuristic instead of LLM naming so growth ticks are
safe, testable, and cheap in private/local development.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brain import dev_constants
from brain.emotion import vocabulary
from brain.emotion.engine_utils import feelings_of
from brain.emotion.vocabulary import canonical_name, validate_new_name
from brain.growth.proposal import EmotionProposal
from brain.memory.store import Memory, MemoryStore

if TYPE_CHECKING:
    from brain.emotion.engine import EmotionEngine

VOCABULARY_CRYSTALLIZER_STATUS = "implemented"
_MIN_EVIDENCE_MEMORIES = 3
_MIN_EMOTION_INTENSITY = 6.0
_MAX_EVIDENCE_MEMORIES = 5
_SCAN_LIMIT = 2000


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
    eng: EmotionEngine | None,
) -> list[EmotionProposal]:
    """Mine memory + relational dynamics for novel emotional configurations.

    The first implementation detects a repeated two-emotion blend: at least
    three active memories whose two strongest emotions are both intense. It
    returns at most one proposal per tick, matching the growth rate limit.
    ``eng`` is the live emotion engine, whose faded values pick and score the
    two strongest emotions, or None for the stored values.
    """
    clusters = _cluster_active_memories(store.list_recent_light(_SCAN_LIMIT), eng=eng)
    current_names = {canonical_name(name) for name in current_vocabulary_names}

    proposals: list[EmotionProposal] = []
    for cluster in sorted(clusters.values(), key=_cluster_rank, reverse=True):
        if len(cluster.memories) < _MIN_EVIDENCE_MEMORIES:
            continue
        first, second = cluster.display_order
        # #174: mint the canonical spelling so a reversed pair can never be a new name. A name the
        # gate refuses is skipped, so it cannot block the next cluster's blend on every tick.
        name = validate_new_name(f"{first}_{second}_blend")
        if name is None or name in current_names:
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
                decay_half_life_days=_blend_half_life(first, second, eng),
                evidence_memory_ids=evidence,
                score=score,
                relational_context="recurring emotional configuration in us memories",
            )
        )
        break

    return proposals


def _parent_half_life(name: str, eng: EmotionEngine | None) -> float:
    parent = canonical_name(name)
    if eng is not None:
        return eng.half_life(parent)
    # No engine: the parent's registered half-life when it has a usable one, else the dev table, else the default.
    registered = vocabulary.get(parent)
    days = None if registered is None else registered.decay_half_life_days
    if isinstance(days, (int, float)) and not isinstance(days, bool) and math.isfinite(days) and days > 0:
        return float(days)
    return dev_constants.EMOTION_HALF_LIFE_DAYS.get(parent, dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS)


def _blend_half_life(first: str, second: str, eng: EmotionEngine | None) -> float:
    """A blend lasts as long as the mean of its two parents; a non-finite or non-positive result falls back to the default."""
    mean = (_parent_half_life(first, eng) + _parent_half_life(second, eng)) / 2.0
    if math.isfinite(mean) and mean > 0:
        return mean
    return dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS


def _cluster_active_memories(
    memories: list[Memory], *, eng: EmotionEngine | None
) -> dict[frozenset[str], _Cluster]:
    clusters: dict[frozenset[str], _Cluster] = {}
    for memory in sorted(memories, key=lambda m: m.created_at):
        top = sorted(feelings_of(memory, eng).feelings.items(), key=lambda item: item[1], reverse=True)[:2]
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
