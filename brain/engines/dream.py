"""Associative dream cycle.

Produces first-person meta-reflection memories by threading associated
experiences together. See spec Section 5 for the cycle step-by-step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from brain.bridge.provider import LLMProvider
from brain.emotion import engine as emotion_engine
from brain.emotion.aggregate import aggregate_state
from brain.emotion.engine_utils import memory_feelings
from brain.emotion.state import EmotionalState
from brain.engines import dream_seeds
from brain.health.json_io import Recovery, append_jsonl, iter_jsonl
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.soul.store import SoulStore
from brain.utils.memory import list_conversation_memories

if TYPE_CHECKING:
    from brain.emotion.engine import EmotionEngine

logger = logging.getLogger(__name__)


class NoSeedAvailable(Exception):  # noqa: N818
    """Raised when there are no conversation memories within the lookback window."""


@dataclass(frozen=True)
class DreamResult:
    """Outcome of a single dream cycle."""

    seed: Memory
    neighbours: list[tuple[Memory, float]]
    system_prompt: str
    prompt: str
    dream_text: str | None
    memory: Memory | None
    strengthened_edges: int


@dataclass
class DreamEngine:
    """Composes memory + emotion + LLM bridge into an associative cycle.

    Mechanism knobs (`lookback_hours`, `depth`, `decay_per_hop`,
    `neighbour_limit`, `strengthen_delta`) are constructor-level calibration,
    not per-call user choices. The brain's owner picks calibration once when
    the engine is built; `run_cycle()` only takes `seed_id` (for heartbeat-
    driven seed pickup) and `dry_run`. Per principle audit 2026-04-25.
    """

    store: MemoryStore
    hebbian: HebbianMatrix
    provider: LLMProvider
    log_path: Path | None = None
    persona_dir: Path | None = None
    persona_name: str = ""
    persona_system_prompt: str = ""
    lookback_hours: int = 24
    depth: int = 2
    decay_per_hop: float = 0.5
    neighbour_limit: int = 8
    strengthen_delta: float = 0.1
    soul_store: SoulStore | None = None
    mood_floor: float = dream_seeds.MOOD_FLOOR
    min_congruent: int = dream_seeds.MIN_CONGRUENT
    refractory_window: int = dream_seeds.REFRACTORY_WINDOW
    w_identity: float = dream_seeds.W_IDENTITY
    w_grief: float = dream_seeds.W_GRIEF
    w_refractory: float = dream_seeds.W_REFRACTORY

    def __post_init__(self) -> None:
        if not self.persona_name:
            raise ValueError(
                "DreamEngine requires persona_name — construct explicitly, don't rely on a default."
            )
        if not self.persona_system_prompt:
            raise ValueError(
                "DreamEngine requires persona_system_prompt — construct "
                "explicitly, don't rely on a default."
            )

    def _feelings_of(self, memory: Memory) -> dict[str, float]:
        """The one read of a memory's feelings: the persona's engine when this DreamEngine has a persona directory (faded values
        once the persona has switched over), the pure read with no engine otherwise (tests). Held to 0-10, canonical names."""
        if self.persona_dir is not None:
            return dict(emotion_engine.engine_for(self.persona_dir).memory_feelings(memory).feelings)
        return dict(memory_feelings(memory, None, None, None).feelings)

    def run_cycle(
        self,
        *,
        seed_id: str | None = None,
        dry_run: bool = False,
    ) -> DreamResult:
        eng = self._live_engine()
        seed = self._select_seed(seed_id=seed_id, lookback_hours=self.lookback_hours, eng=eng)
        neighbours = self._spread_activate(
            seed,
            depth=self.depth,
            decay_per_hop=self.decay_per_hop,
            limit=self.neighbour_limit,
        )
        system_prompt, user_prompt = self._build_prompt(seed, neighbours)

        if dry_run:
            return DreamResult(
                seed=seed,
                neighbours=neighbours,
                system_prompt=system_prompt,
                prompt=user_prompt,
                dream_text=None,
                memory=None,
                strengthened_edges=0,
            )

        from brain.bridge import (
            cli_throttle,  # local import: avoids a circular dependency on brain.bridge
        )

        with cli_throttle.background_slot() as slot:
            if not slot:
                return DreamResult(
                    seed=seed,
                    neighbours=neighbours,
                    system_prompt=system_prompt,
                    prompt=user_prompt,
                    dream_text=None,
                    memory=None,
                    strengthened_edges=0,
                )

            raw_text = self.provider.generate(user_prompt, system=system_prompt)
            dream_text = raw_text if raw_text.startswith("DREAM:") else f"DREAM: {raw_text}"

            dream_memory = self._write_dream_memory(seed, neighbours, dream_text, eng=eng)
            edges = self._strengthen_edges(seed, neighbours, self.strengthen_delta)
            self._log(seed, neighbours, dream_memory)
            self._emit_initiate_candidate(seed, neighbours, dream_memory)

            # The dream was dreamed, so the memories in its prompt were in front of her: one recall each (#587).
            # A dry run, a denied slot or a failed call returns or raises before here and ticks nothing. After
            # the writes above and fail-soft, so a store error here can't cost the dream. An auto-picked seed
            # comes from list reads, which never ticked; a seed asked for by id did.
            try:
                for mem in ([seed] if seed_id is not None else []) + [m for m, _ in neighbours]:
                    self.store.get(mem.id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("dream recall tick failed for seed %s: %s", seed.id, exc)

            return DreamResult(
                seed=seed,
                neighbours=neighbours,
                system_prompt=system_prompt,
                prompt=user_prompt,
                dream_text=dream_text,
                memory=dream_memory,
                strengthened_edges=edges,
            )

    def _live_engine(self) -> EmotionEngine | None:
        """The persona's emotion engine when its switch is on; None when off, unknown, or there is no persona_dir."""
        if self.persona_dir is None:
            return None
        eng = emotion_engine.engine_for(self.persona_dir)
        return eng if eng.is_live() else None

    def _select_seed(
        self, *, seed_id: str | None, lookback_hours: int, eng: EmotionEngine | None
    ) -> Memory:
        if seed_id is not None:
            mem = self.store.get(seed_id, bump=False)  # ticked by run_cycle once the dream is dreamed (#587)
            if mem is None:
                raise NoSeedAvailable(f"Seed id {seed_id!r} not found in store.")
            return mem

        cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)
        convo = [
            m
            for m in list_conversation_memories(self.store, active_only=True)
            if m.created_at >= cutoff
        ]
        grief = [
            m
            for m in self.store.list_by_type("grief_event", active_only=True)
            if m.created_at >= cutoff
        ]
        candidates = convo + grief
        if not candidates:
            raise NoSeedAvailable(
                f"No conversation memories within the last {lookback_hours} hours."
            )

        # Mood from conversation residue only (not grief breadcrumbs) so a
        # breadcrumb cannot self-set the mood that retrieves it. Live, the mood is
        # the engine's short averages: every counted source, and the engine does not
        # count a grief_event either. An empty read during a rebuild is accepted.
        if eng is not None:
            mood = EmotionalState(emotions={name: r.short for name, r in eng.read().items()})
        else:
            mood = aggregate_state(convo)
        crystallizations = self.soul_store.list_active() if self.soul_store else []
        recent = self._recent_seed_ids(self.refractory_window)

        return dream_seeds.select_seed(
            candidates,
            mood,
            crystallizations,
            recent_seed_ids=recent,
            eng=eng,
            mood_floor=self.mood_floor,
            min_congruent=self.min_congruent,
            w_identity=self.w_identity,
            w_grief=self.w_grief,
            w_refractory=self.w_refractory,
        )

    def _recent_seed_ids(self, window: int) -> list[str]:
        """Seed ids from the last `window` dream-log entries (deterministic)."""
        if self.log_path is None:
            return []
        ids: list[str] = []
        for rec in iter_jsonl(self.log_path, recover=Recovery.RAISE):
            sid = rec.get("seed_id")
            if sid:
                ids.append(sid)
        return ids[-window:]

    def _spread_activate(
        self, seed: Memory, *, depth: int, decay_per_hop: float, limit: int
    ) -> list[tuple[Memory, float]]:
        activation = self.hebbian.spreading_activation(
            [seed.id], depth=depth, decay_per_hop=decay_per_hop
        )
        activation.pop(seed.id, None)
        results: list[tuple[Memory, float]] = []
        for mid, act in sorted(activation.items(), key=lambda p: p[1], reverse=True):
            mem = self.store.get(mid, bump=False)  # ticked by run_cycle once the dream is dreamed (#587)
            if mem is not None:
                results.append((mem, act))
                if len(results) >= limit:
                    break
        return results

    def _build_prompt(
        self, seed: Memory, neighbours: list[tuple[Memory, float]]
    ) -> tuple[str, str]:
        if seed.memory_type == "grief_event":
            parts = [
                "This is a grief memory — a loss. Do not weave it together with "
                "other experiences. Sit with it: what it meant, and what is "
                "different now in its absence.",
                "",
                f"  {seed.content}",
                "",
            ]
        else:
            parts = [
                f"Seed memory (domain={seed.domain}):",
                f"  {seed.content}",
                "",
            ]
        if neighbours:
            parts.append("Also present:")
            for mem, _ in neighbours:
                parts.append(f"  - {mem.content[:120]}")
        else:
            parts.append("No other memories resonated with this one yet.")
        return self.persona_system_prompt, "\n".join(parts)

    def _write_dream_memory(
        self,
        seed: Memory,
        neighbours: list[tuple[Memory, float]],
        dream_text: str,
        *,
        eng: EmotionEngine | None,
    ) -> Memory:
        if eng is not None:
            aggregated_emotions = self._faded_means(seed, neighbours, eng)
        else:
            aggregated_emotions = self._stored_sums(seed, neighbours)

        dream = Memory.create_new(
            content=dream_text,
            memory_type="dream",
            domain=seed.domain,
            emotions=aggregated_emotions,
            metadata={
                "seed_id": seed.id,
                "activated": [m.id for m, _ in neighbours],
                "provider": self.provider.name(),
            },
        )
        from brain.memory.pending import route_write

        route_write(self.store, dream, source="dream")
        return dream

    def _stored_sums(self, seed: Memory, neighbours: list[tuple[Memory, float]]) -> dict[str, float]:
        # Sum per-emotion across seed + neighbours. Values are intentionally
        # NOT clamped here — a vivid "love"-saturated dream may legitimately
        # show love=80 across 8 activated memories. Downstream consumers that
        # need a bounded 0..10 range (e.g. EmotionalState.set) clamp on
        # ingest, not here.
        aggregated_emotions: dict[str, float] = self._feelings_of(seed)
        for mem, _ in neighbours:
            for k, v in self._feelings_of(mem).items():
                aggregated_emotions[k] = aggregated_emotions.get(k, 0.0) + v
        return aggregated_emotions

    @staticmethod
    def _faded_means(
        seed: Memory, neighbours: list[tuple[Memory, float]], eng: EmotionEngine
    ) -> dict[str, float]:
        """Per emotion, the mean of the engine's read values across the seed and the neighbours that carry it.

        The read fades each memory's values and holds each to 0-10 (old rows can hold values below 0 or over 10), so the
        mean stays within 0-10 too. A value held to 0 is kept, not dropped.
        """
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for mem in [seed, *(m for m, _ in neighbours)]:
            for name, value in eng.memory_feelings(mem).feelings.items():
                totals[name] = totals.get(name, 0.0) + value
                counts[name] = counts.get(name, 0) + 1
        return {name: total / counts[name] for name, total in totals.items()}

    def _strengthen_edges(
        self,
        seed: Memory,
        neighbours: list[tuple[Memory, float]],
        delta: float,
    ) -> int:
        # Respect forgetting: never reinforce an edge that touches a memory
        # marked for release. A fading seed → reinforce nothing this cycle.
        if seed.state == "fading":
            return 0
        count = 0
        for mem, activation in neighbours:
            if mem.state == "fading":
                continue
            weighted = delta * activation
            if weighted <= 0.0:
                continue
            self.hebbian.strengthen(seed.id, mem.id, delta=weighted)
            count += 1
        return count

    def _log(
        self,
        seed: Memory,
        neighbours: list[tuple[Memory, float]],
        dream_memory: Memory,
    ) -> None:
        if self.log_path is None:
            return
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "seed_id": seed.id,
            "neighbour_ids": [m.id for m, _ in neighbours],
            "dream_id": dream_memory.id,
            "provider": self.provider.name(),
        }
        append_jsonl(self.log_path, entry)

    def _emit_initiate_candidate(
        self,
        seed: Memory,
        neighbours: list[tuple[Memory, float]],
        dream_memory: Memory,
    ) -> None:
        """Emit an initiate candidate sourced from this dream.

        Phase 4.1 of the initiate physiology pipeline. Wrapped in
        try/except so a failure on the initiate side can never break
        the dream cycle itself — dreams are physiology, initiate is
        downstream signal.
        """
        if self.persona_dir is None:
            return
        try:
            # Local import to avoid pulling initiate deps into engines that
            # don't need them and to keep the dream module focused.
            from brain.initiate.emit import emit_initiate_candidate
            from brain.initiate.schemas import (
                EmotionalSnapshot,
                SemanticContext,
            )

            linked_ids = [seed.id, *[m.id for m, _ in neighbours]][:5]
            emit_initiate_candidate(
                self.persona_dir,
                kind="message",
                source="dream",
                source_id=dream_memory.id,
                emotional_snapshot=EmotionalSnapshot(
                    vector=self._feelings_of(dream_memory),
                    rolling_baseline_mean=0.0,
                    rolling_baseline_stdev=0.0,
                    current_resonance=0.0,
                    delta_sigma=0.0,
                ),
                semantic_context=SemanticContext(
                    linked_memory_ids=linked_ids,
                    topic_tags=[],
                ),
            )
        except Exception as exc:
            logger.warning("dream initiate emit failed: %s", exc)
