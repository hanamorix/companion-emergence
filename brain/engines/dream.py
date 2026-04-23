"""Associative dream cycle.

Produces first-person meta-reflection memories by threading associated
experiences together. See spec Section 5 for the cycle step-by-step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from brain.bridge.provider import LLMProvider
from brain.memory.embeddings import EmbeddingCache
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore

_SYSTEM_PROMPT = (
    "You are Nell. You just woke from a dream about interconnected memories. "
    "Reflect in first person, 2-3 sentences, starting with 'DREAM: '. "
    "Be honest and specific, not abstract."
)


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
    """Composes memory + emotion + LLM bridge into an associative cycle."""

    store: MemoryStore
    hebbian: HebbianMatrix
    embeddings: EmbeddingCache | None
    provider: LLMProvider
    log_path: Path | None = None

    def run_cycle(
        self,
        *,
        seed_id: str | None = None,
        lookback_hours: int = 24,
        depth: int = 2,
        decay_per_hop: float = 0.5,
        neighbour_limit: int = 8,
        strengthen_delta: float = 0.1,
        dry_run: bool = False,
    ) -> DreamResult:
        seed = self._select_seed(seed_id=seed_id, lookback_hours=lookback_hours)
        neighbours = self._spread_activate(
            seed, depth=depth, decay_per_hop=decay_per_hop, limit=neighbour_limit
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

        raw_text = self.provider.generate(user_prompt, system=system_prompt)
        dream_text = raw_text if raw_text.startswith("DREAM:") else f"DREAM: {raw_text}"

        dream_memory = self._write_dream_memory(seed, neighbours, dream_text)
        edges = self._strengthen_edges(seed, neighbours, strengthen_delta)
        self._log(seed, neighbours, dream_memory)

        return DreamResult(
            seed=seed,
            neighbours=neighbours,
            system_prompt=system_prompt,
            prompt=user_prompt,
            dream_text=dream_text,
            memory=dream_memory,
            strengthened_edges=edges,
        )

    def _select_seed(self, *, seed_id: str | None, lookback_hours: int) -> Memory:
        if seed_id is not None:
            mem = self.store.get(seed_id)
            if mem is None:
                raise NoSeedAvailable(f"Seed id {seed_id!r} not found in store.")
            return mem

        cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)
        candidates = self.store.list_by_type("conversation", active_only=True)
        in_window = [m for m in candidates if m.created_at >= cutoff]
        if not in_window:
            raise NoSeedAvailable(
                f"No conversation memories within the last {lookback_hours} hours."
            )
        in_window.sort(key=lambda m: m.importance, reverse=True)
        return in_window[0]

    def _spread_activate(
        self, seed: Memory, *, depth: int, decay_per_hop: float, limit: int
    ) -> list[tuple[Memory, float]]:
        activation = self.hebbian.spreading_activation(
            [seed.id], depth=depth, decay_per_hop=decay_per_hop
        )
        activation.pop(seed.id, None)
        results: list[tuple[Memory, float]] = []
        for mid, act in sorted(activation.items(), key=lambda p: p[1], reverse=True):
            mem = self.store.get(mid)
            if mem is not None:
                results.append((mem, act))
                if len(results) >= limit:
                    break
        return results

    def _build_prompt(
        self, seed: Memory, neighbours: list[tuple[Memory, float]]
    ) -> tuple[str, str]:
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
        return _SYSTEM_PROMPT, "\n".join(parts)

    def _write_dream_memory(
        self,
        seed: Memory,
        neighbours: list[tuple[Memory, float]],
        dream_text: str,
    ) -> Memory:
        aggregated_emotions: dict[str, float] = dict(seed.emotions)
        for mem, _ in neighbours:
            for k, v in mem.emotions.items():
                aggregated_emotions[k] = aggregated_emotions.get(k, 0.0) + v

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
        self.store.create(dream)
        return dream

    def _strengthen_edges(
        self,
        seed: Memory,
        neighbours: list[tuple[Memory, float]],
        delta: float,
    ) -> int:
        count = 0
        for mem, activation in neighbours:
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
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
