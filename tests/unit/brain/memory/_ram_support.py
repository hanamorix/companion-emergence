"""Shared test doubles for the reranker memory model (S220): a provider that behaves like the
real one where it matters to the ladder (the tokenizer truncates pairs at the model limit, a
call's time and peak growth are functions of its SHAPE), with hooks to place events inside a
call. Synthetic numbers only."""

from __future__ import annotations

from collections.abc import Callable

from brain.memory import cost_series, ram_fit
from brain.memory import reranker as reranker_mod
from brain.memory.reranker import RerankerProvider

FULL = 1024  # the model's input limit in tokens

# A "true" memory law for the scripted machine, in bytes: a + b*T + c*B*L^2 (clamped at 0).
TRUE_A, TRUE_B, TRUE_C = -20e6, 60e3, 170.0


def true_growth(pairs: int, length: int) -> float:
    return max(0.0, TRUE_A + TRUE_B * pairs * length + TRUE_C * pairs * length * length)


class ScriptedLadderProvider(RerankerProvider):
    """Token count of a pair = characters / 4, truncated at `cap`, like a real tokenizer; a call
    of B pairs padded to the longest costs `overhead + rate * B * L` seconds and grows the
    process peak by `growth_fn(B, L)` bytes (None = unreadable). Records everything."""

    def __init__(
        self,
        *,
        cap: int = FULL,
        overhead: float = 0.1,
        rate: float = 1e-3,
        growth_fn: Callable[[int, int], float | None] | None = true_growth,
        model: str = "scripted-ladder",
    ) -> None:
        self.cap, self.overhead, self.rate, self.growth_fn, self._model = cap, overhead, rate, growth_fn, model
        self.probes = 0
        self.plain: list[tuple[str, list[str]]] = []
        self.timed: list[tuple[int, int]] = []  # (pairs, padded length) of every timed call
        self.on_probe: Callable[[int], None] | None = None
        self.on_warm: Callable[[int], None] | None = None
        self.on_timed: Callable[[int, int], None] | None = None

    def _tokens(self, query: str, document: str) -> int:
        return min(self.cap, (len(query) + len(document)) // 4)

    def rerank(self, query, documents):
        self.plain.append((query, list(documents)))
        if self.on_warm is not None:
            self.on_warm(len(self.plain))
        return [0.0] * len(documents)

    def rerank_timed(self, query, documents):
        pairs = len(documents)
        length = max(self._tokens(query, d) for d in documents)
        self.timed.append((pairs, length))
        if self.on_timed is not None:
            self.on_timed(pairs, length)
        growth = None if self.growth_fn is None else self.growth_fn(pairs, length)
        return [0.0] * pairs, self.overhead + self.rate * pairs * length, growth

    def pair_token_lengths(self, query, documents):
        self.probes += 1
        if self.on_probe is not None:
            self.on_probe(self.probes)
        return [self._tokens(query, d) for d in documents]

    def model_id(self) -> str:
        return self._model


def shapes_in_window(store, model_id: str) -> dict[tuple[int, int], tuple[int, float]]:
    """`{(pairs, length): (samples, total growth)}` of the memory series window."""
    groups = reranker_mod._ram_shapes.shape_groups(store, model_id)  # noqa: SLF001
    assert groups is not None
    return {ram_fit.decode_shape(x)[:2]: (n, total) for x, n, total in groups}


def full_marked(store, model_id: str) -> bool:
    groups = reranker_mod._ram_shapes.shape_groups(store, model_id)  # noqa: SLF001
    return any(ram_fit.decode_shape(x)[2] for x, _n, _t in groups or [])


def seed_window(store, model_id: str, bytes_per_token: float = 245.0) -> None:
    """Four shapes (growth = bytes_per_token x padded tokens, the full-length one marked)."""
    for pairs, length, full in [(1, FULL, True), (1, 512, False), (1, 256, False), (2, 128, False)]:
        x = ram_fit.encode_shape(pairs, length, full=full)
        reranker_mod._ram_shapes.record(store, model_id, x, bytes_per_token * pairs * length)  # noqa: SLF001


def flush(store) -> None:
    cost_series.flush(store)
