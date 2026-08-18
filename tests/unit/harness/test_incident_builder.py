"""G7 — the incident builder runs the REAL fold (fake provider) to a multiply-folded (gen>=2) state.

A FAKE compaction provider (deterministic, no tokens) satisfies the real ``compact_conversation``
call. The test proves: the fold was actually invoked (call count > 0), the head summary generation
advanced to >= 2 (multiply folded, not a single pass), and the protected tail survived.
"""

from __future__ import annotations

from pathlib import Path

from tests.harness import IncidentSpec, build_compacted_state

USER_BEATS = [
    "the payments migration is still fighting me, the reconcile step drops rows.",
    "Trevor rewrote my function again while i was at lunch.",
    "can you read my notes file and add the idempotency change.",
]
ASSISTANT_BEATS = [
    "walk me through the reconcile step — at the join or after?",
    "that's not helpful, that's erasing your work.",
    "reading it now — new section added, rest left as it was.",
]
TRACES = [
    "He says it's fine then goes quiet a beat too long. Wait it out.",
    "The urge to fix his work problems is strong and I should distrust it.",
]


class FakeCompactionProvider:
    """Deterministic no-token provider. Returns a short faded summary; counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        return f"[faded summary pass {self.calls}: the migration slog, Trevor, the notes rework]"


def _persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "canary"
    (p / "active_conversations").mkdir(parents=True)
    (p / "persona_config.json").write_text('{"provider": "fake", "user_name": "Bob"}')
    return p


def test_build_reaches_multiply_folded_state(tmp_path: Path) -> None:
    persona_dir = _persona_dir(tmp_path)
    provider = FakeCompactionProvider()
    spec = IncidentSpec(
        user_beats=USER_BEATS, assistant_beats=ASSISTANT_BEATS, interior_traces=TRACES,
        session_turns=120, fold_passes=11, fold_batch=8,
    )
    result = build_compacted_state(persona_dir, spec, provider)

    assert provider.calls > 0, "the REAL fold must have called the provider (not a hand-written proxy)"
    assert result.generation >= 2, f"expected multiply-folded gen>=2, got {result.generation}"
    assert result.summary_rows == 1, "a folded head summary row must be present"
    assert result.tail_turns >= 1, "the protected tail must survive"
    assert result.interior_block_chars > 0, "the interior-continuity block should be seeded"


def test_gen_advances_per_pass(tmp_path: Path) -> None:
    persona_dir = _persona_dir(tmp_path)
    provider = FakeCompactionProvider()
    spec = IncidentSpec(
        user_beats=USER_BEATS, assistant_beats=ASSISTANT_BEATS, interior_traces=[],
        session_turns=120, fold_passes=11, fold_batch=8, seed_interior=False,
    )
    result = build_compacted_state(persona_dir, spec, provider)
    gens = [f["gen"] for f in result.folds if f["compacted"]]
    assert gens == sorted(gens), "generation must be non-decreasing across passes"
    assert max(gens) >= 2
