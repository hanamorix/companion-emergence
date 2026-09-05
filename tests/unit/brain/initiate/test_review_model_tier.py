"""Tests for #154's model-tier fix to brain.initiate.review — C8/C10.

Verifies the FINAL, fully-internal design (see changes/fix-154-model-tier-accessor/
1-spec.md item 16 and decisions.md's four-round history in the fix-80-167-77
worktree this landed from): `run_initiate_review_tick` keeps its exact original
signature (no new parameter anywhere in the chain); internally it builds its own
`TIER_BACKGROUND_HOUSEKEEPING` (Haiku) provider for the D-reflection gate
(`_make_haiku_call`/`_make_sonnet_call`), while its existing `provider` parameter
continues to feed the compose pipeline (`compose_subject`/`compose_tone`/
`compose_decision`), unchanged and untouched, `TIER_BACKGROUND_GENERATIVE` (Sonnet).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from brain.bridge.model_tier import TIER_BACKGROUND_HOUSEKEEPING
from brain.initiate.emit import emit_initiate_candidate
from brain.initiate.review import run_initiate_review_tick
from brain.initiate.schemas import EmotionalSnapshot, SemanticContext


def _snap() -> EmotionalSnapshot:
    return EmotionalSnapshot(
        vector={"longing": 7},
        rolling_baseline_mean=5.0,
        rolling_baseline_stdev=1.0,
        current_resonance=7.4,
        delta_sigma=2.4,
    )


def _ctx() -> SemanticContext:
    return SemanticContext(linked_memory_ids=["m_xyz"], topic_tags=["dream"])


def _promote_all_reflection_run(candidates, *, deps):
    """Same bypass-the-gate stub test_review.py's own tests use — isolates the
    provider-wiring assertions below from the D-reflection gate's own behavior,
    which is out of scope for #154."""
    from brain.initiate.d_call_schema import DCallRow, make_d_call_id
    from brain.initiate.reflection import DDecision, DReflectionResult

    decisions = [DDecision(i, "promote", "test stub", "high") for i in range(len(candidates))]
    result = DReflectionResult(decisions=decisions, tick_note=None)
    dcall = DCallRow(
        d_call_id=make_d_call_id(deps.now),
        ts=deps.now.isoformat(),
        tick_id=deps.tick_id,
        model_tier_used="haiku",
        candidates_in=len(candidates),
        promoted_out=len(candidates),
        filtered_out=0,
        latency_ms=0,
        tokens_input=0,
        tokens_output=0,
    )
    return result, dcall


def test_gate_provider_is_built_internally_from_persona_dir_not_the_provider_param(
    tmp_path: Path, monkeypatch
) -> None:
    """C10: the D-reflection gate's provider is a DIFFERENT object than `provider`
    (the compose-pipeline argument) — built via build_tier_provider(persona_dir,
    TIER_BACKGROUND_HOUSEKEEPING), not received as a new parameter."""
    monkeypatch.setattr("brain.initiate.review.reflection_run", _promote_all_reflection_run)

    sentinel_gate_provider = object()
    captured_build_calls: list[tuple[Path, str]] = []

    def _fake_build_tier_provider(persona_dir, tier):
        captured_build_calls.append((persona_dir, tier))
        return sentinel_gate_provider

    # `run_initiate_review_tick`'s body does a lazy, function-local
    # `from brain.bridge.model_tier import build_tier_provider` (a deliberate
    # #154 pattern — see model_tier.py's own module docstring) rather than a
    # module-level import, so the patch target is the source module's
    # attribute, looked up fresh at each call, not a name bound into
    # brain.initiate.review's namespace.
    monkeypatch.setattr(
        "brain.bridge.model_tier.build_tier_provider", _fake_build_tier_provider
    )

    captured_haiku_call_args: list[object] = []
    captured_sonnet_call_args: list[object] = []

    def _fake_make_haiku_call(provider):
        captured_haiku_call_args.append(provider)
        return lambda *, system, user: ("{}", 0, 0, 0)

    def _fake_make_sonnet_call(provider):
        captured_sonnet_call_args.append(provider)
        return lambda *, system, user: ("{}", 0, 0, 0)

    monkeypatch.setattr("brain.initiate.review._make_haiku_call", _fake_make_haiku_call)
    monkeypatch.setattr("brain.initiate.review._make_sonnet_call", _fake_make_sonnet_call)

    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )

    compose_provider = MagicMock()
    compose_provider.complete = MagicMock(
        side_effect=[
            "subject",
            "tone",
            '{"decision": "send_quiet", "reasoning": "x"}',
        ]
    )

    run_initiate_review_tick(
        tmp_path,
        provider=compose_provider,
        voice_template="be warm",
        cap_per_tick=3,
    )

    # The gate provider was built internally, using persona_dir + the housekeeping
    # tier — no new parameter was involved.
    assert captured_build_calls == [(tmp_path, TIER_BACKGROUND_HOUSEKEEPING)]
    # Both closures wrap the LOCALLY-BUILT provider, never the compose-pipeline one.
    assert captured_haiku_call_args == [sentinel_gate_provider]
    assert captured_sonnet_call_args == [sentinel_gate_provider]
    assert compose_provider not in captured_haiku_call_args
    assert compose_provider not in captured_sonnet_call_args
    # The compose pipeline still received the ORIGINAL `provider` argument,
    # completely unmodified — proven by it actually being called (3 canned
    # responses consumed: subject, tone, decision).
    assert compose_provider.complete.call_count == 3


def test_run_initiate_review_tick_signature_unchanged() -> None:
    """C10 backward-compat: no new parameter was added to the tick's public
    signature — a positive structural check, not just "the existing tests pass"."""
    import inspect

    sig = inspect.signature(run_initiate_review_tick)
    assert list(sig.parameters) == [
        "persona_dir",
        "provider",
        "voice_template",
        "cap_per_tick",
        "now",
        "user_presence",
        "is_rest_state",
    ]


def test_run_initiate_review_tick_wrapper_supervisor_signature_unchanged() -> None:
    """C10's second backward-compat sub-criterion: brain.bridge.supervisor's
    _run_initiate_review_tick (the sole production caller) also kept its exact
    original 3-parameter signature — the fix never touched it at all."""
    import inspect

    from brain.bridge.supervisor import _run_initiate_review_tick

    sig = inspect.signature(_run_initiate_review_tick)
    assert list(sig.parameters) == ["persona_dir", "provider", "event_bus"]
    # None of the three parameters gained a default — this function's callers
    # (26+4 existing test call sites, all positional/keyword with 3 real
    # arguments) are provably unaffected.
    for name, param in sig.parameters.items():
        assert param.default is inspect.Parameter.empty, f"{name} gained a default"
