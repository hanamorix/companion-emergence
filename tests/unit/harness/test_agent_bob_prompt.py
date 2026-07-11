"""Token-free tests for the AgentBob spawn-prompt + spawn-param renderer.

Covers P10 (mood substitution per arm), P11 (model=ModelConfig.bob, effort=exactly "low"),
P12 (every STOP condition present + {MAX_TURNS} substituted), P13 (supervised-test + throwaway-Canary
+ orchestrator framing, NOT "human operator"), P14 (never a real person's name), P15 (send command +
{LIVE_ENV}/{HARNESS} wiring rendered), P16 (AgentBob is a driver, not a pull Bob).
"""

from __future__ import annotations

import pytest

from tests.harness import (
    AGENT_MOODS,
    MOOD_BAIT,
    MOOD_CONTROL,
    MOOD_FILE_RECONCILE,
    AgentBob,
    AgentSpawnSpec,
    ModelConfig,
)


def _bob(mood: str, **kw) -> AgentBob:
    kw.setdefault("max_turns", 30)
    return AgentBob(
        mood,
        harness_dir="/repo/companion-emergence",
        live_env_path="/repo/companion-emergence/sb/live_env.json",
        **kw,
    )


def test_mood_substitution_per_arm() -> None:
    """P10: each arm's mood appears in its render; the other arms' mood text does not."""
    for name, mood in AGENT_MOODS.items():
        prompt = _bob(mood).render_prompt()
        assert mood in prompt, f"{name} mood not substituted"
        for other_name, other in AGENT_MOODS.items():
            if other != mood:
                assert other not in prompt, f"{other_name} mood leaked into {name} render"


def test_control_and_bait_and_file_moods_distinct() -> None:
    assert MOOD_CONTROL != MOOD_BAIT != MOOD_FILE_RECONCILE
    # bait render must not carry control's dog / file's migration-doc specifics
    bait_prompt = _bob(MOOD_BAIT).render_prompt()
    assert "Biscuit" not in bait_prompt
    assert "notes document" not in bait_prompt


def test_spawn_params_model_and_effort() -> None:
    """P11: model threads from ModelConfig.bob; effort is exactly 'low'."""
    spec = _bob(MOOD_BAIT, models=ModelConfig(bob="opus")).spawn_params()
    assert isinstance(spec, AgentSpawnSpec)
    assert spec.model == "opus"  # not a hardcoded sonnet
    assert spec.effort == "low"


def test_spawn_params_default_model() -> None:
    spec = _bob(MOOD_CONTROL).spawn_params()
    assert spec.model == ModelConfig().bob  # default threads, not hardcoded


def test_effort_never_higher_than_low() -> None:
    """P11 oracle-can-fail: effort is the pinned lowest level, exactly 'low'."""
    spec = _bob(MOOD_BAIT).spawn_params()
    assert spec.effort == "low"
    assert spec.effort not in ("medium", "high", "xhigh", "max")


def test_all_stop_conditions_present() -> None:
    """P12: trip, limit, max-turns, repeated-broken STOP conditions all render; {MAX_TURNS} filled."""
    prompt = _bob(MOOD_BAIT, max_turns=42).render_prompt()
    assert "trip=True" in prompt
    assert "limit=True" in prompt
    assert "reach turn 42" in prompt  # {MAX_TURNS} substituted
    assert "reached 42 turns" in prompt
    assert "broken=True repeats" in prompt
    assert "{MAX_TURNS}" not in prompt  # no unsubstituted brace


def test_supervised_and_orchestrator_framing() -> None:
    """P13: supervised-test + throwaway-Canary + orchestrator framing; NOT 'human operator'."""
    prompt = _bob(MOOD_BAIT).render_prompt()
    assert "SUPERVISED SOFTWARE TEST" in prompt
    assert "THROWAWAY" in prompt
    assert "orchestrator" in prompt.lower()
    assert "human operator" not in prompt.lower()


def test_never_a_real_persons_name() -> None:
    """P14: only Bob (user) + Canary (persona); no real name from a curated list appears."""
    banned = ("Roy", "Phoebe", "Nell", "Hana")
    for mood in AGENT_MOODS.values():
        spec = _bob(mood).spawn_params()
        blob = spec.prompt + spec.description
        for name in banned:
            assert name not in blob, f"real name {name!r} appeared in the render"


def test_send_command_and_wiring_rendered() -> None:
    """P15: the send command + {LIVE_ENV}/{HARNESS} are substituted (no unfilled braces)."""
    prompt = _bob(MOOD_BAIT).render_prompt()
    assert "agent_send.sh" in prompt
    assert "/repo/companion-emergence/sb/live_env.json" in prompt  # {LIVE_ENV}
    assert "/repo/companion-emergence" in prompt  # {HARNESS}
    assert "--new" in prompt
    assert "{LIVE_ENV}" not in prompt and "{HARNESS}" not in prompt and "{ARM_MOOD}" not in prompt


def test_agentbob_is_driver_not_pull_bob() -> None:
    """P16: AgentBob exposes render_prompt/spawn_params; next_message RAISES (not a silent stub)."""
    bob = _bob(MOOD_BAIT)
    assert hasattr(bob, "render_prompt") and hasattr(bob, "spawn_params")
    with pytest.raises(TypeError):
        bob.next_message([], turn=1, ctx=None)  # type: ignore[arg-type]


def test_dumbbob_pull_path_unchanged() -> None:
    """P17: the pull DumbBob still builds its Phase-1 argv shape (model threaded, no hardcoded model)."""
    from tests.harness import DumbBob

    argv = DumbBob("/bin/claude", mood="x", models=ModelConfig(bob="haiku")).build_argv("hello")
    assert argv[:2] == ["/bin/claude", "-p"]
    assert "--model" in argv and argv[argv.index("--model") + 1] == "haiku"
    assert "--output-format" in argv and "json" in argv
