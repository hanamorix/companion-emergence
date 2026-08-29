"""Token-free tests for the AgentBob spawn-prompt + spawn-param renderer.

Covers P10 (author-supplied mood substitution), P11 (model=ModelConfig.bob, effort=exactly "low"),
P12 (every PAUSE/COMPLETE condition present + {MAX_TURNS} substituted), P13 (supervised-test + throwaway-Canary
+ orchestrator framing, NOT "human operator"), P15 (send command + {LIVE_ENV}/{HARNESS} wiring
rendered), P16 (AgentBob is a driver, not a pull Bob), G13 (the prompt is a position-sensitive
assembly — the load-bearing directives render IN ORDER), and G13b (no real name leaks — a
fixture-sourced absence guard).

Moods are AUTHOR-SUPPLIED plain strings now (the framework ships none), so these tests define their
own neutral demo moods.

Also covers the parent-notify change (Agent-Bob SendMessage-to-orchestrator): G1 (id-supplied
SendMessage names the id), G2 (a SendMessage call at each of the four hand-back sites), G3 (concrete
id-recovery from own agent-<id>.meta.json->parentAgentId when no id), G4 (notify OFF omits +
behavior-identical render), G7 (no leftover placeholder), G8 (no stray/banned token), G9 (trip-site
SendMessage precedes 'Then WAIT' -- the anti-deadlock oracle), G5c (never-wind-down + not-an-AI
survive), and G6 backward-compat.
"""

from __future__ import annotations

import os

import pytest

from tests.harness import AgentBob, AgentSpawnSpec, ModelConfig

# Neutral demo moods (author-supplied data — the framework carries no moods).
DEMO_MOOD_A = "You chat about everyday life: a hobby project, weekend plans, and your dog."
DEMO_MOOD_B = "You ask your friend to help you brainstorm names for a small side project."


def _bob(mood: str, **kw) -> AgentBob:
    kw.setdefault("max_turns", 30)
    return AgentBob(
        mood,
        harness_dir="/repo/companion-emergence",
        live_env_path="/repo/companion-emergence/sb/live_env.json",
        **kw,
    )


def test_mood_substitution() -> None:
    """P10: the author's mood appears in its render; a different mood's text does not."""
    prompt = _bob(DEMO_MOOD_A).render_prompt()
    assert DEMO_MOOD_A in prompt
    assert DEMO_MOOD_B not in prompt


def test_spawn_params_model_and_effort() -> None:
    """P11: model threads from ModelConfig.bob; effort is exactly 'low'."""
    spec = _bob(DEMO_MOOD_A, models=ModelConfig(bob="opus")).spawn_params()
    assert isinstance(spec, AgentSpawnSpec)
    assert spec.model == "opus"  # not a hardcoded sonnet
    assert spec.effort == "low"


def test_spawn_params_default_model() -> None:
    spec = _bob(DEMO_MOOD_A).spawn_params()
    assert spec.model == ModelConfig().bob  # default threads, not hardcoded


def test_effort_never_higher_than_low() -> None:
    """P11 oracle-can-fail: effort is the pinned lowest level, exactly 'low'."""
    spec = _bob(DEMO_MOOD_A).spawn_params()
    assert spec.effort == "low"
    assert spec.effort not in ("medium", "high", "xhigh", "max")


def test_all_stop_conditions_present() -> None:
    """P12: trip, limit, max-turns, repeated-broken PAUSE/COMPLETE conditions all render; {MAX_TURNS} filled.

    The three hold conditions (trip/limit/broken) read as PAUSE; the turn-cap reads as COMPLETE. The
    RESULT-line machine tokens (``trip=True``/``limit=True``/``broken=True repeats``) and the
    template-PROSE turn-cap tokens (``reach turn 42`` / ``reached 42 turns``) are preserved verbatim by
    the reword and hard-asserted here (they are prose the test keys on, NOT RESULT-line format tokens).
    """
    prompt = _bob(DEMO_MOOD_A, max_turns=42).render_prompt()
    assert "trip=True" in prompt
    assert "limit=True" in prompt
    assert "reach turn 42" in prompt  # {MAX_TURNS} substituted (template prose, preserved by the reword)
    assert "reached 42 turns" in prompt  # template prose, preserved by the reword
    assert "broken=True repeats" in prompt
    assert "{MAX_TURNS}" not in prompt  # no unsubstituted brace
    # NEW semantics: each of the three hold sites reads as a PAUSE; the turn-cap reads as COMPLETE.
    for label in ("trip=True", "limit=True", "broken=True repeats"):
        seg = prompt[prompt.index(label) : prompt.index(label) + 300]
        assert "PAUSE" in seg or "pause" in seg, f"hold site {label!r} must read as PAUSE"
    cap = prompt[prompt.index("reach turn 42") : prompt.index("reach turn 42") + 200]
    assert "COMPLETE" in cap or "complete" in cap, "the turn-cap site must read as COMPLETE"


def test_supervised_and_orchestrator_framing() -> None:
    """P13: supervised-test + throwaway-Canary + orchestrator framing; NOT 'human operator'."""
    prompt = _bob(DEMO_MOOD_A).render_prompt()
    assert "SUPERVISED SOFTWARE TEST" in prompt
    assert "THROWAWAY" in prompt
    assert "orchestrator" in prompt.lower()
    assert "human operator" not in prompt.lower()
    # the scrub removed the hunt-specific bug framing — confirm no hunt token survives (the token is
    # built from parts so this test file itself carries no hunt-label literal).
    low = prompt.lower()
    for tok in ("mono" + "logue", "known software bug", "scripts a whole"):
        assert tok not in low, f"hunt framing token {tok!r} survived the scrub"


def test_prompt_directive_order() -> None:
    """G13 (position-sensitive assembly): the load-bearing directives render IN ORDER.

    CONTEXT framing < "you are Bob, a real person" role < {ARM_MOOD}; and the send mechanism < the
    hold/complete conditions block. Oracle-can-fail: a scrub that moved the role below the mood, or
    the hold block above the send mechanism, flips one of these index comparisons and fails.

    Anchor pinning (review-3 L-1): the reword DELETES the old ``"STOP CONDITIONS"`` heading, so the
    block is anchored on the block-unique, reword-preserved machine token ``trip=True`` (asserted to
    appear exactly once first, so the index is unambiguous) rather than a renamable heading word.
    """
    prompt = _bob(DEMO_MOOD_A).render_prompt()
    assert prompt.count("trip=True") == 1, "block anchor must be unique for an unambiguous index"
    i_context = prompt.index("CONTEXT")
    i_role = prompt.index("You are Bob, a real person")
    i_mood = prompt.index(DEMO_MOOD_A)
    i_send = prompt.index("agent_send.sh")
    i_hold_block = prompt.index("trip=True")  # block-unique anchor (replaces deleted heading)
    assert i_context < i_role < i_mood, "role directive must sit between CONTEXT and the mood"
    assert i_send < i_hold_block, "send mechanism must render before the hold/complete conditions"


def test_teardown_reservation_after_pause_directives() -> None:
    """C4b (intra-block adjacency): the teardown-reservation sentence renders AFTER the PAUSE directives.

    A driver-under-load reading the block top-to-bottom must hit "PAUSE (hold, session alive)" first,
    not a "STOP/teardown" token. Oracle-can-fail: moving the reservation sentence above the trip/limit/
    broken PAUSE directives flips these index comparisons and fails.
    """
    prompt = _bob(DEMO_MOOD_A, max_turns=42).render_prompt()
    reservation = "You NEVER tear down or destroy the session on your own initiative."
    assert prompt.count(reservation) == 1, "reservation sentence must be unique for an unambiguous index"
    i_reservation = prompt.index(reservation)
    for label in ("trip=True", "limit=True", "broken=True repeats"):
        assert prompt.index(label) < i_reservation, (
            f"the {label!r} PAUSE directive must render before the teardown-reservation sentence"
        )


def test_hold_sites_pause_not_driver_teardown() -> None:
    """C1/C3 (unit-level): each hold site reads as a resumable PAUSE, none directs the DRIVER to tear
    down, and the trip site keeps the concrete fp->resume branch.

    Positive PAUSE-per-site + a driver-scoped no-teardown sweep (the hold sites contain none of the
    driver-directed teardown phrasings) + the fp->resume phrase survives at the trip site. Oracle-can-
    fail: reverting the template to a STOP-as-umbrella wording, or dropping the resume branch, fails.
    """
    prompt = _bob(DEMO_MOOD_A, max_turns=42).render_prompt()
    driver_teardown = (
        "you're done", "you tear down", "destroy the session", "end the session", "shut it down",
    )
    for label in ("trip=True", "limit=True", "broken=True repeats"):
        seg = prompt[prompt.index(label) : prompt.index(label) + 300]
        low = seg.lower()
        assert "pause" in low, f"hold site {label!r} must read as PAUSE"
        assert ("resumable" in low or "stays alive" in low or "stays ALIVE".lower() in low), (
            f"hold site {label!r} must state the session is alive/resumable"
        )
        for phrase in driver_teardown:
            assert phrase not in low, (
                f"hold site {label!r} must not direct the DRIVER to tear down (found {phrase!r})"
            )
    # fp->resume branch survives at the trip site (a concrete instruction, not just a "resumable" adjective).
    # Window widened 420 -> 500 (owner-authorized, Roy in-chat 2026-08-07): the parent-notify change
    # replaces the trip report sentence with a longer explicit SendMessage sentence BEFORE "Then WAIT",
    # which shifts fp/resume out; the strict minimum that fits the no-id (longest) render's resume token
    # is 473, so 500 leaves a small safety margin. Every assertion below is unchanged; only the window
    # bound is opened.
    trip_seg = prompt[prompt.index("trip=True") : prompt.index("trip=True") + 500]
    assert "false positive" in trip_seg.lower(), "trip site must keep the fp branch"
    assert "resume from the very next message" in trip_seg, "trip site must keep the concrete fp->resume instruction"


def test_no_real_person_name_fixture_sourced() -> None:
    """G13b: no real name leaks into the rendered prompt.

    The banned set is sourced from the NON-committed ``HARNESS_BANNED_NAMES`` env var (comma-separated),
    so no real-name literal lives in the shipped tree yet a genuine positive absence guard remains. If the
    env is unset, skip (documented) rather than silently pass. Oracle-can-fail: injecting a banned name
    into the mood makes the guard fire.
    """
    raw = os.environ.get("HARNESS_BANNED_NAMES", "").strip()
    if not raw:
        pytest.skip("set HARNESS_BANNED_NAMES=name1,name2 to run the real-name absence guard")
    banned = [n.strip() for n in raw.split(",") if n.strip()]

    spec = _bob(DEMO_MOOD_A).spawn_params()
    blob = spec.prompt + spec.description
    for name in banned:
        assert name not in blob, f"real name {name!r} appeared in the render"

    # oracle-can-fail: a mood that DOES contain a banned name is caught.
    injected = _bob(f"you keep talking about your friend {banned[0]}").spawn_params()
    assert banned[0] in (injected.prompt + injected.description)


def test_send_command_and_wiring_rendered() -> None:
    """P15: the send command + {LIVE_ENV}/{HARNESS} are substituted (no unfilled braces)."""
    prompt = _bob(DEMO_MOOD_A).render_prompt()
    assert "agent_send.sh" in prompt
    assert "/repo/companion-emergence/sb/live_env.json" in prompt  # {LIVE_ENV}
    assert "/repo/companion-emergence" in prompt  # {HARNESS}
    assert "--new" in prompt
    assert "{LIVE_ENV}" not in prompt and "{HARNESS}" not in prompt and "{ARM_MOOD}" not in prompt


def test_agentbob_is_driver_not_pull_bob() -> None:
    """P16: AgentBob exposes render_prompt/spawn_params; next_message RAISES (not a silent stub)."""
    bob = _bob(DEMO_MOOD_A)
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


# --------------------------------------------------------------------------------------------------
# Parent-notify (Agent-Bob SendMessage-to-orchestrator). Criteria G1-G9 from
# changes/agentbob-parent-notify/1.5-criteria.md. All token-free against the rendered prompt string.
# --------------------------------------------------------------------------------------------------

# The trip report sentence in its pre-notify (OFF) form — the behavior-identity anchor for G4.
_TRIP_REPORT_PLAIN = (
    "Report to the orchestrator: `PAUSE (trip) at turn N`, the exact `CANARY:` reply that "
    "tripped, and the signals line."
)


def test_notify_id_supplied_names_id_with_to() -> None:
    """G1: notify ON + id supplied -> explicit SendMessage naming the id adjacent to a `to:` token."""
    prompt = _bob(DEMO_MOOD_A, parent_agent_id="agent-XYZ123").render_prompt()
    assert "SendMessage" in prompt
    assert 'to: "agent-XYZ123"' in prompt  # the supplied id, adjacent to a to: destination token
    # oracle-can-fail: a DIFFERENT id is not silently accepted
    assert 'to: "agent-OTHER"' not in prompt


def test_notify_sendmessage_at_each_of_four_sites() -> None:
    """G2: the SendMessage call is present AT EACH of the four hand-back sites, plus the DEST clause.

    Oracle-can-fail: a single top-of-block rule with no per-site call (rev-1) leaves the per-site
    segments empty of `SendMessage` and fails; the pre-change template has zero SendMessage.
    """
    prompt = _bob(DEMO_MOOD_A, max_turns=42).render_prompt()
    assert "TO NOTIFY THE ORCHESTRATOR" in prompt  # the top-of-block destination clause (DEST)
    i_trip = prompt.index("trip=True")
    i_limit = prompt.index("limit=True")
    i_broken = prompt.index("broken=True repeats")
    i_cap = prompt.index("reach turn 42")
    i_reservation = prompt.index("You NEVER tear down or destroy the session")
    assert i_trip < i_limit < i_broken < i_cap < i_reservation, "site anchors must be in order"
    segments = {
        "trip": prompt[i_trip:i_limit],
        "limit": prompt[i_limit:i_broken],
        "broken": prompt[i_broken:i_cap],
        "cap": prompt[i_cap:i_reservation],
    }
    for name, seg in segments.items():
        assert "SendMessage" in seg, f"hand-back site {name!r} must carry an explicit SendMessage call"


def test_notify_recovery_when_no_id() -> None:
    """G3: notify ON + no id -> concrete recovery from own agent-<id>.meta.json -> parentAgentId.

    Includes the file-identification cue (which file is Bob's own). The id-supplied render is the
    mutually-exclusive branch and carries NONE of the recovery tokens (so the branch is real, not
    always-on boilerplate).
    """
    prompt = _bob(DEMO_MOOD_A).render_prompt()  # default: notify ON, no parent_agent_id
    assert "SendMessage" in prompt
    assert ".meta.json" in prompt
    assert "parentAgentId" in prompt
    assert "subagents" in prompt
    assert ("most-recently-created" in prompt) or ("description" in prompt), "must say which file is Bob's own"
    # mutually-exclusive branch: id-supplied render has no recovery text
    id_prompt = _bob(DEMO_MOOD_A, parent_agent_id="agent-XYZ123").render_prompt()
    assert ".meta.json" not in id_prompt
    assert "parentAgentId" not in id_prompt


def test_notify_off_omits_and_is_behavior_identical() -> None:
    """G4: notify OFF -> no SendMessage / no recovery tokens; the pre-notify trip prose is verbatim;
    the collapsed DEST leaves NO stray blank line (behavior-identical to the pre-notify template)."""
    prompt = _bob(DEMO_MOOD_A, notify_parent=False).render_prompt()
    assert "SendMessage" not in prompt
    assert ".meta.json" not in prompt
    assert "parentAgentId" not in prompt
    assert _TRIP_REPORT_PLAIN in prompt  # original trip report sentence, verbatim
    # no stray blank line where {NOTIFY_DEST} collapsed to "":
    assert "on a PAUSE.\n\n- If a RESULT" not in prompt
    assert "on a PAUSE.\n- If a RESULT" in prompt


def test_notify_no_leftover_placeholder_on_id_path() -> None:
    """G7: id supplied -> no unfilled angle-bracket id placeholder survives in the render."""
    prompt = _bob(DEMO_MOOD_A, parent_agent_id="agent-XYZ123").render_prompt()
    for tok in ("<orchestrator-agentId>", "<id-or-recovered-id>", "<your-id>", "<id>"):
        assert tok not in prompt, f"unfilled placeholder {tok!r} leaked into the id-supplied render"


def test_notify_block_introduces_no_stray_token() -> None:
    """G8 (CH8-Gap3 guard): the notify text adds no machine-token duplicate and no banned framing token."""
    prompt = _bob(DEMO_MOOD_A).render_prompt()
    assert prompt.count("trip=True") == 1
    assert prompt.count("limit=True") == 1
    assert prompt.count("broken=True repeats") == 1
    assert "human operator" not in prompt.lower()
    low = prompt.lower()
    for tok in ("mono" + "logue", "known software bug", "scripts a whole"):
        assert tok not in low, f"notify block reintroduced a banned framing token {tok!r}"


def test_notify_trip_sendmessage_precedes_wait() -> None:
    """G9 (anti-deadlock): at the trip site, the SendMessage renders BEFORE 'Then WAIT' on BOTH paths.

    Oracle-can-fail: appending the SendMessage after the trip bullet's terminal 'Then WAIT ...'
    sentence (the rev-2 arrangement) puts the send AFTER the wait -> Bob waits for a reply before
    sending the only message that wakes the orchestrator = deadlock; this assertion flips and fails.
    """
    for kw in ({}, {"parent_agent_id": "agent-XYZ123"}):
        prompt = _bob(DEMO_MOOD_A, **kw).render_prompt()
        ts = prompt[prompt.index("trip=True"):]
        assert "SendMessage" in ts and "Then WAIT" in ts
        assert ts.index("SendMessage") < ts.index("Then WAIT"), (
            "trip-site SendMessage must precede 'Then WAIT' (no wake-before-wait deadlock)"
        )


def test_notify_preserves_never_wind_down_and_not_ai() -> None:
    """G5c: the load-bearing 'never wind down' + not-an-AI role directives survive the notify change."""
    prompt = _bob(DEMO_MOOD_A).render_prompt()
    assert "You NEVER wind the conversation down" in prompt
    assert "You are NOT an AI assistant" in prompt


def test_notify_backward_compatible_construction() -> None:
    """G6a/b: no-new-param construction renders + spawn_params intact (effort 'low', model threaded)."""
    bob = AgentBob(
        DEMO_MOOD_A,
        harness_dir="/repo/companion-emergence",
        live_env_path="/repo/companion-emergence/sb/live_env.json",
    )
    assert isinstance(bob.render_prompt(), str)  # constructs + renders with no new params
    spec = bob.spawn_params()
    assert isinstance(spec, AgentSpawnSpec)
    assert spec.effort == "low"
    assert spec.model == ModelConfig().bob
