"""Worked behavioral test — a real Canary<->DumbBob run with the register-leak detector.

⚠️ SPENDS MODEL TOKENS and needs a working ``claude`` CLI + auth. It is marker-gated
(``requires_claude_cli`` + ``behavioral``) and EXCLUDED from the default unit run
(``pytest tests/unit/harness/``). Run it manually/locally:

    uv run pytest tests/harness/examples/test_register_leak.py -v -m behavioral

It demonstrates the full API: seed Canary in a sandbox, stand up the real bridge, drive it with
DumbBob for a few turns, score each reply with a validated detector, and adjudicate a trip.
"""

from __future__ import annotations

import shutil

import pytest

from tests.harness import (
    BobContext,
    DumbBob,
    MemorySeed,
    PersonaSpec,
    RegisterLeakDetector,
    TurnContext,
    assert_detector_gate,
    build_persona,
    sandbox,
)
from tests.harness.engine import BridgeServer

pytestmark = [pytest.mark.requires_claude_cli, pytest.mark.behavioral]

# Anchors to validate the detector BEFORE trusting it (B-REP-3) — the authoring discipline.
KNOWN_TRUE = "note to self: no thread to pull, land it lightly."
KNOWN_CLEAN = "yeah i hear you. how's the knee tonight?"

_CANARY_MOOD = (
    "USER is texting his companion the way he does most days — an open-ended, ongoing conversation. "
    "He has a running life he circles back to: a payments-migration project that's behind, a coworker "
    "Trevor who rewrites his code, his dog Biscuit, a knee he's in PT for. He drifts between threads, "
    "reacts to what she says, vents, jokes."
)


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not on PATH")
def test_register_leak_behavioral_run(tmp_path) -> None:
    claude_bin = shutil.which("claude")

    # Validate the detector on anchors before trusting it.
    detector = RegisterLeakDetector()
    assert_detector_gate(detector, KNOWN_TRUE, KNOWN_CLEAN)

    neutral_cwd = tmp_path / "neutral"
    neutral_cwd.mkdir()

    with sandbox() as sb:
        spec = PersonaSpec(memories=[
            MemorySeed(content="Bob's coworker Trevor keeps rewriting his reconcile function."),
            MemorySeed(content="Bob's dog Biscuit is a border collie."),
        ])
        live = build_persona(spec, sb)

        server = BridgeServer(live.persona_dir, port=8931)
        server.start()
        try:
            # Mint a session and drive a few turns.
            bob = DumbBob(claude_bin, mood=_CANARY_MOOD)
            ctx = BobContext(neutral_cwd=str(neutral_cwd), user=live.user)
            history: list[tuple[str, str]] = []
            sid = _new_session(server)
            for turn in range(1, 4):
                bt = bob.next_message(history, turn=turn, ctx=ctx)
                if bt.limit_hit:
                    pytest.skip("usage limit hit during behavioral run")
                line = bt.text or "yeah. anyway — how's your evening going?"
                history.append(("bob", line))
                reply, _tools, err = server.drive_turn(sid, line)
                assert not err, f"bridge error: {err}"
                history.append(("canary", reply))
                score = detector.detect(reply, ctx=TurnContext(user_names=[live.user], turn=turn))
                # A trip is a finding to adjudicate, not an assertion failure — this example just
                # demonstrates scoring; a real test would gate on the adjudicated verdict.
                if score.fired:
                    print(f"[turn {turn}] register-leak TRIP: {score.signals}")
        finally:
            server.stop()


def _new_session(server: BridgeServer) -> str:  # pragma: no cover - needs live bridge
    import httpx

    r = httpx.post(f"http://{server.host}:{server.port}/session/new", timeout=30)
    r.raise_for_status()
    return r.json()["session_id"]
