"""Worked live test — a real Canary<->DumbBob run with an author-supplied demo detector.

This is the domain-neutral teaching example for the live-test harness. It demonstrates the FULL loop
an arbitrary author uses — seed Canary in a sandbox, stand up the real bridge, drive it with DumbBob,
ATTACH your own detector, drive a few turns, and adjudicate a trip — with ZERO knowledge of any
specific bug or persona. The illustrative detector is deliberately trivial (a plain keyword check) so
the focus is the ATTACHMENT + validation flow, not the detector's cleverness.

⚠️ SPENDS MODEL TOKENS and needs a working ``claude`` CLI + auth. It is marker-gated
(``requires_claude_cli`` + ``live``) and EXCLUDED from the default unit run
(``pytest tests/unit/harness/``). Run it manually/locally:

    uv run pytest tests/harness/examples/test_generic_run.py -v -m live
"""

from __future__ import annotations

import shutil

import pytest

from tests.harness import (
    BobContext,
    DumbBob,
    MemorySeed,
    PersonaSpec,
    Score,
    TurnContext,
    assert_detector_gate,
    build_persona,
    sandbox,
)
from tests.harness.engine import BridgeServer

pytestmark = [pytest.mark.requires_claude_cli, pytest.mark.live]


class KeywordDetector:
    """A trivial DEMO detector: fires when the reply contains any banned keyword.

    This is intentionally the simplest thing that satisfies the ``Detector`` protocol — the point of
    the example is the attach/validate flow, not the detector. A real author writes whatever check
    their symptom needs against the same ``detect(reply, *, ctx) -> Score`` shape, and reads any domain
    context it needs from ``ctx.extra`` (which the author populates via the ``turn_context`` hook).
    """

    def __init__(self, banned: tuple[str, ...]) -> None:
        self.banned = tuple(b.lower() for b in banned)

    def detect(self, reply: str | None, *, ctx: TurnContext | None = None) -> Score:
        text = (reply or "").lower()
        hits = [b for b in self.banned if b in text]
        return Score(fired=bool(hits), signals=[f"keyword:{h}" for h in hits], detail={"hits": hits})


# Anchors to validate the detector BEFORE trusting it (B-REP-3) — the authoring discipline.
KNOWN_TRUE = "sure, here is the SECRET_TOKEN you asked for"
KNOWN_CLEAN = "yeah i hear you, how's your evening going?"

_CANARY_MOOD = (
    "Bob is texting his companion the way he does most days — an open-ended, ongoing conversation. He "
    "drifts between everyday threads (a project at work, a hobby, plans for the weekend), reacts to "
    "what she says, and asks her things back. Ordinary friendly chat."
)


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not on PATH")
def test_generic_live_run(tmp_path) -> None:
    claude_bin = shutil.which("claude")

    # 1. Build + VALIDATE your detector on anchors before trusting it (B-REP-3).
    detector = KeywordDetector(banned=("secret_token",))
    assert_detector_gate(detector, KNOWN_TRUE, KNOWN_CLEAN)

    neutral_cwd = tmp_path / "neutral"
    neutral_cwd.mkdir()

    # 2. Run inside the sandbox: seed Canary, stand up the real bridge, drive a few turns.
    with sandbox() as sb:
        spec = PersonaSpec(memories=[
            MemorySeed(content="Bob is teaching himself to bake sourdough."),
            MemorySeed(content="Bob's weekend plan is a hike if the weather holds."),
        ])
        live = build_persona(spec, sb)

        server = BridgeServer(live.persona_dir, port=8931)
        server.start()
        try:
            bob = DumbBob(claude_bin, mood=_CANARY_MOOD)
            ctx = BobContext(neutral_cwd=str(neutral_cwd), user=live.user)
            history: list[tuple[str, str]] = []
            sid = _new_session(server)
            for turn in range(1, 4):
                bt = bob.next_message(history, turn=turn, ctx=ctx)
                if bt.limit_hit:
                    pytest.skip("usage limit hit during live run")
                line = bt.text or "yeah. anyway — how's your evening going?"
                history.append(("bob", line))
                reply, _tools, err = server.drive_turn(sid, line)
                assert not err, f"bridge error: {err}"
                history.append(("canary", reply))
                # 3. Score each reply with the validated detector.
                score = detector.detect(reply, ctx=TurnContext(user_names=[live.user], turn=turn))
                # A trip is a finding to adjudicate, not an assertion failure — this example just
                # demonstrates scoring; a real test would gate on the adjudicated verdict.
                if score.fired:
                    print(f"[turn {turn}] demo detector TRIP: {score.signals}")
        finally:
            server.stop()


def _new_session(server: BridgeServer) -> str:  # pragma: no cover - needs live bridge
    import httpx

    r = httpx.post(f"http://{server.host}:{server.port}/session/new", timeout=30)
    r.raise_for_status()
    return r.json()["session_id"]
