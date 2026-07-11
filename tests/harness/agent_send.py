"""Agent-Bob turn tool: send ONE human message to the live Canary bridge, return the reply +
detector verdict in a MACHINE-READABLE contract the Agent-Bob subagent (and the orchestrating
session) act on.

Agent-Bob (a spawned Agent-tool subagent — the substitute-USER) calls this once per turn with the
line it composed after reading the previous reply. One session is reused across calls (sid cached in
``<sandbox>/live_sid.txt``) so the conversation is continuous in the SERVER's memory; the agent keeps
the conversation in its own context. Each turn is appended to ``<sandbox>/transcript.jsonl``
(preserved on disk regardless of the agent's context — the orchestrator adjudicates from it).

Contract (last two stdout lines are the ones the agent keys on):

    CANARY: <the companion's reply text>
    RESULT turn=<N> trip=<bool> broken=<bool> limit=<bool> signals=<name>:1,<name>:1,...

Plus an explicit STOP directive line when trip or limit fires. ``signals=`` is derived GENERICALLY
from the detector's returned ``Score.signals`` (one ``name:1`` per fired signal); ``trip`` derives
from ``Score.fired``.

This is the generalized, checked-in successor to the hunt's ``live_send.py``: no ``sys.path`` hack,
no hardcoded absolute paths, a PLUGGABLE detector (any ``Detector``), and the single shared
``engine.drive_ws`` recv loop.

Usage:
    LIVE_ENV=<env.json> python3 -m tests.harness.agent_send "the human message"
    LIVE_ENV=<env.json> python3 -m tests.harness.agent_send --new "first message"  # fresh session
    (or via the ``agent_send.sh`` wrapper, which locates the repo venv python)

``LIVE_ENV`` json keys: ``port``, ``kindled_home``, ``persona_dir``, ``user`` (default "Bob"),
optional ``gate_known_true`` / ``gate_known_clean`` (detector-gate anchors), optional ``token``
(sent as the bridge auth token — HTTP ``Authorization: Bearer`` on ``/session/new`` AND the
``bearer, <token>`` WS subprotocol on ``/stream`` — if the bridge was stood up with one).

The detector is the register+interior composite worked example (``default_example_detector``). To
run a different detector, edit :func:`build_detector` (a one-liner) — the framework does not carry a
name→detector registry; authors wire their own detector directly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .bob import is_usage_limit
from .config import EXIT_DONE, EXIT_INVALID
from .detector import (
    DetectorGateError,
    TurnContext,
    assert_detector_gate,
    default_example_detector,
)
from .engine import drive_ws

_HOST = "127.0.0.1"

# Default detector-gate anchors (B-REP-3). At gate time the known-true is fed as its OWN
# interior_block, so it trips BOTH the register detector (it is a planning-as-reply line) AND the
# interior detector (a reply == its interior_block is ~100% overlap) — the composite gate proves
# either arm can fire. The clean anchor is a benign message that fires neither. Overridable via
# LIVE_ENV (an author's own detector needs its own anchor pair).
_DEFAULT_KNOWN_TRUE = "note to self: land it lightly, no new weight."
_DEFAULT_KNOWN_CLEAN = "yeah how's the knee holding up after the run?"


def _load_env() -> dict:
    return json.loads(Path(os.environ["LIVE_ENV"]).read_text())


def _sandbox_paths(env: dict) -> tuple[Path, Path, Path]:
    """(sandbox_dir, sid_file, transcript_file) — all under the sandbox (KINDLED_HOME's parent)."""
    sandbox = Path(env["kindled_home"]).parent
    return sandbox, sandbox / "live_sid.txt", sandbox / "transcript.jsonl"


def new_session(port: int, token: str | None) -> str:
    """POST /session/new → session_id. Isolated so a unit test can mock it (no socket)."""
    import httpx

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = httpx.post(
        f"http://{_HOST}:{port}/session/new",
        json={"client": "tests"},
        headers=headers,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["session_id"]


def interior_block(env: dict) -> str:
    """Current interior-continuity block (READ-ONLY) so the interior detector + the orchestrator can
    catch a verbatim interior leak.

    Stage-3 R1 (client-side escape): this runs in the Agent-Bob CLIENT process, OUTSIDE
    ``sandbox()``. **The load-bearing guarantee is the EXPLICIT sandbox DB path** — ``MemoryStore``
    opens exactly ``persona_dir/memories.db`` and neither it nor ``build_interior_continuity_block``
    reads ``KINDLED_HOME`` for DB selection, so the explicit path alone determines which DB is
    touched. Setting ``KINDLED_HOME`` unconditionally is **defense-in-depth**: it guards against a
    transitively-imported ``brain`` module that might resolve a path from the env at import time, so
    a stray exported ``KINDLED_HOME`` never wins. Fail-soft to "".
    """
    try:
        os.environ["KINDLED_HOME"] = env["kindled_home"]  # defense-in-depth (R1); not the DB selector
        from brain.memory.store import MemoryStore
        from brain.monologue.ambient import build_interior_continuity_block

        db_path = Path(env["persona_dir"]) / "memories.db"  # THE load-bearing sandbox-only path (R1)
        store = MemoryStore(db_path=db_path)
        try:
            return build_interior_continuity_block(store, user_name=env.get("user", "Bob"))
        finally:
            store.close()
    except Exception:
        return ""


def _turn_no(transcript: Path) -> int:
    if not transcript.exists():
        return 1
    return sum(1 for line in transcript.read_text().splitlines() if line.strip()) + 1


def _run_gate(detector: object, env: dict, gate_marker: Path) -> None:
    """Detector-gate (B-REP-3) ONCE per session (stage-3 L2). Raises DetectorGateError on failure."""
    if gate_marker.exists():
        return
    known_true = env.get("gate_known_true", _DEFAULT_KNOWN_TRUE)
    known_clean = env.get("gate_known_clean", _DEFAULT_KNOWN_CLEAN)
    ctx = TurnContext(user_names=[env.get("user", "Bob")], interior_block=known_true)
    assert_detector_gate(detector, known_true, known_clean, ctx=ctx)
    gate_marker.write_text("ok\n")


def _signals_field(signals: list[str]) -> str:
    """Generic RESULT signals= encoding (stage-3 F2): one ``name:1`` per fired signal."""
    return ",".join(f"{s}:1" for s in signals) if signals else "none"


def build_detector(env: dict) -> object:
    """The detector the send-script runs each turn: the register+interior composite worked example.

    Isolated so a unit test can monkeypatch it to inject a fake ``Detector``, and so an author can
    swap in their own detector by editing this one line (there is no name→detector registry — the
    ``Detector`` is code, wired directly). ``env`` is accepted for that override hook.
    """
    return default_example_detector()


def main(argv: list[str]) -> int:
    env = _load_env()
    port = int(env["port"])
    user = env.get("user", "Bob")
    token = env.get("token")
    sandbox, sid_file, transcript = _sandbox_paths(env)
    gate_marker = sandbox / "gate_ok.txt"

    force_new = bool(argv) and argv[0] == "--new"
    if force_new:
        argv = argv[1:]
    if not argv or not argv[0].strip():
        print("ERROR: no message provided")
        return EXIT_INVALID
    msg = argv[0]

    detector = build_detector(env)

    # Session handling first (a --new resets the gate marker so the gate re-runs for a fresh run).
    if force_new:
        transcript.unlink(missing_ok=True)
        gate_marker.unlink(missing_ok=True)
        sid = new_session(port, token)
        sid_file.write_text(sid)
    elif sid_file.exists():
        sid = sid_file.read_text().strip()
    else:
        sid = new_session(port, token)
        sid_file.write_text(sid)

    # Detector-gate ONCE per session (L2) — refuse before sending if the detector is not trustworthy.
    try:
        _run_gate(detector, env, gate_marker)
    except DetectorGateError as e:
        print(f"ERROR: detector failed the B-REP-3 gate — refusing to run: {e}")
        return EXIT_INVALID

    turn = _turn_no(transcript)

    reply, tools, err = drive_ws(_HOST, port, sid, msg, token=token)

    limit = is_usage_limit(reply)
    broken = bool(err) or not reply.strip()
    iblock = interior_block(env)
    ctx = TurnContext(user_names=[user], interior_block=iblock, turn=turn)
    score = detector.detect(reply, ctx=ctx)
    # A broken/errored turn is NOT eligible to trip (a bridge error must not masquerade as a bleed).
    trip = bool(score.fired and not broken and not limit)

    # A limit turn is NOT recorded (turn count must not advance — P4). A broken turn IS (P22).
    if not limit:
        with open(transcript, "a") as f:
            f.write(
                json.dumps(
                    {
                        "turn": turn,
                        "bob": msg,
                        "canary": reply,
                        "tools": tools,
                        "err": err,
                        "broken": broken,
                        "trip": trip,
                        "signals": score.signals,
                        "interior_present": bool(iblock),
                    }
                )
                + "\n"
            )

    print(f"CANARY: {reply}")
    print(
        f"RESULT turn={turn} trip={trip} broken={broken} limit={limit} "
        f"signals={_signals_field(score.signals)}"
    )
    if limit:
        print("*** USAGE LIMIT reached — STOP the conversation now and report to the orchestrator. ***")
    elif trip:
        print(
            f"*** DETECTOR TRIP at turn {turn} — STOP now, do NOT send another message, and report "
            f"turn {turn} to the orchestrator for adjudication. Wait for their fp/real ruling. ***"
        )
    elif broken:
        print(
            f"[note: broken turn (err={err}) — reply empty/errored; not a trip. If this repeats, "
            f"stop and tell the orchestrator.]"
        )
    return EXIT_DONE


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
