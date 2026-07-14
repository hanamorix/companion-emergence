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

This is the generalized, checked-in send-script: no ``sys.path`` hack, no hardcoded absolute paths, a
PLUGGABLE detector (any ``Detector``), and the single shared ``engine.drive_ws`` recv loop. **It ships
NO default detector** — the author attaches one via ``LIVE_ENV["detector"]``.

Usage:
    LIVE_ENV=<env.json> python3 -m tests.harness.agent_send "the human message"
    LIVE_ENV=<env.json> python3 -m tests.harness.agent_send --new "first message"  # fresh session
    (or via the ``agent_send.sh`` wrapper, which locates the repo venv python)

``LIVE_ENV`` json keys:
    ``port``, ``kindled_home``, ``persona_dir``, ``user`` (default "Bob");
    ``detector`` (REQUIRED) — a ``"module.path:factory"`` dotted path; the module is imported and the
        zero-arg factory called to build the ``Detector`` for this run. No default: core has no opinion
        about what to detect, so it refuses to run without one. There is NO name->detector registry —
        the author names their own code.
    ``turn_context`` (optional) — a ``"module.path:factory"`` dotted path where ``factory(env) -> dict``
        returns the per-turn ``TurnContext.extra`` bag (domain context a detector needs). Absent -> the
        detector sees ``extra={}``. Core never inspects the returned dict.
    ``gate_known_true`` / ``gate_known_clean`` (optional) — detector-gate anchors (B-REP-3).
    ``gate_true_context`` / ``gate_clean_context`` (optional) — a ``"module.path:factory"`` dotted
        path (same shape as ``turn_context``: ``factory(env) -> dict``) supplying the ``extra`` bag
        for the KNOWN-TRUE / KNOWN-CLEAN gate anchor RESPECTIVELY. This lets an author gate a detector
        arm whose stimulus lives in ``ctx.extra`` (a proposed file, a retrieved doc, a tool payload)
        THROUGH THIS SHIPPED SEAM instead of a side-assertion: a sentinel placed in the true anchor's
        ``extra`` no longer leaks onto the clean anchor (the F3 primitive's per-anchor ctx, wired here).
        An anchor whose key is UNSET falls back to the shared ``turn_context`` ``extra``, so setting
        just ``gate_true_context`` is valid (true supplies the sentinel, clean does not). **When
        NEITHER key is set, ``_run_gate`` is byte-identical to before this seam existed** (bare-string
        anchors + one shared ctx whose ``extra`` is ``turn_context``'s output).
    ``log_extra_values`` (optional) — a list of ``extra`` key NAMES whose VALUES to persist into each
        transcript row under an ``extra_values`` field (for post-run adjudication). Opt-in and
        author-named: absent/empty -> the row is byte-identical to today (only ``extra_keys`` names).
        Core reads ONLY the named keys (never the whole ``extra`` bag). Un-serializable values degrade
        to a ``"<unserializable T>"`` placeholder — an author's value never crashes the run.
    ``token`` (optional) — the bridge auth token (HTTP ``Authorization: Bearer`` on ``/session/new``
        AND the ``bearer, <token>`` WS subprotocol on ``/stream``) if the bridge was stood up with one.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

from .bob import is_usage_limit
from .config import EXIT_DONE, EXIT_INVALID
from .detector import (
    Detector,
    DetectorGateError,
    TurnContext,
    assert_detector_gate,
)
from .engine import drive_ws

_HOST = "127.0.0.1"

# Default detector-gate anchors (B-REP-3). These are generic placeholder strings; an author's own
# detector needs its OWN anchor pair (set via LIVE_ENV) that its detector actually fires / stays silent
# on. Core makes NO assumption about what a detector inspects, so these defaults are illustrative only.
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


def _import_factory(spec: str) -> object:
    """Resolve a ``"module.path:factory"`` dotted path → the factory callable.

    Resolution is against ``sys.path`` (via ``importlib.import_module``), so an author can attach a flat
    module they have put on ``sys.path`` (e.g. a git-ignored plug-in dir added by their own test) just as
    easily as an installed package. Raises a clear error on a malformed/unresolvable spec.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"detector/turn_context spec must be 'module.path:factory', got {spec!r}"
        )
    mod_path, _, attr = spec.partition(":")
    module = importlib.import_module(mod_path)
    return getattr(module, attr)


def build_detector(env: dict) -> Detector:
    """Build the detector for this run from the author's ``LIVE_ENV["detector"]`` dotted path.

    **No default, no registry.** If ``env["detector"]`` is absent, raise — core has no opinion about what
    to detect and refuses to run without an author-supplied detector. Isolated as a module-level function
    so a unit test can monkeypatch it to inject a fake ``Detector``.
    """
    spec = env.get("detector")
    if not spec:
        raise ValueError(
            "no detector configured: set LIVE_ENV['detector'] to a 'module.path:factory' dotted path "
            "(core ships no default detector — the author supplies one)"
        )
    factory = _import_factory(spec)
    return factory()


def _extra_from_hook(env: dict, key: str) -> dict:
    """Resolve the author's optional ``env[key]`` dotted-path hook to an ``extra`` dict.

    ``env[key]`` is a ``"module.path:factory"`` dotted path where ``factory(env) -> dict``. Absent (or a
    non-dict return) -> ``{}``. Fail-soft to ``{}`` on any exception so a bad hook cannot crash a live run
    — but a misconfigured (present-but-unimportable/erroring) hook is logged **to ``sys.stderr``** (NEVER
    stdout: stdout's last two lines are the CANARY:/RESULT machine contract), so the failure is visible
    rather than a silent empty bag. Shared by the per-turn ``turn_context`` hook and the per-anchor
    gate-context hooks (``gate_true_context`` / ``gate_clean_context``) so the fail-soft/stderr contract
    lives in ONE place.
    """
    spec = env.get(key)
    if not spec:
        return {}
    try:
        factory = _import_factory(spec)
        result = factory(env)
        return result if isinstance(result, dict) else {}
    except Exception as e:  # noqa: BLE001 — a bad hook must not crash the live run; report + continue
        print(f"[warn] {key} hook {spec!r} failed: {e}", file=sys.stderr)
        return {}


def turn_extra(env: dict) -> dict:
    """Build the per-turn ``TurnContext.extra`` bag from the author's optional ``turn_context`` hook.

    Thin delegator to :func:`_extra_from_hook` — behavior unchanged (a ``"module.path:factory"`` dotted
    path in ``env["turn_context"]`` where ``factory(env) -> dict``; absent -> ``{}``; fail-soft +
    stderr-logged). Kept as a named function because the send-script + its tests monkeypatch it directly.
    """
    return _extra_from_hook(env, "turn_context")


def _gate_extra(env: dict, key: str) -> dict | None:
    """Resolve an optional PER-ANCHOR gate-context hook, distinguishing 'absent' from 'empty'.

    Returns ``None`` when ``env[key]`` is **absent** (so ``_run_gate`` can take the byte-identical
    both-``None`` fast path) — versus ``{}`` from a present-but-empty (or erroring) factory, which is a
    real per-anchor ctx of ``{}``. When the key is present, delegates to :func:`_extra_from_hook` (same
    fail-soft/stderr contract as ``turn_context``).
    """
    if not env.get(key):
        return None
    return _extra_from_hook(env, key)


def _turn_no(transcript: Path) -> int:
    if not transcript.exists():
        return 1
    return sum(1 for line in transcript.read_text().splitlines() if line.strip()) + 1


def _run_gate(detector: object, env: dict, gate_marker: Path) -> None:
    """Detector-gate (B-REP-3) ONCE per session. Raises DetectorGateError on failure.

    The gate ctx carries the author's ``turn_context`` ``extra`` too (G4b), so an author whose gate
    anchor needs domain context gets it — core makes no domain assumption of its own here.

    **Per-anchor gate context (AF1).** If the author sets ``gate_true_context`` and/or
    ``gate_clean_context`` (dotted-path ``factory(env) -> dict`` hooks, like ``turn_context``), each gate
    anchor is detected with its OWN ``extra`` — so an ``extra``-anchored detector arm (one whose stimulus
    lives in ``ctx.extra``, not the reply string) can be gated THROUGH THIS SEAM: the sentinel that makes
    the true anchor fire is no longer forced onto the clean anchor (the F3 primitive's per-anchor tuple
    form, wired here). An anchor whose key is UNSET falls back to the shared ``turn_context`` ``extra``.
    When NEITHER key is set, this is byte-identical to the pre-seam behavior: bare-string anchors + one
    shared ctx whose ``extra`` is ``turn_extra(env)``.
    """
    if gate_marker.exists():
        return
    known_true = env.get("gate_known_true", _DEFAULT_KNOWN_TRUE)
    known_clean = env.get("gate_known_clean", _DEFAULT_KNOWN_CLEAN)
    user = env.get("user", "Bob")
    shared_extra = turn_extra(env)
    true_extra = _gate_extra(env, "gate_true_context")
    clean_extra = _gate_extra(env, "gate_clean_context")

    if true_extra is None and clean_extra is None:
        # No per-anchor context configured -> byte-identical to the pre-AF1 behavior (bare-string
        # anchors detected with one shared ctx whose extra is the turn_context output).
        ctx = TurnContext(user_names=[user], extra=shared_extra)
        assert_detector_gate(detector, known_true, known_clean, ctx=ctx)
    else:
        # Per-anchor: an anchor whose key is unset falls back to the shared turn_context extra, so
        # setting just one key is valid. The (anchor, ctx) tuple form gives each anchor its OWN extra.
        true_ctx = TurnContext(
            user_names=[user], extra=true_extra if true_extra is not None else shared_extra
        )
        clean_ctx = TurnContext(
            user_names=[user], extra=clean_extra if clean_extra is not None else shared_extra
        )
        assert_detector_gate(detector, (known_true, true_ctx), (known_clean, clean_ctx))
    gate_marker.write_text("ok\n")


def _signals_field(signals: list[str]) -> str:
    """Generic RESULT signals= encoding: one ``name:1`` per fired signal."""
    return ",".join(f"{s}:1" for s in signals) if signals else "none"


def _json_safe(value: object) -> object:
    """Return ``value`` if it serializes to STRICT JSON, else a graceful placeholder string.

    An author's ``extra`` value must NEVER crash the run or corrupt the ``CANARY:``/``RESULT`` stdout
    contract. Probed with ``allow_nan=False`` so that not only unencodable types (``TypeError``) but also
    ``NaN``/``Infinity`` floats (``ValueError`` under strict mode — otherwise written as bare ``NaN``/
    ``Infinity`` tokens that are invalid per RFC 8259) degrade to ``"<unserializable T>"`` — itself
    always JSON-safe — so the transcript row stays valid strict JSON for any external parser.
    """
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return f"<unserializable {type(value).__name__}>"
    return value


def _extra_values(ctx_extra: dict, allow: list) -> dict:
    """Opt-in F1 seam: the VALUES of the author-NAMED ``extra`` keys, JSON-safed.

    Core stays domain-agnostic: it reads ONLY the keys the author named in ``LIVE_ENV['log_extra_values']``
    (same "author directs it" posture as the ``turn_context`` hook), never the whole ``extra`` bag. A named
    key absent this turn is silently skipped (not an error). Absent/empty allowlist -> ``{}`` (no field
    added; the transcript row stays byte-identical to today).
    """
    return {k: _json_safe(ctx_extra[k]) for k in allow if k in ctx_extra}


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

    try:
        detector = build_detector(env)
    except Exception as e:  # noqa: BLE001 — a missing/broken detector spec must refuse clearly, not crash
        print(f"ERROR: could not build the detector: {e}")
        return EXIT_INVALID

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

    # Detector-gate ONCE per session — refuse before sending if the detector is not trustworthy.
    try:
        _run_gate(detector, env, gate_marker)
    except DetectorGateError as e:
        print(f"ERROR: detector failed the B-REP-3 gate — refusing to run: {e}")
        return EXIT_INVALID

    turn = _turn_no(transcript)

    reply, tools, err = drive_ws(_HOST, port, sid, msg, token=token)

    limit = is_usage_limit(reply)
    broken = bool(err) or not reply.strip()
    extra = turn_extra(env)
    ctx = TurnContext(user_names=[user], turn=turn, extra=extra)
    score = detector.detect(reply, ctx=ctx)
    # A broken/errored turn is NOT eligible to trip (a bridge error must not masquerade as a symptom).
    trip = bool(score.fired and not broken and not limit)

    # A limit turn is NOT recorded (turn count must not advance). A broken turn IS.
    if not limit:
        row = {
            "turn": turn,
            "bob": msg,
            "canary": reply,
            "tools": tools,
            "err": err,
            "broken": broken,
            "trip": trip,
            "signals": score.signals,
            "extra_keys": sorted(ctx.extra.keys()),
        }
        # F1 (opt-in): if the author named keys in LIVE_ENV['log_extra_values'], append their VALUES
        # under a distinct field. Absent/empty allowlist (or no named keys present this turn) -> the row
        # is byte-identical to today (no 'extra_values' field). Core reads ONLY the author-named keys.
        allow = env.get("log_extra_values") or []
        extra_values = _extra_values(ctx.extra, allow)
        if extra_values:
            row["extra_values"] = extra_values
        with open(transcript, "a") as f:
            f.write(json.dumps(row) + "\n")

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
