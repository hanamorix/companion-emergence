# Behavioral-test harness

A permanent, **sandboxed** framework for behavioral tests that drive the **real**
companion-emergence engine — as opposed to unit tests that mock it. It generalizes the ad-hoc
harness built during the monologue-bleed bug hunt.

The shape of a behavioral test:

> seed a throwaway persona (**Canary**) → stand up the real bridge → drive it with an
> LLM-simulated human (**Bob**) → run a **detector** over the replies → optionally orchestrate
> multiple **arms** that survive usage-limit stalls.

It can exercise ~anything except the GUI.

## Safety first: strong sandbox isolation

The harness is expected to run on developer **laptops** where a real companion lives. Therefore
the #1 design requirement is that **nothing it does may touch anything outside its temporary
sandbox**. Every run goes through the `sandbox()` context manager, which:

- redirects `KINDLED_HOME` (all persona state) and `CLAUDE_CONFIG_DIR` (the `claude` CLI) into a
  fresh temp dir;
- seeds **only** the claude auth credential (never your global `CLAUDE.md` / skills / settings);
- forces the test persona to `notes_enabled=false` + `kindled_relay_url=null` (the only gated
  external write / phone-home paths);
- **fingerprints guarded real-home roots before the run and asserts they are unchanged after** —
  a leak fails the test loudly.

## What it can / can't do

**Can:** drive real multi-turn conversations against a real persona + bridge; build "aged" personas
(a real multiply-folded compaction/incident regime) as fixtures; plug in your own symptom detector;
run multi-arm A/B matrices; survive usage-limit stalls and resume.

**Can't (yet):** test the GUI; run the token-spending behavioral runs in CI (they need real `claude`
auth and cost money — they're marker-gated and run manually/locally); Agent-Bob (a persistent-agent
user-sim) lands in a later phase.

## The isolation guarantees (what `sandbox()` promises)

Every run is a `with sandbox() as sb:` block. Inside it:

- `KINDLED_HOME` → `sb.root`, so **all** persona state (personas, memories, buffers) lands in the
  tempdir. This is the real engine mechanism (`brain/paths.py:58-82`), not a proxy.
- `CLAUDE_CONFIG_DIR` → `sb.root/claude-config`, which the provider respects
  (`brain/bridge/provider.py:174`), so the `claude` CLI reads the sandbox's config, not your
  `~/.claude`.
- `NELLBRAIN_HOME` is **unset** (a stray value would otherwise win the fallback).
- **Auth-only seed:** only `~/.claude/.credentials.json` is copied in — never your `CLAUDE.md`,
  settings, skills, or plugins. On a Mac the credential may be in the Keychain; a fresh
  `CLAUDE_CONFIG_DIR` still authenticates via Keychain (recorded on `sb.auth_source`).
- Every persona is forced to `notes_enabled=false` + `kindled_relay_url=null` — the only two gated
  external-write / phone-home paths.
- The real guarded home roots (the whole platformdirs data/cache/state/log/config family for
  `companion-emergence`, `~/.claude`, plus OS autostart/notes dirs) are **fingerprinted before and
  after** the run; any change raises `SandboxLeak`.
- The env is restored (even if `SandboxLeak` raises) and the tempdir removed (`keep=True` to keep
  it for a post-mortem).

**Concurrency:** `sandbox()` mutates process-global `os.environ` — it is not thread-safe or
nestable, and assumes serial use in a process. pytest-xdist workers are separate processes, so
parallel CI is fine.

**Operational caveat — quit your own companion first.** The leak assertion fingerprints the *real*
companion-emergence data/cache/state dirs. If your **own companion's bridge/service is running**
during a harness run, it writes to those dirs concurrently and will trip a **spurious `SandboxLeak`**
(a false positive — it's *safe*, nothing is corrupted, but it aborts the run). Stop your companion
(and any `launchd`/`systemd` service) before running the behavioral example. The token-free unit
tests run in well under a second, so the window is tiny; longer behavioral runs are the exposure.
_(Follow-up: a runtime pre-check could detect a live service and warn/skip instead of false-tripping.)_

## Usage — authoring a behavioral test

```python
from tests.harness import (
    sandbox, PersonaSpec, MemorySeed, build_persona,
    DumbBob, BobContext, RegisterLeakDetector, TurnContext, assert_detector_gate,
)
from tests.harness.engine import BridgeServer

# 1. Validate your detector on anchors BEFORE trusting it (B-REP-3).
detector = RegisterLeakDetector()
assert_detector_gate(detector, known_true="note to self: land it lightly.", known_clean="yeah, how's the knee?")

# 2. Run inside the sandbox.
with sandbox() as sb:
    live = build_persona(
        PersonaSpec(memories=[MemorySeed(content="Bob's dog Biscuit is a border collie.")]),
        sb,
    )
    server = BridgeServer(live.persona_dir, port=8931)
    server.start()
    try:
        bob = DumbBob("/path/to/claude", mood="...ongoing companion chat...")
        ctx = BobContext(neutral_cwd="/tmp/neutral", user=live.user)
        history = []
        sid = ...  # POST /session/new
        for turn in range(1, 6):
            bt = bob.next_message(history, turn=turn, ctx=ctx)
            history.append(("bob", bt.text))
            reply, tools, err = server.drive_turn(sid, bt.text)
            history.append(("canary", reply))
            score = detector.detect(reply, ctx=TurnContext(user_names=[live.user], turn=turn))
            if score.fired:
                ...  # adjudicate the trip
    finally:
        server.stop()
```

An "aged persona" fixture (a real multiply-folded compaction regime) is built by passing a
`PersonaSpec(incident=IncidentSpec(...))` + a compaction provider to `build_persona`.

**Model toggle:** pass a `ModelConfig(canary=..., bob=..., watchdog=...)` to `build_persona` /
`DumbBob` / `Watchdog` to override the defaults (`sonnet`/`sonnet`/`haiku`).

**Multi-arm + usage stalls:** `Runner(arms, state_path, drive_fn)` runs a list of `ArmSpec`s,
checkpoints on a usage stall (exit 20) and resumes, parks a detector trip (exit 10) and continues.
`Watchdog(ping_fn, marker_path)` detects the usage-reset recovery edge.

## Running the tests

```bash
# The framework's own token-free unit tests (CI; cross-platform; 0 model tokens):
uv run pytest tests/unit/harness/ -v
uv run ruff check tests/harness tests/unit/harness

# The worked behavioral example — SPENDS TOKENS, needs claude auth (NOT in default CI):
uv run pytest tests/harness/examples/test_register_leak.py -v -m behavioral
```

Exclude behavioral runs from a full-suite CI invocation with `-m "not behavioral"`.

## Status

Phase 1 (Dumb-Bob, end-to-end + sandbox isolation) — **built.** AgentBob (Phase 3) is a documented
stub; the authoring skill (Phase 4) is later.
