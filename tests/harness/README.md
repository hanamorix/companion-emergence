# Live-test harness

A permanent, **sandboxed** framework for live tests that drive the **real** companion-emergence
engine — as opposed to unit tests that mock it. ("Live" because it exercises everything except the
GUI, not only model *behavior*.) It generalizes an ad-hoc harness built during a prior bug hunt.

The shape of a live test:

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

**Can't (yet):** test the GUI; run the token-spending live runs in CI (they need real `claude`
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

**Operational caveat — quit your own companion first (now auto-detected).** The leak assertion
fingerprints the *real* companion-emergence data/cache/state dirs. If your **own companion's
bridge/service is running** during a harness run, it writes to those dirs concurrently and would
trip a **spurious `SandboxLeak`** (a false positive — it's *safe*, nothing is corrupted, but it
aborts the run).

To make that failure clear instead of misleading, `sandbox()` runs an **automatic live-service
pre-check** at entry: it scans the engine's real home (`brain.paths.get_home()/personas/*/bridge.json`)
and uses the real `state_file.pid_is_alive` to detect a running bridge **up front**, raising a
distinct **`LiveServiceDetected`** with a "quit your companion bridge (and any
`launchd`/`systemd`/task-scheduler service) first" message — *before* the run starts, rather than
dying later with a confusing `SandboxLeak`. It is **read-only** (it parses `bridge.json` bytes
directly and never calls the heal-on-read `state_file.read()`), so it never writes to your real
files.

Control it with the `live_check=` kwarg: `"raise"` (default) fails fast; `"warn"` warns and
continues (and annotates any later `SandboxLeak` so you know it was your bridge); `"off"` skips the
check (use in CI, where no live bridge exists). An optional `probe=True` adds a double-fingerprint
liveness probe as a complementary net for an external writer that has no discoverable `bridge.json`
(a probe-only hit reports a generic "external writer" message, not a companion-specific one).

Known limit: a live bridge whose `bridge.json` is *currently corrupt but recoverable from a `.bak`*
is not detected by the read-only scan (detecting it would require the heal-on-read path, which
writes) — the post-run `SandboxLeak` remains the backstop for that rare case.

## Usage — authoring a live test

**You supply the detector.** The framework ships NO detector and makes no assumption about what a
detector inspects. A detector is any object with `detect(reply, *, ctx) -> Score`; validate it on
anchors before trusting it (B-REP-3), then score each reply. If a detector needs domain-specific
per-turn context, it reads it from `ctx.extra` (an author-namespaced bag) — core never touches `extra`.

```python
from tests.harness import (
    sandbox, PersonaSpec, MemorySeed, build_persona,
    DumbBob, BobContext, Score, TurnContext, assert_detector_gate,
)
from tests.harness.engine import BridgeServer


# 0. Author your own detector (this trivial one just flags a banned keyword).
class KeywordDetector:
    def __init__(self, banned): self.banned = tuple(b.lower() for b in banned)
    def detect(self, reply, *, ctx=None):
        hits = [b for b in self.banned if b in (reply or "").lower()]
        return Score(fired=bool(hits), signals=[f"keyword:{h}" for h in hits])

# 1. Validate your detector on anchors BEFORE trusting it (B-REP-3).
detector = KeywordDetector(banned=("secret_token",))
assert_detector_gate(detector, known_true="here is the SECRET_TOKEN", known_clean="how's your evening?")

# 2. Run inside the sandbox.
with sandbox() as sb:
    live = build_persona(
        PersonaSpec(memories=[MemorySeed(content="Bob is teaching himself sourdough.")]),
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

The worked example `examples/test_generic_run.py` runs exactly this loop end-to-end (marker-gated).

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

# The worked generic example — SPENDS TOKENS, needs claude auth (NOT in default CI):
uv run pytest tests/harness/examples/test_generic_run.py -v -m live
```

Exclude live runs from a full-suite CI invocation with `-m "not live"`.

## Phase 3 — Agent-Bob (agent-drives-the-loop)

Agent-Bob is the **same substitute-USER role as Dumb-Bob**, but the cheaper / continuous-context /
sometimes-more-capable variant: a spawned **Agent-tool subagent** that holds the whole conversation
in its own context and **drives the loop itself**, calling a send-script each turn and stopping on a
trip/limit/max-turns. Because the Agent tool is a claude-code-runtime capability (not importable),
the live run is **orchestrator-driven — it spends tokens and is NOT a pytest / NOT in CI.** The
checked-in, token-free surface is the *mechanism*: the send-script (`agent_send.py`) and the
spawn-prompt/param renderer (`AgentBob`).

**Who runs this: the orchestrating claude-code SESSION, not a human.** The human only launches a run
and reviews results; the session stands up the bridge, spawns Agent-Bob, and **adjudicates each trip
programmatically** from the on-disk transcript.

### The orchestration protocol (what the session executes)

```python
from tests.harness import sandbox, PersonaSpec, MemorySeed, build_persona, AgentBob, ModelConfig
from tests.harness.engine import BridgeServer
import json, pathlib

with sandbox() as sb:                                   # server-side containment (unchanged)
    live = build_persona(PersonaSpec(memories=[MemorySeed(content="Bob is teaching himself sourdough.")]), sb)
    server = BridgeServer(live.persona_dir, port=8931)  # stood up with auth_token=None (engine.py)
    server.start()
    try:
        # 1. write the LIVE_ENV the send-script reads. You MUST attach your own detector via a
        #    "module.path:factory" dotted path — the framework ships NO default detector. Optionally
        #    attach a turn_context hook (factory(env) -> dict) to populate ctx.extra with domain data.
        live_env = sb.root / "live_env.json"
        live_env.write_text(json.dumps({
            "port": 8931, "kindled_home": str(sb.root),
            "persona_dir": str(live.persona_dir), "user": live.user,
            "detector": "my_pkg.my_detectors:make_detector",       # REQUIRED — your own detector
            "turn_context": "my_pkg.my_context:make_extra",        # OPTIONAL — domain ctx.extra
        }))
        # 2. LOOPBACK SMOKE (A3): the session itself runs one send BEFORE spawning Bob — confirm a
        #    CANARY:/RESULT line comes back (proves loopback reachability under network isolation).
        #    A connection error => the agent needs sandbox-disabled Bash; abort before Agent tokens.
        #       LIVE_ENV=<live_env> ./tests/harness/agent_send.sh --new "hey"
        # 3. render the spawn contract + spawn Agent-Bob via the Agent tool. `mood` is YOUR string.
        spec = AgentBob(
            mood="...your author-supplied test-user mood...", harness_dir=str(pathlib.Path.cwd()),
            live_env_path=str(live_env), max_turns=30, models=ModelConfig(bob="sonnet"),
        ).spawn_params()
        # Agent tool: prompt=spec.prompt, model=spec.model, effort=spec.effort  (== "low")
        # 4. Agent-Bob drives via ./tests/harness/agent_send.sh; on `RESULT ... trip=true` it STOPS + reports.
        # 5. ADJUDICATE PROGRAMMATICALLY: read sb.root/transcript.jsonl row N — rule fp/real from
        #    `canary` + `signals` + `extra_keys` (which of your turn_context keys were present that turn).
        #    Interpret the signals with knowledge of YOUR detector. SendMessage the agent
        #    "false positive, continue" or "real, stop".
    finally:
        server.stop()
```

Token accounting: Canary = the transcript rows; Bob = the subagent's own reported usage (much
cheaper than the per-turn `claude -p` DumbBob — continuous context, no system-prompt reload).

The send-script ships with a **bash wrapper** (`agent_send.sh`) that locates the repo venv python and
runs the module from the repo root. Smoke it on a bash host:
`LIVE_ENV=<env.json> ./tests/harness/agent_send.sh --new "hi"`.

`AgentBob` is a **driver/renderer, NOT a `Bob`** — it renders the spawn prompt + params and does not
implement `next_message` (calling it raises). The pull-based `Runner`/`DumbBob` path is unchanged.

**Scope note — a file-editing mood.** If your author-supplied mood has Agent-Bob issue real
`Write`/`Edit` tool calls (e.g. it owns a notes file it reworks with Canary), those writes are the
**agent's own** filesystem actions — they are NOT covered by the send-script's sandbox guard (which
only owns the transcript/sid/gate files). It is the **orchestrator's responsibility** to scope where
the agent may write (spawn it with a doc path INSIDE the sandbox, and rely on the claude-code runtime's
own permissions) — the server-side `sandbox()` guarantee does not extend to the client agent's
Write/Edit cwd.

## Status

Phase 1 (Dumb-Bob, end-to-end + sandbox isolation) — **built.** Phase 2 (live-service pre-check) —
**built.** Phase 3 (Agent-Bob renderer + `agent_send.py` send-script + the general detector-attachment
seam) — **built**; the live run is orchestrator-driven (marker-gated, not in CI). The authoring skill
(Phase 4) is later.
