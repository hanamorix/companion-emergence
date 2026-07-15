---
name: live-test-authoring
description: Guides authoring a live/behavioral test with the tests/harness/ framework — the ordered flow that drives the REAL companion-emergence engine (seed a Canary persona, author a detector, validate it on anchors before trusting it, drive it with a Bob user-sim, run inside the sandbox, adjudicate trips). Use this whenever writing a live test or behavioral test, building a symptom Detector, standing up a Canary fixture, wiring a DumbBob or Agent-Bob user-simulator, driving the real bridge from a test, or setting up a multi-arm A/B run — even if the request just says "test that the companion does X" rather than naming the harness. It is a PROCEDURE, not a reference; the framework's own tests/harness/README.md is the reference for every knob.
---

# Authoring a live test with `tests/harness/`

A **live test** drives the *real* companion-emergence engine (everything but the GUI), unlike a
unit test that mocks it. The shape is always:

> seed a throwaway **Canary** persona → stand up the real bridge → drive it with an LLM-simulated
> human (**Bob**) → run a **detector** over the replies → **adjudicate** any trip.

This skill is the **order and the discipline** — do the steps in sequence, and do not skip the
one gate that makes a run trustworthy (step 3). For the full API surface and every knob, read
**`tests/harness/README.md`** (the reference); this skill points at it rather than restating it.

**⛔ You USE the framework; you never edit it.** `brain/` and `tests/harness/*.py` are frozen
(drop-in invariant — a new companion-emergence version drops in). Author your test *around* the
harness. The synthetic names are fixed: the persona is **Canary**, the user is **Bob** — never a
real person's name in a fixture.

Live runs spend tokens and need `claude` auth, so they are **marker-gated and out of CI**
(`-m live`). The framework's own logic is covered by token-free unit tests
(`uv run pytest tests/unit/harness/`); the worked end-to-end example is
`tests/harness/examples/test_generic_run.py` — the best thing to copy from.

Follow the six steps **in this order**. The order is load-bearing (see "Why the order matters" at
the end): most of all, **step 3 comes before any run.**

---

## Step 1 — Build a REPRESENTATIVE Canary fixture

Seed a throwaway persona via `build_persona(spec, sb)`, which returns a `LiveEnv`. The spec is a
`PersonaSpec` (a voice, `MemorySeed`s, and optionally an aged/compacted regime):

```python
from tests.harness import PersonaSpec, MemorySeed, build_persona

# inside a `with sandbox() as sb:` block (step 5):
live = build_persona(
    PersonaSpec(memories=[MemorySeed(content="Bob is teaching himself sourdough.")]),
    sb,
)
```

For an **aged persona** (a real multiply-folded compaction/incident regime, built by the real
fold — not a hand-written proxy), pass `PersonaSpec(incident=IncidentSpec(...))` plus a compaction
provider to `build_persona`; see `IncidentSpec` / `build_compacted_state` in `incident.py` and the
README's aged-persona note.

**Hold the discipline — representativeness.** A fixture (or control) must actually **exhibit the
phenomenon** you are testing before you trust any result from it. A persona that can never surface
the symptom turns a green run into noise: the test proves nothing. Ask, before you run: *would this
Canary even be capable of the behavior my detector looks for?* If not, enrich the fixture (more
memories, an incident regime, a voice that invites the behavior) until it can.

---

## Step 2 — Author a `Detector`

A detector runs over the persona's reply each turn and reports whether a symptom fired. **The
framework ships NO detector and makes no assumption about what you inspect — you supply it.** A
detector is any object satisfying the `Detector` protocol: `detect(reply, *, ctx: TurnContext) ->
Score`. Return a valid `Score` for *any* input, including `None`/`""`.

```python
from tests.harness import Score

# A trivial illustrative detector: flag a banned keyword.
class KeywordDetector:
    def __init__(self, banned):
        self.banned = tuple(b.lower() for b in banned)
    def detect(self, reply, *, ctx=None):
        hits = [b for b in self.banned if b in (reply or "").lower()]
        return Score(fired=bool(hits), signals=[f"keyword:{h}" for h in hits])
```

`Score(fired, signals, detail)` carries the trip flag, the fired signal names (for logs and
adjudication), and arbitrary per-signal detail.

If a detector needs **domain-specific per-turn context** (a reference block to compare against, a
proposed file, a retrieved doc), it reads it from `ctx.extra` — an author-namespaced bag on
`TurnContext`. **Core never reads or writes any key of `extra`.** You populate it via the
`turn_context` hook (a `"module:factory"` dotted path in `LIVE_ENV`; `factory(env) -> dict`).

**Attach the detector through the general seam** — the framework has no detector registry, you
name your own code:
- **For the Agent-Bob send-script:** `LIVE_ENV["detector"] = "my_pkg.my_detectors:make_detector"`
  (a `module:factory` dotted path; required — the send-script refuses to run without one).
- **For a `DumbBob` pytest:** construct the detector object and pass it directly (as in the sketch
  above). Combine several with `CompositeDetector(*detectors)` — it ORs `fired` and unions
  `signals` while keeping each sub-detector independently gate-able.

---

## Step 3 — VALIDATE the detector on anchors BEFORE you trust it (B-REP-3) — the load-bearing gate

**This is the single most important discipline in the whole flow — do it before any run.** A
detector that fires on *everything*, or on *nothing*, proves nothing — and you cannot tell which
by staring at its code. Prove it discriminates: give it a `known_true` anchor it MUST fire on and a
`known_clean` anchor it MUST stay silent on.

```python
from tests.harness import assert_detector_gate

detector = KeywordDetector(banned=("secret_token",))
assert_detector_gate(
    detector,
    known_true="here is the SECRET_TOKEN",
    known_clean="how's your evening?",
)
```

`assert_detector_gate(detector, known_true, known_clean, *, ctx=None)` raises `DetectorGateError`
if the detector fails either anchor — refusing to run rather than silently trusting a broken
detector. The send-script runs this gate **once per session** automatically (before it sends
anything), using the `gate_known_true` / `gate_known_clean` anchors from `LIVE_ENV`.

**Anchor whose stimulus lives in `ctx.extra`?** If your detector keys off `ctx.extra` (not the
reply string), a bare-string anchor cannot carry the stimulus, and forcing one shared `ctx` would
leak the true-anchor's sentinel onto the clean anchor. Use the **per-anchor ctx form** — pass each
anchor as an `(anchor_text, ctx)` tuple so each carries its OWN `extra`. Through the send-script,
set `gate_true_context` / `gate_clean_context` (same `module:factory` shape as `turn_context`) so
the true anchor gets the sentinel and the clean one does not.

**Never skip this gate to "save time."** An ungated detector makes every downstream result
meaningless; the minutes here are the cheapest insurance in the run.

---

## Step 4 — Choose the Bob tier

**Bob is the substitute-USER** — the human half of the run. Whichever tier you pick, Bob **REACTS
to the persona's actual reply each turn; he never runs a fixed script** (a fixed script injects
non-sequiturs the moment the persona says something unexpected). Two tiers, same role:

**`DumbBob` — a pull simulator.** The harness owns the loop and pulls the next line each turn via a
fresh `claude -p` call. Simplest; good for a `DumbBob` pytest:

```python
from tests.harness import DumbBob, BobContext, TurnContext

bob = DumbBob("/path/to/claude", mood="...ongoing companion chat...")
ctx = BobContext(neutral_cwd="/tmp/neutral", user=live.user)
history = []
sid = ...  # POST /session/new  (get a fresh session id from the bridge)
for turn in range(1, 6):
    bt = bob.next_message(history, turn=turn, ctx=ctx)
    history.append(("bob", bt.text))
    reply, tools, err = server.drive_turn(sid, bt.text)   # server from step 5
    history.append(("canary", reply))
    score = detector.detect(reply, ctx=TurnContext(user_names=[live.user], turn=turn))
    if score.fired:
        ...  # adjudicate (step 6)
```

**Agent-Bob — agent-drives-the-loop.** The cheaper / continuous-context variant: a spawned
**Agent-tool subagent** holds the whole conversation in its own context and drives the loop
itself, calling the `agent_send` script each turn and stopping on a trip/limit/max-turns.
`AgentBob(mood, *, harness_dir, live_env_path, max_turns, models)` is a **renderer**, not a
`Bob` — call `.spawn_params()` to get the spawn contract; it does not implement `next_message`.
Because the Agent tool is a claude-code-runtime capability, an Agent-Bob run is
**orchestrator-driven**: the orchestrating claude-code **session** stands up the bridge, spawns
Agent-Bob, and **adjudicates each trip** from the on-disk transcript (step 6). Before spending
Agent tokens, run the **loopback smoke** yourself — one send confirms loopback reachability:

```bash
LIVE_ENV=<env.json> ./tests/harness/agent_send.sh --new "hey"
```

**Running more than one arm?** Assemble arms into a `Runner(arms, state_path, drive_fn)` over a
list of `ArmSpec`s for an A/B matrix that checkpoints on a usage stall and resumes; see
`runner.py` and the README's multi-arm note.

---

## Step 5 — Run inside `sandbox()`

**Every run goes through `with sandbox() as sb:`** — this is the safety core, because the harness
is expected to run on developer laptops where a real companion lives. The context manager redirects
`KINDLED_HOME` + `CLAUDE_CONFIG_DIR` into a fresh tempdir, seeds only the claude auth credential
(never your global `CLAUDE.md`/skills/settings), and **fingerprints the guarded real-home roots
before the run and asserts they are unchanged after** — any leak raises `SandboxLeak` and fails the
test loudly.

```python
from tests.harness import sandbox

with sandbox() as sb:
    live = build_persona(PersonaSpec(memories=[...]), sb)
    server = BridgeServer(live.persona_dir, port=8931)   # from tests.harness.engine
    server.start()
    try:
        ...  # drive the loop (step 4)
    finally:
        server.stop()
```

Knobs, all with safe defaults — reach for them only when needed:
- **`live_check=`** — the live-companion pre-check. Default `LIVE_CHECK_RAISE` fails fast with
  `LiveServiceDetected` if your own companion's bridge is running (it would trip a spurious
  `SandboxLeak`); `LIVE_CHECK_WARN` warns and continues; `LIVE_CHECK_OFF` skips it (use in CI,
  where no live bridge exists). Quit your own companion bridge before a run.
- **`editable_paths=`** — the sandbox-boundary extension, **only if** your test must exercise real
  writes OUTSIDE the sandbox. It is sentinel-guarded: **drop the sentinel file
  (`HARNESS_EDITABLE_SENTINEL`, i.e. `.ce-harness-editable`) into the pre-existing target folder,
  or the path is refused with `EditablePathRefused`.** Pass the *same* set to
  `sandbox(editable_paths=...)` and to your confirm step, or the post-run `SandboxLeak` catches the
  mismatch fail-safe.
- **Persona config (author-controlled, safe defaults, opt-in via `PersonaSpec`):**
  `notes_enabled=True` enables a notes folder (a real write there is caught by `SandboxLeak` unless
  the folder is a declared `editable_path`); `kindled_relay_url=...` allows a **network phone-home
  the filesystem leak guard CANNOT catch**, so it stays off by default and emits a loud
  `RuntimeWarning` at build. Leave both off unless you deliberately need them.

---

## Step 6 — Adjudicate trips

A detector trip is a **flag to investigate, not a verdict.** On a trip, read the transcript and
rule **false-positive vs. real** — who rules depends on the tier:
- **DumbBob:** the test code rules inline (it has the `reply`, `signals`, and `ctx` in hand).
- **Agent-Bob:** the **orchestrator** (the claude-code session) rules **from the on-disk
  transcript** (`<sandbox>/transcript.jsonl`) — each row carries `canary`, `signals`, and
  `extra_keys` (which of your `turn_context` keys were present that turn). Agent-Bob just STOPS and
  reports `TRIP at turn N`; the orchestrator SendMessages it "false positive, continue" or "real,
  stop." Interpret the `signals` with knowledge of YOUR detector.

**Hold the discipline — representativeness is held orchestrator-side.** A clean (no-trip) run only
means something if the stimulus for the symptom actually occurred. If the conversation never
exercised the behavior your detector watches for, mark the run **untrusted-negative** — a green
result there is not evidence of absence, it is evidence the test did not run its own premise. This
is the step-1 representativeness judgment, enforced at the end.

---

## Why the order matters

This flow is deliberately ordered, and the order encodes behavior — reordering it teaches the
wrong thing even if every word survives. In particular:

- **Step 3 (validate the detector) MUST precede steps 4–6 (any run).** Trusting an ungated
  detector — running first and validating later — is exactly the failure the gate exists to
  prevent: you would spend tokens gathering results from an instrument you have not shown to
  discriminate. Gate first, run second.
- **Step 1 (representativeness) frames step 6 (adjudication).** The "would this fixture even
  exhibit the symptom?" question you ask up front is the same one that decides whether a clean run
  is trustworthy at the end.

Keep the sequence. If you are tempted to "tidy" it — move the gate after the run, fold the steps
together — that tidy quietly removes the discipline the flow is built to enforce.
