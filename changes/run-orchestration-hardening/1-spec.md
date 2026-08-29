# 1 — Spec: run-orchestration hardening (item B of #139)

## Problem

Live-arm runs repeatedly dead-stall for ~30 minutes. Root cause (issue #139,
live-verified by Testing): an orchestrator agent **backgrounds a step and ends its
turn to wait for it**. A backgrounded child — either a `run_in_background: true` Bash
task OR a background sub-agent (e.g. AgentBob) — routes its completion notification to
**main**, never back to the waiting parent. So the parent hangs until a human pokes it.
This is structurally guaranteed by the agent messaging topology, not a flaky bug: a
completed child cannot wake the specific parent that parked on it. (We lived this inside
item A's own guarded-change loop — every cold-review child routed to main.)

The fix pattern (Testing, owner-endorsed): **one arm = one agent = one turn, everything
FOREGROUND.** Boot, verify, drive all turns, pause, preserve, and report within a single
turn, every step foreground. A foreground call blocks and returns in-turn, so the
orchestrator can never park. The two surfaces that get backgrounded-and-awaited are (a) a
Bash task and (b) a sub-agent; the one rule that closes both is "foreground, one turn."

Today this contract lives only in Testing's gitignored, un-versioned draft
(`hunts/symptom-cluster-rootcause/live-run/RUN-INSTRUCTIONS.md`). There is no tracked,
shipped expression of it, no code that structurally discourages the parkable seam, and no
loud nudge at the moment of temptation (the READY point, where the orchestrator decides
whether to drive now or "wait for the run to finish"). Nothing in the tracked harness
encodes the anti-stall contract, so each new run orchestrator can re-make the same mistake.

## What we are building — a PERMANENT capability in the tracked harness (`tests/harness/`)

Three parts, all tracked:

1. **A run-orchestration contract doc** shipped at `tests/harness/RUN-ORCHESTRATION.md`.
   The tracked, versioned home of the "one arm = one turn, everything foreground" contract.
   It MUST name **both** parkable surfaces (background Bash task AND background sub-agent)
   and state the **one rule** (foreground, one turn), and it references the shipped helper
   and banner as the mechanical support for the rule. Content is adapted/refined from
   Testing's draft (read as input, NOT edited — it is Testing's gitignored lane).

2. **A single blocking foreground helper** in a new tracked module `tests/harness/foreground.py`
   that encapsulates the parkable setup and teardown steps as blocking, in-turn calls with
   no await surface to park on:
   - `boot_and_verify(...)` — one foreground blocking call that boots the arm (bounded by a
     timeout so a dead boot RAISES in-turn rather than hanging) and confirms it reached a clean
     READY: it raises on the ERROR / leak marker or on timeout, and on success parses the READY
     payload and returns a populated `ArmSession`. **Orchestrator-side verification is
     confirmation, not re-execution of the in-process guards** (see the design note below):
     `live_server.py` writes `READY` only after its in-process drop-in guard passes and writes
     `ERROR` on any hard-fail, so *READY-present-without-ERROR is the legitimate
     orchestrator-side confirmation* that the guard/roster passed. `boot_and_verify` optionally
     cross-checks the READY payload's `brain_repo` is under an expected drop-in repo root
     (an orchestrator-side file read, not a process-self check).
   - `assert_tool_callable(...)` — the orchestrator-side tool-side check (RUN-INSTRUCTIONS
     step 4): reads the turn-1 `turn_diag`/transcript and confirms a version-unique tool (e.g.
     `read_full_memory`) is actually in the served MCP roster (not "No such tool available"),
     raising `ToolSideBroken` otherwise. This is the real, load-bearing tool-side verify — a
     broken MCP child fails SILENTLY (the `claude` CLI proceeds tool-less), so it can only be
     caught from the actual served roster in the transcript, orchestrator-side, after turn 1.
   - `preserve_artifacts(...)` — one foreground blocking call that copies the raw run
     artifacts (default `turn_diag.jsonl`, `transcript.jsonl`, plus any caller-named subprocess
     transcript dir) into a preserve dir, returning a `PreserveResult`. A source named in the
     caller's `require=(...)` set that is missing **raises** (`PreservationIncomplete`) rather
     than reporting a silently-green result — the raw artifacts are load-bearing and the VM has
     crashed mid-run before.
   The **drive turns cannot be encapsulated in code** — that is the orchestrator's own
   turn-by-turn reasoning (improvising Bob), or a synchronous AgentBob call. Honestly, the
   helper covers boot+confirm, the tool-side check, and preserve; the drive necessarily sits
   between them and is governed by the DOC's one-turn rule, not by code. The API is shaped to
   discourage backgrounding: each call blocks and returns in-turn, is bounded by a timeout so it
   cannot hang, and the docstrings state loudly that it must never be wrapped in a
   `run_in_background` task.

   **Design note — why `boot_and_verify` does NOT call `assert_brain_under_dropin` /
   `assert_brain_tools_roster` (stage-3 blocker F1).** Those two item-A asserts are
   **process-self-checks**: `assert_brain_under_dropin` checks `sys.prefix ==
   build.venv_root` (dropin.py:217-219) and `assert_brain_tools_roster` first requires the
   in-process `brain` to resolve under `build.repo` (roster_preflight.py:67-68). They are only
   meaningful inside a process running under the drop-in venv — i.e. inside `live_server.py`,
   which is exactly where item A's own handoff wires `assert_brain_under_dropin`
   (`changes/harness-drop-in-code-ingest/9-handoff-live-run-callsite-edits.md:121-131`).
   `boot_and_verify` runs in the **orchestrator** process (the session / Testing's shell),
   which runs under a different venv with `brain` resolving to this clone — so calling those
   asserts there would raise `DropinMismatch` on a *healthy* run. The correct orchestrator-side
   verification is therefore READY-confirmation (their effect) plus the transcript-based
   tool-side check, not re-running the asserts in the wrong process.

3. **A loud structural nudge at the READY decision point** — a harness-owned banner function
   `drive_now_banner(...)` in `tests/harness/foreground.py` whose first line is the exact
   literal string
   `▶ DRIVE NOW, IN THIS SAME TURN — do not end your turn to wait`
   followed by run context and a one-line restatement of the two parkable surfaces + the one
   rule. Because `run_arm.sh` is Testing's gitignored lane, we do NOT edit it; the banner
   text lives in this harness-owned function (the permanent capability), and a thin one-line
   call-site edit is documented for Testing to wire it into `run_arm.sh`'s READY summary
   output. (Same handoff pattern item A used for the roster preflight.)

## Design directions and rationale

- **New module `tests/harness/foreground.py`**, mirroring the shape item A used for
  `roster_preflight.py`: plain synchronous functions raising typed exceptions on failure,
  returning populated result objects on success; exported from `tests/harness/__init__.py`.
- **Boot is a foreground, bounded subprocess seam.** `boot_and_verify` accepts an optional
  `boot_cmd` (a command sequence). When given, it runs it via `subprocess.run(boot_cmd,
  timeout=ready_timeout)` — FOREGROUND, blocking, bounded; a non-zero exit or a timeout
  raises in-turn. When omitted, it does not start anything and instead waits (bounded poll +
  timeout) for the READY marker that an already-foregrounded boot (Testing's `run_arm.sh`,
  which itself already blocks until READY) has produced. Either way the call is bounded and
  cannot hang. The tracked helper hardcodes NO gitignored path — Testing passes its
  `run_arm.sh` invocation as `boot_cmd` if it wants the whole boot+verify to be one call.
- **Verify is orchestrator-side and real (no injected proxy).** `boot_and_verify` confirms the
  booted arm reached a clean READY (raises on ERROR/leak/timeout) and optionally cross-checks
  the READY payload's `brain_repo` against an expected drop-in root; `assert_tool_callable`
  reads the real turn-1 transcript for the served MCP roster. Both are deterministic
  orchestrator-side file operations, so the unit tests exercise the **real** logic against
  crafted READY/transcript fixtures — no live bridge, no tokens, and no injected-fake dodge.
  Item A's process-self asserts stay where they belong (inside `live_server.py`); their
  correctness is covered by item A's tests (`test_roster_preflight.py`, `test_dropin.py`).
- **Banner is a pure function** returning a string with the exact literal first line — a
  trivially discriminating unit test asserts the exact text.

### Note on the em-dash in the banner text (pre-empting a likely review flag)
The project's "no claudisms in prompt strings" rule bans em-dashes in **persona-facing /
prompt strings** — text injected into the companion's own context. The banner is
**operator-facing** (printed to the human/agent running the harness, into `run_arm.sh`'s
READY output); it is never injected into any companion context. The exact banner text,
including its em-dash, is specified verbatim by the item-B charter. It is retained verbatim by
design; the no-claudisms rule does not apply to operator-facing operational output (the
existing `tests/harness/README.md` uses em-dashes and arrows freely, consistent with this).

## Expected touched files (tracked, on branch `ThinkerOfThoughts/harness-drop-in-code-ingest`)

- **NEW** `tests/harness/foreground.py` — `boot_and_verify`, `assert_tool_callable`,
  `preserve_artifacts`, `drive_now_banner`, result dataclasses, and the typed exceptions
  (`ForegroundBootError`, `ToolSideBroken`, `PreservationIncomplete`).
- **NEW** `tests/harness/RUN-ORCHESTRATION.md` — the tracked, canonical anti-stall contract doc.
- **NEW** `tests/unit/harness/test_foreground.py` — unit tests (mechanical facts only).
- **EDIT** `tests/harness/__init__.py` — export the new public symbols in `__all__`.
- **EDIT** `tests/harness/README.md` — replace the orchestration-protocol section's intro with
  a prominent pointer to `RUN-ORCHESTRATION.md` as the canonical anti-stall contract (single
  source of truth; avoids the two-copies drift the stage-3 review flagged as F4).
- **NEW** `changes/run-orchestration-hardening/9-handoff-drive-banner-callsite.md` — thin
  call-site handoff for Testing (banner wire-in to `run_arm.sh` + how to call the helper).
- Plus this change folder's stage docs.

## Explicitly OUT of scope / hard constraints

- **Tracked `tests/harness/` ONLY.** Do NOT edit the gitignored
  `hunts/symptom-cluster-rootcause/live-run/` files (`run_arm.sh`, `live_server.py`,
  `RUN-INSTRUCTIONS.md`) or the main clone. Call-site edits are documented as a handoff.
- **No production (`brain/**`) changes.** None are needed. If the build finds it needs one,
  STOP and ask the orchestrator.
- **No behavioral agent-compliance test.** The anti-stall guarantee is structural/contractual
  (see `1.5-criteria.md` and `8-harness.md`), not enforced by an agent-behavior harness —
  that is a known money-pit here (the measurement-apparatus trap). We prove only the
  mechanical facts.
- Lands on the #137 branch; #139 closes on merge (referenced in the commit).
