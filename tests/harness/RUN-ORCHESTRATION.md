# RUN-ORCHESTRATION.md — the anti-stall contract for live-arm runs

Read this BEFORE booting, driving, or tearing down a live-test arm. It exists because
orchestrators keep dead-stalling for ~30 minutes at a time, and the stall is always the same
one mistake. This is the tracked, canonical home of the contract; do not duplicate the rule
elsewhere (`README.md`'s orchestration-protocol section points here instead of restating it).

## THE ONE RULE

**One arm = one agent = one turn, everything FOREGROUND.**

Boot, verify, drive all the turns, pause, preserve, and report **within a single turn**,
running every step in the FOREGROUND. Never background a step and end your turn to wait for
it.

## Why this is not optional

There are exactly **two parkable surfaces**, and both fail the same way:

1. A **`run_in_background: true` Bash task.**
2. A **background sub-agent** (e.g. an AgentBob spawned without `run_in_background: false`).

Each one routes its completion notification to **main**, never back to the specific
orchestrator that parked on it. This is structurally guaranteed by the agent messaging
topology, not a flaky bug: a completed child cannot wake the specific parent that parked on it.
The moment you end your turn waiting on either surface, nothing wakes you — you hang until a
human notices and pokes you. This has cost multiple ~30-minute dead stalls.

A FOREGROUND call blocks and returns to you in-turn, so you can never park on it. That is what
`tests/harness/foreground.py` gives you for the steps that CAN be encapsulated in code:
`boot_and_verify`, `assert_tool_callable`, and `preserve_artifacts`. The drive itself (the
turn-by-turn conversation) cannot be encapsulated in code — it is either your own inline
reasoning or a **synchronous** AgentBob call — but it stays inside the same rule: run it in
this same turn, foreground, never backgrounded.

**If you catch yourself about to end your turn with a background task or sub-agent still
pending: STOP. That is the bug. Foreground it instead.**

## THE ONE-TURN SEQUENCE

All of the following happens in a single turn:

1. **Clean slate first.** No other `live_server.py` / `box_mirror` / `brain.mcp_server`
   running, nothing listening on the target port. One harness server at a time — two
   concurrent servers cause silent `ws:ConnectionClosedOK` turn failures.
2. **Boot + confirm, FOREGROUND, via `boot_and_verify`.** Call
   `tests.harness.foreground.boot_and_verify(spec, boot_cmd=...)` (or, if the boot is already
   an external foregrounded process such as Testing's `run_arm.sh`, call it with `boot_cmd=None`
   to bounded-poll for the `READY` it writes). It blocks, is bounded by `spec.ready_timeout`,
   and raises `ForegroundBootError` in-turn on a non-zero boot, an error marker, a bad/absent
   READY, or a timeout. It never hangs.
3. **Tool-side check, after turn 1, via `assert_tool_callable`.** The startup guard confirmed
   by step 2 is PROMPT-SIDE ONLY: a broken MCP tool child fails SILENTLY — the `claude` CLI
   proceeds tool-less, no abort. After turn 1, call
   `tests.harness.foreground.assert_tool_callable(source, "read_full_memory")` (or whatever
   tool is unique to the version under test) against the turn's `turn_diag`/transcript. It
   raises `ToolSideBroken` if the tool is not actually in the served roster. If it raises: STOP
   and report "TOOL-SIDE BROKEN" — do not run a silently tool-less arm.
4. **Drive the turns, FOREGROUND.** An improvising "Bob" who reacts to each reply (no canned
   lines). Two valid ways, both foreground, both in THIS turn:
   - **Inline (preferred, zero nesting, cannot park):** send each turn yourself, read the
     reply, compose the next.
   - **OR** spawn AgentBob with **`run_in_background: false`** (synchronous — returns to you
     in-turn). NEVER background it.
5. **Pause + preserve immediately, FOREGROUND, via `preserve_artifacts`.** Write the pause
   marker (not the STOP/teardown sentinel). Call
   `tests.harness.foreground.preserve_artifacts(run_dir, subproc_dirs=(...), require=(...))`
   right away — the raw artifacts are load-bearing (the VM has crashed mid-run before) and a
   `require=`d source that's missing raises `PreservationIncomplete` rather than reporting a
   silently-green preserve.
6. **Report** to whoever spawned you (paths, per-turn tool lists, drive validity), then hold.

## The banner

`tests.harness.foreground.drive_now_banner(...)` renders the loud nudge at the READY decision
point — the moment an orchestrator is most tempted to "wait for the run to finish." Its first
line is always the literal headline

```
▶ DRIVE NOW, IN THIS SAME TURN — do not end your turn to wait
```

followed by the run context and a restatement of the two parkable surfaces + the one rule.
Wire it into your boot's READY summary output (see the handoff note for the exact call-site
edit); read it, believe it, act on it.

## NEVER background the helper calls

`boot_and_verify`, `assert_tool_callable`, and `preserve_artifacts` are all FOREGROUND,
blocking, bounded calls. Do NOT wrap any of them in a `run_in_background: true` Bash task or a
background sub-agent — doing so recreates exactly the stall this module exists to prevent, just
one layer further down. Call them directly, inline, in the same turn as everything else.

## Pitfall: a `boot_cmd` that leaves a pipe-holder process behind

`boot_and_verify`'s `boot_cmd` branch runs `subprocess.run(boot_cmd, capture_output=True, ...)`.
If `boot_cmd` leaves a surviving process holding the captured stdout/stderr pipe -- e.g. an
un-detached background server launched from within the command -- `subprocess.run` will block
until `ready_timeout` even on an otherwise-successful boot, because the pipe never reaches EOF
while a child still holds it open. It is still bounded (`boot_and_verify` raises
`ForegroundBootError` at `ready_timeout`, it does not hang forever), so this is a usage pitfall,
not a correctness bug. For booting a long-lived server, use poll mode (`boot_cmd=None`, step 2
above) after foregrounding the boot separately, or make sure `boot_cmd` fully detaches its
children's stdio.

## On a `ForegroundBootError`: clean-slate before retry

If `boot_and_verify` raises `ForegroundBootError`, do not simply retry the boot. An interrupted
`boot_cmd` (a timeout, a killed subprocess) can leave a `nohup`'d server **orphaned** —
listening on the port, holding the run dir — which violates "one harness server at a time" and
will make the retry's own boot fail or, worse, silently attach to the stale process. Re-run the
clean-slate check (step 1) before any retry.
