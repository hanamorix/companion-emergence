# dragonfly config — companion-emergence (Layer 2, Hana / macOS)

Per-project config for the `dragonfly` skill. Parameterizes the agnostic diagnosis loop for
this repo on Hana's macOS dev machine. See `~/.claude/skills/dragonfly/METHODOLOGY.md` for the
contract. Sibling of `guarded-change` (which makes the fix).

```yaml
project: companion-emergence

redteam_context:          # PRIORITY ORDER — read top-down; a cold subagent can't read the whole
                          # tree, so each entry says what to check there first.
  - path: "the project root/brain"
    note: "The dev source we run from (uv run, repo .venv). Authority for any behavior claim on this machine — we execute THIS tree, not a bundled runtime. Start at chat/{prompt,engine,tool_loop}.py and bridge/provider.py. A diagnostic that cites a name/call must match THIS code."
  - path: "~/Library/Application Support/companion-emergence/personas/nell/chat_usage.jsonl"
    note: "Ground truth for cost/cache/num_turns symptoms. Confirm a field EXISTS before a hypothesis or test relies on it (e.g. there is NO background/foreground marker on generate rows). Read Nell's logs as evidence — do NOT run the hunt's tests through the nell persona."
  - path: "~/Library/Application Support/companion-emergence/personas/nell/tool_invocations.log.jsonl"
    note: "Ground truth for tool/file behavior, incl. record_monologue firing. request_id groups rows (not always stamped); no session_id, no reply-boundary field — confirm before treating request_id as 'one reply'."
  - path: "~/Library/Application Support/companion-emergence/personas/nell/monologue_digest.jsonl"
    note: "Synchronous record_monologue write-at-record-time. Presence/absence of a row for a turn is the most direct evidence of whether the tool fired."
  - path: "the project root/tests"
    note: "pytest suite (uv run pytest from repo root). Through-path tests are the cheapest non-token repro for prompt-assembly behavior — prefer a failing test over a live chat turn when the question is structural."

reproduction:             # how to exercise the suspect behavior in this project
  how: >
    PREFER a pytest through-path test (no tokens) when the question is structural — e.g. "is the
    record_monologue directive present in the assembled prompt the provider receives, and in what
    position." Only drive a real chat turn when the question is genuinely behavioral (does the model
    actually call the tool). A real chat turn against a DEV persona (Test or nell.sandbox, never the
    live `nell`) CONSUMES TOKENS via the Claude CLI → by the triage's highest-priority rule it is a
    full-guarded-change artifact, not lite.
  logs: >
    Nell's JSONL telemetry under ~/Library/Application Support/companion-emergence/personas/nell/
    is ground truth for OBSERVING past behavior; read the fresh tail rather than reasoning about what
    "should" have been logged. For a NEW repro, run against a dev persona and read THAT persona's tail.

ledgers:
  dir: "hunts/<slug>/"     # symptom-ledger.md + observation-ledger.md live here; must survive a
                           # session restart (the cold-start guard recommends restarting).

iteration_cap:
  N: 3                     # convergence-gate cap; matches the Layer-1 default. Raise only with a
                           # recorded reason if a genuinely multi-layer bug needs more cycles.
```

## Notes specific to this project

- **⛔ Never run the hunt's tests/repros through the live `nell` persona.** Use a throwaway/dev
  persona (`Test`, `nell.sandbox`, or a fresh temp `NELLBRAIN_HOME`/`KINDLED_HOME`). Reading Nell's
  *files/logs* as ground truth is fine; *driving chat turns through her* is not — it pollutes her
  continuity and burns the shared Claude subscription. (Project hard rule, mirrors the wizard-validation rig.)
- **Cite the code we actually run.** On this machine that is the repo checkout (`uv run` against the
  repo `.venv`), NOT a bundled .app runtime. Confabulated variable names / misplaced calls are the
  motivating failure class — the stage-1/4/7 cold reviewer must verify every cited identifier against
  `the project root/brain`.
- **Prefer a no-token through-path test over a live turn.** Structural prompt-assembly questions are
  answerable by a pytest that inspects `build_static_system_message` / `build_volatile_context`
  output — cheaper and deterministic. Reserve token-burning chat-turn repros for genuinely behavioural
  questions, and route them through full guarded-change.
- **Read the fresh log tail, don't infer.** An observation about cost/cache/tool/monologue behaviour is
  only trustworthy if it cites a real row written by the repro run — not a prediction of what the code
  "should" log. Confirm fields exist (several expected ones do not).
- **The two logs cannot be joined** (chat_usage has session_id + no request_id; tool log has
  request_id + no session_id). Any "tool activity per chat message" observation is uncomputable from
  the logs alone — flag it rather than fabricating a correlation.
