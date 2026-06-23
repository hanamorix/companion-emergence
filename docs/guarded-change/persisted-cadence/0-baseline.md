# 0 — Baseline (persisted-cadence, defer #21)

**Verdict: no relevant metric baseline → stage-8 runs conformance-only (+ full-suite regression backstop).**

The Layer-2 config (`guarded-change.companion.md`) standing metrics are all
chat-path cost/cache/tool measures (`cost_per_chat_call_usd`,
`cache_*`, `num_turns`, tool-call counts). **#21 does not touch the chat path** —
it changes *when background supervisor ticks fire* (voice reflection, forgetting+
narrative maintenance, finalise), not how a chat turn is built or billed. None of
the standing metrics can move as a result of this change, so a cost/cache baseline
would measure noise.

**Regression backstop used instead:** the full backend test suite. Current green
baseline on `main` (pre-change, 2026-06-23): **3729 passed, 1 skipped, 1 xfailed**
(`uv run pytest -q -p no:randomly`, brain `__version__` 0.0.38). Stage 8 re-runs
this; any new failure = regression.

**Behavioural-shift guard (the real regression risk):** the three tick *bodies*
must be byte-unchanged — only the cadence *guard* (monotonic → wall-clock) and the
*footer* (advance+save) change. If a tick body changed, decay-rate / narrative /
voice / finalise behaviour could shift. Stage 8 verifies the call sites are
identical via diff (criterion C5).
