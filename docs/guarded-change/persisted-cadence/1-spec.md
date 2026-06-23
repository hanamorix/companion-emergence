# 1 — Spec (persisted-cadence, defer #21)

Full design: `docs/superpowers/specs/2026-06-23-persisted-cadence-design.md`.

**Problem.** `run_folded` (`brain/bridge/supervisor.py`) paces the
`voice_reflection`, `maintenance` (forgetting+narrative), and `finalize`
cadences off `time.monotonic()`, which resets on restart and freezes during
sleep. On a desktop app not running continuously, any interval longer than a
session under-fires: forgetting never decays memories, narrative threads never
close, voice never evolves, stale buffers past 24h silence linger.

**Prior art.** Soul review (`brain/soul/cadence.py`), self-model, and kindled
already use persisted wall-clock cadences. This applies the same fix to the
three highest-bite remaining cadences.

**Constraints.**
- Tick bodies, fault-isolation, disable (`interval_s=None`) semantics, and
  finalise's 24h silence threshold unchanged.
- Fail toward running (missing/corrupt state → due-now).
- advance+save runs after the tick regardless of outcome (no tight retry storm).
- No failure-backoff (finalise owns F-011; forgetting is no-LLM).
- One generic helper, not three near-duplicate modules.

**Out of scope.** heartbeat/initiate_review/log_rotation (≤1h, fire within a
session); the existing persisted cadences (soul/self_model/kindled).
