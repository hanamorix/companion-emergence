# decisions.md — gate log (persisted-cadence, defer #21)

Append-only. One line per gate (stage 4, 7, 8): gate, worst severity, route, rationale.

- stage 0: no relevant metric baseline (cost/cache orthogonal to cadence timing) → conformance-only + full-suite (3730 green) regression backstop.
- stage 4 (plan red-team): worst=MINOR. Cold reviewer verified signature/line-numbers/maintenance-can't-raise/finalise-test-safe against source (citations spot-checked real). One real finding M1: the always-advance-on-exception contract (plan's own "major if violated") has no test. Route: fix-in-place → add a raising-tick canary during build, then proceed to stage 5. Nitpicks A1 (now() twice = matches soul, accept), A4 (C4 hardcoded count = update to actual) logged.
- stage 7 (code red-team): worst=MINOR. Cold reviewer verified C1/C2/C3/C5 met against source (tick bodies byte-identical, grep empty, fail-safe branches, helper tests pass). One minor: stale "monotonic timer" comment at supervisor.py:376-378. Two nitpicks: no maintenance-raises canary, no disabled-cadence no-file test. Route: fix-in-place → corrected the comment + added both hardening tests (6 cadence tests pass), then proceed to stage 8.
