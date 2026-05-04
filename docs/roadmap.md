# Roadmap

This roadmap keeps the project’s remaining work honest after the April 2026 audit remediation. It is not a public release promise; companion-emergence is still private/local-first during development.

## Current posture

The framework is a private prototype with enough implemented surface for local smoke testing:

- CLI entry point: `nell`
- local persona storage via `NELLBRAIN_HOME`
- bridge daemon and HTTP/WebSocket API
- chat/session lifecycle
- memory ingest and retrieval
- safe memory inspection through `nell memory list/search/show`
- body/emotion context
- soul candidate review
- growth scheduler/crystallizers
- MCP tool server with privacy-aware audit logging
- health checks and data-file self-healing
- test/lint gates across Linux, macOS, and Windows CI

## Near-term priorities

### 1. Keep private release readiness boring and repeatable

- Maintain `CHANGELOG.md` for every meaningful behavior, privacy, packaging, or release-readiness change.
- Keep `docs/release-checklist.md` as the gate before any tagged/public build.
- Add or maintain smoke-test coverage for the full local companion loop:
  - create session
  - send chat turn
  - close session
  - ingest memories
  - run active soul/growth paths
  - confirm bridge auth does not leak tokens in URLs or logs

### 2. Replace stubs with useful CLI commands one at a time

Current intentional stubs:

- `nell rest`
- `nell works`

Rules for stubs:

- Stubs must exit non-zero.
- Stubs must say they are not implemented.
- Stubs must stay listed here until implemented.
- Each implementation needs tests before the stub is removed.

Suggested order:

1. ~~`nell supervisor` — expose bridge/supervisor lifecycle in one operator-facing place.~~ *(shipped 2026-05-04 — see `docs/superpowers/specs/2026-05-04-nell-supervisor-design.md`)*
2. `nell rest` — clarify whether this is sleep/rest cadence, bridge rest, or old-plan residue before implementing.
3. `nell works` — define the user story before building; the name is currently ambiguous.

### 3. Firm up packaging before public release automation

Current packaging state:

- `pyproject.toml` defines package metadata and the `nell` script.
- Wheel packaging includes the `brain` package.
- Release automation is intentionally deferred.

Before public/tagged release:

- Build and inspect wheel/sdist artifacts.
- Install the built wheel in a clean virtual environment.
- Run CLI smoke tests from the installed package, not just the source tree.
- Decide whether non-Python assets need explicit package-data rules.
- Add public contributor docs only when the project is meant to receive outside contributions.

### 4. Keep audit findings reconciled with code reality

The April 30 audit remains useful, but several findings have since been fixed. When remediation lands:

- Update the relevant audit note or release checklist.
- Prefer verified status over stale text.
- Do not follow old audit ordering blindly without checking current code and tests.

## Public release blockers

These block a public/tagged release, but do not block private local development:

- Remaining intentional CLI stubs are not fully documented in release notes.
- No clean-install wheel/sdist smoke test has been recorded.
- Public contributor/onboarding docs are missing.
- Public API/CLI compatibility policy is not defined.
- Signing/distribution story is not applicable yet because this is not a desktop app package; if that changes, write a separate release plan.
- Remove the deprecated `nell bridge` alias. Removing it does not affect `nell chat` auto-spawn — chat uses `brain.bridge.daemon` internals directly (inside `_chat_handler`), not the CLI surface.

## Done recently

- Implemented `nell supervisor` as the canonical bridge lifecycle command (start/stop/status/restart/tail-events/tail-log), with `nell bridge` kept as a deprecating alias until v0.1.
- Resolved audit reliability issues around chat persistence, soul queue reporting, soul review idempotency, bridge API validation, memory search/listing, pytest markers, MCP audit privacy, add-memory error visibility, and vocabulary crystallization.
- Hardened cross-platform CI assumptions for Windows PID probing, POSIX-only permission assertions, and timestamp precision flakes.
- Added `nell status` as the first non-stub operational status surface.
- Added `nell memory list/search/show` for safe local inspection of a persona's memory store.
