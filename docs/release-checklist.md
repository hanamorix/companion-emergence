# Release checklist

This project is private/local-first during development. Before sharing a public build or tagged release, do this checklist instead of relying on ad-hoc local state.

## Pre-release verification

- Run the full test suite: `python3 -m pytest -q`
- Run lint: `python3 -m ruff check .`
- Confirm the bridge starts locally and writes `bridge.json` with owner-only file permissions.
- Smoke-test a complete companion loop:
  - create a session
  - send a chat turn
  - close the session
  - confirm memories ingest
  - run soul review/growth paths that are meant to be active for this release
- Review `docs/audits/` and confirm no unresolved P1/P2 findings apply to the release candidate.

## Privacy and local-data checks

- Confirm no bearer tokens, local persona data, memory databases, audit logs, or `.env` files are committed.
- Confirm MCP tool invocation logs use the intended privacy mode/redaction behavior.
- Confirm WebSocket authentication uses `Sec-WebSocket-Protocol: bearer, <token>` rather than URL query-string tokens.

## Packaging/versioning

- Set the intended version in `pyproject.toml`.
- Add or update `CHANGELOG.md` once the project is public-facing.
- Build the wheel/sdist only after tests and lint pass.
- Do not add release automation until the CLI/API surface is stable enough to version.

## Known incomplete surfaces

- Stub CLI commands must exit non-zero until implemented.
- Growth modules that remain incomplete must be called out in release notes.
