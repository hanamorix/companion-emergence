# Changelog

All notable changes to companion-emergence will be tracked here.

The project is in active OSS development as of v0.0.1-alpha. Entries below describe pre-1.0 release readiness, not a public stable API promise.

## 0.0.1 - Unreleased

### Added

- Local-first persona data layout under the platform-aware `NELLBRAIN_HOME` root.
- `nell` CLI entry point with migration, chat, bridge, health, soul, growth, dream, heartbeat, reflex, research, interest, status, and memory inspection surfaces.
- `nell status` for checking a persona directory, provider/searcher config, memory database count, and bridge process state without contacting live providers.
- `nell memory list/search/show` for safe local memory inspection without contacting live providers.
- Bridge HTTP/WebSocket server with local bearer-token authentication, session lifecycle endpoints, health checks, audit-safe event streaming, and dirty-shutdown recovery hooks.
- Chat/session flow with memory retrieval, body/emotion state context, persistence metadata, and local ingestion on session close.
- SQLite memory store, Hebbian associations, embedding cache, and health/self-healing support for local data files.
- Soul candidate queue, review workflow, audit logging, duplicate-safe acceptance, and revocation support.
- Growth scheduler and crystallizers for identity, preferences, relationships, style, and vocabulary.
- MCP server tools with configurable audit logging modes: `off`, `metadata`, `redacted`, and `full`.
- Release checklist for private smoke testing and future public/tagged releases.
- `nell supervisor` lifecycle command — canonical operator surface for the per-persona bridge daemon. Actions: `start`, `stop`, `status`, `restart`, `tail-events`, `tail-log`. Wraps the existing bridge daemon implementation; same args, same exit codes, plus sequential `restart` (stop-then-start, gated on stop success) and cross-platform `tail-log`.
- `nell works` — brain-authored creative artifact portfolio. Nell decides via the `save_work` MCP tool when something she's written (story, code, planning doc, idea, role-play scene, letter) is worth preserving. Stored at `persona/<name>/data/works/<id>.md` with a SQLite + FTS5 index. Operators browse via `nell works list/search/read --persona X`; the brain herself recalls via the MCP tools (`list_works`, `search_works`, `read_work`) and via the bridge `GET /self/works[*]` endpoints (per source spec §15.2). Type taxonomy: story, code, planning, idea, role_play, letter, other.

### Changed

- Stub CLI commands intentionally exit non-zero until implemented, so incomplete surfaces are visible instead of silently pretending to work.
- Provider-backed growth paths are guarded so local tests and dry runs do not accidentally hang on live provider calls.
- Memory hot paths can avoid expensive integrity checks while health checks retain deeper verification.
- Empty memory text searches are rejected; callers that want broad listing must use explicit listing APIs.
- `NELLBRAIN_HOME` now also redirects `get_log_dir()` and `get_cache_dir()` (to `$NELLBRAIN_HOME/logs/` and `$NELLBRAIN_HOME/cache/` respectively). Previously only persona data honored the override; bridge logs and cache state went to the platformdirs default. The new layout makes a sandboxed `NELLBRAIN_HOME` actually sandboxed. Users with `NELLBRAIN_HOME` set will see bridge logs move from the platformdirs location to `$NELLBRAIN_HOME/logs/` on next bridge start.

### Security and privacy

- WebSocket authentication uses bearer subprotocol headers instead of URL query-string tokens.
- Bridge state files are protected with owner-only permissions where the platform supports POSIX permission bits.
- MCP tool audit logging redacts sensitive arguments by default.
- Status output does not print bridge bearer tokens.

### Fixed

- Chat persistence failures are surfaced in response metadata instead of disappearing silently.
- Soul queue write failures increment explicit ingest error counts.
- Soul review acceptance is retry/duplicate safe.
- Windows CI no longer uses POSIX-only PID liveness probes.
- Tests no longer assume POSIX chmod behavior on Windows or distinct timestamps from immediate back-to-back calls.

### Deprecated

- `nell bridge` — use `nell supervisor` instead. The alias still works and forwards to the new command, but prints a deprecation warning to stderr. Will be removed in v0.1.

### Removed

- `nell rest` stub command. Rest is a body-state physiology concern (energy depletes from writing and long sessions, recovers from dreams and idle time), not a user-facing CLI command. The mechanics live in `brain/body/state.py`; see source spec §15.9 (rewritten) and §0 (framework principles, new).

### Known incomplete surfaces

- `nell works` remains an intentional future-work stub.
- Public release automation is deferred until the CLI/API surface is stable enough to version.
- Public contributor documentation is in progress.
