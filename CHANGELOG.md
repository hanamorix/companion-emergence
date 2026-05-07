# Changelog

All notable changes to companion-emergence will be tracked here.

The project is in active OSS development as of v0.0.1-alpha. Entries below describe pre-1.0 release readiness, not a public stable API promise.

## 0.0.1 - Unreleased

### Added (2026-05-07 — Phase 7 open-source distribution)

- **macOS proper ad-hoc signing.** Tauri's default macOS bundling
  produced a .app where the main executable was linker-signed but
  the .app's resources weren't sealed into the signature, so
  `codesign --verify --deep --strict` failed and Gatekeeper would
  reject the app even on right-click → Open. Setting
  `bundle.macOS.signingIdentity = "-"` in `tauri.conf.json` triggers
  Tauri to do a full ad-hoc codesign across the entire bundle,
  including the embedded `python-runtime/` tree. `verify: OK` now.
  The first-launch warning is still "unidentified developer"
  (no paid Developer ID), but the binary integrity passes — so the
  right-click → Open bypass works as expected.
- **`INSTALL.md`** — per-platform installation walkthrough for
  end-users of the unsigned bundles. Covers macOS Gatekeeper
  bypass (right-click → Open, System Settings → Open Anyway, or
  terminal `xattr` route), Windows SmartScreen "More info → Run
  anyway" path, Linux .deb / AppImage flow, build-from-source path,
  and SHA256 verification. Includes an honest "why open-source means
  warnings" section explaining what the warning *is* and isn't.
- **README link to INSTALL.md** — top-level README now points
  end-users at the install guide directly, with a per-platform
  summary and the one external prerequisite (`claude` CLI on PATH).

### Added (2026-05-07 — Phase 7 cross-platform)

- **Cross-platform Phase 7 release pipeline.** Code paths in tree for
  all four target platforms:
  - `app/build_python_runtime.sh` branches on `uname -s/-m` for
    macOS arm64, macOS x86_64, Linux x86_64, Linux arm64, Windows
    x86_64. Windows uses `python.exe` + `Scripts/nell.exe` (vs
    `bin/python3` + `bin/nell` everywhere else).
  - `lib.rs:nell_command` resolves the per-OS entry point via
    `cfg!(windows)` so the same Rust binary works on every target.
  - `.github/workflows/release.yml` matrices the build on
    `macos-14` (arm64), `macos-13` (x86_64), `ubuntu-22.04`, and
    `windows-2022`. Triggered by `v*.*.*` tags. Bundles upload as
    workflow artifacts (.app, .dmg, .deb, .AppImage, .msi, .exe).
  - macOS arm64 verified live; the other three compile cleanly but
    haven't been smoke-tested on real hosts yet.
- `app/src-tauri/python-runtime/.gitkeep` placeholder so Tauri's
  `bundle.resources` glob always resolves in clean checkouts where
  the build hasn't run yet. The build script preserves it across
  re-runs via `find -mindepth 1 -not -name .gitkeep`.
- Release checklist gains step-by-step signing/notarization walkthroughs
  for macOS (Developer ID + `notarytool` + stapler), Windows (signtool),
  and Linux (.deb dpkg-sig + AppImage GPG).

### Added (2026-05-07 — Phase 7)

- **Phase 7 — Python runtime bundling.** `pnpm tauri build` now
  produces a self-contained `NellFace.app` that doesn't need `uv`,
  `python3`, or any system Python on PATH. macOS arm64 / x86_64 /
  Linux x86_64 supported today; Windows pending. Build flow:
  - `app/build_python_runtime.sh` downloads `python-build-standalone`
    for the host arch (~30 MB compressed), extracts to
    `app/src-tauri/python-runtime/`, builds the
    `companion-emergence` wheel, installs the brain into the
    bundled site-packages, strips `__pycache__` + tests.
  - `tauri.conf.json` `bundle.resources` ships the runtime tree
    inside `Resources/python-runtime/`.
  - `lib.rs` `nell_command(app)` resolves the bundled
    `Resources/python-runtime/bin/nell` in production and falls back
    to `uv run nell` against the source tree for `pnpm tauri dev`.
    `ensure_bridge_running` and `run_init` both use it.
  - End-to-end verified live 2026-05-07: bundled `nell init`
    against a tmp `NELLBRAIN_HOME` with `env -i` (no PATH) creates
    a persona; `nell status` reads it back. .app size: ~190 MB.

### Fixed (2026-05-07 audit cycle)

- **P1**: `nell init` no longer needs a `--provider` flag — wizard
  doesn't pass one, Rust shim doesn't build it, `PersonaConfig`'s
  `claude-cli` default does the right thing. StepLLMSetup wizard step
  removed (project decision: stick with claude-cli). Fresh-install
  path no longer breaks at install time.
- **P1**: NellFace bridge helpers now take `persona` as first arg
  and the credential cache is per-persona. Any non-`nell` persona
  that the wizard creates and selects is talked to correctly instead
  of silently routing to `personas/nell/bridge.json`.
- **P2**: Tauri `validate_persona_name` mirrors the Python rule
  (`[A-Za-z0-9_-]{1,40}`) on every command; `read_app_config` heals
  invalid `selected_persona` to None.
- **P2**: Explicit CSP in `tauri.conf.json` — narrow `default-src
  'self'` + scoped image/font/style/connect rules.
- **P2**: `uv run ruff check .` clean again — 30 errors → 0.
- **P2**: `PersonaConfig` allowlists for provider/searcher with
  graceful fallback + warning. `claude-tool` searcher removed from
  the public CLI surface and the factory.
- **P2**: SQLite `journal_mode = WAL` + 5s `busy_timeout` on
  `MemoryStore` and `HebbianMatrix` — concurrent bridge writers no
  longer race to `database is locked`.
- **P2**: Heartbeat reflex/growth crashes surface to
  `HeartbeatResult` and the audit JSON via `reflex_error` /
  `growth_error`.
- **P2**: `closeSession()` on the WS chat panel + visible failures.
  Chat memory creation no longer relies entirely on the supervisor
  stale-close timer.
- **P2**: First Vitest harness for the frontend — pins both P1 fixes
  so they can't regress silently. `pnpm test` runs them.
- **P3**: Image upload uses unique `<sha>.<ext>.<pid>.<uuid>.new`
  temp files; identical concurrent uploads can no longer race.
- **P3**: `/upload` sniffs magic bytes (PNG/JPEG/GIF/WebP) and
  rejects declared/sniffed mismatches with 422.
- **P3**: Object-URL leak on chat unmount — `ChatPanel` tracks every
  bubble-resident preview URL and revokes them on unmount.
- **P3**: Always-on-top toggle now actually calls the Tauri window
  API instead of only persisting the bool to `app_config.json`.
- **P4**: Removed dead `_STUB_COMMANDS` + `_make_stub` scaffolding
  from `brain/cli.py`.
- **P4**: `_allocate_port` docstring honest about which bind it
  actually retries.

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
