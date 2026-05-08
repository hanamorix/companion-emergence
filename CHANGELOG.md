# Changelog

All notable changes to companion-emergence will be tracked here.

The project is in active OSS development as of v0.0.1-alpha. Entries below describe pre-1.0 release readiness, not a public stable API promise.

## 0.0.1 - Unreleased

### Added (2026-05-08 — second full repository audit + targeted hardening)

- **`docs/audits/2026-05-08-full-audit.md`** — second whole-repo
  audit, covering release integrity, stress coverage, lifecycle
  edge cases, supply chain, observability, security, and
  accessibility. 0×P0, 2×P1, 11×P2, 12×P3, 6×P4 = 31 findings
  across the codebase. The doc captures both the failure mode and
  a four-phase plan of action (release blockers → launchd
  production hardening → runtime/release reproducibility → app
  lifecycle/UX resilience → polish + a11y + coverage).

### Fixed (2026-05-08 — audit follow-through, phases 0-4 partial)

- **P1-1: image provider error detail surfaces.** `claude --print`
  failures on the image path were collapsing to bare `502
  provider_failed` without the structured quota/API detail the
  text path already exposed. Provider now extracts the same
  diagnostic shape across both surfaces; isolated image stress
  test rewritten + 66 new chat-provider tests cover the parity.
- **P1-2: sdist no longer ships generated runtime + local
  scratch.** `pyproject.toml` gains explicit `[tool.hatch.build]`
  exclusions for `app/src-tauri/python-runtime/`, `app/src-tauri/target/`,
  `bugs/`, `mock-ups/`, and `dist/`. Wheel smoke script tightens
  the import check so it can't false-positive against the source
  tree. Sdist size will track wheel size on the next release.
- **P2-1, P2-2: production launchd install is now resilient.**
  Tauri's `install_supervisor_service` resolves and passes the
  *bundled* `nell` path to `nell service install` instead of
  whatever happened to be on PATH at install time, so a
  Finder-launched .app installs against its own embedded runtime.
  A new helper detects DMG-translocated and unstable bundle
  paths up front and refuses with a friendly "Move the .app to
  /Applications first" message rather than wiring a plist that
  breaks on app upgrade. Cargo unit test
  (`unstable_macos_app_path_detects_dmg_and_translocation`)
  guards the regression.
- **P2-3: bundled runtime install honors `uv.lock`.**
  `build_python_runtime.sh` reproduces the locked dep set instead
  of resolving fresh on every build, so two builds from the same
  commit produce byte-identical runtime trees.
- **P2-5: non-macOS service install is honest.** The launchd
  service-install path now returns `success=false` with a "not
  yet implemented on this OS" message on Linux/Windows instead of
  the previous synthetic success that pretended the agent had
  been wired.
- **P2-7, P3-5: `/sessions/close` reports errors honestly.** The
  endpoint stops reporting `closed: true` when ingest fails;
  callers now get a 4xx/5xx with the underlying error and the
  session buffer is preserved for retry. The chat panel surfaces
  close failures inline instead of silently swallowing them.
- **P2-8: bridge-start failure is recoverable.** App.tsx no
  longer parks the user on an unrecoverable boot screen; the
  failure surface gets a "Retry" button and a "Go to settings"
  escape hatch so a transient launchd hiccup doesn't trap them.
- **P2-9: WebSocket chat stream has timeouts + cancel.**
  `streamChat` gained connection / first-byte / inactivity
  timeouts plus an explicit cancel signal wired to UI cancel.
  Hung claude-cli calls no longer leave the UI staring at a
  spinning bubble forever.
- **P2-10: credential rotation retry covers every bridge call.**
  `bridge.ts` was previously only retrying state-poll on auth
  failure; chat / session / upload / stream now go through the
  same retry-after-rotate path.
- **P2-11: StepReady doesn't gate on emotions.** Emotion
  aggregation can take up to 15 minutes (next heartbeat) for a
  fresh persona; the wizard now treats reachable `/persona/state`
  as ready and surfaces "emotions warming up" as a separate,
  non-blocking signal.
- **P3-1: bundled runtime trimmed further.** `build_python_runtime.sh`
  removes pip / idle / pydoc / Tcl-Tk / setuptools internals
  from the shipped tree on top of the earlier prune, dropping the
  bundle by another ~15-20 MB.
- **P3-3: stress harnesses are safe to share.** `stress_test_voice.py`
  rewritten with explicit budget caps, a dry-run mode, and clear
  "this hits the live provider" warnings. Image stress mirrors
  the pattern.
- **P3-9: missing `@keyframes spin`.** Several spinners across the
  wizard / boot screens referenced an undefined `spin` animation;
  added to `styles.css`.
- **P4-1: wizard accessibility (first pass).** Radio-group
  semantics on mode pickers, `aria-pressed` on toggle-style
  cards, label associations on form inputs, focus-ring rules in
  CSS for keyboard nav.
- **P4-5: detached bridge spawn closes its parent log fd.**
  `spawn_detached` was leaking the wrapper's log filehandle into
  the daemon child, so the wrapper's pipe stayed alive longer
  than intended.

### Added (2026-05-08 — Plan C: launchd supervisor agent)

- **Brain-as-a-service.** The supervisor now runs as a user-level
  macOS LaunchAgent (`com.companion-emergence.supervisor.<persona>`)
  instead of being spawned by the .app on demand. The .app is now a
  thin viewer that connects to a brain that already exists. Killing,
  uninstalling, or rebuilding the .app no longer touches the
  supervisor; launchd's `KeepAlive` restarts the brain on crash; the
  agent runs at login. Solves the entire "talk to Nell, close the
  app, brain dies" class of bug the audit cycle motivated, plus the
  related "rebuilt .app reuses an old supervisor with stale code"
  drift.
- **`brain/service/launchd.py`** — full plist build / install /
  uninstall / status / doctor surface. Idempotent reinstall via
  `launchctl bootout` then `bootstrap` then `kickstart -k`. Plist
  embeds an absolute `nell_path` (resolved at install time), the
  user's `~/Library/LaunchAgents/`-canonical layout, plus a
  `launchd-PATH` that includes `~/.local/bin` so Anthropic's
  `claude` CLI resolves out of the box.
- **`nell service` CLI** — `print-plist` / `install` / `uninstall`
  / `status` / `doctor`. `doctor` runs eight non-mutating preflight
  checks (platform, persona name, persona dir, nell path,
  LaunchAgents dir, log dir, claude CLI on PATH, NELLBRAIN_HOME)
  before any mutation.
- **`nell supervisor run`** — foreground mode designed for
  process-supervisors (launchd, systemd, etc.) to manage. Default
  `--idle-shutdown 0` (never auto-shut). The bridge runner gains
  a `cmd_run` entry point that does NOT detach; the bridge daemon
  `--client-origin launchd` choice signals the lifecycle source.
- **Wizard auto-install.** `runInstall` in the wizard's
  StepInstalling now calls `install_supervisor_service(persona)`
  after `nell init` succeeds, so first-launch users land on the
  launchd-managed model directly. Failure is surfaced inline but
  doesn't block the wizard — the legacy Tauri-spawn path is still
  the safety net.
- **Connection panel install button.** Existing personas (users
  who finished the wizard before this work landed) get a one-click
  "install launchd supervisor" button in the new "Supervisor"
  section of ConnectionPanel. Idempotent — safe to click again.
  Three explicit states: idle / running / ✓ installed / retry-on-error.
- **Plan doc.** `docs/superpowers/plans/2026-05-08-launchd-supervisor-agent.md`
  records the design, the migration story, and the surprises hit
  during live validation (claude CLI path resolution, `--env-path`
  default drift between library and CLI, supervisor module-cache
  requiring `kickstart -k` to pick up source edits).

### Added (2026-05-08 — transparent UI polish)

- **Companion Emergence rename.** `productName`, window title,
  `<title>`, App.tsx docstring all read "Companion Emergence" now.
  Internal crate / package names left as `nellface` because
  they're build artifacts, not user-facing.
- **Transparent window.** Tauri `transparent: true` +
  `macOSPrivateApi: true` + `titleBarStyle: Overlay`. Removed the
  body radial-glow + washi-grain layers. Cards (PanelShell,
  bubbles, chat input row, icon rail) keep their own cream-on-cream
  panel chrome so they read against any wallpaper.
- **Drag region.** `getCurrentWindow().startDragging()` from
  `onMouseDown` on the avatar (and a 28px top handle for the
  hidden title bar) — works around CSS `-webkit-app-region: drag`
  failing under WebKit's hit-test for transformed / blend-modes
  elements. Required `core:window:allow-start-dragging` permission
  in `app/src-tauri/capabilities/default.json`; without it, the
  call was silently rejected at the Tauri permissions layer.
- **Avatar halo retired.** Circular radial gradients (glow + tint)
  read as a perceptual ring against any colored wallpaper. Replaced
  with no halo at all in normal operation; provider-down keeps a
  dim crimson `drop-shadow` so a real LLM fault still signals
  visibly. The mood-tint colour palette is retained in source for
  future non-circular reuse.
- **Speech bubble dots.** `<TypingDots />` now renders inside the
  empty Hana bubble while streaming, replacing the standalone
  block that floated below it. Bounce animation + 0.18s stagger
  preserved.
- **Per-persona placeholder.** `Write to ${capitalize(persona)}…`
  instead of the hard-coded `write to nell...`.
- **Reflex / dream italics rendering.** Markdown `*setting line*`
  in interior summaries now renders as `<em>` instead of literal
  asterisks. Inline-only — matches the actual content shape.
- **`DREAM` deduplication.** Models writing reflex / dream / research
  output sometimes prefix it with the section name (`"DREAM: I was
  back..."`); the InteriorPanel was rendering that label twice
  (heading + body prefix). Stripped at the `_build_interior` data
  boundary.

### Fixed (2026-05-08 — bridge stability)

- **Idle-watcher false positive.** `_check_idle` treated
  `last_chat_at is None` as "idle", so a freshly launched bridge
  with no chat traffic SIGTERM'd itself ~60s after every launch.
  That cascaded into the close-heartbeat (decay + dream + reflex +
  growth) firing on every relaunch, which read in the UI as the
  brain "flooding" between sessions. Fixed by treating bridge
  startup as activity (`last_chat_at OR started_at`) so a fresh
  launch gets the full `idle_shutdown_seconds` window before the
  watcher fires.
- **Close-heartbeat debounce.** When the bridge shuts down within
  5 minutes of the previous close, skip the heartbeat tail
  (decay/dream/reflex/growth) and exit clean. Session-drain still
  runs unconditionally; the tail is what causes "flooding"
  perception during dev iteration cycles where the .app is
  rebuilt + relaunched repeatedly.
- **Bridge dev polling auth.** Tauri dev mode could not poll
  `/state` because the bearer-token check rejected requests from
  `localhost:1420` (Vite). Allowed-origins list and CORS gating
  now distinguish auth-required from origin-required surfaces
  correctly.
- **Memory lifecycle on app quit.** ChatPanel's unmount handler
  now closes any active session via `closeSession(persona, sid)`
  before the .app exits, so the buffered turn becomes a memory
  rather than waiting for the supervisor's stale-session sweep.
  `beforeunload` fallback covers Tauri webview teardown.

### Fixed (2026-05-08 — daemon-state summary cap + service drift)

- **Reflex / dream / research summaries no longer cut mid-clause.**
  `daemon_state.py` capped summaries at 250 chars with a hard
  `summary[:250]` slice, which landed in the middle of "...like
  her body still expects him to" on a 571-char journal entry. The
  cap is now 1500 (paragraph-sized) and the truncation walks back
  to the rightmost sentence terminator (`. ` / `! ` / `? ` /
  paragraph break). Fallback to ellipsis when no break exists in
  the window. Context-summary cap (the prompt-exposed slice)
  bumped from 200 → 600 with the same sentence-aware cut.
- **`claude` not found in launchd PATH.** Two-part fix.
  `DEFAULT_LAUNCHD_PATH` was a static string missing
  `~/.local/bin` (where Anthropic's installer puts the binary);
  it's now resolved at plist-build time. The CLI's
  `_add_service_common` was also hardcoding its own copy of the
  same path string as the `--env-path` default and silently
  shadowing the library's value — now lazy-imports the canonical
  default. Surfaced when `nell service doctor --persona nell`
  reported `claude not found` despite the library's plist
  builder including it.

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
  `python3`, or any system Python on PATH. Initial implementation
  shipped macOS arm64 / x86_64 / Linux x86_64; the cross-platform
  block above (2026-05-07) extended this to Linux arm64 + Windows
  x86_64. See that block for the canonical platform support matrix.
  Build flow:
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
