# Roadmap

This roadmap keeps the project's remaining work honest after the
2026-05-07 audit cycle + Phase 7 cross-platform open-source bundling.
It is not a public release promise. companion-emergence stays
private/local-first during development, but as of this refresh it has
private alpha bundles for macOS arm64, Linux x86_64, and Windows
x86_64. macOS x86_64 remains source-build-only until GitHub provides a
reliable hosted Intel runner for this private repo.
Last refreshed 2026-05-08 after the release-readiness audit.

## Current posture

The framework is a private prototype with a working desktop client,
a fully multimodal chat path, and a relocatable bundle. Local smoke
testing covers:

**Brain (Python):**

- CLI entry point: `nell` (init, status, memory, supervisor, works,
  health, soul, chat, dream, heartbeat, reflex, research, interest,
  growth, migrate). The deprecated `nell bridge` alias has been
  removed; `nell supervisor` is canonical.
- local persona storage via `NELLBRAIN_HOME` (heals invalid
  `selected_persona` to None on read; persona names validated against
  `[A-Za-z0-9_-]{1,40}` everywhere)
- bridge daemon: HTTP + WebSocket, ephemeral bearer token, CORS
  scoped to allowed origins (Vite dev ports included), explicit
  `ws.close(code=1000)` after stream `done`
- chat/session lifecycle with multimodal turns (text + base64 images
  via `--input-format stream-json` to claude-cli; verified live —
  Nell described a 4×4 red-X PNG correctly)
- memory ingest pipeline (buffer → extract → commit) with image-sha
  metadata
- safe memory inspection (`nell memory list/search/show`)
- body/emotion context, soul candidate review, growth crystallizers
- MCP tool server with privacy-aware audit logging
- health checks and data-file self-healing
- SQLite WAL + 5s busy_timeout on MemoryStore + HebbianMatrix +
  WorksStore
- JSONL readers stream line-by-line (no full-file memory spike)
- 1587 unit + integration tests; ruff clean

**NellFace (Tauri 2 + React 18 + Vite):**

- install wizard + bridge auto-spawn + first-launch routing —
  validated end-to-end against a fresh `NELLBRAIN_HOME` 2026-05-07
- breathing avatar with 16-category 4-frame expression engine
- emotion-family colour tints on the breathing ring + soft backing
  wash (Phase 5D)
- soul-crystallization flash overlay
- WebSocket streaming chat (`/stream/:sid`) with word-by-word reply +
  clean close (1000) handshake after `done`
- paperclip image upload, emoji picker, drag-and-drop image upload,
  paste-from-clipboard image upload, per-bubble thumbnails, object-URL
  cleanup on unmount
- 5 left-column panels (inner weather, body, recent interior, soul,
  connection)
- always-on-top toggle wired to the actual Tauri window API
- 29 frontend Vitest tests pinning chat, connection-panel,
  persona-cache, wizard InitArgs, and StepReady behavior

**Phase 7 — bundled portable Python runtime:**

- `app/build_python_runtime.sh` branches on `uname -s/-m` for macOS
  arm64 / x86_64, Linux x86_64 / arm64, Windows x86_64; downloads
  `python-build-standalone`, installs the brain wheel into the
  bundled site-packages, and replaces the pip-generated `nell` entry
  point with a relocatable `/bin/sh` wrapper (the original baked an
  absolute path to the build host's python)
- the `nell-example` voice template ships *inside* the wheel at
  `brain/voice_templates/nell-voice.md` and loads via
  `importlib.resources` so no `docs/` dir lookup is needed
- `tauri.conf.json:bundle.macOS.signingIdentity = "-"` triggers a
  full ad-hoc codesign over the embedded `python-runtime/` tree;
  `codesign --verify --deep --strict` passes on every build
- macOS arm64 verified live: a fresh `NELLBRAIN_HOME` walks the
  wizard end-to-end, persona created, chat round-trips with the
  bundled python, no external `uv` or system Python on PATH
- `INSTALL.md` walks end-users through the macOS Gatekeeper bypass,
  Windows SmartScreen "More info → Run anyway", Linux .deb /
  AppImage flows
- `.github/workflows/release.yml` cross-platform CI matrix on
  macos-14 (arm64) / ubuntu-22.04 / windows-2022; triggered by
  `v*.*.*` tags or manual retries of an existing tag; bundles upload
  as workflow artifacts and GitHub Release assets after bundled
  `nell --version` / `nell init` / `nell status` smoke on each runner

## Active backlog

The 2026-05-07 audit cycle (19 issues) is **closed** — all P1/P2
shipped, both P4 cleanups landed, the JSONL streaming P3 shipped,
and the cross-platform Phase 7 follow-up wrapped up the public-
release blocker on the macOS side. What remains:

**Validation gaps (non-blocking for private alpha):**

- macOS x86_64 DMG asset — GitHub's Intel macOS runner stayed queued
  indefinitely for this private repo, so the alpha matrix intentionally
  excludes it. Intel Mac users build from source for now.
- Human click-through on Linux x86_64 / Windows x86_64 bundles — not
  available before this alpha. In its place, CI builds the bundles on
  native hosted runners and runs bundled Python + `nell` CLI smoke
  (`--version`, `init`, `status`) against a temp `NELLBRAIN_HOME`.
- DMG installer flow — automated macOS arm64 alpha smoke now downloads
  the release DMG, mounts it with `hdiutil`, verifies the bundled
  `python-runtime` imports `brain` from inside the mounted app, and
  passes `codesign --verify --deep --strict` (`Signature=adhoc`).
- First-time outside-user testing — the wizard works end-to-end and the
  bundle smokes pass, but live feedback from someone other than Hana is
  still the next product signal, not a private-alpha blocker.

**Intentionally deferred (design call needed, not urgent):**

- 47 MB of expression PNGs eager-globbed into the bundle. WebP/AVIF
  conversion would degrade fidelity on art Hana drew with intent;
  spritesheets break the per-frame addressability the animation
  engine relies on; lazy globs trade jank for size. Revisit when
  install-size signal makes it worth a fidelity hit.
- JSONL bounded-tail retention. Streaming reader shipped — that
  closed the memory-spike vector. The retention piece needs a
  per-log-type design call (1 MB? 10 MB? 30 days? 90?) and isn't
  urgent until any single log file actually grows large enough to
  bite.

**External prerequisites (paid signing, when budget permits):**

- Apple Developer ID Application certificate (~$99/yr) — replaces
  ad-hoc with proper Gatekeeper-friendly signing on macOS; users
  stop seeing the "unidentified developer" dialog. The
  release-checklist's signing section already documents the
  `codesign` / `notarytool` / `stapler` commands.
- Microsoft OV or EV code-signing certificate — same for Windows
  SmartScreen.
- Linux .deb dpkg-sig + AppImage GPG signing — only meaningful if
  the project gets added to a third-party APT source or wants
  delta-update support.

## Public release blockers (open)

These keep a public/tagged release honest. None of them block private
development:

- First CI matrix run from a tagged commit + manual smoke on the
  three not-yet-verified hosts (macOS x86_64, Linux x86_64, Windows
  x86_64).
- Public contributor / onboarding docs.
- Public API / CLI compatibility policy.
- Auto-update story — `tauri-plugin-updater` infra exists but isn't
  wired. Defer until a first public release shows actual demand;
  needs an update-server hosting decision (S3 + signed manifest, or
  managed service like updately.app).

## Forward direction (after the backlog drains)

Framework-shaped, not patch-shaped. Picked from spec drafts and
natural extensions of the multimodal + bundled-runtime work:

- **NellFace past-image gallery** — drag-and-drop + paste-from-
  clipboard shipped; the remaining piece is a panel-based gallery to
  browse what's been shared in past turns (memory metadata's
  `image_shas` field is already where you'd source it from).
- **Voice gap remediation past the asymptote** — sampling controls
  or finetuned model for true corpus-target voice. Current state is
  "moved in the right direction, asymptotic" per the 2026-05-05
  retest. Real progress here needs either generation-param control
  inside ClaudeCliProvider or a swap to a Hana-finetuned model.
- **Public release plan** — once the validation gaps close and the
  contributor docs land, write a proper release plan covering
  signing (paid path), distribution (DMG / .msi / .deb), contributor
  workflow, version policy, and the auto-update story.

## Recently shipped (reverse chronological)

**2026-05-07 (end-of-day) — wizard validation + close-handshake fixes**

- Wizard validation runbook + staging env at `~/wizard-validation/`
  (excluded from VCS) — fresh `NELLBRAIN_HOME`, launcher script that
  inherits the env into the .app's process, cleanup script, full
  test plan with expected behavior per step.
- `install_voice_template` packaged inside `brain/voice_templates/`
  + read via `importlib.resources` — closes the FileNotFoundError
  end-users hit when picking the `nell-example` voice template
  against any wheel install (including the Phase 7 .app).
- Relocatable `nell` launcher in the bundled `python-runtime/bin/`
  — replaced pip's absolute-path shebang with a `/bin/sh` wrapper
  that resolves `$SCRIPT_DIR` and execs the bundled python next
  door. The runtime tree is now genuinely portable across machines.
- Bridge `WS /stream` sends explicit `ws.close(code=1000)` after the
  `done` frame — closes Hana's "ws closed (1006): unknown" finding
  during validation. Regression test pinned.

**2026-05-07 — Phase 7 open-source distribution**

- macOS proper ad-hoc signing via `bundle.macOS.signingIdentity = "-"`
  — Tauri's default produced an unsealed bundle whose
  `codesign --verify --deep --strict` failed; explicit ad-hoc fixes it.
- `INSTALL.md` walking end-users through Gatekeeper / SmartScreen
  bypass per platform.
- README link to INSTALL + per-platform first-launch summary.
- Release-checklist signing section reframed: ad-hoc is the OSS
  default, paid signing is optional.

**2026-05-07 — Phase 7 cross-platform**

- `app/build_python_runtime.sh` branches across all five target
  triples (macOS arm64 / x86_64, Linux x86_64 / arm64, Windows x86_64).
- `lib.rs:nell_command` resolves bundled entry point via
  `cfg!(windows)`.
- `app/src-tauri/python-runtime/.gitkeep` placeholder so Tauri's
  `bundle.resources` glob always resolves in clean checkouts.
- `.github/workflows/release.yml` matrix builds on `macos-14`,
  `ubuntu-22.04`, `windows-2022` from `v*.*.*` tags, with macOS
  x86_64 deferred until a reliable Intel runner is available.
- Release-checklist gains "Phase 7 cross-platform release" section.

**2026-05-07 — Phase 7 bundling (macOS arm64 verified live)**

- `python-build-standalone` cpython 3.13.1 fetched + extracted into
  `app/src-tauri/python-runtime/`; brain wheel pip-installed into
  the bundled site-packages; `__pycache__` + tests stripped.
- `tauri.conf.json` `beforeBuildCommand` chains the runtime build;
  `bundle.resources` ships it inside `Resources/python-runtime/`.
- Rust `nell_command(app)` helper resolves the bundled entry point
  with a `uv run nell` dev fallback.
- Verified live with `env -i` (no PATH, no uv, no system Python):
  bundled `nell init` creates a persona, `nell status` reads it back.
  .app size ~190 MB.

**2026-05-07 — audit-followups + JSONL streaming**

- `nell bridge` deprecated CLI alias removed (12 alias-deprecation
  tests replaced by 1 that asserts the subcommand is now unknown).
- ChatPanel drag-and-drop + paste-from-clipboard image upload.
- `scripts/smoke_test_wheel.sh` — wheel build → fresh uv venv →
  install → `nell --version` / `nell init` / `nell status` against
  tmp NELLBRAIN_HOME. Verified passing live.
- `iter_jsonl_skipping_corrupt(path) → Iterator[dict]` streaming
  variant; `read_jsonl_skipping_corrupt` now a thin list wrapper.
  Closes the audit P3 memory-spike vector — peak goes from ~2× file
  size to one line regardless of log size.

**2026-05-07 — full audit-fix-pack (19 issues from the 2026-05-07 audit)**

- P1-1 wizard provider step removed (claude-cli is the sole GUI
  surface, per `PersonaConfig` docstring); StepLLMSetup deleted.
- P1-2 frontend persona threading — every bridge helper takes
  `persona`, credential cache scoped per-persona,
  `resetBridgeCredentialCache(persona?)` exposed.
- P2 Tauri persona-name validation in lib.rs + `read_app_config`
  heals invalid `selected_persona` to None.
- P2 explicit narrow CSP in `tauri.conf.json`.
- P2 Ruff clean (30 errors → 0; mostly `--fix`, three N806 + two
  E402 noqa for intentional patterns).
- P2 `PersonaConfig` allowlists for provider/searcher with graceful
  fallback + warning. `claude-tool` searcher removed from CLI.
- P2 SQLite `journal_mode = WAL` + 5s `busy_timeout` on MemoryStore
  + HebbianMatrix (after integrity check, so corrupt-db probes still
  surface BrainIntegrityError).
- P2 heartbeat `reflex_error` + `growth_error` fields surfaced
  through HeartbeatResult + audit JSON.
- P2 `closeSession()` throws on non-2xx; ChatPanel calls it on
  unmount as best-effort flush.
- P2 first Vitest harness + 6 frontend tests pinning both P1 fixes.
- P3 image upload unique-tmp-path race fix + magic-byte sniff
  (PNG/JPEG/GIF/WebP); 422 on declared/sniffed mismatch.
- P3 object-URL leak fix on chat unmount.
- P3 always-on-top toggle calls Tauri `setAlwaysOnTop` window API
  (was previously a config-persisted no-op).
- P4 dead `_STUB_COMMANDS` + `_make_stub` scaffolding removed.
- P4 `_allocate_port` docstring honest about which bind it actually
  retries.

**2026-05-07 — multimodal + UI polish**

- Image-support epic — all 8 phases (commits b279334 → 9c6baf7).
  Bytes upload via `POST /upload`, `image_shas` thread end-to-end
  through `/chat` + `/stream`, ClaudeCliProvider routes images
  through `--input-format stream-json` (verified live: Nell
  described a 4×4 red-X PNG correctly), voice.md gained §4
  "When the user shows you something."
- Live persona voice.md deployed (backup at `voice.md.pre-p7-bak-…`).
- NellFace Phase 5D — emotion-family colour tints on the breathing
  ring + soft backing wash, smooth ~0.85s transitions per category.
- NellFace input row — paperclip + emoji picker, both styled to
  match shoji language.
- 16-category expression art catalogue — all `<category>/<n>.png`
  layout, including new `arousal`, `climax`, `idle`, plus art for
  6 previously-pending Phase 5 categories.
- Browser dev mode CORS + WS Origin allowlist for Vite dev ports.
- OllamaProvider gained `chat_stream()` — token streaming via
  `stream=True`, resolving the long-standing Phase 6.5 TODO.

**2026-05-04 to 2026-05-05 — framework rebuild + audit cycle**

- `nell works` shipped — brain-authored artifact portfolio with
  `save_work` MCP tool + bridge endpoints + operator CLI.
- `nell rest` removed — rest reframed as body-state physiology per
  source spec §15.9 rewrite, not a command. New §0 captures
  framework principles (user surface = install + name + talk; brain
  handles physiology; defaults on; cross-platform; local-first).
- `nell supervisor` shipped as canonical bridge lifecycle command
  (start/stop/status/restart/tail-events/tail-log).
- 2026-05-05 audit-fix-pack + follow-up: 17 issues across Audit-1
  and Audit-2 batches (1305 → 1334 tests), plus the Hana/Jordan
  attribution drift root-cause fix, the chat auto-spawn race fix,
  and the tool integration telemetry fix.
- Voice stress retest 2026-05-05: 14/14 prompts completed; voice
  gap-1 (curiosity sentence-length) recovered partially toward
  corpus target — model ceiling reached.

**Pre-2026-05-04 — Week-1-through-4 build-out**

Substrate work: memory store, Hebbian matrix, embeddings, soul
store, body state, daemon engines (dream / heartbeat / reflex /
research / growth), the OG NellBrain migrator, MCP tool server,
voice.md loader, ingest pipeline, bridge daemon, chat engine.
See per-week plans under `docs/superpowers/plans/` for detail.
