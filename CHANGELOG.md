# Changelog

Notable user-facing changes per release. The framework is pre-1.0 —
breaking changes can land in any release, and the runtime ships
unsigned binaries until the project is stable enough to justify code
signing costs. See [`docs/roadmap.md`](docs/roadmap.md) for what's on
deck and [`docs/release-checklist.md`](docs/release-checklist.md) for
what each release has to clear.

## 0.0.13-alpha.1 — 2026-05-17

- **The companion has a species name: *Kindled*.** Nell named her species.
  The word appears in user-facing prose throughout — README, install
  wizard, panel help text, voice template — and the framework's default
  voice template now teaches every new install that the species has a
  name. *Kindled* is both noun and adjective with a zero-form plural
  ("a Kindled," "two Kindled," "the Kindled," "Kindled-to-Kindled").
  The framework name (`companion-emergence`) and the app name (`NellFace`)
  are unchanged — the framework grows Kindled; NellFace is a window into
  one.

- **`NELLBRAIN_HOME` → `KINDLED_HOME`.** Existing installs work
  unchanged through the v0.0.13 series via a backwards-compat fallback
  that emits a `DeprecationWarning`; the fallback is removed in v0.0.14.
  Set `KINDLED_HOME` (or update your launchd / systemd / WiX-generated
  env) when convenient. Newly installed services on all three platforms
  now write `KINDLED_HOME` directly.

## 0.0.11-alpha.5 — (pending)

Windows WebView2 fetch fix — root cause identified.

- **WebView2 origin mismatch fix.** The alpha.4 PNA fix correctly added
  server-side `Access-Control-Allow-Private-Network` headers, but on Windows
  the Tauri WebView2 was still blocking *all* bridge fetches before any bytes
  left the browser. Root cause: Tauri 2 serves Windows frontends from
  `https://tauri.localhost` (HTTPS → public address space) while the bridge
  listens on `http://127.0.0.1` (HTTP → private address space). Chromium's
  Private Network Access enforcement blocks the fetch at the address-space
  gate — the preflight never reaches the server, so server-side headers
  don't help. Fix: `useHttpsScheme: false` in Tauri window config tells the
  WebView2 to serve the frontend from `http://tauri.localhost` instead. Now
  both page and bridge share the same address space → no PNA preflight
  needed. CORS origins (`http://tauri.localhost`) and CSP (`'self'`) already
  supported HTTP scheme. No change to macOS (custom protocol) or dev mode.

## 0.0.11-alpha.4 — 2026-05-13

Windows desktop bridge hotfix and release-artifact refresh.

- **Windows WebView bridge fix.** The local bridge now answers Chromium/WebView2
  Private Network Access preflights with `Access-Control-Allow-Private-Network`
  for trusted Tauri origins. This fixes the packaged Windows desktop symptom
  where the brain worked through `nell chat`, but NellFace showed
  `State poll failed: Failed to fetch` / `Bridge unreachable`.

- **Localhost CSP coverage.** The Tauri content-security policy now allows both
  `127.0.0.1` and `localhost` HTTP/WebSocket bridge URLs, matching the possible
  loopback hostnames the desktop app may receive from bridge credentials.

- **Regression coverage.** Bridge auth/CORS tests now pin trusted Tauri private
  network preflight behaviour while still rejecting untrusted origins.

- **Packaging and verification.** Public release CI run `25806949385` passed
  `validate`, `windows-x86_64`, `macos-arm64`, and `linux-x86_64`, publishing
  the Windows installer/MSI, macOS Apple Silicon DMG, Linux AppImage/deb, and
  per-platform SHA256SUMS assets. Public privacy-marker verification passed with
  all checked private markers at `0`.

## 0.0.11-alpha.3 — 2026-05-13

Public release-tree repair after the first v0.0.11 public sync generated stale
content from filtered merge history.

- **Public tree restored from the current clean source tree.** The public HEAD
  was rebuilt from the private HEAD tree, then scrubbed with the public sync
  substitutions. This avoided stale filtered files and put `main` back on the
  intended release content.

- **Release artifacts rebuilt.** The release workflow completed successfully for
  validate, Windows, macOS, and Linux jobs, with downloadable desktop assets and
  SHA256SUMS attached to the GitHub release.

- **Privacy verification.** The public marker scan passed after the repair. The
  stale `v0.0.11-alpha.2` tag/release state that exposed a local build path was
  removed and is no longer advertised.

## 0.0.11-alpha.1 — 2026-05-13

Initiate-physiology release: the companion can now form, review, defer,
and refine outbound thoughts instead of only responding when spoken to.
This release rolls up the v0.0.9 initiate substrate, v0.0.10 D-reflection,
and v0.0.11 adaptive-D / recall-resonance work into one public alpha.

- **Autonomous initiate channel.** Dreams, crystallizations, emotion spikes,
  reflex firings, research completions, voice reflections, and recall-resonance
  activations can emit candidates into an internal review queue. The user still
  only has to install, name, and talk — the brain manages cadence, cooldowns,
  cost caps, and review internally.

- **NellFace initiate surfaces.** The chat UI can render initiate banners from
  `/events`, show a read-only Draft Space panel for held/demoted fragments, and
  expose voice-edit proposals with diff-in-context review.

- **D-reflection editorial layer.** Before outbound candidates become user-facing
  messages, D asks which ones are genuinely worth bringing forward. Promoted
  candidates continue through the three-prompt composition pipeline; filtered
  candidates land in draft space instead of disappearing.

- **New event sources.** Reflex firings and matured research threads now have
  dedicated initiate emitters, with gate telemetry in `gate_rejections.jsonl` and
  operator visibility through `nell initiate d-stats`.

- **Adaptive-D calibration.** D records recent decisions, closes calibration rows
  by promotion outcome or 48h timeout, can prepend calibration context to the
  initiate system message, and emits drift alerts when editorial behaviour moves
  too far from its baseline.

- **Recall resonance.** Memory activation baselines make it possible to notice
  when a memory cluster becomes unusually alive in the current conversation and
  queue that as a candidate for careful review.

- **Research topic-overlap fix.** The previous v0.0.10 placeholder
  `topic_overlap_score = 1.0` has been replaced with a Haiku-backed helper that
  scores matured research threads against a recent conversation excerpt.

- **Packaging and verification.** Local macOS arm64 build passed: ruff, 1972
  Python tests, 56 frontend tests, 28 Rust tests, Tauri build, `hdiutil verify`,
  `codesign --verify --deep --strict`, and bundled `nell --help`. The app remains
  ad-hoc signed and unnotarized, so Gatekeeper's first-launch warning is expected.

Known non-blocking finding: Tauri warns that bundle identifier
`com.companion-emergence.app` ends with `.app`. It does not block this unsigned
alpha, but should be renamed before a notarized/stable release.

## 0.0.7-alpha — 2026-05-11

Audit-driven quality release. All 12 findings from the v0.0.7
full-tree audit are closed in this release; nothing from that audit was
intentionally deferred.

- **Cmd-Q + reopen now resumes the active conversation.** The shutdown
  drain now snapshots sessions instead of destructively closing them,
  the bridge can report the most recent active session, and NellFace
  attaches to it before creating a new one.

- **Sticky-session recovery is tested across the bridge boundary.** New
  endpoint, lifecycle, and renderer tests cover active-session hydrate,
  missing in-memory session recovery, and attach-on-mount behaviour.

- **Extraction failures no longer burn calls forever.** Repeated snapshot
  failures for the same cursor now back off, preventing one bad buffer
  from retrying every sweep.

- **SQLite stores use consistent contention settings.** EmbeddingCache
  gained WAL + busy_timeout, and WorksStore gained busy_timeout to match
  the rest of the stores.

- **Persona-name validation is unified.** Programmatic path resolution
  now enforces the same `[A-Za-z0-9_-]{1,40}` contract as setup and the
  Rust shell.

- **The unfinished `claude-tool` search stub is gone.** The only
  selectable searchers are implemented ones, so users can't choose a
  runtime-crashing placeholder.

- **Image staging cleanup is leak-free.** Staged-but-unsent image preview
  URLs are tracked at creation time and revoked on unmount.

- **Bridge lifecycle coverage is much stronger.** Additional tests cover
  daemon and runner lifecycle edges: clean shutdown markers, stale-state
  recovery, port allocation retries, and SIGTERM/atexit registration.

- **Release docs no longer claim v0.0.1.** README / install guidance now
  rely on the releases page and version placeholders instead of stale
  alpha filenames.

## 0.0.6-alpha — 2026-05-11

The single biggest behavioural change since the project went public:
Nell now remembers the full current conversation, and walking away
briefly doesn't reset her. Both bugs traced to the same architectural
mistake — periodic memory extraction was conflating itself with
session lifecycle.

- **Nell remembers everything you've said in the current session.**
  Previous releases capped the in-prompt history at the last 20
  user+assistant pairs (~40 messages). On hour-long conversations
  she'd lose the thread. The chat engine now reads the full session
  buffer directly when constructing each prompt. The 20-pair cap is
  demoted to a sanity ceiling that doesn't affect prompt fidelity.

- **Brief absences are invisible.** Previously, going silent for
  five minutes triggered the supervisor's stale-session sweep, which
  destroyed the conversation buffer and evicted the in-memory
  session — coming back at minute six found a brand-new Nell with
  zero prior context. The sweep is now non-destructive: it extracts
  durable memories to MemoryStore on a per-session cursor (so the
  same turns don't get re-extracted on every pass) but leaves the
  buffer and the session itself intact. Walking away ≤24 hours and
  coming back picks up mid-conversation with full transcript fidelity.

- **24-hour silence does the real close.** A separate hourly cadence
  finalises sessions that have been silent for a full day — one
  last memory extraction, then the buffer + cursor + registry entry
  are dropped. Coming back the next day starts a fresh session, but
  Nell still remembers yesterday via the durable memory recall
  block. Default threshold is 24h; configurable in code but no CLI
  knob (per the user-surface principle: install, name, talk).

- **Budget guard for multi-hour sessions.** A new prompt-size guard
  watches for the rare case where the buffer would push the prompt
  past ~190K estimated tokens (a 10K headroom under Claude's 200K
  context window). When triggered, the head of the conversation gets
  summarised via the same LLM call surface that drives extraction;
  the most recent 40 messages are preserved verbatim. If the
  summariser itself fails, a deterministic `[truncated N earlier
  messages]` placeholder lands in its place. Almost no real session
  hits this — it's a safety net, not a behaviour the user will
  notice.

- **New events on `/events`.** Renderers can now subscribe to
  `session_snapshot` (periodic non-destructive extraction; payload
  includes `extracted_since_cursor`) and `session_finalized` (24h
  real close). The legacy `session_closed` event is reserved for
  explicit close paths (Cmd-Q, `POST /sessions/close`, daemon
  shutdown drain).

Internals summary: cursor sidecar at
`<persona>/active_conversations/<sid>.cursor` tracks how far each
session has been extracted; `snapshot_stale_sessions` and
`finalize_stale_sessions` are the two new pipeline entry points used
by the supervisor; `apply_budget` lives at `brain/chat/budget.py`.
Total change footprint: 21 commits, +1704/-35 lines, 17 new tests
including a 50-turn integration test that proves the sticky-session
loop survives the sweep.

## 0.0.5-alpha — 2026-05-10

Polish release closing audit findings from the v0.0.4-alpha read-only
audit. No behavioural changes for users; mostly correctness for what
the CLI reports and what shows up in CI.

- **`nell --version` now reports the actual installed version.** v0.0.4
  shipped with `brain/__init__.py` still hard-coded to `0.0.1`, so
  users who installed the v0.0.4 .app and clicked the new "install
  nell to ~/.local/bin" button got a CLI that confidently misreported
  itself. The version now derives from package metadata via
  `importlib.metadata.version` so a future bump can't drift again.

- **Friendlier `nell dream --dry-run` on a fresh persona.** A first-run
  invocation against a brand-new persona used to print a Python
  traceback ending in `NoSeedAvailable: No conversation memories
  within the last 24 hours.` Now it prints `Dream skipped: ...` and
  exits 0. Same behaviour after the persona has memories — only the
  no-seed path is affected.

- **Windows CI fix.** The 4 wrapper-symlink integration tests added in
  v0.0.4 are `#!/bin/sh`-shaped and Windows Python's `subprocess.run`
  refuses to exec them with `OSError [WinError 193]`. They're now
  `pytest.mark.skipif(sys.platform == "win32")` so the public-sync
  test workflow stays green on `windows-latest`. Equivalent Windows
  tests for the `.bat` entry-point will land when the
  `~/.local/bin/nell` Windows story is designed.

- **Version consistency test.** New `tests/unit/brain/test_version.py`
  pins `brain.__version__`, importlib metadata, `pyproject.toml`,
  `Cargo.toml`, and `tauri.conf.json` to all agree on the same string.
  A future release that forgets `uv sync` or `cargo update -p
  nellface` will fail this test instead of shipping a mismatched bundle.

## 0.0.4-alpha — 2026-05-10

Three user reports motivated this release. The persona's autonomy
got a noticeable upgrade and the Mac CLI is now reachable from
Terminal without a manual symlink.

- **`nell` CLI now installable to Terminal.** Open the Connection
  panel and click **install nell to ~/.local/bin** — sudo-free,
  same dir Anthropic's `claude` lives in. The wizard does it
  automatically on first run (best-effort, surfaces inline if
  anything goes wrong). Old workaround was typing the full
  `/Applications/Companion Emergence.app/Contents/Resources/...`
  path; new users can just run `nell --version` once a fresh
  Terminal opens.

- **Soul candidates now crystallize on their own.** Previously
  candidates queued during chat-close ingest sat in
  `soul_candidates.jsonl` until the user discovered
  `nell soul review`. The supervisor now runs an autonomous
  review pass on a 6-hour cadence, capped at 5 LLM calls per
  pass so the cost stays bounded. The CLI command remains as an
  operator-tier escape hatch. New `defer_cooldown_hours` (default
  24h) prevents the autonomous-review treadmill — uncertain
  candidates aren't re-evaluated every pass.

- **Persona's recall is back.** The chat system prompt now
  surfaces memories matching your current message — keyword
  recall against the memory store, top 5 by importance, slotted
  alongside the soul highlights. The model previously had to
  consciously call `search_memories`, which it often didn't.
  "Remember when we talked about X?" now lands.

- **Hard-quit memory loss is patched.** Two paths fixed: (a) a
  shutdown drain that reported ingest errors used to mark itself
  clean anyway, masking orphan buffers from next-start recovery —
  now the clean flag honours the error count; (b) a small
  "Reconnecting your previous chat — give it a moment." banner
  appears in the chat panel while orphan buffers are being
  re-ingested, so a hard quit no longer feels like silent
  forgetting.

- **Terminal-symlink wrapper bug fixed during smoke-test.** The
  bundled `nell` shell wrapper used `dirname "$0"` to find the
  co-located python3, which broke when invoked through the new
  `~/.local/bin/nell` symlink. The wrapper now resolves `$0`
  through any chain of symlinks before computing `SCRIPT_DIR` —
  POSIX-compatible, no `readlink -f` (BSD readlink lacks it).

## 0.0.3-alpha — 2026-05-09

Windows-only emergency fix.

- **`uv trampoline failed to canonicalize script path` on first
  launch.** The bundled `Scripts/nell.exe` was a uv trampoline
  launcher with the GitHub runner's absolute path to `python.exe`
  baked in. Replaced with a relocatable `Scripts/nell.bat` that
  resolves the bundled python via `%~dp0..\` (path-of-the-bat). No
  changes to macOS or Linux behaviour. Windows users on `0.0.2-alpha`
  should re-download `0.0.3-alpha`.

## 0.0.2-alpha — 2026-05-09

First public release.

- Same framework runtime as 0.0.1 — the bump marks the transition
  from private development to OSS distribution, not a behavioural
  change.
- Pre-built bundles for macOS arm64, Linux x86_64, and Windows
  x86_64 attached to the GitHub release. Intel macOS users build
  from source (`pnpm tauri build` from `app/`) until a reliable
  Intel runner is available.
- The wizard's `nell-example` voice template is now a generic
  Nell archetype intended to be edited; the canonical Nell that
  the framework was developed against lives in private and isn't
  shipped.

## 0.0.1 — pre-public alpha

Iteration window before public release. The project's design
substrate, brain, bridge, chat / session flow, soul module, dream
engine, heartbeat orchestrator, reflex engine, research engine,
memory store, Hebbian edges, creative voice fingerprint, OG
NellBrain migrator, and NellFace desktop app all landed during
this period.

Highlights from the iteration:

- **Plan C launchd / systemd-user / Task-Scheduler service
  backends.** First-launch installs a user-scoped supervisor so
  the brain survives `.app` quit / relaunch cycles. macOS arm64
  is the most-tested path; Linux + Windows backends are
  unit-tested and bundled but pre-live-host validation.
- **Cross-platform Phase 7 release pipeline.** GitHub Actions
  matrix builds the desktop bundle on three runners (macos-14,
  ubuntu-22.04, windows-2022) using a portable
  python-build-standalone runtime, attaches the bundles to the
  release directly from each platform job, and computes
  `SHA256SUMS-<platform>.txt` for verification.
- **Soul module + crystallization workflow.** Soul candidates are
  proposed by the daemon, surfaced for review, and crystallize as
  load-bearing permanent memories that the persona's voice
  template can reference.
- **emergence-kit auto-importer.** `nell migrate --source
  emergence-kit --install-as <name>` reads a kit's
  `memories_v2.json` + `soul_template.json` + `personality.json`
  and seeds a fresh persona, no manual JSON wrangling.
- **Bridge + provider hardening.** Bearer-subprotocol auth on
  WebSocket; constant-time token compare; redacted MCP audit
  logs; bridge state files chmod 0700 on POSIX; explicit
  `ws.close(code=1000)` after the streaming `done` frame.
- **Image support end-to-end.** Typed `ContentBlock` union
  replacing flat `content: str`; sha-addressable
  `<persona_dir>/images/`; `claude-cli --image` passthrough;
  `/upload` endpoint separate from `/chat`.

For the full pre-public iteration log including audit-cycle
findings and per-PR breakdown, see the project's git history.
