# NellFace — desktop client for companion-emergence

Tauri 2 + React 18 + TypeScript + Vite. Talks to the running bridge daemon
via HTTP (`/persona/state`, `/chat`, `/session/new`, `/sessions/close`).

## Phase 2 (this commit)

Shipped:

- Tauri 2 scaffold with the shoji visual language (warm radial bg, washi
  grain, breathing avatar glow).
- `NellAvatar` component — emotion-driven expression switch (heuristic
  mapping from `/persona/state.emotions` → expression PNG); breathing
  animation; mode-aware filter (offline → desaturate, provider_down →
  crimson glow).
- `ChatPanel` — opens a session against the bridge on mount, sends turns
  via HTTP `/chat`, renders bubbles with timestamps, typing dots while
  awaiting Nell's reply.
- `bridge.ts` client — reads bridge port + auth token via Rust command
  in production builds (`get_bridge_credentials` reads
  `<NELLBRAIN_HOME>/personas/<persona>/bridge.json`); falls back to
  `VITE_BRIDGE_URL` + `VITE_BRIDGE_TOKEN` for browser dev.
- `expressions/` symlinked into `public/` so Vite serves all 28
  expression variants (7 categories × 4 each).
- `/persona/state` polling every 5s — drives the avatar's expression.

Subsequent phases shipped:

- Inner Weather, Recent Interior, Soul highlight, Connection panels — Phase 3
- Install wizard + bridge auto-spawn + first-launch routing — Phase 4
- Animation engine + heuristic stack mapping — Phase 5A
- Soul-crystallization flash overlay — Phase 5C
- WS streaming chat (`/stream/:sid`) — Phase 6
- Paperclip image upload + emoji picker in chat input — image-support P3 + UI
- image_shas thread end-to-end through /chat + /stream into the ingest
  buffer + extract markers + memory metadata — image-support P5 + P6
- Image bytes flow into the Claude provider via `--input-format
  stream-json` so Nell actually *sees* attached images (verified live
  2026-05-07) — image-support P4
- Voice coaching teaches Nell to react from the seeing rather than
  imagining when an image arrives — image-support P7
- Emotion-family colour tints on the breathing ring + soft backing
  wash so the room temperature follows her mood — Phase 5D

- Bundled portable Python runtime so the .app is zero-install for
  Python — `pnpm tauri build` ships a self-contained NellFace.app
  that doesn't need `uv` or system Python on PATH — Phase 7
  (macOS arm64 / x86_64 / Linux x86_64 today; Windows next)

Not yet built:

- Linux .deb / Windows .msi cross-platform release pipeline
- App distribution story (signing, notarization, auto-update)

## Run it (development)

For fast iteration without rebuilding the bundled Python every cycle:

```bash
pnpm install
pnpm tauri dev
```

The Rust backend's `nell_command` helper falls back to `uv run nell`
against the source tree when the bundled runtime isn't present — so
dev mode just needs `uv` on PATH.

Bridge auto-spawn on first chat handles `nell supervisor start` for
you; if you want to start it manually:

```bash
# from the repo root
uv run nell supervisor start --persona nell
```

The app reads `NELLBRAIN_HOME` from the environment to find the
persona dir; falls back to platformdirs default (matches `brain.paths`).

## Build the production bundle

```bash
pnpm tauri build
```

This runs `app/build_python_runtime.sh` as a `beforeBuildCommand`,
which downloads `python-build-standalone` for the host arch, creates
a portable Python tree under `app/src-tauri/python-runtime/`, builds
the `companion-emergence` wheel, installs the brain into the bundled
site-packages, and strips `__pycache__` + tests. Tauri then bundles
that tree into `Resources/python-runtime/` inside `NellFace.app`.

The resulting `.app` is ~190 MB on macOS arm64 (Python + httpx +
fastapi + sqlite + ML deps; the full size is mostly third-party
deps, not brain itself). The user does not need `uv`, `python3`,
or any system Python — only `claude` (claude-cli) on PATH for the
LLM provider.

## Browser dev mode (no Tauri)

```bash
VITE_BRIDGE_URL=http://127.0.0.1:<port> \
VITE_BRIDGE_TOKEN=<token-from-bridge.json> \
pnpm dev
```

Open `http://localhost:1420`. Useful for fast iteration on UI without
rebuilding the Rust shell.

## Production build

```bash
pnpm tauri build
```

Produces a `.app` (macOS), `.deb`/`.AppImage` (Linux), or `.msi` (Windows)
in `src-tauri/target/release/bundle/`.
