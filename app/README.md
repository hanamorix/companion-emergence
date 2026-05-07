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

Not yet built:

- Emotion-family colour tints over the avatar — Phase 5D
- Image bytes flowing all the way through to the provider so Nell
  visually *sees* them (today she gets `[image: <sha[:8]>]` markers
  in the prompt, not the pixels). The Claude CLI subprocess surface
  has no native image-attach flag; the next session swaps the
  multimodal path to either Anthropic SDK or `--input-format
  stream-json` with base64 blocks — image-support P4 (deferred from
  the P5+P6 bundle because the CLI flag the original plan assumed
  doesn't exist)
- Voice coaching for seeing-images — image-support P7 (gated on P4)
- Python+uv runtime bundling for true zero-install — Phase 7

## Run it

The bridge daemon must be running. Start it first:

```bash
# from the repo root
uv run nell supervisor start --persona nell
```

Then in this directory:

```bash
pnpm install
pnpm tauri dev
```

The app reads `NELLBRAIN_HOME` from the environment to find the
persona dir; falls back to platformdirs default (matches `brain.paths`).

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
