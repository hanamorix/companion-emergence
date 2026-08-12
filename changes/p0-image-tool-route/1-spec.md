# P0 image-tool-route — Spec

**Branch:** `ThinkerOfThoughts/p0-image-tool-route` (off `ThinkerOfThoughts/diagnose-monologue-bleed-memory-gap`; no merge of main).
**Owner direction:** documented, and a feasibility spike PASSED — see
`~/.claude/plans/memory-dream-rework-p0-image-tool-spike.md` (`## RESULT`) and the P0 entry in
`~/.claude/plans/memory-dream-rework-plan.md`.

## Problem — three defects, one shared root

The companion has a **special image transport** (`ClaudeCliProvider._chat_with_images`, an
`--input-format stream-json` base64 path) that exists only because the built-in `Read` tool is
disallowed (`provider.py:247` `_BUILTIN_TOOLS_DISALLOWED`, the MCP-only posture). That transport is a
second, parallel assembly path alongside the normal text/tool path, and its existence produces three
distinct problems:

1. **#43 — can't attach a non-image file.** The frontend button (`ChatPanel.tsx:879`, `aria-label="attach
   image"`, `ACCEPTED_IMAGE_TYPES` restricts to png/jpeg/webp/gif) only accepts images. A user cannot hand
   the companion a text/PDF/other file at all.
2. **#72 — image/text fork divergence.** Two assembly paths that must be kept in lockstep and drift.
3. **Volatile-drop-after-image (correctness bug, undocumented issue draft on hold).** After any image in a
   session, **every later text turn loses its `volatile_suffix`** (all ambient/volatile context). Mechanism:
   the engine picks its builder from **this turn's** `image_shas` (`engine.py:187`), while the provider
   picks its transport from `has_images` over **replayed history** (`provider.py:623`,
   `_message_has_image`). On a text turn *following* an image turn, the engine takes the text branch
   (builds a `volatile_suffix`, `engine.py:198-208`) but the provider sees the replayed image block in
   history, routes to `_chat_with_images`, and **`_chat_with_images` never carries `volatile_suffix`** — so
   the volatile context is silently dropped. The two components disagree on what "an image turn" is.

**Owner's chosen direction (already decided — this spec elaborates it, it does not re-decide it):**
*dissolve the image transport.* Route the shared image through the curated `read_file` MCP tool,
generalized to return a **viewable image** (an MCP image content block). Delete `_chat_with_images` + the
provider image branch. With one assembly path, #72's fork disappears and the volatile-drop bug cannot
occur (there is no longer a transport that bypasses `volatile_suffix`). The owner's direction also framed
this as "turn the attach-image button into a send-file button (closes #43)"; **the runner's elaboration
found that fully closing #43 needs more than the transport change — see the scope decision below.**

**What THIS change delivers — SCOPE = A per owner ruling (2026-08-12, see decisions.md):**
- ✅ **Images become viewable via `read_file`** (spike-proven), replacing the transport.
- ✅ **The volatile-drop bug is fixed** and the #72 image/text fork is dissolved (single assembly path).
- ✅ **#43 storage + hand-off is closed in THIS PR** (text fully readable; PDF/binary stored + path-surfaced).
  Widen the backend so a non-image file reaches disk and can be handed to the read tool: `/upload`
  (`server.py:2150-2201`) today hard-refuses every non-image at three gates (415 on declared type, 422 on
  `sniff_media_type`, and it stores only via `save_image_bytes`, images only, `images.py:106-122`). Scope A
  **widens `/upload` + adds a non-image content-addressable store** (`brain/files.py`) and adds the
  **end-to-end #43 criterion C16** (send a non-image text file → stored → readable via `read_file`).
  - **Readability scope (honest, reconciled with C16 — final-red-team Finding 1):** a **text** file is
    fully readable (`read_file` returns its content). A **PDF/other non-UTF-8 binary** is stored and its
    path surfaced to the model, but `read_file` returns the existing "binary — not shown" note — PDF **text
    extraction** is a further step (a new dependency) and is **NOT** built here. The owner's ruling named
    "text/PDF … readable"; because PDF-readability needs that extra step, **it is escalated to the owner at
    the post-build gate**, not silently narrowed. This spec makes no claim that PDFs are readable in this PR.
  - `server.py` is a ⚠ Hana-coordination surface — flagged in the footprint below.

**Image-seeing shifts from *automatic* to *opt-in-via-tool-call* (owner-awareness, red-team F-3):** today
an attached image is always in-context (that was #48's fix); after the change the model must *choose* to
call `read_file` on the surfaced path. The change carries a **gating** criterion (C13) that proves the
*organic* image-share turn — the actual #48 scenario — still works before ship; if it does not, the design
proactively emits the read (open axis 4). Not a silent substitution — an owner-awareness item at sign-off.

## The one capability this rests on — already verified

An MCP tool that returns an image content block **does** render to the model as a viewable image under
the companion's real posture (`claude -p --mcp-config --disallowedTools <11 builtins incl. Read>`). Proven
by the P0 spike: secret token `QV8KT` rendered as an image, tool returned an image block, model read back
`QV8KT` exactly; negative control (text-only result) → `NO IMAGE`. **Caveat carried forward:** the spike
was ONE positive run (usage emergency), on the `sonnet` alias. This spec's criteria therefore **re-confirm
the mechanism ≥3× on fresh secrets against the companion's pinned chat model**, through the *real*
`brain.mcp_server` (the spike used a throwaway FastMCP server; the real server uses the low-level
`mcp.server.Server` API and returns `list[TextContent]` — the image path must be added *there*).

## Design (RUNNER'S PROPOSAL — challengeable; the owner-sign-off gate ratifies it)

The owner ratified the *direction*. The elaborations below (E-marked) are the runner's proposal and are
exactly what the post-red-team owner-sign-off gate is for. Each names an open axis.

- **E1 — generalize `read_file` to return images.** In `brain/tools/impls/read_file.py`, when the resolved
  file is an image of an allowed type within an image size cap, return a result that signals "this is an
  image" carrying the raw bytes + mime type (instead of today's `{"note": "(binary file … not shown)"}` at
  line 145). The **MCP adapter** `brain/mcp_server/tools.py:_call_tool` (currently always emits
  `[TextContent(...)]`) detects that signal and emits an `mcp.types.ImageContent` block (base64 + mimeType).
  Text files keep returning `TextContent` unchanged.
- **E2 — readable-types + image-size tunables (`files.*`).** Add `files.image_types` (default
  png/jpeg/webp/gif — mirrors `ACCEPTED_IMAGE_TYPES`) and `files.image_max_bytes` (a cap sized for real
  images, separate from the 256KB text `files.read_max_bytes`). **Open axis:** exact default cap value and
  whether non-image binaries stay refused (proposal: yes, refused as today).
- **E3a — image send-path change (IN SCOPE, images).** `ChatPanel.tsx`: change the send path so an
  uploaded **image**'s **path is surfaced to the model as text on the normal turn** rather than as an
  `image_shas` payload that triggers the transport. The button stays image-attach; accepted types stay the
  image set (images already upload+store today via `/upload`). This is the part that dissolves the transport
  and is independent of the #43 decision. **Open axis:** exact UX + how the path is surfaced (E4).
- **E3b — "send file" relabel + non-image support (CONTINGENT on option A of the #43 scope decision).**
  Only if the owner puts #43 in this PR: relabel `attach image` → `send file`, widen `ACCEPTED_IMAGE_TYPES`,
  AND widen the backend `/upload` + `_ALLOWED_UPLOAD_MEDIA_TYPES` + add a non-image storage path (⚠
  `server.py`). Under the runner's default (option B) E3b is deferred to the follow-up PR and the frontend
  stays image-only.
- **E4 — surface the path through the NORMAL text path (this is what fixes volatile-drop).** The uploaded
  image lands on disk (existing `/upload` → `<persona_dir>/images/<sha>.<ext>` content-addressable store;
  extended to non-images only under option A of the #43 decision). The engine surfaces its **path** in the
  user turn's text (e.g. a rendered line "[the user shared a file: <path>]") and **stops setting
  `image_shas`/`ImageBlock`**, so the turn is an ordinary text turn that carries `volatile_suffix` and the
  model calls `read_file <path>`. **No branch on images anywhere in engine/provider.** The path line is
  injected only into the outgoing user message, NOT into the salience/volatile input (see plan step 5 /
  C14). Open axis: exact surfaced-line wording/mechanism.
- **E5 — force-recruit `read_file` on a file-send turn.** `read_file` is **not** in `REFLEXIVE_CORE`
  (`tool_recruit.py:21`); it is a salience-recruited `_FILE_TOOLS` member (`tool_recruit.py:41`). A
  file-send turn with little text could fail to recruit it, leaving the model unable to read the file it was
  just handed. The change must **guarantee `read_file` is in the allowed set whenever a file was shared this
  turn** (force-include, independent of salience score).
- **E6 — delete the transport.** Remove `_chat_with_images` (`provider.py:1088`), the `has_images` branch
  (`provider.py:619-637`), `_message_has_image`, `_build_stream_json_user_message`, and the engine's
  `image_shas` builder-fork (`engine.py:187-208`, `_build_user_message` image blocks, buffer-replay
  `ImageBlock` at `engine.py:394-405`, `_persist_turn` `image_shas`). **Open axis — deletion reach:** how
  far the `image_shas` plumbing is torn out (bridge `server.py` `/chat` + `/ws` `image_shas` params +
  `ChatReq` validator; `ImageBlock`; `brain.images.read_image_bytes`). Proposal: remove the
  transport-driving plumbing; **keep** `/upload` + on-disk image storage as-is (widened to general-file only
  under option A of the #43 decision). Back-compat of the wire `image_shas` field is an owner call.

## Footprint — expected touched files (declared for the reviewer's closed set)

Hana-coordination surfaces are flagged ⚠ (per the task: `provider.py`, `engine.py`, frontend).

- ⚠ `brain/bridge/provider.py` — delete `_chat_with_images`, image branch, `_message_has_image`,
  `_build_stream_json_user_message`; touches `chat()` routing.
- ⚠ `brain/chat/engine.py` — remove image builder-fork + `image_shas` plumbing; unify on the text/volatile path.
- ⚠ `app/src/components/ChatPanel.tsx` (+ its tests) — image path-surfacing send path (E3a, IN SCOPE);
  the send-file relabel + accepted-types widening (E3b) is CONTINGENT on option A of the #43 decision.
- `brain/tools/impls/read_file.py` — image-returning branch + size/type gating.
- `brain/mcp_server/tools.py` — `_call_tool` emits `ImageContent` for image results.
- `brain/tools/schemas.py` — `read_file` description (it can now show a shared image).
- `brain/chat/tool_recruit.py` — force-recruit `read_file` on file-send turns.
- `brain/tunables.py` usage / `read_file.py` — `files.image_types`, `files.image_max_bytes`, `files.upload_max_bytes`.
- **NEW** `brain/files.py` — general content-addressable non-image store (`save_file_bytes`/`file_path`),
  sha-only on-disk path (traversal-proof), mirrors `images.py`'s idempotent pattern.
- `brain/bridge/server.py` — `/chat`+`/ws`+`ChatReq` `image_shas` plumbing removal (extent = deletion-reach
  axis). ⚠ **CONTINGENT (option A only):** widening `/upload` + `_ALLOWED_UPLOAD_MEDIA_TYPES` + a non-image
  storage path — the #43-closing work; excluded under the runner's default (option B).
- Possibly `brain/bridge/chat.py` (`ImageBlock`), `brain/images.py` — if `image_shas` plumbing is torn out.
- Tests under `brain/` and `app/src/` covering the above.

## Constraints

- **Preserve the MCP-only disallowed-builtins posture.** Do NOT re-enable built-in `Read`. The viewable
  image must arrive via the MCP `ImageContent` block (the spike's mechanism), through `brain.mcp_server`.
- **Do not break existing chat.** Text-only turns must be byte-for-byte unaffected on the caching-relevant
  assembly (the Option-A/A+ static-system + volatile-suffix split must be preserved).
- **The chat model's tool results flow through the MCP subprocess path** (`_chat_with_mcp_tools`), not the
  in-process `tool_loop.py:420` dispatch (which `json.dumps` results). The image block must be emitted on
  the MCP path. The plan must confirm the in-process path is not the chat path (or handle both).
- Naming in any test/fixture/GitHub-facing text: synthetic user = **Bob**, persona = **Canary**; Phoebe is
  never a fixture; no personal details in GitHub-facing text.
- Local CI before "done": `uv run ruff check .` + `uv run pytest -m "not live and not requires_claude_cli
  and not integration"` (py3.12).

## Prior art

- Spike harness `~/p0-image-spike/` (`mcp_server.py` prototype uses FastMCP's `Image`; the real server
  needs `mcp.types.ImageContent` — different API, same wire result).
- `_chat_with_images` was itself the fix for now-closed **#48** ("attached images not seen"). Whatever
  replaces it must still let her *see* images. C1/C2 prove the **instructed** read renders; **C13** proves
  the **organic** image-share turn (#48's actual scenario — no "call read_file" instruction) still works,
  and is the gating guard on the automatic→opt-in shift.
