# P0 image-tool-route — Plan

How to build the change, how each criterion is measured, what instrumentation is needed, and the
severity→routing thresholds. Scope A adds a **new content-addressable file store** (`brain/files.py`,
step 9); it mirrors `save_image_bytes`'s proven idempotent tmp-then-`os.replace`+dedup pattern and adds no
new read-modify-write window over shared mutable state, so the concurrency-lens plan duties (ST2b) still do
not fire — but the final red-team re-checks this against the new store rather than inheriting the judgment.
The prompt-assembly change IS position-sensitive, so C5/C9/C14 carry execution-based behavior-preservation
checks.

## Build steps

1. **`read_file` returns images (E1/E2).** In `brain/tools/impls/read_file.py`:
   - **Cap-ordering is load-bearing (fixes red-team F-1).** The existing text size-cap check is at
     `read_file.py:102-114` (`if size > files.read_max_bytes: return {"error": "file too large…"}`) — it
     fires **before** the current UTF-8 decode at :131. The image branch MUST be inserted **before that
     text-cap check (before :102)**, not merely "before the decode": otherwise every image between the
     256KB text cap and `files.image_max_bytes` is refused by the text cap and the separate image cap E2
     introduces is dead code (effective image cap collapses to `min(256KB, image_max_bytes)`).
   - Concretely: after the existing existence check (:89-100), **sniff the media type from a small byte
     prefix** — read the first ~32 bytes and pass to `brain.images.sniff_media_type(bytes)` (verified to
     take `bytes`, magic-byte prefix suffices; returns a media_type or `None`). If the sniff yields an
     image type in `files.image_types`: gate on the **image** cap — `if size > files.image_max_bytes:`
     refuse with an image-specific "image too large" error (its own audit `error="image too large"`);
     else read the full bytes and return a **structured image result** — e.g. `{"path": ..., "image":
     {"media_type": <mt>, "data_b64": <base64>}}` (a discriminable key the MCP adapter keys off). Do NOT
     base64 into the text `content` field. Dedup + audit still apply on the image path (audit summarizes
     as `image/<mt> <N>B`, NOT the base64 — see step 2 / red-team G5).
   - If the sniff is `None` / not an image type, **fall through unchanged** to the existing text-cap check
     at :102 and today's text handling (256KB text cap, decode, dedup, head-cap, audit). Non-image
     binaries stay refused (`{"note": "(binary file …)"}`) unless the owner-sign-off widens this.
   - Register `files.image_types` and `files.image_max_bytes` via `tunables.register(...)` next to
     `files.read_max_bytes`. The image size cap is checked on the image branch **separately from and
     ahead of** the text cap.
2. **MCP adapter emits `ImageContent` (E1).** In `brain/mcp_server/tools.py:_call_tool`, after `dispatch`
   returns, if the result dict carries the `image` key, return `[ImageContent(type="image", data=<b64>,
   mimeType=<mt>)]` (import from `mcp.types`) instead of `[TextContent(...)]`. Text results are unchanged.
   Preserve the audit-log call in both branches (log the invocation as `ok`; summarize the image as e.g.
   `image/png <N>B` so the audit stays useful and doesn't dump base64).
3. **Confirm the chat path is the MCP subprocess path.** Verify chat tool-calls flow through
   `_chat_with_mcp_tools` (`provider.py:657`) → the model sees `_call_tool`'s blocks. The in-process
   `tool_loop.py:420` dispatch (`json.dumps(result)`) is a *different* provider path; confirm it is not the
   companion chat path (or, if it is ever reachable for chat, degrade the image result there to a graceful
   text note — not a crash). Record the finding in `decisions.md`.
4. **Force-recruit `read_file` on file-send turns (E5).** Thread a "a file was shared this turn" signal into
   recruitment so `read_file` is force-included in `select_tools`'s output on that turn, independent of
   salience score. Smallest form: `engine.respond` passes a flag to `select_tools`/`build_tools_list` that
   unions `{"read_file"}` into `allowed` when a file path was shared.
5. **Surface the path through the normal text turn (E4).** In `engine.respond`, replace the `image_shas`
   builder-fork (`engine.py:187-208`): when a file was shared, render its path into the user turn's **text**
   (a clearly-marked line, e.g. `[the user shared a file: <path>]`) and take the **normal** text/volatile
   branch (`build_static_system_message` + `build_volatile_context`). Delete the `image_shas` path from
   `_build_user_message`, buffer replay (`engine.py:394-405`), and `_persist_turn`. The shared-file record
   in the buffer becomes plain text (the path), so replay needs no image handling.
   - **Do NOT let the surfaced path line contaminate salience/volatile computation (fixes red-team F-4/G4).**
     `build_volatile_context` and `assess_salience` are fed `user_input` (`engine.py:199-208`); if the
     `[the user shared a file: <path>]` line is prepended into the `user_input` that feeds them, the path
     string perturbs salience flags and volatile memory-seeding. Compute salience/volatile on the user's
     **actual typed text**, and inject the path line only into the **outgoing user message** handed to the
     provider (after volatile/salience are computed). This keeps the file-send turn's volatile *content*
     equivalent to the same turn without a file — verified by C14.
6. **Delete the transport (E6).** Remove `_chat_with_images` (`provider.py:1088-1281`), the `has_images`
   branch (`provider.py:619-637`), `_message_has_image` (`provider.py:1780`), `_build_stream_json_user_message`
   (`provider.py:1787`), and the engine image-block fork (`engine.py:187-208` image branch, the `_build_user_message`
   image-block build `engine.py:321-342`, buffer replay `engine.py:356-401`, `_persist_turn` image_shas write
   `engine.py:417-432`). **Deletion reach (was axis 1; runner's bounded call per owner "plumbing calls stand"):**
   the transport symbols above go; the wire path becomes a **file reference the engine resolves to a path**
   (step 5a). Keep the content-addressable stores. `ImageBlock` (`brain/bridge/chat.py:34`) and
   `brain.images.read_image_bytes` become unused by the chat path — remove them if no non-test caller remains
   (verified by C6's grep), else leave dormant with a note. Ingest metadata (`buffer.py`/`commit.py`/`extract.py`
   `image_shas`) is downstream of the buffer; since file-send turns now persist plain text (step 5), those
   paths simply receive no image_shas — leave them, they are not the transport.
   - **Accepted-risk — migration of pre-existing buffers (red-team lap-2 CH8-2; ratified by owner).** Deleting
     the buffer-replay image handling (`engine.py:356-401`) means a session buffer that shared an image *before*
     the upgrade replays as text-only (the image silently vanishes from later context). Owner-accepted as a
     one-time pre-release cost (see decisions.md).
7. **Frontend send-path change (E3a, IN SCOPE) + "send file" relabel (E3b, scope A).** `ChatPanel.tsx`: change
   the send path so a staged upload's **reference (sha + kind/ext)** is sent with the normal message (the
   engine resolves it to a path, step 5a) instead of an `image_shas` transport payload; relabel `attach image`
   → `send file` and widen the accepted-types set to include ≥1 non-image type (backed by the widened `/upload`,
   step 9). Update `ChatPanel.*.test.tsx` + `streamChat.ts`/`bridge.ts` wire shape. Covers C8a + C8b.
8. **Schema copy (`brain/tools/schemas.py`).** Update `read_file`'s description so the model knows it can be
   handed any shared file (incl. images) and read/see it — without encouraging proactive reads.

### Settled build specifics (runner's bounded calls under "plumbing calls stand")
- **Wire contract:** replace `ChatReq.image_shas: list[str]` (+ `/ws` param + `engine.respond(image_shas=)`
  + `_respond_blocking`) with `shared_files: list[{kind: "image"|"file", sha: 64hex, media_type?, filename?}]`
  (length-capped like image_shas). The frontend's `/upload` returns the ref; the message carries it.
- **Bounded deletion reach:** the engine now persists a file-send turn as **plain text** (the path line lives
  in the user text), so `_persist_turn` stops writing `image_shas` and buffer replay drops the ImageBlock
  branch. The downstream ingest `image_shas` metadata (`buffer.py`/`commit.py`/`extract.py`) and the
  `/images` gallery + `_extract_recent_image_shas` simply stop being populated for new turns — **leave them
  (dead-but-harmless; not the transport).** Pre-change buffers keep their old rows (owner-accepted migration).
- **Path-line format:** `[the user shared a file: <abs-path>]` (one line per file), appended to the OUTGOING
  user text only; if a filename is known, `[the user shared a file "<name>": <abs-path>]`.
- **PDF/binary:** stored + path surfaced; `read_file` returns today's "binary — not shown" note for non-UTF-8
  bytes (no extraction this PR). PDF-text-readability is escalated to the owner at the post-build gate.
- **Live criteria C1/C13 (and the live arm of C16) need the real `claude` CLI + pinned model — usage-gated,
  run at stage 8; the build lands the code + the NON-live pytest/vitest criteria + local CI now.**

### Scope-A build steps (close #43 — widen upload + non-image storage; owner ruling)
9. **General file store (new `brain/files.py`, mirrors `images.py`).** `save_file_bytes(persona_dir, data)
   -> FileRecord(sha, size_bytes)`, content-addressable at `<persona_dir>/files/<sha>` — **the on-disk name
   is the validated 64-hex sha ONLY; no client-supplied filename or extension is ever a filesystem path
   component (fixes final-red-team Finding 2, path-traversal).** `_validate_sha` (reuse the `images.py`
   regex) gates the sha; `file_path(persona_dir, sha)` resolves under `persona_dir/files/` and nowhere else.
   Atomic per-writer tmp + `os.replace` + dedup, exactly mirroring `save_image_bytes`'s idempotent race
   handling. The original filename is preserved only as **display metadata** returned to the caller (surfaced
   in the path-line text for the model, step 11) — never used to build the path. Because the on-disk key is
   pure sha, dedup is by content alone (dissolves the (sha,ext) dedup wrinkle, Finding 4). A general size cap
   `files.upload_max_bytes` (register in tunables; default ≥ the 20MB `/upload` image cap so caps are
   coherent). This is a **new content-addressable store**; the concurrency lens re-checked it and it adds no
   new read-modify-write window beyond the proven idempotent `save_image_bytes` pattern.
10. **Widen `/upload` (`server.py:2150-2201`, ⚠ Hana surface).** Sniff bytes: if `sniff_media_type` is an
    image → existing `save_image_bytes` path (unchanged, incl. its sniff-vs-declared integrity check). Else →
    `save_file_bytes` (general store), gated by `files.upload_max_bytes` and a widened accept policy (drop the
    image-only 415; accept non-image files up to the cap; keep the 413 size gate). Return a discriminated ref:
    `{kind: "image"|"file", sha, media_type?, filename?, size_bytes}` — the on-disk path is derived from the
    validated sha (+ media_type for images), NOT from `filename` (which is display-only, Finding 2).
11. **Engine resolves a file reference to a path (E4/step 5a).** On a turn carrying a shared-file reference
    (sha + kind), the engine resolves the on-disk path via `images.image_path(persona_dir, sha, media_type)`
    (kind=image) or `files.file_path(persona_dir, sha)` (kind=file) — both sha-validated, both resolving under
    the persona dir only — and surfaces `[the user shared a file: <path>]` (optionally annotating the original
    filename as display text) on the outgoing user message (not into salience/volatile — step 5), and
    force-recruits `read_file` (step 4). No `ImageBlock`, no transport. Persist the reference as plain text in
    the buffer.

## Measurement — how each criterion is verified (stage 8)

| Criterion | How measured | Instrumentation needed |
|---|---|---|
| C1 viewable image via real server | Extend spike harness to drive `python -m brain.mcp_server --persona-dir <sandbox>` under real posture; 3 fresh-secret runs, pinned model; string-eq. **Harness resolves the model from the persona config (the same resolution `provider.py` uses, `DEFAULT_MODEL` fallback), NOT a hardcoded alias** (red-team C1 caveat) | New harness under `changes/p0-image-tool-route/harness/` (seeds from `~/p0-image-spike/`) |
| C2 negative control | Same harness, image path disabled → assert secret absent | same harness |
| C3 text still text | Unit test: `read_file` on a text fixture returns known line | pytest |
| C4 posture preserved | Unit assertion on `_BUILTIN_TOOLS_DISALLOWED` + `_apply_lean_flags` | pytest |
| C5 volatile survives post-file turn | Two-turn engine test; assert suffix present post-change AND absent pre-change (base commit) | pytest + a pre-change run |
| C6 no fork remains | `git grep` removed symbols = empty in source; one assembly path exercised | pytest + grep |
| C7 path reaches model + read_file available | Assert path substring in assembled turn; `read_file ∈ select_tools()` for file-send turn | pytest |
| C8a image send-path change (gating) | `app/src` component test: send posts normal message with a file reference, sets no `image_shas` transport payload | vitest/jest |
| C8b send-file relabel + non-image (gating, scope A) | `app/src` component test: "send file" label + ≥1 non-image accepted type | vitest/jest |
| C16 #43 end-to-end (gating, scope A) | pytest: POST a text fixture to widened `/upload` → stored → `read_file <path>` returns the known line; shown-able-to-fail = pre-change `/upload` rejects the text file (415/422) | pytest (FastAPI TestClient + read_file) |
| C9 text-only byte-preserving | Diff assembled static system + suffix placement pre/post | pytest |
| C10 CI green | `ruff` + `pytest` exit codes + frontend suite | existing CI |
| C11 cost/cache (adv) | Config regression metrics over representative turns | existing logs (advisory only per config H8) |
| C12 large image not refused | `read_file` on an image 256KB < size ≤ `image_max_bytes` returns an image result (not "too large"); shown-able-to-fail = an image > `image_max_bytes` IS refused, AND the same >256KB image returns "too large" on pre-change/mis-ordered code | pytest (unit; no model needed) |
| C13 organic (uninstructed) image-seeing | Real-server harness: a **natural** turn (image renders a fresh secret; user asks about it with NO "call read_file" instruction) → model reports the secret; shown-able-to-fail = text-only/no-image control → secret absent | C1/C2 harness, natural-prompt arm |
| C14 file-send turn volatile equivalence | Assemble a file-send turn and the same turn's text without a file; assert the volatile/salience-derived content is equivalent (path line not fed to salience/volatile) | pytest |

**Instrument-before-build (CP3):** the only new instrumentation is the C1/C2/C5 harness + the pre-change
comparison runs; all are authored in stage 8's harness dir. No production telemetry gap is opened (the
change removes a path; it does not add an unmeasured one). The audit log already records `read_file`
invocations — step 2 keeps that, adding an image-size summary field so image reads remain auditable.

## Severity → routing thresholds (gates 4/7/8)

- **Blocker** (restart loop / escalate to owner): the mechanism does not hold on the real server/pinned
  model (C1 fails ≥1 of 3, or C2 leaks the secret) → the whole direction's premise is void; stop, escalate
  to owner (fall back to the spike's documented image-transport alternative — NB: distinct from the #43
  "scope A"). Also blocker: **C13 fails — the model does NOT
  reliably see an organically-shared image without an explicit read instruction** (the #48 requirement is
  not met by opt-in-via-tool-call → the design must change, e.g. proactively emit the `read_file` result
  on a file-send turn; this is an owner-facing design decision, not a silent runner fix); text-only chat
  regresses (C9 diff non-empty); or CI red (C10).
- **Major** (bounce to build): C5 does not reproduce-pre / fix-post; C6 leaves a live fork; C7 leaves
  `read_file` un-recruitable on a file turn; **C12 refuses a >256KB image** (cap-ordering regression, F-1);
  **C14 shows the surfaced path line contaminating salience/volatile**; **C16 fails — the widened `/upload`
  does not store a non-image file or its path is not readable via `read_file`** (#43 not closed); **C8a/C8b
  frontend send-path/relabel wrong**.
- **Minor/advisory:** C11 deltas; audit-summary formatting (incl. the G5/C15 base64-in-audit check).

## Axes — RESOLVED by owner ruling 2026-08-12 (see decisions.md)

1. **Deletion reach** — runner's bounded call stands (owner: "your other plumbing/wording calls stand"):
   remove the transport symbols; wire becomes an engine-resolved file reference; remove `ImageBlock`/
   `read_image_bytes` if C6's grep shows no non-test caller, else leave dormant. (Step 6.)
2. **`files.image_max_bytes` default ≥ 20MB** (owner-ratified — no dead zone). The new
   **`files.upload_max_bytes`** tunable is a scope-A addition under "your other plumbing calls stand" (NOT
   the dead-zone ruling); set ≥ 20MB for coherence with the `/upload` cap (runner's call).
3. **Path-surfacing wording** `[the user shared a file: <path>]`, injected only on the outgoing user
   message (step 5); exact wording is the runner's call (stands).
4. **Seeing = OPT-IN** (owner-ratified, aligned with issue #124 — Kindled agency). C13 stays a GATING
   guard; its failure routes as a blocker to the owner.
5. **Scope = A** (owner-ratified): close #43 in this PR — steps 9-11 widen `/upload` + add the general
   store; C8b + C16 are gating.
6. **Single-file per turn** — out of scope (unchanged; matches current `ChatPanel` `stagedImage`).

The direction and these axes are owner-ratified; the run now takes the final confirmation red-team on the
settled criteria, freezes criteria, and builds.
