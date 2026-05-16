# Full-Session Context + Sticky Sessions — Design Spec

**Date:** 2026-05-10
**Status:** Approved design — ready for plan + implementation
**Scope:** Phase B of the conversation-context work; also closes the
companion 5-minute context-loss bug discovered during Phase A.

---

## Problem

Nell forgets. Two related bugs, same root cause family:

1. **P1 (audit-found):** `brain/chat/session.py:24` truncates `session.history`
   to the last 20 user+assistant pairs. Past ~40 messages the model can't see
   the earlier conversation, even if it just happened minutes ago in the same
   session.
2. **P2 (audit-found):** `brain/ingest/extract.py:114-116` tail-caps the
   transcript at `max_tokens*4` characters (~24KB) before extraction. Long
   sessions lose head context during ingest.
3. **5-minute context-loss (newly reported):** The supervisor's
   `close_stale_sessions(silence_minutes=5.0)` sweep at
   `brain/bridge/supervisor.py:106-116` calls `close_session`, which (a) runs
   the ingest pipeline and (b) **deletes the buffer file and removes the
   session from `_SESSIONS`** at `brain/ingest/pipeline.py:188-189` and
   `brain/bridge/supervisor.py:116`. A user who walks away for 6 minutes
   returns to a brand-new session with zero prior context — even though the
   transcript on disk and the in-memory history both already existed.

All three share the same root: the system treats "time to extract memories"
and "time to end the session" as the same event, and the prompt window is
artificially small.

## Goals

- Nell remembers the full current conversation, end-to-end, for as long as
  the session is alive — no 20-turn cliff.
- Brief absences (under 24 hours, configurable) are invisible: same
  `session_id`, same transcript, conversation continues mid-thread.
- Memory extraction continues at the existing 5-minute cadence but is
  **non-destructive** — it commits durable items to `MemoryStore` without
  deleting the buffer or evicting the session.
- Long absences (≥24h) and explicit close still finalize the session: ingest
  one last time, delete the buffer, drop from `_SESSIONS`.
- Edge cases (multi-hour sessions, transcript bloat) handled by a budget
  guard that compresses oldest turns only when the prompt would exceed the
  model's context window.
- **No new user-visible CLI knobs.** Cadence constants live in code.

## Non-Goals

- Cross-session memory linking (already handled by `MemoryStore` +
  `HebbianMatrix`).
- Streaming summaries to disk on every tick. The buffer is already the
  durable record; we read from it, we don't double-write summaries.
- Changing the extraction prompt or `ExtractedItem` schema.
- Removing the old `session.history` field outright — keep it as a debugging
  aid / fallback, but stop *relying* on it for the prompt.

---

## Architecture

Four moving parts, each small. Order of dependency is roughly engine →
pipeline → supervisor → budget guard.

### 1. Live prompt reads the buffer (`brain/chat/engine.py`)

`engine.respond` currently builds the messages list as:

```python
messages = [system, *session.history, user]
```

Change it to read the buffer file:

```python
buffer_turns = read_session(persona_dir, session.session_id)
history_msgs = _buffer_turns_to_messages(buffer_turns)
messages = [system, *history_msgs, user]
```

- `_buffer_turns_to_messages` converts the JSONL records produced by
  `_persist_turn` into `ChatMessage` objects. Speaker `"user"` → user role,
  `"assistant"` → assistant role. Image-bearing user turns reconstruct
  `(TextBlock, *ImageBlock)` content tuples from the recorded `image_shas`,
  same shape as `_build_user_message` already produces for the live turn.
- The buffer is *the* source of truth for in-session memory now.
  `session.history` is still maintained by `append_turn` for backwards
  compatibility (anything reading it directly), but `engine.respond` no
  longer touches it for prompt construction.
- `HISTORY_MAX_TURNS = 20` stays as a sanity ceiling on the in-memory list,
  but it's no longer load-bearing. We could raise it to 5000 or leave it at
  20 — it doesn't affect prompt fidelity anymore.
- **Ordering invariant:** `_persist_turn` writes BOTH the user and assistant
  turn *before* `session.append_turn` runs (engine.py:178-185). On the next
  call, `read_session` therefore sees turn N already on disk before turn N+1
  starts — the read returns a *complete* prior history with no torn turn.

### 2. Sticky sessions — sweep extracts, doesn't destroy

`close_stale_sessions` currently calls `close_session`, which both extracts
*and* deletes. Split the responsibility:

- New function: `extract_session_snapshot(persona_dir, session_id, *, store,
  hebbian, provider, embeddings=None, config=None) -> IngestReport`. Runs
  stages BUFFER → EXTRACT → SCORE → DEDUPE → COMMIT → SOUL → LOG. **Does
  not** call `delete_session_buffer`. Marks the snapshot's tail timestamp in
  a new sidecar file `<persona>/active_conversations/<sid>.cursor` so the
  next snapshot only re-extracts turns added since the previous one.
- Sidecar format: a single line containing the ISO-8601 `ts` of the most
  recent turn that was included in the last successful extraction pass. If
  the cursor is missing or malformed, fall back to "extract from the
  beginning" (current behavior).
- `extract_session_snapshot` reads turns whose `ts > cursor` (or all turns,
  if no cursor). If that filtered list is empty, the function logs
  `conversation_snapshot_empty` and returns an empty report — no LLM call,
  no buffer change.
- After a successful extraction, write the new cursor (the `ts` of the last
  turn in the filtered list) atomically (`.tmp` + `os.replace`).
- `close_stale_sessions` continues to walk active sessions and apply the
  silence window, but now calls `extract_session_snapshot` instead of
  `close_session`. It still returns an `IngestReport` list — same shape, so
  existing event-publish + observability code is untouched.
- The supervisor stops calling `remove_session(sid)` for sessions returned
  by the snapshot sweep. The session stays in `_SESSIONS`; the buffer file
  stays on disk; the user comes back and continues.

### 3. Real close — long-silence OR explicit

A second cadence in the supervisor handles true session end:

- New helper `finalize_stale_sessions(persona_dir, *, finalize_after_hours:
  float = 24.0, store, hebbian, provider, embeddings=None, config=None) ->
  list[IngestReport]`. Iterates active buffers, computes
  `session_silence_minutes` (existing helper at `buffer.py:101`), and for any
  session at or beyond `finalize_after_hours * 60` minutes:
  1. Runs `extract_session_snapshot` one last time (catches any turns added
     after the most recent snapshot).
  2. Calls `delete_session_buffer(persona_dir, sid)`.
  3. Returns the report; the supervisor follows up with
     `remove_session(sid)`.
- The supervisor calls `finalize_stale_sessions` on its own cadence —
  default once per hour, so we don't waste ticks scanning for 24h-old
  sessions every 60 seconds. Constant lives in
  `brain/bridge/supervisor.py`.
- Explicit close (`POST /sessions/close` and the REPL's exit path) still
  calls the existing `close_session` — which already deletes the buffer and
  removes the session. **One change:** `close_session` should check for and
  remove the `<sid>.cursor` sidecar file too, so a clean close leaves no
  litter.

### 4. Budget guard — fallback compression

For multi-hour sessions whose buffer would push the system prompt past the
model's window, compress the oldest portion of the buffer when constructing
the prompt:

- New module: `brain/chat/budget.py`.
- `apply_budget(messages: list[ChatMessage], *, max_tokens: int = 190_000,
  preserve_tail_msgs: int = 40, provider: LLMProvider) -> list[ChatMessage]`.
  Crude size estimate using `len(text) // 4` per message; only triggers when
  the estimate exceeds `max_tokens`.
- When triggered, partition messages into `[head_to_compress, preserved_tail]`
  (last 40 messages always preserved verbatim). Concatenate head into a
  single transcript, call `provider.generate` with a compact
  "Summarize for context preservation, preserve names, decisions, emotional
  beats, and unresolved threads" prompt, return a single
  `ChatMessage(role="system", content="[Earlier in this conversation: ...]")`
  followed by the preserved tail.
- Result inserted between the original system message and the preserved tail
  in `engine.respond`. The original system message is never compressed.
- Budget guard fires on the construction path, not on the buffer. The buffer
  on disk stays complete and faithful.
- 200K Claude Sonnet budget − ~10K headroom for the response = 190K default.
  Tuning lives in the function signature; no env var.

---

## Data Flow

### Happy path — turn N in an existing session

1. `engine.respond(persona_dir, user_input, session=existing_session, ...)`.
2. `_build_user_message` constructs the live user `ChatMessage`.
3. **New:** `read_session(persona_dir, session.session_id)` returns prior
   turns as dicts.
4. **New:** `_buffer_turns_to_messages(turns)` reconstructs the message list.
5. **New:** `apply_budget(messages, max_tokens=190_000, provider=...)` if the
   crude estimate exceeds the budget — otherwise pass through.
6. `run_tool_loop(messages, ...)` proceeds unchanged.
7. `_persist_turn` writes both user and assistant turn to the buffer.
8. `session.append_turn` updates `session.history` + `last_turn_at`.
9. Return `ChatResult`.

### Brief absence — 5-minute snapshot, user returns at 6 minutes

1. Supervisor tick at T=5min: `close_stale_sessions` finds the session whose
   last turn was 5min ago. **Now calls** `extract_session_snapshot`. Reads
   buffer, filters by cursor (no cursor on first pass = full buffer),
   extracts memories, commits to `MemoryStore`, writes cursor file. Returns
   an `IngestReport`. Buffer stays. Session stays in `_SESSIONS`.
2. Event published: `session_snapshot` (new type, see Observability below).
3. User sends a message at T=6min. Bridge looks up the existing
   `session_id`, finds the still-live `SessionState`, calls
   `engine.respond(session=existing)`. `read_session` returns the full prior
   transcript. Conversation continues mid-thread.

### Long absence — 25 hours, no return

1. Supervisor's hourly `finalize_stale_sessions` pass at T=25h sees the
   session's last turn is 25h old.
2. Runs `extract_session_snapshot` one last time (cursor likely already
   covers everything, so it's a fast no-op).
3. Calls `delete_session_buffer` + supervisor calls `remove_session`.
4. Event published: `session_finalized`.
5. User opens the app two days later. New session created via
   `/session/new`. Prior conversation lives in `MemoryStore` as extracted
   memories, surfaced through the existing recall block in
   `build_system_message`.

### Explicit close — Cmd-Q or `/sessions/close`

Unchanged from today: `close_session` runs, buffer is deleted, session
removed. We additionally clear the `<sid>.cursor` sidecar.

### Edge: image turn replay

A user turn that originally carried `image_shas` is replayed from buffer.
`_buffer_turns_to_messages` reconstructs `ImageBlock` entries using
`media_type_for_sha`. If the image bytes were deleted between session start
and replay, log a warning and drop that image block from the replayed
message (same defensive behavior as `_build_user_message:222-234`).

---

## Components

| File | Change |
|---|---|
| `brain/chat/engine.py` | Switch prompt builder to read buffer; insert budget guard. |
| `brain/chat/budget.py` | **New.** `apply_budget(messages, ..., provider) -> list[ChatMessage]`. |
| `brain/ingest/pipeline.py` | **New** `extract_session_snapshot`. Existing `close_session` clears `.cursor` sidecar. `close_stale_sessions` switches to snapshot path. **New** `finalize_stale_sessions`. |
| `brain/ingest/buffer.py` | **New** `read_cursor` / `write_cursor` / `delete_cursor` helpers. **New** `read_session_after(persona_dir, session_id, after_ts)`. |
| `brain/bridge/supervisor.py` | Replace destructive sweep with snapshot sweep; stop calling `remove_session` for snapshot returns; add hourly `finalize_stale_sessions` cadence; new event types. |
| `brain/chat/session.py` | Optional: raise `HISTORY_MAX_TURNS` to 5000 (or leave) — `session.history` is now informational. Document the demotion in module docstring. |
| `tests/unit/brain/chat/test_engine.py` | Add cases for buffer-driven prompt, image-bearing replay, budget pass-through. |
| `tests/unit/brain/chat/test_budget.py` | **New.** Unit tests for `apply_budget` (no-op, compression triggers, tail preservation). |
| `tests/unit/brain/ingest/test_pipeline.py` | Add cases for `extract_session_snapshot` (cursor write, idempotent re-run, empty filtered list, cursor recovery from malformed). |
| `tests/unit/brain/ingest/test_buffer.py` | Add cases for cursor helpers + `read_session_after`. |
| `tests/unit/brain/bridge/test_supervisor.py` | Update existing close-stale tests: assert buffer survives, assert `_SESSIONS` retains the id, assert cursor is written. Add new `finalize_stale_sessions` tests. |
| `tests/integration/brain/bridge/` | **New** integration: live session → 5-min sweep → user returns → conversation continues. |

---

## Error Handling

- **Cursor read corrupt / missing:** Treat as "no prior cursor — extract
  from the beginning." Logged at INFO. Never raises.
- **Cursor write fails (disk full, permission):** Log WARN, return the
  report anyway. Next snapshot pass will re-extract the same turns —
  duplicates absorbed by `is_duplicate` in COMMIT stage. No data loss.
- **Snapshot extraction fails (LLM error):** Same as today's
  `close_session` failure path — log + return an `IngestReport` with
  `errors=1`. Buffer + cursor untouched. Retry next sweep.
- **Buffer file unreadable mid-session (e.g. transient FS error):** Engine
  logs ERROR, falls back to `session.history` for that turn so the user
  doesn't get an empty-context Nell. After the turn, normal flow resumes
  and the buffer write goes through.
- **Budget guard summarization fails:** Log WARN, fall back to a
  deterministic head-truncation (`messages = [system, *messages[-tail:]]`
  with a `[truncated N earlier messages]` system note). Conversation
  degrades, but never crashes.
- **Finalize sweep dies mid-run:** Per-session try/except inside the loop,
  identical to today's supervisor. One bad session can't kill the loop.

---

## Observability

New events on `event_bus`:

- `session_snapshot` — emitted by the snapshot sweep instead of the old
  `session_closed`. Fields: `session_id`, `committed`, `deduped`,
  `soul_candidates`, `extracted_since_cursor`, `errors`, `at`.
- `session_finalized` — emitted by `finalize_stale_sessions`. Fields:
  `session_id`, `total_silence_hours`, `final_committed`, `final_deduped`,
  `errors`, `at`.
- `session_closed` — kept for the explicit-close path (`/sessions/close`,
  REPL exit). Same shape as today. Renderers that listen for "session
  ended" should listen on `session_closed | session_finalized`.
- `budget_compression_triggered` — emitted by `apply_budget` when it fires.
  Fields: `session_id`, `head_msgs_compressed`, `estimated_tokens_before`,
  `estimated_tokens_after`, `at`. Lets us measure how often the rare path
  runs in production.

Logging:

- `conversation_snapshot session=<sid> turns_seen=<N> turns_extracted=<M>
  committed=<C> deduped=<D> cursor=<ts>` on every snapshot pass.
- `conversation_finalized session=<sid> silence_hours=<H>` on every
  finalize pass.

---

## Testing Strategy

TDD per `superpowers:test-driven-development`. Each new function lands
with its test first. Roughly:

1. **Cursor primitives** (`test_buffer.py`): write/read/delete cursor;
   read_session_after with cursor in the middle / at the tail / before
   the head / missing file.
2. **`extract_session_snapshot`** (`test_pipeline.py`): empty buffer (no
   LLM call), no cursor (full buffer extracted, cursor written), existing
   cursor (only post-cursor turns extracted), idempotent re-run after a
   commit (no duplicates because dedupe + cursor advance), malformed
   cursor (fall back, recover).
3. **`finalize_stale_sessions`** (`test_pipeline.py`): under-threshold
   session skipped, at-threshold session finalized (buffer deleted,
   `.cursor` deleted, report returned), per-session try/except on a
   forced provider error.
4. **`apply_budget`** (`test_budget.py`): below threshold → identity
   transform, above threshold → tail preserved + compressed head present,
   summarization failure → deterministic fallback note + tail.
5. **Engine** (`test_engine.py`): respond reads from buffer, not
   `session.history`; image turns replay correctly; engine recovers when
   buffer read fails by falling back to `session.history`.
6. **Supervisor** (`test_supervisor.py`): snapshot sweep does not delete
   the buffer or remove the session id; emits `session_snapshot` event;
   finalize cadence runs at the configured interval; explicit close still
   clears the cursor.
7. **Integration**: fake LLM provider, real persona dir on tmp_path,
   simulate a 50-turn conversation, sleep past the silence window, run
   a supervisor tick, send a new turn, assert the prompt sent to the
   fake provider contains all 50 prior turns.

Coverage targets follow the existing brain conventions — every new
public function in `pipeline.py` / `buffer.py` / `budget.py` has at least
one unit test for the happy path and one for the error path.

---

## Migration / Backwards Compatibility

- **Existing buffers** at upgrade time: have no `.cursor` sidecar. First
  snapshot pass extracts the full buffer (just like today). Subsequent
  passes use the cursor. No migration script needed.
- **Existing in-memory sessions** at process restart: lost — that's
  already the case today. The buffer file remains, the next user message
  on that `session_id` (if the client retries with the same id) will
  re-attach to the buffer transparently. If the client creates a fresh
  `session_id`, the orphaned buffer is picked up by the next snapshot
  sweep just like today. Note: HTTP clients always create a new
  `session_id` on `/session/new`, so this only affects the REPL path
  where the user could theoretically resume.
- **`session.history`** is no longer authoritative but stays populated.
  Anything reading it (tests, telemetry, debugging) keeps working.
- **Event consumers** (renderers listening on `/events`): need to handle
  `session_snapshot` and `session_finalized` in addition to today's
  `session_closed`. The renderer currently treats `session_closed` as
  "session is over" — that semantic now belongs to `session_finalized` +
  the explicit-close path. The Phase B plan includes the corresponding
  renderer-side change.

---

## Open Questions (resolved)

- **Q1: Context budget.** Answered: 200K Claude Sonnet. Compression only
  triggers at >190K estimated tokens.
- **Q2: Lifecycle cadence.** Working assumption: **24h** silence before
  real close. (Constant lives in `supervisor.py`. Easy to tune later if
  Hana wants 12h or 48h.)
- **Q3: Sidecar location.** Cursor lives next to the buffer
  (`<persona>/active_conversations/<sid>.cursor`) so deleting the
  buffer + cursor stays one cohesive operation.

## Out of Scope

- Multi-device session resumption (today's bridge is single-process).
- Memory consolidation triggered by snapshot rate (current heartbeat
  cadence is unchanged).
- Compressing the buffer file on disk — only the prompt construction
  path compresses, when needed.
