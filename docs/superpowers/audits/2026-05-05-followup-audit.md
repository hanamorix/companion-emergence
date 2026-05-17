# Follow-up audit — companion-emergence

**Date:** 2026-05-05 (follow-up to the original 2026-05-05 audit fix pack)
**Auditor:** code-reviewer agent
**Scope:** `brain/` (tools, works, migrator, soul, chat, bridge, ingest, body)
**Skipped (already scheduled):** I-5 (concurrency lock for `nell migrate`), M-1 (extract `save_with_backup_text` helper)

This pass re-walked the audited surface after commit `f5c62b5` shipped fixes for I-1, I-2, I-3, I-4, I-6 and confirms those land correctly. Findings below are NEW issues uncovered while re-reading the code, plus a handful of latent issues the original audit did not call out.

---

## Important issues

### I-1: `cmd_tail` builds a WS URL with `?token=…` — server doesn't accept that, and the token leaks into URL space

**File:** `brain/bridge/daemon.py`
**Line(s):** 256–275 (`cmd_tail`), specifically 267–268

```python
token_qs = f"?token={s.auth_token}" if s.auth_token else ""
url = f"ws://127.0.0.1:{s.port}/events{token_qs}"
```

**Description.** The server's WebSocket auth path (`brain/bridge/server.py:484-518`) reads the token from `Sec-WebSocket-Protocol: bearer, <token>` only. It never reads `?token=` from the query string. So in production (auth enabled — `runner.py` always generates a token), `nell bridge tail` will attempt to connect, the server will reject with `4001 missing token`, and the user sees a confusing connection drop. Worse: query strings end up in `ps` output, OS-level network logs, and any HTTP/WS proxy access logs the user has running. The bearer token is the *only* thing protecting the bridge HTTP surface — leaking it is the same severity as leaking a session cookie.

**Severity rationale.** Privacy + correctness. Token-in-URL is a textbook auth-token-leak antipattern; combined with the broken auth path it means the documented `nell bridge tail` workflow is silently busted in the configuration that ships in real use.

**Suggested fix.** Use `websockets.sync.client.connect(url, additional_headers=...)` with `Sec-WebSocket-Protocol: bearer, <token>` (websockets supports `subprotocols=["bearer", token]`). Drop the `?token=` query string. Add a regression test with auth enabled that exercises the tail path against the real server.

---

### I-2: `ingest_turn` writes to `<persona>/active_conversations/<session_id>.jsonl` with no `session_id` validation — path traversal / clobber risk

**File:** `brain/ingest/buffer.py`
**Line(s):** 31–55

`_session_path` interpolates the `session_id` straight into a filename: `_active_conversations_dir(persona_dir) / f"{session_id}.jsonl"`. The HTTP bridge constrains `session_id` to a UUID at the request model layer (`brain/bridge/server.py:276,281`), but `ingest_turn` is called from other paths too — `brain/chat/engine.py:_persist_turn` passes the engine-supplied `session_id`, and any future direct caller (CLI, tests, MCP) inherits the bug. A `session_id="../../etc/passwd"` would write outside the persona dir; `session_id=""` would create `.jsonl`; `session_id=".."` would write `..jsonl` next to the target dir.

**Severity rationale.** Path-traversal class bug on a write path that consumes user-influenced input on the bridge surface. Today the only public entrypoint validates, but the function's contract advertises "Optional: session_id" and silently accepts arbitrary strings. One careless future caller turns this into RCE-adjacent data corruption.

**Suggested fix.** In `_session_path`, validate session_id with a regex like `^[A-Za-z0-9_-]{1,64}$` (UUID4 + the `sess_<8 hex>` fallback both fit). Raise `ValueError` on mismatch. Mirror the validation that `brain/works/storage.py:_work_path` already does for work IDs.

---

### I-3: `WorksStore.search` is the only FTS5 OperationalError catch — `WorksStore.list_recent` and the callers downstream still raise

**File:** `brain/works/store.py`
**Line(s):** 129–141 (`list_recent`); compare with the protected 143–167 (`search`)

The audit fix-pack for I-2 wrapped `search()` with `try: ... except sqlite3.OperationalError: return []`, which is correct. But `list_recent` and `get` still execute against the `works` table without the same protection. If the SQLite file is corrupted, the WAL is partial after a crash, or a future schema bump breaks rollover, an `OperationalError` will surface as a 500 from `/self/works` and `/self/works/{id}`. The bridge route handlers call these directly without catching SQLite-class exceptions (only `read_work` checks for "no works.db / not found / corrupt" via its own dict-error contract).

**Severity rationale.** Same shape as the original I-2 — a SQLite operational state shouldn't take down a chat-bridge HTTP route. Already caught one path; the symmetric paths are still exposed.

**Suggested fix.** Either wrap each `list_recent`/`get` execute in the same `try/except sqlite3.OperationalError`, OR push the catch up one level into the bridge route handlers and translate to 503 with a structured error body. Prefer the former — keeps store surface uniform, mirrors the I-2 fix.

---

### I-4: `save_work` reorder is correct, but `WorksStore.insert` is *not* atomic across `works` + `works_fts` insertion

**File:** `brain/works/store.py`
**Line(s):** 85–118 (`insert`)

The `insert` method does two writes inside one `with self._connect() as conn:` block:

```python
conn.execute("INSERT INTO works ...", ...)
conn.execute("INSERT INTO works_fts ...", ...)
conn.commit()
```

If the second insert fails (FTS5 corruption, disk full mid-statement), the explicit `conn.commit()` never runs and the connection's context manager rolls back. Good so far. But sqlite3's connection context manager only rolls back on exception — and in WAL mode (`PRAGMA journal_mode = WAL` line 74) there's a subtle gotcha: every `conn.execute(...)` outside an explicit `BEGIN` runs as its own implicit transaction in autocommit semantics for DDL. For DML it's deferred-transactional, so the rollback path *should* hold — but the code never opens an explicit `BEGIN IMMEDIATE` and never tests the FTS-failure branch. Combined with `save_work`'s reorder (store-first), if FTS5 insert silently succeeds but produces a malformed segment, the works row exists with no searchable index, and `search_works` will never return it.

**Severity rationale.** Lower than I-1 in the original audit (the markdown-orphan bug) but in the same family — partial commit between two related rows means the index can lag the canonical row indefinitely. Not data loss, but a queryability hole that's hard to notice.

**Suggested fix.** Wrap the two inserts in an explicit `with conn:` block (`conn` itself is a context manager that commits on exit and rolls back on exception) AND add an explicit `BEGIN IMMEDIATE` to lock writers out during the pair. Add a test that monkeypatches the second `execute` to raise and asserts the works row is also absent.

---

### I-5: `record_climax_event` swallows `Exception` then continues to behavioral_log — partial-failure paths leak inconsistent state

**File:** `brain/body/events.py`
**Line(s):** 75 (`except Exception: # noqa: BLE001`), 84–98

If `Memory.create_new` or `store.create` raises (e.g., disk full, sqlite locked), the function logs at WARN, returns `None`, and skips behavioral_log. Fine. But if `journal.id` is set (memory created) and then `store.create` raises *after* the in-memory `Memory` was constructed but *during* commit, the bare `except Exception` masks the actual failure type — the operator can't tell the disk filled up vs the schema is broken. Worse, the variable `journal` is referenced on line 88 (`name=journal.id`) but assigned inside the try block — if creation raised before the assignment, `journal` is unbound on the path that catches OSError on the behavioral log write at 94. The narrow tuple `(OSError, ValueError)` on 94 saves us only because nothing else uses `journal` after the bare-Exception catch.

**Severity rationale.** Latent bug in a privacy-sensitive path (climax journal entry — should never silently lose). The bare `except Exception` is the same anti-pattern the I-3 fix-pack stamped out in the migrator.

**Suggested fix.** Replace bare `except Exception` with a narrower tuple `(OSError, sqlite3.Error, ValueError)` — same standard the migrator now uses. If something else raises (KeyError, TypeError), let it crash; that's a programming bug, not an I/O event. Also early-return after the catch so the behavioral_log block can't run with `journal` in an undefined state.

---

### I-6: `chat/engine.py:_persist_turn` only catches `(OSError, ValueError)` — but `ingest_turn` could raise broader (path traversal validation, etc.)

**File:** `brain/chat/engine.py`
**Line(s):** 217–223

```python
except (OSError, ValueError) as exc:
    logger.warning(...)
    return False, str(exc)
```

If I-2 above is fixed (path-traversal validation in `_session_path`), `ingest_turn` will start raising `ValueError` for invalid session_ids — already handled, good. But if the validation is implemented as a `re.error` or `TypeError` on non-string input, this block would let the exception propagate up through `respond()` and into the bridge `/chat` handler, returning 502 to the user despite the docstring promising "Errors are logged, not raised — the chat response is delivered."

**Severity rationale.** The stated contract is "persistence errors never break the chat response." A narrower catch can violate that contract whenever the underlying buffer adds new failure modes.

**Suggested fix.** Either widen to `(OSError, ValueError, TypeError)` OR — preferred — catch `Exception` *with* `logger.exception(...)` so the warning message includes the full traceback for diagnosis. The only loss is masking programming bugs in `ingest_turn` itself, and the contract here explicitly chooses chat-response durability over surfacing them.

---

### I-7: Migrator `migration-report.md` and `source-manifest.json` are written non-atomically at the end of `run_migrate`

**File:** `brain/migrator/cli.py` (lines 329, 331), `brain/migrator/report.py:write_source_manifest` (line 189), `brain/migrator/og_legacy.py:migrate_legacy_files` (line 76), `brain/migrator/og_reflex_log.py:migrate_reflex_log` (line 73), `brain/migrator/og_soul_candidates.py:migrate_soul_candidates` (line 80)

Five places in the migrator write directly via `path.write_text(...)` or `path.write_bytes(...)` with no `.new` + `os.replace` rotation:

- `cli.py:329` — `write_source_manifest(work_dir / "source-manifest.json", manifest)` — calls `report.py:189` which does `path.write_text(...)`.
- `cli.py:331` — `(work_dir / "migration-report.md").write_text(...)`.
- `og_legacy.py:76` — verbatim file copy via `dest.write_bytes(src.read_bytes())`.
- `og_reflex_log.py:73` — `dest.write_text(json.dumps(...))`.
- `og_soul_candidates.py:80` — `with dest.open("w") as f: ...`.

Compare with `og_reflex.py:156-161` which does `tmp_path.write_text + os.replace` correctly. The mid-run paths (`og_reflex.py`, `og_journal_dna.py:save_creative_dna` via `save_with_backup`, the `vocab_target` write at `cli.py:131-136`, `interests_target` at `cli.py:189-194`) are atomic; the artifacts written at the *end* of the run are not.

**Severity rationale.** A migration is the canonical "long-running, hard-to-rerun" job. If the user kills the process between `print(report_text)` and the report write completing, they get a half-written `migration-report.md` with no way to know the migration actually succeeded — the SQLite DBs were all committed before the report write. The same applies to legacy files (preservation is the whole point of the legacy directory; a torn copy of `nell_journal.json` is worse than no copy).

**Suggested fix.** Pull out a small helper (per the M-1 task already scheduled) that does `tmp = path.with_suffix(suffix + ".new"); tmp.write_text(...); os.replace(tmp, path)`, and route all five sites through it. The report + manifest writes are tiny so the cost is rounding error.

---

### I-8: `brain/chat/session.py:_SESSIONS` is shared mutable state with no lock — supervisor thread + asyncio handlers race

**File:** `brain/chat/session.py`
**Line(s):** 73 (`_SESSIONS: dict[str, SessionState] = {}`), and all callers

`create_session`, `get_session`, `remove_session`, `all_sessions`, `reset_registry` all touch the module-level dict without any lock. The bridge supervisor thread runs `close_stale_sessions` (which itself calls `list_active_sessions` from the buffer dir — different state — but the in-memory `_SESSIONS` dict is touched from both the supervisor thread (via the heartbeat shutdown path at `bridge/server.py:435-442`) and the asyncio event loop's HTTP handlers (every `/session/new`, `/chat`, `/sessions/close`).

CPython's GIL makes individual dict ops atomic, but the *compound* operations are not: e.g., `all_sessions()` returns `list(_SESSIONS.values())` — fine in isolation, but `state_endpoint` then does `if session_id in s.in_flight_locks` against a separate dict; between those two reads a `remove_session` from another thread can leave `state_endpoint` returning a stale view.

**Severity rationale.** Latent race; no reproduction yet. The blast radius is "occasional flake in concurrent bridge usage" rather than "data loss." Worth fixing before it bites someone in the nascent multi-client scenarios (Tauri + CLI on the same bridge).

**Suggested fix.** Wrap the dict in a small `_SessionRegistry` class with a `threading.Lock`. Or use `threading.RLock()`. Audit the call sites for compound ops (`get + check + remove`) and serialize them under the same lock.

---

### I-9: `soul_candidates.jsonl` append in `queue_soul_candidate` is non-atomic and not flushed/fsynced

**File:** `brain/ingest/soul_queue.py`
**Line(s):** 65–72

```python
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
```

If the process is killed mid-write (SIGKILL or OOM), the JSONL file may end with a partial line. The reader (`brain/health/jsonl_reader.py`) tolerates corrupt lines — good — but the partial line still occupies bytes that the next append will *concatenate to*, potentially merging two records into one corrupted line that *also* parses (e.g., the partial JSON closes against the next line's open). A `flush()` after the write reduces the window to OS-buffer-flush latency; an `os.fsync(fh.fileno())` makes it durable. Same pattern in `brain/soul/audit.py:60-63` (audit log appends — and the audit log is the safety rail, *should* be the most durable thing on disk).

**Severity rationale.** Soul candidates and the soul audit log are both load-bearing for the autonomous-soul-decision contract (spec §11). A torn append → "we lost the audit entry that explained this decision" → the safety rail has a hole.

**Suggested fix.** After `fh.write(...)`, call `fh.flush()` and `os.fsync(fh.fileno())` for the soul_audit and soul_candidates paths specifically. Buffer-write paths like behavioral_log and active_conversations buffer can keep current behavior — they're recoverable from chat history.

---

### I-10: `migrate_legacy_files` does verbatim byte copy with no integrity check — a torn OG file is silently preserved-as-torn

**File:** `brain/migrator/og_legacy.py`
**Line(s):** 70–77

The function copies bytes via `dest.write_bytes(src.read_bytes())`. If the source file is mid-write (the `check_preflight` lock check only catches `memories_v2.json.lock`, not the 16 legacy files), or if the read returns short due to FS issues, the destination silently has truncated content. There's no SHA validation, no size assertion, and no rollback if the read fails partway. The docstring explicitly says "broken files preserve their broken bytes for future migrator-authors" — fair as a design choice — but there's no diagnostic surfacing *which* files are short or torn versus intact.

**Severity rationale.** Migration is a one-shot operation users run during persona transitions. Silent truncation of `nell_journal.json` (a years-deep biographical record) with no indicator is the kind of bug Hana cares about specifically (per the original audit's I-4 framing).

**Suggested fix.** Compute SHA-256 + size of source AND destination after the copy; if they differ, raise (preferred) or record in the migration report's source manifest as a `legacy_file_integrity` warning. Add the legacy files to the `_verify_sources_unchanged` post-run check so OG-mid-write detection covers them too.

---

## Minor issues

### M-1: `body/words.py:count_words_in_session` will always return 0 — no memory of `memory_type="conversation"` is ever written

**File:** `brain/body/words.py`
**Line(s):** 42

`store.list_by_type("conversation", active_only=True)` is queried — but `brain/ingest/commit.py:52` writes memories with `memory_type=item.label` (one of `observation/feeling/decision/question/fact/note`), and `add_memory.py`/`add_journal.py` use other types too. Grep across `brain/` confirms no path produces `memory_type="conversation"`. The function silently returns 0, so the energy compute (`brain/body/state.py:_compute_energy`) gets 0 word-drain for every session — body state misreads as "fully energetic" even after a 3-hour writing session.

**Suggested fix.** Either query session-buffer JSONL files directly (the actual conversation source-of-truth) and sum `len(text.split())` for assistant turns within the window; or filter `list_active` by `metadata.get("speaker") == "assistant"` and tags for conversation-class memories. Matches the OG behavior the spec describes.

### M-2: `migrator/og_journal_dna.py:_migrate_tendencies` doesn't handle the case where `og_tendencies` is neither list nor dict

**File:** `brain/migrator/og_journal_dna.py`
**Line(s):** 61–105

`if isinstance(og_tendencies, list)` falls through to `og_tendencies.get("active", [])` for any non-list input — but if it's an int, str, None, or anything else without `.get`, the migrator crashes with AttributeError. Caller's `try/except (OSError, json.JSONDecodeError, ValueError)` does NOT catch AttributeError, so a malformed OG file kills the migration mid-run.

**Suggested fix.** Add an explicit `elif isinstance(og_tendencies, dict)` branch and a final `else: return {"active": [], "emerging": [], "fading": []}`.

### M-3: `brain/works/store.py:WorksStore` opens a fresh connection every call — `WorksStore(db_path)` inside `search_works` is recreated per HTTP request

**File:** `brain/tools/impls/search_works.py:21`, `brain/tools/impls/list_works.py`, `brain/tools/impls/read_work.py:18`

Each tool call constructs `WorksStore(db_path)` which runs `_connect`, executes `_SCHEMA_SQL` (CREATE IF NOT EXISTS for tables + indexes + virtual table + PRAGMA), and commits. That's 3-4 SQLite operations per call before the actual query runs. Not catastrophic, but on a hot search path the schema-init noise is wasteful and turns every read into a write transaction.

**Suggested fix.** Move the `_SCHEMA_SQL` execution into a one-time `init` or guard with `_initialised: ClassVar[set[Path]]`. Or open the DB read-only when the file already exists (skip schema init).

### M-4: `tools/dispatch.py:dispatch` mutates the caller's `arguments` dict on `get_body_state` (line 119)

**File:** `brain/tools/dispatch.py`
**Line(s):** 117–124

```python
if name == "get_body_state" and "session_hours" in arguments:
    try:
        arguments["session_hours"] = float(arguments["session_hours"])
```

In-place mutation of the LLM's tool_call arguments. The chat tool loop then logs `arguments` into the invocations record (`brain/chat/tool_loop.py:103-105`). If the original was the literal string `"1.5"` from the LLM's JSON, the log now says `1.5` (float) — small audit-trail drift, and a footgun if a future caller re-dispatches the same `arguments` dict expecting unchanged input.

**Suggested fix.** Make a shallow copy: `arguments = {**arguments, "session_hours": float(...)}`. Don't mutate input dicts.

### M-5: `brain/migrator/cli.py:_ensure_clobber_safe` capitalises `kind` in the error message but `kind` is passed lowercase from one call site only

**File:** `brain/migrator/cli.py`
**Line(s):** 357–366

Only one call site exists: `cli.py:55` passes `kind="output directory"`. The `kind.capitalize()` then renders `"Output directory"` — fine for that one site. But the function is non-private (no `_` prefix isn't quite right — it has one, but it's exported via the module), so future callers passing already-capitalised strings would get `"Output Directory"` from `.capitalize()` (capitalize lower-cases everything past the first char). Brittle.

**Suggested fix.** Either drop the `.capitalize()` and document that `kind` should be lowercase, or use `kind[0].upper() + kind[1:]` to capitalise just the first letter. Or take a `kind_label: str` plus an `error_subject: str` parameter.

### M-6: `brain/bridge/runner.py:_write_clean_shutdown` swallows `Exception` with `pass` — masks failures during shutdown diagnostics

**File:** `brain/bridge/runner.py`
**Line(s):** 67–68

```python
except Exception:
    pass  # best-effort only; don't re-raise at exit time
```

If `state_file.write` ever starts raising for a permission, schema, or path reason, every dirty shutdown becomes a silent dirty shutdown — `recovery_needed()` will fire on next start but the operator has no log indicating *why*. A `logger.exception(...)` cost is zero and gives a forensic trail.

**Suggested fix.** Replace `pass` with `logger.warning("clean-shutdown write failed", exc_info=True)`. Doesn't change exit-time behaviour (still no re-raise), just leaves a breadcrumb.

### M-7: `brain/bridge/server.py:health` catches `Exception` from `walk_persona`/`compute_pending_alarms` (line 529) — bare `except Exception` here matches the I-3 anti-pattern

**File:** `brain/bridge/server.py`
**Line(s):** 526–532

```python
try:
    anomalies = walk_persona(s.persona_dir)
    alarms = compute_pending_alarms(s.persona_dir)
except Exception:
    logger.warning("health walk failed", exc_info=True)
    anomalies = []
    alarms = []
```

Same pattern the original audit's I-3 swept out of the migrator: bare `except Exception` masking programming bugs. The `/health` endpoint should return 200 even on degraded state, but if `walk_persona` raises a `KeyError` due to a code bug, swallowing it means health stays green forever and the operator has no signal.

**Suggested fix.** Narrow to `(OSError, ValueError, sqlite3.Error)` or whatever the legitimate runtime failures are. Let real bugs surface as 500.

### M-8: `brain/migrator/og.py:_LIVE_LOCK_THRESHOLD_SECONDS = 5 * 60` — five minutes is generous; an OG bridge that ticks every 60s could appear dead

**File:** `brain/migrator/og.py`
**Line(s):** 15

Five minutes is the threshold for "the OG bridge looks active." OG NellBrain's supervisor ticks every 60 seconds (per its README), so a bridge running healthily that paused mid-tick by 5+ minutes (e.g., a long heartbeat) would *not* trip the preflight. The check is correct in intent but possibly under-aggressive given OG's actual cadence.

**Suggested fix.** Reduce to 90 seconds (1.5x tick interval) or document the rationale for 5 minutes. Add a `--force-preflight` flag for users who know what they're doing.

### M-9: `brain/works/storage.py:_atomic_write_text` doesn't fsync the parent dir after `os.replace`

**File:** `brain/works/storage.py`
**Line(s):** 50–77

The write-to-`.new` + `os.replace` pattern is correct for atomic file content, but the parent directory entry isn't fsynced. On a power loss between the rename and the dir-block flush, the old name *or* new name may be visible — POSIX guarantees the rename is atomic but not that it's durable until the dir is fsynced. Same pattern in `brain.health.attempt_heal.save_with_backup` (referenced at `bridge/state_file.py:24`) which presumably has the same omission.

**Suggested fix.** After `os.replace(new_path, path)`, do `fd = os.open(path.parent, os.O_DIRECTORY); os.fsync(fd); os.close(fd)` (POSIX only — guard with `if os.name == "posix"`). The cost is one syscall per save; the benefit is durability.

### M-10: `brain/tools/dispatch.py:dispatch` raises `ToolDispatchError` on unknown tool — but the tool loop catches `Exception` and surfaces `str(exc)` to the LLM, leaking the "Known tools: …" list

**File:** `brain/tools/dispatch.py:96-98`, `brain/chat/tool_loop.py:117-120`

When the LLM calls a tool that doesn't exist, `dispatch` raises `ToolDispatchError(f"unknown tool: {name!r}. Known tools: {known}")`, the tool loop catches it, and the next iteration's tool message contains the full list of canonical tool names (currently 13). That's training signal we hand the LLM for free. Probably benign — most LLMs don't latch onto error-message contents — but the helpful-dev habit of listing-known-options leaks the tool surface to anyone reading invocation logs / metadata.

**Suggested fix.** Drop the "Known tools: …" tail from the exception message, or at least don't include it in the JSON returned to the LLM (`tool_loop.py:121`).

---

## Summary of recommended next-bundle scope

**Ship in next bundle (high-impact, narrow change):**
- I-1 (cmd_tail token leak + broken auth) — privacy + correctness, single file
- I-2 (session_id path traversal validation) — security, one regex check
- I-3 (Works store list_recent/get OperationalError catch) — symmetry with already-shipped I-2 fix
- I-9 (fsync on soul_audit + soul_candidates appends) — safety-rail durability
- M-1 (body/words.py "conversation" type fix) — silent zero-bug worth a real fix
- M-6 (runner.py log-instead-of-pass) — diagnostics

**Defer or evaluate further:**
- I-4 (works.db transaction wrapping) — needs a reproduction case before committing to the BEGIN IMMEDIATE refactor
- I-5 (record_climax_event narrow exception tuple) — straightforward but not blocking
- I-6 (engine._persist_turn catch widening) — depends on I-2 fix shape
- I-7 (migrator end-of-run atomic writes) — pairs naturally with the M-1 (audit-original) `save_with_backup_text` extraction; ship together
- I-8 (session registry lock) — latent race, no repro yet; nice-to-have
- I-10 (legacy file integrity) — design discussion; current "preserve broken bytes" is intentional
- M-2 through M-5, M-7, M-8, M-10 — polish; bundle when convenient

**Findings count:** 10 Important + 10 Minor.
