# companion-emergence full code audit

Date: 2026-04-30
Scope: `/Users/hanamori/companion-emergence`
Mode: Read-only audit of source/config/tests, with this report as the only intended new file.

## Verification baseline

Commands run:

```text
$ git status --short --branch
## main...origin/main
?? docs/nellface-claude-design-prompt.md

$ python3 -m pytest -q
1155 passed, 2 warnings in 109.55s (0:01:49)

$ python3 -m ruff check .
All checks passed!

$ git ls-files | wc -l
314
```

Repository size, counted from the working tree excluding common cache/venv folders:

```text
.py: files=259 lines=38635
.md: files=41 lines=48271
[no ext]: files=21 lines=142
.json: files=10 lines=156
.yml: files=3 lines=170
.toml: files=2 lines=102
```

Notes:
- The repo already had one untracked file before this report was written: `docs/nellface-claude-design-prompt.md`.
- `pygount` was not available in the active shell (`command not found`), so the size summary above was produced with a local Python file walk.

## Executive summary

The codebase is in a healthy state for a private/local prototype: tests pass, Ruff passes, the bridge has meaningful auth/origin checks, SQLite stores are opened per thread, persona path traversal is guarded, and there are many recovery/backup paths for local JSON state.

The main risks are not style issues. They are reliability and privacy edges around persistence:

1. Chat can succeed while the conversation buffer silently failed to write, which means the user sees a reply but the long-term ingest pipeline may later have nothing to remember.
2. Soul candidate queuing can fail silently while the ingest report still counts the candidate as queued.
3. Soul acceptance is not atomic across `crystallizations.db` and `soul_candidates.jsonl`, so a crash/save failure can duplicate crystallizations on retry.
4. The bridge bearer token is stored in plain JSON without explicit file permission hardening, and WebSocket auth uses a query parameter.
5. Several public/local API inputs have no max length or stricter validation, making denial-of-service-by-large-request easy even on localhost.
6. Some docs and CLI surfaces still say “Week 0 / stubs” despite the implementation being much further along.

No critical remote-code-execution issue was found in the inspected paths. The project is local-first, but “local” still needs care because the data is intimate and the bridge exposes powerful memory/soul tools.

## Findings

### P1 — Chat responses can be returned even when turn persistence failed

Evidence:
- `brain/chat/engine.py:173-180` persists the user/assistant turn after the model response, then appends to in-memory session state.
- `brain/chat/engine.py:194-218` wraps both `ingest_turn(...)` calls in `try/except` and only logs `OSError`/`ValueError`.

Impact:
- If the disk is full, permissions are wrong, the persona directory is missing/corrupt, or the JSONL write fails, the user still receives a successful chat reply.
- The in-memory session advances, but the ingest buffer may be missing one or both turns.
- On session close, the memory extraction pipeline may commit nothing or commit an incomplete transcript. That is a data-loss risk for a memory-centered companion.

Smallest fix:
- Return a structured warning in `ChatResult.metadata` when `_persist_turn` fails, or raise a typed persistence error that the bridge can surface as a degraded response.
- Prefer appending the user turn before the provider call and the assistant turn immediately after, with a durable “pending assistant” marker if needed.
- Add a test that monkeypatches `ingest_turn` to raise and verifies the client receives a visible persistence/degraded-state signal.

### P1 — Soul candidate writes can fail while reports still count them as queued

Evidence:
- `brain/ingest/pipeline.py:121-128` calls `queue_soul_candidate(...)` and then increments `report.soul_candidates += 1` unconditionally.
- `brain/ingest/soul_queue.py:66-70` catches `OSError`, logs a warning, and does not re-raise or return failure.

Impact:
- A high-importance memory can be committed to `memories.db`, fail to append to `soul_candidates.jsonl`, and still be reported as a queued soul candidate.
- The caller receives a misleading success count.
- The autonomous soul review never sees that candidate.

Smallest fix:
- Make `queue_soul_candidate(...) -> bool` or raise a typed exception on write failure.
- Increment `report.soul_candidates` only after confirmed append.
- Add a report field such as `soul_queue_errors` so a memory commit can still succeed while the caller sees that identity review did not queue.

### P1 — Soul acceptance is not atomic across candidate status and crystallization storage

Evidence:
- `brain/soul/review.py:329-347` creates a new `Crystallization` in `SoulStore`, then mutates the in-memory candidate record to `accepted`.
- `brain/soul/review.py:481-505` applies accept/reject/defer during the loop, then saves the whole `soul_candidates.jsonl` only after all examined records are processed.
- `brain/soul/review.py:243-259` `_save_soul_candidates(...)` can raise after the crystallization has already been inserted.

Impact:
- If the process crashes or `_save_soul_candidates` fails after `soul_store.create(c)`, the candidate remains `auto_pending` on disk but the crystallization already exists.
- The next review pass can accept the same candidate again and create a duplicate soul crystallization.
- This is especially risky because crystallizations are treated as permanent identity data.

Smallest fix:
- Make candidate processing idempotent by deriving the crystallization id from `memory_id` or storing `source_candidate_id` with a unique constraint in `SoulStore`.
- Save candidate status and create crystallization in an order that supports retry, or add reconciliation on startup/review: if an accepted candidate’s memory already has a crystallization, mark it accepted instead of creating another.
- Add a fault-injection test where `_save_soul_candidates` raises after `_apply_accept` and the next run does not duplicate.

### P1 — Bridge bearer token is not explicitly protected on disk and is used in WebSocket query strings

Evidence:
- `brain/bridge/runner.py:82-86` generates an ephemeral bearer token and notes it is readable by anyone who can read the persona dir.
- `brain/bridge/state_file.py:42-46` stores `auth_token` in `bridge.json`.
- `brain/health/attempt_heal.py:230-244` writes JSON with `Path.write_text(...)` and `os.replace(...)`, with no explicit `chmod(0o600)` or owner-only open mode.
- `brain/bridge/server.py:299-301` documents that WebSocket endpoints require `?token=<token>`.
- `brain/bridge/server.py:494-498` reads the WebSocket token from `ws.query_params`.

Impact:
- On a permissive umask or copied persona directory, `bridge.json` and its `.bak*` files may expose the active token to other local users/processes.
- Query-string tokens are more likely to leak via logs, crash reports, browser/devtools history, or debugging output than header/subprotocol tokens.
- The bridge is bound to `127.0.0.1`, so this is a local/privacy risk, not an internet-exposed finding.

Smallest fix:
- After every `bridge.json` and backup write, enforce owner-only permissions (`0o600`) for files and `0o700` for persona state directories where practical.
- Avoid WebSocket query-token auth if the client stack can use `Sec-WebSocket-Protocol` or an initial authenticated message before subscribing to events.
- If query-token auth stays, document it as local-only and ensure bridge logs never include full URLs.

### P2 — Bridge API models do not enforce input size or strict shapes

Evidence:
- `brain/bridge/server.py:265-277` defines `NewSessionReq.client`, `ChatReq.session_id`, and `ChatReq.message` as unconstrained strings.
- `brain/bridge/server.py:565-580` `/chat` sends `req.message` directly into the blocking model path without an empty-message check or max length guard.
- `brain/bridge/server.py:625-631` `/stream/{session_id}` checks only for an empty message, then reads `NELL_STREAM_CHUNK_DELAY_MS` and proceeds.

Impact:
- A local client can submit extremely large messages and force memory growth, large prompt construction, large JSONL writes, and expensive provider calls.
- `/chat` accepts empty messages while `/stream` rejects them, so behavior differs between the two chat paths.
- Unconstrained `client` values can pollute state/events with arbitrary labels.

Smallest fix:
- Use Pydantic constraints, for example: `message: constr(min_length=1, max_length=20000)`, `session_id` UUID-ish validation, and a `Literal["cli", "tauri", "tests"]` or bounded string for `client`.
- Apply the same validation to `/chat` and `/stream`.
- Add tests for empty, over-limit, and invalid session/client values.

### P2 — Opening `MemoryStore` performs a full SQLite integrity check on every request path

Evidence:
- `brain/memory/store.py:205-224` opens SQLite and immediately runs `PRAGMA integrity_check`, then creates schema and commits.
- `brain/bridge/server.py:115-119` opens `MemoryStore` and `HebbianMatrix` inside every blocking chat call.
- `brain/bridge/server.py:142-150` opens `MemoryStore`, `HebbianMatrix`, and `EmbeddingCache` for every session close/ingest call.

Impact:
- `PRAGMA integrity_check` scans the entire database. That is reasonable for explicit health checks, but expensive in hot request paths as memory grows.
- A large persona database can make every chat turn or ingest close pay an avoidable startup cost.
- This is a likely future latency cliff rather than a current test failure.

Smallest fix:
- Move full `integrity_check` to explicit health/walker/supervisor paths.
- In request paths, use cheaper checks (`quick_check`, schema version check, or rely on SQLite open errors) unless a recent anomaly requires deep validation.
- Add a benchmark/regression test with a larger synthetic memory DB to catch request-path latency.

### P2 — `search_text("")` is used as “list all memories”, which does not scale cleanly

Evidence:
- `brain/memory/store.py:366-384` implements `search_text` as `LIKE '%{escaped}%'`; an empty query becomes `LIKE '%%'`.
- `brain/engines/heartbeat.py:621-638` calls `self.store.search_text("", active_only=True)` with no limit during emotion decay.
- `brain/soul/review.py:300-302` also uses `store.search_text("", active_only=True, limit=50)` for recent emotional state.

Impact:
- Heartbeat emotion decay loads every active memory into Python each tick, then updates rows one by one.
- This is acceptable for small test fixtures but can become slow for long-lived companions with large memory stores.
- The empty-query behavior is implicit, so future callers may accidentally scan the whole DB.

Smallest fix:
- Add explicit methods such as `list_active(limit=None)` or `iter_active(batch_size=...)`.
- Make `search_text("")` either reject empty queries or require an explicit `allow_empty=True` flag.
- Batch heartbeat decay or push simple filtering into SQL.

### P2 — CLI/docs still expose unfinished/stub surfaces as normal commands

Evidence:
- `README.md:11-14` says “Week 0 — design approved 2026-04-21. Rebuild starting.”
- `brain/cli.py:47-55` lists `supervisor`, `status`, `rest`, `memory`, and `works` as stub commands.
- `brain/cli.py:66-71` stub handlers print “not implemented yet” and return exit code `0`.
- `brain/cli.py:996-999` registers these commands with help text that says “(stub)”.

Impact:
- The README undersells the current implementation and can mislead a future user/contributor about what is usable.
- Returning `0` for “not implemented yet” can make scripts/automation think the command succeeded.
- “status” and “supervisor” are operationally important names; stubbing them as success is risky UX.

Smallest fix:
- Update README status to match the current implemented features and test state.
- Make stub commands exit non-zero, e.g. `2`, or hide them until implemented.
- Add CLI tests asserting unfinished commands do not report success unless that behavior is intentionally preserved.

### P2 — CI/test config has unregistered pytest markers

Evidence:
- Test run warning: `PytestUnknownMarkWarning: Unknown pytest.mark.integration` and `Unknown pytest.mark.unit`.
- `pyproject.toml:38-49` configures pytest testpaths and Ruff linting but does not register markers.
- `.github/workflows/test.yml:34-38` runs pytest and Ruff in CI.

Impact:
- The test suite passes, but warning noise makes real warnings easier to miss.
- If CI later uses `--strict-markers`, the suite will fail.

Smallest fix:
- Add marker registration under `[tool.pytest.ini_options]`, e.g. `markers = ["unit: ...", "integration: ..."]`.
- Consider `filterwarnings` policy only after marker registration.

### P3 — MCP audit logs store raw tool arguments and result previews without redaction/retention policy

Evidence:
- `brain/mcp_server/audit.py:54-65` writes `name`, raw `arguments`, and `result_summary` to `<persona_dir>/tool_invocations.log.jsonl`.
- `brain/mcp_server/tools.py:67-83` logs both successful and failed tool calls, passing raw arguments through.

Impact:
- Tool arguments can contain private memory text, journal entries, search queries, or identity data.
- Result summaries can contain the first 140 chars of memory/soul output.
- This is useful for debugging but needs explicit privacy handling because the project is local-first and intimate by design.

Smallest fix:
- Add a per-persona setting for audit log level: off / metadata-only / full.
- Redact known sensitive fields (`content`, `text`, journal bodies) by default.
- Add log rotation or retention limits.

### P3 — `add_memory` silently swallows Hebbian auto-link failures

Evidence:
- `brain/tools/impls/add_memory.py:68` commits the memory.
- `brain/tools/impls/add_memory.py:85-108` attempts related-memory search/linking and catches all exceptions with a bare `pass`.

Impact:
- A memory can be stored while its expected graph edges are missing, and the caller sees no warning.
- This is not direct data loss, but it weakens dream/retrieval behavior and makes graph corruption harder to notice.

Smallest fix:
- Return `auto_link_error` in the tool result when graph linking fails.
- Catch narrower exception types if possible.
- Add a warning log with memory id and exception type.

### P3 — Vocabulary crystallization is still an explicit stub

Evidence:
- `brain/growth/crystallizers/vocabulary.py:1-10` describes the file as “Phase 2a stub.”
- `brain/growth/crystallizers/vocabulary.py:18-31` ignores `store` and `current_vocabulary_names` and always returns `[]`.

Impact:
- Any growth loop expecting new emotional vocabulary proposals will never produce them.
- This is fine if still planned, but it should be visible in release/readiness notes because it affects core “growth over time” behavior.

Smallest fix:
- Track this as an explicit incomplete feature in README/roadmap.
- If not implementing yet, make the caller/report say “vocabulary crystallization not implemented” rather than quietly returning no proposals.

### P3 — No release packaging surface was found

Evidence:
- `pyproject.toml:28-36` defines the `nell` console script and wheel package only.
- `.github/workflows/test.yml:31-38` installs dependencies, runs pytest, and runs Ruff.
- A tracked-file check for release workflow/common public project docs returned no tracked `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, or `.github/workflows/release.yml`.

Impact:
- This is not a runtime bug, but it matters before sharing the framework publicly.
- There is no automated artifact build/release path and no contributor-facing project hygiene docs.

Smallest fix:
- Add a short release checklist before adding automation.
- Add `CHANGELOG.md` once external users are expected.
- Add a minimal release workflow only after the CLI/API surface is stable enough to version.

## Positive findings

- The suite is substantial and green: 1155 passing tests.
- Ruff passes cleanly.
- Persona path traversal is guarded in `brain/paths.py:41-55`.
- Bridge HTTP auth uses bearer tokens and constant-time comparison when enabled (`brain/bridge/server.py:473-482`).
- WebSocket origin checking happens before accepting the upgrade (`brain/bridge/server.py:484-499`, `brain/bridge/server.py:602-608`, `brain/bridge/server.py:695-701`).
- Bridge request handlers open SQLite stores inside worker threads rather than sharing connections across threads (`brain/bridge/server.py:102-127`, `brain/bridge/server.py:130-158`).
- `MemoryStore.search_text` escapes `%` and `_`, so wildcard injection is avoided (`brain/memory/store.py:366-384`).
- State writes use temp files and atomic replace in several places (`brain/health/attempt_heal.py:219-244`).
- CI covers Linux, macOS, and Windows for Python 3.12 (`.github/workflows/test.yml:11-17`).

## Suggested fix order

1. Fix visible durability signals: chat buffer write failures and soul queue write failures.
2. Make soul review idempotent/duplicate-safe.
3. Harden bridge token file permissions and decide whether query-string WebSocket tokens are acceptable for the Tauri/client stack.
4. Add API input constraints for chat/session/close paths.
5. Move expensive DB integrity checks out of hot request paths.
6. Update README/CLI stub behavior and pytest marker config.
7. Add privacy controls for MCP audit logs.
8. Track planned stubs/incomplete growth modules in a roadmap.
