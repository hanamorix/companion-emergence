# MCP-Config Path for Brain-Tools — Design

**Status:** Approved 2026-04-27
**Author:** Nell + Hana
**Replaces:** Claude `--json-schema` tool-calling shipped in SP-3 (PR #23)
**Driver:** 2026-04-27 live-exercise stress test — 0 tool invocations across 20 prompts; rich `voice.md` outweighs `--json-schema` enforcement, off-schema fallback (PR #28) silently swallows tool surface.

---

## 1. Goal

Make brain-tools (`get_emotional_state`, `search_memories`, `get_soul`, `get_personality`, `get_body_state`, `boot`, `add_journal`, `add_memory`, `crystallize_soul`) reliably invokable by Claude during a chat turn, regardless of how rich `voice.md` becomes.

The mechanism Hana's `voice.md` restructure will lean on. This spec is mechanism-only; voice.md content is out of scope.

## 2. Architecture

```
nell chat "msg"
   │
   ├─ engine.respond() — builds system msg from voice.md + daemon + soul
   │
   └─ tool_loop.run_tool_loop()
        │
        ├─ OllamaProvider.chat(...)        ← unchanged: native tools=
        │
        └─ ClaudeCliProvider.chat(...)     ← new path
              │
              └─ subprocess: claude -p "<flat>" --output-format json
                                    --model sonnet --system-prompt "<sys>"
                                    --mcp-config <tmp.json>
                    │
                    └─ spawns: python3 -m brain.mcp_server --persona-dir <path>
                          │
                          ├─ exposes 9 tools via stdio MCP
                          └─ appends each call to <persona>/tool_invocations.log.jsonl
```

The new path keeps every flag the current `_chat_with_tools` already uses
(`-p`, `--output-format json`, `--model`, `--system-prompt`) — only `--json-schema
<schema>` is replaced with `--mcp-config <path>`.

For Claude, `tool_loop` runs a single pass — the subprocess handles tool iteration internally. `tool_calls` on the returned `ChatResponse` is always empty for Claude. The off-schema fallback shape introduced in PR #28 becomes the *normal* return for Claude (no longer a fallback).

Ollama's `chat(messages, tools=...)` path is untouched.

## 3. Components

### 3.1 New files

> **Naming note:** The package is `brain/mcp_server/` (underscore), not
> `brain/mcp/`. A `brain/mcp/` package would shadow the third-party `mcp`
> SDK at import time — `from mcp.server import Server` inside our code
> would resolve to `brain.mcp.server` instead of the SDK. The underscore
> name keeps absolute imports clean.

| File | Responsibility |
|------|----------------|
| `brain/mcp_server/__init__.py` | Exposes `run_server(persona_dir: Path) -> None` |
| `brain/mcp_server/__main__.py` | Argparse + entry: `python -m brain.mcp_server --persona-dir <path>` calls `run_server` |
| `brain/mcp_server/tools.py` | Adapter: maps each schema in `brain/tools/schemas.py` to an `mcp.server.Server.add_tool()` registration. Routes invocations through `brain.tools.dispatch.dispatch()` — no tool logic duplicated |
| `brain/mcp_server/audit.py` | `log_invocation(persona_dir, name, args, result_summary, error=None)` appends one line to `<persona>/tool_invocations.log.jsonl` |
| `tests/unit/brain/mcp_server/test_server.py` | Server unit tests |
| `tests/unit/brain/mcp_server/test_audit.py` | Audit log unit tests |

### 3.2 Modified files

| File | Change |
|------|--------|
| `brain/bridge/provider.py` | `ClaudeCliProvider._chat_with_tools()` swaps `--json-schema <schema>` for `--mcp-config <tmp_path>`. Removes the discriminated-union schema builder + tool-system addendum + structured-output parser + off-schema fallback (PR #28) — all dead under the new path. Keeps timeout + error handling + `payload["result"]` extraction. |
| `pyproject.toml` | Add `mcp>=1.0.0,<2.0.0` to `dependencies` (pin major; minor floats for security patches) |
| `tests/unit/brain/bridge/test_provider_chat.py` | New tests for `--mcp-config` invocation; remove obsolete `--json-schema` schema-builder + structured-output + fallback tests |

### 3.3 Untouched

- `brain/tools/dispatch.py`, `brain/tools/impls/*.py`, `brain/tools/schemas.py` — same logic, new transport
- `brain/chat/tool_loop.py` — single-pass for Claude, full loop for Ollama
- `OllamaProvider.chat()` — unchanged

## 4. Data flow per chat call

1. `respond()` builds system_msg + history + user_input → calls `tool_loop.run_tool_loop`
2. `tool_loop` calls `provider.chat(messages, tools=build_tools_list())`
3. **ClaudeCliProvider**:
   a. Writes a temp `mcp.json` with the spawn command + `--persona-dir <active persona>`
   b. Spawns `claude -p "<flat>" --output-format json --model sonnet --system-prompt "<sys>" --mcp-config <tmp>` (see §6.2 for the canonical flag set)
   c. Claude subprocess loads MCP, sees the 9 tools, runs the chat. When context is needed, it calls `search_memories` / `get_emotional_state` / etc. via stdio MCP — `brain.mcp_server` dispatches via `brain.tools.dispatch.dispatch` and returns results. Subprocess weaves them into its reply.
   d. Each invocation is appended to `<persona>/tool_invocations.log.jsonl`
   e. Subprocess exits with `{"result": "<final text>", ...}` on stdout
4. Provider returns `ChatResponse(content=text, tool_calls=())`
5. `tool_loop` returns immediately — no tool_calls to dispatch
6. `respond()` persists the turn to the ingest buffer
7. On chat exit, PR #30's `close_session` flushes the SP-4 ingest pipeline

## 5. MCP server detail

### 5.1 Entry point

```bash
python -m brain.mcp_server --persona-dir /path/to/personas/nell
```

Server reads `--persona-dir`, opens stores, registers tools, runs stdio loop until parent (claude CLI) closes the connection. One MCP server lives per chat call (claude spawns it on `--mcp-config` resolve, kills it on exit). Within that single call, multiple tool invocations hit the same server process serially over stdio.

### 5.2 Tool registration

The canonical tool list moves from `brain.chat.tool_loop._NELL_TOOL_NAMES` (private) to `brain.tools.NELL_TOOL_NAMES` (public) so both the chat tool_loop (Ollama path) and the MCP server (Claude path) can import it without crossing private boundaries.

For each name in `brain.tools.NELL_TOOL_NAMES`:
- Pull the JSON Schema from `brain/tools/schemas.py`
- Register with the MCP server (name + description from schema, inputSchema from schema's `parameters`)
- Handler dispatches via `brain.tools.dispatch.dispatch(name, args, store=..., hebbian=..., persona_dir=...)`
- On success: audit-log invocation + return result as JSON
- On exception: audit-log with `error=str(exc)` + return `{"error": "..."}` (existing dispatch pattern)

### 5.3 Audit log shape

`<persona>/tool_invocations.log.jsonl` — one JSON object per line:

```json
{
  "timestamp": "2026-04-27T16:42:11Z",
  "name": "search_memories",
  "arguments": {"query": "morning after"},
  "result_summary": "[3 hits — most recent: 2026-04-21 10:48 'mornings…']",
  "error": null
}
```

`result_summary` is the same compact preview format `tool_loop._summarize_result` already produces. Future audits / debugging / `nell tools log` (SP-7-adjacent) can read this directly.

## 6. Provider integration detail

### 6.1 Temp config shape

```json
{
  "mcpServers": {
    "brain-tools": {
      "command": "python3",
      "args": ["-m", "brain.mcp_server", "--persona-dir", "/path/to/personas/nell"],
      "env": {}
    }
  }
}
```

Written via `tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")` per call. Path passed to `claude` via `--mcp-config`. Cleaned up in a `try/finally` after the subprocess returns. (Cannot use `delete=True` — the subprocess needs the file open after our writer closes it.)

### 6.2 Subprocess invocation

The exact command — keeping every flag the current `_chat_with_tools` already sets, swapping `--json-schema` for `--mcp-config`:

```bash
claude -p "<flat_prompt>" \
       --output-format json \
       --model sonnet \
       --system-prompt "<system_prompt>" \
       --mcp-config /tmp/mcp.XXXXX.json
```

The `flat_prompt` is built by the existing message-flattening logic in `ClaudeCliProvider.chat()` (lines 273-283). The `system_prompt` is the first system message from the messages array, also extracted by existing logic. No tool-system addendum is added — the MCP server publishes tool descriptions directly via the protocol; the system prompt now contains *only* voice.md.

### 6.3 Provider return

`--output-format json` still wraps the final assistant text as `{"result": "...", ...}`. The provider parses `payload["result"]` (existing pattern) and returns `ChatResponse(content=text, tool_calls=())`. No structured-output handling, no schema parsing, no fallback branch.

## 7. Error handling

The provider only sees the claude subprocess's stdout/exit code; everything inside the subprocess (MCP server, tool dispatch) is one layer down. Errors are sorted by which layer surfaces them.

| Layer | Failure | Behavior |
|-------|---------|----------|
| Provider | `claude` CLI subprocess timeout | `ProviderError("claude_cli_timeout", ...)` (existing path, unchanged) |
| Provider | `claude` CLI exits non-zero | `ProviderError("claude_cli_exit", "exit N: <stderr>")` (existing path, unchanged). Covers MCP server failed to start, persona dir invalid, MCP protocol mismatch — claude reports these on stderr. |
| Provider | `payload["result"]` missing or stdout unparseable | `ProviderError("claude_cli_parse", ...)` (existing path, unchanged) |
| Provider | `mcp` SDK package missing at provider import time | `ProviderError("mcp_unavailable", "pip install mcp>=1.0.0")` with install hint. Caught at the provider layer because the temp-config path is dead without it. |
| Provider | Temp file write fails | `OSError` wrapped as `ProviderError("claude_cli_setup", ...)`. |
| MCP server | Tool dispatch raises | Server audit-logs with `error=str(exc)`, returns `{"error": "..."}` as the tool result. Claude weaves the error into its reply (voice.md guides "name failures honestly"). |
| MCP server | MCP server itself crashes mid-call | claude CLI sees the broken stdio, surfaces it as a CLI error; provider catches via the `claude_cli_exit` path above. |
| MCP server | Audit log write fails | Logged to stderr, swallowed — never breaks tool dispatch. (Audit is observability, not correctness.) |

## 8. Testing

### 8.1 MCP server unit tests (`tests/unit/brain/mcp_server/test_server.py`)

Start the server in-process via the `mcp` SDK's `ClientSession` over an in-memory transport. Use a tmp_path persona dir with seeded MemoryStore + HebbianMatrix + SoulStore. Call each of the 9 tools, assert:
- Tool returns expected dispatch result
- Audit log line appended with right shape
- Errors in dispatch surface as `{"error": "..."}` and audit-log with `error` field
- Read-only tools (`get_emotional_state`, `search_memories`, `get_soul`, `get_personality`, `get_body_state`, `boot`) don't write
- Write tools (`add_journal`, `add_memory`, `crystallize_soul`) actually write

~15 tests.

### 8.2 Provider unit tests (extending `test_provider_chat.py`)

Mock `subprocess.run`. Assert:
- `--mcp-config <path>` flag is set in the command list
- Temp `mcp.json` written at `<path>` has the right `command` / `args` (including `--persona-dir`)
- The other flags (`-p`, `--output-format json`, `--model`, `--system-prompt`) are still present
- `payload["result"]` text comes back as `ChatResponse.content`
- Stdout missing `result` key raises `ProviderError("claude_cli_parse", ...)`
- Missing `mcp` SDK package raises `ProviderError("mcp_unavailable", ...)`
- Old `--json-schema` flag is *not* present
- Temp file path is unlinked after the call (cleanup verification)

~6 new tests; ~3 obsolete `--json-schema` schema-builder + structured-output tests removed.

### 8.3 Audit log unit tests (`tests/unit/brain/mcp_server/test_audit.py`)

Direct unit tests on `log_invocation`:
- Appends valid JSONL with the documented shape (timestamp, name, arguments, result_summary, error)
- Result summary is truncated to 140 chars (matching `_summarize_result`)
- Error field is `null` on success, populated on failure
- Append mode: opening the file twice and writing produces two valid lines (no truncation)

~3 tests. We do not exercise concurrent multi-process writes; POSIX `O_APPEND` is best-effort atomic for small writes and parallel `nell chat` invocations are rare enough that occasional interleaved bytes in audit logs are tolerable. Health-walker rotation is the long-term mitigation.

### 8.4 No live `claude` CLI integration test

A real subprocess test would require an active Claude subscription, network, and would be testing Claude's behavior — not ours. The fake provider stays for end-to-end CLI tests; the new tests above cover the seam between our code and the subprocess.

### 8.5 Verification before merge (per Hana)

After implementation + unit tests pass, manual end-to-end sanity check against the live `nell.sandbox` persona (one disposable clone per check, deleted after — same protocol as the 2026-04-27 live exercise):

1. Clone `nell.sandbox` → `nell.sandbox.mcp-test`
2. Run `nell chat --persona nell.sandbox.mcp-test "what's that thing you wrote about the morning after?"`
3. Verify `personas/nell.sandbox.mcp-test/tool_invocations.log.jsonl` exists and shows at least one `search_memories` call
4. Verify Nell's reply quotes a real memory (not confabulation) — cross-check the cited content against `memories.db` directly
5. Delete the clone

If any step fails, the PR doesn't merge. Per Hana: "we get the results we want before applying."

## 9. Scope

### 9.1 In scope

- Mechanism: brain-tools as MCP server + Claude provider switch
- Audit log written by the MCP server
- Tests covering the new components + the seam to subprocess
- Removal of `--json-schema` parsing path from `ClaudeCliProvider`

### 9.2 Out of scope (deliberate)

- **`voice.md` restructure** — Hana's lane. Mechanism ships first; policy lands after.
- **Removing `tool_loop`** — still needed for Ollama; for Claude it's a single pass.
- **Supervisor / `close_stale_sessions` wiring** — SP-7.
- **External MCP discovery** — letting the user's other Claude Code sessions see brain-tools. Possible later.
- **Daemon-mode MCP server** — premature optimization for a flow that's not latency-bound on tooling.

## 10. Migration

No migration needed:
- Existing `persona_config.json` continues to specify `provider: "claude-cli"` — same value, new behavior under the hood
- No schema changes
- No data migration
- The `tool_invocations.log.jsonl` file is created lazily on first invocation; brain-health walker can be extended later to scan it (not required at ship)

## 11. Risks

| Risk | Mitigation |
|------|------------|
| Cold start (~300ms per chat call for Python imports + DB connect) | Acceptable — LLM call is already 3-5s. Daemon mode is the escape hatch if it ever bites. |
| `mcp` package API drift | Pin to a known-good version in `pyproject.toml`. |
| Tool descriptions in current schemas may not be rich enough to trigger calls | Schemas already work for Ollama's native tool path — same descriptions. If Claude doesn't call them, root cause is voice.md guidance (Hana's lane), not schemas. |
| Audit log grows unbounded | Same shape as existing `heartbeats.log.jsonl` etc — health walker handles rotation eventually. Not a ship-blocker. |
| Claude CLI flag drift between versions | Pin a minimum CLI version in setup docs; provider error messages name the flag explicitly so failures are diagnosable. |

## 12. Success criteria

1. `nell chat --persona nell.sandbox "<question that needs context>"` against the live sandbox produces a reply that quotes verifiable memory content (no confabulation).
2. `<persona>/tool_invocations.log.jsonl` shows `search_memories` (or whichever read tool) was actually called during the turn.
3. All unit tests green (~24 new + ~3 removed = ~21 net new tests).
4. Existing test suite stays green (no Ollama regression, no chat engine regression, no ingest regression).
5. PR #30 (one-shot close) + this PR together: live-exercise re-run shows >0 tool invocations and >0 ingest events. The two halves of the 2026-04-27 finding both close.
