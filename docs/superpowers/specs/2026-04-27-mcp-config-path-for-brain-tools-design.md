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
              └─ subprocess: claude --mcp-config <tmp.json> --print "<system>" "<user>"
                    │
                    └─ spawns: python3 -m brain.mcp_server --persona-dir <path>
                          │
                          ├─ exposes 9 tools via stdio MCP
                          └─ appends each call to <persona>/tool_invocations.log.jsonl
```

For Claude, `tool_loop` runs a single pass — the subprocess handles tool iteration internally. `tool_calls` on the returned `ChatResponse` is always empty for Claude. The off-schema fallback shape introduced in PR #28 becomes the *normal* return for Claude (no longer a fallback).

Ollama's `chat(messages, tools=...)` path is untouched.

## 3. Components

### 3.1 New files

| File | Responsibility |
|------|----------------|
| `brain/mcp/__init__.py` | Package marker |
| `brain/mcp/server.py` | Entry: `python -m brain.mcp.server --persona-dir <path>`. Opens MemoryStore + HebbianMatrix + SoulStore, registers tools, runs stdio loop |
| `brain/mcp/tools.py` | Adapter: maps each schema in `brain/tools/schemas.py` to an `mcp.server.Server.add_tool()` registration. Routes invocations through `brain.tools.dispatch.dispatch()` — no tool logic duplicated |
| `brain/mcp/audit.py` | `log_invocation(persona_dir, name, args, result_summary, error=None)` appends one line to `<persona>/tool_invocations.log.jsonl` |
| `tests/unit/brain/mcp/test_server.py` | Server unit tests |
| `tests/unit/brain/mcp/test_audit.py` | Audit log unit tests |

### 3.2 Modified files

| File | Change |
|------|--------|
| `brain/bridge/provider.py` | `ClaudeCliProvider.chat()` writes temp `mcp.json`, invokes `claude --mcp-config <tmp> --print` instead of `--json-schema`. Removes the off-schema fallback branch (it's the only path now). Keeps timeout + error handling. |
| `pyproject.toml` | Add `mcp` to `dependencies` |
| `tests/unit/brain/bridge/test_provider_chat.py` | New tests for `--mcp-config` invocation; remove obsolete `--json-schema` parsing tests |

### 3.3 Untouched

- `brain/tools/dispatch.py`, `brain/tools/impls/*.py`, `brain/tools/schemas.py` — same logic, new transport
- `brain/chat/tool_loop.py` — single-pass for Claude, full loop for Ollama
- `OllamaProvider.chat()` — unchanged

## 4. Data flow per chat call

1. `respond()` builds system_msg + history + user_input → calls `tool_loop.run_tool_loop`
2. `tool_loop` calls `provider.chat(messages, tools=build_tools_list())`
3. **ClaudeCliProvider**:
   a. Writes a temp `mcp.json` with the spawn command + `--persona-dir <active persona>`
   b. Spawns `claude --mcp-config <tmp> --print -- <flattened messages>`
   c. Claude subprocess loads MCP, sees the 9 tools, runs the chat. When context is needed, it calls `search_memories` / `get_emotional_state` / etc. via stdio MCP — `brain.mcp.server` dispatches via `brain.tools.dispatch.dispatch` and returns results. Subprocess weaves them into its reply.
   d. Each invocation is appended to `<persona>/tool_invocations.log.jsonl`
   e. Subprocess exits with final text on stdout
4. Provider returns `ChatResponse(content=text, tool_calls=())`
5. `tool_loop` returns immediately — no tool_calls to dispatch
6. `respond()` persists the turn to the ingest buffer
7. On chat exit, PR #30's `close_session` flushes the SP-4 ingest pipeline

## 5. MCP server detail

### 5.1 Entry point

```bash
python -m brain.mcp.server --persona-dir /path/to/personas/nell
```

Server reads `--persona-dir`, opens stores, registers tools, runs stdio loop until parent (claude CLI) closes the connection. Idle servers don't exist — every chat call spawns a fresh server.

### 5.2 Tool registration

For each name in `brain.chat.tool_loop._NELL_TOOL_NAMES`:
- Pull the JSON Schema from `brain/tools/schemas.py`
- Register with the MCP server (name + description from schema, inputSchema from schema's `parameters`)
- Handler dispatches via `brain.tools.dispatch.dispatch(name, args, store=..., hebbian=..., persona_dir=...)`
- On success: log invocation + return result as JSON
- On exception: log invocation with `error=str(exc)` + return `{"error": "..."}` (existing dispatch pattern)

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
      "args": ["-m", "brain.mcp.server", "--persona-dir", "/path/to/personas/nell"],
      "env": {}
    }
  }
}
```

Written via `tempfile.NamedTemporaryFile(suffix=".json", delete=True)` per call. Auto-cleaned on provider return.

### 6.2 Subprocess invocation

```bash
claude --mcp-config /tmp/mcp.XXXXX.json \
       --print \
       --append-system-prompt "<system_message>" \
       -- "<user_text>"
```

Multi-turn history is prepended to user_text the same way the current provider already does (the existing `_flatten_messages` helper stays — only the flag set changes).

### 6.3 Provider return

Always `ChatResponse(content=stdout_text, tool_calls=())`. No JSON parsing. No structured_output handling. Plain text in, plain text out.

## 7. Error handling

| Failure | Behavior |
|---------|----------|
| MCP server crash mid-call | subprocess returns whatever it had; provider raises `ProviderError("claude_cli_empty", ...)` if stdout is empty. User sees an error, not silent breakage. |
| Tool dispatch raises inside MCP server | Server logs invocation with `error=...`, returns `{"error": "..."}` as the tool result. Claude weaves the error into its reply. Voice.md already guides Nell to name tool failures honestly, not confabulate. |
| `mcp` package missing | ImportError at module load. Provider raises `ProviderError("mcp_unavailable", "pip install mcp")` with install hint. |
| Temp file fails to write | `OSError` propagates as `ProviderError("claude_cli_setup", ...)`. |
| Persona dir doesn't exist | MCP server fails fast at startup; subprocess exits non-zero; provider raises `ProviderError("claude_cli_failed", ...)`. |
| Audit log write fails | Logged to stderr, swallowed — never breaks tool dispatch. (Audit is observability, not correctness.) |

## 8. Testing

### 8.1 MCP server unit tests (`test_server.py`)

Start the server in-process via `mcp.client.session.ClientSession` over an `mcp.client.stdio` in-memory transport. Use a tmp_path persona dir with seeded MemoryStore + HebbianMatrix + SoulStore. Call each of the 9 tools, assert:
- Tool returns expected dispatch result
- Audit log line appended with right shape
- Errors in dispatch surface as `{"error": "..."}` and audit-log with `error` field
- Read-only tools (`get_emotional_state`, `search_memories`, `get_soul`, `get_personality`, `get_body_state`, `boot`) don't write
- Write tools (`add_journal`, `add_memory`, `crystallize_soul`) actually write

~15 tests.

### 8.2 Provider unit tests (extending `test_provider_chat.py`)

Mock `subprocess.run`. Assert:
- `--mcp-config` flag is set
- Temp `mcp.json` has the right `command` / `args` (including `--persona-dir`)
- Stdout text comes back as `ChatResponse.content`
- Empty stdout raises `ProviderError("claude_cli_empty", ...)`
- Missing `mcp` package raises `ProviderError("mcp_unavailable", ...)`
- Old `--json-schema` flag is *not* present

~5 new tests; ~3 obsolete `--json-schema` parsing tests removed.

### 8.3 Audit log unit tests (`test_audit.py`)

Direct unit tests on `log_invocation`:
- Appends valid JSONL
- Result summary is truncated to 140 chars (matching `_summarize_result`)
- Error field omitted on success, populated on failure
- Concurrent writes don't corrupt the file (use `O_APPEND` / line buffering)

~3 tests.

### 8.4 No live `claude` CLI integration test

A real subprocess test would require an active Claude subscription, network, and would be testing Claude's behavior — not ours. The fake provider stays for end-to-end CLI tests; the new tests above cover the seam between our code and the subprocess.

### 8.5 Verification before merge (per Hana)

After implementation + unit tests pass, manual end-to-end sanity check:
- Run `nell chat --persona nell.sandbox "what's that thing you wrote about the morning after"` against Hana's actual sandbox persona
- Verify `tool_invocations.log.jsonl` shows `search_memories` was called
- Verify Nell's reply quotes a real memory (not confabulation)

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
3. All unit tests green (~23 new + ~3 removed = ~20 net new tests).
4. Existing test suite stays green (no Ollama regression, no chat engine regression, no ingest regression).
5. PR #30 (one-shot close) + this PR together: live-exercise re-run shows >0 tool invocations and >0 ingest events. The two halves of the 2026-04-27 finding both close.
