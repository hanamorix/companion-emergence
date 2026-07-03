# Claude Code CLI — provider capability ledger

The brain shells out to the user's `claude` CLI (`ClaudeCliProvider`). Long-lived
**incorrect beliefs** about what that CLI exposes each drove build-then-rollback
churn (extended thinking, prompt caching, token streaming, in-subprocess turn
caps). This is the ONE canonical, dated home for every CLI-capability claim.
Before building anything on a CLI behaviour, check here; if it's not here or is
stale, **re-spike and add a dated row** rather than trusting memory or a scattered
gotcha.

Each row: the claim, the verdict, the date last verified, and how it was
established (spike / live log / commit). Newest concerns first.

| Capability | Verdict | Verified | How known |
|---|---|---|---|
| **Server-side prompt caching** | ✅ REAL — Anthropic prefix cache is active across separate `claude -p` calls (5-min TTL, keyed on exact prefix). Lever = a **byte-stable prefix** (freeze system prompt, push volatile to stdin tail). | 2026-06-22 | Live `chat_usage.jsonl` shows nonzero `cache_creation` + `cache_read` every row; A/B (Options A/A+) measured `cache_read` +59%/turn. Corrects the old "no cross-call caching" gotcha. |
| **Prompt REORDER wins a cache** | ❌ NO — reordering the *same* payload toward a cache is dead effort; order isn't the lever, a stable **prefix** is. | 2026-06-06 | v0.0.31 P0 spike. (This is what the spike actually refuted — it over-generalised to "no cache", which was wrong; see the row above.) |
| **Token-by-token streaming** | ❌ NO — the CLI emits the whole assistant reply as ONE `--output-format stream-json` frame, zero `content_block_delta` frames, for BOTH text and image turns. Any word-by-word UI streaming is downstream artifice (`_word_chunks`). | 2026-06-15 | P0 spike vs claude 2.1.172: image turn 333 chars in one frame @ +9.2s; text turn 256 chars in one frame @ +29.1s. Do NOT build "native image streaming in chat_stream". |
| **Extended thinking via CLI** | ❌ NO — the CLI consumes thinking blocks internally and never returns them on stdout in any `--output-format`. Subscription-only forbids direct SDK use. | v0.0.26 (2026-05-31) | v0.0.25's extended-reasoning toggle never produced output; removed in v0.0.26. `record_monologue` replaces the ambition within what the CLI exposes. If Anthropic ever surfaces thinking via `--output-format`, re-spike. |
| **Bounding the in-subprocess tool loop** | Tool calls run INSIDE the `claude` subprocess; the brain can't bound iterations. Only `--max-budget-usd` (a $ ceiling) exists; there is NO `--max-turns`. | 2026-06-19 | File-tool cost fix (v0.0.38 regression). The budget backstop is the only lever. |
| **`--safe-mode`** | Disables hooks/plugins/CLAUDE.md/skills — but ALSO disables an explicitly-passed `--mcp-config`, so it breaks the brain-tools MCP (Nell's tools). Unusable for the chat path. | 2026-07-01 | Direct test: a brain-tools tool call failed under `--safe-mode`, worked without it. |
| **`--bare`** | Skips hooks/plugins/CLAUDE.md — but disables keychain reads; auth becomes strictly `ANTHROPIC_API_KEY`/apiKeyHelper. Incompatible with subscription OAuth. | 2026-07-01 | `--help` + the subscription-only constraint. |
| **`--disable-slash-commands`** | Removes the built-in skills catalogue from context — but ALSO breaks brain-tools MCP tool loading (skills + MCP tool resolution are entangled). Unusable for the chat path. | 2026-07-01 | Direct test: tool call failed with the flag, worked without it. |
| **Explicit `CLAUDE_CONFIG_DIR`** | Setting it — even to the default `~/.claude` — forces file-based auth and BYPASSES the macOS keychain. The dedicated brain config dir therefore needs its OWN `claude auth login`; a copied (expired) `.credentials.json` won't authenticate. | 2026-07-01 | Isolated-dir tests all returned "Not logged in" until a real login was done in that dir. Underpins the clean-login onboarding feature. |
| **`claude auth login` flow** | Opens a browser, then requires the user to PASTE A CODE back into a TTY prompt (`redirect_uri` = remote `platform.claude.com/oauth/code/callback`, not localhost). Not silent, not localhost-callback. | 2026-07-01 | Observed the flow directly. Shapes any in-app authorise UX (paste-code, not auto-callback). |
| **Irreducible base-harness context** | Even with zero plugins + no CLAUDE.md, the CLI injects its OWN built-in skills catalogue (~13 commands), built-in agent types, and MCP deferred-tools framing — baked into the binary, not the user config. Can't be stripped without `--safe-mode`/`--bare`. | 2026-07-01 | Probed under a clean config dir. The `_HARNESS_FENCE` (system prompt) tells Nell to ignore this residue. |

## How to add a row

1. Re-spike the behaviour in a **clean shell** (`scripts/clean_shell.sh` — the agent shell's `ANTHROPIC_BASE_URL` proxy makes nested `claude -p` 401, masking real behaviour).
2. Add a dated row with the verdict + how-known.
3. If the new fact contradicts a CLAUDE.md gotcha, correct the gotcha too and point it here.
