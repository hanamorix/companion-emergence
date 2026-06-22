# 2 — Plan: how + measurement + instrumentation + thresholds

## Build order (TDD, smallest-blast-radius first)

The three parts are sequenced so the **measurement organs land before the behavioural change**
they measure — "instrument before you build" (METHODOLOGY core principle).

### Phase P1 — Instrumentation (no behaviour change)
1. Port `scripts/cache_replay_workload.py` from `7baa145b` verbatim (it's standalone; no merge
   conflict — new file). Confirm it runs against our `main` tree (the `respond()` signature it
   calls predates the caching change, so it drives OLD and NEW identically).
2. Port `_maybe_log_cache_debug` into `brain/bridge/provider.py` **without** the
   `volatile_suffix` threading yet — i.e. log `system_sha256` + char/token estimates on each
   `chat`/`chat_stream` call, gated on `NELL_CACHE_DEBUG=1`, fail-soft. This alone proves the
   *current* (pre-A) per-turn system-prompt churn (expect **many** distinct hashes — the
   regression made visible).
3. Tests: port the relevant unit tests; add one asserting `cache_debug` is a no-op when the env
   var is unset (zero file writes, no exception).

**P1 gate to self:** run the replay on the unmodified `main` → record OLD metrics into
`0-baseline.md` (real baseline, not the scratch A/B). Expect OLD C1 to FAIL (many hashes) —
that's the baseline showing the bug.

### Phase P2 — Option A (freeze system prompt)
4. Split `build_system_message` → `build_static_system_message` + `build_volatile_context` in
   `brain/chat/prompt.py`. Keep `build_system_message` intact for the image path.
   - **Care point:** enumerate ALL ~19 volatile blocks and assign each to static-vs-volatile.
     Any per-turn-varying byte left in the static half breaks C1. The cold red-team must verify
     this partition against the actual `build_system_message` body (it is the highest-risk
     correctness point — see §risks).
5. Wire the chat path (non-image) to pass the static message as `--system-prompt-file` and hold
   the volatile chunk for P3. Until P3 lands, append volatile to the prompt body as today
   (behavioural no-op vs current) so P2 is independently shippable/testable.
6. Tests: port `test_prompt_caching_split.py` C1–C4 (static/volatile split is pure-function
   testable — no `claude` call needed). Add the C7-auto image-path byte-identity test.

**P2 gate to self:** replay → C1 should now PASS (1 hash). C2/C3 directionally improved.

### Phase P3 — Option A+ (volatile → stdin tail)
7. Thread `volatile_suffix` + `include_block_clock=False` through `run_tool_loop` chat_options
   to every `provider.chat`/`chat_stream` (main loop, recruit-rerun, final forced pass), per
   `7baa145b`'s engine.py/tool_loop.py diff. Relocate the `Current time:` anchor to the tail.
8. Verify the suffix is appended in BOTH `_format_claude_print_prompt` branches (single-message
   first turn AND multi-message) so it is never dropped on a session's first turn — this is an
   explicit fork comment; assert it in a test.
9. Tests: port the engine test; add a through-path test asserting the volatile suffix reaches
   the provider on the recruit-rerun and final-forced-pass calls (not just the main call) — the
   "wire it back" / Organ-DoD discipline (a writer tested in isolation rots). **Assert via a
   unit/mock spy on `provider.chat`, NOT via `cache_debug.jsonl`** — the final-forced-pass call
   passes `options` without `persona_dir` (deliberate, to preserve no-usage-log), so
   `_maybe_log_cache_debug` early-returns and records nothing for it (red-team finding). The
   suffix still reaches the provider; only the debug-log can't see it.

### Phase P4 — Measurement on realistic + tools workloads (C4, C5/C2-live)
10. Build a realistic fixture: 16KB `voice.md` + a real `active_conversations` JSONL capped
    under 80 msgs. Run the text-path A/B (`--history-file --history-msgs 79`) → record C4
    magnitude.
11. **C5 / C2-live — measure the dominant tool-schema term on the LIVE path (red-team
    correction).** Do NOT instrument `_chat_with_mcp_tools` — stage-3 verified it is the
    non-streaming path the live UI never calls; the live path is `_StreamingProxy.chat()` →
    `chat_stream`, which **already logs** `cache_creation`/`cache_read` (`provider.py:888`).
    Instead: run a **fixed prompt script through the real bridge with tools enabled**, on the OLD
    build then the NEW build, and diff the fresh `chat_usage.jsonl` `call_type=="chat"` rows
    (mean `cache_creation`, `cache_read_ratio`). This exercises the MCP tool-schema mass the
    replay strips. (Optional, separate: stamping a correlation key on ALL tool rows to unblock
    the config's BLOCKED `tool_calls_per_request` metrics is its own change — out of scope here.)

### Phase P5 — Record correction + wiring
12. Correct the CLAUDE.md "no caching" gotcha + the deferred-memory copy (repo edits — **only
    after the kindled session releases `main`**; rebase this branch onto the post-Phase-7 main
    first).
13. Refresh `docs/maturity-manifest.md` if the split changes an organ's wiring description.

## Measurement (how each criterion is verified)

| Criterion | Instrument | Command |
|---|---|---|
| C1 | `cache_debug.jsonl` | replay (P1+), count distinct `system_sha256` (replay only) |
| C2-text, C3 | `chat_usage.jsonl` via replay | `cache_replay_workload.py --compare OLD NEW` (text path) |
| C2-live, C5 | live `chat_usage.jsonl` (streaming+tools, already logged `provider.py:888`) | fixed prompt script through real bridge, OLD vs NEW, diff fresh chat rows |
| C4 | replay, realistic fixture | replay `--history-file <real> --history-msgs 79` |
| C6 | test suites | `uv run pytest` + ruff + `pnpm test` + `pnpm build` |
| C7-auto | image-path test (pin `now` seam) | targeted unit test, `_format_claude_context_block(now=...)` fixed |
| C7 | replay `--dump-replies` side-by-side | Hana reads OLD-vs-NEW, ≥8 turns, ≥1 tools run |
| A1 (advisory) | uncontrolled live `chat_usage.jsonl` | tail vs `0-baseline.md` (sanity only) |

## Instrumentation added to scope
- `cache_debug.jsonl` (P1) — the byte-stability probe. **This is the signal that would have
  made the original regression impossible to miss.**
- `cache_replay_workload.py` (P1) — the **comparable replay workload** the project config
  declares missing. Adopting it **flips `cost_per_chat_call_usd`, `num_turns_per_chat_call`,
  `cache_creation_per_chat_call`, `cache_read_ratio` from advisory → gating** (config:
  `gating: true` means "gating once a comparable workload exists"). Update
  `guarded-change.companion.md` to record the workload now exists.
- **No new tools-path instrumentation needed for C5** (red-team correction). The live
  streaming+tools path already logs cache tokens (`provider.py:888`); C2-live reads existing
  rows. (Unblocking the config's BLOCKED `tool_calls_per_request` metrics — stamp a correlation
  key on ALL tool rows — is a separate change, out of scope here.)

## Thresholds → routing (gating vs advisory)

- **Gating:** C1, C2-text, C2-live, C3, C4, C5, C6, C7-auto, C7(human). A fail on any bounces
  per the severity model. (C2-live/C5 is the gate on the dominant tool-schema term.)
- **Advisory:** A1 (live-log delta) — surfaced in `8-harness.md`, never auto-bounces (live
  turns are not a comparable workload).
- **Severity mapping for likely findings:**
  - Volatile block mis-partitioned into static (C1 fails) → **blocker** (defeats the change's
    entire premise).
  - A+ suffix dropped on a call path (first turn / recruit-rerun / final pass) → **major**
    (correct goal, broken wiring).
  - Magnitude smaller than hoped but real (C4 records a small positive drop) → **minor / human
    call** — not a bounce; a recorded tradeoff (the win is the create→read shift even if small).
  - Tools-path effect unmeasured at ship (C5 unsatisfied) → **blocker** (ships an unmeasured
    regression on the real path — the exact failure this loop exists to prevent).
  - Voice (c)-failure in C7 (NEW narrates ambient instead of replying) → **blocker**.

## Wiring (§Wiring — required by project hard rules)
- **Reads from:** `build_system_message`'s existing volatile blocks (unchanged content);
  `chat_usage.jsonl` cache fields (already logged by `usage_log.py`); the `run_tool_loop`
  chat_options seam.
- **Feeds into:** the `claude` CLI prompt assembly (smaller per-turn create); the
  guarded-change harness (the replay workload becomes the standing gating instrument for the
  whole cost/cache class — every future change in this class now has a real gate).
- **Downstream consumers unaffected:** salience, tool-recruitment, reflection-debounce,
  monologue/attunement pass-2 (the volatile *content* is unchanged; only its position moves).

## Build-hold condition
Phases P1–P4 build is **held until the kindled-link Phase 7 session merges to `main`** (it owns
the main worktree and touches none of these files, but P2/P3 edit `prompt.py`/`engine.py`/
`provider.py`/`tool_loop.py` and we rebase onto post-Phase-7 `main` to apply the design cleanly
once, not against a moving target). P1 (new files only) could land earlier if desired — it has
no conflict surface. This plan + the stage-3 red-team are produced now so the build is ready to
execute the moment the tree frees.
