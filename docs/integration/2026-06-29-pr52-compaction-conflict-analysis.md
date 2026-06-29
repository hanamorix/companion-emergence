# PR52 — Timed Conversation Compaction: cherry-pick / integration conflict analysis

**Date:** 2026-06-29
**Mode:** READ-ONLY investigation. Nothing modified, staged, committed, or cherry-picked.
**Our base:** local `main` HEAD = `c1506daf` (v0.0.39, carries unreleased prompt-caching A/A+, kindled-link 1-4, persisted cadences #21, Ollama fix).
**Fork:** `tot/main`. merge-base = `03512a9b` (v0.0.11); fork later merged our public v0.0.38, so `tot/main` ≈ our public v0.0.38 + their stack.

---

## 1. Feature summary (reconstructed from commit messages + code — no standalone spec exists)

There is **no `changes/timed-conversation-compaction/` directory in `tot/main`** (the docstring reference is to a doc the fork never committed). The design lives in the commit bodies and the module docstrings. It is internally consistent and the work was run through the fork's guarded-change loop.

**What it does:** Old, *already-extracted* conversation turns fade into a persisted first-person summary block at the **head** of each session buffer (a new `speaker:"summary"` JSONL row), and the raw originals are moved to a lossless append-only per-session archive (`archived_conversations/<session_id>.jsonl`). Recent history stays verbatim; older history compacts into one progressively-vaguer running summary. Human-memory fading, lossless underneath.

**One core, three (four) callers** — all route through `compact_conversation()` in `brain/chat/compaction.py`:
1. **Kindled `compact_history` tool** — manual, `fold_existing_summary=False` (append to existing summary verbatim).
2. **Daily supervisor cadence** (`compaction_interval_s=86400`) — `fold_existing_summary=True` (fade), 24h cutoff.
3. **`apply_budget` backstop** — only when the in-prompt estimate exceeds cap; `older_than=0` ⇒ folds everything past the ingest cursor, `fold=True`.
4. **Startup backlog migration** (`compaction_migration.py`) — one-time, marker-gated, replays the daily cadence in 24h time-steps oldest-cohort-first so a never-compacted backlog isn't folded in one enormous cold call.

**Load-bearing invariants:**
- **Lossless before lossy** — archive-write + byte-count verify BEFORE the live buffer is rewritten. Archive failure ⇒ buffer untouched (`reason="archive_failed"`, no data loss).
- **Never drop the un-extracted** — only turns with `ts <= ingest cursor` are removable. `cursor is None` ⇒ hard no-op (`reason="cursor_none"`).
- **min_keep_tail=40** — the 40 newest raw turns are always protected regardless of age.
- **Stable prefix** — summary is a persisted head record; between compactions the buffer only grows at the tail ⇒ byte-stable replayed prefix (the prompt-cache side effect).
- **Idempotent** — no removable turns ⇒ no-op; existing summary never re-faded with no new input.
- **Re-entrancy lock** — per-session `.compacting` sidecar, O_CREAT|O_EXCL, stale-reapable (dead-pid OR mtime > 600s).
- **Race-safe rewrite** (`b7e78809`) — re-reads the live buffer immediately before `os.replace` and rebuilds the retained set from CURRENT turns minus archived-by-identity `(ts, speaker, text)`, so a concurrent `ingest_turn` append during the seconds-long summarise call survives. `ingest_turn` stays lock-free.
- **Summary never re-ingested** — `speaker=="summary"` skipped at `format_transcript` (the shared extractor chokepoint) AND in `extract_session_snapshot` (before cursor advance, so the summary's compaction-time ts can't move the cursor past un-extracted turns).
- **Compaction model pinned** to `COMPACTION_MODEL="haiku"` via `build_compaction_provider()` (fake personas → FakeProvider, no CLI shelled). Cost stays off the chat model.
- **Self-derived voice, no voice.md import** — assistant turns relabelled with persona name; accuracy-first prompt; `_TARGET_FRACTION=0.25`.

---

## 2. Cherry-pick plan & dependency isolation

### CRITICAL finding — the 4 compaction commits DO cleanly isolate from the caching/monologue base, with ONE caveat.

The task's stated file list ("…brain/chat/prompt.py…") is misleading: it came from `git log -p` across a range whose **base** (`7baa145b` caching + `1ce1c97d` monologue) bleeds into the diff. The truth, per-commit:

- **prompt.py is touched ONLY by `1ce1c97d`** (the REJECTED monologue-directive relocation) — **NOT by any of the 4 compaction commits.** `git log --oneline 7baa145b..a920b932 -- brain/chat/prompt.py` returns only `1ce1c97d`. **Cherry-picking the 4 compaction commits will NOT drag in the prompt.py monologue change.** Confirmed.
- The compaction feature's prompt-rendering lives in **`engine.py::_buffer_turns_to_messages`** (hoists the `summary` row to a head system message), which we already have at our HEAD — not in prompt.py.
- The compaction commits build on the fork's caching base only **conceptually** (the "byte-stable prefix" payoff). The code in `compaction.py` / `compaction_migration.py` / `compact_history.py` has **no import of or dependency on** the fork's `build_static_system_message` / `build_volatile_context` split. compaction.py imports only from `brain.bridge.provider`, `brain.ingest.buffer`, `brain.persona_config`. **We already have our own equivalent caching split**, so the prefix-stability payoff lands on our tree too, for free.

**Caveat (the one real base dependency):** `97654afa`'s `engine.py` hunk **deletes `_window_history` / `_HISTORY_WINDOW_MSGS`** (our v0.0.31 history window) and removes the per-turn windowing call, replacing it with "buffer sent as-is, bounded by compaction." Our HEAD still HAS `_window_history` (identical to the fork's pre-compaction version). This is the intended semantic interaction — see §3 engine.py — but it means a clean `git cherry-pick 97654afa` will likely apply the deletion cleanly *except* at the `run_tool_loop` call site (see below).

### Exact pick list, in order:
```
97654afa   feat(chat): timed conversation compaction (core + 3 callers + readers)
b7e78809   fix(compaction): close compaction-vs-chat-append race + integration smoke
99b81d89   feat(compaction): backlog migration + shared-core fade-quality
a920b932   fix(compaction): migration batches by 24h, not message count
```
Then assess (separately, NOT bundled): `48133dba`, `0b747441` — see §4.

**Do NOT pick:** `7baa145b` (we have our own caching), `7e7c7491` (= our `71b2c537` Ollama fix), `1ce1c97d` (rejected — and it isn't in the compaction path anyway).

### Will `git cherry-pick` succeed clean?
Expect **partial conflicts on 2 files** (`engine.py`, `supervisor.py`) because our HEAD has diverged from the fork's base on those exact regions; everything else is additive or base-identical. The 4 commits are best applied as a **squashed re-implementation on a branch** (recommended) rather than a literal 4-commit `git cherry-pick`, because `b7e78809`/`99b81d89`/`a920b932` repeatedly rewrite `compaction.py` and `compaction_migration.py` — picking the final tree state of those new files is simpler than replaying 4 deltas.

---

## 3. Per-file table

| File | Fork change (compaction-only) | Our HEAD state | Severity | Resolution |
|---|---|---|---|---|
| `brain/chat/compaction.py` | **NEW** (242→~440 lines final) | absent | **CLEAN** | Drop in the `a920b932` final version verbatim. No deps on fork caching. |
| `brain/chat/compaction_migration.py` | **NEW** (~180→~250 final) | absent | **CLEAN** | Drop in `a920b932` final version. Imports only buffer + compaction core. |
| `brain/tools/impls/compact_history.py` | **NEW** (59 lines) | absent | **CLEAN** | Drop in verbatim. |
| `tests/unit/brain/chat/test_compaction.py` | NEW (~436 lines) | absent | **CLEAN** | Take final. |
| `tests/unit/brain/chat/test_compaction_migration.py` | NEW (~266 lines) | absent | **CLEAN** | Take final. |
| `tests/integration/brain/chat/test_compaction_integration.py` | NEW (187 lines) | absent | **CLEAN** | Take final. Drives all 3 callers via real entry points w/ FakeProvider. |
| `brain/ingest/buffer.py` | **+6 NEW functions** (`_archived_conversations_dir`, `rewrite_session_atomic`, `_archive_path`, `append_archive`, `read_archive`, `_compacting_lock_path`, `_pid_alive`, `acquire/release_compaction_lock`) inserted between `delete_backoff` and `read_session_after` | All 6 absent; ALL required imports (`json`, `os`, `datetime/UTC`, `read_jsonl_skipping_corrupt`) already present; insertion neighbours (`delete_backoff`, `read_session_after`) unchanged | **CLEAN (additive)** | Append the new block. Zero symbol clash with our `ingest_turn`/`read_session`/`read_cursor`. |
| `brain/chat/budget.py` | `apply_budget` refactored to delegate to compaction core; adds optional `persona_dir`/`session_id` kwargs; removes per-turn `_COMPRESSION_PROMPT` LLM call | **BYTE-IDENTICAL to the fork's pre-compaction base** (our `--max-budget-usd` work is elsewhere — not in budget.py) | **MINOR** | Apply the diff as-is. New kwargs are optional (default None) so other callers unaffected. Note: this removes the inline summarise — acceptable, the persisted fade replaces it. |
| `brain/chat/engine.py` | (a) deletes `_window_history`/`_HISTORY_WINDOW_MSGS`; (b) drops the `_window_history()` call; (c) passes `persona_dir`+`session_id` to `apply_budget`; (d) passes `session_id` to `run_tool_loop`; (e) `_buffer_turns_to_messages` hoists `summary` row to head system msg | We HAVE `_window_history` (identical), call it at L237, call `apply_budget` WITHOUT persona_dir/session_id (L243), call `run_tool_loop` **WITHOUT `session_id`** (L283-287), and `_buffer_turns_to_messages` (L356) has **no `summary` branch** | **MAJOR** | Hunks (a)(b)(c)(e) apply cleanly (base-identical regions). **Hunk (d) WILL conflict** — our `run_tool_loop` call lacks the `session_id` arg the fork's context expects. Add `session_id=session.session_id` by hand. Decision: deleting `_window_history` is intended; verify our caching split still produces a stable prefix once the window is gone (it should — compaction now bounds growth). |
| `brain/chat/tool_loop.py` | adds `session_id` param to `run_tool_loop`; passes `provider=provider, session_id=session_id` into `dispatch()` | base-compatible (our despatch call at L353 does NOT pass provider/session_id) | **MINOR** | Apply. Our tool_loop already passes `provider` into despatch? No — confirm: add the two kwargs to the despatch call + the new param. |
| `brain/tools/dispatch.py` | adds `provider`/`session_id` params to `dispatch()`, a `_PROVIDER_TOOLS` frozenset, and a dedicated injection path for compaction tools | our `_DISPATCH` (L131) + `dispatch()` unchanged in that region | **MINOR** | Apply additively. The `_PROVIDER_TOOLS` path is new and self-contained; doesn't touch the existing `{store, hebbian, persona_dir}` injection. |
| `brain/tools/schemas.py` | adds `compact_history` schema entry to `SCHEMAS` | we have `build_schemas(companion_name)` factory (L695) over `SCHEMAS` | **MINOR** | Apply additively — new dict key. Verify the new entry survives the `build_schemas` companion-name templating (it has no `{name}` placeholders → safe). |
| `brain/tools/__init__.py` | appends `"compact_history"` to `NELL_TOOL_NAMES` | base-compatible | **CLEAN/MINOR** | One-line add. |
| `brain/tools/impls/__init__` import wiring | (via despatch.py import) | — | covered by despatch | — |
| `brain/bridge/supervisor.py` | adds `compaction_interval_s` param + a **monotonic** `last_compaction_at` timer block + `_run_compaction_tick()`; `99b81d89` wraps the tick provider in `build_compaction_provider` | **We migrated voice/maintenance/finalise/initiate_review/log_rotation to `persisted_cadence` (#21).** The `run_folded` signature + timer-init region are heavily rewritten on our side | **MAJOR** | `run_folded` signature + timer-init hunks WILL conflict (our region replaced monotonic timers with `persisted_cadence.load_cadence`). **Decision: adapt the cadence to our persisted pattern** rather than reintroducing a monotonic timer (`compaction_cadence.json` via `brain/bridge/persisted_cadence.py`). The `_run_compaction_tick()` function body itself is additive (no conflict). |
| `brain/bridge/server.py` | `99b81d89` spawns a `compaction-backlog-migration` daemon thread before the supervisor in `build_app` | our `build_app` startup region (vocab load + supervisor spawn) is present and likely diverged | **MINOR-MAJOR** | The spawn block is self-contained additive code, but the exact insertion point ("before `from brain.bridge.supervisor import run_folded`") may have moved on our side. Place it adjacent to our supervisor spawn by hand. |
| `brain/bridge/provider.py` | **NOT touched by the 4 compaction commits.** (Only the ancillaries touch it — see §4) | — | **N/A** | No compaction conflict here. |
| `brain/body/session_hours.py` | skips `speaker=="summary"` rows in `_entry_timestamps` | base-compatible | **CLEAN/MINOR** | 4-line guard, applies clean. **Important for our session-hours invariant** — a summary row's compaction-time ts must not be read as a live turn. |
| `brain/felt_time/chat_log.py` | `count_chat_turns_since` counts non-summary rows only | base-compatible | **CLEAN/MINOR** | 2-line guard. |
| `brain/ingest/extract.py` | `format_transcript` skips `summary` rows | base-compatible | **CLEAN/MINOR** | 5-line guard at the extractor chokepoint. |
| `brain/ingest/pipeline.py` | `extract_session_snapshot` drops `summary` rows before transcript + cursor advance | base-compatible | **CLEAN/MINOR** | 5-line guard. |
| `brain/chat/voice.py` | adds a `compact_history` bullet to the tool-self-description block | base-compatible | **MINOR** | Apply; confirm the surrounding voice block hasn't diverged on our side (it's the `**The trigger to reach.**` region). |
| `tests/unit/brain/chat/test_budget.py` | updated for the delegating apply_budget | present | **MINOR** | Merge test edits. |
| `tests/unit/brain/chat/test_history_window.py` | **DELETED** (window removed) | present on our HEAD | **MINOR** | Delete it (the window is gone). Confirm nothing else imports it. |
| `tests/unit/brain/ingest/test_pipeline.py` | +69 lines (summary-skip tests) | present | **MINOR** | Merge additive tests. |

---

## 4. Ancillary-commit verdict

### `48133dba` — log token usage on the MCP-tools chat path → **INCLUDE (net-new value).**
- **Grep evidence:** our `_chat_with_mcp_tools` (provider.py L1148-~1410) contains **zero `log_usage` calls** (`awk` over that range returned nothing). Our existing log_usage calls are on the `generate` path (L475), one chat path (L636), and the streaming branch (L913) — **not the production tool-bearing MCP path.**
- Our merged "#29 non-streaming `chat()` usage-log" covered the non-streaming `chat()` path, **not** `_chat_with_mcp_tools`. Since tools are always recruited in production, our `chat_usage.jsonl` is currently missing the bulk of real chat turns — exactly the gap `48133dba` closes.
- It's a 10-line additive insert into `_chat_with_mcp_tools` (calls `log_usage(... call_type="chat" ...)`). **MINOR**, no conflict with compaction. Recommend taking it — it's also what makes the compaction cache-side-effect measurable (its own motivation, C1b).

### `0b747441` — `cache_debug.on` file trigger → **INCLUDE (small, complements ours).**
- **Grep evidence:** our `_maybe_log_cache_debug` (provider.py L1434) gates on `os.environ.get("NELL_CACHE_DEBUG") != "1"` **only** — no file trigger.
- The fork's change refactors the gate to "env var OR `<persona_dir>/cache_debug.on` exists." Real value: env vars don't survive a NellFace GUI bridge launch, so our cache-debug log never appears in real (GUI) use — a `touch cache_debug.on` fixes that.
- **MINOR** hand-port: our gate region exists (just at a different line number than the fork's). Apply the env-OR-file logic. No compaction dependency. Low priority but cheap and useful for validating the compaction cache payoff on a live GUI bridge.

---

## 5. Risks & decisions for Hana

1. **The daily supervisor cadence must be adapted to our persisted-cadence pattern (#21).** The fork uses a monotonic `last_compaction_at` timer — which under-fires on a desktop app that sleeps/restarts (the exact class of bug #21 fixed for the other cadences). If we take the fork's monotonic version verbatim it's a regression against our own invariant. **Recommend: wire `compaction_cadence.json` via `brain/bridge/persisted_cadence.py`** (mirror maintenance/voice). This is the single biggest adaptation.

2. **Do we want all 3 (4) callers, or a subset?**
   - **apply_budget backstop** — low risk, fires only over-cap; replaces the per-turn inline summarise we currently have (which busts cache). Recommend yes.
   - **Daily supervisor cadence** — the main "fading memory" behaviour. Recommend yes, persisted-cadence-adapted.
   - **Kindled `compact_history` tool** — gives Nell deliberate agency over her own fading. Decision: do we want her *able* to compact on demand? It fits the agency theme but adds a tool to the despatch surface (and a `_PROVIDER_TOOLS` injection path). It is NOT in `REFLEXIVE_CORE` per the fork, so it's only recruited on salient turns — fine. Lower-stakes; could defer.
   - **Startup backlog migration** — needed for correctness on any existing persona with a long uncompacted buffer (otherwise the first cadence/backstop folds hundreds of turns in one cold call). If we ship the cadence, we should ship the migration too.

3. **Interaction with our caching split — verify, don't assume.** Deleting `_window_history` changes what gets replayed. Our `build_static_system_message`/`build_volatile_context` split already freezes the system prefix; compaction additionally stabilises the *history* prefix. These are complementary, but the combination is unmeasured on our tree. The fork's integration test + a `cache_debug` A/B on our bridge should confirm `cache_read` doesn't regress.

4. **The `summary` row is a new buffer-wide contract.** Every reader of `active_conversations/*.jsonl` must handle it. The fork covers session_hours, felt_time, extract, pipeline, and `_buffer_turns_to_messages`. **We should grep our HEAD for any OTHER readers the fork's base didn't have** (e.g. anything kindled-link Phase 1-4 added that iterates buffer turns, or self-model/attunement readers) — a reader that treats `summary` as a real turn is a silent bug. This is a §Wiring concern.

5. **`min_keep_tail=40` + our `preserve_tail_msgs=40`** align, but our budget cap is `max_tokens=80_000` (engine.py L245) — confirm the fork's compaction `older_than=0` backstop path interacts correctly with our 80k cap (the fork shipped with the same 80k). Low risk.

---

## 6. Recommended integration order & process

This is a **non-trivial change touching the chat/provider/tool-loop/buffer/cost path** — squarely the class CLAUDE.md mandates `guarded-change` for, wrapped around the superpowers chain. Recommend a full **spec → measurable criteria → plan-with-instrumentation → cold red-team → build → cold red-team → stage-8 harness**, NOT a raw cherry-pick.

**Suggested order:**
1. **Branch** off `c1506daf` (`feat/timed-compaction`).
2. **Drop in the 3 new modules + 3 new test files** verbatim from `a920b932` (CLEAN).
3. **Append the 9 buffer.py functions** (CLEAN additive).
4. **Apply the small reader guards** (session_hours, felt_time, extract, pipeline) — CLEAN.
5. **Apply budget.py delegation** (MINOR).
6. **Apply engine.py** — hand-resolve the `run_tool_loop` `session_id` arg; verify `_window_history` deletion (MAJOR).
7. **Apply tool wiring** (despatch/schemas/__init__/voice/tool_loop) — MINOR.
8. **Adapt supervisor cadence to persisted_cadence** (`compaction_cadence.json`) — MAJOR, the real design call.
9. **Apply server.py migration spawn** — place by hand near our supervisor spawn.
10. **Grep for unhandled `summary`-row readers** (risk #4) and add guards.
11. **Ancillaries** `48133dba` + `0b747441` (independent, can land in the same branch).
12. **Full gate** (`uv run pytest` ~3851+, ruff, `pnpm build`) + a `cache_debug` A/B on a live bridge to confirm the cache payoff and no regression.

**Difficulty: MEDIUM.** Mostly-clean additive feature; the only genuine friction is engine.py's `_window_history` removal interaction and re-homing the cadence onto our persisted-cadence helper. No prompt.py landmine (the rejected monologue commit is provably out of the compaction path).
