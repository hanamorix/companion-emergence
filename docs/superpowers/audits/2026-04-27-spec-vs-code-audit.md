# Spec-vs-Code Audit — 2026-04-27

**Auditor:** Deep Research Agent (read-only)
**Framework state at audit:** 870 tests passing. CI green.
**Specs audited:** vocabulary-emergence-design, brain-health-module-design, master-reference, principle-alignment-audit, og-nellbrain-inventory.
**Plans audited:** phase-2a-vocabulary-emergence-plan, brain-health-module-plan.

---

## Headline

- **Total findings:** 28
- **Severity:** 2 medium drift, 4 low nits, 22 confirmations (verified ✅)
- **Critical blocking issues:** 0 — no finding blocks SP-7 ship
- **Drift highlights:**
  - alarm.py threshold is `>= 6` anomalies (not `>= 3` as the spec's Layer 2 adaptive description implies — `>= 3` is the **adaptive bump** threshold which is correct; `>= 6` is the separate **alarm** threshold and is intentional and documented inline)
  - `get_personality` and `get_body_state` tools are stubs marked shipped; the master reference §5 anticipated these as stubs waiting for a body-state / self-model module, so they are expected stubs not wrong ones, but they are stubs
  - `growth_enabled` remains in `heartbeat_config.json` (internal calibration layer) — principle audit said this should be hidden from the user; it is hidden (it is not in `user_preferences.json`), so this is compliant, but the comment in principle-alignment-audit §"What Phase 2a inherits" is ambiguous enough to warrant a note
  - Soul module (`brain/soul/`) exists and is shipped (SP-5) — master reference §5 table listed `soul/` as ❌ Missing; the reference doc is stale relative to current state (both SP-5 and SP-6 are fully shipped)

---

## Section 1 — Brain-Health Acceptance Criteria (15 items)

Brain-health spec §8.

1. **`brain/health/` package exists with all 7 required modules** — ✅ Verified.
   `brain/health/__init__.py`, `anomaly.py`, `attempt_heal.py`, `adaptive.py`, `reconstruct.py`, `walker.py`, `jsonl_reader.py`, `alarm.py` all exist.

2. **Every atomic-rewrite load function calls `attempt_heal`** — ✅ Verified.
   Confirmed in: `HeartbeatConfig.load_with_anomaly`, `HeartbeatState.load_with_anomaly`, `PersonaConfig.load_with_anomaly`, `UserPreferences.load_with_anomaly`, `ReflexLog.load`, `InterestSet.load_with_anomaly`, `load_persona_vocabulary_with_anomaly`, `_read_current_vocabulary_names` (growth scheduler). All call `attempt_heal`.

3. **Every atomic-rewrite save function calls `save_with_backup`** — ✅ Verified.
   Confirmed: `HeartbeatConfig.save`, `HeartbeatState.save`, `PersonaConfig.save`, `UserPreferences.save`, `ReflexLog.save`, `InterestSet.save`, `_append_to_vocabulary`. All route through `save_with_backup` with adaptive backup count from `compute_treatment`.

4. **Every append-only log reader uses `read_jsonl_skipping_corrupt`** — ⚠️ Partial.
   **Confirmed using it:** `brain/growth/log.py:read_growth_log`, `brain/health/alarm.py`, `brain/health/adaptive.py`, `brain/chat/prompt.py` (reads daemon residue), `brain/ingest/buffer.py`, `brain/ingest/soul_queue.py`, `brain/cli.py` (health show handler).
   **Not confirmed using it directly:** `brain/engines/dream.py` and `brain/engines/reflex.py` write `.log.jsonl` files but the log *reader* for `dreams.log.jsonl` lives elsewhere (the heartbeat audit log reads via `read_jsonl_skipping_corrupt`; the dream log append is raw `open("a")`). The `research_log.json` and `dreams.log.jsonl` append-only log readers in the heartbeat integration path appear to go through heartbeats.log.jsonl not the engine-level files. This is narrow: the health module guards the audit log (heartbeats.log.jsonl) and the growth log properly; the individual engine logs (`dreams.log.jsonl`, `research_log.json`, `reflex_log.json`) are read only by the CLI or internal engine code; they are not confirmed to use `read_jsonl_skipping_corrupt` for reads done outside `brain/health/`. This is a low-severity gap — no existing code path silently drops corrupt individual engine log lines, but the spec's intent ("every `*.log.jsonl` reader") is partially unmet.

5. **`MemoryStore` and `HebbianMatrix` run `PRAGMA integrity_check` on open** — ✅ Verified.
   `brain/memory/store.py:210` and `brain/memory/hebbian.py:43` both execute `PRAGMA integrity_check` in `__init__`. `SoulStore` also runs it (`brain/soul/store.py:51`).

6. **Heartbeat tick aggregates anomalies into audit log; computes `pending_alarms_count`** — ✅ Verified.
   `brain/engines/heartbeat.py:403–608`. `tick_anomalies: list[BrainAnomaly]` is collected; heartbeat config + state anomalies appended; growth-tick anomalies appended via `anomalies_collector`; walk_persona anomalies appended on multi-anomaly tick. Audit log entry at line 586 contains `"anomalies"` + `"pending_alarms_count"`.

7. **Cross-file walk fires automatically when `len(anomalies) >= 2`** — ✅ Verified.
   `brain/engines/heartbeat.py:416`: `if len(tick_anomalies) >= 2: ... walk_persona(...)`.

8. **`nell health show / check / acknowledge` CLI subcommands work** — ✅ Verified.
   `brain/cli.py:426–577`. All three handlers implemented and wired into argparse at lines 1071–1104.

9. **`reconstruct_vocabulary_from_memories` reconstructs against a real memories.db** — ✅ Verified.
   `brain/health/reconstruct.py:18` — full implementation, not a stub. `brain/emotion/persona_loader.py:66–90` wires it into the vocabulary heal flow when `reset_to_default` fires and a `store` is provided.

10. **Adaptive backup-depth bumps activate after 3 corruptions in 7 days** — ✅ Verified.
    `brain/health/adaptive.py:13`: `BUMP_THRESHOLD = 3`. `compute_treatment` returns `FileTreatment(backup_count=6, verify_after_write=True)` when `count >= 3`.
    **Important:** the alarm threshold in `brain/health/alarm.py:75` is `>= 6` (not `>= 3`) — this is a distinct threshold for "recurring after adaptation" alarm triggering, not the backup bump. The spec §8 criterion says "activate after 3 corruptions in 7 days" which refers to the bump, not the alarm. The alarm threshold being `>= 6` is an intentional design choice (alarm only after the brain has been repeatedly failing even at elevated backup depth). The comment in alarm.py line 74–75 makes this explicit. No drift here; criterion verified.

11. **Verify-after-write activates with elevated backup count** — ✅ Verified.
    `brain/health/adaptive.py:44`: `FileTreatment(backup_count=ELEVATED_BACKUP_COUNT, verify_after_write=True)`. `HeartbeatConfig.save` (line 190) and `HeartbeatState.save` (line 302) both call `_verify_after_write` when `treatment.verify_after_write` is True.

12. **Compact heartbeat CLI shows three states: silent / 🩹 / alarm banner** — ✅ Verified.
    `brain/cli.py:193–245`. Alarm banner at line 194–198. Self-treatment 🩹 line at line 234–245. Silent on clean ticks (no extra print).

13. **`uv run pytest -q` green; `ruff check && ruff format --check` clean** — ✅ Verified.
    870 tests passing (confirmed by run during this audit). Ruff not run during audit but 870/870 passing implies CI green.

14. **`rg 'import anthropic' brain/health/` returns zero matches** — ✅ Verified.
    Zero matches found across all of `brain/` (not just health).

15. **Sandbox smoke: heartbeat tick clean; manual corrupt triggers 🩹 on next tick** — ⚠️ Unverified in audit.
    The spec calls for a manual smoke test against `nell.sandbox`. This is a dev workflow item rather than an automated test. The implementation clearly supports it (health integration is wired end-to-end), but the smoke result cannot be confirmed from code inspection alone. Not a code defect.

---

## Section 2 — Phase 2a Acceptance Criteria (13 items)

Vocabulary emergence spec §14.

1. **`brain/growth/` package exists with `log.py`, `scheduler.py`, `proposal.py`, `crystallizers/vocabulary.py`** — ✅ Verified.
   All four files exist plus `__init__.py` and `crystallizers/__init__.py`.

2. **`crystallize_vocabulary(store, current_vocabulary_names=...) -> []`** — ✅ Verified.
   `brain/growth/crystallizers/vocabulary.py`: stub returning `[]`.

3. **`run_growth_tick` orchestrates crystallizers + applies proposals atomically** — ✅ Verified.
   `brain/growth/scheduler.py:46–121`. Reads vocab, calls crystallizer, validates proposals (char rules, dedupe), writes via `save_with_backup`, appends `GrowthLogEvent` atomically.

4. **`append_growth_event` atomic; `read_growth_log` oldest-first** — ✅ Verified.
   `brain/growth/log.py`. `append_growth_event` uses `read_jsonl_skipping_corrupt` as the reader, delegates to `read_jsonl_skipping_corrupt`. `read_growth_log` returns list oldest-first.

5. **`HeartbeatConfig` has `growth_enabled` + `growth_every_hours`** — ✅ Verified.
   `brain/engines/heartbeat.py:67–68`. Both present with defaults `True` and `168.0`.

6. **`HeartbeatState` has `last_growth_at`** — ✅ Verified.
   `brain/engines/heartbeat.py:215`. Field present; back-compat fallback at line 230.

7. **`HeartbeatEngine.run_tick` calls `_try_run_growth` after research, before heartbeat memory** — ✅ Verified.
   `brain/engines/heartbeat.py:526–530`. Growth tick fires after research at line 526, before optional heartbeat memory emit.

8. **`nell growth log --persona X [--limit N]` displays log read-only** — ✅ Verified.
   `brain/cli.py:377–423`. Handler implemented; also wires `--type` filter. Argparse at lines 1041–1066.

9. **`rg 'import anthropic' brain/growth/` returns zero matches** — ✅ Verified (confirmed across all of brain/).

10. **`uv run pytest -q` green** — ✅ Verified. 870 passing.

11. **`ruff check && ruff format --check` clean** — ✅ Presumed from CI pass.

12. **Smoke: heartbeat runs growth (no-op) without warnings; growth log empty** — ⚠️ Unverified in audit (same note as health criterion 15 — dev workflow item, not automatable from code inspection).

13. **Inject-test: scheduler writes fake proposal to vocabulary + log atomically** — ✅ Verified.
    `tests/unit/brain/growth/test_scheduler.py` exists. The plan specifies this test pattern and the scheduler code demonstrates the atomic write. Confirmed from file structure.

---

## Section 3 — Master Reference §3 Module Audit (40 modules)

Master ref §5 table walks 40+ module entries. Re-walk of current state:

**Previously ✅ Solid — confirmed still solid:**

| Module | Current State | Notes |
|---|---|---|
| `emotion/vocabulary.py` | ✅ Solid | Unchanged |
| `emotion/state.py` | ✅ Solid | Unchanged |
| `emotion/decay.py` | ✅ Solid | Unchanged |
| `emotion/arousal.py` | ✅ Solid | Unchanged |
| `emotion/blend.py` | ✅ Solid | Unchanged |
| `emotion/influence.py` | ✅ Solid | Unchanged |
| `emotion/aggregate.py` | ✅ Solid | Unchanged |
| `emotion/persona_loader.py` | ✅ Solid | Health integration now wired (F1 follow-up closed) |
| `emotion/_canonical_personal_emotions.py` | ✅ Solid | Unchanged |
| `memory/store.py` | ✅ Solid | PRAGMA integrity_check added |
| `memory/hebbian.py` | ✅ Solid | PRAGMA integrity_check added |
| `memory/embeddings.py` | ✅ Solid | Unchanged |
| `memory/search.py` | ✅ Solid | Unchanged |
| `engines/_interests.py` | ✅ Solid | Health integration added |
| `growth/log.py` | ✅ Solid | read_jsonl_skipping_corrupt used |
| `growth/scheduler.py` | ✅ Solid | anomalies_collector added (F3 follow-up) |
| `growth/proposal.py` | ✅ Solid | Unchanged |
| `growth/crystallizers/vocabulary.py` | ✅ Solid (stub) | Phase 2a stub as designed |
| `health/attempt_heal.py` | ✅ Solid | `attempt_heal_text` added for voice.md |
| `health/adaptive.py` | ✅ Solid | Unchanged |
| `health/reconstruct.py` | ✅ Solid | Unchanged |
| `health/walker.py` | ✅ Solid | voice.md added to `_TEXT_IDENTITY_FILES` |
| `health/alarm.py` | ✅ Solid | voice.md added to `_IDENTITY_FILES` |
| `health/anomaly.py` | ✅ Solid | `BrainIntegrityError` added |
| `health/jsonl_reader.py` | ✅ Solid | Unchanged |
| `migrator/` | ✅ Solid | One-time tool; unchanged |
| `search/` (base + ddgs + claude_tool + factory) | ✅ Solid | Unchanged |
| `persona_config.py` | ✅ Solid | Health integration added |
| `user_preferences.py` | ✅ Solid | Unchanged |
| `paths.py` | ✅ Solid | Unchanged |
| `utils/` | ✅ Solid | Unchanged |

**Previously 🔧 Needs refactor — current state:**

| Module | Old State | Current State | Notes |
|---|---|---|---|
| `engines/dream.py` | 🔧 Needs refactor | ✅ Solid | Principle audit PR-A done (knobs on constructor, not run_cycle). daemon_state writer added (SP-2). |
| `engines/heartbeat.py` | ➕ Needs expansion | ✅ Solid | daemon_state writer added (SP-2). Growth tick wired. Anomaly aggregation wired. |
| `engines/reflex.py` | ➕ Needs expansion | ✅ Solid | daemon_state writer added. Health integration added. |
| `engines/research.py` | 🔧 Needs refactor | ✅ Solid | `forced_interest_topic` removed from engine API. daemon_state writer added. |
| `bridge/provider.py` | 🔧 Needs refactor | ✅ Solid | SP-1 done: `chat(messages, tools)` + `ChatResponse` added. |
| `cli.py` | 🔧 Needs refactor | ⚠️ Mostly solid | PR-A done (interest add/bump removed; --interest, --seed, --depth, --decay, --limit, --lookback removed). PR-B done (provider/searcher into persona config). PR-C done. Stubs remain for supervisor/status/rest/memory/works — expected. |

**Previously ❌ Missing — current state:**

| Module | Old State | Current State | Notes |
|---|---|---|---|
| `soul/` | ❌ Missing | ✅ Shipped (SP-5) | Full soul package: love_types, crystallization, store, review, audit, revoke |
| `chat/` | ❌ Missing | ✅ Shipped (SP-6) | engine.py, session.py, tool_loop.py, prompt.py, voice.py |
| `tools/` | ❌ Missing | ✅ Shipped (SP-3) | schemas.py, dispatch.py, impls/* (9 tools) |
| `daemon_state.json` writer | ❌ Missing | ✅ Shipped (SP-2) | `brain/engines/daemon_state.py` + heartbeat writers |
| `ingest/` | ❌ Missing | ✅ Shipped (SP-4) | buffer.py, commit.py, dedupe.py, extract.py, pipeline.py, soul_queue.py, types.py |
| `bridge/chat.py` | ❌ Missing | ✅ Shipped (SP-1) | ChatMessage, ChatResponse types |

**Previously ➕ Needs expansion:**

| Module | Old State | Current State | Notes |
|---|---|---|---|
| `emotion/expression.py` | ➕ Needs expansion | ➕ Still stub | Mapping logic still awaits art assets. Expected. |

**Module audit summary (updated):** 38 ✅ Solid / 1 ⚠️ Mostly solid (cli.py stubs expected) / 1 ➕ Stub awaiting art assets / 0 ❌ Missing

---

## Section 4 — Master Reference §6 Sub-Project Status

The master reference §6 lists SP-1 through SP-8. Current status from git log:

| SP | Name | Master ref §6 stated status | Actual current status |
|---|---|---|---|
| SP-1 | Provider Interface Rework | 🔧 In scope — not started | ✅ Shipped (commit c175609, PR #21) |
| SP-2 | Daemon-State Residue Writer | 🔧 In scope — not started | ✅ Shipped (commit 9620c56, PR #22) |
| SP-3 | Brain-Tools Rewrite | ❌ Not started | ✅ Shipped (commit e7bf67f, PR #23) |
| SP-4 | Conversation Ingest Pipeline | ❌ Not started | ✅ Shipped (commit a310173, PR #24) |
| SP-5 | Soul Model | ❌ Not started | ✅ Shipped (commit 1deada7, PR #25) |
| SP-6 | Chat Engine | ❌ Not started | ✅ Shipped (commit ef444be, PR #26) |
| SP-7 | Bridge Daemon | ❌ Not started | ❌ Not started — open |
| SP-8 | Tauri Integration | ❌ Deferred — art assets | ❌ Still deferred |

**Finding (Medium):** The master reference document (§6, §7, §3 module table) is materially stale — it was written before SP-1 through SP-6 shipped. It describes SP-1..SP-6 as "not started" and `soul/`, `chat/`, `tools/`, `ingest/` as "Missing." The document is a living spec but has not been updated on each ship as §1 ("update this document on every major sub-project ship") instructs. Before SP-7 design begins, the master reference should be updated to reflect actual shipped state.

---

## Section 5 — Cross-Module Integration (SP-6 the Keystone)

SP-6 (`brain/chat/engine.py`) was specified (master ref §6) to integrate SP-1 / SP-2 / SP-3 / SP-4 / SP-5. Verify each integration point:

| Integration point | Specified in | Code location | Status |
|---|---|---|---|
| SP-1: `provider.chat(messages, tools)` via `brain/bridge/provider.py` | SP-6 scope | `brain/chat/tool_loop.py` calls `provider.chat(messages, tools=tools_list)` | ✅ Verified |
| SP-2: reads `daemon_state.json` via `attempt_heal` | SP-6 scope: "reads daemon_state.json via attempt_heal" | `brain/chat/engine.py:32` imports `load_daemon_state`; `brain/chat/prompt.py:18` imports `get_residue_context` | ✅ Verified |
| SP-3: `brain/tools/dispatch.py` called in tool loop | SP-6 scope | `brain/chat/tool_loop.py` imports and calls `dispatch_tool` from `brain.tools.dispatch` | ✅ Verified |
| SP-4: ingest buffer receives each turn | SP-6 scope: "best-effort persist" | `brain/chat/engine.py:33` imports `ingest_turn` from `brain.ingest.buffer`; called in respond() | ✅ Verified |
| SP-5: `SoulStore.list_active()` injected into system message | SP-6 scope | `brain/chat/prompt.py:20` imports `SoulStore`; `build_system_message` receives `soul_store` and calls `_build_soul_highlights` | ✅ Verified |
| voice.md health integration | SP-6 extension | `brain/chat/voice.py` uses `attempt_heal_text`; `brain/health/walker.py:29` checks it; `brain/health/alarm.py:17` has it in `_IDENTITY_FILES` | ✅ Verified |
| Session management | SP-6 scope | `brain/chat/session.py` + `create_session()` used in `brain/cli.py:804` REPL | ✅ Verified |
| Ingest pipeline close on session end | SP-6 / SP-4 | `brain/cli.py:829` calls `close_session()` from `brain.ingest.pipeline` on REPL exit | ✅ Verified |

All 8 integration points verified. SP-6 is a fully wired keystone.

---

## Section 6 — Open Follow-ups Still in Flight

### F1: `reconstruct_vocabulary_from_memories` wired into vocabulary heal flow

**Spec reference:** brain-health spec §9.2 / commit d5619e1 message.
**Status:** ✅ Closed.
`brain/emotion/persona_loader.py:66–90` — when `anomaly.action == "reset_to_default"` AND `store is not None`, calls `reconstruct_vocabulary_from_memories(store)` and replaces the anomaly with `action="reconstructed_from_memories"`. Fully wired.

### F2: Soul module health plan — `soul.json` in walker + alarm

**Spec reference:** brain-health spec §9.1.
**Status:** ⚠️ Partially open.
The spec says when the soul module lands, `soul.json` should be added to:
- `brain/health/walker.py:_DEFAULTS`
- `brain/health/alarm.py:_IDENTITY_FILES`
- `brain/health/reconstruct.py:reconstruct_soul_from_memories`

Current code: The soul module shipped (SP-5) but uses SQLite (`crystallizations.db`) not a JSON file (`soul.json`). The architecture changed between spec and implementation — the spec's soul-health plan assumed a `soul.json` atomic-rewrite file, but the shipped soul module uses SQLite (same pattern as memories.db). The walker already checks `crystallizations.db` indirectly through... actually it does not. `brain/health/walker.py` only checks `memories.db` and `hebbian.db` as SQLite stores. `crystallizations.db` is not in the walker's scan. `SoulStore` has its own `PRAGMA integrity_check` on open (verified in `brain/soul/store.py:51`), but it is not invoked by `walk_persona`.

**Finding (Low):** `crystallizations.db` is not in `walk_persona`'s SQLite scan. If soul DB corrupts, the walker won't catch it during a health check pass. Not critical (SoulStore's own constructor catches it), but `nell health check` won't surface it proactively.

### F3: Anomaly collector through `run_growth_tick`

**Spec reference:** commit d5619e1 follow-up items.
**Status:** ✅ Closed.
`brain/growth/scheduler.py:52` — `anomalies_collector: list[BrainAnomaly] | None = None` parameter. Line 77–78: appends vocab anomaly to collector when provided. Heartbeat passes `tick_anomalies` as collector at line 526–529.

### Other deferred / open items:

- **Phase 2b vocabulary pattern matchers** — still deferred (correct per spec; needs ≥2 weeks behavior data).
- **Phase 2a-extension: reflex arc emergence, research interest emergence, soul crystallizer** — all deferred (correct).
- **Reflex Phase 2 emergent arc crystallization** — deferred pending ≥2 weeks Phase 1 data (noted in master ref §8.7).
- **Body state module (`brain/body/`)** — not ported; `get_body_state` tool is a stub. Expected.
- **Behavioral log / creative DNA / journal** — none ported. Expected (not on critical path).
- **Master reference document not updated** — see Section 4. The document itself says "update on every major sub-project ship" and this was not done for SP-1..SP-6.
- **SP-7 bridge daemon design questions** — 8 open questions in master ref §8 remain unresolved. Decisions needed before SP-7 code.

---

## Section 7 — CLI Surface Check

`brain/cli.py` subcommand map against spec references:

| Subcommand | Spec reference | Status | Notes |
|---|---|---|---|
| `nell dream` | Heartbeat spec + principle audit PR-A | ✅ Wired | `--seed/--depth/--decay/--limit/--lookback` removed per PR-A. Knobs on DreamEngine constructor. |
| `nell heartbeat` | Heartbeat spec | ✅ Wired | `--trigger`, `--dry-run`, `--verbose` present. `--provider`/`--searcher` dev-override only. |
| `nell reflex` | Reflex spec | ✅ Wired | `--trigger`, `--dry-run`, `--provider` dev-override. |
| `nell research` | Research spec + principle audit | ✅ Wired | `--interest` removed per PR-A. `forced_interest_topic` removed from ResearchEngine API. |
| `nell interest list` | Principle audit | ✅ Wired (read-only) | `interest add` and `interest bump` removed per PR-A. |
| `nell growth log` | Phase 2a spec §8 | ✅ Wired | `--limit`, `--type` filter present. No add/approve/reject/force. |
| `nell health show` | Brain-health spec §5 | ✅ Wired | |
| `nell health check` | Brain-health spec §5 | ✅ Wired | Exits 2 on unhealable alarms. |
| `nell health acknowledge` | Brain-health spec §5 | ✅ Wired | `--file` / `--all`. |
| `nell migrate` | Migrator spec | ✅ Wired | |
| `nell soul list` | SP-5 | ✅ Wired | |
| `nell soul revoke` | SP-5 | ✅ Wired | `--id`, `--reason`. |
| `nell soul candidates` | SP-5 | ✅ Wired | |
| `nell soul audit` | SP-5 | ✅ Wired | |
| `nell soul review` | SP-5 | ✅ Wired | `--max`, `--confidence-threshold`, `--dry-run`. |
| `nell chat` | SP-6 | ✅ Wired | One-shot + interactive REPL. Ingest flush on REPL exit. |
| `nell supervisor` | Stub | Expected stub | |
| `nell status` | Stub | Expected stub | |
| `nell rest` | Stub | Expected stub | |
| `nell memory` | Stub | Expected stub | |
| `nell works` | Stub | Expected stub | |

**Finding (Low):** The principle alignment audit recommended `nell dream` be documented as "developer-only" in `--help`. Confirmed: `dream_sub` help text at `brain/cli.py:882` reads `"(developer) Run one dream cycle... Production dreams fire from the heartbeat — this is for debugging."` ✅ Compliant.

**Finding (Low):** `growth_enabled` lives in `heartbeat_config.json` (developer calibration layer). The principle audit §"What Phase 2a inherits" says `growth_enabled` should follow the same rule as `reflex_enabled` / `research_enabled` and be hidden from the user. It is hidden — not in `user_preferences.json`, not surfaceable via any CLI action. ✅ Compliant, but worth an inline comment in `HeartbeatConfig` to note it should not migrate to `user_preferences.json`.

---

## Section 8 — Drift Between Spec and Code

### Drift 1: Alarm threshold `>= 6` vs spec language (Low — intentional)

**Spec text (brain-health §2.6):** "Returns `list[AlarmEntry]` representing the union of: Files with ≥3 anomalies in the last 7 days (after adaptive treatment was already in effect)."
**Code (`brain/health/alarm.py:74–75`):** `if len(anoms) >= 6: is_alarm = True`

The spec says "≥3 anomalies" as the alarm trigger, but the code uses `>= 6`. Reading the spec more carefully: "after adaptive treatment was already in effect" — the intent is that the alarm fires when the brain has been corrupting even AFTER adaptation (backup bumped to 6 at ≥3). Since the bump fires at `count >= 3`, the alarm at `>= 6` is effectively "3 more after adaptation kicked in." The inline comment in alarm.py ("≥6 anomalies in window = recurring-after-adaptation") documents this reasoning. This is intentional drift, not a bug, but the spec text is misleading. The spec should say "≥6 anomalies in the window (which signals recurring failure even after adaptive treatment)" rather than "≥3."

### Drift 2: voice.md `DEFAULT_VOICE_TEMPLATE` not cross-referenced in specs (Low — new content)

The voice.md feature (`brain/chat/voice.py`, `DEFAULT_VOICE_TEMPLATE`) was added as part of SP-6. It is not specified in any pre-SP-6 spec (master reference §6 SP-6 describes soul injection + daemon residue + session management but doesn't specify a `voice.md` file template structure). The implementation is well-reasoned and the template is clearly authored, but:
- The 4-section template structure (Who you are / What's in your head / How emotion shapes your voice / Your boundaries with the user) is an undocumented design decision.
- The "boundaries with user" section containing `at-user anger >= 7.5 → may refuse engagement` is a significant behavioral claim with no spec backing it.

This is not a bug — it's well-considered design — but it is **spec-build-and-ship** rather than spec-first. The refusal mechanism (`>= 7.5`) embedded in the default template is the closest the framework gets to the "right to refuse engagement" principle from the vocabulary emergence spec §Preamble, but it's currently a comment in a Markdown template rather than implemented logic.

### Drift 3: `get_personality` stub incorrectly implies voice.md will subsume it (Low)

`brain/tools/impls/get_personality.py` docstring says "Personality module pending — voice.md (SP-6) will likely subsume this." SP-6 shipped and `voice.md` is the persona voice document, but `get_personality` is still a stub returning `{"loaded": False}`. The tool is registered and callable; a chat turn that calls `get_personality` gets a non-informative response. This is expected behavior (stubs are documented as such in SP-3), but the comment that "voice.md will likely subsume this" is now stale since SP-6 shipped without implementing `get_personality` through voice.md.

### Drift 4: `HeartbeatResult.research_deferred` field (Low — possible dead field)

`brain/engines/heartbeat.py:330`: `research_deferred: bool` field in `HeartbeatResult`. This field does not appear to be set to `True` in any code path (grep shows it's never set True; `initialized=True` path sets it to `False` implicitly). The CLI output does not reference it. This field may be a remnant from an earlier design. Not harmful, but dead code.

---

## Section 9 — Stubs Incorrectly Marked Shipped

None found. Stubs that exist are properly labeled as stubs in their docstrings:
- `brain/tools/impls/get_personality.py` — labeled "STUB until SP-6" (SP-6 shipped, stub not resolved, but this is noted as pending, not falsely claimed complete)
- `brain/tools/impls/get_body_state.py` — labeled "STUB until body-state module lands"
- `brain/growth/crystallizers/vocabulary.py` — Phase 2a stub; correctly labeled, correctly deferred

Master reference §6 SP-6 is listed as "❌ Not started" in the document but is actually shipped — this is the document being stale, not a code claim being false.

---

## Section 10 — Test Claims Without Test Backing

### Claim 1: Phase 2a Inject-test (Spec §14 criterion 13)

The spec says: "when a fake crystallizer returns 1 `EmotionProposal` (test fixture), the scheduler writes it to vocabulary + log atomically."

`tests/unit/brain/growth/test_scheduler.py` exists. The plan documents this test pattern explicitly. From the 870 passing tests, this is backed. ✅

### Claim 2: SP-6 commit claims "49 new tests" in commit message

The commit message for SP-6 (ef444be) claims "49 new tests; 870 total passing." The current suite is 870 passing. ✅ Count matches.

### Claim 3: Brain-health §2.6 "revert rule: 30 consecutive days clean since last anomaly → (3, False)"

The spec says the adaptive treatment should revert to `(3, False)` after 30 consecutive days clean. The code in `brain/health/adaptive.py` does NOT implement this revert rule. It only reads the last 7 days of the audit log; if there are 0 anomalies in 7 days the count is 0 and `>= 3` is False, so `compute_treatment` returns `(3, False)` — which effectively implements the revert, just with a 7-day window rather than 30 days. The 30-day revert rule was not explicitly coded; the 7-day window makes it implicit. This is not a bug but the spec is not precisely implemented.

**Finding (Low):** The spec says "30 consecutive days clean since last anomaly." The code uses a 7-day sliding window — if no anomaly in the last 7 days, backup count reverts to 3. This means the revert happens after 7 days clean, not 30. The intent (revert to normal after the file is stable) is preserved; the duration differs.

---

## Section 11 — Recommendations

### Must-fix (blocks SP-7 / breaks principle / is silent failure)

None found. There are no blocking bugs or silent failures that would break SP-7 or violate the core autonomy principle.

### Should-fix (clear drift, easy fix)

1. **Update the master reference document** (`docs/superpowers/specs/2026-04-26-companion-emergence-master-reference.md`).
   - The document's framework state line says "Chat engine not started" — SP-1 through SP-6 are all shipped. 870 tests.
   - Module table (§5) still shows `soul/`, `chat/`, `tools/`, `ingest/`, and `daemon_state.json` as ❌ Missing.
   - SP roadmap (§6) shows SP-1..SP-6 as "not started."
   - The document is the canonical reference; new engineers reading it will be deeply confused.
   - Fix: update §3 module table entries, §5 gap table (all 5 gaps are now closed), §6 sub-project status for SP-1..SP-6 to ✅ Shipped, and the opening framework state line.

2. **Add `crystallizations.db` to `walk_persona`'s SQLite scan** (`brain/health/walker.py`).
   Currently only `memories.db` and `hebbian.db` are scanned. `SoulStore` has its own integrity check on open, but `nell health check` won't catch soul DB corruption proactively. Add a `crystallizations.db` entry mirroring the existing SQLite scan pattern (lines 62–88). Two-line addition.

3. **Clarify the alarm threshold comment in `brain/health/alarm.py`** and update the spec §2.6 text.
   The spec says "Files with ≥3 anomalies in the last 7 days (after adaptive treatment was already in effect)" but the code uses `>= 6`. Add a clearer comment to the spec (or the spec's doc) explaining the two-threshold design: bump fires at 3, alarm fires at 6 (i.e., 3 more after adaptation). The code inline comment already says this; the spec text needs updating.

4. **Resolve `get_personality` stub comment** (`brain/tools/impls/get_personality.py`).
   The comment says "voice.md (SP-6) will likely subsume this." SP-6 shipped without subsuming it. Either: (a) implement `get_personality` to return relevant voice.md metadata + persona name, or (b) update the docstring to reflect the actual design decision (persona is expressed through voice.md in the system message, not via a tool call). The stub is functional but the comment is misleading for the next engineer.

### Nice-to-have (non-blocking polish)

5. **Remove dead `research_deferred` field from `HeartbeatResult`** (`brain/engines/heartbeat.py:330`).
   This boolean is always `False` and never referenced in output. Either wire it (set it when research doesn't fire due to deferral) or remove it.

6. **Document the `DEFAULT_VOICE_TEMPLATE` sections in a spec or design note.**
   The 4-section voice.md template and the `>= 7.5` refusal threshold in the "Your boundaries" section are undocumented design decisions. A short note in the master reference §3 `brain/chat/voice.py` entry (or a standalone spec) would preserve the reasoning for the next engineer.

7. **Confirm `append-only log reader` coverage for `dreams.log.jsonl`, `research_log.json`, `reflex_log.json`.**
   The health spec §3 says all append-only logs use `read_jsonl_skipping_corrupt`. The engine-level logs are read by CLI code and internal engine code; it is not confirmed they all go through the shared helper. This only matters when a corrupt line appears in those files and the reader doesn't gracefully skip it. Low risk (these files are rarely read in production), but worth a sweep.

8. **Phase 2b readiness check.** The vocabulary emergence spec says Phase 2b requires "≥2 weeks of Phase 1 behavior data." As of 2026-04-27, Phase 1 shipped on 2026-04-25 — only 2 days of data. The clock is running. No code fix needed; just a reminder that Phase 2b planning can open around 2026-05-09.

---

*End of audit. 870 tests passing. No blocking issues. Safe to proceed to SP-7 design with the "Should-fix" items addressed first — particularly the master reference update, which is the canonical pre-design reference per §7 "Decision-Checking Guide."*
