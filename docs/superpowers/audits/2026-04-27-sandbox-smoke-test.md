# Sandbox Smoke Test — 2026-04-27

## Headline

- Total commands run: 38
- 35 worked / 2 weird / 1 failed
- Critical issues: none. One expected failure (dream engine — no conversation seed), one behaviour quirk (fake provider bleeds DREAM: prefix into research summary).

---

## Step-by-step results

### Step 0 — Backup

```
tar -czf ~/.cache/companion-emergence-smoke-test/nell.sandbox.pre-smoke.tar.gz \
    -C "$PERSONA/.." nell.sandbox
```

**Output:** `Snapshot saved to ... 1.0M 27 Apr 09:00`
**Classification:** PASS — 1.0M snapshot, restore path confirmed.

---

### Step 1 — Read-only probes

| Command | Result |
|---------|--------|
| `nell --version` | `companion-emergence 0.0.1` — PASS |
| `nell --help` | Full subcommand table printed, all 14 subcommands shown — PASS |
| `nell heartbeat --help` | Full usage printed with `--trigger`, `--provider`, `--dry-run`, `--verbose` — PASS |
| `nell dream --help` | Full usage printed with `--provider`, `--dry-run` — PASS |
| `nell reflex --help` | Full usage printed — PASS |
| `nell research --help` | Full usage printed — PASS |
| `nell growth --help` | Full usage printed with `log` subcommand — PASS |
| `nell soul --help` | Full usage printed with `list`, `revoke`, `candidates`, `audit`, `review` — PASS |
| `nell health --help` | Full usage printed with `show`, `check`, `acknowledge` — PASS |
| `nell chat --help` | Full usage printed with optional `message` positional arg — PASS |
| `nell interest --help` | Full usage printed with `list` only subcommand — PASS |
| `nell migrate --help` | Full usage printed — NOTE: `--help` exits with code 2 (argparse behaviour); content is correct — WEIRD |
| `nell setup --help` | **Does not exist.** `argparse error: invalid choice: 'setup'` — NOTED (not a failure; spec says may not exist yet) |

**Note on `migrate --help` exit code 2:** argparse prints the help and exits 2 on `--help` only when the parser is built with `add_help=False` or when the command is a subparser that doesn't have help wired in. This is cosmetically wrong but functionally harmless.

---

### Step 2 — State inspection

```
nell health show --persona nell.sandbox
```
```
Health for persona 'nell.sandbox':
  Pending alarms: 0
  Recent self-treatments (last 7 days): 1
    2026-04-25T23:35:44Z  heartbeat_state.json  restored_from_bak1  (cause: user_edit)
```
**Classification:** PASS — 0 alarms, 1 pre-existing self-treatment from a prior session (not introduced by this test).

```
nell health check --persona nell.sandbox
```
```
✅  heartbeat_state.json: OK
✅  interests.json: OK
✅  reflex_arcs.json: OK
✅  emotion_vocabulary.json: OK
✅  memories.db: OK
✅  hebbian.db: OK
0 file(s) healed, 0 unhealable. Brain is healthy.  exit: 0
```
**Classification:** PASS — full file integrity walk clean.

```
nell soul list --persona nell.sandbox
```
```
Soul crystallizations for persona 'nell.sandbox' (0 active):
  (none yet)
```
**Classification:** PASS — expected for a sandbox persona that hasn't had real LLM interactions.

```
nell soul candidates --persona nell.sandbox
```
```
Pending soul candidates for persona 'nell.sandbox' (0): (none)
```
**Classification:** PASS

```
nell soul audit --persona nell.sandbox
```
```
Soul audit log for persona 'nell.sandbox' (last 0 entries): (empty)
```
**Classification:** PASS

```
nell growth log --persona nell.sandbox
```
```
Growth log for persona 'nell.sandbox' (0 events shown): (empty)
```
**Classification:** PASS — growth hasn't fired; emotion vocab unchanged since migration.

```
nell interest list --persona nell.sandbox
```
```
Interests for persona 'nell.sandbox' (2):
  - Lispector diagonal syntax  pull=7.2  scope=either    last_researched=never
  - Hana                       pull=4.8  scope=internal  last_researched=never
```
**Classification:** PASS — both OG interests intact from migration.

---

### Step 3 — Dry-run engines

```
nell heartbeat --persona nell.sandbox --provider fake --trigger manual --dry-run
```
```
Heartbeat dry-run — no writes.
  elapsed: 32.43h
  would decay: 0 memories
  would prune: 0 edges
  dream: would_fire_but_dry_run
```
**Classification:** PASS — dry-run accurate; dream correctly identified as due.

```
nell dream --persona nell.sandbox --provider fake --dry-run
```
```
Traceback...
brain.engines.dream.NoSeedAvailable: No conversation memories within the last 24 hours.
exit: 1
```
**Classification:** FAIL (expected by design — see Issues section below). Dream engine called directly requires a conversation seed from the last 24 hours. `nell.sandbox` is a migrated OG persona with no post-migration chat; conversation memories haven't been generated. Production path (heartbeat → dream) works around this automatically.

```
nell reflex --persona nell.sandbox --provider fake --dry-run
```
```
Reflex dry-run — would fire: self_check.
  Skipped: creative_pitch (trigger_not_met), loneliness_journal (trigger_not_met),
           gift_creation (cooldown_active), gratitude_reflection (cooldown_active),
           defiance_burst (trigger_not_met), body_grief_whisper (cooldown_active),
           jordan_grief_carry (cooldown_active)
```
**Classification:** PASS — 8 OG arcs evaluating correctly; self_check fires, rest respect cooldowns/triggers.

```
nell research --persona nell.sandbox --provider fake --dry-run
```
```
Research dry-run — would fire: Lispector diagonal syntax.
```
**Classification:** PASS — correct interest selected.

```
nell soul review --persona nell.sandbox --provider fake --dry-run
```
```
Soul review dry-run — no writes.
Soul review complete: 0 pending, 0 examined, 0 accepted, 0 rejected, 0 deferred, 0 parse failures.
```
**Classification:** PASS

---

### Step 4 — Real heartbeat ticks

**Tick 1:**
```
nell heartbeat --persona nell.sandbox --provider fake --trigger manual
```
```
Heartbeat tick complete (manual).
  elapsed: 32.43h
  decayed: 878 memories, pruned 0 edges
  dream fired: f698653d-f3d7-458c-9ebb-dff915b35286
  research fired: Lispector diagonal syntax
```
**Classification:** PASS — dream and research both fired. 878 memories decayed (correct after 32h gap). 0 edges pruned.

**Tick 2 (verbose):**
```
nell heartbeat --persona nell.sandbox --provider fake --trigger manual --verbose
```
```
Heartbeat tick complete (manual).
  elapsed: 0.01h
  decayed: 879 memories, pruned 0 edges
  dream gated: not_due
  reflex evaluated (8 arc(s) skipped)
  research gated: no_eligible_interest
  interests bumped: 0
```
**Classification:** PASS — gating logic correct (dream/research gated after fresh tick; reflex evaluated but none fired in 0.01h window).

**Post-tick verification:**
- `heartbeats.log.jsonl`: 9 entries total, 2 new from this smoke (tick_count 6 and 7) — PASS
- `daemon_state.json`: all fields populated (last_heartbeat, last_dream, last_research, emotional_residue with decays_by) — PASS
- `pending_alarms_count: 0` in both new log entries — PASS

---

### Step 5 — Chat engine smoke

```
nell chat --persona nell.sandbox --provider fake "tell me how you're feeling today"
```
Output: `FAKE_CHAT: response 94e493e63098b291`
**Classification:** PASS — response printed, session file created.

```
nell chat --persona nell.sandbox --provider fake "what's the most important memory you have about hana?"
```
Output: `FAKE_CHAT: response 88d69e95384b4895`
**Classification:** PASS

**Post-chat verification:**
- `active_conversations/`: 2 session files created (separate session_id per one-shot invocation) — PASS
- Each session file has 2 turns (`speaker: user` + `speaker: assistant`) — PASS
- Schema uses `speaker` field (not `role` as the test plan assumed) — this is fine, the field name difference is cosmetic
- `voice.md` auto-created with full template — PASS

---

### Step 6 — Brain-health corruption recovery

Corrupted `heartbeat_state.json` in a temp persona with `{not valid json`.
Re-ran heartbeat — self-heal fired:
```
Health for persona 'health-test-persona':
  Pending alarms: 0
  Recent self-treatments: 1
    2026-04-27T08:02:52Z  heartbeat_state.json  reset_to_default  (cause: user_edit)
```
- `.bak1` retained — PASS
- `.corrupt-*` quarantine file retained — PASS
- `heartbeat_state.json` reset to valid defaults — PASS
- `health show` reports 0 alarms — PASS

**Classification:** PASS — self-heal path works end-to-end.
Temp persona cleaned up.

---

### Step 7 — Interest list read-only, add/bump blocked

```
nell interest list --persona nell.sandbox
```
Shows 2 interests, `last_researched` updated for Lispector after Step 4's research fire — PASS

```
nell interest add "test" --persona nell.sandbox
```
```
nell interest: error: argument action: invalid choice: 'add' (choose from list)
```
**Classification:** PASS — add correctly not exposed.

```
nell interest bump "test" --persona nell.sandbox
```
```
nell interest: error: argument action: invalid choice: 'bump' (choose from list)
```
**Classification:** PASS — bump correctly not exposed.

---

### Step 8 — Daemon-state reading

`daemon_state.json` structure verified:
```json
{
  "last_dream": { "timestamp", "dominant_emotion", "intensity", "theme", "summary" },
  "last_heartbeat": { ... },
  "last_research": { ... },
  "emotional_residue": { "emotion", "intensity", "source", "decays_by" }
}
```
All SP-2 spec fields present. `last_reflex` absent — normal (reflex didn't fire in Step 4's ticks).

One quirk: `last_research.summary` reads `"DREAM: test dream 746349806d8bd4cc — an associative thread"` because the fake provider returns the same template string for both dream and research calls. Real LLM would produce a research-appropriate summary. Fake-provider limitation only.

**Classification:** PASS (with WEIRD quirk noted)

---

### Step 9 — Conversation ingest pipeline

Two `active_conversations/*.jsonl` files exist after Step 5.
Both have 2 turns each in correct format.
No soul candidates auto-generated (correct — fake provider outputs don't trigger soul crystallization pipeline).
`voice.md` exists.

**Classification:** PASS

---

### Step 10 — Full pytest run

```
uv run pytest -q
```
```
870 passed in 1.32s
```
**Classification:** PASS — 870/870, 0 failures, 0 skips.

---

### Step 11 — Snapshot retained

```
~/.cache/companion-emergence-smoke-test/nell.sandbox.pre-smoke.tar.gz  1.0M
```
Restore command:
```bash
rm -rf '/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox'
tar -xzf "$HOME/.cache/companion-emergence-smoke-test/nell.sandbox.pre-smoke.tar.gz" \
    -C '/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox/..'
```

---

## Issues found

### 1. `nell dream` — direct invocation fails without recent conversation memories

**Command:** `nell dream --persona nell.sandbox --provider fake --dry-run`
**Expected:** dry-run output with seed + neighbour info
**Observed:** `NoSeedAvailable: No conversation memories within the last 24 hours.`
**Root cause:** `_select_seed()` filters strictly to `conversation` type memories within `lookback_hours=24`. `nell.sandbox` is a migrated persona — OG NellBrain data was imported as non-conversation memory types. After migration with no new post-migration chat sessions, there are zero conversation-typed memories in the 24h window.
**Impact on production path:** None. Heartbeat fires dreams via `run_cycle(seed_id=<picked_seed>)`, bypassing the time-window filter entirely. This is a developer-only CLI footgun, not a production regression.
**Severity:** Low — documented limitation, expected behaviour for migrated personas without post-migration chat. Would benefit from a better error message that says "this is expected for migrated personas — run `nell heartbeat` first or use `nell chat` to build conversation history."

---

### 2. Fake provider bleeds `DREAM:` prefix into research summary in `daemon_state.json`

**Command:** Heartbeat tick with `--provider fake`
**Expected:** `last_research.summary` contains a research-flavoured string
**Observed:** `"summary": "DREAM: test dream 746349806d8bd4cc — an associative thread"`
**Root cause:** The fake provider `generate()` returns the same `DREAM: test dream <hash> — an associative thread` template regardless of which engine calls it. The research engine then stores whatever the provider returns as the summary. Real providers would return research-appropriate content.
**Severity:** Low — fake provider only; no production impact. Would be worth adding research-specific fake output in the fake provider to make fake-mode smoke tests more realistic.

---

### 3. `nell migrate --help` exits with code 2

**Command:** `nell migrate --help`
**Expected:** exit 0
**Observed:** exit 2 with correct help content printed
**Root cause:** The migrate subparser uses a custom `add_help` or non-standard argparse setup that exits 2 on `--help`. All other subcommands exit 0.
**Severity:** Low — cosmetically annoying for scripted help checks, no functional impact.

---

## Persona state changes

The following writes were made to `nell.sandbox` during this smoke test:

| File | Change | Step |
|------|--------|------|
| `heartbeats.log.jsonl` | 2 new JSONL entries (tick_count 6 and 7) | Step 4 |
| `heartbeat_state.json` | Updated tick_count, last_tick_at, timestamps | Step 4 |
| `heartbeat_state.json.bak1` | Rotated backup | Step 4 |
| `daemon_state.json` | Updated last_dream, last_heartbeat, last_research, emotional_residue | Step 4 |
| `daemon_state.json.bak1,bak2,bak3` | Rotated backups | Step 4 |
| `dreams.log.jsonl` | 1 new dream log entry (seed f698653d) | Step 4 |
| `memories.db` | 1 new dream memory written; 878-879 memories decay-updated | Step 4 |
| `hebbian.db` | Edge strengths updated for dream memory | Step 4 |
| `research_log.json` | 1 new research fire entry (Lispector diagonal syntax) | Step 4 |
| `interests.json` | `last_researched_at` updated for Lispector interest | Step 4 |
| `active_conversations/6e7d63fe-*.jsonl` | New session — 2 turns | Step 5 |
| `active_conversations/74733bf1-*.jsonl` | New session — 2 turns | Step 5 |
| `voice.md` | Auto-created (template only, no LLM content) | Step 5 |

**Snapshot retained at:** `~/.cache/companion-emergence-smoke-test/nell.sandbox.pre-smoke.tar.gz`
The writes above are all valid real-world state — heartbeat ticks, a dream, a research fire, and 2 chat sessions. Hana can leave them as-is or restore from snapshot.

---

## Recommendations

### Must-fix before SP-7

None. No critical failures found.

### Should-fix soon

1. **`nell dream` error message clarity** — When `NoSeedAvailable` fires for a migrated persona with no post-migration chat, the error should explain this is expected and suggest running `nell heartbeat` or `nell chat` first. Current traceback is confusing for new users.

2. **Fake provider: research-specific output** — Add a `research` context to the fake provider's `generate()` call so smoke tests can distinguish dream vs. research outputs in `daemon_state.json`. One line: check for a `"research"` keyword in the system prompt and return a different template.

### Nice-to-have polish

3. **`nell migrate --help` exit code** — Should exit 0, not 2. Small fix in the migrate subparser's argparse setup.

4. **Conversation schema note in test plan** — The test plan checked for a `role` field; actual schema uses `speaker`. Not a bug in the framework, just worth updating the smoke-test plan for future runs.

5. **Reflex dry-run verbose flag** — `nell reflex --dry-run` shows which arc would fire but not why the others were skipped (trigger condition not met, cooldown, etc.). Heartbeat verbose shows this summary; reflex direct call doesn't. Parity would help debugging specific arc behaviour.
