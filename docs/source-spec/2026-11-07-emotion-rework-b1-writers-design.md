# Emotion rework: B1 writers (design)

Umbrella: #240. Parent: `2026-10-28-emotion-rework-hana-half-design.md`, whose §4 B1 items this spec replaces. Fixes #125 (closes when Engine 6 merges).

Verified against `origin/main` @ `c9589209` on 2026-11-07. Line numbers are from that commit. The engine half is merged through Readers 1 (#632) and stays off until Engine 6 (#633) writes `state/emotion_switch.json`.

## 1. Outcome

Every writer feeds the engine honestly once the switch is on: signed changes can lower a feeling, every stored value is on the 0-10 scale, peers count in proportion to real trust, research carries its own feelings and rating, and per-memory reads use faded values.

**With the switch off, nothing she feels changes.** Every writer keeps today's path, and tests pin that.

Done means: all seven PRs are merged with their tests, the `test_intended_*` pins from #671 are rewritten against the live path, and the full gate is green (`uv run pytest`, ruff, `pnpm test`, `pnpm build`).

## 2. Decisions (Hana, 2026-11-07)

1. **Faded values:** every per-memory emotion read that B1 touches uses `faded_value`.
2. **Raw trust:** `peer_trust` returns `trust_score` with no relationship-stage cap (§6 records the risk).
3. **Inline gating:** each writer checks `is_live()` itself. There is no shared gate helper.
4. **Seven PRs:** the parent's five, with `peer_trust` split out as its own first PR, plus a cleanup PR after Engine 6.

## 3. Rules every PR follows

**Gating.** Production code calls `engine.engine_for(persona_dir).is_live()`. Pure functions take `live: bool = False`, as `brain/engines/reflex.py:473`, `self_model/ambient.py:74` and `chat/prompt.py:2133` already do. A long-running operation reads the switch once at its top. `persona_dir=None` counts as off.

**Faded reads.** Pure functions that read per-memory values take `eng` with no default, following salience's `_emotion_input`; `None` means use stored values. Iterate the memory's own keys, because `faded_value` raises `KeyError` for an absent emotion.

**Tests.**
- Live tests use the real engine and a real switch file in a temp `KINDLED_HOME`, and run through the real entry point, not the helper.
- Each PR pins the off path against today's behaviour.
- Each PR removes the `xfail` from the `test_intended_*` pins it fixes and rewrites them with the switch on. The `test_today_*` pins stay as the switch-off record.
- Tests that pass on base carry `no_base_failure(reason=...)`.

## 4. The PRs

### PR 1: `peer_trust`

New `peer_trust(persona_dir, peer_id) -> float` in `kindled_link/relationship.py`:

1. If `kindled_db_path(persona_dir)` doesn't exist, return 0.0. Reading must never create the database.
2. Open `KindledLinkStore(path, integrity_check=False)`, read `get_relationship_state(store, peer_id).trust_score`, and close.
3. A missing row gives 0.0, the dataclass default. `sqlite3.Error` logs a warning and returns 0.0.

There is no caching, marked with a `shortcut:` comment, because it runs once per peer row under the engine lock. ThinkerOfThoughts swaps out the stand-in `engine_files.peer_trust` (`:270`) in Engine 6.

Tests: no database (and none created), missing row, real row, corrupt database.

### PR 2: signed changes

Files: `chat/extractor.py` `_apply_emotion_delta` (`:360`), `self_model/reconcile.py` `_write_self_authored_delta` (`:94`).

New `signed_change_values(delta)` in the extractor: `sign(v) × min(|v| × 10, 3.0)`, skipping zeros. Reconcile imports it.

Switch on: `fold_change(values, None, source)` with sources `monologue_emotion` and `self_model_reconcile`, and no memory row. In reconcile, `True` means folded. Switch off: today's path.

The existing accepted repeat when pass 2 processes an entry again stays as it is.

Tests, through the real pass-2 and reconcile entry points: a negative value lowers `read()[name].short`, and no row is written. Flips the two sign pins. #671 must merge first. The PR says `Refs #125`.

### PR 3: nudges and peer memories

**Nudges, ×10 when live:**
- `maker/wiring.py` `making_emotion_delta(..., live=)`, called at `maker/making_runner.py:62`.
- `initiate/reach_emotion.py` `reach_emotions_for(source, live=)`, called at `initiate/review.py:340`.
- `notes/runner.py:141`, inline (0.12 becomes 1.2).

**`run_relationship_reflection` (`kindled_link/relationship.py:156`):**
- Read `is_live()` once at the top. The production caller (`kindled_link/tick.py:354`) passes `persona_dir`.
- A second version of the prompt's last segment asks for 0-10 values. The old text stays byte-identical (`prompt_strings.toml`, `[kindled_link.relationship]`).
- Live values: drop negative and non-finite values, and hold the rest to 10.
- `apply_peer_emotion` isn't called on the live path, so the bucket stops applying.
- `hard_breach` still returns early.

Old rows and pending-queue rows keep their small values, with no migration.

Tests: each writer through its real entry point. A stubbed reflection returning `{"joy": 9, "grief": -3, "love": 40}` stores joy 9 and love 10 and drops grief. The live prompt asks for 0-10. Off pinned. Flips the two peer pins. Prompt-string tests may need the new segment.

### PR 4: research

`parse_session_output` (`research_session.py:106`) isn't touched. New `extract_felt_lines(raw) -> (rest, emotions, rating)` in `research_session.py` runs first on the live path. It walks back from the end of the reply over one-line markers only, accepting bold and underscore styling.

A second version of `render_system_prompt_segments` asks for the three existing markers, then:

```
EMOTIONS: {"<emotion name>": <0-10>}
ENGAGEMENT: <0-10>
```

This is working text; ThinkerOfThoughts may edit the rating line (their design, item 11).

- **Switch:** `is_live()` once per fire (`engines/research.py:203`).
- **Emotions:** a JSON object. Drop unregistered, zero, negative and non-finite values, and hold the rest to 10.
- **Rating:** the starting number (`7/10` gives 7). A negative is treated as missing. Held to 10.
- **Storage:** `build_research_memory` (`:616`) gains optional `emotions` and `engagement_rating`. The metadata key `engagement_rating` is written only when present.
- **Failure:** a missing or broken line never fails the fire. The memory is written and `record_parse_failure(site="research_felt")` logs it.
- `research_recovery.py` is unchanged.

Tests: a table of reply shapes, including `Engagement: 7` in the middle of the notes (not taken). A real fire with a stub provider and the real engine. A live reply with only the three markers. Off pinned.

### PR 5: dreams, grief, growth

**Dream row (`engines/dream.py` `_write_dream_memory`, `:241`).** Live: for each emotion, the mean of the faded values across the seed and neighbours that carry it, held to 10. Off or `persona_dir=None`: today's sum. The emotions snapshot at `dream.py:342` gets the mean too. Flips the dream pin.

**Mood (`dream.py:172`).** Live: `EmotionalState(emotions={name: r.short ...})`, as `research.py:213` does. This widens mood to all counted sources. A breadcrumb still can't set its own mood, because `grief_event` isn't counted. An empty read during a rebuild is accepted.

**Seeds (`engines/dream_seeds.py`).** `emotional_congruence` (`:27`) and `grief_pull` (`:94`) take `eng` through `composite_score` (`:108`) and `select_seed` (`:128`). No default, so existing tests gain `eng=None`. `DreamEngine` passes the engine when it has a `persona_dir`.

**Grief (`grief/__init__.py` `handle_drop`, `:53`).** Intensity and residue use faded values, inside the existing `try`. This runs after `hard_delete` (`forgetting/__init__.py:288`), so `faded_value` must work from the memory object alone (§8).

**Growth.**
- `_aggregate_recent_emotion_vector` in `growth/scheduler.py` (`:285`), `crystallizers/reflex.py` (`:504`) and `crystallizers/creative_dna.py` (`:468`) take `persona_dir`. Live, they return short averages above 0. This is context only.
- `crystallizers/vocabulary.py`: `_cluster_active_memories` (`:76`) takes the top two emotions by faded value, with the 6.0 floor applied to faded values. `crystallize_vocabulary` (`:31`) gains `eng` (`scheduler.py:106`).

Tests: a real `run_cycle`, aged memories, the real forgetting pass reaching `handle_drop`, old memories not forming a blend, off pinned.

### PR 6: monologue note and remaining writers

**Trace.** `write_trace_memory(store, monologue, *, feed_digest, persona_dir)` (`monologue/trace.py:34`). Its one production caller is `chat/monologue_capture.py:99`. Live:
- Content is `feed_digest`.
- Emotions are the strongest short-average and strongest long-average feelings from `read()`, as one key with the larger value if they are the same emotion. Zeros are filtered, and an empty read stores `{}`.
- Importance comes from the existing `_trace_importance`.

The engine doesn't count `monologue_trace`, so there is no feedback loop. Off: the verbatim monologue and `aggregate_state`, as today.

**#93 guard.** Live, it compares `recent[0].content` with `feed_digest`. Accepted edges: the same digest on a different thought is dropped, including its pass 2; the same thought with a different digest is kept twice.

**Unchanged.** `monologue_digest.jsonl`, the Feed, `surface`, and `monologue_text` for pass 2.

**Climax journal.** `tools/impls/add_memory.py:77` uses `body/events.CLIMAX_THRESHOLD` (`:36`), with the import above the `if`. Same value, no gate.

**Other writers.** Ingest, file-commit and grief get tests that they store 0-10.

**One bug fix.** `ingest/emotion_backfill.py` `_normalize` (`:149`) lets NaN through, because `min(nan, 10.0)` returns NaN. The check becomes `not math.isfinite(v) or v <= 0.0`, matching #466. Failing test first.

**Consequences once live:**
- `recall_monologue` and the interest sweep search at most 400 characters of digest instead of up to 3,000 of raw thought.
- Pass 2's queue is in memory, so a restart before it drains loses that thought's extraction, with only the digest remaining.
- Old traces keep their verbatim text until forgetting reaches them.
- Live traces hold today's third-person digest until PR 7.

This PR's work also rewrites the `CLAUDE.local.md` gotcha that says raw monologue is persisted synchronously.

Tests: capture through the real `_dispatch_record_monologue` with a switch file (digest, two feelings, importance, `monologue_text` still raw). Dedup by digest. One key when both feelings match. Empty read. All three trace readers. Off pinned. Climax at the threshold. Backfill NaN and inf. The writers table.

### PR 7: cleanup (after Engine 6, before Engine 7)

- Delete the old `aggregate_state` branches in `engines/dream.py`, `monologue/trace.py`, `growth/scheduler.py`, `growth/crystallizers/creative_dna.py` and `growth/crystallizers/reflex.py`.
- Change the `feed_digest` and `surface` descriptions in `tools/schemas.py` to first person. The Feed's voice changes here.

## 5. Wiring

| PR | Reads from | Feeds into |
|---|---|---|
| 1 `peer_trust` | `relationship_state.trust_score` | the engine's peer modifier, once Engine 6 swaps out the stand-in |
| 2 signed changes | pass 2's extraction, the `reconcile_self_read` call | `fold_change`, then the engine's averages and every reader |
| 3 nudges | making, reach-out and note events | memory rows, then the commit hook and the engine |
| 3 peer memories | the model's reading of the exchange | memory rows, then the engine, weighted by trust |
| 4 research | the session's `EMOTIONS` and `ENGAGEMENT` lines | memory row and `engagement_rating`, then the engine through the research modifier |
| 5 dreams | faded seed values, `read()` for mood | dream row (not counted), recall |
| 5 grief | faded values of the dropped memory | breadcrumb row (not counted), recall |
| 5 growth | short averages, faded values | crystallizer context, vocabulary blends |
| 6 monologue note | `feed_digest`, `read()` | interior-continuity block, monologue recall, interest sweep |

## 6. Deferred / kept on purpose

| Item | Why | Revisit |
|---|---|---|
| Raw trust can jump | One reflection can move a peer's weight sharply; `trust_score` is model-proposed and only clamped | If peer memories swing her feelings |
| Old rows and pending-queue rows | Keep their small values, no migration | Only if a reader shows the mix matters |
| `peer_emotion_window` table | Stays under the additive-only rule; nothing writes to it on the live path | #388 |
| Research limits | Multi-line JSON and felt lines before `VERDICT` count as missing and are logged | If `research_felt` failures show up |
| Duplicate-check edges | §4 PR 6 | If digests repeat in practice |
| Dynamic nudges (#566) | Out of scope | #566 |

These also go in `project_companion_emergence_deferred.md` and the next brainstorm.

## 7. Merge order and coordination

- PR 1 can land any time.
- PR 2 waits for #671.
- PRs 3 to 6 are independent of each other. All six land before Engine 6.
- PR 7 lands after Engine 6 and before Engine 7.
- #673 (draft) removes `relationship_emotion_delta` and `_EMOTION_BASE`, so PR 3 and PR 7 don't touch them. Whichever of #673 and PR 3 lands second rebases in `relationship.py`.
- #543 has merged, so the parent's "rebase after #543" no longer applies.

## 8. Questions for ThinkerOfThoughts (#633)

Answers to their open questions, posted once this spec is approved:

1. The nudge is gated: yes.
2. Faded values: yes.
3. `peer_trust` shape: yes, with raw trust.
4. `ensure_engine_ready` in the dream handler is fine (`_dream_handler`, `cli.py:918`, is developer-only; production dreams come from the heartbeat).
5. The old branches go in PR 7.

New question: `handle_drop` runs after `hard_delete`. Today's `faded_value` reads only the memory object, so this works. Engine 6 replaces it, so it must keep working for a row that is already deleted. PR 5 pins that with a test.

## 9. Not verified

- Whether reach-out review treats a dream vector's size as meaningful.
- Whether anything outside `grief` reads a memory after it is deleted. Only `handle_drop`'s one caller was checked.
- The grief breadcrumb's upper bound (`grief/breadcrumb.py:123`); the PR 6 test settles it.
- How often pass 2's queue loses work on a restart, and whether a 400-character digest gives the interest sweep enough. Both only show in live use.
