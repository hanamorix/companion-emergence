# Emotion rework — Hana's half (design)

Umbrella: #240. Shared plan: "Emotion system rework: plan and split" (ThinkerOfThoughts + Planning, 2026-10-08; going into `docs/source-spec/` with its own PR). This spec covers only the parts §8 of that plan assigns to Hana — A1 writer-side tests, A2 cleanup, A4 vocabulary, B1 writers — plus the particulars §8 leaves to each owner. Where it disagrees with the shared plan, it's because the code differs; §2 lists each case.

Verified against `origin/main` @ `44af2447` on 2026-10-27/28. #321 is merged (2026-10-15), so nothing here waits on it.

## 1. Outcome

Her feelings come from what she experiences, and the writers feed that honestly: signed changes can lower a feeling, every writer stores values on the same 0-10 scale, research and peers contribute in proportion to their source, dreams stay in range, and only words that pass one gate become part of her vocabulary. Until the phase C switch flips, **nothing she feels changes** — every B1 behaviour sits behind the per-persona switch from step 0.

Done means: each work item in §4 is merged with its tests, the A1 expected-failures it targets have flipped to passes, and the full gate is green (`uv run pytest`, ruff, `pnpm test`, `pnpm build`).

## 2. Where the code differs from the shared plan

| Plan | Code (file:line at verification) | Consequence |
|---|---|---|
| Old memories register their words when reappraisal brings them back (§5) | `consolidation._handle_reappraisals` (`brain/engines/consolidation.py:833`) re-scores importance and names only; it never reads emotions | Replaced: one final heal scan, then delete (§4 A4) |
| Heal scan runs every 15 min | `persona_loader._heal_referenced_but_unregistered` (`brain/emotion/persona_loader.py:195`) runs on every `load_persona_vocabulary(store=…)`: heartbeat (`supervisor.py:2026`), soul review (`:2138`), bridge startup (`server.py:160`), 7 CLI sites (`cli.py:718,939,1000,1122,1178,1811,2154`) | Removing it drops `store=` at all 10 store-passing callers |
| Reuse the "letters and underscores, ≤100 chars" rule | No emotion-name rule exists. `known_names.MAX_NAME_LENGTH = 100` has no character rule; `growth/scheduler._is_valid_name` only rejects `/ \ { }` | The gate is new code |
| `add_memory` must set last-accessed | Confirmed it doesn't (`Memory.create_new` leaves `last_accessed_at=None`). Also its auto-link `store.search_text(kw, limit=4)` bumps last-accessed and recall count on related memories | Fix both in A4 |
| Emotion-only rows "hold no content" | They hold a bracketed channel list, e.g. `[monologue emotion influence: joy:1.50]` | None; noted for accuracy |
| `feed_digest` is "today only logged" | Also read into the user-visible Feed by `bridge/feed.py::build_monologue_entries` (`:370`) | B1-5 keeps that path |
| The trace's three readers "still work" | All three read only `.content`. Interior-continuity (`monologue/ambient.py:22`) gets the full monologue, newest 5, ≤1200 chars | Works, but the prompt block gets thinner — needs ThinkerOfThoughts' agreement (§6) |
| #93 guard "compares full text" | Compares full text, but only against the newest queued trace (`monologue_capture.py:87`) | B1-5 keeps that scope, compares digests |
| Peer cap is "0.6 per peer per day" | Leaky bucket on the summed size of all emotions in one change, draining linearly over 24 h (`kindled_link/limits.py:34`, `store.py:610`, `relationship.py:355`) | A1 pins the real shape |
| Climax journal | Only `add_memory` triggers it, with a literal `>= 7.0` (`tools/impls/add_memory.py:77`), writing via `store.create` (no gate) | B1-5: use `CLIMAX_THRESHOLD` |
| Research rating "from the call that already reads it" | That call emits NOTES / MEMORY / VERDICT markers (`engines/research_session.py:11`, `prompt_strings.toml:538`) | Two new markers, not one field |
| Decay erases over-10 dream values | True, indirectly: `EmotionalState.set` raises above the clamp and the heartbeat drops that key on rewrite (`heartbeat.py:919-931`). It also drops negatives, unregistered names and anything < 0.01 | A1 pins it; the ×10 data fix (C, ThinkerOfThoughts) only rescales what's left |
| `update_baseline` feeds the felt-time intensity driver | It also feeds the chat-activity baseline (`supervisor.py:1944`) | ThinkerOfThoughts' C removal must keep chat activity working (§6) |

## 3. Decisions (Hana, 2026-10-28)

1. **Old words:** run the heal scan one final time, then delete it. No reappraisal hook.
2. **B1 split:** five PRs by mechanism (§4).
3. **Digest voice:** first person. The schema text for `record_monologue.feed_digest` asks for "I…". Final wording goes to the end-of-plan wording pass.

## 4. Work items

Each is one issue, one branch, one draft PR based on `main`. Tests come first (TDD); each item lands complete with its tests.

### A1 — writer-side safety net (now)

New file `tests/unit/brain/emotion/test_rework_writer_pins.py`, temp `KINDLED_HOME` (no mocked state files). Each bug gets two tests: one pinning today's behaviour (passes now), one asserting the intended behaviour marked `xfail(strict=True)`, so the PR that fixes it must remove the marker.

| Pin | Through-path | Flipped by |
|---|---|---|
| Negative extractor `emotion_delta` raises the feeling | `extractor._apply_emotion_delta` → the felt-state read | B1-1 (on switch) |
| Negative `reconcile_self_read` delta raises the feeling | `reconcile_self_read(...)` → the felt-state read | B1-1 |
| A dream emotion over 10 is erased by one heartbeat | `dream._write_dream_memory` → `heartbeat._apply_emotion_decay` | B1-4 |
| A negative peer value is erased by one heartbeat | `write_kindled_peer_memory` → `_apply_emotion_decay` | B1-2 (values become 0-10) |
| Spelling variants pool as separate channels | `aggregate_state` on `comfort-seeking` + `comfort_seeking` | The one-line fix in §6.6, or C |
| Peer bucket caps the summed size across emotions | `relationship.apply_peer_emotion` twice within 24 h | B1-2 (cap unused on switch) |

"The felt-state read" is `aggregate_state` today. The B1 PRs switch the assertion to the engine's read call with the switch on.

### A2a — cleanup (now)

- Delete `brain/emotion/blend.py` + `tests/unit/brain/emotion/test_blend.py`.
- Delete the residue queue: `ResidueEntry`, `EmotionalState.residue`, `add_residue` and their `copy/to_dict/from_dict` handling (`brain/emotion/state.py:25-117`). `EmotionalState` is never loaded from disk, but `from_dict` keeps ignoring an unknown `residue` key.
- Delete the dead peer table: `_EMOTION_BASE`, `_EMOTION_ACCENT`, `relationship_emotion_delta` (`kindled_link/relationship.py:336-352`) + `tests/kindled_link/test_relationship_emotion.py`.
- Remove `protected` from `brain/migrator/transform.py:159`, `brain/cli.py:635`, `brain/recovery/source_reader.py:66`. The `Memory` field, the column and the heartbeat check stay until ThinkerOfThoughts removes them in C.
- Fix wrong comments in Hana-owned files. Tool-text wording waits for the final pass.

### A2b — weather-shift anchor removal (after #385 merges)

- `brain/felt_time/weather_shift.py`: delete `detect_shift`, `ShiftDetection`, `_LABEL_TEMPLATES`/`_label_for`, `append_shift_log` and their constants. Keep `Baseline`, `update_baseline`, `BASELINE_WINDOW`.
- Remove the `weather_shift` anchor from `anchors.py:39`, `state.py:16,97`, `prompt.py:18,26,146`, `tool.py:61-64`, `tools/schemas.py:504,529`.
- Comments: `supervisor.py:1864,2486,2540`, `narrative_memory/arc.py:33`.
- JSON-IO guard: `tests/unit/json_io_allowlist.txt:21`, `tests/unit/test_no_direct_json_io.py:786`, `tests/unit/brain/health/test_json_io.py:733` (needs another held-writer example).
- Tests: trim `tests/felt_time/test_weather_shift.py` to the `update_baseline` test; update `tests/felt_time/test_prompt.py`, plus the `weather_shift` cases #385 adds.
- `docs/maturity-manifest.md:81`.
- `FeltTimeState.weather_baselines` stays: saved state may only change by adding fields.

### A4 — vocabulary (now)

- **Gate:** `vocabulary.validate_new_name(raw) -> str | None` canonicalises, then accepts `^[a-z_]{1,100}$`. Refusals are logged. Used by `add_memory` and the crystallizer only; every other writer keeps dropping unregistered names.
- **Thread safety:** `register()` takes a module `threading.Lock`; the "startup-only" docstring is corrected.
- **Freshness:** `persona_loader.refresh_if_changed(path)` re-reads the file when `(st_mtime_ns, st_size, st_ino)` changes, the known-names pattern without SQLite's change counter. `load_persona_vocabulary` (already called by the heartbeat) uses it, and the engine's read call calls it (§6.5), so no new call sites are needed in ThinkerOfThoughts' files.
- **`add_memory`** (`tools/impls/add_memory.py`, `tools/schemas.py:240`):
  - gate and register new words at write time;
  - optional `emotion_half_lives: {name: days}`, defaulting to the dev-file default;
  - set `last_accessed_at` on the new memory;
  - auto-link search passes `bump=False`.
- **Crystallizer** (`growth/crystallizers/vocabulary.py`, `growth/scheduler.py`):
  - `register()` straight after `_append_to_vocabulary`;
  - half-life = the mean of the parents' half-lives (a parent with none uses the default);
  - read the emotion column only, newest first, with a limit, through a new additive getter `MemoryStore.list_recent_emotions(limit)`.
- **Vocab repair:** `_excerpts_for_name` passes `bump=False` (`health/vocab_repair.py:280`). Its stale "startup only" docstring is fixed.
- **Final heal:** vocab repair's startup step runs `_heal_referenced_but_unregistered` once, then writes the marker `cadence/vocab_heal_final.done` through `brain.paths.cadence_state_path`. Then the `store=` heal path is deleted from `load_persona_vocabulary` and all its callers.
- **Half-life source:** the per-persona vocabulary file keeps `decay_half_life_days` per word. The dev-file default and the baseline table are ThinkerOfThoughts' step 0 / dev-constants work; A4 reads the default through his lookup (stubbed until it lands).

### B1 — writers (after the step 0 stub; every change gated on the switch)

With the switch off, each writer keeps today's code path exactly.

- **B1-1 Signed changes** — `chat/extractor.py`, `self_model/reconcile.py`.
  - Switch on: send a signed change entry to the engine (`value = sign(v) × min(|v| × 10, 3.0)`), and write no `monologue_emotion` / `self_model_reconcile` row.
  - Flips the two A1 sign pins.
- **B1-2 Nudge scale + kindled-link**
  - `maker/wiring.py`, `initiate/reach_emotion.py`, `notes/runner.py`: constants ×10.
  - `kindled_link/relationship.py`, `prompt_strings.toml:772`: ask for 0-10 values; stop using the bucket cap. The `peer_emotion_window` table stays.
  - New `kindled_link.peer_trust(persona_dir, peer_id) -> float` (0-1) reading `relationship_state.trust_score`.
- **B1-3 Research** — `engines/research.py`, `research_session.py`, `prompt_strings.toml:538`.
  - Add `EMOTIONS:` (JSON, 0-10) and `ENGAGEMENT:` (0-10) markers.
  - Emotions are stored on the research memory, filtered to registered names; engagement goes in metadata under the step 0 field name.
  - A missing or unparsable marker means no emotions / no rating, never a failure.
- **B1-4 Dreams** — `engines/dream.py`.
  - Each emotion's value is the sum over the memories in the dream that carry it, divided by how many carry it, held to 10.
  - The mood read (`dream.py:172`) moves to the engine's short average.
- **B1-5 Monologue note + remaining writers**
  - `monologue/trace.py`, `chat/monologue_capture.py`, `tools/dispatch.py:170-205`, `tools/schemas.py:597-640`.
  - Switch on: the trace's content is the first-person `feed_digest`. Its emotions hold the strongest short-average and strongest long-average feelings from the engine (one key with the larger value if both are the same emotion), with importance from the existing `_trace_importance`. The #93 guard compares digests. `monologue_digest.jsonl` and the Feed are unchanged.
  - `add_memory`'s climax journal uses `body/events.CLIMAX_THRESHOLD`.
  - Ingest, emotion backfill, file-commit and grief writers: tests that they store 0-10 values; no code change expected. (Emotion backfill is also touched by #543 — rebase after it.)

## 5. Wiring

| Item | Reads from | Feeds into |
|---|---|---|
| Signed changes (B1-1) | pass-2 extraction; her `reconcile_self_read` call | engine short/long averages → every reader in shared plan §1 |
| Nudges (B1-2) | making / reach-out / note events | memory rows → commit hook → engine |
| Peer memories (B1-2) | model's reading of the exchange; `peer_trust` | memory rows → engine, weighted by trust (A3 modifier) |
| Research (B1-3) | the research session's own output | memory row + engagement → engine via the modifier |
| Dreams (B1-4) | engine short average (mood read); seed memories | dream row (not counted); recall |
| Monologue note (B1-5) | `feed_digest`; engine averages | interior-continuity block, monologue recall, interest sweep, Feed |
| Vocabulary gate (A4) | `add_memory`; crystallizer | vocabulary file → both processes via `refresh_if_changed` → engine name lookup |

## 6. Needed from ThinkerOfThoughts at step 0

1. **Hook placement:** in `MemoryStore.create` and the consolidation promotion path, dated by `created_at`. Peer memories and the climax journal call `store.create` directly; gated types reach `memories.db` only on promotion.
2. **Row updates:** emotion backfill writes via `store.update(..., emotions=…)` (`ingest/emotion_backfill.py:479`), which a create-hook won't see. Decide whether backfilled emotions fold in.
3. **`update_baseline`:** also drives the chat-activity baseline (`supervisor.py:1941-1944`), so C can't just delete it.
4. **Interior-continuity block:** gets digests (≤400 chars each) instead of full monologues after the switch. Agree it, since the prompt is his.
5. **Vocabulary freshness:** the engine's read call should call `refresh_if_changed`.
6. **Spelling pooling:** a one-line `canonical_name` key in `aggregate_state` (his file, deleted in C) would flip A1's spelling pin before C. Optional.
7. **Simplification:** the new nudge scale starts exactly at the switch-over time (B1 is gated on the switch), so "record when the new nudge scale started" can reuse the switch-over time. No extra field.

## 7. Coordination

- Assign #125 to Hana (its fix is B1-1), or note it there.
- #385 (ThinkerOfThoughts) touches `felt_time/state.py` and pins `weather_shift`, so A2b waits for it.
- #543 (ThinkerOfThoughts) touches `ingest/emotion_backfill.py`; B1-5 rebases after it.
- #417 (Hana) touches `tools/schemas.py`. A4 / A2b / B1-5 edit different tools there; whichever lands second rebases.
- Discord heads-up at each sequential point (shared plan §9).

## 8. Deferred / kept on purpose

| Item | Why | Revisit |
|---|---|---|
| `memories.protected` column | Older brains still read it (shared plan §4) | Backwards-compatibility issue (#388) |
| `FeltTimeState.weather_baselines` field | Saved state may only change by adding fields | #388 |
| `peer_emotion_window` table | Same, for `kindled_link.db` | #388 |
| Junk words the old heal scan already registered | The gate only stops new ones | Bugs-ledger sweep at the end |
