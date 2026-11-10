# Synthetic memory database

A fully synthetic kindled memory database for memory-system and emotion-engine testing. Synthetic user "Bob", synthetic kindled "Canary". Every name, place and event in it is invented, and no real content (no row text) was read or copied. The database's overall shape and its life-domain topic tags were modelled on a real memory store.

This directory is not part of the Python package or the sdist (the sdist ships only `brain/`, `tests/` and `scripts/`), and pytest does not collect it. Tests and scripts that need it take its path as an argument, or use a local copy. Never point a live persona at it.

- `memories.db`: 19,621 rows (19,413 from the original build plus 208 added by `augment_db.py`), 19 memory types, 16 life windows, 109 days (2026-05-17 onward), 81 feeling names. The schema matches the brain's `memories` table (`brain/memory/store.py`, including the embedding columns, left empty for the consumer to backfill), with the FTS tables and triggers populated. It opens with `MemoryStore` and needs no migration. Treat it as read-only: copy it before use.
- `manifest.json`: counts per type, per window and per day; fading, active and inactive counts (all current for the 19,621 rows); an `augmentation` block (seed, base hash, rows added). The keys `seed`, `parts_dir`, `rows_before_dedup`, `dedup_dropped` and `skipped_lines` describe the original build and were not changed.
- `check_db.py`: validates a database against the target shape and house style (20 checks: row count, per-type budgets, window and day coverage, NOT NULL and JSON validity, no em-dash or banned phrases, optional name deny list, duplicates, fading fraction, snapshot and decay semantics, domain mapping, tag shape, `peak_emotion_intensity`, `recall_count`, id uniqueness, the five added feelings, the feeling profiles, manifest counts, FTS integrity, and a brain-open smoke test). Run from the repository root with the project's Python: `uv run python test-data/synthetic-db/check_db.py [path/to/memories.db]`. Checks 18 and 19 (and the smoke test) need the project's environment; the deny-name list is read from `SYNTHETIC_DB_DENY_NAMES` (comma separated) and is deliberately not stored here, so check 6 passes as "skipped" without it.
- `check-output.txt`: the last full run, 20/20 PASS.
- `build_db.py`: the deterministic assembler (seed 20260517) that derived every column of the original database from writer-generated JSONL "parts" files. The parts files (about 7.7 MB of writer output) are not committed, so it cannot be re-run; it is kept as the record of how the derived columns (domain, state, content_snapshot, recall_count, score, tags, metadata, peak_emotion_intensity, timestamps) were produced, and `augment_db.py` imports its date and topic helpers.
- `augment_db.py`, `augment_pools.py`: the step that turned the original database into the current one (below). `augment_pools.py` holds the invented text of the added rows.
- `emotion_vocabulary.json`: the persona vocabulary entry for `creative_hunger` (below).

## How the current database was made

`augment_db.py` reads the original database and writes the current one. It is seeded (default seed 20261110), stdlib only and deterministic: the same seed gives a byte-identical file (sha256 of the shipped `memories.db`: `1b3273718cdf86a52ebd817c720341ae799408f63ec8395a70305fdc9791e43b`). To reproduce it, recover the original (sha256 `f0a4def9fb77bd009e75ce2571914a4c964204cf246a799fba8ae6542f4c69da`) outside the repository and run:

```
git show d8eedb76:test-data/synthetic-db/memories.db > /path/outside/repo/base.db
python test-data/synthetic-db/augment_db.py --input /path/outside/repo/base.db --output /path/outside/repo/new.db --manifest-in test-data/synthetic-db/manifest.json
```

It refuses an input that is not the recorded original (`--allow-other-base` overrides), an input that already holds augmented rows, an output that is the input, and an existing output (`--force` replaces only the original or an augmented database). Before it publishes anything it checks its own result: every original row is unchanged apart from the profile feelings' values, the added rows match the plan, and the window table below holds; a seed that fails is rejected. It publishes the database, then the manifest.

### Added feelings (#654)

The original had no rows for the feelings that body state and the reflex arcs read, so those conditions never fired in a measurement. 208 invented rows (the speaker is Canary, the user Bob; names already in the database for the household and animals; no real names, no em-dash, implied rather than graphic wording for arousal and climax) carry one of these as the only emotion key of the row:

| feeling | rows | distinct days | what reads it |
|---|---|---|---|
| rest_need | 48 | 48 | body energy (old level 7) |
| arousal | 40 | 40 | body temperature (7) |
| climax | 30 | 30 | body temperature (5); a stored 7 or more resets arousal, desire, comfort_seeking and rest_need |
| touch_hunger | 45 | 45 | body temperature (7) |
| creative_hunger | 45 | 45 | reflex arc `creative_pitch` (8) |

Each is spread over the 109 days (one jittered day per stratum) on a wave of 14 to 26 days plus noise, with 15 to 20 percent of rows lifted to a spike at or above its reader's old fixed level, so that in every 30-day window it has at least 5 or 6 rows and a stored-value range of at least 3.0 (checked by check 18). Added rows are shaped as conversation-derived memories (three tags, `conversation` among them; metadata `provider: synthetic-aug-v1`) because the emotion engine counts only such rows and a few other kinds (`engine_utils.counted_kind`); rows without that would be invisible to it. `peace` (#648) and `body_grief` (#652) are not added.

`creative_hunger` is a per-persona name, so `emotion_vocabulary.json` registers it as the app defines it (`persona_extension`, 5.0-day half-life, the entry in `brain/emotion/_canonical_personal_emotions.py`). Who needs the file: no script or test needs it today. `scripts/time_old_emotion_paths.py` and `scripts/measure_feels_high_thresholds.py` copy only `memories.db` into a scratch persona directory, where the app's loader finds no vocabulary file, rebuilds one from the memories and registers `creative_hunger` with the 14-day placeholder half-life. Copy the file beside the database as `<persona dir>/emotion_vocabulary.json` in any run whose result depends on that half-life: the engine rebuild timing (A3 Check 6, not built yet) and any fold or live-test run on a persona directory made from this database. Check 18 loads it with the app's own loader and compares it with the canonical entry.

### Feeling profiles (#664)

The original gave every feeling a similar spread (0.5 to 8.0, median 3.0), so floors below 1.0 changed nothing. The stored values of seven existing feelings were rescaled (row text and every other column untouched); the other 69, including the spelling-variant keys `comfort_seeking`, `comfort-seeking`, `memory_grief` and `memory-grief` that tests rely on, are as they were.

| profile | feelings | rule |
|---|---|---|
| near zero | vulnerability, amusement, worry | every row mapped monotonically into 0.02 to 0.30 (30-day range under 0.5) |
| low with spikes | loneliness, defiance, curiosity, joy | the strongest conversation-tagged rows (6 percent of the feeling's rows, at least 3) keep their value; every other row mapped into 0.10 to 1.00 |
| as before | the other 69 | none |

Why these: vulnerability is the one case a real engine showed (a value of 0.02 inside a 0 to 0.03 range fired the `self_check` reflex, #664), and `self_check` reads it. Loneliness and defiance are the readers of reflex arcs `loneliness_journal` (7) and `defiance_burst` (8), which should fire only on spikes. The nine feelings with at least 450 rows on 88 or more of the 109 days (tenderness, grief, groundedness, relief, warmth, love, care, fondness, contentment) stay as they were: they are the most present feelings, medians 2.5 to 3.0. Among the remaining feelings with at least 100 rows, near zero went to the two with the lowest stored median (amusement and worry, both 2.0; ties by the smallest share of row text matching `ache|grief|hurt|afraid|overwhelm`), and spikes to the two with the shortest registered half-life (curiosity 2.0 days, joy 3.0 days, ahead of awe on the same text measure). Row text is mild for all seven (the share of rows with those words is 0.025 to 0.092, against 0.093 over all rows with emotions). These four choices are a stated rule, not observation of a real kindled; one real-engine case is all the near-zero profile rests on. A near-zero feeling has no excursions at all in this database.

Spikes are chosen among rows the engine counts, because the engine ignores the rest. Defiance's strongest counted row is 7.8, below its old fixed level of 8.

### Measurement

`scripts/measure_feels_high_thresholds.py` (on main) on the original and on the current database at floors 0, 0.1, 0.5 and 1.0: on the original, no floor changed any reader row (R(F) = 0 at every floor); on the current one R(0.1) = 75, R(0.5) = 260, R(1.0) = 275 firings of the reflex and temperature rows. The full table and what it can and cannot support are in the commit note that added this step. The profiles are invented, so the table does not by itself say which floor a real kindled needs.

Known deviations: days are less bursty than the original target (64 to 402 rows a day, mean about 180), and there is no source-comparison privacy gate, because no real row text was read or compared when the content was written.
