# Synthetic memory database

A fully synthetic kindled memory database for memory-system and emotion-engine testing. Synthetic user "Bob", synthetic kindled "Canary". Every name, place and event in it is invented, and no real content (no row text) was read or copied. The database's overall shape and its life-domain topic tags were modelled on a real memory store.

This directory is not part of the Python package or the sdist (the sdist ships only `brain/`, `tests/` and `scripts/`), and pytest does not collect it. Tests and scripts that need it take its path as an argument, or use a local copy.

- `memories.db`: 19,413 rows, 19 memory types, 16 life windows, 109 days (2026-05-17 onward). The schema matches the brain's `memories` table (`brain/memory/store.py`, including the embedding columns, left empty for the consumer to backfill), with the FTS tables and triggers populated. It opens with `MemoryStore` and needs no migration. Treat it as read-only: copy it before use, and never point a live persona at it.
- `manifest.json`: counts per type, per window and per day; fading, active and inactive counts; dedup stats.
- `check_db.py`: validates a database against the target shape and house style (17 checks: row count, per-type budgets, window and day coverage, NOT NULL and JSON validity, no em-dash or banned phrases, optional name deny list, duplicates, fading fraction, snapshot and decay semantics, domain mapping, tag shape, id uniqueness, FTS integrity, and a brain-open smoke test). Run from the repository root with the project's Python: `uv run python test-data/synthetic-db/check_db.py [path/to/memories.db]`. Set `SYNTHETIC_DB_DENY_NAMES` (comma separated) to have it also scan for a private list of names; the list is deliberately not stored here.
- `check-output.txt`: the last full run, 17/17 PASS.
- `build_db.py`: the deterministic assembler (seed 20260517) that derived every column from writer-generated JSONL "parts" files. The parts files (about 7.7 MB of writer output) are not committed; the database is the artifact. The script is kept as the record of how the derived columns (domain, state, content_snapshot, recall_count, score, tags, metadata, peak_emotion_intensity, timestamps) were produced.

Known deviations: days are less bursty than the original target (62 to 398 rows a day, mean about 178), and there is no source-comparison privacy gate, because no real row text was read or compared when the content was written.
