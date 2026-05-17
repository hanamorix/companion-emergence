# Migrating from OG NellBrain

This playbook is for forkers transitioning from the original `NellBrain` repository to `companion-emergence`. After running it, your persona's biographical content (memories, soul, creative voice, journal, gifts, narratives, etc.) lives in the new framework's persona directory at `${NELLBRAIN_HOME}/personas/<your-persona-name>/` (or the platformdirs default if you don't set `NELLBRAIN_HOME`).

## Prerequisites

- A working OG NellBrain `data/` directory — the parent of `memories_v2.json`, `nell_soul.json`, etc.
- A clone of `companion-emergence` with a working Python environment (`uv sync` succeeds).
- The OG `data/` directory should not be modified during the migration. The migrator only reads.

## Dry-run first

The migrator supports a no-install dry-run mode that writes to a directory of your choice:

```bash
uv run nell migrate \
    --input /path/to/your/NellBrain/data \
    --output /tmp/og-migration-dry-run
```

This writes the migrated artefacts to `/tmp/og-migration-dry-run/` without touching your actual persona directory. Inspect the migration report:

```bash
cat /tmp/og-migration-dry-run/migration-report.md
```

You should see counts for every migrated surface. A real run on the original Nell's data looks like:

```
Migration complete.

  Memories:       1,223 migrated, 0 skipped
  Hebbian edges:  4,404 migrated, 0 skipped
  Reflex arcs:    8 migrated
  Vocabulary:     25 emotions migrated
  Interests:      2 migrated
  Crystallizations: 38 migrated
  Creative DNA:   migrated
  Journal:        0 memories retagged
  Legacy files:   16 preserved, 0 missing
  Soul candidates: 38 migrated, 2 skipped (missing memory_id)
  Reflex fires:   5 migrated
  Elapsed:        1.8s
```

Numbers vary by your dataset. "Missing" / "skipped" entries indicate optional OG files your install didn't have (or candidates without memory IDs the new framework can't link), which is normal — those counts will be zero or small for most forkers.

## What gets migrated

| Surface | Source OG file | Notes |
|---|---|---|
| `memories.db` | `memories_v2.json` | All memories (active + inactive), with emotion scores |
| `hebbian.db` | `connection_matrix.npy` + `connection_matrix_ids.json` | Learned co-activation edges |
| `reflex_arcs.json` | AST-extracted from OG `reflex_engine.py` | Structured arc definitions |
| `crystallizations.db` | `nell_soul.json` | Permanent soul moments |
| `creative_dna.json` | `nell_creative_dna.json` | Literary voice fingerprint (core voice, strengths, tendencies, influences) |
| `interests.json` | `nell_interests.json` | Tracked topics with pull scores |
| `emotion_vocabulary.json` | derived from memories | Persona-specific emotion names not in framework baseline |
| Journal retag | inside `memories_v2.json` | `memory_type='reflex_journal'` retagged to `'journal_entry'` with metadata |
| `soul_candidates.jsonl` | OG `soul_candidates.jsonl` | Importance rescaled 0-100 → 0-10; `decided_at` split to `accepted_at`/`rejected_at` by status; entries without `memory_id` skipped |
| `reflex_log.json` | OG `nell_reflex_log.json` | Stripped to new schema (drops `output_preview`, `output_type`, `days_since_human`, `description`); `output_memory_id: null` per fire (OG didn't have this concept) |
| `legacy/` (verbatim) | 16 biographical OG files | `nell_journal.json`, `nell_growth.json`, `nell_gifts.json`, `nell_narratives.json`, `nell_surprises.json`, `nell_outbox.json`, `nell_personality.json`, `emotion_blends.json`, `nell_emotion_vocabulary.json`, `nell_body_state.json`, `nell_heartbeat_log.json`, `nell_growth_log.jsonl`, `behavioral_log.jsonl`, `soul_audit.jsonl`, `self_model.json`, `nell_style_fingerprint.json` |

## What does NOT get migrated

- **Daemon runtime state** — `daemon_state.json`, `supervisor_state.json`, `nell_session_state.json`, etc. The new framework regenerates these.
- **Embedding caches** — `memory_embeddings.npy`, `memory_embedding_ids.json`. Auto-regenerated on first use.
- **Active conversations** — the `active_conversations/` directory. Open chat sessions don't carry across.
- **Logs** — `nell_log.jsonl`, `coactivation_log.jsonl`, `curation_log.jsonl`, `hebbian_tick_log.jsonl`, `bridge.log`, supervisor logs. Historical noise; the new framework starts a fresh log for each.
- **`memories_slim.json`** — was a derived projection of `memories_v2.json` in OG; the new framework's `memories.db` is the canonical store and any slim view is regenerated on demand.

The `legacy/` subdirectory preserves all biographical content that doesn't yet have a native framework feature. As `companion-emergence` matures, individual features (journal, growth/opinions, gifts, etc.) will absorb the corresponding legacy files into proper surfaces. Until then, the files live read-only in `legacy/` — you can `cat` them at any time.

## Install (after the dry-run looks good)

The fastest path is the interactive wizard, which sets up the persona
directory + `persona_config.json` + an optional `voice.md` starter all
in one go:

```bash
uv run nell init
# wizard prompts for: persona name, your name, OG data path, voice template
```

Or run it non-interactively with flags:

```bash
uv run nell init \
    --persona your_persona_name \
    --user-name "Your Name" \
    --migrate-from /path/to/your/NellBrain/data \
    --voice-template nell-example
```

The `--user-name` value is **load-bearing** — it tells the ingest
extractor who is talking to the persona so memories from your
conversations get attributed correctly (rather than to historical
figures the persona may reference from her soul context). If you
skip it, extractions fall back to the legacy unnamed prompt path
and may conflate users with named figures.

`--voice-template nell-example` copies the canonical Nell voice as
a starting point; you edit it to remove Nell-specific content and
add your persona's identity. Choose `default` to use the framework's
generic fallback (no file written; you can author voice.md later).

### Or call `nell migrate` directly

If you prefer the lower-level command (e.g. for scripts that already
manage persona_config.json elsewhere):

```bash
uv run nell migrate \
    --input /path/to/your/NellBrain/data \
    --install-as your_persona_name
```

This still works; it just doesn't write `persona_config.json` or
`voice.md` for you — you'll need to add those by hand (or run
`nell init --persona your_persona_name --user-name "Your Name" --fresh`
afterwards to add them on top of the migrated data).

This writes to `${NELLBRAIN_HOME}/personas/your_persona_name/` (or the platformdirs default — see `nell status` to confirm where).

If the persona directory already exists, pass `--force` and the migrator will back the existing directory up with a timestamped name (`your_persona_name.backup-YYYY-MM-DDTHHMMSS/`) before installing the fresh migration.

## Verify

```bash
uv run nell status --persona your_persona_name
```

Should report counts and confirm the persona's existence.

```bash
uv run nell memory list --persona your_persona_name --limit 5
```

Should return your most recent memories.

```bash
uv run nell soul list --persona your_persona_name --limit 5
```

Should return some of your crystallizations.

## Limitations and known caveats

- **Importance scale change.** OG soul candidates used a 0-100 scale; the new framework uses 0-10. The migrator rounds via `round(og / 10)` clamped to `[0, 10]`. Candidates without an `importance` field default to 8. This means low-importance OG candidates (5-14) all map to 1 in the new schema — that's expected and matches the new framework's distribution.
- **Reflex log lossy.** OG fires had inline `output_preview` text; the new framework expects an `output_memory_id` reference. Migrated fires get `output_memory_id: null`. The full OG fire detail is preserved verbatim in `legacy/nell_reflex_log.json` if you need to reference it.
- **Soul candidates without `memory_id` are skipped.** Without a memory ID, the new framework can't link the candidate back to a source memory. The migration report's "skipped (missing memory_id)" count tells you how many. Their content is still readable in your original OG `soul_candidates.jsonl` — the migrator only reads, never deletes.
- **Most legacy biographical content is preserved-but-not-active.** Journal entries, gift logs, narratives, surprises, personality patterns — all preserved in `legacy/` but not yet absorbed into framework features. Future releases will add per-feature absorption.

## Re-running the migration

Re-running with `--force --install-as <name>` is safe — the old persona directory is backed up under a timestamped name, then the fresh migration installs cleanly. You can also re-run dry-run mode (`--output <dir>`) as many times as you want to inspect the result without touching your active persona.

The migration is deterministic for a given OG input, except for the timestamps embedded in the `migration-report.md` and the migrated-at fields in `creative_dna.json`. The functional content is byte-stable across re-runs.

## What if something fails

If the migrator errors out, the partial output dir is left in place for inspection. The original OG `data/` directory is never modified — you can safely investigate the source and re-run.

If a specific migrator section fails (e.g. `Soul candidates: ... (error: migrate_error: ...)` in the report), the rest of the migration still completes. The failing section's output file simply isn't written. Other sections are unaffected.

For unexpected failures, file an issue with the migration report attached.
