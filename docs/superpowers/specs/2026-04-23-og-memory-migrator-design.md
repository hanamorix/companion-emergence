# OG NellBrain Memory Migrator — Design Spec

> **Week 3.5** — bridges the gap between Week 3 (memory substrate shipped) and Week 4 (engines) by porting OG NellBrain's memory data into the new companion-emergence SQLite stores. Lets Week 4 engines run against real accumulated history instead of synthetic tests.

**Status:** design approved by Hana 2026-04-23; plan to be written next.
**Scope:** memory substrate only (memories + Hebbian edges). Embeddings, soul, personality, self-model, journals, and logs are explicitly out of scope.

---

## 1. Goal

Port 1,141 memories + 8,808 Hebbian edges from `/Users/hanamori/NellBrain/data/` into a new companion-emergence persona using the Week 3 stores (`MemoryStore`, `HebbianMatrix`), WITHOUT touching or modifying the OG data.

After the migrator runs and the user confirms, running the new framework with `uv run brain run nell` should surface Nell's actual memory history.

## 2. Non-goals

- Embeddings — 822 pre-computed 768-dim OG vectors are NOT carried over. They will regenerate naturally in Week 5 when the Ollama bridge lands and `MemorySearch.semantic_search` is first called. Rationale: our `EmbeddingCache` keys on content-hash only with no provider identity, so carrying pre-computed vectors from a possibly-different model risks silent drift when we commit to a Week 5 provider.
- Persona identity layer (`nell_soul.json`, `nell_personality.json`, `self_model.json`, `nell_emotion_vocabulary.json`). These map to layers the new framework hasn't built yet; separate migrator passes once those layers exist.
- Behavioural layer (`nell_creative_dna.json`, `nell_body_state.json`, `nell_interests.json`, `nell_journal.json`, `nell_narratives.json`, `nell_style_fingerprint.json`, `nell_gifts.json`, `nell_surprises.json`). Same reason.
- Logs / runtime state / engine output (`behavioral_log.jsonl`, `curation_log.jsonl`, `coactivation_log.jsonl`, `nell_log.jsonl`, `nell_heartbeat_log.json`, `nell_reflex_log.json`, `nell_research_log.jsonl`, `nell_outbox.json`, `daemon_state.json`, `nell_session_state.json`, `scheduler_log.jsonl`, `soul_audit.jsonl`, `self_monitor_state.json`, `hebbian_tick_log.jsonl`, `nell_growth*`, `nell_token_state.json`, `nanoclaw_sync_state.json`, `research_queue.jsonl`, `bridge.log`, `research_engine.log`, `nell_brain.log`). Not mnemonic data.
- **Absolutely** no writes to OG — the migrator is pure ETL-read-only on the source.

## 3. Observed OG data (ground truth, 2026-04-23)

```
memories_v2.json         1,141 memories (3.9 MB)
memory_embeddings.npy    822 × 768 float32 embeddings (2.5 MB) — SKIP
memory_embedding_ids.json  822 ids aligned with above — SKIP
connection_matrix.npy    855 × 855 float32 dense matrix (2.9 MB)
connection_matrix_ids.json  855 ids aligned with rows/cols
hebbian_state.json       metadata only (293 bytes)

Memory coverage in Hebbian graph: 8,808 non-zero edges
Average weight (non-zero): 0.956 (most edges at max strength)
```

OG Memory record shape (sample, 2026-04-23):
```json
{
  "id": "a317a2d4-...",
  "content": "...",
  "memory_type": "conversation",
  "domain": "us",
  "created_at": "2024-03-21T16:41:02+00:00",
  "source_date": "2024-03-15",
  "source_summary": "...",
  "importance": 8.0,
  "tags": ["love", "first"],
  "emotional_tone": "tender",
  "active": true,
  "supersedes": "...",
  "access_count": 3,
  "last_accessed": "2024-04-10T12:00:00",
  "emotions": {"love": 9.0, "tenderness": 8.0},
  "emotion_score": 17.0,
  "emotion_count": 2,
  "intensity": 8.5,
  "schema_version": 2,
  "connections": ["...", "..."]
}
```

## 4. Week 3 amendment — `metadata` field

The new `Memory` dataclass gains one field:

```python
metadata: dict[str, Any] = field(default_factory=dict)
```

Stored as a new `metadata_json TEXT NOT NULL DEFAULT '{}'` column in the `memories` table. This absorbs OG-only fields (`source_date`, `source_summary`, `emotional_tone`, `supersedes`, `access_count`, `emotion_count`, `intensity`, `schema_version`, `connections`) without proliferating the dataclass signature.

Rationale: a metadata bag is forward-compatible. A future engine that needs `supersedes` reads from `metadata["supersedes"]`. Nothing is lost; nothing is prematurely elevated to a first-class field.

Required updates to existing code:
- `Memory` dataclass adds the field.
- `Memory.to_dict` / `Memory.from_dict` preserve the dict verbatim (JSON serialisable at schema level).
- `MemoryStore._SCHEMA` adds the column with the default.
- `MemoryStore.create` / `_row_to_memory` / `update` handle the new column.
- New round-trip tests for `metadata` at both Memory and MemoryStore levels.

## 5. Field mapping

| OG field | New field | Transform |
|---|---|---|
| `id` | `id` | verbatim (UUID string) |
| `content` | `content` | verbatim, required (skip if missing/empty) |
| `memory_type` | `memory_type` | verbatim, default `"conversation"` if missing |
| `domain` | `domain` | verbatim, default `"us"` if missing |
| `created_at` | `created_at` | `_coerce_utc` shared helper (Week 3) |
| `last_accessed` | `last_accessed_at` | rename + `_coerce_utc`, None if missing |
| `tags` | `tags` | verbatim, `[]` if missing |
| `importance` | `importance` | verbatim, default 0.0 |
| `active` | `active` | verbatim, default True |
| `emotions` | `emotions` | verbatim if dict-of-numeric; skip memory if any value non-numeric |
| `emotion_score` | `score` | rename; on mismatch with `sum(emotions.values())` log a warning but use OG's value |
| *(not in OG)* | `protected` | default False |
| *(not in OG)* | `metadata` | OG-only fields stuffed in verbatim |

**OG-only fields → `metadata[...]`:** `source_date`, `source_summary`, `emotional_tone`, `supersedes`, `access_count`, `emotion_count`, `intensity`, `schema_version`, `connections`. Any other unknown key also lands in `metadata` (future-proof against OG schema drift).

## 6. Hebbian transform

1. Read `connection_matrix_ids.json` → list of 855 memory ids, index-aligned with rows/cols of the matrix.
2. Read `connection_matrix.npy` → 855×855 float32 array.
3. For each `(i, j)` with `i < j` and `weight > 0.0`:
   - Lookup ids: `a = ids[i]`, `b = ids[j]`
   - Call `HebbianMatrix.strengthen(a, b, delta=float(weight))`
4. Canonicalisation is handled by `HebbianMatrix._canonical` — (a,b)/(b,a) already share one row.
5. If a memory id referenced in the matrix does NOT exist in the migrated `MemoryStore` (because that memory was skipped as malformed), the edge is still written — the store's referential integrity is not enforced by foreign key (by design; spreading_search gracefully handles orphan edges via `store.get(mid) is None` guard).
6. Report: edges migrated, edges skipped (zero-weight — always zero in an OG matrix, but logged for paranoia).

## 7. Malformed data handling — permissive

Skipping policy (log, don't halt):
- Missing or empty `content` → skip
- Unparseable `created_at` → skip
- `emotions` contains non-numeric value → skip
- Duplicate id within OG (shouldn't happen but defend) → skip second+
- OG-only extra keys with unrecognised names → absorb into `metadata`, not a skip

Each skipped record gets a `SkippedMemory(id, reason, field, raw_snippet)` entry. Report groups them by reason at end.

## 8. CLI surface

One new subcommand under the existing `brain` CLI (Week 1 scaffolding):

```bash
# Default: inspect-first. Writes to a named output dir for review.
uv run brain migrate \
    --input /Users/hanamori/NellBrain/data \
    --output ./migrated-nell

# After inspection, install as a persona:
uv run brain migrate \
    --input /Users/hanamori/NellBrain/data \
    --install-as nell

# Force flags (safety overrides):
#   --force         overwrite existing output dir / persona
#   --no-backup     skip the persona backup on install
```

Exactly one of `--output` / `--install-as` is required. Passing both is an error.

## 9. Safety — no OG writes, no silent overwrites

**OG side (read-only):**
1. All OG files opened with `"rb"` / `"r"` mode. Never `"wb"` / `"a"`.
2. Pre-flight: if `memories_v2.json.lock` exists and is recent (< 5 min mtime), refuse — the OG bridge is live and writing. Print "stop the bridge first" message.
3. For every OG file read, record `(path, size_bytes, sha256, mtime_utc)` in `<output>/source-manifest.json`. Cryptographic audit trail.
4. After all reads complete, re-stat each source file — if any mtime or size changed mid-run, abort with clear error and leave output partial + a warning.

**Output side (refuse-to-clobber by default):**
1. `--output <dir>`: if `<dir>` exists AND is non-empty → refuse unless `--force`. Empty dir is fine (lets user pre-create).
2. `--install-as <name>`: if `<platformdirs>/companion-emergence/<name>/` exists → refuse unless `--force`. With `--force`, the old dir is renamed to `<name>.backup-<YYYY-MM-DDTHHMMSS>/` first.
3. `--install-as` is atomic: write to `<name>.new/` in the parent dir, verify all files land, then `os.rename` swap. No half-installed personas.
4. Migration artefacts in `--output <dir>`:
   - `memories.db` — SQLite MemoryStore
   - `hebbian.db` — SQLite HebbianMatrix
   - `source-manifest.json` — audit trail
   - `migration-report.md` — the end-of-run report (same content that prints to stdout)

## 10. End-of-run report

Printed to stdout AND written to `<output>/migration-report.md`:

```
Migration complete.

  Memories:       1,128 migrated, 13 skipped
  Hebbian edges:  8,808 migrated, 0 skipped
  Elapsed:        2.3s

Skipped memories (13):
  - 3 missing required field 'content'
  - 7 emotions dict had non-numeric values (e.g. "high")
  - 2 unparseable created_at
  - 1 duplicate id (second occurrence)

Source manifest:
  memories_v2.json             3,895,916 bytes  sha256=ab12...
  connection_matrix.npy        2,924,228 bytes  sha256=cd34...
  connection_matrix_ids.json      34,200 bytes  sha256=ef56...
  hebbian_state.json                 293 bytes  sha256=789a...

Next steps:
  1. Inspect the output:
       sqlite3 ./migrated-nell/memories.db "SELECT COUNT(*) FROM memories;"
       sqlite3 ./migrated-nell/memories.db \
         "SELECT domain, COUNT(*) FROM memories GROUP BY domain;"
       sqlite3 ./migrated-nell/hebbian.db "SELECT COUNT(*) FROM hebbian_edges;"
       cat ./migrated-nell/migration-report.md

  2. When satisfied, install as the 'nell' persona:
       uv run brain migrate \
           --input /Users/hanamori/NellBrain/data \
           --install-as nell
```

## 11. Package structure

```
brain/
├── migrator/
│   ├── __init__.py
│   ├── og.py          # OG data readers (memories_v2.json, .npy, ids, manifest)
│   ├── transform.py   # per-memory field-mapping + validators; SkippedMemory dataclass
│   ├── report.py      # migration report formatter, source-manifest writer
│   └── cli.py         # `brain migrate` subcommand (argparse wiring)
└── cli.py             # existing — adds the `migrate` subcommand dispatch

tests/unit/brain/migrator/
├── __init__.py
├── test_og.py         # readers with fixture inputs
├── test_transform.py  # field mapping + edge cases (missing fields, bad timestamps, non-numeric emotions, unknown keys into metadata)
└── test_report.py     # report formatting
tests/integration/
└── test_full_migration.py   # runs migrator end-to-end against tests/fixtures/og_mini/
tests/fixtures/og_mini/
├── memories_v2.json          # 5 hand-crafted memories, incl. 1 malformed
├── connection_matrix.npy     # 5×5 sparse-ish
├── connection_matrix_ids.json
└── hebbian_state.json
```

## 12. Testing strategy

- **Unit tests** — each transform function gets a table-driven test of OG shapes → new shapes + edge cases. No filesystem for transform tests.
- **Report tests** — verify output strings match expected structure (easier to read than asserting substrings in stdout).
- **Integration test** — `test_full_migration.py` builds a tiny `og_mini/` fixture (5 memories incl. 1 malformed, 5×5 matrix), runs the migrator end-to-end, opens the output SQLite dbs, asserts counts + specific records + skip reasons.
- **No CI run against real OG data.** The 3.9 MB production file is not in the repo and stays on Hana's machine. CI uses fixtures only.
- **Expected test count:** 20-30 new tests (roughly 4-5 for og, 12-15 for transform, 3-4 for report, 1 integration test with a dozen assertions).

## 13. Dependencies added

None. `numpy` is already a dep from Week 3. `argparse` is stdlib. `hashlib` is stdlib.

## 14. Rollout sequence

1. Write implementation plan (this spec → writing-plans skill).
2. Execute plan in phases: Week 3 `metadata` amendment → migrator modules → CLI → integration test → end-to-end dry run on real OG.
3. Hana runs `--output` mode, inspects output.
4. Hana runs `--install-as nell`, verifies `uv run brain run nell` surfaces real memory history.
5. Week 4 engines can now be tested against lived-in data.

## 15. Success criteria

- All 1,141 OG memories processed (migrated or accounted-for in the skip report).
- All 8,808 non-zero Hebbian edges migrated.
- `source-manifest.json` matches on post-run re-stat (proof OG untouched).
- Round-trip: write to `memories.db` via migrator → read via `MemoryStore.get(id)` → every non-skipped OG memory retrievable.
- `metadata` field round-trips cleanly through the amended Memory dataclass.
- Tests green on macOS + Windows + Linux in CI (fixtures only).
- Hana runs it against real OG data, inspects, approves, installs as `nell`.

---

*End of spec. Implementation plan to follow via superpowers:writing-plans.*
