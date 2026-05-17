# Migrator: legacy preservation of biographical OG files

**Date:** 2026-05-04
**Status:** Design — pending implementation plan
**Owner:** Hana
**Layer:** 2A of the migration audit (see brainstorm transcript 2026-05-04)
**Closes:** 11 Tier-1 + 3 Tier-1-supplement + 2 Tier-3 OG files that would otherwise be silently dropped during `nell migrate`

## Why

The Layer 2 audit identified ~16 biographical OG NellBrain files that have **no current migrator** and **no current new-framework surface** (or have a regenerable surface where the historical snapshot still has value). Without a preservation step, every forker following the OG → companion-emergence migration playbook silently loses content like:

- `nell_journal.json` — 221 timestamped journal entries
- `nell_personality.json` — daily_rhythms (morning_state, post_writing_exhaustion patterns)
- `nell_growth.json` — opinion convictions with strength counters
- `nell_gifts.json` — story pitches given to Hana
- `nell_narratives.json` — fiction projects in progress (e.g. "The Tyrant and the Difficult Boy", chapter 5)
- `nell_surprises.json` — planted surprises with bloom triggers
- … and ten others

Each of these deserves a real framework feature eventually (that's Layer 2C — a multi-PR backlog). But until those features exist, **dropping the content is the wrong default**. This spec ships a single small preservation step that copies the files verbatim to `persona/<name>/legacy/` so they live on disk for future migrators to absorb (or for a forker to read manually).

The principle aligns with framework spec §0.1 and §0.4 — the brain handles physiology naturally, but the framework MUST not destroy biographical content during migration.

## What ships

A new module `brain/migrator/og_legacy.py` with:

- `LEGACY_FILES: tuple[str, ...]` — frozen positive list of 16 OG filenames
- `migrate_legacy_files(*, og_data_dir: Path, persona_dir: Path) -> tuple[list[str], list[str]]` — copies present files to `persona_dir/legacy/`, returns `(preserved, missing)` lists of basenames

Wired into `run_migrate` between journal memories and post-run re-stat. Two new fields on `MigrationReport`. One new line in `format_report`.

## The positive list (16 files)

Defined as a frozen tuple in `brain/migrator/og_legacy.py`:

```python
LEGACY_FILES: tuple[str, ...] = (
    # Tier 1 — biographical, no migrator, no surface (11)
    "nell_journal.json",
    "nell_growth.json",
    "nell_gifts.json",
    "nell_narratives.json",
    "nell_surprises.json",
    "nell_outbox.json",
    "nell_personality.json",
    "emotion_blends.json",
    "nell_emotion_vocabulary.json",
    "nell_body_state.json",
    "nell_heartbeat_log.json",
    # Tier 1 supplements (3)
    "nell_growth_log.jsonl",        # pairs with nell_growth.json
    "behavioral_log.jsonl",         # OG biographical record (different scope from new framework)
    "soul_audit.jsonl",             # audit history of crystallization decisions
    # Tier 3 — regenerable, but historical snapshot is biographical (2)
    "self_model.json",
    "nell_style_fingerprint.json",
)
```

Why these and not others (positive-list rationale):
- **Excluded — Tier 2:** `soul_candidates.jsonl`, `nell_reflex_log.json`. These get real schema migrators in Layer 2B; preserving them as legacy would just double-write.
- **Excluded — Tier 4 runtime state, logs, caches, system files:** ~22 files (`nell_log.jsonl`, `daemon_state.json`, `memory_embeddings.npy`, etc.). Not biographical. Auto-regenerated or genuinely "starts fresh."

## Architecture

### `brain/migrator/og_legacy.py` (new)

```python
"""brain.migrator.og_legacy — verbatim preservation of OG NellBrain files.

Copies a positive list of biographical OG files to persona/<name>/legacy/<basename>
during nell migrate. Pure verbatim byte copy — no JSON parsing, no schema
validation, no transformation. The whole point is "do not lose content";
broken files preserve their broken bytes for future migrator-authors.

Files in LEGACY_FILES that are missing from the OG dir are silently skipped
and counted in the migration report as 'missing'. This is the typical case
for forkers who didn't use every OG NellBrain feature.

Idempotent in the simple sense: re-running overwrites with the same content.
The migrator's --force flag handles the broader install-rerun case at a
higher level (backup + atomic rename of the whole persona dir).
"""
from __future__ import annotations

from pathlib import Path

LEGACY_FILES: tuple[str, ...] = (
    # ... (full tuple as above)
)


def migrate_legacy_files(
    *,
    og_data_dir: Path,
    persona_dir: Path,
) -> tuple[list[str], list[str]]:
    """Copy each LEGACY_FILES entry from og_data_dir to persona_dir/legacy/.

    Args:
        og_data_dir: The OG NellBrain `data/` directory (where memories_v2.json lives).
        persona_dir: The new framework persona dir (where memories.db will live).

    Returns:
        (preserved, missing) — both lists of basenames. Order matches LEGACY_FILES
        for deterministic test assertions.

    Raises:
        Nothing under normal operation. OSError propagates if persona_dir is
        unwritable (e.g. permissions); that's a hard environmental failure
        the existing migrator would also surface.
    """
    legacy_dir = persona_dir / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    preserved: list[str] = []
    missing: list[str] = []
    for name in LEGACY_FILES:
        src = og_data_dir / name
        if not src.exists():
            missing.append(name)
            continue
        dest = legacy_dir / name
        dest.write_bytes(src.read_bytes())
        preserved.append(name)
    return preserved, missing
```

### `brain/migrator/cli.py` — wire it in

After the `# ---- journal memories (reflex_journal → journal_entry retag) ----` section (just shipped in Layer 1) and before `# ---- post-run source re-stat ----`, insert:

```python
    # ---- legacy preservation ----
    legacy_files_preserved = 0
    legacy_files_missing = 0
    legacy_skipped_reason: str | None = None
    try:
        preserved, missing = migrate_legacy_files(
            og_data_dir=args.input_dir,
            persona_dir=work_dir,
        )
        legacy_files_preserved = len(preserved)
        legacy_files_missing = len(missing)
    except OSError as exc:
        legacy_skipped_reason = f"copy_error: {exc}"
```

The `try/except OSError` is narrower than the existing migrators' `try/except Exception` because `migrate_legacy_files` only raises OSError under environmental failure (no JSON parsing, no schema validation, no `# noqa: BLE001` needed). Match the established pattern of catching the realistic failure mode.

Add the import near the existing migrator imports:

```python
from brain.migrator.og_legacy import migrate_legacy_files
```

Pass three new kwargs when constructing `MigrationReport(...)` at line ~227:

```python
        legacy_files_preserved=legacy_files_preserved,
        legacy_files_missing=legacy_files_missing,
        legacy_skipped_reason=legacy_skipped_reason,
```

### `brain/migrator/report.py` — schema + format

Add three new fields to `MigrationReport` after the journal_memories fields:

```python
    legacy_files_preserved: int = 0
    legacy_files_missing: int = 0
    legacy_skipped_reason: str | None = None
```

Add one new `lines.append(...)` block to `format_report` between the Journal line and the Elapsed line:

```python
    lines.append(
        f"  Legacy files:   {report.legacy_files_preserved:,} preserved, "
        f"{report.legacy_files_missing:,} missing"
        + (
            f" (skipped: {report.legacy_skipped_reason})"
            if report.legacy_skipped_reason
            else ""
        )
    )
```

## Behavior contract

**Verbatim preservation.** `Path.read_bytes() → Path.write_bytes()`. No JSON parsing. No schema validation. The file's exact bytes go to disk under `persona_dir/legacy/<basename>`.

**Missing files are not errors.** The OG file simply doesn't exist on this forker's system. Counted in `missing`. Not logged. Not surfaced to user beyond the "X missing" line in the report.

**Corrupt files are preserved as-is.** If a forker's `nell_journal.json` is malformed JSON, it lands at `legacy/nell_journal.json` with the malformed bytes intact. The future Layer 2C migrator (when journal becomes a framework feature) inherits the corruption and can decide how to handle it. Layer 2A's job is "don't drop content," not "validate and reject."

**Non-JSON files work fine.** `.jsonl` files (`behavioral_log.jsonl`, `soul_audit.jsonl`, `nell_growth_log.jsonl`) and any other text/binary content go through the same `read_bytes/write_bytes` path. No special casing.

**Re-runs overwrite.** Running `nell migrate --force --install-as ...` twice produces the same `legacy/` content. The outer `--force` already gates persona-level reruns by backing up the existing persona dir; we don't need versioned legacy backups.

**No silent failure modes.** OSError on write (e.g. permission denied) is captured into `legacy_skipped_reason` and surfaced in the report. Existing migrator pattern.

## Test plan

5 new tests, target 1287 (1282 + 5).

**`tests/unit/brain/migrator/test_og_legacy.py`** (new file):

```python
def test_legacy_copies_present_files(tmp_path: Path) -> None:
    """OG with a subset of LEGACY_FILES → those land in persona_dir/legacy/."""
    og_data = tmp_path / "og_data"
    og_data.mkdir()
    (og_data / "nell_journal.json").write_text('{"entries": []}')
    (og_data / "nell_gifts.json").write_text('{"gifts": []}')
    (og_data / "nell_personality.json").write_text('{"version": "1.0"}')

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    preserved, missing = migrate_legacy_files(
        og_data_dir=og_data, persona_dir=persona_dir
    )
    assert set(preserved) == {"nell_journal.json", "nell_gifts.json", "nell_personality.json"}
    assert "nell_journal.json" not in missing
    assert "nell_outbox.json" in missing  # not seeded → in missing
    assert (persona_dir / "legacy" / "nell_journal.json").read_text() == '{"entries": []}'


def test_legacy_handles_jsonl_files(tmp_path: Path) -> None:
    """Non-JSON content (behavioral_log.jsonl) preserves byte-for-byte."""
    og_data = tmp_path / "og_data"
    og_data.mkdir()
    raw_jsonl = b'{"event": "a"}\n{"event": "b"}\n'
    (og_data / "behavioral_log.jsonl").write_bytes(raw_jsonl)

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    preserved, _ = migrate_legacy_files(og_data_dir=og_data, persona_dir=persona_dir)
    assert "behavioral_log.jsonl" in preserved
    assert (persona_dir / "legacy" / "behavioral_log.jsonl").read_bytes() == raw_jsonl


def test_legacy_overwrites_on_rerun(tmp_path: Path) -> None:
    """Calling twice produces the same final state; no error."""
    og_data = tmp_path / "og_data"
    og_data.mkdir()
    (og_data / "nell_journal.json").write_text("v1")
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    migrate_legacy_files(og_data_dir=og_data, persona_dir=persona_dir)
    (og_data / "nell_journal.json").write_text("v2")
    preserved, _ = migrate_legacy_files(og_data_dir=og_data, persona_dir=persona_dir)

    assert "nell_journal.json" in preserved
    assert (persona_dir / "legacy" / "nell_journal.json").read_text() == "v2"


def test_legacy_empty_og_returns_all_missing(tmp_path: Path) -> None:
    """Fresh OG dir with none of the legacy files → all missing, none preserved."""
    og_data = tmp_path / "og_data"
    og_data.mkdir()
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    preserved, missing = migrate_legacy_files(
        og_data_dir=og_data, persona_dir=persona_dir
    )
    assert preserved == []
    assert len(missing) == len(LEGACY_FILES)
    # legacy/ should still be created (empty)
    assert (persona_dir / "legacy").exists()
```

**`tests/unit/brain/migrator/test_cli.py`** (extend):

```python
def test_run_migrate_preserves_legacy_files(tmp_path: Path) -> None:
    """End-to-end: OG with biographical files → output/legacy/<file> exists; report counts match."""
    og_data = tmp_path / "og" / "data"
    og_data.mkdir(parents=True)
    (og_data / "memories_v2.json").write_text("[]")
    (og_data / "nell_journal.json").write_text('{"entries": [{"timestamp": "2026-01-01", "entry": "test", "private": true}]}')
    (og_data / "nell_gifts.json").write_text('{"gifts": []}')

    output = tmp_path / "out"
    args = MigrateArgs(input_dir=og_data, output_dir=output, install_as=None, force=False)
    report = run_migrate(args)

    assert report.legacy_files_preserved == 2
    assert report.legacy_files_missing == 14  # len(LEGACY_FILES) - 2
    assert report.legacy_skipped_reason is None
    assert (output / "legacy" / "nell_journal.json").exists()
    assert (output / "legacy" / "nell_gifts.json").exists()
```

**`tests/unit/brain/migrator/test_report.py`** (extend):

```python
def test_format_report_shows_legacy_line() -> None:
    """Migrated + missing legacy file counts render in the report."""
    report = MigrationReport(
        memories_migrated=0,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
        legacy_files_preserved=14,
        legacy_files_missing=2,
    )
    text = format_report(report)
    assert "Legacy files:" in text
    assert "14 preserved" in text
    assert "2 missing" in text
```

## Out of scope

- **Layer 2B** — schema migrators for `soul_candidates.jsonl` and `nell_reflex_log.json`. Separate PR; those files have new-framework counterparts and need real field-mapping migration, not legacy-preserve.
- **Layer 2C** — building real framework features for journal, growth/opinions, gifts, narratives, etc. Multi-PR backlog. Each feature absorbs the corresponding legacy file when it ships.
- **Versioned legacy backups** — re-runs overwrite. The `--force` install pattern already provides outer-level versioning via timestamped persona-dir backups.
- **Discovery / opt-in CLI** — no `nell legacy list` or similar. Forkers see the count in the migration report; they can `ls persona/<name>/legacy/` to discover content.
- **JSON schema validation** — verbatim only. The whole point is "don't drop content."

## Backwards compatibility

Purely additive. No existing test fixtures break (the three new `MigrationReport` fields default to `0`/`0`/`None`). No existing migrator behavior changes. A persona dir that doesn't have a `legacy/` subdir continues to work exactly as before — the framework never reads from `legacy/`.

## Forker docs implication

Anyone running the migration playbook now sees in their report:
```
  Legacy files:   X preserved, Y missing
```

Where the X files (whichever they have) are stashed under `persona/<name>/legacy/`. They can be told: "this content is preserved; future framework releases may absorb specific files into native features. Until then, the files live read-only in legacy/. You can `cat` them at any time."
