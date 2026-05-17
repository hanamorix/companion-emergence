# Migrator: wire `og_journal_dna` into `nell migrate`

**Date:** 2026-05-04
**Status:** Design — pending implementation plan
**Owner:** Hana
**Severity:** P1 — silent biographical loss for every forker following the OG → companion-emergence migration playbook
**Closes:** Layer 1 of the migration audit (see brainstorm transcript 2026-05-04)

## Why

`brain/migrator/og_journal_dna.py` ships two functions:
- `migrate_creative_dna(*, persona_dir: Path, og_root: Path) -> bool` — converts OG `nell_creative_dna.json` (two schema variants) to the new `creative_dna.json` format. Idempotent.
- `migrate_journal_memories(*, persona_dir: Path, store: MemoryStore) -> int` — retags memories from `memory_type='reflex_journal'` to `memory_type='journal_entry'` and stamps `metadata.private=True, source='reflex_arc', auto_generated=True`. Idempotent.

Both have direct unit/integration coverage at `tests/integration/brain/migrator/test_journal_dna_migration.py`. **Neither is imported or called by `brain/migrator/cli.py`.** A `nell migrate` run on an OG NellBrain with a fully-populated `nell_creative_dna.json` (Hana's: 4.8 KB literary fingerprint — em-dash lover, garnets leitmotif, post-writing exhaustion patterns) silently produces no `creative_dna.json` and leaves journal-tagged memories with the wrong `memory_type`.

The bug snuck in because the test suite covers the module functions in isolation but the CLI test (`tests/unit/brain/migrator/test_cli.py`) has zero `creative_dna` / `reflex_journal` assertions. Green CI; broken user surface.

This affects every forker who runs the migration — they inherit the same silent loss. P1 release-readiness blocker.

## What ships

Five small changes inside `brain/migrator/`:

| Change | File | Estimated size |
|---|---|---|
| Import the two functions | `brain/migrator/cli.py` | 1 line |
| Add a `# ---- creative_dna + journal ----` section to `run_migrate` between "soul" and "post-run source re-stat" | `brain/migrator/cli.py` | ~25 lines (matches existing migrator section style) |
| Pass new fields when constructing `MigrationReport` at line 227 | `brain/migrator/cli.py` | ~6 lines |
| Add 4 fields to `MigrationReport` dataclass | `brain/migrator/report.py` | 4 lines |
| Update `format_report` to render two new lines | `brain/migrator/report.py` | ~12 lines |
| Add 2 CLI tests + 1 report test | `tests/unit/brain/migrator/test_cli.py`, `tests/unit/brain/migrator/test_report.py` | ~80 lines |

Total estimated PR: ~50 lines production + ~80 lines tests.

## Architecture / wiring

Inside `run_migrate()` at `brain/migrator/cli.py`, between the existing `# ---- soul crystallizations ----` section (line ~195) and the `# ---- post-run source re-stat ----` section (line ~220), insert:

```python
# ---- creative_dna ----
creative_dna_migrated = False
creative_dna_skipped_reason: str | None = None
try:
    creative_dna_migrated = migrate_creative_dna(
        persona_dir=work_dir,
        og_root=args.input_dir.parent,
    )
    if not creative_dna_migrated:
        creative_dna_skipped_reason = "og file not present"
except Exception as exc:  # noqa: BLE001
    creative_dna_skipped_reason = f"migrate_error: {exc}"

# ---- journal memories (reflex_journal → journal_entry retag) ----
journal_memories_retagged = 0
journal_memories_skipped_reason: str | None = None
try:
    journal_memories_retagged = migrate_journal_memories(
        persona_dir=work_dir,
        store=store,
    )
except Exception as exc:  # noqa: BLE001
    journal_memories_skipped_reason = f"migrate_error: {exc}"
```

**Why this position:**
- `migrate_journal_memories` operates on the `MemoryStore` populated earlier in the function — must run after the memories section finishes.
- `migrate_creative_dna` has no order dependency, but bundling it with the journal step keeps the journal_dna module's two halves visually paired.
- Placing both before "post-run source re-stat" means the source-manifest snapshot includes any OG file mutations seen during these new migrators — same guarantee the existing migrators get.
- Placing both before report construction means the new fields can be populated cleanly.

**Why `args.input_dir.parent`:** the existing migrators take `args.input_dir` (the `data/` directory) directly. `migrate_creative_dna` was written with a different convention — it expects the NellBrain repo root and constructs `og_root / "data" / "nell_creative_dna.json"` internally. Rather than refactor the function signature (touches integration tests, broadens scope), pass `.parent` at the callsite. The convention asymmetry is documented as a follow-up cleanup.

**Error handling:** mirror the existing migrators' pattern — wrap each call in `try / except Exception` and stash the error message in the corresponding `_skipped_reason`. Same noqa BLE001 the rest of the file uses. Migration never raises to the user.

## `MigrationReport` schema delta

Add 4 fields to `brain/migrator/report.py:MigrationReport` (frozen dataclass). Position them after `crystallizations_skipped_reason` to maintain alphabetical-ish grouping with sibling migrator fields:

```python
creative_dna_migrated: bool = False
creative_dna_skipped_reason: str | None = None
journal_memories_retagged: int = 0
journal_memories_skipped_reason: str | None = None
```

All four default to "nothing happened" so the dataclass stays backwards-compatible with any existing test fixtures or callers that don't pass these.

## `format_report` delta

Add two new `lines.append(...)` calls to `format_report` in `brain/migrator/report.py`, positioned after the existing "Crystallizations" line and before "Elapsed":

```python
lines.append(
    f"  Creative DNA:   "
    + ("migrated" if report.creative_dna_migrated else "not migrated")
    + (
        f" (skipped: {report.creative_dna_skipped_reason})"
        if report.creative_dna_skipped_reason
        else ""
    )
)
lines.append(
    f"  Journal:        {report.journal_memories_retagged:,} memories retagged"
    + (
        f" (skipped: {report.journal_memories_skipped_reason})"
        if report.journal_memories_skipped_reason
        else ""
    )
)
```

**Output sample after the change** (Hana's NellBrain dry-run rerun):

```
Migration complete.

  Memories:       1,220 migrated, 0 skipped
  Hebbian edges:  4,404 migrated, 0 skipped
  Reflex arcs:    8 migrated
  Vocabulary:     25 emotions migrated
  Interests:      2 migrated
  Crystallizations: 38 migrated
  Creative DNA:   migrated
  Journal:        14 memories retagged
  Elapsed:        2.0s
```

(Numbers for Creative DNA + Journal are illustrative — actual counts vary by OG data.)

When the OG file is missing (typical fresh install case):
```
  Creative DNA:   not migrated (skipped: og file not present)
  Journal:        0 memories retagged
```

## Test plan

Three new tests, two files:

**`tests/unit/brain/migrator/test_cli.py`** (extend existing file):

```python
def test_run_migrate_writes_creative_dna_when_og_has_it(tmp_path: Path) -> None:
    """End-to-end: OG dir with nell_creative_dna.json → output has creative_dna.json."""
    og_dir = tmp_path / "og" / "data"
    og_dir.mkdir(parents=True)
    # minimal valid memories file for the existing migrators to consume
    (og_dir / "memories_v2.json").write_text("[]")
    (og_dir / "nell_creative_dna.json").write_text(json.dumps({
        "version": "1.0",
        "writing_style": {
            "core_voice": "literary",
            "strengths": ["close listening"],
            "tendencies": {"active": ["em-dashes"], "emerging": [], "fading": []},
            "influences": ["clarice lispector"],
            "avoid": [],
        },
    }))

    output = tmp_path / "out"
    args = MigrateArgs(input_dir=og_dir, output_dir=output, install_as=None, force=False)
    report = run_migrate(args)

    assert report.creative_dna_migrated is True
    assert report.creative_dna_skipped_reason is None
    assert (output / "creative_dna.json").exists()


def test_run_migrate_creative_dna_graceful_when_og_missing_file(tmp_path: Path) -> None:
    """OG dir without nell_creative_dna.json → migrate runs cleanly, report flags 'not present'."""
    og_dir = tmp_path / "og" / "data"
    og_dir.mkdir(parents=True)
    (og_dir / "memories_v2.json").write_text("[]")

    output = tmp_path / "out"
    args = MigrateArgs(input_dir=og_dir, output_dir=output, install_as=None, force=False)
    report = run_migrate(args)

    assert report.creative_dna_migrated is False
    assert report.creative_dna_skipped_reason == "og file not present"
    assert not (output / "creative_dna.json").exists()


def test_run_migrate_retags_journal_memories(tmp_path: Path) -> None:
    """Memories with memory_type='reflex_journal' get retagged to 'journal_entry'
    and metadata.private=True, source='reflex_arc', auto_generated=True."""
    og_dir = tmp_path / "og" / "data"
    og_dir.mkdir(parents=True)
    (og_dir / "memories_v2.json").write_text(json.dumps([
        {
            "id": "j1",
            "content": "a journal entry",
            "memory_type": "reflex_journal",
            "domain": "us",
            "created_at": "2026-04-01T00:00:00+00:00",
            "emotions": {"reflection": 5.0},
            "emotion_score": 5.0,
        },
        {
            "id": "m1",
            "content": "a regular conversation",
            "memory_type": "conversation",
            "domain": "us",
            "created_at": "2026-04-01T00:00:00+00:00",
            "emotions": {},
            "emotion_score": 0.0,
        },
    ]))

    output = tmp_path / "out"
    args = MigrateArgs(input_dir=og_dir, output_dir=output, install_as=None, force=False)
    report = run_migrate(args)

    assert report.journal_memories_retagged == 1
    assert report.journal_memories_skipped_reason is None

    # Verify the actual retag landed in the SQLite store
    store = MemoryStore(db_path=output / "memories.db")
    try:
        journal_mems = store.list_by_type("journal_entry", active_only=True)
        assert len(journal_mems) == 1
        assert journal_mems[0].id == "j1"
        assert journal_mems[0].metadata.get("private") is True
        assert journal_mems[0].metadata.get("source") == "reflex_arc"
        assert journal_mems[0].metadata.get("auto_generated") is True

        # Conversation-type memory must NOT be retagged
        conv_mems = store.list_by_type("conversation", active_only=True)
        assert len(conv_mems) == 1
        assert conv_mems[0].id == "m1"
    finally:
        store.close()
```

**`tests/unit/brain/migrator/test_report.py`** (extend existing file):

```python
def test_format_report_shows_creative_dna_migrated_line() -> None:
    report = MigrationReport(
        memories_migrated=0,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
        creative_dna_migrated=True,
        journal_memories_retagged=14,
    )
    text = format_report(report)
    assert "Creative DNA:" in text
    assert "migrated" in text
    assert "Journal:" in text
    assert "14 memories retagged" in text


def test_format_report_shows_skipped_reasons() -> None:
    report = MigrationReport(
        memories_migrated=0,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
        creative_dna_migrated=False,
        creative_dna_skipped_reason="og file not present",
    )
    text = format_report(report)
    assert "Creative DNA:" in text
    assert "not migrated" in text
    assert "skipped: og file not present" in text
```

**Existing tests:** none of the existing tests in `test_cli.py` or `test_report.py` should regress — the new dataclass fields all have safe defaults (`False`, `None`, `0`).

## Out of scope

- **Refactoring `migrate_creative_dna` signature** to take `og_data_dir` instead of `og_root`. Real consistency win but doubles the diff surface and touches integration tests. Filed as a follow-up; the workaround at the callsite (`args.input_dir.parent`) is unambiguous.
- **Layer 2A (legacy preservation migrator)** — separate brainstorm + spec + plan + PR.
- **Layer 2B (soul_candidates + reflex_log schema migrators)** — separate.
- **Layer 2C (framework features for Tier 1 content)** — separate, multi-PR backlog.
- **Running the install for Hana** — that's Layer 3, after all fix PRs land. One command.

## Backwards compatibility

Purely additive at the public surface. The `MigrationReport` dataclass gains four optional fields with safe defaults — any existing caller constructing a `MigrationReport` without these fields continues to work. Output `format_report` text grows by 2 lines; that's a human-readable surface, not a contract.

The `nell_creative_dna.json` migration is idempotent (per the module docstring) — re-running on a persona that already has `creative_dna.json` is a no-op. Same for journal-memory retagging (memories already at `journal_entry` type aren't found by `list_by_type("reflex_journal")`).
