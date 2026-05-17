# Audit fix pack — close the confirmed Important bugs

**Date:** 2026-05-05
**Status:** Design — pending implementation plan
**Owner:** Hana
**References:** Session audit transcript 2026-05-05; full audit report from the `superpowers:code-reviewer` agent

## Why

The 2026-05-05 session audit identified six Important issues. Stress-test exercise confirmed three as real bugs with empirical reproduction:

- **I-1 (`save_work` non-atomic)** — write_markdown succeeds, store.insert raises → orphan markdown file on disk, no SQLite row, work unreachable via API.
- **I-2 (`search_works` 500 on malformed FTS5)** — six malformed query patterns all raise `sqlite3.OperationalError` through to the bridge HTTP layer as 500.
- **I-4 (`--force` silent clobber)** — `nell migrate --output X --force` against an unrelated directory deletes the directory's contents without warning. Confirmed against an innocent test dir with personal-looking files.

Plus two not-empirically-tested but audit-confirmed:

- **I-3 (exception-handling inconsistency)** — six `try/except` blocks in `run_migrate` use four different patterns (`except Exception`, `except OSError`, `except (ValueError, OSError)`, `except (ValueError, FileNotFoundError, OSError)`). Programming bugs in some migrators surface as crashes; in others they get absorbed into report text.
- **I-6 (stale "nine tools" references)** — multiple tests + a docstring still hardcode "9 registered tools" after we shipped 4 more (now 13).

This spec ships a single PR that closes all five. Lower-priority items (I-5 concurrency lock, M-1 through M-5) are deferred to a follow-up PR; tracked as known limitations.

## What ships

| Fix | File | Change |
|---|---|---|
| I-1 | `brain/tools/impls/save_work.py` | Reorder: insert into store FIRST, then write_markdown. On store failure, no file written. On markdown failure, the row exists pointing at a missing file — `read_work` already handles this case ("indexed but file missing") |
| I-2 | `brain/works/store.py` | Wrap `WorksStore.search` execute in `try: ... except sqlite3.OperationalError: return []`. Malformed FTS5 → empty results, no exception |
| I-3 | `brain/migrator/cli.py` | Standardize all six `run_migrate` try-blocks on narrow exception tuples (`OSError | json.JSONDecodeError | ValueError`). Bare `except Exception` becomes `except (OSError, json.JSONDecodeError, ValueError)`. Programming bugs surface as crashes; expected I/O failures absorb into report |
| I-4 | `brain/migrator/cli.py` | In `_ensure_clobber_safe(work_dir, force, kind="output directory")`: when `force=True` and dir non-empty, require dir to contain at least one of `{migration-report.md, source-manifest.json, memories.db}` (i.e., evidence of a prior migration target). Else raise `FileExistsError` regardless of `--force`. Custom error message names the missing markers |
| I-6 | `brain/tools/__init__.py`, `brain/tools/dispatch.py`, `tests/unit/brain/tools/test_schemas.py`, `tests/unit/brain/tools/test_dispatch.py`, `tests/unit/brain/mcp_server/test_tools.py` | Replace hardcoded "nine" / `9` literals with the appropriate dynamic reference (`len(_DISPATCH)`, `len(NELL_TOOL_NAMES)`, etc.). Rename test functions e.g. `test_register_tools_advertises_all_nine` → `test_register_tools_advertises_all_dispatch_tools` |

Plus 3 new tests:

- **G-1** `tests/unit/brain/tools/test_works_tools.py::test_save_work_no_orphan_file_on_store_failure` — monkeypatch `WorksStore.insert` to raise `sqlite3.OperationalError`. Assert: no file at `data/works/<id>.md` after the failure (because store insert ran first and failed before write_markdown was reached).
- **G-2** `tests/unit/brain/works/test_store.py::test_search_returns_empty_on_malformed_fts5_query` — call `WorksStore.search(query='lighthouse"')` with unbalanced quote. Assert: returns `[]`, no exception.
- **G-3** `tests/unit/brain/migrator/test_cli.py::test_run_migrate_refuses_force_clobber_of_non_migrator_directory` — pre-populate output dir with arbitrary files, run with `--force`, assert FileExistsError + dir contents preserved.

Total estimated PR: ~50 lines production + ~80 lines tests. Test count delta: 1301 → 1304 (3 new G-tests; the I-6 renames don't change count).

## Architecture

### I-1 — `save_work` reorder

Current at `brain/tools/impls/save_work.py:55-56`:
```python
write_markdown(persona_dir, work, content=content)
WorksStore(persona_dir / "data" / "works.db").insert(work, content=content)
```

Fix:
```python
WorksStore(persona_dir / "data" / "works.db").insert(work, content=content)
write_markdown(persona_dir, work, content=content)
```

If the store insert fails: no file is written, error propagates as before, persona dir state is unchanged. If the markdown write fails after a successful store insert: `read_work` already catches `FileNotFoundError` and returns `{"error": "work {id} indexed but file missing"}`. The orphan-row case is the recoverable one — operators can manually re-run save with the same content (idempotent via content-hash dedup) to repopulate the file.

### I-2 — `WorksStore.search` exception catch

Current at `brain/works/store.py:139-148` (search method):
```python
sql = """
SELECT works.* FROM works
JOIN works_fts ON works.id = works_fts.id
WHERE works_fts MATCH ?
"""
# ... appends type filter and limit ...
with self._connect() as conn:
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_work(r) for r in rows]
```

Fix: wrap the `conn.execute` in a `try/except` that catches `sqlite3.OperationalError`:
```python
with self._connect() as conn:
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []
    return [_row_to_work(r) for r in rows]
```

The `OperationalError` is FTS5's signal for "your query syntax is bad." Returning empty results is the right behaviour: an LLM emitting a malformed query gets back "no matches" and can rephrase, instead of a stack trace.

### I-3 — exception-handling consistency in `run_migrate`

The audit identified four different patterns. Fix: standardize on `except (OSError, json.JSONDecodeError, ValueError)` for all six try-blocks (creative_dna, journal-memories, soul-candidates, reflex-log, legacy, plus the existing vocabulary/reflex_arcs/interests/crystallizations sections).

This catches the realistic I/O + parse failure modes without absorbing programming bugs (`KeyError`, `AttributeError`, `TypeError`). The `# noqa: BLE001` directive comes off too — it's no longer needed.

### I-4 — `--force` clobber safety

Current at `brain/migrator/cli.py:344-354` (`_ensure_clobber_safe`):
```python
def _ensure_clobber_safe(path: Path, force: bool, *, kind: str) -> None:
    """Refuse to clobber a non-empty dir without --force."""
    if not path.exists():
        return
    if not any(path.iterdir()):
        return  # empty dir is safe to use
    if not force:
        raise FileExistsError(
            f"{kind} {path} is not empty; pass --force to overwrite."
        )
    # --force: clobber
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
```

Fix: even with `--force`, refuse to clobber unless the dir contains a migrator marker file:
```python
_MIGRATOR_MARKER_FILES = frozenset({"migration-report.md", "source-manifest.json", "memories.db"})


def _ensure_clobber_safe(path: Path, force: bool, *, kind: str) -> None:
    """Refuse to clobber a non-empty dir without --force, and even with --force
    refuse to clobber a directory that doesn't look like a prior migration target.
    """
    if not path.exists():
        return
    if not any(path.iterdir()):
        return  # empty dir is safe to use
    if not force:
        raise FileExistsError(
            f"{kind} {path} is not empty; pass --force to overwrite."
        )
    # --force: only clobber if the directory looks like a prior migration target
    has_marker = any((path / m).exists() for m in _MIGRATOR_MARKER_FILES)
    if not has_marker:
        raise FileExistsError(
            f"{kind} {path} is not empty and does not contain any of "
            f"{sorted(_MIGRATOR_MARKER_FILES)} — refusing to clobber an "
            f"unrelated directory even with --force. Choose a different output "
            f"path or remove the directory first."
        )
    # --force + has marker: safe to clobber
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
```

The error message is operator-friendly — names the missing markers, suggests the fix.

### I-6 — stale "nine tools" references

Located via:
```
brain/tools/dispatch.py:77       — docstring "must be one of the 9 registered tools"
tests/unit/brain/tools/test_schemas.py:24    — def test_schemas_has_all_nine_tools
tests/unit/brain/tools/test_dispatch.py:131  — def test_all_nine_tools_dispatch_without_crash
tests/unit/brain/mcp_server/test_tools.py:26 — def test_register_tools_advertises_all_nine
```

Fix:
- Docstring: rephrase to reference the dispatch table (`one of the registered tools (see _DISPATCH)`).
- Test names: rename to `_all_dispatched_tools_*` and reference `len(_DISPATCH)` or iterate `NELL_TOOL_NAMES` instead of hardcoding 9.
- Inside the test bodies that already had the count fixed to 13 (during the works branch), replace literal `13` with the iterated length so future tool additions don't require yet-another bump.

## Test plan

3 new tests + 3 renamed tests. Total: 1301 → 1304 (renames don't change count; new tests add 3).

The new tests are above (G-1, G-2, G-3). The renamed tests just change function names + de-hardcode the counts; they exercise the same surface they always did.

## Backwards compatibility

- I-1 is internal reordering. The save_work API contract is unchanged — same inputs, same outputs, same error shape.
- I-2 changes search behavior: malformed queries used to crash, now return `[]`. **This is a behavior change.** No existing test relies on the exception path, but third-party code that catches `OperationalError` from `search_works` would no longer see it. Acceptable — the exception was undocumented and is the bug being fixed.
- I-3 is internal exception-class refinement. No external API change.
- I-4 is a new check that fires on a previously-permissive path. **Forkers who scripted `nell migrate --output ~/some-dir --force` against non-migration directories will now get `FileExistsError`.** This is the intended behavior — that path was the bug. Documented in the migration playbook (we'll add a sentence).
- I-6 is pure cleanup, no behavior change.

No new dependencies, no schema changes, no migrations.

## Out of scope

- **I-5 (concurrency lock)** — lockfile pattern requires choosing where to put the lock (target persona dir? OG data dir?), how to handle stale locks, integration with the daemon's existing lock pattern. Substantive — separate PR.
- **M-1 (extract save_with_backup_text)** — refactor; nice but no behavior change. Separate PR.
- **M-2, M-3, M-4, M-5, M-6** — polish. Separate PR or skip.
- **G-4, G-5, G-6** — test gaps for items not being fixed in this PR. Add when those items get fixed.
