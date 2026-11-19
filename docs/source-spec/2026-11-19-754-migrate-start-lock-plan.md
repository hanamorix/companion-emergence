# A forced migrate install takes the start lock (#754): plan

**Spec:** `docs/source-spec/2026-11-19-754-migrate-start-lock-design.md`

**Goal:** a forced `--install-as` refuses while the persona's start lock is held, and holds that lock across the folder swap.

## Global constraints

- Tests use a temp `KINDLED_HOME`; no mocked state files.
- Text reads pass `encoding="utf-8"`; Windows behaviour is exercised by patching `install_swap._IS_WINDOWS`, not assumed.
- Full gate before commit (pytest, ruff, Windows lint, `pnpm test`, `pnpm build`), then `scripts/new_tests_fail_on_base.py --base origin/main`.

## Review focus

- A held lock is refused before any work, and again at the swap: the live folder never moves.
- Windows: our own lock handle must not be open during the rename.
- The lock file is never unlinked (`release_lock` closes the fd only), so no second inode appears.
- A waiter that wakes after the swap locks the file now at the path.
- A first install (no existing folder) behaves as before.

## Tasks

1. **`brain/migrator/install_swap.py`.** `PersonaInUseError`, `refuse_if_in_use(final_dir)`, `install_over(final_dir, work_dir) -> Path | None`. Tests in `tests/unit/brain/migrator/test_install_swap_754.py`: a held lock is refused by both functions with nothing moved; the lock is held during the renames (POSIX) and free after; the patched Windows branch releases first and turns a first-rename `PermissionError` into the refusal; a missing folder just renames.
2. **Wire the three migrators.** `cli.py` (`nellbrain`), `emergence_kit.py` and `companion_emergence.py` call `refuse_if_in_use` before any work when `--force` meets an existing folder, and `install_over` at the swap. `companion_emergence` builds in `<name>.new`. `_dispatch` catches `PersonaInUseError` → stderr, exit 1. Tests: each migrator refuses with the live folder untouched and no backup; `companion_emergence` leaves no `.new` after success; `nell migrate` exits 1 with no traceback.
3. **`acquire_lock` relocks a swapped file (POSIX).** After the lock is taken, compare `os.fstat(fd)` with `os.stat(path)` (`st_dev`, `st_ino`); on a mismatch, unlock, close and lock again. Test in `tests/bridge/` or alongside the existing lock tests: patch `fcntl.flock` so the first call swaps the persona folder, then assert the returned fd is the file now at the path.
