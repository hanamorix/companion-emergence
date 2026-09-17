# Update script — apply the current git state over an existing install (#179)

**Date:** 2026-09-13 · **Status:** approved (Hana, 2026-09-13) · **Issue:** #179

## Problem

No updater exists. A user on a bundled install (`.deb`, `.app`, `.msi`) who wants a fix
that is on `main` but not yet in a tagged release has no supported path except waiting
for the next tag. Source installs have one (`git pull && uv sync`) but nothing names it
or restarts the supervisor around it. `nell paths` resolves persona *data* locations only;
nothing exposes where the *code* is installed, so any updater must guess.

## Decisions (brainstorm, 2026-09-12/13)

| # | Decision | Chosen |
|---|---|---|
| 1 | Which install shape | **B** — both bundled and source, auto-detected |
| 2 | Where the updater lives | **A** — `scripts/update.sh` (bash, POSIX); Windows is a documented gap |
| 3 | Source ref | `main` by default, `--ref`; `--source <dir>` reuses an existing checkout |
| 4 | Dependencies (bundled) | Mirror `app/build_python_runtime.sh`: `uv export --locked` + `uv pip install --require-hashes`, then the wheel `--no-deps` |

## Design

### 1. `nell paths` gains three global keys

Added to `_paths_for_persona` in `brain/cli.py`:

| key | value |
|---|---|
| `install_root` | `sys.prefix` — bundled: `.../Resources/python-runtime`; source: `<repo>/.venv` |
| `brain_package` | `Path(brain.__file__).parent` |
| `install_kind` | `source` when `brain_package.parent / "pyproject.toml"` exists, else `bundled` |

`install_kind` is a plain string carried in the same dict; `--json` renders it with the
usual `{path, exists}` shape (exists is meaningless there and stays `false`). The single-key
form `nell paths install_kind` works unchanged.

### 2. `scripts/update.sh` (safety tier 3 — live persona, mutating)

```
update.sh [--persona NAME] [--ref REF] [--source DIR] [--nell PATH]
          [--no-restart] [--dry-run] [--allow-app-rewrite]
```

1. **Preflight.** `git` and `uv` on `PATH`, else one clear message and exit 2. Resolve
   `nell`: `--nell`, then `PATH`, then `~/.local/bin/nell`. Read
   `nell paths --json --persona P` → `install_root`, `brain_package`, `install_kind`.
2. **Source.** `--source DIR` (must contain `pyproject.toml`) or
   `git clone --depth 1 --branch REF <origin-url> <tmp>`. The origin URL is the repo's
   canonical `https://github.com/hanamorix/companion-emergence`.
3. **Stop.** `nell service stop --persona P` unless `--no-restart`.
4. **Apply.**
   - *source:* `git -C <repo> pull --ff-only` (only when no `--source`, i.e. the install
     IS the checkout) then `uv sync --all-extras` in `<repo>`.
   - *bundled:* in the source tree `uv build --wheel`; `uv export --format requirements-txt
     --no-dev --no-emit-project --locked`; `uv pip install --python <install_root>/bin/python3
     --require-hashes -r req.txt`; `uv pip install --python ... --no-deps <wheel>`; restore
     `<install_root>/bin/nell` from the copy taken before the install (pip regenerates the
     entry point with a baked shebang; the shipped file is a relocatable wrapper).
   - `install_root` not writable → re-exec under `sudo` (Linux `.deb`). On macOS when
     `install_root` is inside a `.app` bundle, refuse unless `--allow-app-rewrite`
     (rewriting Resources invalidates the ad-hoc signature; Gatekeeper may re-prompt).
5. **Verify.** `nell --version` from the same binary must equal `version` in the source
   tree's `pyproject.toml`, else fail.
6. **Start.** `nell service start --persona P`, then `nell service status --persona P`.
7. **Never leave the brain down.** Any failure after step 3 runs `service start` before
   exiting non-zero.

`--dry-run` prints the command plan (one command per line, prefixed `plan:`) and executes
nothing past preflight. This is the testing seam.

### 3. Tests

- `tests/unit/test_cli_paths_install.py` — the three keys for both kinds (monkeypatch
  `brain.__file__`, `sys.prefix`); JSON shape; single-key lookup.
- `tests/unit/scripts/test_update_sh.py` — runs the script under `bash` with `--dry-run`
  against a fake `nell` shim in `tmp_path` that prints canned `paths --json`. Asserts the
  plan for source and bundled kinds, the `sudo` branch (unwritable root), the macOS `.app`
  refusal, the missing-tool message, and `--no-restart`. No real git/uv/wheel. Skipped on
  Windows (no bash), same as the other bash-script tests.

### 4. Docs

- `scripts/README.md` inventory row (tier 3).
- `README.md` "Updating" subsection after Quick start.
- `docs/releases/cross-platform-validation.md` Windows row: no updater (bash absent).
- Comment on #179 linking the PR.

## Wiring

Consumes: `nell paths`, `nell service stop/start/status`, `app/build_python_runtime.sh`'s
install recipe (copied, not shared — bash cannot import bash safely across those scripts'
`set -e` contexts; drift risk accepted and named in the script header).
Feeds: nothing in the brain. Ops surface only; no organ.

## Deferred

- `nell update` subcommand wrapping the script (YAGNI until asked) — #256.
- Windows updater (needs a PowerShell twin; runtime ships no bash) — #255.
Both filed as issues at spec time.

## Out of scope

The Tauri desktop shell's own release updater; anything that touches persona data.
