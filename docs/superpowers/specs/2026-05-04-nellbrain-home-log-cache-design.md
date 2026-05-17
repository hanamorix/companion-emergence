# `NELLBRAIN_HOME` honors log + cache dirs

**Date:** 2026-05-04
**Status:** Design — pending implementation plan
**Owner:** Hana
**Related:** Surfaced during the `nell supervisor` branch (Task 2 / `cmd_tail_log` test workaround). Filed as task #15.

## Why

`brain/paths.py` currently has split-personality behaviour around the `NELLBRAIN_HOME` env var:

- `get_home()` honors `NELLBRAIN_HOME` if set, else falls back to `platformdirs.user_data_path`.
- `get_persona_dir(name)` correctly nests under `get_home() / "personas" / name` — so personas follow the override.
- `get_log_dir()` and `get_cache_dir()` go straight to `platformdirs.user_log_path` / `user_cache_path` — they ignore `NELLBRAIN_HOME` entirely.

Concretely: a developer or forker who sets `NELLBRAIN_HOME=/tmp/test-nell` to point Nell at a sandbox gets persona data redirected to `/tmp/test-nell/personas/`, but bridge logs still write to `~/Library/Logs/companion-emergence/` (macOS) or the equivalent platformdirs path. The two halves of the on-disk state are now in different places — surprising, harder to clean up, and harder to reason about during forker setup or test isolation.

The inconsistency surfaced during the `nell supervisor` branch when `cmd_tail_log` tests had to monkeypatch `paths.get_log_dir` directly because setting `NELLBRAIN_HOME` in the test fixture didn't redirect the log path. The workaround works, but it leaks the implementation detail: a test fixture that pretends to set up an isolated home has to know about *which* path functions honor the env var.

## What ships

`get_log_dir()` and `get_cache_dir()` gain the same override branch `get_home()` already has:

- If `NELLBRAIN_HOME` is set: return `get_home() / "logs"` (and `/ "cache"` for cache).
- Otherwise: fall back to platformdirs as today.

Layout under `NELLBRAIN_HOME` becomes:

```
$NELLBRAIN_HOME/
  personas/<name>/      ← already works
  logs/                 ← new: was ~/Library/Logs/companion-emergence/ etc.
  cache/                ← new: was ~/Library/Caches/companion-emergence/ etc.
```

The `cmd_tail_log` test helper drops its `monkeypatch.setattr(paths, "get_log_dir", ...)` line — setting `NELLBRAIN_HOME` becomes sufficient.

`get_cache_dir()` has zero callers in the brain/ codebase today. Fixing it is symmetry, not a hot path. Done because the inconsistency itself is the bug — when a future caller appears, the env var should already work.

## Implementation

`brain/paths.py` — replace the two one-line bodies:

```python
def get_cache_dir() -> Path:
    """Return the cache directory (embeddings, computed matrices, etc).

    Resolution order matches get_home():
    1. NELLBRAIN_HOME / "cache" if NELLBRAIN_HOME is set
    2. platformdirs user_cache_path for the current OS
    """
    override = os.environ.get("NELLBRAIN_HOME")
    if override:
        return (Path(override).expanduser() / "cache").resolve()
    return _dirs.user_cache_path.resolve()


def get_log_dir() -> Path:
    """Return the log file directory (per-persona bridge logs etc).

    Resolution order matches get_home():
    1. NELLBRAIN_HOME / "logs" if NELLBRAIN_HOME is set
    2. platformdirs user_log_path for the current OS
    """
    override = os.environ.get("NELLBRAIN_HOME")
    if override:
        return (Path(override).expanduser() / "logs").resolve()
    return _dirs.user_log_path.resolve()
```

The override branch mirrors `get_home()` exactly except for the trailing subdir component. Same `expanduser()` for `~` support, same `.resolve()` to canonicalize symlinks (matters on Linux CI runners with symlinked home dirs — same reason `get_home()` does it).

## Test plan

**`tests/unit/brain/test_paths.py`** — extend with override-respecting tests for both functions:

```python
def test_get_log_dir_respects_env_override(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NELLBRAIN_HOME redirects get_log_dir to <HOME>/logs."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    result = paths.get_log_dir()
    assert result == (tmp_path / "logs").resolve()


def test_get_log_dir_falls_back_to_platformdirs(clean_env: None) -> None:
    """Without NELLBRAIN_HOME, get_log_dir() returns platformdirs path."""
    result = paths.get_log_dir()
    assert isinstance(result, Path)
    assert result.is_absolute()
    # Platform-specific assertion: should NOT be under any tmp_path
    assert "companion-emergence" in str(result).lower()


def test_get_cache_dir_respects_env_override(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NELLBRAIN_HOME redirects get_cache_dir to <HOME>/cache."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    result = paths.get_cache_dir()
    assert result == (tmp_path / "cache").resolve()


def test_get_cache_dir_falls_back_to_platformdirs(clean_env: None) -> None:
    """Without NELLBRAIN_HOME, get_cache_dir() returns platformdirs path."""
    result = paths.get_cache_dir()
    assert isinstance(result, Path)
    assert result.is_absolute()
```

The existing `test_get_log_dir_is_absolute_path` and `test_get_cache_dir_is_absolute_path` tests stay — they assert the type-and-shape contract regardless of env state.

**`tests/unit/brain/bridge/test_daemon_extras.py`** — simplify `_patch_paths`. The current helper sets `NELLBRAIN_HOME` AND monkeypatches `paths.get_log_dir`. After this change, only the env var is needed:

```python
def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, persona: str = "nell") -> Path:
    """Wire NELLBRAIN_HOME so get_persona_dir + get_log_dir resolve under tmp_path.

    Returns the log directory path (also created on disk).
    """
    home = tmp_path / "home"
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _make_persona(tmp_path, persona)  # creates home/personas/<persona>
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))
    return log_dir
```

The `from brain import paths` import inside the helper goes away. The cmd_tail_log tests themselves don't change — they still consume `log_dir` from the helper and write fixture log content to it.

## Documentation

**`CHANGELOG.md`** — under `## 0.0.1 - Unreleased`, add to `### Changed`:

```
- `NELLBRAIN_HOME` now also redirects `get_log_dir()` and `get_cache_dir()` (to `<HOME>/logs` and `<HOME>/cache` respectively). Previously only persona data honored the override; bridge logs and cache state went to the platformdirs default. The new layout makes a sandboxed `NELLBRAIN_HOME` actually sandboxed. Users with `NELLBRAIN_HOME` set will see bridge logs move from the platformdirs location to `$NELLBRAIN_HOME/logs/` on next bridge start.
```

No README changes — README doesn't currently mention `NELLBRAIN_HOME` layout details.

## Backwards compatibility

**For users WITHOUT `NELLBRAIN_HOME` set:** zero behaviour change. Both functions return platformdirs paths exactly as today.

**For users WITH `NELLBRAIN_HOME` set:** the bridge log file location moves from the platformdirs path to `$NELLBRAIN_HOME/logs/`. On next `nell supervisor start`, the bridge writes to the new location. Old logs at the platformdirs path are *not* migrated. This is a one-time, deliberate shift — the whole point of `NELLBRAIN_HOME` is "put all my state here," and today's behaviour silently violated that.

The project is private/local-first pre-v0.1 with a single operator (Hana) plus one or two test sandboxes. Migration friction is bounded.

## Out of scope

- Auto-migrating existing platformdirs logs/cache into `NELLBRAIN_HOME/`. YAGNI — the only realistic affected user is Hana, and old log files are throwaway.
- Adding `NELLBRAIN_LOG_DIR` / `NELLBRAIN_CACHE_DIR` for finer-grained override. The single `NELLBRAIN_HOME` knob is the contract; multiplying env vars is the wrong direction.
- Refactoring callers to share an in-tree `_resolve_with_override(subdir)` helper. Two two-line bodies don't justify a helper. If a third path function ever needs the same pattern, *then* extract.
- Documenting `NELLBRAIN_HOME` layout in README. Belongs in v0.1 contributor docs, not this PR.
