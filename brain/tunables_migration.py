"""One-time, idempotent migration of the pre-lull idle-tuning tunable keys
into the single ``chat.idle_lull_seconds`` key (ram-spike-fix INC-6, spec §7,
ledger S25/S30/S37/S52/S71).

Retired keys (ram-spike-fix INC-6, C4):
  - ``throttle.background_min_idle_seconds`` (old default 300.0) — carried
    forward as ``chat.idle_lull_seconds``'s override IFF its value differs
    from the old default (a genuine user customization); a plain-default
    override (300.0, indistinguishable from "never customized") is dropped,
    landing on the new code default (600.0) instead of carrying a stale
    number forward as if it were still meaningful.
  - ``chat.pass2_min_idle_seconds`` — removed, value not carried (pass 2 now
    shares the one lull like every other caller).
  - ``self_model.articulate_min_idle_seconds`` — removed, value not carried
    (same reason).
  - ``judge_selftune.gate_handful_decisions`` (S71) — removed, value not
    carried (this key moved to ``brain/dev_constants.py`` in INC-1 and must
    not be a registered tunable at all, C13/C26).

Called synchronously in the bridge lifespan BEFORE the supervisor thread
starts (server.py, S37) — the migration must complete before any code can
read the (possibly still-old) file.

Fail-safe: any error during the rewrite restores the original file from the
backup taken just before writing (if the on-disk file was touched at all)
and logs a WARNING — this module must never raise into bridge startup.
Idempotent: if none of the retired keys are present, this is a no-op (no
write at all, not even a no-op rewrite) — running it twice produces
byte-identical files.

Concurrency (#529): this migration reads and rewrites the shared, cross-persona KINDLED_HOME/tunables.json, and so
do ``tunables.set_overrides`` and ``tunables.write_defaults_section``. The rewrite (read, write, verify and restore)
therefore runs under the same cross-process lock they take, with a bounded wait; the file is re-read inside the lock.
The pre-check for a retired key runs outside the lock, so an ordinary boot never locks. The shared ``.bak-migrate``
sidecar is only touched while the lock is held.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from brain import dev_constants
from brain.health.json_io import Recovery, read_json, write_json
from brain.utils.file_lock import LockTimeoutError, locked_within

logger = logging.getLogger(__name__)

_FILE_NAME = "tunables.json"
_BACKUP_SUFFIX = ".bak-migrate"

_OLD_THROTTLE_KEY = "throttle.background_min_idle_seconds"
_OLD_THROTTLE_DEFAULT = 300.0
_NEW_LULL_KEY = "chat.idle_lull_seconds"

# Keys whose VALUE is never carried forward — only ever removed.
_DROP_ONLY_KEYS = (
    "chat.pass2_min_idle_seconds",
    "self_model.articulate_min_idle_seconds",
    "judge_selftune.gate_handful_decisions",
)

# Test seam: called inside the locked window, after the re-read and just before the write (deterministic interleaving).
_LOCKED_WINDOW_HOOK: Callable[[], None] | None = None


def _read_state(path: Path) -> tuple[bytes, dict[str, Any], dict[str, Any]] | None:
    """``(original bytes, parsed file, overrides)``, or None when there is nothing to migrate or the file is unusable."""
    try:
        original_bytes = path.read_bytes()
    except FileNotFoundError:
        return None  # nothing to migrate: a fresh persona has no old keys
    except OSError as exc:
        logger.warning("tunables_migration: cannot read %s (%s) — skipping", path, exc)
        return None

    try:
        data = json.loads(original_bytes.decode("utf-8"))  # json-io: ok -- byte-exact backup needs the original bytes
    except Exception as exc:  # noqa: BLE001 — corrupt file: leave it for tunables.py's own repair
        logger.warning(
            "tunables_migration: %s unreadable (%s) — skipping (not this module's job to repair)",
            path, exc,
        )
        return None

    if not isinstance(data, dict):
        logger.warning("tunables_migration: %s is not a JSON object — skipping", path)
        return None

    overrides = data.get("overrides")
    if not isinstance(overrides, dict):
        overrides = {}
    return original_bytes, data, overrides


_RETIRED_KEYS_PRESENT = [_OLD_THROTTLE_KEY, *_DROP_ONLY_KEYS]


def migrate_idle_keys(home: Path) -> None:
    """Migrate ``home/tunables.json``'s retired idle-tuning keys in place.

    No-op (no lock, no write) when the file is absent or none of the retired keys are present in its ``overrides``: that
    cheap pre-check runs OUTSIDE the lock, so an ordinary boot pays no lock round trip. When a retired key is present the
    file is locked (the same cross-process lock ``tunables.set_overrides`` takes, bounded wait) and RE-READ inside the
    lock; the lock covers the whole read, write, verify and restore sequence, so a concurrent ``set_overrides`` can
    neither be overwritten by the write nor clobbered by the restore. The bytes written back are always the ones read
    inside the lock. If another process holds the lock for the whole wait the migration is skipped (retried at the next
    start).
    """
    path = home / _FILE_NAME
    state = _read_state(path)
    if state is None or not any(k in state[2] for k in _RETIRED_KEYS_PRESENT):
        return  # idempotent no-op — nothing to migrate, no write at all
    try:
        with locked_within(path, dev_constants.SETTINGS_FILE_LOCK_WAIT_S):
            state = _read_state(path)
            if state is None or not any(k in state[2] for k in _RETIRED_KEYS_PRESENT):
                return  # another process migrated it between the pre-check and the lock
            if _LOCKED_WINDOW_HOOK is not None:
                _LOCKED_WINDOW_HOOK()
            _rewrite(path, *state)
    except LockTimeoutError:
        logger.warning(
            "tunables_migration: another process held the lock on %s for %.1f s, so the migration was skipped (retried at the next start)",
            path, dev_constants.SETTINGS_FILE_LOCK_WAIT_S,
        )


def _rewrite(path: Path, original_bytes: bytes, data: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Rewrite the file without the retired keys. Called with the file lock held. Never raises."""
    retired_keys_present = _RETIRED_KEYS_PRESENT

    new_overrides: dict[str, Any] = {
        k: v for k, v in overrides.items() if k not in retired_keys_present
    }

    if _NEW_LULL_KEY not in overrides and _OLD_THROTTLE_KEY in overrides:
        old_val = overrides[_OLD_THROTTLE_KEY]
        if isinstance(old_val, (int, float)) and not isinstance(old_val, bool) and old_val != _OLD_THROTTLE_DEFAULT:
            new_overrides[_NEW_LULL_KEY] = float(old_val)
        # else: old value equals the old default (or is malformed) — treated
        # as "never customized"; the new code default (600.0) applies, no
        # override key written.

    new_data = dict(data)
    new_data["overrides"] = new_overrides

    backup_path = path.with_name(path.name + _BACKUP_SUFFIX)
    try:
        backup_path.write_bytes(original_bytes)
    except OSError as exc:
        logger.warning(
            "tunables_migration: could not write backup %s (%s) — skipping migration "
            "(original left untouched)",
            backup_path, exc,
        )
        return

    try:
        write_json(path, new_data, indent=2, sort_keys=True, fsync=False, nonfinite="null")

        # Re-load via the SAME reader tunables.py uses and verify.
        reloaded = read_json(path, recover=Recovery.RAISE, expect=dict)
        reloaded_overrides = reloaded.get("overrides", {})
        for k in retired_keys_present:
            if k in reloaded_overrides:
                raise ValueError(f"retired key {k!r} still present after migration")
        if _NEW_LULL_KEY in new_overrides:
            if reloaded_overrides.get(_NEW_LULL_KEY) != new_overrides[_NEW_LULL_KEY]:
                raise ValueError("migrated chat.idle_lull_seconds value did not round-trip")
    except Exception as exc:  # noqa: BLE001 — fail-safe: restore original, never raise
        logger.warning(
            "tunables_migration: migration of %s failed (%s) — restoring original", path, exc,
        )
        try:
            if path.read_bytes() != original_bytes:
                path.write_bytes(original_bytes)
        except OSError as restore_exc:  # noqa: BLE001 — last-resort log only
            logger.error(
                "tunables_migration: could not restore original %s after a failed "
                "migration (%s) — the file may be in a partially-migrated state",
                path, restore_exc,
            )
        return
    finally:
        try:
            backup_path.unlink(missing_ok=True)
        except OSError:
            pass

    # the file's signature changed under tunables.py's own cache: the next get_tunable()
    # call re-reads (tunables.py:_load_overrides_locked), no extra reset needed.
    logger.info(
        "tunables_migration: migrated %s (%d retired key(s) removed%s)",
        path,
        len(retired_keys_present),
        f", chat.idle_lull_seconds={new_overrides[_NEW_LULL_KEY]!r} carried forward"
        if _NEW_LULL_KEY in new_overrides
        else "",
    )
