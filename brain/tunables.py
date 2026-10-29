"""Ops-tier tunables: the install's settings file, ``$KINDLED_HOME/tunables.toml``.

Spec: docs/superpowers/specs/2026-07-04-ops-tunables-design.md (reshaped in #529)

Call sites register their key at import time and read at call time:

    _MY_TIMEOUT: float = tunables.register("area.my_timeout_seconds", 60.0, description="Seconds to wait for it.")
    ...
    timeout = tunables.get_tunable("area.my_timeout_seconds", _MY_TIMEOUT)

The file holds ONE flat entry per registered setting, its value pre-filled with the shipped default and the default shown as a
comment above it (the setting's one-line description, then ``# default: <value>``). It is the single source: reads and
writes go to the value. A person edits the value by hand; the app edits it in place (``set_values``) and never touches a
comment, a blank line, an entry it does not know or the file's line endings. All TOML reading and writing goes through
``brain.utils.toml_io``.

At every bridge boot ``sync_file`` adds the entries a new version registered and moves a value that still equals the
RECORDED default (the comment above it) onto the new default, because the user never changed it. A value the user changed
is left alone. Because a setting equal to the shipped default follows later default changes, nobody can pin the current
default by typing it (owner-decided, E7). Entries this process does not register are never removed (a module that registers
lazily, or only in another entry point, would lose its user's value); only an explicit retired-key rule deletes.

An unreadable file (syntax error, duplicate key, not UTF-8) is LEFT ALONE: no writer, no pass and no migration changes it. The
reader fails open to code defaults, and a notice (``tunables.notice.json``) names the failing line and column so the user can
fix that line. Ops tier ONLY. Physiology (forgetting, emotion, salience, cadences) is
fenced by the user-surface principle and must not gain keys here, with
ONE named exception: ``chat.idle_lull_seconds`` (below, ``CHAT_IDLE_LULL_SECONDS``),
the single ops-timing key that gates every background/cadence caller via
``is_chat_idle`` (brain/bridge/cli_throttle.py). No other ops-timing key
may be registered here (S51/I7/C27).
Fail-open throughout: this module never raises into a turn.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain import dev_constants, paths
from brain.health.json_io import read_json, write_json
from brain.utils import toml_io
from brain.utils.file_lock import LockTimeoutError, file_lock, locked_within
from brain.utils.log_throttle import warn_throttled

logger = logging.getLogger(__name__)

_FILE_NAME = "tunables.toml"
_NOTICE_FILE_NAME = "tunables.notice.json"

_lock = threading.Lock()
_registry: dict[str, Any] = {}
_descriptions: dict[str, str] = {}
_values_cache: dict[str, Any] = {}
# What the cached values were parsed from: (st_mtime_ns, st_size, st_ino). Writes replace the file by rename, so the
# inode changes on every write and two writes inside one coarse timestamp tick are still told apart.
_cache_sig: tuple[int, int, int] | None = None
_io_noted_sig: tuple[int, int, int] | None = None
_logged_sig: tuple[int, int, int] | None = None
_warned_keys: set[str] = set()

# Test seam: called inside the locked window of a read-modify-write accessor, just before it writes. Tests inject the
# competing write here (deterministic interleaving). None in production.
_LOCKED_WINDOW_HOOK: Callable[[], None] | None = None

_HEADER = (
    "# Settings for this install. Safe to edit by hand.\n"
    '# The comments above each setting say what it does and show its shipped default ("# default: ...").'
    " Change the value to change the setting.\n"
    "# A setting you have not changed follows the shipped default when a new version changes it; a setting you have changed\n"
    "# is left alone. Keep one setting per line, outside any [section]. Comments you add are kept: put them above a setting's\n"
    "# description or after its value.\n"
    '# Do not edit the description and "# default:" lines above each setting: the app keeps them up to date, and the\n'
    '# "# default:" line is how a setting you changed is told from one you did not.\n'
)
_DEFAULT_PREFIX = "default:"


class TunablesWriteError(OSError):
    """``tunables.toml`` could not be changed: unreadable, an entry the editor will not touch, unwritable, or the lock wait expired.

    Nothing was written. The text names the file and, for a file that does not parse, the failing line and column. A writer
    never repairs or resets the file: the user's text is never lost to a settings write.
    """


def _file_path() -> Path:
    return paths.get_home() / _FILE_NAME


def register(key: str, default: Any, description: str = "") -> Any:
    """Record (key, default) and its one-line description in the registry; return default unchanged.

    The description is what a person reads above the setting in ``tunables.toml``. It lives in a parallel dict so the
    ``_registry`` key -> default shape stays as it was.
    """
    with _lock:
        _registry[key] = default
        if description:
            _descriptions[key] = description
    return default


# The ONE ops-timing key this module's docstring fence allows (S47/S51/S68/C27):
# the chat-idle lull duration. Read in exactly one place —
# brain/bridge/cli_throttle.py's private `_lull_seconds()`, called only from
# `is_chat_idle` (C4(b)) — never re-exported or read elsewhere.
CHAT_IDLE_LULL_SECONDS: float = register(
    "chat.idle_lull_seconds", 600.0,
    description="Seconds of quiet after your last message before the kindled starts background work.",
)


def get_tunable(key: str, default: Any) -> Any:
    """Return the file's value for *key* if present and type-compatible, else *default*."""
    with _lock:
        _registry.setdefault(key, default)
        values = _load_values_locked()
    if key not in values:
        return default
    value = values[key]
    if not _type_ok(default, value):
        _warn_key(key, default, value)
        return default
    if isinstance(default, float) and isinstance(value, int):
        return float(value)
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)  # TOML has arrays, not tuples: an untouched key reads as the code default's own type
    return value


def positive_or_default(key: str, value: Any, default: Any, largest: float | None = None) -> Any:
    """Return *value* (what ``get_tunable`` returned for *key*) when it is an int or a float that is finite and above 0, else *default*.

    For a setting that is a duration or a size cap: ``get_tunable`` checks the type only, so a hand-edited ``0``, a
    negative number, ``nan`` or ``inf`` reaches the code (a loop that never waits, an upload cap that refuses everything).
    *largest*, when a reader has one, is the biggest value it can use (inclusive, for example the longest wait the platform
    allows); a value above it is rejected the same way, not clamped. Without it any finite number above 0 is kept, so a reader
    whose consumer has a limit passes it. A rejected value (a bool or a non-number too, for a direct caller) is logged once
    per key and value, naming both and the limit.
    *default* is the setting's registered default, which the caller passes the same way it passes it to ``get_tunable``.
    Only the readers that need it call this; ``register`` takes no bounds.
    """
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
        and (not isinstance(value, float) or math.isfinite(value))
        and (largest is None or value <= largest)
    ):
        return value
    try:
        shown = repr(value)
    except Exception:  # an object whose repr raises, or an int too long to print: reading a setting never raises
        shown = f"<{type(value).__name__}>"
    seen = f"{key}={shown}:not-positive-finite"
    with _lock:
        first = seen not in _warned_keys
        _warned_keys.add(seen)
    if first:
        logger.warning(
            "tunables: value %r=%s is not a finite number above 0%s; using the default %r",
            key, shown, "" if largest is None else f" and at most {largest!r}", default,
        )
    return default


def get_raw(key: str) -> Any:
    """The raw value for *key* exactly as the file holds it (any type), or None when the file has none.

    No type guard and no default: for a caller that must tell the user what text is in the file (a model setting that
    fails its grammar).
    """
    with _lock:
        values = _load_values_locked()
    return values.get(key)


def registered_keys() -> frozenset[str]:
    """The keys registered in THIS process so far (a CLI process knows only the modules it imported; the bridge imports all)."""
    with _lock:
        return frozenset(_registry)


def _type_ok(default: Any, value: Any) -> bool:
    # bool is an int subclass — check it first and strictly, both directions.
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if isinstance(default, float):
        return isinstance(value, (int, float))
    if isinstance(default, int):
        return isinstance(value, int)
    if isinstance(default, str):
        return isinstance(value, str)
    return True


def _warn_key(key: str, default: Any, value: Any) -> None:
    with _lock:
        if key in _warned_keys:
            return
        _warned_keys.add(key)
    logger.warning(
        "tunables: value %r=%r has wrong type (expected %s); ignored",
        key, value, type(default).__name__,
    )


def _unreadable_text(path: Path, res: toml_io.TomlRead) -> str:
    """Plain words for a file that could not be read: the file, the line and column, the reason, and that it was left alone."""
    if res.line is not None:
        where = f"line {res.line}" + (f", column {res.column}" if res.column is not None else "")
        return f"{path} could not be read at {where} ({res.reason}); it was left as it is"
    return f"{path} could not be read ({res.reason}); it was left as it is"


def _load_values_locked() -> dict[str, Any]:
    """Return the file's values as dotted keys, re-parsing only when the file changes (mtime_ns, size or inode)."""
    global _values_cache, _cache_sig, _io_noted_sig, _logged_sig
    path = _file_path()
    try:
        st = path.stat()
    except OSError:
        _values_cache, _cache_sig = {}, None
        return _values_cache
    sig = (st.st_mtime_ns, st.st_size, st.st_ino)
    if sig == _cache_sig:
        return _values_cache
    # An unreadable or malformed file reads as no values: the module logs it, fail-open by design. It is also persisted as the
    # "unreadable" notice (once per file change), so a hand edit that breaks the file while the bridge runs is not silently
    # ignored until the next restart (#462). The runtime never repairs or rewrites the file.
    res = toml_io.read_toml_result(path)
    if res.outcome == "io":
        # A read that failed on the filesystem (for example a Windows replace in flight) says nothing about the content:
        # keep the previous values in force and do NOT cache the failure, so the next call reads the file again.
        if _io_noted_sig != sig:
            _io_noted_sig = sig
            _note_unreadable_at_runtime(path, res)
        return _values_cache
    if res.outcome == "damaged":
        if _logged_sig != sig:
            _logged_sig = sig
            _log_unreadable(path, res)
        _note_unreadable_at_runtime(path, res)
        values: dict[str, Any] = {}
    else:
        values = toml_io.flatten_toml(res.data)
    _values_cache, _cache_sig = values, sig  # cache a CONTENT failure too, until the file changes
    return _values_cache


def file_mtime() -> float | None:
    """The settings file's modification time (epoch seconds), or None when it does not exist."""
    try:
        return _file_path().stat().st_mtime
    except OSError:
        return None


def _reset_values_cache() -> None:
    """Forget the cached values and the per-file-change log flags but keep the registry and ``_warned_keys`` (only ``_reset_for_tests`` clears those)."""
    global _values_cache, _cache_sig, _io_noted_sig, _logged_sig
    with _lock:
        _values_cache, _cache_sig, _io_noted_sig, _logged_sig = {}, None, None, None


def _reset_for_tests() -> None:
    with _lock:
        _registry.clear()
        _descriptions.clear()
        _warned_keys.clear()
    _reset_values_cache()


def _notice_path() -> Path:
    return _file_path().with_name(_NOTICE_FILE_NAME)


def _write_marker(record: dict[str, Any]) -> None:
    """Persist the notice beside tunables.toml, best effort: a failure is logged and NEVER stops the boot pass."""
    path = _notice_path()
    try:
        with file_lock(path):
            write_json(path, record, indent=2, fsync=False, nonfinite="null")
    except OSError:  # the filesystem (a lock or a write): telling the user must never break bridge startup
        logger.warning(
            "tunables: could not write the notice file %s; the app will not be told about the unreadable tunables.toml",
            path, exc_info=True,
        )


def _log_unreadable(path: Path, res: toml_io.TomlRead) -> None:
    where = ""
    if res.line is not None:
        where = f" at line {res.line}" + (f", column {res.column}" if res.column is not None else "")
    logger.warning(
        "tunables: %s could not be read%s (%s), so your settings are not in effect and the defaults are being used; "
        "the file was left as it is",
        path, where, res.reason,
    )


def _record_unreadable(path: Path, res: toml_io.TomlRead) -> None:
    """The file could not be read: remember where for every bridge, and say so in the log."""
    try:
        mtime_ns: int | None = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = None
    _write_marker({
        "kind": "unreadable",
        "at": datetime.now(UTC).isoformat(),
        "mtime_ns": mtime_ns,
        "line": res.line,
        "column": res.column,
        "reason": res.reason,
    })


def _note_unreadable_at_runtime(path: Path, res: toml_io.TomlRead) -> None:
    """The runtime reload found tunables.toml unusable: persist the "unreadable" notice once per file change (best effort)."""
    try:
        mtime_ns = path.stat().st_mtime_ns
        marker = _read_marker()
        if marker is not None and marker["kind"] == "unreadable" and marker.get("mtime_ns") == mtime_ns:
            return  # already recorded for this very file state
        _record_unreadable(path, res)
    except OSError:
        logger.warning("tunables: could not record that %s is unreadable", path, exc_info=True)


def _read_marker() -> dict[str, Any] | None:
    marker = read_json(_notice_path(), default=lambda: None, expect=dict)
    if marker is None or not isinstance(marker.get("kind"), str):
        return None
    return marker


def _marker_is_active(marker: dict[str, Any]) -> bool:
    """Is the user still unaware of / not done with what the marker says?

    Only the ``unreadable`` kind is ever acted on. The notice ends as soon as the file reads and parses: a content failure
    ends when the file is fixed, and a transient I/O failure (a Windows replace in flight, a momentary permission problem)
    ends at the next successful read even though the file's mtime never changed, because it said nothing about the content.
    A marker written by an older version (kinds ``reset`` and ``restored``, which described a JSON file the app no longer
    reads) is inactive and is removed at boot.
    """
    if marker["kind"] != "unreadable":
        return False
    path = _file_path()
    try:
        path.stat()
    except OSError:
        return True
    return toml_io.read_toml_result(path).outcome not in ("ok", "empty")


def current_notice() -> dict[str, Any] | None:
    """The persisted notice about tunables.toml, or None. A PURE READ: never writes, never raises.

    Called on every `/persona/state` poll by every bridge, so a fact recorded by whichever process found the file
    unreadable reaches the user whichever persona they are looking at. ``line`` and ``column`` are None when the parser gave none.
    """
    try:
        marker = _read_marker()
        if marker is None or not _marker_is_active(marker):
            return None
        return {
            "kind": marker["kind"],
            "line": marker.get("line"),
            "column": marker.get("column"),
            "reason": marker.get("reason"),
        }
    except Exception:  # noqa: BLE001 — the state endpoint must never fail on this
        warn_throttled(logger, "tunables:notice", "tunables: could not read the notice file; reporting none", exc_info=True)
        return None


def _drop_inactive_marker() -> None:
    """Delete the marker once it is inactive. Re-evaluated under the marker lock so a newer marker is never deleted."""
    try:
        marker = _read_marker()
        if marker is None or _marker_is_active(marker):
            return
        with file_lock(_notice_path()):
            marker = _read_marker()
            if marker is not None and not _marker_is_active(marker):
                _notice_path().unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 — housekeeping must never break bridge startup
        logger.warning("tunables: could not clear the notice file", exc_info=True)


# ---------------------------------------------------------------------------------------------------------------
# The file's layout, the boot pass and the writer
# ---------------------------------------------------------------------------------------------------------------


def _equal(a: Any, b: Any) -> bool:
    """Typed equality with numeric leniency: ``600`` equals ``600.0``; a bool equals only a bool; a tuple equals a list elementwise."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


def _default_comment(default: Any) -> str:
    return f"# {_DEFAULT_PREFIX} {toml_io.dumps_value(default)}"


def _description_lines(key: str, descriptions: dict[str, str]) -> list[str]:
    text = " ".join(descriptions.get(key, "").split())
    return [f"# {text}"] if text else []


def _recorded_default(run: list[str]) -> tuple[int | None, bool, Any]:
    """``(index of the "# default:" line in the comment run or None, whether its value parsed, the value)``."""
    for i in range(len(run) - 1, -1, -1):
        body = run[i].lstrip()[1:].strip()
        if body.startswith(_DEFAULT_PREFIX):
            text = body[len(_DEFAULT_PREFIX) :].strip()
            try:
                return i, True, toml_io.parse_toml("x = " + text)["x"]
            except toml_io.TomlFileError:
                return i, False, None
    return None, False, None


def _fresh_document() -> toml_io.TomlDocument:
    return toml_io.TomlDocument.from_bytes(_HEADER.encode("utf-8"))


def _registry_snapshot() -> tuple[dict[str, Any], dict[str, str]]:
    with _lock:
        return dict(_registry), dict(_descriptions)


def _looks_like_description(line: str) -> bool:
    """Could this comment line be a setting description the app wrote? One sentence: starts with a capital, ends with a period,
    at most 100 characters (the rules every registered description follows)."""
    body = line.lstrip()[1:].strip()
    return 0 < len(body) <= 100 and body[0].isupper() and body.endswith(".")


def _place_description(key: str, run: list[str], index: int | None, wanted: str) -> None:
    """Make ``wanted`` (the full comment line, ``# <text>``) the description directly above the entry's ``# default:`` line, in ``run`` (the comment lines above the
    entry, edited in place), without ever replacing a line the user wrote.

    Every entry carries its description directly above the default line, so the line there is the app's. The app replaces it
    only when it can tell it from a user's note: the line looks like a description and the line above it does not. In any
    other case the new description is inserted just above the default line and the existing lines stay. Comments the user
    typed above an entry, or between its description and its default, are therefore never the line that goes. An entry with
    no default line gets the description and then the default, so a note typed above it ends up above the description.
    """
    if any(line.strip() == wanted for line in run):
        return  # already there
    text = wanted
    if index is None:  # the default comment was just appended last: the description goes directly before it
        run.insert(len(run) - 1, text)
        return
    slot = index - 1
    if slot >= 0 and _looks_like_description(run[slot]) and not (slot >= 1 and _looks_like_description(run[slot - 1])):
        logger.info("tunables: %s: the description %r is replaced by the registered one", key, run[slot].lstrip()[1:].strip())
        run[slot] = text
    else:
        run.insert(index, text)


def _apply_sync(doc: toml_io.TomlDocument, registry: dict[str, Any], descriptions: dict[str, str]) -> dict[str, Any]:
    """Apply the boot-pass rules to ``doc`` in memory; return the values the pass intends to have changed or added.

    For each registered key, with ``D_new`` the code default, ``V`` the file's value and ``D_old`` the default recorded in the
    comment above it: an absent key is inserted with ``V := D_new``; ``V`` equal to ``D_old`` while ``D_new`` differs takes
    ``D_new`` (the user never touched it); a ``V`` that differs from ``D_old`` is kept, even a bad-typed one; a missing or
    unparsable ``D_old`` keeps ``V``. The recorded default is refreshed to ``D_new`` whenever it differs, and the registered
    description is kept directly above it (``_place_description``: written on the first pass, refreshed when it changes, never
    over a line the user wrote). Nothing else in the document changes, and an entry the editor cannot place (several lines,
    two spellings) is skipped with a log.
    """
    intended: dict[str, Any] = {}
    for key in sorted(registry):
        default = registry[key]
        try:
            comment = _default_comment(default)
        except toml_io.TomlFileError as exc:
            logger.warning("tunables: %s: the default cannot be written to the file (%s); skipped", key, exc.reason)
            continue
        state = doc.state(key)
        if state == "absent":
            doc.insert(key, default, [*_description_lines(key, descriptions), comment])
            intended[key] = default
            continue
        if state != "single":
            logger.warning(
                "tunables: %s is %s in tunables.toml, so the boot pass leaves it as it is",
                key, "written in two spellings" if state == "ambiguous" else "spread over several lines or in an inline table",
            )
            continue
        current = doc.values()[key]
        run = doc.comment_above(key)
        index, known, old_default = _recorded_default(run)
        if known and _equal(current, old_default) and not _equal(old_default, default):
            doc.set_value(key, default)
            intended[key] = default
        new_run = list(run)
        if not known or not _equal(old_default, default):
            if index is None:
                new_run.append(comment)
            else:
                new_run[index] = comment
        wanted = _description_lines(key, descriptions)
        if wanted:
            _place_description(key, new_run, index, wanted[0])
        if new_run != run:
            doc.set_comment_above(key, new_run)
    return intended


def sync_file() -> None:
    """The boot pass: create ``tunables.toml`` with every registered key, add new keys, move untouched values onto new defaults.

    Called once at bridge boot. Fail-open: any error is logged and swallowed, boot must not break. The whole
    read-modify-write runs under the file's cross-process lock (the same one ``set_values`` takes), waiting at most
    ``dev_constants.SETTINGS_FILE_LOCK_WAIT_S``; if another process holds it that long the pass is skipped with a log line
    (nothing is written) and happens at the next start.

    A file that already holds exactly the result is not rewritten, so its mtime stays put: the rejected-model fallback
    (``brain.bridge.model_tier``) trusts a rejection record only while it is newer than this file, and a rewrite at every
    boot would un-trust every other running bridge's records each time one more bridge started.

    A file that does not parse is NEVER repaired, reset or set aside (owner ruling E9): the pass records the ``unreadable``
    notice (with the failing line and column, AFTER the lock is released) and returns.
    """
    path = _file_path()
    unreadable: toml_io.TomlRead | None = None
    try:
        with locked_within(path, dev_constants.SETTINGS_FILE_LOCK_WAIT_S):
            res = toml_io.read_toml_result(path)
            if res.outcome in ("damaged", "io"):
                unreadable = res
            else:
                registry, descriptions = _registry_snapshot()
                fresh = res.outcome != "ok"
                if fresh and not registry:
                    return
                doc = _fresh_document() if fresh else toml_io.TomlDocument.from_bytes(res.raw, path)
                expected = {} if fresh else dict(doc.values())
                expected.update(_apply_sync(doc, registry, descriptions))
                if not fresh and doc.to_bytes() == res.raw:
                    logger.debug("tunables: %s already holds the current defaults; not rewritten", path)
                elif not doc.verify(expected):
                    logger.error("tunables: the boot pass would not read back as intended; %s was not changed", path)
                else:
                    if _LOCKED_WINDOW_HOOK is not None:
                        _LOCKED_WINDOW_HOOK()
                    # Known and accepted: a hand edit saved in an editor at the same instant this write replaces the
                    # file can lose that one edit (an editor does not take the settings lock). The file is never
                    # corrupted. The same holds for the write in set_values.
                    toml_io.write_toml_text(path, doc.to_bytes())
                    _reset_values_cache()
    except LockTimeoutError:
        logger.warning(
            "tunables: another process held the lock on %s for %.1f s, so the settings file was not synced "
            "(it is synced at the next start)",
            path, dev_constants.SETTINGS_FILE_LOCK_WAIT_S,
        )
        return
    except Exception:  # noqa: BLE001 — never break bridge startup
        logger.exception("tunables: sync_file failed (ignored)")
        return
    if unreadable is not None:
        _log_unreadable(path, unreadable)
        _record_unreadable(path, unreadable)
    _drop_inactive_marker()


def set_values(updates: dict[str, Any], *, only_if_default: bool = False) -> dict[str, Any]:
    """Change the named settings in ``tunables.toml`` IN PLACE and return the ones actually written (key -> value).

    A value of ``None`` resets the setting to its registered default (the entry stays; an unregistered key raises
    ``ValueError``). A value equal to the current one writes nothing, and a call that writes nothing does not touch the file
    (no lock churn beyond the read, no mtime change, a missing file is not created). With ``only_if_default=True`` a key is
    written only while its current value still equals its registered default (an absent key counts as the default), so a
    value the user already changed is left alone.

    Under ONE cross-process lock (taken with a bounded wait, never re-entrant: nothing may call this while holding the
    tunables lock) it re-reads the file, edits only the value text of each named entry (every other byte stays), inserts an
    entry that is missing, creates the file with the ``sync_file`` content when it is absent or zero bytes, checks that the
    result reads back as intended, writes it atomically and durably (fsynced) and resets this process's cache.

    Raises ``TunablesWriteError`` and writes nothing when the file does not parse (the text names the line and column; it is
    never repaired here), an entry cannot be edited in place (two spellings, several lines), the file cannot be written, or
    the lock wait expired.
    """
    registry, descriptions = _registry_snapshot()
    for key, value in updates.items():
        if value is None and key not in registry:
            raise ValueError(f"{key} is not a registered setting, so there is no default to reset it to")
        try:
            toml_io.dumps_value(registry[key] if value is None else value)  # a value the file cannot hold fails here, before any lock
        except toml_io.TomlFileError as exc:
            raise TunablesWriteError(f"{key}: the value cannot be written to the settings file ({exc.reason or exc}); nothing was written") from exc
    path = _file_path()
    try:
        with locked_within(path, dev_constants.SETTINGS_FILE_LOCK_WAIT_S):
            res = toml_io.read_toml_result(path)
            if res.outcome in ("damaged", "io"):
                raise TunablesWriteError(_unreadable_text(path, res))
            if res.outcome == "ok":
                doc = toml_io.TomlDocument.from_bytes(res.raw, path)
            else:
                doc = _fresh_document()
                _apply_sync(doc, registry, descriptions)
            current = doc.values()
            expected = dict(current)
            written: dict[str, Any] = {}
            for key, value in updates.items():
                target = registry[key] if value is None else value
                if only_if_default:
                    if key in registry:
                        if key in current and not _equal(current[key], registry[key]):
                            continue
                    elif key in current:
                        continue
                if key in current:
                    if toml_io.same_value(current[key], target):
                        continue
                    doc.set_value(key, target)
                else:
                    if key in registry and _equal(registry[key], target):
                        continue  # absent means "the default": nothing to write
                    comments = [*_description_lines(key, descriptions), _default_comment(registry[key])] if key in registry else []
                    doc.insert(key, target, comments)
                expected[key] = target
                written[key] = target
            if not written:
                return {}
            if _LOCKED_WINDOW_HOOK is not None:
                _LOCKED_WINDOW_HOOK()
            if not doc.verify(expected):
                raise TunablesWriteError(f"the change to {path} would not read back as intended; nothing was written")
            # A hand edit saved in an editor at the same instant can be lost (see the note in sync_file); never corrupted.
            toml_io.write_toml_text(path, doc.to_bytes())
    except LockTimeoutError as exc:
        raise TunablesWriteError(
            f"could not lock {path} within {dev_constants.SETTINGS_FILE_LOCK_WAIT_S:g} s (another process holds it); "
            "nothing was written"
        ) from exc
    except TunablesWriteError:
        raise
    except toml_io.TomlFileError as exc:
        raise TunablesWriteError(f"{path} was not changed: {exc.reason or exc}") from exc
    except OSError as exc:
        raise TunablesWriteError(f"could not write {path}: {exc}") from exc
    _reset_values_cache()
    return written
