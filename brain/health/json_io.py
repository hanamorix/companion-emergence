"""The one reader and writer for every JSON and JSONL file in brain/ (#438, #436, #434).

Why one module: the app read and wrote its JSON files in dozens of places, each with its
own idea of which errors fall back (invalid UTF-8 escaped eight readers' fallbacks, #436),
what to log, how to replace a file atomically, and what to do with a torn JSONL tail (#434).
Everything file-I/O specific lives here; what a default value *means* ("no data yet" versus
"deny") stays at the call site.

Reading a whole file::

    read_json(path, default=dict)                              # fail soft, evidence logged
    read_json(path, recover=Recovery.RAISE)                    # fail closed: JsonFileError
    read_json_result(path, default=f, recover=Recovery.HEAL)   # quarantine + restore + anomaly
    read_json(path, default=dict, recover=Recovery.RESTORE)    # restore from a good backup, else raise

Recovery policies: DEFAULT returns ``default()`` and leaves the file; RAISE raises
:class:`JsonFileError`; RESTORE restores the freshest valid backup (or a valid ``fallbacks``
sibling) and otherwise raises, leaving the corrupt file exactly where it is (a corrupt budget
file must keep failing closed, never reset to an empty counter); HEAL is RESTORE that resets
to ``default()`` when nothing is valid. A file that merely cannot be *read* (kind ``io``) is
never quarantined by any policy.

Writing: :func:`write_json` (atomic, optional ``.bak`` chain, optional verify-after-write),
:func:`append_jsonl` (one write, torn-tail safe), :func:`write_jsonl`. Reading JSONL:
:func:`iter_jsonl`, :func:`read_jsonl`, :func:`read_jsonl_tail`, :func:`parse_jsonl`.

Output is always UTF-8 and pure ASCII (``ensure_ascii``): a torn write cannot cut a
multi-byte character (#433), a raw U+2028 cannot split a line for a ``splitlines()`` reader,
and a lone surrogate cannot make a persistence write raise. One constant, ``_ENSURE_ASCII``.

Locks: appends use ``file_lock(path)`` (the shared ``<name>.lock`` sidecar). The heal and any
save of a file with a backup chain use their own sidecar ``<name>.heal.lock``: ``<name>.lock``
is held for a bridge's whole life (``bridge.json.lock``) and for a whole heartbeat tick
(``heartbeat_state.json.lock``), so taking it here would block or deadlock.

The heal is deliberately its own implementation (not ``attempt_heal._heal_from_baks``) so the
legacy function and its fifteen callers do not change; it is deleted together with
``attempt_heal`` when the last caller has migrated.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from brain.health.anomaly import BrainAnomaly
from brain.health.attempt_heal import (
    _REPLACE_RETRY_SLEEP_S,
    _classify_cause,
    _filename_safe_timestamp,
    _replace_with_retry,
    save_with_backup_text,
)
from brain.utils.file_lock import file_lock
from brain.utils.log_throttle import warn_throttled

logger = logging.getLogger(__name__)

_ENSURE_ASCII = True
_IS_WINDOWS = os.name == "nt"
# A read holds the file for microseconds, so it rides out a rename in progress with far fewer
# attempts than the writer's 8: three attempts, sleeping 0.05 s then 0.10 s (Windows only).
_READ_ATTEMPTS = 3
_BOM = b"\xef\xbb\xbf"
_TAIL_CHUNK = 8192
_PREVIEW = 201  # characters of exception text / line preview in a log message
_DETAIL = 500  # anomaly detail cap (as attempt_heal)

# Test seams and switches. Nothing in brain/ reads them but this module; tests patch them to
# inject an interleaving or to run a guard switched off (the failing baseline).
_HEAL_REVALIDATE = True
_SAVE_TAKES_HEAL_LOCK = True
_HEAL_LOCK_USES_SHARED_SIDECAR = False


def _seam(name: str, path: Path) -> None:
    """Called at named points inside the heal and the append; the default does nothing."""


_NO_DEFAULT: Any = object()


class Recovery(enum.Enum):
    DEFAULT = "default"
    RAISE = "raise"
    RESTORE = "restore"
    HEAL = "heal"


class JsonFileError(OSError, ValueError):
    """A JSON file could not be used. ``kind`` is one of missing, io, empty, malformed, encoding,
    shape, schema, line.

    An OSError *and* a ValueError so a legacy ``except (OSError, ValueError)`` still catches it.
    """

    def __init__(
        self, path: Path | str, kind: str, detail: str = "", line: int | None = None
    ) -> None:
        message = f"{path}: {kind}"
        if detail:
            message += f": {detail}"
        if line is not None:
            message += f" (line {line})"
        super().__init__(message)
        self.path = Path(path)
        self.kind = kind
        self.detail = detail
        self.line = line

    def __str__(self) -> str:
        return str(self.args[0])

    def __reduce__(self):
        return (_rebuild_error, (type(self), str(self.path), self.kind, self.detail, self.line))


class JsonFileNotFound(JsonFileError, FileNotFoundError):  # noqa: N818 - named for what it is
    """kind == "missing", also a FileNotFoundError for callers that let that escape before."""


def _rebuild_error(cls, path, kind, detail, line):
    return cls(path, kind, detail, line)


@dataclass(frozen=True)
class JsonRead:
    data: Any
    outcome: Literal["ok", "missing", "defaulted", "restored", "reset"]
    kind: str | None = None
    anomaly: BrainAnomaly | None = None


# --------------------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------------------


def _short(exc: object) -> str:
    return str(exc)[:_PREVIEW]


def _report(path: Path, kind: str, exc: object, action: str) -> None:
    """One WARNING per (path, kind) per throttle interval (#429's helper), DEBUG between; with traceback."""
    warn_throttled(
        logger,
        f"json_io:{path}:{kind}",
        "json file %s unusable (%s: %s); %s",
        path,
        kind,
        _short(exc),
        action,
        exc_info=isinstance(exc, BaseException),
    )


# --------------------------------------------------------------------------------------------------
# Whole-file read
# --------------------------------------------------------------------------------------------------


class _BadContentError(Exception):
    """Internal: the bytes were read but are not usable. Never escapes this module."""

    def __init__(self, kind: str, cause: BaseException) -> None:
        super().__init__(kind)
        self.kind = kind
        self.cause = cause


def _signature(st: os.stat_result) -> tuple[int, int, int]:
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _read_bytes(path: Path) -> tuple[bytes, tuple[int, int, int]]:
    """Read the whole file. Windows retries a transient sharing violation briefly; POSIX never does."""
    attempts = _READ_ATTEMPTS if _IS_WINDOWS else 1
    for attempt in range(attempts):
        try:
            with open(path, "rb") as fh:
                sig = _signature(os.fstat(fh.fileno()))
                return fh.read(), sig
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(_REPLACE_RETRY_SLEEP_S * (attempt + 1))
    raise AssertionError("unreachable")  # pragma: no cover


def _check(
    raw: bytes,
    expect: type | tuple[type, ...] | None,
    validate: Callable[[Any], None] | None,
) -> Any:
    if raw.startswith(_BOM):
        raw = raw[len(_BOM) :]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _BadContentError("encoding", exc) from exc
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise _BadContentError("empty" if not text.strip() else "malformed", exc) from exc
    if expect is not None and not isinstance(data, expect):
        raise _BadContentError("shape", TypeError(f"expected {expect}, got {type(data).__name__}"))
    if validate is not None:
        try:
            validate(data)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise _BadContentError("schema", exc) from exc
    return data


def read_json(
    path: Path,
    *,
    default: Callable[[], Any] = _NO_DEFAULT,
    recover: Recovery = Recovery.DEFAULT,
    expect: type | tuple[type, ...] | None = None,
    validate: Callable[[Any], None] | None = None,
    fallbacks: Iterable[Path] = (),
) -> Any:
    """The data of ``path``; see :func:`read_json_result` for the arguments."""
    return read_json_result(
        path,
        default=default,
        recover=recover,
        expect=expect,
        validate=validate,
        fallbacks=fallbacks,
    ).data


def read_json_result(
    path: Path,
    *,
    default: Callable[[], Any] = _NO_DEFAULT,
    recover: Recovery = Recovery.DEFAULT,
    expect: type | tuple[type, ...] | None = None,
    validate: Callable[[Any], None] | None = None,
    fallbacks: Iterable[Path] = (),
) -> JsonRead:
    """Read a whole JSON file under a recovery policy and say what happened.

    ``default`` is a zero-argument factory (called per use). It is required for DEFAULT, RESTORE
    and HEAL; under RAISE it applies only to a missing file. ``expect`` is a type (or tuple) the
    top-level value must be; ``validate`` raises to reject. ``fallbacks`` (RESTORE/HEAL only) are
    sibling files tried in preference to the backup chain when they are valid and not older than
    the freshest valid backup. A fallback is *removed* when installed, so it must be a throwaway
    sibling (a stray ``.tmp``); one that resolves to the live file or a backup is a ValueError.
    """
    path = Path(path)
    fallbacks = tuple(Path(f) for f in fallbacks)
    if recover is not Recovery.RAISE and default is _NO_DEFAULT:
        raise TypeError(f"read_json: recover={recover.name} needs a default factory")
    _check_fallbacks(path, fallbacks)
    try:
        raw, sig = _read_bytes(path)
    except FileNotFoundError:
        if default is _NO_DEFAULT:
            raise JsonFileNotFound(path, "missing") from None
        return JsonRead(default(), "missing")
    except OSError as exc:
        if recover is Recovery.DEFAULT:
            _report(path, "io", exc, "using the default")
            return JsonRead(default(), "defaulted", "io")
        raise JsonFileError(path, "io", _short(exc)) from exc
    try:
        return JsonRead(_check(raw, expect, validate), "ok")
    except _BadContentError as bad:
        if recover is Recovery.DEFAULT:
            _report(path, bad.kind, bad.cause, "using the default")
            return JsonRead(default(), "defaulted", bad.kind)
        if recover is Recovery.RAISE:
            raise JsonFileError(path, bad.kind, _short(bad.cause)) from bad.cause
        return _heal(path, bad, sig, default, recover, expect, validate, fallbacks)


def _check_fallbacks(path: Path, fallbacks: tuple[Path, ...]) -> None:
    if not fallbacks:
        return
    forbidden = {path.resolve()} | {
        path.with_name(f"{path.name}.bak{i}").resolve() for i in (1, 2, 3, 4, 5, 6)
    }
    for f in fallbacks:
        if f.resolve() in forbidden:
            raise ValueError(
                f"fallback {f} is the live file or one of its backups; a fallback is removed when installed"
            )


# --------------------------------------------------------------------------------------------------
# Locks
# --------------------------------------------------------------------------------------------------


@contextmanager
def _heal_guard(path: Path) -> Iterator[bool]:
    """Hold ``<name>.heal.lock`` (never ``<name>.lock``); yields whether the lock is actually held.

    Only *acquiring* the lock is guarded: a filesystem that refuses flock degrades to unlocked with
    a warning, but an exception raised by the body propagates once and the body is never re-run.
    """
    stack = ExitStack()
    locked = False
    target = path if _HEAL_LOCK_USES_SHARED_SIDECAR else path.with_name(path.name + ".heal")
    try:
        stack.enter_context(file_lock(target))
        locked = True
    except OSError as exc:
        warn_throttled(
            logger,
            f"json_io:{path}:lock",
            "could not take the lock for %s (%s); continuing without it",
            path,
            _short(exc),
        )
    with stack:
        yield locked


# --------------------------------------------------------------------------------------------------
# Heal / restore
# --------------------------------------------------------------------------------------------------


class _ChangedError(Exception):
    """Internal: a destructive step's precondition no longer holds (someone else acted)."""


def _expect_unchanged(path: Path, sig: tuple[int, int, int]) -> None:
    if not _HEAL_REVALIDATE:
        return
    try:
        if _signature(path.stat()) != sig:
            raise _ChangedError(f"{path} changed")
    except FileNotFoundError:
        raise _ChangedError(f"{path} vanished") from None


def _expect_absent(path: Path) -> None:
    if _HEAL_REVALIDATE and os.path.lexists(path):
        raise _ChangedError(f"{path} appeared")


def _after_change(path: Path, expect, validate, original: _BadContentError) -> JsonRead:
    """A competing actor touched the file while we healed: report what is there now, destroy nothing."""
    try:
        raw, _sig = _read_bytes(path)
    except FileNotFoundError:
        raise JsonFileError(path, original.kind, "file vanished during the heal") from None
    except OSError as exc:
        raise JsonFileError(path, "io", _short(exc)) from exc
    try:
        return JsonRead(_check(raw, expect, validate), "ok")
    except _BadContentError as bad:
        raise JsonFileError(path, bad.kind, _short(bad.cause)) from bad.cause


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def _valid(path: Path, expect, validate) -> tuple[bool, bool]:
    """(readable, valid) for a candidate file."""
    try:
        raw, _sig = _read_bytes(path)
    except OSError:
        return False, False
    try:
        _check(raw, expect, validate)
    except _BadContentError:
        return True, False
    return True, True


def _heal(
    path: Path,
    bad: _BadContentError,
    sig: tuple[int, int, int],
    default: Callable[[], Any],
    recover: Recovery,
    expect,
    validate,
    fallbacks: tuple[Path, ...],
) -> JsonRead:
    _seam("before_heal", path)
    with _heal_guard(path):
        # Re-validate under the lock: another healer, or a writer, may have fixed the file since we judged it.
        if _HEAL_REVALIDATE:
            try:
                raw, sig = _read_bytes(path)
            except FileNotFoundError:
                return JsonRead(default(), "missing")
            except OSError as exc:
                raise JsonFileError(path, "io", _short(exc)) from exc
            try:
                return JsonRead(_check(raw, expect, validate), "ok")
            except _BadContentError as still_bad:
                bad = still_bad
        try:
            return _heal_locked(path, bad, sig, default, recover, expect, validate, fallbacks)
        except _ChangedError:
            return _after_change(path, expect, validate, bad)


def _heal_locked(path, bad, sig, default, recover, expect, validate, fallbacks) -> JsonRead:
    now = datetime.now(UTC)
    stamp = _filename_safe_timestamp(now)
    likely_cause = _classify_cause(path)  # before the live file is moved
    anomaly_kind = (
        "json_parse_error" if bad.kind in ("encoding", "empty", "malformed") else "schema_mismatch"
    )
    detail = _short_detail(bad.cause)

    # Read-only walk: which backup, and which fallback, would be installed?
    walked_corrupt: list[Path] = []
    chosen_bak: tuple[int, Path] | None = None
    for i in (1, 2, 3):
        bak = path.with_name(f"{path.name}.bak{i}")
        if not bak.exists():
            continue
        readable, ok = _valid(bak, expect, validate)
        if ok:
            chosen_bak = (i, bak)
            break
        if readable:
            walked_corrupt.append(bak)
    chosen_fallback: Path | None = None
    bak_mtime = _mtime(chosen_bak[1]) if chosen_bak else None
    for fb in fallbacks:
        if not fb.exists():
            continue
        _readable, ok = _valid(fb, expect, validate)
        if ok and (bak_mtime is None or _mtime(fb) >= bak_mtime):
            chosen_fallback = fb
            break

    if chosen_bak is None and chosen_fallback is None and recover is Recovery.RESTORE:
        raise JsonFileError(path, bad.kind, _short(bad.cause)) from bad.cause

    _expect_unchanged(path, sig)
    _seam("before_quarantine", path)
    _expect_unchanged(path, sig)
    quarantine = path.with_name(f"{path.name}.corrupt-{stamp}")
    try:
        _replace_with_retry(path, quarantine)
    except FileNotFoundError:
        raise _ChangedError(f"{path} vanished") from None
    except OSError as exc:
        raise JsonFileError(path, "io", _short(exc)) from exc
    _seam("inside_heal", path)
    for corrupt in walked_corrupt:
        try:
            _replace_with_retry(corrupt, corrupt.with_name(f"{corrupt.name}.corrupt-{stamp}"))
        except OSError:
            logger.warning("could not quarantine corrupt backup %s", corrupt, exc_info=True)

    if chosen_bak is None and chosen_fallback is None:
        data = default()
        _expect_absent(path)
        save_with_backup_text(
            path,
            json.dumps(data, indent=2, ensure_ascii=_ENSURE_ASCII) + "\n",
            keep_backups=False,
            newline="\n",
        )
        action, outcome = "reset_to_default", "reset"
        detail_out = detail
    else:
        _seam("before_install", path)
        _expect_absent(path)
        try:
            if chosen_fallback is not None:
                raw, _sig = _read_bytes(chosen_fallback)
                data = _check(raw, expect, validate)
                save_with_backup_text(
                    path, raw.decode("utf-8-sig"), keep_backups=False, newline="\n"
                )
                chosen_fallback.unlink(missing_ok=True)
                action, outcome = "restored_from_bak1", "restored"
                detail_out = f"{detail} | restored from fallback {chosen_fallback.name}"[:_DETAIL]
            else:
                index, bak = chosen_bak
                raw, _sig = _read_bytes(bak)
                data = _check(raw, expect, validate)
                _replace_with_retry(bak, path)
                action, outcome = f"restored_from_bak{index}", "restored"
                detail_out = detail
        except (
            FileNotFoundError
        ):  # the backup was rotated away (a save without the heal sidecar): someone else acted
            raise _ChangedError("candidate vanished") from None
    anomaly = BrainAnomaly(
        timestamp=now,
        file=path.name,
        kind=anomaly_kind,  # type: ignore[arg-type]
        action=action,  # type: ignore[arg-type]
        quarantine_path=quarantine.name,
        likely_cause=likely_cause,  # type: ignore[arg-type]
        detail=detail_out,
    )
    logger.warning(
        "%s: %s (%s); %s, quarantined as %s", path, bad.kind, detail, action, quarantine.name
    )
    return JsonRead(data, outcome, bad.kind, anomaly)  # type: ignore[arg-type]


def _short_detail(exc: object) -> str:
    return str(exc)[:_DETAIL]


# --------------------------------------------------------------------------------------------------
# Whole-file write
# --------------------------------------------------------------------------------------------------


def write_json(
    path: Path,
    data: Any,
    *,
    backups: int = 0,
    indent: int | None = 2,
    sort_keys: bool = False,
    encode_default: Callable[[Any], Any] | None = None,
    fsync: bool = True,
    mode: int | None = None,
    verify: bool = False,
    if_missing: bool = False,
    on_error: Literal["raise", "log"] = "raise",
) -> bool:
    """Atomically write ``data`` as JSON. True if written, False if skipped or (``on_error="log"``) failed.

    Serialization happens first and strictly (no NaN, no silent ``str()``), so a payload that cannot be
    encoded touches nothing. ``backups=N`` keeps ``.bak1..N`` (0 keeps none); a file with a backup chain
    is saved under the heal sidecar so a save cannot land inside a heal. ``mode`` is applied when the new
    file is created and to every backup. ``verify=True`` re-reads the result and restores ``.bak1`` if it
    is not what was written (never raises). ``if_missing=True`` creates the file only if it does not exist
    (False if it does). ``on_error="log"`` turns an ``OSError`` into a throttled warning and False.
    """
    text = (
        json.dumps(
            data,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=_ENSURE_ASCII,
            allow_nan=False,
            default=encode_default,
        )
        + "\n"
    )
    return _save_text(Path(path), text, backups, fsync, mode, verify, if_missing, on_error)


def _save_text(
    path: Path,
    text: str,
    backups: int,
    fsync: bool,
    mode: int | None,
    verify: bool,
    if_missing: bool,
    on_error: str,
) -> bool:
    guarded = (backups > 0 or if_missing) and _SAVE_TAKES_HEAL_LOCK
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _heal_guard(path) if guarded else nullcontext(False):
            if if_missing and os.path.lexists(path):
                return False
            save_with_backup_text(
                path,
                text,
                max(backups, 1),
                keep_backups=backups > 0,
                mode=mode,
                fsync=fsync,
                newline="\n",
            )
    except OSError as exc:
        if on_error == "raise":
            raise
        _report(path, "write", exc, "write skipped")
        return False
    if verify:
        return _verify(path, text, backups > 0)
    return True


def _verify(path: Path, text: str, has_backups: bool) -> bool:
    expected = text.encode("utf-8")
    try:
        if path.read_bytes() == expected:
            return True
        problem = "content differs from what was written"
    except OSError as exc:
        problem = f"unreadable ({_short(exc)})"
    logger.error("verify-after-write failed for %s: %s", path, problem)
    bak1 = path.with_name(f"{path.name}.bak1")
    if has_backups and bak1.exists():
        try:
            save_with_backup_text(
                path, bak1.read_bytes().decode("utf-8"), keep_backups=False, newline="\n"
            )
            logger.error("restored %s from %s", path, bak1.name)
        except (OSError, UnicodeDecodeError):
            logger.exception("could not restore %s from %s", path, bak1.name)
    return False
