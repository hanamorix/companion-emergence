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

Recovery policies: DEFAULT returns ``default()`` and leaves the file (an I/O fault too); CONTENT does the same for bad
content but raises on an I/O fault; RAISE raises
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

This module is now the only JSON heal (the legacy ``attempt_heal`` and its fifteen callers are gone). Two legacy
helpers remain for the held felt-time files only: ``attempt_heal.save_with_backup`` and
``jsonl_reader.iter_jsonl_skipping_corrupt``; ``jsonl_reader`` also still owns the line parser this module
imports. ``Recovery.CONTENT`` is for a caller whose handler swallowed bad content but never an I/O fault: a
migration never widens what a caller swallows.
"""

from __future__ import annotations

import enum
import errno
import gzip
import io
import json
import logging
import os
import shutil
import time
import zlib
from collections import deque
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
from brain.health.jsonl_reader import JsonlLineError, _parse_lines
from brain.utils.file_lock import file_lock
from brain.utils.log_throttle import warn_throttled

logger = logging.getLogger(__name__)

_ENSURE_ASCII = True
_IS_WINDOWS = os.name == "nt"
# A read holds the file for microseconds, so it rides out a rename in progress with far fewer
# attempts than the writer's 8: three attempts, sleeping 0.05 s then 0.10 s (Windows only).
_READ_ATTEMPTS = 3
_BOM = b"\xef\xbb\xbf"
# JSONDecodeError is a ValueError, and so is the error for an integer of more than 4300 digits.
_JSON_PARSE_ERRORS = (ValueError, RecursionError)
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


def _as_path(path: Path | str) -> Path:
    """``path`` as a Path without constructing one when it already is (a test that patches ``os.name`` to simulate
    Windows makes ``Path(x)`` pick WindowsPath, which cannot be instantiated on POSIX)."""
    return path if isinstance(path, Path) else Path(path)

Nonfinite = Literal["raise", "null"]


def _null_nonfinite(obj: Any, default: Callable[[Any], Any] | None, seen: frozenset[int] = frozenset()) -> Any:
    """Copy ``obj`` with every NaN/Infinity replaced by None, walking dicts, lists and tuples (never text).

    A value ``json`` cannot encode goes through ``default`` first and its result is walked too, so a hook
    that returns a dict holding a NaN is covered. A cycle raises ValueError like ``json.dumps`` does.
    """
    if isinstance(obj, float):
        return obj if obj - obj == 0.0 else None
    if obj is None or isinstance(obj, (str, int)):
        return obj
    if id(obj) in seen:
        raise ValueError("Circular reference detected")
    seen = seen | {id(obj)}
    if isinstance(obj, dict):
        return {k: _null_nonfinite(v, default, seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_null_nonfinite(v, default, seen) for v in obj]
    if default is not None:
        return _null_nonfinite(default(obj), default, seen)
    return obj  # json.dumps raises its own TypeError for it


def _dumps(
    data: Any,
    *,
    indent: int | None,
    sort_keys: bool,
    encode_default: Callable[[Any], Any] | None,
    nonfinite: Nonfinite,
) -> str:
    """The one encoder call: strict JSON, ASCII. ``nonfinite="null"`` retries a failed strict encode with NaN/Infinity as null."""
    kwargs = {
        "indent": indent,
        "sort_keys": sort_keys,
        "ensure_ascii": _ENSURE_ASCII,
        "allow_nan": False,
        "default": encode_default,
    }
    try:
        return json.dumps(data, **kwargs)
    except ValueError:
        if nonfinite != "null":
            raise
        return json.dumps(_null_nonfinite(data, encode_default), **kwargs)


class Recovery(enum.Enum):
    DEFAULT = "default"
    # CONTENT: bad content (empty, malformed, bad encoding, wrong shape or schema) takes the default exactly
    # as DEFAULT does, but an I/O fault raises JsonFileError(kind="io") exactly as RAISE does. For a caller
    # whose handler swallowed bad content and let OSError propagate: a migration never widens what a
    # caller swallows.
    CONTENT = "content"
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
        self.path = _as_path(path)
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
    except _JSON_PARSE_ERRORS as exc:
        raise _BadContentError("empty" if not text.strip() else "malformed", exc) from exc
    if expect is not None and not isinstance(data, expect):
        raise _BadContentError("shape", TypeError(f"expected {expect}, got {type(data).__name__}"))
    if validate is not None:
        try:
            validate(data)
        except (ValueError, TypeError) as exc:  # a validator rejects with these; anything else is its bug and propagates
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
    path = _as_path(path)
    fallbacks = tuple(_as_path(f) for f in fallbacks)
    if recover is not Recovery.RAISE and default is _NO_DEFAULT:
        raise TypeError(f"read_json: recover={recover.name} needs a default factory")
    _check_fallbacks(path, fallbacks)
    return _read(path, default, recover, expect, validate, fallbacks, waited=False)


def _read(path, default, recover, expect, validate, fallbacks, waited: bool) -> JsonRead:
    try:
        raw, sig = _read_bytes(path)
    except FileNotFoundError:
        return _missing(path, default, recover, expect, validate, fallbacks, waited)
    except OSError as exc:
        if recover is Recovery.DEFAULT:
            _report(path, "io", exc, "using the default")
            return JsonRead(default(), "defaulted", "io")
        raise JsonFileError(path, "io", _short(exc)) from exc
    try:
        return JsonRead(_check(raw, expect, validate), "ok")
    except _BadContentError as bad:
        if recover in (Recovery.DEFAULT, Recovery.CONTENT):
            _report(path, bad.kind, bad.cause, "using the default")
            return JsonRead(default(), "defaulted", bad.kind)
        if recover is Recovery.RAISE:
            raise JsonFileError(path, bad.kind, _short(bad.cause)) from bad.cause
        return _heal(path, bad, sig, default, recover, expect, validate, fallbacks)


def _missing(path, default, recover, expect, validate, fallbacks, waited: bool) -> JsonRead:
    """The file is not there. For RESTORE and HEAL that may be a heal in progress, not an absence.

    A module heal never makes the path absent, but a legacy ``attempt_heal`` (no sidecar) does, and a
    crash can. If the heal sidecar exists, wait for a heal in progress to finish and look again. Under
    RESTORE a valid backup is then restored instead of returning ``default()``: a fail-closed policy must
    never answer "no data yet" while a good copy exists. HEAL keeps ``attempt_heal``'s meaning (a deleted
    file is a reset, not a corruption).
    """
    if recover in (Recovery.RESTORE, Recovery.HEAL):
        if not waited and os.path.lexists(path.with_name(path.name + ".heal.lock")):
            with _heal_guard(path):
                pass
            return _read(path, default, recover, expect, validate, fallbacks, waited=True)
        if recover is Recovery.RESTORE:
            restored = _restore_missing(path, default, recover, expect, validate, fallbacks)
            if restored is not None:
                return restored
    if default is _NO_DEFAULT:
        raise JsonFileNotFound(path, "missing") from None
    return JsonRead(default(), "missing")


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


def _valid(path: Path, expect, validate) -> tuple[bool, bool, tuple[int, int, int] | None]:
    """(readable, valid, signature) for a candidate file."""
    try:
        raw, sig = _read_bytes(path)
    except OSError:
        return False, False, None
    try:
        _check(raw, expect, validate)
    except _BadContentError:
        return True, False, sig
    return True, True, sig


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
                # It vanished between our first read and the lock (a legacy healer, a manual delete). HEAL keeps
                # attempt_heal's meaning (the default); RESTORE must not say "no data yet" while a good copy exists.
                if recover is Recovery.RESTORE:
                    restored, retry = _restore_missing_locked(path, expect, validate, fallbacks)
                    if restored is not None:
                        return restored
                    if retry:
                        return _after_change(path, expect, validate, bad)
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


def _pick(path: Path, expect, validate, fallbacks: tuple[Path, ...]):
    """Read-only walk: (chosen backup, chosen fallback, corrupt backups walked past with their signatures)."""
    walked: list[tuple[Path, tuple[int, int, int]]] = []
    chosen_bak: tuple[int, Path] | None = None
    for i in (1, 2, 3):
        bak = path.with_name(f"{path.name}.bak{i}")
        if not bak.exists():
            continue
        readable, ok, bsig = _valid(bak, expect, validate)
        if ok:
            chosen_bak = (i, bak)
            break
        if readable and bsig is not None:
            walked.append((bak, bsig))
    chosen_fallback: Path | None = None
    bak_mtime = _mtime(chosen_bak[1]) if chosen_bak else None
    for fb in fallbacks:
        if not fb.exists():
            continue
        _readable, ok, _fsig = _valid(fb, expect, validate)
        if ok and (bak_mtime is None or _mtime(fb) >= bak_mtime):
            chosen_fallback = fb
            break
    return chosen_bak, chosen_fallback, walked


def _install(path: Path, chosen_bak, chosen_fallback, expect, validate, detail: str):
    """Put the chosen candidate at ``path`` with one atomic replace; returns (data, action, detail)."""
    try:
        if chosen_fallback is not None:
            raw, _sig = _read_bytes(chosen_fallback)
            data = _check(raw, expect, validate)
            save_with_backup_text(path, raw.decode("utf-8-sig"), keep_backups=False, newline="\n")
            chosen_fallback.unlink(missing_ok=True)
            return (
                data,
                "restored_from_bak1",
                f"{detail} | restored from fallback {chosen_fallback.name}"[:_DETAIL],
            )
        index, bak = chosen_bak
        raw, _sig = _read_bytes(bak)
        data = _check(raw, expect, validate)
        _replace_with_retry(bak, path)
        return data, f"restored_from_bak{index}", detail
    except (FileNotFoundError, _BadContentError):
        # the candidate was rotated away or rewritten (a save that took no heal sidecar): someone else acted
        raise _ChangedError("candidate vanished or changed") from None


def _quarantine_copy(path: Path, stamp: str) -> Path:
    """Keep the corrupt bytes as ``<name>.corrupt-<stamp>`` WITHOUT moving the live file.

    A hard link (a copy where links are unsupported): the live path is never absent during a heal, so a
    reader that arrives mid-heal sees either the corrupt file (and waits on the heal lock) or the restored
    one, never "missing", and a failed install leaves the live file where the next read will heal it again.
    """
    for attempt in range(100):
        q = path.with_name(f"{path.name}.corrupt-{stamp}" + (f"-{attempt}" if attempt else ""))
        try:
            os.link(path, q)
            return q
        except FileExistsError:
            continue
        except FileNotFoundError:
            raise _ChangedError(f"{path} vanished") from None
        except OSError:
            pass  # links unsupported or refused: fall back to a copy
        try:
            with open(path, "rb") as src, open(q, "xb") as dst:
                shutil.copyfileobj(src, dst)
            shutil.copystat(path, q)
            return q
        except FileExistsError:
            continue
        except FileNotFoundError:
            q.unlink(missing_ok=True)
            raise _ChangedError(f"{path} vanished") from None
        except OSError:
            q.unlink(missing_ok=True)  # never leave a half-written quarantine file behind
            raise
    raise OSError(errno.EEXIST, f"no free quarantine name for {path}")


def _heal_locked(path, bad, sig, default, recover, expect, validate, fallbacks) -> JsonRead:
    now = datetime.now(UTC)
    stamp = _filename_safe_timestamp(now)
    likely_cause = _classify_cause(path)
    anomaly_kind = (
        "json_parse_error" if bad.kind in ("encoding", "empty", "malformed") else "schema_mismatch"
    )
    detail = _short_detail(bad.cause)
    chosen_bak, chosen_fallback, walked_corrupt = _pick(path, expect, validate, fallbacks)

    if chosen_bak is None and chosen_fallback is None and recover is Recovery.RESTORE:
        raise JsonFileError(path, bad.kind, _short(bad.cause)) from bad.cause

    reset_data = reset_text = None
    if chosen_bak is None and chosen_fallback is None:
        # Build and strictly encode the default BEFORE anything is quarantined or written: a default holding a NaN
        # (or a factory that raises) leaves the directory exactly as it was.
        reset_data = default()
        reset_text = _dumps(reset_data, indent=2, sort_keys=False, encode_default=None, nonfinite="raise") + "\n"

    _expect_unchanged(path, sig)
    _seam("before_quarantine", path)
    _expect_unchanged(path, sig)
    try:
        quarantine = _quarantine_copy(path, stamp)
    except OSError as exc:
        raise JsonFileError(path, "io", _short(exc)) from exc
    _seam("inside_heal", path)
    # From here to the replace the live path still holds the corrupt file: a failure leaves it for the next heal.
    try:
        _seam("before_install", path)
        _expect_unchanged(path, sig)
        if chosen_bak is None and chosen_fallback is None:
            data = reset_data
            save_with_backup_text(path, reset_text, keep_backups=False, newline="\n")
            action, outcome, detail_out = "reset_to_default", "reset", detail
        else:
            data, action, detail_out = _install(
                path, chosen_bak, chosen_fallback, expect, validate, detail
            )
            outcome = "restored"
    except OSError as exc:
        raise JsonFileError(path, "io", _short(exc)) from exc
    # corrupt backups last, best effort, each only if it is unchanged
    for corrupt, csig in walked_corrupt:
        try:
            _seam("before_quarantine_backup", corrupt)
            _expect_unchanged(corrupt, csig)
            _replace_with_retry(corrupt, corrupt.with_name(f"{corrupt.name}.corrupt-{stamp}"))
        except _ChangedError:
            logger.warning("backup %s changed during the heal; left alone", corrupt)
        except OSError:
            logger.warning("could not quarantine corrupt backup %s", corrupt, exc_info=True)
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


def _restore_missing(path, default, recover, expect, validate, fallbacks) -> JsonRead | None:
    """RESTORE only: the file is absent; install a valid backup or fallback if there is one, else None."""
    candidates = [path.with_name(f"{path.name}.bak{i}") for i in (1, 2, 3)] + list(fallbacks)
    if not any(c.exists() for c in candidates):
        return None  # nothing to restore from; do not create a sidecar (or directories) for a first run
    with _heal_guard(path):
        result, retry = _restore_missing_locked(path, expect, validate, fallbacks)
    if retry:  # outside the lock: the normal read path may need it
        return _read(path, default, recover, expect, validate, fallbacks, waited=True)
    return result


def _restore_missing_locked(path, expect, validate, fallbacks) -> tuple[JsonRead | None, bool]:
    """With the heal sidecar held: (restored result or None, whether to read again). Never takes the lock."""
    if os.path.lexists(path):
        return None, True  # someone put it back while we waited for the lock
    chosen_bak, chosen_fallback, _walked = _pick(path, expect, validate, fallbacks)
    if chosen_bak is None and chosen_fallback is None:
        return None, False
    try:
        _expect_absent(path)
        data, action, detail = _install(
            path, chosen_bak, chosen_fallback, expect, validate, "live file missing"
        )
    except _ChangedError:
        return None, True
    except OSError as exc:
        raise JsonFileError(path, "io", _short(exc)) from exc
    anomaly = BrainAnomaly(
        timestamp=datetime.now(UTC),
        file=path.name,
        kind="json_parse_error",
        action=action,  # type: ignore[arg-type]
        quarantine_path=None,
        likely_cause="unknown",
        detail=detail,
    )
    logger.warning("%s was missing; %s", path, action)
    return JsonRead(data, "restored", "missing", anomaly), False


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
    nonfinite: Nonfinite = "raise",
) -> bool:
    """Atomically write ``data`` as JSON. True if written, False if skipped or (``on_error="log"``) failed.

    Serialization happens first and strictly (no NaN, no silent ``str()``), so a payload that cannot be
    encoded touches nothing. ``backups=N`` keeps ``.bak1..N`` (0 keeps none); a file with a backup chain
    is saved under the heal sidecar so a save cannot land inside a heal. ``mode`` is applied when the new
    file is created and to every backup. ``verify=True`` re-reads the result and restores ``.bak1`` if it
    is not what was written (never raises). ``if_missing=True`` creates the file only if it does not exist
    (False if it does). ``on_error="log"`` turns an ``OSError`` into a throttled warning and False.
    ``nonfinite="null"`` writes NaN/Infinity as null instead of raising (for a best-effort writer whose
    payload can carry a computed float); the default raises ValueError before anything is touched.
    """
    text = (
        _dumps(
            data, indent=indent, sort_keys=sort_keys, encode_default=encode_default, nonfinite=nonfinite
        )
        + "\n"
    )
    return _save_text(_as_path(path), text, backups, fsync, mode, verify, if_missing, on_error)


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
            if not verify:
                return True
            # Verify under the same lock as the save: no module writer can land between them.
            try:
                after = _signature(path.stat())
            except OSError:
                after = None
            _seam("after_save", path)
            return _verify(path, text, backups > 0, after)
    except OSError as exc:
        if on_error == "raise":
            raise
        _report(path, "write", exc, "write skipped")
        return False


def _verify(path: Path, text: str, has_backups: bool, after_save) -> bool:
    """Did our write land intact? Restore ``.bak1`` only if the file is damaged in place or unreadable.

    A different file that replaced ours after the save (a new signature: another saver that takes no heal
    sidecar) is a newer save, not damage: it is left alone and the write counts as done.
    """
    expected = text.encode("utf-8")
    try:
        if path.read_bytes() == expected:
            return True
        try:
            now = _signature(path.stat())
        except OSError:
            now = None
        if after_save is not None and now is not None and now != after_save:
            logger.warning("%s was replaced after it was saved; leaving the newer file alone", path)
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


# --------------------------------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------------------------------

_JSONL_IO_ERRORS = (OSError, EOFError, gzip.BadGzipFile, zlib.error)


def encode_jsonl(
    records: dict | Iterable[dict],
    *,
    encode_default: Callable[[Any], Any] | None = None,
    nonfinite: Nonfinite = "raise",
) -> bytes:
    """The one JSONL encoder: one ASCII line per dict, ``\\n`` terminated. Raises before anything is written.

    A record that is not a dict is a TypeError, a NaN a ValueError, an unencodable value a TypeError
    (unless ``encode_default`` handles it). ``nonfinite="null"`` writes NaN/Infinity as null instead.
    """
    rows = [records] if isinstance(records, dict) else list(records)
    out = []
    for rec in rows:
        if not isinstance(rec, dict):
            raise TypeError(f"a JSONL record must be a dict, got {type(rec).__name__}")
        out.append(
            _dumps(rec, indent=None, sort_keys=False, encode_default=encode_default, nonfinite=nonfinite)
            + "\n"
        )
    return "".join(out).encode("ascii" if _ENSURE_ASCII else "utf-8")


def _check_jsonl_recover(recover: Recovery) -> None:
    if recover not in (Recovery.DEFAULT, Recovery.RAISE):
        raise ValueError(f"JSONL reads take Recovery.DEFAULT or Recovery.RAISE, not {recover!r}")


def parse_jsonl(
    data: bytes | str, *, source: Path | str = "<memory>", strict: bool = False
) -> Iterator[dict]:
    """Parse JSONL text or bytes already read: the one place bad lines are skipped (or, strict, raised)."""
    raw = data.encode("utf-8", "surrogateescape") if isinstance(data, str) else data
    src = _as_path(source)
    try:
        yield from _stream(io.BytesIO(raw), src, strict)
    except _StrictLineError as exc:
        raise JsonFileError(exc.path, "line", exc.reason, exc.line) from exc


def _stream(binary, source: Path, strict: bool) -> Iterator[dict]:
    wrapper = io.TextIOWrapper(binary, encoding="utf-8-sig", errors="surrogateescape", newline="\n")
    try:
        yield from _parse_lines(wrapper, source, strict=strict)
    except JsonlLineError as exc:
        raise _StrictLineError(source, exc.reason, exc.line) from exc
    finally:
        wrapper.detach()


class _StrictLineError(Exception):
    """Internal carrier so a strict-line failure is not mistaken for an I/O failure by the readers' handlers."""

    def __init__(self, path: Path, reason: str, line: int) -> None:
        super().__init__(reason)
        self.path, self.reason, self.line = path, reason, line


def iter_jsonl(
    path: Path, *, strict: bool = False, recover: Recovery = Recovery.DEFAULT
) -> Iterator[dict]:
    """Yield the dict records of a JSONL file (``.gz`` by suffix), skipping bad lines with a warning.

    A missing file yields nothing. ``strict=True`` raises ``JsonFileError(kind="line")`` on the first bad
    line (after yielding the good ones before it). An I/O failure (including a truncated gz) ends the
    iteration with a throttled warning (DEFAULT) or raises ``JsonFileError(kind="io")`` (RAISE).
    """
    path = _as_path(path)
    _check_jsonl_recover(recover)
    try:
        opener = gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")
    except FileNotFoundError:
        return
    except _JSONL_IO_ERRORS as exc:
        yield from _io_stop(path, exc, recover)
        return
    try:
        with opener as fh:
            yield from _stream(fh, path, strict)
    except _StrictLineError as exc:
        raise JsonFileError(exc.path, "line", exc.reason, exc.line) from exc
    except _JSONL_IO_ERRORS as exc:
        yield from _io_stop(path, exc, recover)


def _io_stop(path: Path, exc: BaseException, recover: Recovery) -> Iterator[dict]:
    if recover is Recovery.RAISE:
        raise JsonFileError(path, "io", _short(exc)) from exc
    _report(path, "io", exc, "stopped reading; returning what was read")
    return
    yield  # pragma: no cover - makes this a generator


def read_jsonl(
    path: Path, *, strict: bool = False, recover: Recovery = Recovery.DEFAULT
) -> list[dict]:
    return list(iter_jsonl(path, strict=strict, recover=recover))


def iter_jsonl_raw(
    path: Path,
    *,
    strict: bool = False,
    recover: Recovery = Recovery.DEFAULT,
    bad_lines: bool = False,
) -> Iterator[tuple[str, dict | None]]:
    """Like :func:`iter_jsonl` but yields ``(line_text, record)``: the line without its terminator, exactly as read.

    For a caller that re-writes the lines it read and must not re-serialise them (log rotation) or that must
    still see a line it cannot parse (``recall_diagnostics._prune`` removes it). Blank lines are never yielded.
    With ``bad_lines=True`` a line that cannot be used (malformed, invalid UTF-8, not an object) is yielded as
    ``(text, None)`` after its usual warning; otherwise it is only skipped. ``strict`` and ``recover`` behave
    as in :func:`iter_jsonl`.
    """
    path = _as_path(path)
    _check_jsonl_recover(recover)
    try:
        opener = gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")
    except FileNotFoundError:
        return
    except _JSONL_IO_ERRORS as exc:
        yield from _io_stop(path, exc, recover)  # type: ignore[misc]
        return
    try:
        with opener as fh:
            yield from _stream_raw(fh, path, strict, bad_lines)
    except _StrictLineError as exc:
        raise JsonFileError(exc.path, "line", exc.reason, exc.line) from exc
    except _JSONL_IO_ERRORS as exc:
        yield from _io_stop(path, exc, recover)  # type: ignore[misc]


def _stream_raw(binary, source: Path, strict: bool, bad_lines: bool) -> Iterator[tuple[str, dict | None]]:
    wrapper = io.TextIOWrapper(binary, encoding="utf-8-sig", errors="surrogateescape", newline="\n")
    try:
        for line_index, raw in enumerate(wrapper, start=1):
            text = raw.rstrip("\r\n")
            try:
                raw.encode("utf-8")
            except UnicodeEncodeError as exc:
                reason, shown = f"invalid UTF-8 ({exc.reason})", text.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
                record = None
            else:
                if not text.strip():
                    continue
                try:
                    data = json.loads(text)
                except _JSON_PARSE_ERRORS as exc:
                    reason, shown, record = f"malformed JSON ({exc})", text, None
                else:
                    if isinstance(data, dict):
                        yield text, data
                        continue
                    reason, shown, record = f"not a JSON object ({type(data).__name__})", text, None
            if strict:
                raise _StrictLineError(source, reason, line_index)
            logger.warning(
                "skipping malformed jsonl line %d in %s: %s | content: %r",
                line_index, source, reason, shown[:_PREVIEW],
            )
            if bad_lines:
                yield text, record
    finally:
        if not binary.closed:  # an abandoned iterator can be closed after the caller's `with` closed the file
            wrapper.detach()


def read_jsonl_tail(
    path: Path, n: int, *, strict: bool = False, recover: Recovery = Recovery.DEFAULT
) -> list[dict]:
    """The records on the last ``n`` non-blank lines, read backwards in byte chunks (bounded I/O).

    Blank lines do not count toward ``n``. The window is cut on a newline *byte* before any decoding, so a
    multi-byte character split by a chunk boundary costs only the dropped partial first line, and a bad
    byte is skipped with a warning (or, ``strict=True``, raises) rather than turned into U+FFFD. The scan is
    linear in the bytes read, even for one very long line.

    A ``.gz`` path cannot be read backwards: it is streamed through :func:`iter_jsonl` and the last ``n``
    *records* (after bad lines are skipped) are returned.
    """
    path = _as_path(path)
    _check_jsonl_recover(recover)
    if n <= 0:
        return []
    if path.suffix == ".gz":
        return list(deque(iter_jsonl(path, strict=strict, recover=recover), maxlen=n))
    try:
        lines = _tail_lines(path, n)
    except FileNotFoundError:
        return []
    except _JSONL_IO_ERRORS as exc:
        if recover is Recovery.RAISE:
            raise JsonFileError(path, "io", _short(exc)) from exc
        _report(path, "io", exc, "returning no records")
        return []
    try:
        return list(_stream(io.BytesIO(b"\n".join(lines)), path, strict))
    except _StrictLineError as exc:
        raise JsonFileError(exc.path, "line", exc.reason, exc.line) from exc


def _tail_lines(path: Path, n: int) -> list[bytes]:
    """The last ``n`` non-blank lines as bytes (CR stripped), in file order. Each byte is looked at once."""
    with open(path, "rb") as f:
        remaining = f.seek(0, os.SEEK_END)
        if remaining == 0:
            return []
        head: deque[bytes] = (
            deque()
        )  # pieces of the partial line at the left edge of what has been read
        found: list[bytes] = []  # complete non-blank lines, rightmost first
        while remaining > 0 and len(found) < n:
            size = min(_TAIL_CHUNK, remaining)
            remaining -= size
            f.seek(remaining)
            chunk = f.read(size)
            if b"\n" not in chunk:
                head.appendleft(chunk)  # the partial line extends further left; do not re-scan it
                continue
            parts = chunk.split(b"\n")
            tail_line = parts[-1] + b"".join(head)  # the newline before it makes it complete
            head = deque([parts[0]])
            for line in [tail_line, *parts[-2:0:-1]]:  # rightmost complete line first
                if line.strip():
                    found.append(line.rstrip(b"\r"))
                    if len(found) >= n:
                        break
        if remaining == 0 and len(found) < n:  # the head is the first line of the file
            first = b"".join(head)
            if first.startswith(_BOM):
                first = first[len(_BOM) :]
            if first.strip():
                found.append(first.rstrip(b"\r"))
    return found[::-1]


def _payload(records, nonfinite: Nonfinite = "raise") -> bytes:
    """Encoded records: pass ``encode_jsonl`` output through (it must end a line), encode anything else."""
    if isinstance(records, (bytes, bytearray)):
        payload = bytes(records)
        if payload and not payload.endswith(b"\n"):
            raise ValueError(
                "encoded JSONL bytes must end with a newline (a half line would fuse with the next record)"
            )
        return payload
    return encode_jsonl(records, nonfinite=nonfinite)


def _tail_is_torn(fh) -> bool:
    fh.seek(0, os.SEEK_END)
    size = fh.tell()
    if size == 0:
        return False
    fh.seek(size - 1)
    return fh.read(1) != b"\n"


def append_jsonl(
    path: Path,
    records: dict | Iterable[dict] | bytes,
    *,
    lock: Literal["block", "try", "held"] = "block",
    fsync: bool = False,
    on_error: Literal["raise", "log"] = "raise",
    nonfinite: Nonfinite = "raise",
) -> int | None:
    """Append dict records as JSONL in ONE write; return the record bytes written, or None if nothing was.

    ``records`` is a dict, an iterable of dicts, or the bytes ``encode_jsonl`` returned. Everything is encoded
    before the file is touched. Under the lock, if the file is non-empty and does not end in a newline (a torn
    tail, #434) the same write begins with a newline, so only the torn record is lost. ``lock``: "block" waits
    for ``file_lock(path)``; "try" returns None if it is busy; "held" takes no lock (the caller holds it). Under
    "block" a filesystem that refuses the lock degrades to unlocked with a warning. Under "try" it cannot:
    ``file_lock(blocking=False)`` reports every flock error as busy, so a refusing filesystem reads as busy
    and the record is not written (the caller, like the recall path, buffers on None). ``on_error="log"`` turns
    an OSError into a throttled warning and None. ``nonfinite`` applies only when ``records`` are dicts;
    pre-encoded bytes are written as given (encode them with ``encode_jsonl(..., nonfinite=...)``).
    """
    path = _as_path(path)
    if lock not in ("block", "try", "held"):
        raise ValueError(f"lock must be 'block', 'try' or 'held', not {lock!r}")
    payload = _payload(records, nonfinite)
    if not payload:
        return 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stack = ExitStack()
        acquired = True
        if lock != "held":
            try:
                acquired = stack.enter_context(file_lock(path, blocking=(lock == "block")))
            except OSError as exc:
                warn_throttled(
                    logger,
                    f"json_io:{path}:lock",
                    "could not take the lock for %s (%s); continuing without it",
                    path,
                    _short(exc),
                )
                acquired = True
        with stack:
            if not acquired:
                return None
            with open(path, "ab+") as fh:
                prefix = b""
                if _tail_is_torn(fh):
                    prefix = b"\n"
                    logger.warning(
                        "%s ended in a torn record (no newline); starting a new line before appending",
                        path,
                    )
                _seam("between_check_and_write", path)
                fh.write(prefix + payload)
                fh.flush()
                if fsync:
                    os.fsync(fh.fileno())
    except OSError as exc:
        if on_error == "raise":
            raise
        _report(path, "append", exc, "record dropped")
        return None
    return len(payload)


def write_jsonl(
    path: Path,
    records: dict | Iterable[dict] | bytes,
    *,
    backups: int = 0,
    fsync: bool = True,
    mode: int | None = None,
    on_error: Literal["raise", "log"] = "raise",
    nonfinite: Nonfinite = "raise",
) -> bool:
    """Atomically replace a JSONL file with ``records`` (same writer as :func:`write_json`)."""
    payload = _payload(records, nonfinite)
    return _save_text(
        _as_path(path), payload.decode("utf-8"), backups, fsync, mode, False, False, on_error
    )
