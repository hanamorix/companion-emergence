"""What a failure depends on decides what happens to the item it hit (#459, #460).

A batch job that fails on one item has to choose between keeping the item for a later attempt and setting it aside, and the
choice is not about severity but about cause:

- ``"environment"``: the failure depends on the store or the filesystem, not on the item. It would hit any item and clears
  when the environment is fixed, so the item is kept and tried again later.
- ``"item"``: the failure depends on the item's own data and will fail the same way again, so retrying costs work and gets
  nowhere; the item is set aside with the error as evidence.

The mapping is by exception TYPE, never by message text. Each branch states why it sits where it does:

- ``ThrottleDeferred`` (an expired login, a shutdown refusal): a refusal that says "not now" (#473). It is a ``RuntimeError``,
  which would otherwise read as an item fault, so it is checked first.
- ``OSError`` (``PermissionError``, ``FileNotFoundError``, disk full, I/O error, timeout, and ``JsonFileError`` of every kind):
  raised by the filesystem the job reads and writes, not by the item; it recurs for any item and clears when the environment
  does.
- ``sqlite3.IntegrityError``, ``DataError``, ``ProgrammingError``, ``InterfaceError``: a constraint, a size or a binding: they
  depend on the row.
- any other ``sqlite3.Error`` (``OperationalError``: locked, busy, full, I/O, "no such table"; ``DatabaseError``: corrupt or not a
  database; ``InternalError``; ``NotSupportedError``): store-wide, the same for every item.
- everything else, including ``TypeError``, ``ValueError``, ``KeyError``, ``MemoryError`` and any unknown type: deterministic in
  the item (or a bug in the job); setting the item aside keeps its data with evidence instead of retrying it forever.

``sqlite3.ProgrammingError`` is also what a CLOSED store and a store used from the wrong thread raise, for every item. So before
anyone acts on an item verdict, :func:`settle_scope` confirms the store itself is usable with one cheap read.
"""

from __future__ import annotations

import sqlite3
from typing import Literal

Scope = Literal["environment", "item"]

_ITEM_SQLITE = (sqlite3.IntegrityError, sqlite3.DataError, sqlite3.ProgrammingError, sqlite3.InterfaceError)


def failure_scope(exc: BaseException) -> Scope:
    """Classify ``exc`` by type: ``"environment"`` (keep the item, retry later) or ``"item"`` (set it aside)."""
    # Imported here: brain.utils must not depend on brain.bridge at import time.
    from brain.bridge.cli_throttle import ThrottleDeferred

    if isinstance(exc, ThrottleDeferred):
        return "environment"
    if isinstance(exc, OSError):
        return "environment"
    if isinstance(exc, _ITEM_SQLITE):
        return "item"
    if isinstance(exc, sqlite3.Error):
        return "environment"
    return "item"


def settle_scope(exc: BaseException, store) -> Scope:
    """``failure_scope`` plus a store-health confirmation of an item verdict.

    A per-row failure leaves the store usable. If one cheap read of the store (``store.get("", bump=False)``, which returns
    None on a healthy store) itself raises, the store is closed, on the wrong thread or broken, the failure would hit every
    item, and it is reclassified as ``"environment"``.
    """
    scope = failure_scope(exc)
    if scope == "item":
        try:
            store.get("", bump=False)
        except Exception:  # noqa: BLE001 — any failure of the probe means the store is unusable, which is the answer
            return "environment"
    return scope
