"""Tolerant construction of persisted-state dataclasses (#286 §7).

Rollback invariant: the brain's code can roll back (a user reverts a brain
updated from main to the release brain) but persona data does not. A record
written by a newer brain may carry fields this brain does not know. Every
reader of persisted state builds its dataclass through `from_known_fields`,
which keeps the fields the class declares and ignores the rest, so a rollback
never crashes a reader or silently drops a record.

The matching rule for writers: persisted-state changes are additive only —
add fields; never rename or remove one, or change what an existing one means,
without a migration spec. tests/unit/test_no_strict_state_constructors.py
fails if a `Cls(**record)` construction comes back.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


def from_known_fields[T](cls: type[T], data: Mapping[str, Any]) -> T:
    """Build dataclass `cls` from `data`, ignoring keys `cls` doesn't declare.

    A missing required field still raises TypeError, as `cls(**data)` would.
    """
    names = {f.name for f in dataclasses.fields(cls)}  # TypeError if not a dataclass
    return cls(**{k: v for k, v in data.items() if k in names})
