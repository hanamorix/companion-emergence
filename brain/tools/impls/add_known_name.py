"""add-name tool implementation: the kindled adds a name to her known-names list.

The third way a name reaches the list (spec §5, "Names are added three ways",
item 3; the other two are the temp gate's judge and the on-recall re-appraiser).
Dispatched in the MCP subprocess like ``add_memory`` (not bridge-routed): it only
writes the small known-names file under the OS file lock.

All admission goes through ``brain.memory.known_names.admit_names`` (S70): an
entry that is a recall stopword ("will", "the") is rejected with nothing written;
every other entry is admitted as given, its display form kept intact (S87).

The tool's identifier and its persona-facing text are PLACEHOLDERS owned by the
owner (I10); they live in ``brain/tools/schemas.py``. This module's own name is
not persona-facing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from brain.memory.known_names import admit_names

logger = logging.getLogger(__name__)


def add_known_name(name: Any, *, persona_dir: Path, **_unused: Any) -> dict:
    """Add ``name`` to the persona's known-names list (source ``tool``).

    Returns structured fields only, no prose:
      - ``added``: True when the name is on the list after the call (a name
        already listed counts, the list is a set); False when the admission filter
        rejected it (stopword entry, no word in it, not a string) or the write
        could not be made (logged, never raised).
      - ``name``: the name as given, whitespace-trimmed (display form intact); an
        empty string when the argument was not a string.

    Never raises for a bad argument or an I/O failure: a tool result is data, and
    the model deciding what to do about ``added: false`` is its own business.
    """
    given = name.strip() if isinstance(name, str) else ""
    if not given:
        return {"added": False, "name": given}
    admitted = admit_names(persona_dir, [given], "tool")
    return {"added": bool(admitted), "name": given}
