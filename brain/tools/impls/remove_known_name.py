"""remove_name tool implementation: the kindled removes a name from her list.

Names increment (S105, S107, S135): she can remove any listed name, whoever added
it. The name goes on the rejects list, so the gate judge and the re-appraiser never
add it again; she can add it back with ``add_name`` through the confirm step
(S137). Removing a name does not change any memories. Dispatched in the MCP
subprocess, like ``add_name``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from brain.memory.known_names import remove_names


def remove_known_name(name: Any, *, persona_dir: Path, **_unused: Any) -> dict:
    """Remove ``name`` (compared after normalizing, so "Tuesday" and "tuesday" are
    one name) from the persona's known-names list.

    Returns structured fields: ``removed`` (True when a listed name was removed;
    False when it was not on the list, the argument was not a name, or the write
    could not be made, logged) and ``name`` (as given, trimmed). Never raises.
    """
    given = name.strip() if isinstance(name, str) else ""
    if not given:
        return {"removed": False, "name": given}
    removed = remove_names(persona_dir, [given], "tool")
    return {"removed": bool(removed), "name": given}
