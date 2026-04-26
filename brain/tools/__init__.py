"""brain.tools — brain-query tools for the chat engine (SP-3).

Public surface:
    schemas     — SCHEMAS dict + LOVE_TYPES constant
    dispatch    — dispatch(name, arguments, *, store, hebbian, persona_dir) -> dict
    ToolDispatchError — raised on dispatch failures

OG reference: NellBrain/nell_tools.py (841 lines, 9 impls + SCHEMAS).
Master ref §6 SP-3.
"""

from brain.tools.dispatch import ToolDispatchError, dispatch
from brain.tools.schemas import LOVE_TYPES, SCHEMAS

__all__ = [
    "SCHEMAS",
    "LOVE_TYPES",
    "dispatch",
    "ToolDispatchError",
]
