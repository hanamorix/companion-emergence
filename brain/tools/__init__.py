"""brain.tools — brain-query tools for the chat engine (SP-3).

Public surface:
    SCHEMAS           — JSON schemas keyed by tool name
    LOVE_TYPES        — enum of crystallization love-types
    NELL_TOOL_NAMES   — canonical ordered tuple of all 9 tools
    dispatch          — dispatch(name, arguments, *, store, hebbian, persona_dir) -> dict
    ToolDispatchError — raised on dispatch failures

OG reference: NellBrain/nell_tools.py (841 lines, 9 impls + SCHEMAS).
Master ref §6 SP-3.
"""

from brain.tools.dispatch import ToolDispatchError, dispatch
from brain.tools.schemas import LOVE_TYPES, SCHEMAS

# Canonical tool list, in the order the LLM should see them.
# Ported verbatim from OG NELL_TOOLS (nell_bridge.py:172-185).
NELL_TOOL_NAMES: tuple[str, ...] = (
    "get_emotional_state",
    "get_soul",
    "get_personality",
    "get_body_state",
    "boot",
    "search_memories",
    "add_journal",
    "add_memory",
    "crystallize_soul",
    "save_work",
    "list_works",
    "search_works",
    "read_work",
)

__all__ = [
    "SCHEMAS",
    "LOVE_TYPES",
    "NELL_TOOL_NAMES",
    "dispatch",
    "ToolDispatchError",
]
