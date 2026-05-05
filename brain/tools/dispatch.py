"""Tool dispatch — name → callable, with argument validation.

dispatch(name, arguments, *, store, hebbian, persona_dir) is the single
entry-point for the chat engine's tool loop. It validates required fields
against the schema, type-checks critical args, then delegates to the impl.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools.impls.add_journal import add_journal
from brain.tools.impls.add_memory import add_memory
from brain.tools.impls.boot import boot
from brain.tools.impls.crystallize_soul import crystallize_soul
from brain.tools.impls.get_body_state import get_body_state
from brain.tools.impls.get_emotional_state import get_emotional_state
from brain.tools.impls.get_personality import get_personality
from brain.tools.impls.get_soul import get_soul
from brain.tools.impls.list_works import list_works
from brain.tools.impls.read_work import read_work
from brain.tools.impls.save_work import save_work
from brain.tools.impls.search_memories import search_memories
from brain.tools.impls.search_works import search_works
from brain.tools.schemas import SCHEMAS


class ToolDispatchError(RuntimeError):
    """Raised on unknown tool name, missing required args, or type mismatch.

    Subclasses RuntimeError so callers can catch it without importing this
    module just for the exception class — but ToolDispatchError is the
    canonical name to use in tools/dispatch-facing code.
    """


# ---------------------------------------------------------------------------
# Dispatch table — name → impl callable
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, Any] = {
    "get_emotional_state": get_emotional_state,
    "get_personality": get_personality,
    "get_body_state": get_body_state,
    "search_memories": search_memories,
    "add_journal": add_journal,
    "add_memory": add_memory,
    "boot": boot,
    "get_soul": get_soul,
    "crystallize_soul": crystallize_soul,
    "save_work": save_work,
    "list_works": list_works,
    "search_works": search_works,
    "read_work": read_work,
}


_WORKS_TOOLS = frozenset({"save_work", "list_works", "search_works", "read_work"})


def dispatch(
    name: str,
    arguments: dict[str, Any],
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Dispatch a tool call by name with arguments.

    Parameters
    ----------
    name:
        Tool name — must be a key in the dispatch table (see ``_DISPATCH``).
    arguments:
        Parsed JSON args dict from the LLM tool_call.  Will be validated
        against the schema's "required" list before the impl is called.
    store, hebbian, persona_dir:
        Injected by the chat engine — not passed by the LLM.

    Raises
    ------
    ToolDispatchError
        - Unknown tool name
        - Missing required argument
        - Type mismatch (e.g. emotions not a dict)

    Returns
    -------
    dict — the impl's return value, JSON-serialisable.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        # M-10: don't enumerate known tools in the error returned to the LLM.
        # The tool surface is implementation detail; the unknown-tool message
        # ends up in invocation logs and (via tool_loop) the next LLM context
        # window. Dispatch errors are for operators, not training signal.
        raise ToolDispatchError(f"unknown tool: {name!r}")

    schema = SCHEMAS.get(name, {})
    params = schema.get("parameters", {})
    required = params.get("required", [])

    # Validate required args present
    for field in required:
        if field not in arguments:
            raise ToolDispatchError(f"tool {name!r} missing required argument: {field!r}")

    # Type-check critical args that dispatch would silently mishandle
    if name == "add_memory" and "emotions" in arguments:
        if not isinstance(arguments["emotions"], dict):
            raise ToolDispatchError(
                f"tool 'add_memory' arg 'emotions' must be a dict, "
                f"got {type(arguments['emotions']).__name__!r}"
            )

    if name == "get_body_state" and "session_hours" in arguments:
        try:
            session_hours_float = float(arguments["session_hours"])
        except (TypeError, ValueError) as exc:
            raise ToolDispatchError(
                f"tool 'get_body_state' arg 'session_hours' must be a number, "
                f"got {type(arguments['session_hours']).__name__!r}"
            ) from exc
        # M-4: don't mutate the caller's dict — shallow-copy then update.
        # The chat engine logs `arguments` straight into the invocations
        # record; mutating it bleeds float coercion into the audit trail.
        arguments = {**arguments, "session_hours": session_hours_float}

    if name in _WORKS_TOOLS:
        try:
            return fn(**arguments, persona_dir=persona_dir)
        except TypeError as exc:
            raise ToolDispatchError(f"bad arguments to tool {name!r}: {exc}") from exc

    injected = {"store": store, "hebbian": hebbian, "persona_dir": persona_dir}

    try:
        return fn(**arguments, **injected)
    except TypeError as exc:
        raise ToolDispatchError(f"bad arguments to tool {name!r}: {exc}") from exc
