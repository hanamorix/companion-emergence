"""Tool dispatch — name → callable, with argument validation.

dispatch(name, arguments, *, store, hebbian, persona_dir) is the single
entry-point for the chat engine's tool loop. It validates required fields
against the schema, type-checks critical args, then delegates to the impl.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from brain.felt_time.tool import felt_time_now as _felt_time_now_impl
from brain.felt_time.tool import pressure_since as _pressure_since_impl
from brain.forgetting.tool import recall_forgotten as _recall_forgotten_impl
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.monologue.recall import recall_monologue
from brain.narrative_memory.tool import list_open_arcs as _list_open_arcs_impl
from brain.narrative_memory.tool import recall_arc as _recall_arc_impl
from brain.tools.impls.add_journal import add_journal
from brain.tools.impls.add_memory import add_memory
from brain.tools.impls.boot import boot
from brain.tools.impls.compact_history import compact_history
from brain.tools.impls.crystallize_soul import crystallize_soul
from brain.tools.impls.get_body_state import get_body_state
from brain.tools.impls.get_emotional_state import get_emotional_state
from brain.tools.impls.get_personality import get_personality
from brain.tools.impls.get_soul import get_soul
from brain.tools.impls.list_directory import list_directory
from brain.tools.impls.list_works import list_works
from brain.tools.impls.reach_for_capability import reach_for_capability
from brain.tools.impls.read_file import read_file
from brain.tools.impls.read_work import read_work
from brain.tools.impls.save_work import save_work
from brain.tools.impls.search_memories import search_memories
from brain.tools.impls.search_works import search_works
from brain.tools.schemas import SCHEMAS

log = logging.getLogger(__name__)


class ToolDispatchError(RuntimeError):
    """Raised on unknown tool name, missing required args, or type mismatch.

    Subclasses RuntimeError so callers can catch it without importing this
    module just for the exception class — but ToolDispatchError is the
    canonical name to use in tools/dispatch-facing code.
    """


# ---------------------------------------------------------------------------
# Dispatch table — name → impl callable
# ---------------------------------------------------------------------------


def _felt_time_now_wrapper(*, store, hebbian, persona_dir, **_):
    return _felt_time_now_impl(persona_dir=persona_dir)


def _pressure_since_wrapper(*, store, hebbian, persona_dir, anchor_type=None, **_):
    return _pressure_since_impl(arguments={"anchor_type": anchor_type}, persona_dir=persona_dir)


def _recall_forgotten_wrapper(*, store, hebbian, persona_dir, query=None, **_):
    result = _recall_forgotten_impl(
        arguments={"query": query} if query is not None else {},
        persona_dir=persona_dir,
    )
    # Fault-isolated grief write — wired per spec §6 / Phase 9.1.
    # handle_recall_touch is internally fault-isolated; this outer try guards
    # only against import-time failures on brain.grief / brain.felt_time.
    try:
        hits = result.get("hits") or []
        hit_ids = [h["memory_id"] for h in hits if "memory_id" in h]
        if hit_ids:
            from brain.felt_time.state import load_or_recover as _load_felt_time
            from brain.grief import handle_recall_touch

            felt_state, _ = _load_felt_time(persona_dir)
            handle_recall_touch(
                touched_ids=hit_ids,
                graveyard_entries=hits,
                persona_dir=persona_dir,
                store=store,
                lived_age_hours_now=felt_state.lived_age_hours,
            )
    except Exception:  # noqa: BLE001
        log.exception("grief.handle_recall_touch failed inside recall_forgotten dispatch")
    return result


def _list_open_arcs_wrapper(*, store, hebbian, persona_dir, **_):
    return _list_open_arcs_impl(persona_dir=persona_dir)


def _recall_arc_wrapper(*, store, hebbian, persona_dir, query="", **_):
    if not isinstance(query, str):
        raise ToolDispatchError("recall_arc: 'query' must be a string")
    return _recall_arc_impl(query=query, persona_dir=persona_dir)


def _reconcile_self_read_wrapper(
    *, store, hebbian, persona_dir, action, channel=None, delta=None, name=None, **_
):
    from brain.self_model.reconcile import reconcile_self_read as _reconcile_impl

    return _reconcile_impl(
        persona_dir=persona_dir,
        action=action,
        channel=channel,
        delta=delta,
        name=name,
    )


def _surface_makings_wrapper(*, persona_dir, **_):
    from brain.maker.disclosure import surface_makings

    return surface_makings(persona_dir=persona_dir)


def _propose_write_wrapper(*, persona_dir, path, op, content=None, making_id=None, **_):
    from brain.tools.impls.propose_write import propose_write

    return propose_write(
        path=path, content=content, op=op, making_id=making_id, persona_dir=persona_dir
    )


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
    "felt_time_now": _felt_time_now_wrapper,
    "pressure_since": _pressure_since_wrapper,
    "recall_forgotten": _recall_forgotten_wrapper,
    "list_open_arcs": _list_open_arcs_wrapper,
    "recall_arc": _recall_arc_wrapper,
    "recall_monologue": recall_monologue,
    "record_monologue": None,  # replaced below after function definition
    "read_file": read_file,
    "list_directory": list_directory,
    "reach_for_capability": reach_for_capability,
    "reconcile_self_read": _reconcile_self_read_wrapper,
    "propose_write": _propose_write_wrapper,
    "surface_makings": _surface_makings_wrapper,
    "compact_history": compact_history,
}


def _dispatch_record_monologue(
    *,
    monologue: str = "",
    feed_digest: str = "",
    surface: bool = True,
    store: MemoryStore,
    persona_dir: Path,
    **_unused: Any,
) -> dict[str, Any]:
    """Real handler for record_monologue (MCP and direct-dispatch paths).

    Calls capture_monologue() synchronously: persists the Tier-2 trace memory
    + the gated Tier-3 digest. On success returns a dict carrying the monologue
    text — downstream code (tool_loop._find_monologue_text and the MCP audit
    pass-through in provider._read_audit_lines_since) reads ``monologue_text``
    to decide whether to spawn pass 2.

    On CaptureRejected, returns an error dict without raising — the tool
    result is surfaced as a normal (soft) error rather than a crash.
    """
    from brain.chat.monologue_capture import CaptureRejected, capture_monologue

    try:
        captured = capture_monologue(
            persona_dir=persona_dir,
            store=store,
            monologue=monologue,
            feed_digest=feed_digest,
            surface=surface,
        )
        return {"ok": True, "monologue_text": captured}
    except CaptureRejected as exc:
        return {"error": str(exc)}


_DISPATCH["record_monologue"] = _dispatch_record_monologue


_WORKS_TOOLS = frozenset({"save_work", "list_works", "search_works", "read_work"})

# Tools that need a live LLM provider injected (compaction summarises via the
# provider). Routed through a dedicated path so the global injected dict stays
# clean — existing tools never receive an unexpected ``provider`` or
# ``session_id`` kwarg.
_PROVIDER_TOOLS = frozenset({"compact_history"})


def dispatch(
    name: str,
    arguments: dict[str, Any],
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
    provider: Any = None,
    session_id: str | None = None,
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
    provider:
        Live LLM provider — required only for tools in ``_PROVIDER_TOOLS``
        (currently ``compact_history``). Optional for all other tools.
    session_id:
        Active session identifier — required only for ``_PROVIDER_TOOLS``.

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

    if name == "get_body_state":
        if "session_hours" in arguments:
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
        else:
            # Caller didn't provide session_hours — the LLM never knows
            # the session age and the MCP tool schema doesn't ask. Inject
            # the same live value the UI's /persona/state body block
            # already uses (active-conversation-buffer age), so the
            # brain's self-read matches what the panel shows. Bug
            # surfaced 2026-05-17: without this, get_body_state always
            # returned session_hours=0.0 + fresh-persona defaults
            # (energy 7, exhaustion 0) regardless of session age.
            from datetime import UTC, datetime

            from brain.body.session_hours import compute_active_session_hours

            arguments = {
                **arguments,
                "session_hours": compute_active_session_hours(persona_dir, now=datetime.now(UTC)),
            }

    if name in _WORKS_TOOLS:
        try:
            return fn(**arguments, persona_dir=persona_dir)
        except TypeError as exc:
            raise ToolDispatchError(f"bad arguments to tool {name!r}: {exc}") from exc

    if name in _PROVIDER_TOOLS:
        if provider is None:
            raise ToolDispatchError(f"tool {name!r} requires a provider but none was injected")
        if not session_id:
            raise ToolDispatchError(f"tool {name!r} requires a session_id but none was injected")
        try:
            return fn(
                **arguments,
                persona_dir=persona_dir,
                provider=provider,
                session_id=session_id,
            )
        except TypeError as exc:
            raise ToolDispatchError(f"bad arguments to tool {name!r}: {exc}") from exc

    injected = {"store": store, "hebbian": hebbian, "persona_dir": persona_dir}

    try:
        return fn(**arguments, **injected)
    except TypeError as exc:
        raise ToolDispatchError(f"bad arguments to tool {name!r}: {exc}") from exc
