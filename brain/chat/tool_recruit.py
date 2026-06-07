"""Per-turn tool recruitment — which tools go into --allowedTools this turn.
The MCP server exposes ALL tools; this only gates what Nell may call this turn.
Reflexive core is always present (incl. reach_for_capability so she can always
escalate). Heavier faculties recruited on salience flags. Maximal signal (the
fail-open case) recruits everything → behaves exactly like today."""
from __future__ import annotations

from brain.chat.salience import SalienceSignal
from brain.tools import NELL_TOOL_NAMES

# SalienceSignal.maximal() yields score=1.0; 0.999 absorbs float drift only.
_MAXIMAL_SCORE = 0.999

# Tiers cover the common per-turn faculties. boot (startup-only), crystallize_soul,
# and the works tools are intentionally omitted — they arrive via the full suite on a
# maximal signal or via reach_for_capability's full re-invoke (see tool_loop, Task 2.3).

# Always available — interior voice, self-state reads, and the escalation valve.
REFLEXIVE_CORE: tuple[str, ...] = (
    "record_monologue",
    "recall_monologue",
    "reach_for_capability",
    "get_emotional_state",
    "get_body_state",
    "get_soul",
    "get_personality",
    "felt_time_now",
    "pressure_since",
)

_MEMORY_TOOLS = (
    "search_memories",
    "recall_forgotten",
    "recall_arc",
    "list_open_arcs",
    "add_memory",
    "add_journal",
)
_FILE_TOOLS = ("read_file", "list_directory")


def select_tools(
    signal: SalienceSignal,
    *,
    base: tuple[str, ...] = NELL_TOOL_NAMES,
) -> list[str]:
    """Return the list of tool names allowed for this turn.

    Fails open: maximal signal → full suite (today's behaviour). Non-maximal
    turns get REFLEXIVE_CORE plus whichever heavier faculties the salience
    flags call for. Result is ordered by *base* so the LLM sees a stable list.
    """
    if signal.score >= _MAXIMAL_SCORE:  # maximal / fail-open → full suite
        return list(base)

    keep: set[str] = set(REFLEXIVE_CORE)

    if signal.references_past or signal.mentions_entity_or_date or signal.topic_shift:
        keep.update(_MEMORY_TOOLS)

    if signal.mentions_file_or_path:
        keep.update(_FILE_TOOLS)

    # Preserve base ordering; drop anything not in base.
    return [t for t in base if t in keep]
