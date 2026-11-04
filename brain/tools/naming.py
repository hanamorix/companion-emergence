"""Brain-tool names as the model must call them (#417, #419).

The Claude CLI registers the brain tools as ``mcp__brain-tools__<name>``; Ollama takes them bare.
The provider declares its prefix (``LLMProvider.tool_name_prefix``) and each text the model reads
gets it where the text is built: static strings carry ``TOOL_PREFIX_SLOT`` in front of each tool
name, filled by :func:`fill_tool_prefix`; voice.md keeps plain names on disk and goes through
:func:`prefix_tool_names` when it is loaded into the prompt.
"""

from __future__ import annotations

import re

TOOL_PREFIX_SLOT = "{tool_prefix}"


def tool_prefix_of(provider: object) -> str:
    """The provider's tool-name prefix; a duck-typed provider without one names tools bare.

    A wrapper around a real provider must forward ``tool_name_prefix`` (``_StreamingProxy`` does),
    or its turns fall back to bare names here.
    """
    return getattr(provider, "tool_name_prefix", "")


def fill_tool_prefix(text: str, prefix: str) -> str:
    return text.replace(TOOL_PREFIX_SLOT, prefix)


# The result fields the brain itself words (gate reasons, the add_name reply, refusals). Content
# fields (a file, a memory) are left as written, even if they happen to contain the slot.
_WORDED_RESULT_KEYS = ("reason", "message", "error")


def fill_tool_result(result: object, prefix: str) -> object:
    """A dispatch result with the prefix filled into the fields the brain words itself."""
    if not isinstance(result, dict):
        return result
    return {
        k: fill_tool_prefix(v, prefix) if k in _WORDED_RESULT_KEYS and isinstance(v, str) else v
        for k, v in result.items()
    }


def prefix_tool_names(text: str, prefix: str) -> str:
    """Prefix each backticked brain-tool name (`name` or `name(...)`) in free text.

    Only quoted names are touched: plain words like "boot" stay prose, and a name that already
    carries a prefix is not matched again.
    """
    if not prefix:
        return text
    from brain.tools import NELL_TOOL_NAMES  # brain.tools imports schemas, which imports this

    names = "|".join(sorted(map(re.escape, NELL_TOOL_NAMES), key=len, reverse=True))
    return re.sub(rf"`({names})(?=[`(])", lambda m: f"`{prefix}{m.group(1)}", text)
