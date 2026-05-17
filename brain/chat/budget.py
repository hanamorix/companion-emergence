"""Prompt-size guard for engine.respond.

apply_budget inspects the message list, estimates its size via
len(content_text) // 4 per message, and when it exceeds ``max_tokens``:
  1. Splits into [system, *head_to_compress, *preserved_tail] where the
     preserved tail is the last ``preserve_tail_msgs`` messages.
  2. Concatenates the head into a transcript and asks the provider to
     summarise.
  3. Returns [system, compressed_head_system_note, *preserved_tail].

On provider failure, falls back to a deterministic truncation note. The
original system message is never compressed.
"""

from __future__ import annotations

import logging

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import LLMProvider

logger = logging.getLogger(__name__)

_COMPRESSION_PROMPT = """Summarize the following conversation for context preservation.
Preserve: names of people and places, decisions made, emotional beats,
unresolved threads, anything that would be referenced later.
Drop: pleasantries, repetition, formatting noise.
Output prose only, no headers or lists.

CONVERSATION:
{transcript}

SUMMARY:"""


def _estimate_tokens(messages: list[ChatMessage]) -> int:
    """Crude char-based token estimate matching brain.ingest.extract."""
    total_chars = 0
    for m in messages:
        total_chars += len(m.content_text())
    return total_chars // 4


def apply_budget(
    messages: list[ChatMessage],
    *,
    max_tokens: int = 190_000,
    preserve_tail_msgs: int = 40,
    provider: LLMProvider,
) -> list[ChatMessage]:
    """Return a message list that fits inside ``max_tokens``.

    Identity transform when the estimate is below max_tokens OR when the
    message list is too short to have a head to compress (i.e. fewer than
    2 + preserve_tail_msgs entries: system + preserved tail).

    The original system message is never compressed. Only the
    head-between-system-and-preserved-tail gets summarised.
    """
    if _estimate_tokens(messages) <= max_tokens:
        return messages

    if len(messages) < 2 + preserve_tail_msgs:
        return messages

    system_msg = messages[0]
    head = messages[1 : len(messages) - preserve_tail_msgs]
    tail = messages[-preserve_tail_msgs:]

    if not head:
        return messages

    transcript = "\n".join(f"{m.role}: {m.content_text()}" for m in head)
    summary_msg: ChatMessage
    try:
        summary = provider.generate(prompt=_COMPRESSION_PROMPT.format(transcript=transcript))
        summary_msg = ChatMessage(
            role="system",
            content=f"[Earlier in this conversation: {summary.strip()}]",
        )
    except Exception:
        logger.exception("apply_budget: provider summarisation failed; falling back")
        summary_msg = ChatMessage(
            role="system",
            content=f"[truncated {len(head)} earlier messages]",
        )

    return [system_msg, summary_msg, *tail]
