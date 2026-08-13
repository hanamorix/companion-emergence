"""Prompt-size guard for engine.respond — the last-resort backstop.

**Normal compaction is startup/idle-only (owner ruling 2026-08-13).** Routine
compaction runs at startup + during idle ticks (supervisor), never mid-exchange.
This module is the *emergency net* for the one case that gate cannot cover: a long
ACTIVE session with no idle gap whose live prompt has ALREADY crossed the cap
(properly Phase 7 / ROOT1's job). apply_budget estimates the assembled prompt size
(len(content_text) // 4 per message) and, only when it exceeds ``max_tokens``:
  1. LAST-RESORT persisted fold: fires the compaction core (emergency_fold_24h) on
     the buffer — folding old, already-extracted turns into the head summary block
     and archiving them, so the NEXT turn is back under cap. This is a rarely-hit
     safety valve: with startup/idle compaction keeping normal sessions bounded, it
     only fires when that has failed to keep up (a pathological active session). It
     is NOT routine mid-exchange compaction; it triggers only when already over-cap.
  2. For the CURRENT over-cap prompt, applies a DETERMINISTIC truncation note as
     the in-prompt floor — a non-LLM, no-buffer-write WINDOWING trim (the per-turn
     provider.generate summary this module used to insert is removed; it busted
     prompt caching every turn). **This step, not step 1, is what actually bounds
     the within-session prompt** — so the size bound survives regardless of the
     idle-gate, and idle-gating routine compaction cannot cause unbounded growth.

The original system message is never compressed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import LLMProvider

logger = logging.getLogger(__name__)

# Head marker of the persisted compaction-summary system block that
# engine._buffer_turns_to_messages inserts at index 1 ("[Earlier in this
# conversation: ...]"). Must stay in sync with that f-string (engine.py).
_COMPACTION_SUMMARY_PREFIX = "[Earlier in this conversation:"


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
    persona_dir: Path | None = None,
    session_id: str | None = None,
) -> list[ChatMessage]:
    """Return a message list that fits inside ``max_tokens`` (last-resort backstop).

    Identity transform when the estimate is below max_tokens OR when the message
    list is too short to have a head to compress (fewer than 2 + preserve_tail_msgs
    entries: system + preserved tail).

    When over cap, fires the persisted compaction core on the buffer (so the
    fade is durable + archived and the next turn is back under cap) and applies a
    deterministic in-prompt truncation note for the current turn. The original
    system message is never compressed; only the head-between-system-and-tail is
    replaced by the note.
    """
    if _estimate_tokens(messages) <= max_tokens:
        return messages

    # 1. Persisted fade of the buffer (mirrors the daily cadence). Best-effort:
    #    a failure here must not break the turn — the deterministic note below
    #    still bounds the current prompt. older_than=0 ⇒ cutoff = ingest cursor,
    #    so all *extracted* turns past the tail fold in (un-extracted ones are
    #    left intact by the core's cursor guard).
    if persona_dir is not None and session_id:
        try:
            from brain.chat.compaction import (
                build_compaction_provider,
                emergency_fold_24h,
            )

            # 24h-only emergency fold on the sectioned row (bounds the live head
            # in-turn; does NOT run the full age-gated re-bucket, which stays on
            # the daily tick). Tiers 2/3 are left untouched. See plan §1.3 / C22.
            emergency_fold_24h(
                persona_dir,
                session_id,
                # Compaction always folds with COMPACTION_MODEL, not the chat model.
                provider=build_compaction_provider(persona_dir),
                min_keep_tail=preserve_tail_msgs,
            )
        except Exception:
            logger.exception(
                "apply_budget: persisted compaction failed session=%s; using in-prompt floor",
                session_id,
            )

    # 2. Deterministic in-prompt floor for the current over-cap turn (no LLM).
    if len(messages) < 2 + preserve_tail_msgs:
        return messages
    system_msg = messages[0]
    # Preserve the persisted compaction-summary block if present. engine inserts
    # it as a role="system" ChatMessage at index 1 ("[Earlier in this
    # conversation: ...]"); it carries the faded old context and must survive the
    # truncation instead of being swallowed into the "[truncated N]" note (#11).
    preserved_head = [system_msg]
    body_start = 1
    if (
        len(messages) > 1
        and messages[1].role == "system"
        and messages[1].content_text().startswith(_COMPACTION_SUMMARY_PREFIX)
    ):
        preserved_head.append(messages[1])
        body_start = 2
    head = messages[body_start : len(messages) - preserve_tail_msgs]
    tail = messages[-preserve_tail_msgs:]
    if not head:
        return messages
    summary_msg = ChatMessage(
        role="system",
        content=f"[truncated {len(head)} earlier messages]",
    )
    return [*preserved_head, summary_msg, *tail]
