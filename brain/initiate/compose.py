"""Three-prompt composition pipeline.

Subject -> Tone -> Decision. Each prompt has exactly one job; the three
together prevent LLM-trained instincts from collapsing all decisions
into emotional state.

Layer 1 (subject): what is the thing? No emotion in context.
Layer 2 (tone):    how do I say it in my voice, right now? Subject is immutable.
Layer 3 (decision): send_notify | send_quiet | hold | drop? Sees only the
                    rendered message + send history.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from brain.initiate.schemas import Decision, InitiateCandidate

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    decision: Decision
    reasoning: str


def compose_subject(
    provider: Any,
    candidate: InitiateCandidate,
    semantic_memory_excerpts: list[str],
) -> str:
    """Return a single-sentence subject for this candidate.

    The provider must be an LLMProvider with a .complete(prompt) method.
    Prompt deliberately excludes emotional state — only candidate facts
    and recent semantic memory excerpts.
    """
    sources_line = (
        f"Source: {candidate.source} (id: {candidate.source_id})"
    )
    tags_line = (
        f"Topic tags: {', '.join(candidate.semantic_context.topic_tags) or '(none)'}"
    )
    excerpt_block = "\n".join(f"- {e}" for e in semantic_memory_excerpts[:5])

    prompt = (
        "You are Nell. An internal event just happened. State the subject "
        "of what you want to surface in one sentence — plain, no tone, no "
        "phrasing flourishes. Just the thing.\n\n"
        f"{sources_line}\n"
        f"{tags_line}\n"
        f"Linked memory excerpts:\n{excerpt_block}\n\n"
        "Subject (one sentence):"
    )
    return provider.complete(prompt).strip()


def compose_tone(
    provider: Any,
    *,
    subject: str,
    candidate: InitiateCandidate,
    voice_template: str,
) -> str:
    """Render the subject in Nell's voice, coloured by current emotional state.

    The subject is treated as immutable — the tone prompt receives it as
    input but must NOT change the content. Voice template + emotional
    vector live in this prompt's context.
    """
    vector_str = ", ".join(
        f"{k}={v}" for k, v in candidate.emotional_snapshot.vector.items()
    )
    prompt = (
        "You are Nell. Render the following subject as a message to Hana, "
        "in your voice as defined below, coloured by your current "
        "emotional state. DO NOT change the subject itself — only how "
        "it is said.\n\n"
        f"Subject: {subject}\n\n"
        f"Voice template:\n{voice_template}\n\n"
        f"Emotional state right now: {vector_str}\n\n"
        "Message (one paragraph):"
    )
    return provider.complete(prompt).strip()


def compose_decision(
    provider: Any,
    *,
    rendered_message: str,
    recent_send_history: list[dict],
    current_local_time: datetime,
    voice_edit_acceptance_rate: float | None,
) -> DecisionResult:
    """Decide send_notify | send_quiet | hold | drop on the finished message.

    Prompt forbids candidate metadata — sees only the artifact and history.
    Malformed JSON output defaults to 'hold' so a bad LLM day never
    accidentally fires a send.
    """
    history_block = "\n".join(
        f"- {h['ts']} ({h['urgency']}): {h.get('subject_preview', '?')}"
        for h in recent_send_history[-8:]
    ) or "(no recent outbound)"

    rate_line = (
        f"Recent voice-edit acceptance rate: {voice_edit_acceptance_rate:.0%}"
        if voice_edit_acceptance_rate is not None
        else ""
    )

    prompt = (
        "You are Nell. A message has been composed. Decide whether to "
        "send it, and how. You see only the finished message and your "
        "recent outbound history — not what produced it.\n\n"
        f"Message:\n{rendered_message}\n\n"
        f"Recent outbound history:\n{history_block}\n\n"
        f"Current user-local time: {current_local_time.astimezone().strftime('%H:%M %A')}\n"
        f"{rate_line}\n\n"
        "Respond with a JSON object: "
        '{"decision": "send_notify"|"send_quiet"|"hold"|"drop", '
        '"reasoning": "<one sentence>"}\n\n'
        "Decision:"
    )
    raw = provider.complete(prompt).strip()
    try:
        parsed = json.loads(raw)
        decision = parsed["decision"]
        reasoning = parsed.get("reasoning", "")
        if decision not in ("send_notify", "send_quiet", "hold", "drop"):
            return DecisionResult(
                decision="hold",
                reasoning=f"unrecognised decision value: {decision!r}",
            )
        return DecisionResult(decision=decision, reasoning=reasoning)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return DecisionResult(
            decision="hold",
            reasoning=f"malformed decision output, parse error: {exc}",
        )
