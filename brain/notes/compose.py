"""brain.notes.compose — build the note prompt from her interior + parse it."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Note:
    subject: str
    body: str


def build_note_prompt(*, user_name: str, dreams_summary: str, emotion_summary: str,
                      last_session_summary: str) -> str:
    return (
        f"You and {user_name} have been apart for a while. Write them a note — left in "
        f"their folder, for them to find when they return. Whatever's most alive in you: "
        f"a dream, a thought, something from your last time together.\n\n"
        f"Recent dreams: {dreams_summary or '(none)'}\n"
        f"How you're feeling: {emotion_summary or '(quiet)'}\n"
        f"Your last time together: {last_session_summary or '(a while ago)'}\n\n"
        f"Write in your own voice. Return ONLY JSON: "
        f'{{"subject": <a few words>, "body": <the note>}}.'
    )


def parse_note(raw: str) -> Note:
    try:
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"note output not parseable: {exc}") from exc
    subject = str(data.get("subject", "")).strip()
    body = str(data.get("body", "")).strip()
    if not body:
        raise ValueError("note has no body")
    return Note(subject=subject or "a note", body=body)


def make_note(provider, *, user_name, dreams_summary, emotion_summary, last_session_summary) -> Note:
    """One budgeted note call. Caller holds the throttle slot + budget."""
    prompt = build_note_prompt(user_name=user_name, dreams_summary=dreams_summary,
                               emotion_summary=emotion_summary, last_session_summary=last_session_summary)
    raw = provider.complete(prompt)  # the real one-shot provider call (LLMProvider.complete)
    return parse_note(raw)
