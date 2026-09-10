"""brain.notes.compose — build the note prompt from her interior + parse it."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from brain import prompt_strings

logger = logging.getLogger(__name__)

# Text externalized to prompt_strings.toml [notes.compose] (issue #129 stage 2c).
_BUILD_NOTE_PROMPT_SEGMENTS = prompt_strings.register_segments("notes.compose.build_note_prompt_segments")


@dataclass
class Note:
    subject: str
    body: str


def build_note_prompt(*, persona_name: str, user_name: str, dreams_summary: str,
                      emotion_summary: str, last_session_summary: str) -> str:
    # #170: without an identity line the model signs the note "-Claude".
    seg = _BUILD_NOTE_PROMPT_SEGMENTS
    return (
        seg[0] + persona_name + seg[1] + user_name + seg[2]
        + (dreams_summary or '(none)') + seg[3]
        + (emotion_summary or '(quiet)') + seg[4]
        + (last_session_summary or '(a while ago)') + seg[5]
        + persona_name + seg[6] + persona_name + seg[7]
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


def make_note(provider, *, persona_name, user_name, dreams_summary, emotion_summary,
              last_session_summary) -> Note:
    """One budgeted note call. Caller holds the throttle slot + budget."""
    prompt = build_note_prompt(persona_name=persona_name, user_name=user_name,
                               dreams_summary=dreams_summary,
                               emotion_summary=emotion_summary, last_session_summary=last_session_summary)
    raw = provider.complete(prompt)  # the real one-shot provider call (LLMProvider.complete)
    return parse_note(raw)
