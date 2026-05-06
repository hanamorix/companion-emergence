"""SP-4 EXTRACT + SCORE stages — LLM-based memory extraction from transcripts.

Calls LLMProvider.generate() with a structured extraction prompt, parses the
JSON array response, and returns normalized ExtractedItem instances.

Defense strategy:
- LLMs may wrap output in code fences; strip them.
- LLMs may prepend prose; locate first '[' and last ']'.
- On any parse failure, log a WARNING and return None so the retry loop kicks.
- After max_retries exhausted, return [] — callers never see an exception.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from brain.bridge.provider import LLMProvider
from brain.ingest.types import ExtractedItem

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT_LEGACY = """You are extracting durable memories from a conversation transcript.
Return ONLY a JSON array. Each item: {{"text": str, "label": one of [observation, feeling, decision, question, fact, note], "importance": 1-10}}.
Skip pleasantries. Keep items concrete. No prose, no commentary.

TRANSCRIPT:
{transcript}

JSON:"""

EXTRACTION_PROMPT_NAMED = """You are extracting durable memories from a conversation transcript.

Speakers in this transcript:
- {user_name} is the human user the assistant is talking to. Statements
  attributed to {user_name} (her actions, her decisions, her words) belong
  to {user_name}, not to anyone else.
- {assistant_name} is the assistant — the AI persona. Her replies may
  reference other people by name (from her memories, her soul, her
  history). Those are HISTORICAL references, not the current speaker.
  Do NOT attribute the current user's actions to anyone {assistant_name}
  mentioned by name.

Return ONLY a JSON array. Each item:
{{"text": str, "label": one of [observation, feeling, decision, question, fact, note], "importance": 1-10}}.
Skip pleasantries. Keep items concrete. No prose, no commentary.

TRANSCRIPT:
{transcript}

JSON:"""

# Backward-compat alias — older callers still import EXTRACTION_PROMPT.
EXTRACTION_PROMPT = EXTRACTION_PROMPT_LEGACY


@dataclass(frozen=True)
class ExtractionOutcome:
    """Result of an extraction attempt.

    ``items=[]`` is valid when the transcript contained no durable memories.
    ``failed=True`` means provider/parse failures exhausted retries and the
    caller should retain the source buffer for a future retry.
    """

    items: list[ExtractedItem]
    failed: bool = False
    error: str | None = None


def format_transcript(
    turns: list[dict],
    max_tokens: int = 6000,
    *,
    user_name: str | None = None,
    assistant_name: str | None = None,
) -> str:
    """Format turns as "speaker: text" lines, capped to max_tokens*4 chars.

    The crude 4-chars-per-token estimate matches the OG implementation.
    When the transcript is too long, the *tail* is kept — recent context
    is more relevant for extraction.

    user_name / assistant_name: when provided, replace the generic
    "user:" / "assistant:" speaker labels with the actual names. This
    disambiguates extraction when the assistant's replies reference
    historical figures by name (Bug A in the 2026-05-05 audit-3:
    Hana's framework work was attributed to Jordan because Jordan was
    mentioned in the assistant's soul-context lines).
    """
    user_label = user_name if user_name else "user"
    assistant_label = assistant_name if assistant_name else "assistant"
    lines = []
    for t in turns:
        speaker = t.get("speaker", "?")
        if speaker == "user":
            label = user_label
        elif speaker == "assistant":
            label = assistant_label
        else:
            label = speaker
        lines.append(f"{label}: {t.get('text', '')}")
    text = "\n".join(lines)
    cap = max_tokens * 4
    if len(text) > cap:
        text = text[-cap:]
    return text


def parse_extraction(raw: str | None) -> list[ExtractedItem] | None:
    """Parse the LLM's raw text into a list of ExtractedItems.

    Returns:
        A list (possibly empty) on success.
        None on any parse failure — the caller should retry.

    Handles:
    - Code fences (```json ... ```)
    - Leading/trailing prose (locate first '[' and last ']')
    - Empty arrays (valid — transcript had only pleasantries)
    """
    if not raw:
        return None

    text = raw.strip()

    # Strip code fences if the model wrapped the output.
    if text.startswith("```"):
        # Remove leading ``` and optional language tag line.
        text = text.lstrip("`")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            # If the first line after stripping backticks is a language tag
            # (like "json"), discard it.
            if not first_line.strip().startswith("[") and not first_line.strip().startswith("{"):
                text = rest
            else:
                text = first_line + "\n" + rest
        # Remove trailing ```
        text = text.rstrip("`").strip()

    # Locate the JSON array boundaries — ignore any prose before/after.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        logger.warning("parse_extraction: could not locate JSON array in: %r", text[:200])
        return None

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("parse_extraction: JSON decode failed: %s | text: %r", exc, text[:200])
        return None

    if not isinstance(data, list):
        logger.warning("parse_extraction: expected list, got %s", type(data).__name__)
        return None

    out: list[ExtractedItem] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        out.append(
            ExtractedItem(
                text=str(entry.get("text", "")),
                label=str(entry.get("label", "observation")),
                importance=entry.get("importance", 5),
            )
        )
    return out


def extract_items(
    transcript: str,
    *,
    provider: LLMProvider,
    max_retries: int = 1,
    user_name: str | None = None,
    assistant_name: str | None = None,
) -> list[ExtractedItem]:
    """Call the provider, parse the JSON array. Retry once on failure.

    On all retries exhausted, logs a warning and returns [].
    Errors are never raised — the pipeline must keep running.

    When user_name AND assistant_name are both provided, uses the named
    extraction prompt that explicitly disambiguates current-user
    statements from assistant references to historical figures.
    Otherwise falls back to the legacy prompt (backward-compatible).
    """
    return extract_items_with_status(
        transcript,
        provider=provider,
        max_retries=max_retries,
        user_name=user_name,
        assistant_name=assistant_name,
    ).items


def extract_items_with_status(
    transcript: str,
    *,
    provider: LLMProvider,
    max_retries: int = 1,
    user_name: str | None = None,
    assistant_name: str | None = None,
) -> ExtractionOutcome:
    """Call the provider and distinguish valid-empty from failed-empty.

    Backward-compatible ``extract_items`` intentionally keeps returning just a
    list. The ingest pipeline uses this richer outcome so provider/parse
    failures do not look like successful empty extractions.
    """
    if not transcript.strip():
        return ExtractionOutcome(items=[])

    if user_name and assistant_name:
        prompt = EXTRACTION_PROMPT_NAMED.format(
            transcript=transcript,
            user_name=user_name,
            assistant_name=assistant_name,
        )
    else:
        prompt = EXTRACTION_PROMPT_LEGACY.format(transcript=transcript)

    last_error: str | None = None
    for attempt in range(max_retries + 1):
        raw_text: str | None = None
        try:
            raw_text = provider.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            last_error = f"provider.generate failed on attempt {attempt}: {exc}"
            logger.warning(
                "extract_items: provider.generate failed on attempt %d: %s", attempt, exc
            )
            continue

        items = parse_extraction(raw_text)
        if items is not None:
            return ExtractionOutcome(items=items)

        last_error = f"parse failed on attempt {attempt}/{max_retries}"
        logger.warning("extract_items: parse failed on attempt %d/%d", attempt, max_retries)

    logger.warning("extract_items: gave up after %d retries", max_retries)
    return ExtractionOutcome(
        items=[],
        failed=True,
        error=last_error or f"gave up after {max_retries} retries",
    )
