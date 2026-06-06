"""Per-turn salience signal — cheap, deterministic, no LLM. Keystone for
tool recruitment (select_tools) and reflection debounce (should_reflect).

Fails open: any exception → SalienceSignal.maximal() so a scorer bug can only
ever cost MORE (attach tools / reflect), never silently strip agency."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

_PAST_CUES = (
    "remember", "last time", "you said", "earlier", "before", "yesterday",
    "the other day", "we talked", "we discussed", "back when", "used to",
)
_FILE_CUES = ("file", "folder", "directory", "desktop", "read ", "open ", "/", "~", ".txt", ".md", ".py")
_DATE_RE = re.compile(r"\b(\d{1,4}[/-]\d{1,2}([/-]\d{1,4})?|\d{4}|\d{1,2}(am|pm))\b", re.I)
_CAP_MIDSENTENCE_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}\b")
_BASE_AFFECT = frozenset({
    "love", "hate", "afraid", "scared", "angry", "sad", "happy", "anxious",
    "tired", "exhausted", "lonely", "hurt", "grief", "joy", "fear", "ashamed",
    "proud", "guilty", "hopeful", "worried", "overwhelmed",
})


def _emotion_names() -> frozenset[str]:
    from brain.emotion import vocabulary
    return frozenset(e.name.lower() for e in vocabulary.list_all())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


@dataclass(frozen=True)
class SalienceSignal:
    score: float
    references_past: bool
    mentions_entity_or_date: bool
    emotional_density: float
    is_question: bool
    mentions_file_or_path: bool
    topic_shift: bool
    word_count: int

    @classmethod
    def maximal(cls) -> SalienceSignal:
        return cls(1.0, True, True, 1.0, True, True, True, 0)


def assess_salience(
    user_input: str,
    *,
    prior_user_text: str | None = None,
    persona_dir=None,  # accepted for interface symmetry; unused today
) -> SalienceSignal:
    try:
        text = user_input or ""
        low = text.lower()
        toks = _tokens(text)
        wc = len(toks)

        references_past = any(cue in low for cue in _PAST_CUES)
        mentions_file = any(cue in low for cue in _FILE_CUES)
        is_question = "?" in text
        mentions_entity_or_date = bool(_DATE_RE.search(text)) or bool(_CAP_MIDSENTENCE_RE.search(text))

        affect = _BASE_AFFECT | _emotion_names()
        hits = sum(1 for t in toks if t in affect)
        emotional_density = (hits / wc) if wc else 0.0

        topic_shift = False
        if prior_user_text:
            prev = set(_tokens(prior_user_text))
            cur = set(toks)
            if prev and cur:
                jacc = len(prev & cur) / len(prev | cur)
                topic_shift = jacc < 0.12

        # Composite — any strong signal lifts it; length contributes mildly.
        score = 0.0
        score += 0.30 if references_past else 0.0
        score += 0.20 if mentions_entity_or_date else 0.0
        score += 0.20 if mentions_file else 0.0
        score += 0.15 if is_question else 0.0
        score += 0.15 if topic_shift else 0.0
        score += min(0.20, emotional_density * 0.8)
        score += min(0.15, wc / 100.0)
        score = min(1.0, score)

        return SalienceSignal(
            score=score,
            references_past=references_past,
            mentions_entity_or_date=mentions_entity_or_date,
            emotional_density=emotional_density,
            is_question=is_question,
            mentions_file_or_path=mentions_file,
            topic_shift=topic_shift,
            word_count=wc,
        )
    except Exception:  # noqa: BLE001 — fail open: never strip agency on a scorer bug
        log.debug("assess_salience failed; returning maximal signal", exc_info=True)
        return SalienceSignal.maximal()
