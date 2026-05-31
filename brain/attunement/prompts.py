"""Assembled system prompts for the attunement detector."""
from __future__ import annotations

_DETECTOR_SYSTEM_PROMPT = """You are extracting attunement signals from a conversation.

Your output is a JSON object matching this schema:
{
  "current_read": {
    "tone_label": "warm" | "frayed" | "raw" | "fizzy" | "focused" | "distant" | "measured" | "withdrawn",
    "tone_justification": "one-sentence ground for the label",
    "cadence_label": "terse" | "measured" | "expansive",
    "cadence_justification": "one-sentence ground for the label",
    "mood_valence": -1.0 to +1.0,
    "mood_intensity": 0.0 to 1.0,
    "predicted_arc_shape": "one short sentence on where this conversation seems to be heading"
  },
  "pattern_candidates": [
    {
      "category": "tone" | "cadence",
      "canonical_key": "stable identifier",
      "description": "one-sentence natural language",
      "evidence_quote": "VERBATIM substring from one of the turns below",
      "evidence_turn_id": "the id of the turn the quote came from"
    }
  ]
}

CRITICAL RULES:

1. Every pattern_candidate MUST include `evidence_quote` — verbatim text
   copied from one of the user's turns. If you cannot quote, OMIT the
   candidate. Do not paraphrase. Do not fabricate.

2. Every pattern_candidate MUST include `evidence_turn_id` — the id of
   the turn the quote came from. If you can't identify the turn, OMIT.

3. Categories in this release: only "tone" and "cadence". Do not emit
   topic_affinity, response_shape, or relational candidates.

4. When the input is empty, ambiguous, or too short to read:
   - Return `tone_label`: "unknown" and `cadence_label`: "unknown"
   - Return an empty pattern_candidates list
   - Do NOT guess. Decline cleanly.

5. The `canonical_key` for a candidate must be stable across runs:
   if you observe "warm tone when discussing the dog" twice, both runs
   should produce the same canonical_key. Use kebab-case descriptive
   keys like "tone:warm-when-dog".

Your output is consumed by code that re-validates every quote against
the source. Fabricated quotes are silently rejected and logged. You
gain nothing by claiming what you cannot ground."""


def build_detector_system_prompt() -> str:
    """Return the detector system prompt. Deterministic; tests pin its content."""
    return _DETECTOR_SYSTEM_PROMPT
