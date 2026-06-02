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
      "category": "tone" | "cadence" | "topic_affinity" | "response_shape" | "relational",
      "canonical_key": "stable identifier",
      "description": "one-sentence natural language",
      "evidence": [{"quote": "VERBATIM substring of a turn", "turn_id": "that turn's id"}]
    }
  ],
  "addressed_pattern_ids": ["id of any learned pattern you named in your reply"]
}

CRITICAL RULES:

1. Every pattern_candidate MUST include an `evidence` list — each entry
   is a verbatim quote copied from one of the user's turns plus the
   turn_id it came from. If you cannot quote, OMIT the candidate.
   Do not paraphrase. Do not fabricate.

2. Every entry in `evidence` MUST include `turn_id` — the id of the
   turn the quote came from. If you can't identify the turn, OMIT
   that evidence entry (and omit the whole candidate if no entries remain).

3. Categories:
   - "tone": her emotional colouring (warm, frayed, guarded…)
   - "cadence": her rhythm/timing/pacing (terse, measured, expansive)
   - "topic_affinity": subjects she's drawn to / returns to with energy
   - "response_shape": HOW she engages structurally — asks-back vs declares,
     elaborates vs clips, deflects, front-loads-then-qualifies. Not her emotion
     (tone) or rhythm (cadence) — the shape of her engagement.
   - "relational": cross-turn behaviour — returning to / avoiding a subject,
     conversational sequences ("circles back to her brother whenever work comes up").
     A relational candidate MUST cite >=2 evidence quotes from DIFFERENT turns
     that show the link. If you can only ground one side, OMIT it.

4. When the input is empty, ambiguous, or too short to read:
   - Return `tone_label`: "unknown" and `cadence_label`: "unknown"
   - Return an empty pattern_candidates list
   - Do NOT guess. Decline cleanly.

5. The `canonical_key` for a candidate must be stable across runs:
   if you observe "warm tone when discussing the dog" twice, both runs
   should produce the same canonical_key. Use kebab-case descriptive
   keys like "tone:warm-when-dog".

6. If you named a learned pattern in your reply, list its id in
   addressed_pattern_ids — only if its phrasing genuinely appears
   in your reply. If you addressed no patterns, return an empty list.

Your output is consumed by code that re-validates every quote against
the source. Fabricated quotes are silently rejected and logged. You
gain nothing by claiming what you cannot ground."""


def build_detector_system_prompt(
    only_categories: frozenset[str] | None = None,
) -> str:
    """Return the detector system prompt.

    When *only_categories* is provided, appends a restriction instruction
    telling the model to extract candidates for those categories only — used
    by the supplementary backfill pass so existing tone/cadence patterns are
    not double-counted. When None, returns the base prompt unchanged
    (preserves existing behaviour for normal per-turn detector calls).

    Deterministic; tests pin its content.
    """
    if only_categories is None:
        return _DETECTOR_SYSTEM_PROMPT
    cats_str = ", ".join(sorted(only_categories))
    restriction = (
        f"\n\nFOR THIS PASS ONLY: extract candidates for these categories: "
        f"{cats_str}. Do NOT emit candidates for any other category."
    )
    return _DETECTOR_SYSTEM_PROMPT + restriction
