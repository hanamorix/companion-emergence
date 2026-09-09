"""Assembled system prompts for the attunement detector."""
from __future__ import annotations

from brain import prompt_strings

# Text externalized to prompt_strings.toml [attunement.prompts] (issue #129 stage 1).
_DETECTOR_SYSTEM_PROMPT = prompt_strings.register("attunement.prompts.detector_system_prompt")


_IDENTITY_TEMPLATE = prompt_strings.register("attunement.prompts.identity_template")


def build_detector_system_prompt(
    only_categories: frozenset[str] | None = None,
    *,
    companion_name: str = "",
    user_name: str = "",
    user_pronouns: object = None,
) -> str:
    """Return the detector system prompt.

    When *only_categories* is provided, appends a restriction instruction
    telling the model to extract candidates for those categories only — used
    by the supplementary backfill pass so existing tone/cadence patterns are
    not double-counted. When None, returns the base prompt unchanged
    (preserves existing behaviour for normal per-turn detector calls).

    When *companion_name* is provided, appends an identity-grounding block.
    Without it the CLI-wrapped detector has no name for the companion and
    falls back to its own self-concept, writing "Claude" into pattern
    descriptions (live report, 2026-06-11). *user_name* names the user in
    the same block (generic "the user" when empty) — both come from runtime
    persona state (persona_dir.name / PersonaConfig.user_name), never
    hardcoded. Empty companion_name → base prompt unchanged (no dangling
    block).

    Deterministic; tests pin its content.
    """
    prompt = _DETECTOR_SYSTEM_PROMPT
    if only_categories is not None:
        cats_str = ", ".join(sorted(only_categories))
        prompt += (
            f"\n\nFOR THIS PASS ONLY: extract candidates for these categories: "
            f"{cats_str}. Do NOT emit candidates for any other category."
        )
    if companion_name:
        from brain.pronouns import resolve

        pr = resolve(user_pronouns)
        prompt += _IDENTITY_TEMPLATE.format(
            companion_name=companion_name,
            user_label=user_name or "the user",
            subject=pr.subject,
            object=pr.object,
        )
    return prompt
