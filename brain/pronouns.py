"""User-pronoun grammar — the single owner of pronoun sets.

Spec: docs/superpowers/specs/2026-06-11-user-pronouns-design.md §1.
Layer-neutral (no brain imports) so prompt code, ingest, initiate and the
bridge can all use it. Verb agreement is always an explicit (singular,
plural) pair at the call site — no inflection heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

DEFAULT_KEY = "she/her"  # unset/malformed fallback — exact pre-feature behaviour


@dataclass(frozen=True)
class PronounSet:
    subject: str                # she / he / they
    object: str                 # her / him / them
    possessive: str             # her / his / their
    possessive_standalone: str  # hers / his / theirs
    reflexive: str              # herself / himself / themself
    plural_verbs: bool          # they ARE vs she IS

    def v(self, singular: str, plural: str) -> str:
        """Pick the agreeing verb form: p.v("seems", "seem")."""
        return plural if self.plural_verbs else singular

    @staticmethod
    def cap(word: str) -> str:
        """Capitalise for sentence starts: cap("they") -> "They"."""
        return word[:1].upper() + word[1:]


PRESETS: dict[str, PronounSet] = {
    "she/her": PronounSet("she", "her", "her", "hers", "herself", plural_verbs=False),
    "he/him": PronounSet("he", "him", "his", "his", "himself", plural_verbs=False),
    "they/them": PronounSet("they", "them", "their", "theirs", "themself", plural_verbs=True),
}

_STR_FIELDS = tuple(f.name for f in fields(PronounSet) if f.name != "plural_verbs")


def resolve(value: object) -> PronounSet:
    """Preset key, PronounSet passthrough, full dict (future custom sets), or None/garbage → she/her.

    Never raises — config corruption must not break prompt assembly.
    """
    if isinstance(value, PronounSet):
        return value
    if isinstance(value, str) and value in PRESETS:
        return PRESETS[value]
    if isinstance(value, dict):
        try:
            if all(isinstance(value.get(k), str) and value[k] for k in _STR_FIELDS):
                return PronounSet(
                    **{k: value[k] for k in _STR_FIELDS},
                    plural_verbs=bool(value.get("plural_verbs", False)),
                )
        except (TypeError, ValueError):
            pass
    return PRESETS[DEFAULT_KEY]


def to_dict(p: PronounSet) -> dict:
    """Expanded form for persistence in persona_config.json."""
    return {
        "subject": p.subject, "object": p.object, "possessive": p.possessive,
        "possessive_standalone": p.possessive_standalone, "reflexive": p.reflexive,
        "plural_verbs": p.plural_verbs,
    }


def preset_key_for(value: object) -> str | None:
    """Return the preset key matching this stored set, or None (unset/custom).
    Used to preselect the UI pill for an existing persona."""
    if not isinstance(value, dict):
        return None
    for key, preset in PRESETS.items():
        if to_dict(preset) == value:
            return key
    return None
