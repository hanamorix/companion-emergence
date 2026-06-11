"""Pronoun core — presets, resolve fallback, verb agreement.

Spec: docs/superpowers/specs/2026-06-11-user-pronouns-design.md §1.
"""
from brain.pronouns import DEFAULT_KEY, PRESETS, resolve


def test_three_presets_exist_with_full_sets():
    assert set(PRESETS) == {"she/her", "he/him", "they/them"}
    she = PRESETS["she/her"]
    assert (she.subject, she.object, she.possessive) == ("she", "her", "her")
    assert (she.possessive_standalone, she.reflexive) == ("hers", "herself")
    assert she.plural_verbs is False
    they = PRESETS["they/them"]
    assert (they.subject, they.object, they.possessive) == ("they", "them", "their")
    assert (they.possessive_standalone, they.reflexive) == ("theirs", "themself")
    assert they.plural_verbs is True
    he = PRESETS["he/him"]
    assert (he.subject, he.object, he.possessive) == ("he", "him", "his")
    assert (he.possessive_standalone, he.reflexive) == ("his", "himself")
    assert he.plural_verbs is False


def test_resolve_none_falls_back_to_she_her():
    assert resolve(None) == PRESETS[DEFAULT_KEY] == PRESETS["she/her"]


def test_resolve_preset_key():
    assert resolve("they/them") == PRESETS["they/them"]


def test_resolve_full_dict_custom_set():
    """Tripwire (spec §Deferred, custom-pronoun UI): the schema capability
    must not rot — resolve() accepts a full custom dict faithfully.
    Ledger: project_companion_emergence_deferred.md."""
    custom = {
        "subject": "xe", "object": "xem", "possessive": "xyr",
        "possessive_standalone": "xyrs", "reflexive": "xemself",
        "plural_verbs": False,
    }
    p = resolve(custom)
    assert p.subject == "xe" and p.reflexive == "xemself" and p.plural_verbs is False


def test_resolve_garbage_falls_back():
    for garbage in ("zir", 42, {"subject": "xe"}, {"subject": 1}, []):
        assert resolve(garbage) == PRESETS["she/her"]


def test_verb_agreement_and_cap():
    she, they = PRESETS["she/her"], PRESETS["they/them"]
    assert she.v("seems", "seem") == "seems"
    assert they.v("seems", "seem") == "seem"
    assert they.v("hasn't", "haven't") == "haven't"
    assert she.cap(she.subject) == "She"
    assert they.cap(they.subject) == "They"
