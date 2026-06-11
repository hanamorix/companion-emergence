"""Tests for brain.bridge.pronoun_nudge — one-time pronoun-setting nudge.

Spec: docs/superpowers/specs/2026-06-11-user-pronouns-design.md §5.
"""
from __future__ import annotations

import json


def test_nudge_marker_written_when_pronouns_unset(tmp_path):
    from brain.bridge.pronoun_nudge import maybe_write_pronoun_nudge

    assert maybe_write_pronoun_nudge(tmp_path, companion_name="Mira") is True
    assert (tmp_path / "pronoun_nudge.json").exists()
    assert maybe_write_pronoun_nudge(tmp_path, companion_name="Mira") is False  # once ever


def test_nudge_not_written_when_pronouns_set(tmp_path):
    from brain.bridge.pronoun_nudge import maybe_write_pronoun_nudge
    from brain.pronouns import PRESETS, to_dict

    (tmp_path / "persona_config.json").write_text(
        json.dumps({"user_pronouns": to_dict(PRESETS["he/him"])})
    )
    assert maybe_write_pronoun_nudge(tmp_path, companion_name="Mira") is False
    assert not (tmp_path / "pronoun_nudge.json").exists()


def test_feed_emits_nudge_entry(tmp_path):
    from brain.bridge.pronoun_nudge import build_pronoun_nudge_entries, maybe_write_pronoun_nudge

    maybe_write_pronoun_nudge(tmp_path, companion_name="Mira")
    entries = build_pronoun_nudge_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0].type == "pronoun_nudge"
    assert "pronouns" in entries[0].body and "Connection panel" in entries[0].body
