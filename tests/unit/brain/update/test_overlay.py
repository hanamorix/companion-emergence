"""brain.update.overlay — the overlay's on-disk state (#286 §3.3)."""

from __future__ import annotations

import json

import pytest

from brain.update import overlay

ENTRY_A = {"dir": "aaaa", "commit": "a" * 40, "brain_version": "0.0.42", "bundle_id": "b1"}
ENTRY_B = {"dir": "bbbb", "commit": "b" * 40, "brain_version": "0.0.43", "bundle_id": "b1"}


def test_missing_state_reads_as_no_overlay(tmp_path):
    assert overlay.read_state(tmp_path) == {"active": None, "previous": None}


def test_activate_moves_the_old_active_to_previous(tmp_path):
    overlay.activate(tmp_path, ENTRY_A)
    overlay.activate(tmp_path, ENTRY_B)
    assert overlay.read_state(tmp_path) == {"active": ENTRY_B, "previous": ENTRY_A}


def test_reactivating_the_same_dir_keeps_previous(tmp_path):
    overlay.activate(tmp_path, ENTRY_A)
    overlay.activate(tmp_path, ENTRY_B)
    overlay.activate(tmp_path, ENTRY_B)
    assert overlay.read_state(tmp_path)["previous"] == ENTRY_A


def test_revert_clears_active_and_remembers_it(tmp_path):
    overlay.activate(tmp_path, ENTRY_A)
    overlay.revert(tmp_path)
    assert overlay.read_state(tmp_path) == {"active": None, "previous": ENTRY_A}


def test_rollback_restores_previous(tmp_path):
    overlay.activate(tmp_path, ENTRY_A)
    overlay.activate(tmp_path, ENTRY_B)
    overlay.rollback(tmp_path)
    assert overlay.read_state(tmp_path) == {"active": ENTRY_A, "previous": None}


def test_corrupt_or_non_object_state_reads_as_no_overlay(tmp_path):
    (tmp_path / "current.json").write_text("[1, 2]", encoding="utf-8")
    assert overlay.read_state(tmp_path) == {"active": None, "previous": None}
    (tmp_path / "current.json").write_text("{not json", encoding="utf-8")
    assert overlay.read_state(tmp_path) == {"active": None, "previous": None}


def test_unknown_keys_are_ignored(tmp_path):
    (tmp_path / "current.json").write_text(
        json.dumps({"active": ENTRY_A, "previous": None, "_future_field": 1}), encoding="utf-8")
    assert overlay.read_state(tmp_path)["active"] == ENTRY_A


def test_prune_keeps_only_active_and_previous(tmp_path):
    for name in ("aaaa", "bbbb", "cccc", ".staging-dddd-1"):
        (tmp_path / name).mkdir()
    overlay.activate(tmp_path, ENTRY_A)
    overlay.activate(tmp_path, ENTRY_B)
    overlay.prune(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == ["aaaa", "bbbb"]


def test_lock_refuses_a_second_live_holder(tmp_path):
    with overlay.overlay_lock(tmp_path):
        with pytest.raises(overlay.OverlayBusy):
            with overlay.overlay_lock(tmp_path):
                pass
    assert not (tmp_path / ".lock").exists()


def test_lock_takes_over_from_a_dead_holder(tmp_path):
    (tmp_path / ".lock").write_text("999999999", encoding="utf-8")  # no such pid
    with overlay.overlay_lock(tmp_path):
        assert (tmp_path / ".lock").read_text(encoding="utf-8") == str(__import__("os").getpid())
