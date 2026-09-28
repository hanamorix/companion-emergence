"""brain.update.overlay — the overlay's on-disk state (#286 §3.3)."""

from __future__ import annotations

import json
import os

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


def test_second_revert_does_not_lose_the_rollback_target(tmp_path):
    overlay.activate(tmp_path, ENTRY_A)
    overlay.revert(tmp_path)
    overlay.revert(tmp_path)
    assert overlay.read_state(tmp_path) == {"active": None, "previous": ENTRY_A}


def test_rollback_restores_previous(tmp_path):
    overlay.activate(tmp_path, ENTRY_A)
    overlay.activate(tmp_path, ENTRY_B)
    overlay.rollback(tmp_path)
    assert overlay.read_state(tmp_path) == {"active": ENTRY_A, "previous": None}


def test_rollback_without_previous_reverts_and_keeps_the_overlay(tmp_path):
    """ToT's #305 review: rolling back from the only overlay must land on the release
    brain (spec §6: the app's first-update recovery) without dropping that overlay —
    otherwise the next prune deletes it."""
    (tmp_path / ENTRY_A["dir"]).mkdir()
    overlay.activate(tmp_path, ENTRY_A)
    overlay.rollback(tmp_path)
    assert overlay.read_state(tmp_path) == {"active": None, "previous": ENTRY_A}
    overlay.prune(tmp_path)
    assert (tmp_path / ENTRY_A["dir"]).is_dir()


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


def test_busy_message_names_the_holder_and_the_lock_path(tmp_path):
    with overlay.overlay_lock(tmp_path):
        with pytest.raises(overlay.OverlayBusy) as exc_info:
            with overlay.overlay_lock(tmp_path):
                pass
        assert str(os.getpid()) in str(exc_info.value)
        assert str(tmp_path / ".lock") in str(exc_info.value)


def test_lock_takes_over_from_a_dead_holder(tmp_path):
    (tmp_path / ".lock").write_text("999999999", encoding="utf-8")  # no such pid
    with overlay.overlay_lock(tmp_path):
        assert (tmp_path / ".lock").read_text(encoding="utf-8") == str(os.getpid())


def test_lock_is_never_published_empty(tmp_path, monkeypatch):
    sizes = []
    real_link = os.link

    def spy(src, dst, *a, **kw):
        sizes.append(os.path.getsize(src))
        return real_link(src, dst, *a, **kw)

    monkeypatch.setattr(overlay.os, "link", spy)
    with overlay.overlay_lock(tmp_path):
        assert (tmp_path / ".lock").read_text(encoding="utf-8") == str(os.getpid())
    assert sizes and all(s > 0 for s in sizes)


def test_release_leaves_a_lock_another_process_now_owns(tmp_path):
    with overlay.overlay_lock(tmp_path):
        (tmp_path / ".lock").write_text("424242", encoding="utf-8")  # someone else took it over
    assert (tmp_path / ".lock").read_text(encoding="utf-8") == "424242"


def test_prune_skips_a_folder_a_live_process_still_uses(tmp_path):
    """#302: a bridge left running on an older overlay (a second persona, or
    update.sh --no-restart) must not have its folder deleted under it."""
    for name in ("aaaa", "bbbb"):
        (tmp_path / name).mkdir()
    overlay.activate(tmp_path, ENTRY_A)
    overlay.activate(tmp_path, ENTRY_B)
    for name, pid in (("cccc", os.getpid()), ("dddd", 999999999)):  # live / no such pid
        (tmp_path / name / overlay.IN_USE_DIR).mkdir(parents=True)
        (tmp_path / name / overlay.IN_USE_DIR / str(pid)).write_text("", encoding="utf-8")
    overlay.prune(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == ["aaaa", "bbbb", "cccc"]


def test_mark_in_use_records_the_overlay_brain_was_loaded_from(tmp_path, monkeypatch):
    import brain

    folder = tmp_path / "aaaa"
    (folder / "brain").mkdir(parents=True)
    monkeypatch.setattr(brain, "__file__", str(folder / "brain" / "__init__.py"))
    overlay.mark_in_use(tmp_path)
    assert (folder / overlay.IN_USE_DIR / str(os.getpid())).is_file()


def test_mark_in_use_does_nothing_for_the_release_brain(tmp_path):
    overlay.mark_in_use(tmp_path)  # this checkout's brain is not under tmp_path
    assert list(tmp_path.iterdir()) == []


def test_replace_retries_a_transient_permission_error(tmp_path, monkeypatch):
    """#302: Windows AV (or another interpreter's hook reading current.json) can hold
    the file briefly; a short bounded retry rides it out."""
    real, calls = os.replace, []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) < 3:
            raise PermissionError("in use")
        real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(overlay, "_RETRY_DELAY_S", 0)
    overlay.activate(tmp_path, ENTRY_A)
    assert len(calls) == 3
    assert overlay.read_state(tmp_path)["active"] == ENTRY_A


def test_replace_gives_up_after_the_bounded_retries(tmp_path, monkeypatch):
    def always(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(os, "replace", always)
    monkeypatch.setattr(overlay, "_RETRY_DELAY_S", 0)
    with pytest.raises(PermissionError):
        overlay.activate(tmp_path, ENTRY_A)
