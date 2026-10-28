"""#465 C2: the hand-written temp-and-replace writers now go through the shared atomic writer.

One section per site: the identity key (S1/S2), the judge pointer (S3), the migrator's legacy copy (S4), the image and
file blobs (S5/S6), the launchd plist (S7), the systemd unit refresh (S8) and the directory fsync callers (S9). Each
asserts the result, that no old-shape temp is left, that two transient PermissionErrors on the replace are retried and
that a failed write keeps the previous content. Nothing here imports the shared module, so the tests also run (and
fail where the hardening is missing) against the tree before the change.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
import threading
from pathlib import Path

import pytest

TEMP_RE = re.compile(r"\.t\d+-(?:k[0-9a-f]{12}-)?[0-9a-f]+\.new$")
posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")


def _flaky_replace(monkeypatch, suffix: str, failures: int = 2) -> dict:
    real = os.replace
    state = {"n": failures}

    def flaky(src, dst):
        if state["n"] > 0 and str(dst).endswith(suffix):
            state["n"] -= 1
            raise PermissionError(32, "sharing violation")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    return state


def _record_replace_sources(monkeypatch) -> list[str]:
    seen: list[str] = []
    real = os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        return real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    return seen


def _boom_fsync(monkeypatch):
    def boom(fd):
        raise OSError(28, "disk full")

    monkeypatch.setattr(os, "fsync", boom)


def _leftovers(directory: Path) -> list[str]:
    return sorted(
        p.name
        for p in directory.iterdir()
        if p.name.endswith((".tmp", ".new")) or ".tmp-" in p.name
    )


# --- S1/S2: identity key ------------------------------------------------------------------------------------------

_KEY_REL = Path("kindled_link") / "identity_ed25519.key"


def test_identity_create_and_rotate_write_a_0600_key_and_leave_no_temp(tmp_path):
    from brain.kindled_link.identity import KindledIdentity

    a = KindledIdentity.load_or_create(tmp_path)
    key = tmp_path / _KEY_REL
    assert len(key.read_bytes()) == 32
    new, old = KindledIdentity.rotate(tmp_path)
    assert old.key_id == a.key_id and new.key_id != a.key_id
    assert KindledIdentity.load_or_create(tmp_path).key_id == new.key_id
    assert _leftovers(key.parent) == []
    if os.name != "nt":
        assert stat.S_IMODE(key.stat().st_mode) == 0o600


def test_identity_create_retries_transient_permission_errors(tmp_path, monkeypatch):
    from brain.kindled_link.identity import KindledIdentity

    state = _flaky_replace(monkeypatch, "identity_ed25519.key")
    idn = KindledIdentity.load_or_create(tmp_path)
    assert state["n"] == 0
    assert KindledIdentity.load_or_create(tmp_path).key_id == idn.key_id


def test_identity_rotate_retries_transient_permission_errors(tmp_path, monkeypatch):
    from brain.kindled_link.identity import KindledIdentity

    KindledIdentity.load_or_create(tmp_path)
    state = _flaky_replace(monkeypatch, "identity_ed25519.key")
    new, _old = KindledIdentity.rotate(tmp_path)
    assert state["n"] == 0
    assert KindledIdentity.load_or_create(tmp_path).key_id == new.key_id


def test_identity_key_is_published_through_the_shared_temp_shape(tmp_path, monkeypatch):
    from brain.kindled_link.identity import KindledIdentity

    seen = _record_replace_sources(monkeypatch)
    KindledIdentity.load_or_create(tmp_path)
    assert len(seen) == 1 and TEMP_RE.search(seen[0]), seen


def test_identity_failed_create_leaves_the_key_absent_and_no_temp(tmp_path, monkeypatch):
    from brain.kindled_link.identity import KindledIdentity

    _boom_fsync(monkeypatch)
    with pytest.raises(OSError):
        KindledIdentity.load_or_create(tmp_path)
    assert not (tmp_path / _KEY_REL).exists()
    assert _leftovers((tmp_path / _KEY_REL).parent) == []
    monkeypatch.undo()
    assert KindledIdentity.load_or_create(tmp_path).key_id.startswith("kid_")


def test_identity_failed_rotate_keeps_the_old_key_bytes_and_leaves_no_temp(tmp_path, monkeypatch):
    from brain.kindled_link.identity import KindledIdentity

    KindledIdentity.load_or_create(tmp_path)
    key = tmp_path / _KEY_REL
    before = key.read_bytes()
    _boom_fsync(monkeypatch)
    with pytest.raises(OSError):
        KindledIdentity.rotate(tmp_path)
    assert key.read_bytes() == before
    assert _leftovers(key.parent) == []


# --- S3: judge pointer --------------------------------------------------------------------------------------------


def test_swap_champion_pointer_writes_the_name_and_leaves_no_temp(tmp_path):
    from brain.memory import judge_lora

    root = tmp_path / "relevance_judge"
    judge_lora.swap_champion_pointer(root, root / "adapter-abc")
    assert (root / "current").read_text(encoding="utf-8") == "adapter-abc"
    assert _leftovers(root) == []


def test_swap_champion_pointer_retries_and_uses_the_shared_temp_shape(tmp_path, monkeypatch):
    from brain.memory import judge_lora

    root = tmp_path / "relevance_judge"
    state = _flaky_replace(monkeypatch, "current")
    seen = _record_replace_sources(monkeypatch)
    judge_lora.swap_champion_pointer(root, root / "adapter-abc")
    assert state["n"] == 0
    assert seen and all(TEMP_RE.search(n) for n in seen), seen
    assert (root / "current").read_text(encoding="utf-8") == "adapter-abc"


def test_swap_champion_pointer_failed_write_keeps_the_old_pointer(tmp_path, monkeypatch):
    from brain.memory import judge_lora

    root = tmp_path / "relevance_judge"
    judge_lora.swap_champion_pointer(root, root / "adapter-old")
    _boom_fsync(monkeypatch)
    with pytest.raises(OSError):
        judge_lora.swap_champion_pointer(root, root / "adapter-new")
    assert (root / "current").read_text(encoding="utf-8") == "adapter-old"
    assert _leftovers(root) == []


# --- S4: migrator legacy copy -------------------------------------------------------------------------------------


def _legacy_setup(tmp_path):
    og = tmp_path / "og"
    og.mkdir()
    (og / "nell_journal.json").write_bytes(b'{"entries": [1, 2, 3]}\xff\xfe raw')
    persona = tmp_path / "persona"
    persona.mkdir()
    return og, persona


def test_legacy_copy_is_byte_identical_and_leaves_no_temp(tmp_path):
    from brain.migrator.og_legacy import migrate_legacy_files

    og, persona = _legacy_setup(tmp_path)
    preserved, _missing, issues = migrate_legacy_files(og_data_dir=og, persona_dir=persona)
    assert preserved == ["nell_journal.json"] and issues == []
    dest = persona / "legacy" / "nell_journal.json"
    assert (
        hashlib.sha256(dest.read_bytes()).digest()
        == hashlib.sha256((og / "nell_journal.json").read_bytes()).digest()
    )
    assert _leftovers(dest.parent) == []


def test_legacy_copy_retries_transient_permission_errors(tmp_path, monkeypatch):
    from brain.migrator.og_legacy import migrate_legacy_files

    og, persona = _legacy_setup(tmp_path)
    state = _flaky_replace(monkeypatch, "nell_journal.json")
    seen = _record_replace_sources(monkeypatch)
    preserved, _m, issues = migrate_legacy_files(og_data_dir=og, persona_dir=persona)
    assert state["n"] == 0 and preserved == ["nell_journal.json"] and issues == []
    assert seen and all(TEMP_RE.search(n) for n in seen), seen


# --- S5/S6: image and file blobs ----------------------------------------------------------------------------------

_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63606060600000000400015e36b8c80000000049454e44ae426082"
)


def _save_image(tmp_path):
    from brain.images import save_image_bytes

    return save_image_bytes(tmp_path, _PNG, "image/png")


def _save_file(tmp_path, data=b"some non image file"):
    from brain import file_store

    return file_store.save_file_bytes(tmp_path, data)


@pytest.mark.parametrize("which", ["image", "file"])
def test_blob_names_and_residue_after_success(tmp_path, which):
    rec = _save_image(tmp_path) if which == "image" else _save_file(tmp_path)
    directory = tmp_path / ("images" if which == "image" else "files")
    expected = f"{rec.sha}.png" if which == "image" else rec.sha
    assert [p.name for p in directory.iterdir()] == [expected]


@pytest.mark.parametrize("which", ["image", "file"])
def test_blob_second_save_does_no_write(tmp_path, monkeypatch, which):
    save = _save_image if which == "image" else _save_file
    save(tmp_path)
    calls: list[str] = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda s, d: (calls.append(str(d)), real(s, d))[1])
    save(tmp_path)
    assert calls == []


@pytest.mark.parametrize("which", ["image", "file"])
def test_blob_retries_transient_permission_errors(tmp_path, monkeypatch, which):
    state = _flaky_replace(monkeypatch, "")  # every replace of this save
    save = _save_image if which == "image" else _save_file
    rec = save(tmp_path)
    assert state["n"] == 0
    directory = tmp_path / ("images" if which == "image" else "files")
    assert len(list(directory.iterdir())) == 1 and rec.size_bytes > 0


@pytest.mark.parametrize("which", ["image", "file"])
def test_blob_is_fsynced_before_it_is_published(tmp_path, monkeypatch, which):
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            events.append("fsync")
        return real_fsync(fd)

    def replace(src, dst):
        events.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    (_save_image if which == "image" else _save_file)(tmp_path)
    assert events == ["fsync", "replace"]


@pytest.mark.parametrize("which", ["image", "file"])
def test_blob_failure_with_the_target_absent_raises_and_leaves_no_temp(
    tmp_path, monkeypatch, which
):
    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        (_save_image if which == "image" else _save_file)(tmp_path)
    directory = tmp_path / ("images" if which == "image" else "files")
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize("which", ["image", "file"])
def test_blob_failure_with_the_target_present_means_a_concurrent_writer_won(
    tmp_path, monkeypatch, which
):
    """b557882d: identical concurrent uploads end with one complete file and no error."""
    real = os.replace
    monkeypatch.setattr("time.sleep", lambda _s: None)
    data = _PNG if which == "image" else b"some non image file"

    def competitor_then_denied(src, dst):
        if not Path(dst).exists():
            Path(dst).write_bytes(data)
        raise PermissionError(32, "held open")

    monkeypatch.setattr(os, "replace", competitor_then_denied)
    rec = (_save_image if which == "image" else _save_file)(tmp_path)
    monkeypatch.setattr(os, "replace", real)
    directory = tmp_path / ("images" if which == "image" else "files")
    assert len(list(directory.iterdir())) == 1 and rec.size_bytes == len(data)


@pytest.mark.parametrize("which", ["image", "file"])
def test_blob_threads_saving_the_same_content_end_with_one_file(tmp_path, which):
    save = _save_image if which == "image" else _save_file
    barrier = threading.Barrier(6)
    errors: list[BaseException] = []

    def work():
        try:
            barrier.wait()
            save(tmp_path)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    directory = tmp_path / ("images" if which == "image" else "files")
    assert errors == [] and len(list(directory.iterdir())) == 1


# --- S7: launchd plist --------------------------------------------------------------------------------------------


@pytest.fixture
def launchd_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "data" / "personas" / "nell").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Path.home() on Windows
    monkeypatch.setenv("KINDLED_HOME", str(home / "data"))
    monkeypatch.delenv("NELLBRAIN_HOME", raising=False)
    nell = tmp_path / "nell"
    nell.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nell.chmod(0o755)
    return nell


def test_launchd_plist_is_the_built_xml_mode_0644_and_leaves_no_temp(launchd_env):
    from brain.service import launchd

    path = launchd.write_launchd_plist(persona="nell", nell_path=launchd_env)
    assert path.read_text(encoding="utf-8") == launchd.build_launchd_plist_xml(
        persona="nell", nell_path=launchd_env
    )
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert _leftovers(path.parent) == []


def test_launchd_plist_retries_transient_permission_errors(launchd_env, monkeypatch):
    from brain.service import launchd

    state = _flaky_replace(monkeypatch, ".plist")
    path = launchd.write_launchd_plist(persona="nell", nell_path=launchd_env)
    assert state["n"] == 0 and path.exists()


def test_launchd_failed_write_keeps_the_previous_plist_and_leaves_no_temp(launchd_env, monkeypatch):
    from brain.service import launchd

    path = launchd.write_launchd_plist(persona="nell", nell_path=launchd_env)
    before = path.read_bytes()
    other = launchd_env.parent / "other"
    other.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    other.chmod(0o755)
    _boom_fsync(monkeypatch)
    with pytest.raises(OSError):
        launchd.write_launchd_plist(persona="nell", nell_path=other)
    assert path.read_bytes() == before
    assert _leftovers(path.parent) == []


# --- S8: systemd unit refresh -------------------------------------------------------------------------------------


@pytest.fixture
def unit_env(tmp_path, monkeypatch):
    from brain.service import systemd

    nell = tmp_path / "bin" / "nell"
    nell.parent.mkdir()
    nell.write_text("#!/bin/sh\n")
    nell.chmod(0o755)
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()
    monkeypatch.setattr(systemd, "user_unit_dir", lambda: unit_dir)
    monkeypatch.setattr(systemd, "_user_home", lambda: tmp_path)
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(systemd, "run_systemctl", lambda args: None)
    path = unit_dir / systemd.service_unit_name("nell")
    path.write_text(
        systemd.build_systemd_unit_text(persona="nell", nell_path=nell).replace(
            "KillMode=mixed\n", ""
        ),
        encoding="utf-8",
    )
    return systemd, path


@pytest.mark.skipif(sys.platform == "win32", reason="systemd unit text uses POSIX paths")
def test_unit_refresh_text_is_unchanged_apart_from_the_killmode_line_and_leaves_no_temp(unit_env):
    systemd, path = unit_env
    before = path.read_text(encoding="utf-8")
    assert systemd.refresh_installed_unit("nell") is True
    after = path.read_text(encoding="utf-8")
    assert after == before.replace("TimeoutStopSec=300\n", "TimeoutStopSec=300\nKillMode=mixed\n")
    assert _leftovers(path.parent) == []


@pytest.mark.skipif(sys.platform == "win32", reason="systemd unit text uses POSIX paths")
def test_unit_refresh_retries_and_uses_the_shared_temp_shape(unit_env, monkeypatch):
    systemd, path = unit_env
    state = _flaky_replace(monkeypatch, path.name)
    seen = _record_replace_sources(monkeypatch)
    assert systemd.refresh_installed_unit("nell") is True
    assert state["n"] == 0 and seen and all(TEMP_RE.search(n) for n in seen), seen


@pytest.mark.skipif(sys.platform == "win32", reason="systemd unit text uses POSIX paths")
def test_unit_refresh_failed_write_keeps_the_unit_and_leaves_no_temp(unit_env, monkeypatch):
    systemd, path = unit_env
    before = path.read_bytes()
    _boom_fsync(monkeypatch)
    with pytest.raises(OSError):
        systemd.refresh_installed_unit("nell")
    assert path.read_bytes() == before
    assert _leftovers(path.parent) == []


# --- S9: the directory fsync callers ------------------------------------------------------------------------------


def test_append_archive_new_segment_and_new_marker_fsync_the_directory(tmp_path, monkeypatch):
    from brain.ingest import buffer

    seen: list[Path] = []
    monkeypatch.setattr(buffer, "fsync_dir", lambda d: seen.append(Path(d)))
    buffer.append_archive(
        tmp_path, "sess1", [{"role": "user", "content": "hi", "ts": "2026-06-15T10:00:00+00:00"}]
    )
    archive_dir = buffer._archived_conversations_dir(tmp_path)
    assert seen == [archive_dir]
    seen.clear()
    buffer.append_archive(
        tmp_path, "sess1", [{"role": "user", "content": "again", "ts": "2026-06-15T10:00:01+00:00"}]
    )
    assert seen == []  # the segment already exists: no new directory entry
    marker_dir = buffer._archive_marker_path(tmp_path, "sess1").parent
    marker_dir.mkdir(parents=True, exist_ok=True)
    buffer.write_archive_marker(tmp_path, "sess1", "key-1")
    assert len(seen) >= 1 and all(p == marker_dir for p in seen)

