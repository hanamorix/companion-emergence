"""#607: the write guard compares directories by identity, not by spelling.

On a case-insensitive volume (APFS default, NTFS) ``realpath`` keeps the case the
caller typed, so a string compare let ``~/.SSH/...`` or a case-changed persona dir
past the deny list while the write still landed inside it. The same applies to a
Unicode normalisation variant on APFS. Driven through ``check_write_target`` and
``is_within_authorized`` with a temp HOME; no mocked state files.
"""
import unicodedata
from pathlib import Path

import pytest

from brain.files.write_guard import check_write_target, is_within_authorized


def _case_insensitive(d: Path) -> bool:
    probe = d / "CaseProbe607"
    probe.mkdir()
    try:
        return (d / "caseprobe607").exists()
    finally:
        probe.rmdir()


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "home"
    (h / ".ssh").mkdir(parents=True)
    (h / "Documents").mkdir()
    monkeypatch.setattr(Path, "home", lambda: h)
    return h


@pytest.fixture
def ci(tmp_path):
    if not _case_insensitive(tmp_path):
        pytest.skip("needs a case-insensitive volume: the bypass only exists there")


@pytest.mark.parametrize("rel", [".SSH/authorized_keys", ".Ssh/x", ".ssh/../.SSH/config"])
def test_case_variant_of_a_denied_home_dir_is_refused(home, ci, tmp_path, rel):
    r = check_write_target(str(home / rel), op="create", persona_dir=tmp_path / "persona")
    assert not r.ok and "denied" in r.error.lower()


def test_case_changed_persona_dir_is_refused(home, ci, tmp_path):
    persona = tmp_path / "persona"
    persona.mkdir()
    r = check_write_target(str(tmp_path / "PERSONA" / "f.txt"), op="create", persona_dir=persona)
    assert not r.ok and "denied" in r.error.lower()


def test_persona_dir_passed_in_another_case_still_guards_the_real_one(home, ci, tmp_path):
    persona = tmp_path / "persona"
    persona.mkdir()
    r = check_write_target(str(persona / "f.txt"), op="create", persona_dir=tmp_path / "Persona")
    assert not r.ok and "denied" in r.error.lower()


def test_case_variant_of_a_denied_dir_that_does_not_exist_yet_is_refused(home, tmp_path):
    # commit mkdirs parents, so ~/.AWS/credentials would create the dir the AWS CLI later reads as ~/.aws.
    # Nothing exists to compare by identity, so the name rule has to hold in any case.
    assert not (home / ".aws").exists()
    r = check_write_target(str(home / ".AWS" / "credentials"), op="create", persona_dir=tmp_path / "persona")
    assert not r.ok and "denied" in r.error.lower()


def test_case_variant_of_a_denied_system_root_is_refused(home, tmp_path):
    if not Path("/Library").is_dir() or not Path("/library").exists():
        pytest.skip("needs macOS's /Library on a case-insensitive volume")
    r = check_write_target("/library/x-607", op="create", persona_dir=tmp_path / "persona")
    assert not r.ok and "denied" in r.error.lower()


def test_unicode_normalisation_variant_of_the_persona_dir_is_refused(home, tmp_path):
    nfc, nfd = unicodedata.normalize("NFC", "café"), unicodedata.normalize("NFD", "café")
    persona = tmp_path / nfc
    persona.mkdir()
    if not (tmp_path / nfd).exists():
        pytest.skip("needs a normalisation-insensitive volume (APFS)")
    r = check_write_target(str(tmp_path / nfd / "f.txt"), op="create", persona_dir=persona)
    assert not r.ok and "denied" in r.error.lower()


def test_real_case_is_still_refused_and_documents_still_allowed(home, tmp_path):
    persona = tmp_path / "persona"
    persona.mkdir()
    assert not check_write_target(str(home / ".ssh" / "authorized_keys"), op="create", persona_dir=persona).ok
    assert not check_write_target(str(persona / "f.txt"), op="create", persona_dir=persona).ok
    r = check_write_target(str(home / "Documents" / "note.md"), op="create", persona_dir=persona)
    assert r.ok and r.resolved == (home / "Documents" / "note.md").resolve()


def test_authorized_folder_given_in_another_case_still_matches(ci, tmp_path):
    folder = tmp_path / "Notes"
    folder.mkdir()
    assert is_within_authorized((folder / "n.md").resolve(), tmp_path / "NOTES") is True


def test_authorized_check_does_not_casefold_on_a_case_sensitive_volume(tmp_path):
    if _case_insensitive(tmp_path):
        pytest.skip("needs a case-sensitive volume, where Notes and notes are two folders")
    (tmp_path / "Notes").mkdir()
    (tmp_path / "notes").mkdir()
    assert is_within_authorized(tmp_path / "notes" / "n.md", tmp_path / "Notes") is False


def test_authorized_folder_that_does_not_exist_authorizes_nothing(tmp_path):
    assert is_within_authorized(tmp_path / "gone" / "n.md", tmp_path / "gone") is False
