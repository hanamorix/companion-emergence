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

# The base-code check runs on Linux (ext4: case- and normalisation-sensitive), where these skip.
# On macOS APFS each one fails against main (#607); the macOS and Windows jobs run them on the head.
_VOLUME = pytest.mark.no_base_failure(
    reason="needs a case- or normalisation-insensitive volume, so it skips on the Linux base check; fails on main on macOS APFS (#607)"
)
_REGRESSION = pytest.mark.no_base_failure(
    reason="regression guard: pins behaviour main already has (refusal in the real case, Documents allowed, no casefolding on a case-sensitive volume) against the #607 rewrite"
)


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


@_VOLUME
@pytest.mark.parametrize("rel", [".SSH/authorized_keys", ".Ssh/x", ".ssh/../.SSH/config"])
def test_case_variant_of_a_denied_home_dir_is_refused(home, ci, tmp_path, rel):
    r = check_write_target(str(home / rel), op="create", persona_dir=tmp_path / "persona")
    assert not r.ok and "denied" in r.error.lower()


@_VOLUME
def test_case_changed_persona_dir_is_refused(home, ci, tmp_path):
    persona = tmp_path / "persona"
    persona.mkdir()
    r = check_write_target(str(tmp_path / "PERSONA" / "f.txt"), op="create", persona_dir=persona)
    assert not r.ok and "denied" in r.error.lower()


@_VOLUME
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


@_VOLUME
def test_case_variant_of_a_denied_system_root_is_refused(home, tmp_path):
    if not Path("/Library").is_dir() or not Path("/library").exists():
        pytest.skip("needs macOS's /Library on a case-insensitive volume")
    r = check_write_target("/library/x-607", op="create", persona_dir=tmp_path / "persona")
    assert not r.ok and "denied" in r.error.lower()


@_VOLUME
def test_unicode_normalisation_variant_of_the_persona_dir_is_refused(home, tmp_path):
    nfc, nfd = unicodedata.normalize("NFC", "café"), unicodedata.normalize("NFD", "café")
    persona = tmp_path / nfc
    persona.mkdir()
    if not (tmp_path / nfd).exists():
        pytest.skip("needs a normalisation-insensitive volume (APFS)")
    r = check_write_target(str(tmp_path / nfd / "f.txt"), op="create", persona_dir=persona)
    assert not r.ok and "denied" in r.error.lower()


@_REGRESSION
def test_real_case_is_still_refused_and_documents_still_allowed(home, tmp_path):
    persona = tmp_path / "persona"
    persona.mkdir()
    assert not check_write_target(str(home / ".ssh" / "authorized_keys"), op="create", persona_dir=persona).ok
    assert not check_write_target(str(persona / "f.txt"), op="create", persona_dir=persona).ok
    r = check_write_target(str(home / "Documents" / "note.md"), op="create", persona_dir=persona)
    assert r.ok and r.resolved == (home / "Documents" / "note.md").resolve()


@_VOLUME
def test_authorized_folder_given_in_another_case_still_matches(ci, tmp_path):
    folder = tmp_path / "Notes"
    folder.mkdir()
    assert is_within_authorized((folder / "n.md").resolve(), tmp_path / "NOTES") is True


@_REGRESSION
def test_authorized_check_does_not_casefold_on_a_case_sensitive_volume(tmp_path):
    if _case_insensitive(tmp_path):
        pytest.skip("needs a case-sensitive volume, where Notes and notes are two folders")
    (tmp_path / "Notes").mkdir()
    (tmp_path / "notes").mkdir()
    assert is_within_authorized(tmp_path / "notes" / "n.md", tmp_path / "Notes") is False


def test_authorized_folder_that_does_not_exist_authorizes_nothing(tmp_path):
    assert is_within_authorized(tmp_path / "gone" / "n.md", tmp_path / "gone") is False


@_VOLUME
def test_missing_denied_dir_under_a_normalisation_variant_of_home_is_refused(tmp_path, monkeypatch):
    # Neither half alone covers this: ~/.aws doesn't exist (nothing to compare by identity) and the
    # home is spelled in the other Unicode normalisation (the whole-path spelling differs too).
    nfc, nfd = unicodedata.normalize("NFC", "café"), unicodedata.normalize("NFD", "café")
    home = tmp_path / nfc / "home"
    home.mkdir(parents=True)
    if not (tmp_path / nfd).exists():
        pytest.skip("needs a normalisation-insensitive volume (APFS)")
    monkeypatch.setattr(Path, "home", lambda: home)
    for rel in (".aws/credentials", ".AWS/credentials"):
        r = check_write_target(str(tmp_path / nfd / "home" / rel), op="create", persona_dir=tmp_path / "p")
        assert not r.ok and "denied" in r.error.lower(), rel


def test_a_deleted_notes_folder_gets_a_consent_card_not_an_auto_commit(tmp_path, monkeypatch):
    # The allow-list matches by identity, so a notes folder the user deleted after enabling notes
    # authorizes nothing: propose_write queues a card instead of silently recreating the folder.
    import json
    from datetime import UTC, datetime

    from brain.files.pending import list_pending
    from brain.tools.impls.propose_write import propose_write

    h = tmp_path / "home"
    (h / "Documents").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: h)
    notes = h / "Documents" / "Canary Notes"  # enabled, then deleted: never created here
    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / "persona_config.json").write_text(
        json.dumps({"notes_enabled": True, "notes_folder": str(notes)}), encoding="utf-8"
    )
    out = propose_write(path=str(notes / "today.md"), content="a thought", op="create", persona_dir=persona)
    assert out.get("status") != "written", out
    assert not notes.exists()
    assert len(list_pending(persona, now=datetime.now(UTC))) == 1


@_REGRESSION
def test_shared_identity_helper_does_not_swallow_a_nul_path(tmp_path):
    # The ancestor walk moved into path_identity; main's read guard raised on a NUL path (fail closed)
    # and must not start answering "not denied" for one.
    from brain.files.read_guard import is_read_denied

    persona = tmp_path / "persona"
    persona.mkdir()
    with pytest.raises(ValueError):
        is_read_denied(tmp_path / "a\x00b", persona_dir=persona)


@pytest.mark.parametrize("seg", [".aws.", ".aws ", ".AWS. .", ".aws..."])
def test_windows_trailing_dots_and_spaces_on_a_missing_denied_dir_are_refused(home, tmp_path, monkeypatch, seg):
    # Win32 drops trailing dots and spaces from each segment, so commit's mkdir of ``.aws.`` creates ``.aws``.
    # Through the platform seam, so it runs (and bites on main) on every OS.
    import brain.files.write_guard as wg

    monkeypatch.setattr(wg, "_is_windows", lambda: True)
    r = check_write_target(str(home / seg / "credentials"), op="create", persona_dir=tmp_path / "persona")
    assert not r.ok and "denied" in r.error.lower()


@pytest.mark.parametrize("seg", [".aws::$INDEX_ALLOCATION", "notes.md:hidden"])
def test_windows_stream_suffix_is_refused(home, tmp_path, monkeypatch, seg):
    # An NTFS stream name (``dir::$INDEX_ALLOCATION`` creates the directory) can't be compared by
    # spelling or identity, and Win32 refuses ``:`` in an ordinary name anyway, so refuse it outright.
    import brain.files.write_guard as wg

    monkeypatch.setattr(wg, "_is_windows", lambda: True)
    r = check_write_target(str(home / "Documents" / seg / "x"), op="create", persona_dir=tmp_path / "persona")
    assert not r.ok


@pytest.mark.parametrize("rel", [".CONFIG/GH/hosts.yml", ".config/GH/hosts.yml", "library/launchagents/x.plist"])
def test_multi_component_missing_denied_path_is_refused(home, tmp_path, rel):
    # ~/.config/gh and ~/Library/LaunchAgents don't exist here: the anchor is home and the tail is two parts.
    r = check_write_target(str(home / rel), op="create", persona_dir=tmp_path / "persona")
    assert not r.ok and "denied" in r.error.lower(), rel


@_REGRESSION
@pytest.mark.parametrize("rel", [".config/ghx/f", ".config/g/f", ".awsx/f", ".ssh-notes/f", "Documents/.ssh.txt"])
def test_tail_match_does_not_refuse_a_neighbour(home, tmp_path, rel):
    r = check_write_target(str(home / rel), op="create", persona_dir=tmp_path / "persona")
    assert r.ok, (rel, r.error)
