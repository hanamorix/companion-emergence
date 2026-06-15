"""Notes autonomous-path safety: a note may ONLY land inside its authorized
folder. is_within_authorized must be escape-proof (realpath), like _is_within."""
from brain.files.write_guard import is_within_authorized


def test_path_inside_folder_allowed(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    assert is_within_authorized((folder / "2026-06-15 — hi.md").resolve(), folder) is True


def test_path_outside_folder_refused(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    assert is_within_authorized((tmp_path / "elsewhere.md").resolve(), folder) is False


def test_dotdot_escape_refused(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    sneaky = folder / ".." / "escaped.md"
    assert is_within_authorized(sneaky, folder) is False


def test_symlink_escape_refused(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = folder / "link"
    link.symlink_to(outside)
    assert is_within_authorized(link / "x.md", folder) is False  # realpath lands outside


def test_none_folder_refused(tmp_path):
    assert is_within_authorized(tmp_path / "x.md", None) is False
