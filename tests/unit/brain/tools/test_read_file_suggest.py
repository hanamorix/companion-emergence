from brain.tools.impls.read_file import read_file


def test_miss_suggests_real_filename(tmp_path):
    (tmp_path / "Phoebes_notes.md").write_text("hi", encoding="utf-8")
    (tmp_path / "other.txt").write_text("x", encoding="utf-8")
    out = read_file(str(tmp_path / "phoebe_notes.md"), persona_dir=tmp_path / "p")
    assert "error" in out
    assert "Phoebes_notes.md" in out.get("did_you_mean", [])


def test_miss_no_suggestions_when_dir_absent(tmp_path):
    out = read_file(str(tmp_path / "nope" / "x.md"), persona_dir=tmp_path / "p")
    assert "error" in out
    assert out.get("did_you_mean", []) == []
