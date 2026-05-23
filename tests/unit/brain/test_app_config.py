import json

from brain import app_config


def test_write_if_missing_creates_file(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    app_config.write_if_missing("phoebe")
    cfg = json.loads((tmp_path / "app_config.json").read_text())
    assert cfg["selected_persona"] == "phoebe"
    assert cfg["always_on_top"] is False
    assert cfg["reduced_motion"] is False


def test_write_if_missing_no_op_when_already_set(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    (tmp_path / "app_config.json").write_text(json.dumps({
        "selected_persona": "nell", "always_on_top": True, "reduced_motion": False,
    }))
    app_config.write_if_missing("phoebe")
    cfg = json.loads((tmp_path / "app_config.json").read_text())
    assert cfg["selected_persona"] == "nell"
    assert cfg["always_on_top"] is True


def test_write_if_missing_no_op_when_existing_has_null_selection(tmp_path, monkeypatch):
    """Per spec §Open Questions: only write when file is missing entirely.
    Existing file with null selection is handled by boot-side autodetect."""
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    (tmp_path / "app_config.json").write_text(json.dumps({
        "selected_persona": None, "always_on_top": False, "reduced_motion": False,
    }))
    app_config.write_if_missing("phoebe")
    cfg = json.loads((tmp_path / "app_config.json").read_text())
    assert cfg["selected_persona"] is None
