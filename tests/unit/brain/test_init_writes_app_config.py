import json
from pathlib import Path

from brain import cli


def _run_init(monkeypatch, tmp_path: Path, argv: list[str]) -> int:
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    return cli.main(["init", *argv])


_FRESH_FLAGS = [
    "--persona", "phoebe",
    "--user-name", "",
    "--fresh",
    "--voice-template", "skip",
    "--model", "sonnet",
]


def test_init_writes_app_config_when_missing(tmp_path, monkeypatch):
    rc = _run_init(monkeypatch, tmp_path, _FRESH_FLAGS)
    assert rc == 0
    cfg = json.loads((tmp_path / "app_config.json").read_text())
    assert cfg["selected_persona"] == "phoebe"


def test_init_leaves_existing_app_config_alone(tmp_path, monkeypatch):
    (tmp_path / "app_config.json").write_text(json.dumps({
        "selected_persona": "nell", "always_on_top": False, "reduced_motion": False,
    }))
    rc = _run_init(monkeypatch, tmp_path, _FRESH_FLAGS)
    assert rc == 0
    cfg = json.loads((tmp_path / "app_config.json").read_text())
    assert cfg["selected_persona"] == "nell"
