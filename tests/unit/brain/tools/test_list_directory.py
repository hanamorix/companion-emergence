"""Tests for brain.tools.impls.list_directory."""
from __future__ import annotations

from brain.tools.impls.list_directory import list_directory


def test_lists_entries_with_types(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    out = list_directory(path=str(tmp_path), persona_dir=tmp_path)
    names = {e["name"]: e["type"] for e in out["entries"]}
    assert names["a.txt"] == "file" and names["sub"] == "dir"


def test_non_dir_clean_error(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    out = list_directory(path=str(f), persona_dir=tmp_path)
    assert "error" in out


def test_writes_audit_line(tmp_path):
    import json
    list_directory(path=str(tmp_path), persona_dir=tmp_path)
    line = json.loads((tmp_path / "file_access.jsonl").read_text().strip().splitlines()[-1])
    assert line["tool"] == "list_directory"


def test_non_dir_writes_failure_audit_line(tmp_path):
    import json
    persona = tmp_path / "persona"
    persona.mkdir()
    f = tmp_path / "a.txt"
    f.write_text("x")
    list_directory(path=str(f), persona_dir=persona)
    line = json.loads((persona / "file_access.jsonl").read_text().strip().splitlines()[-1])
    assert line["tool"] == "list_directory" and line["ok"] is False


def test_through_dispatch(tmp_path):
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.tools import NELL_TOOL_NAMES, dispatch
    from brain.tools.schemas import SCHEMAS

    assert "list_directory" in NELL_TOOL_NAMES and "list_directory" in SCHEMAS
    (tmp_path / "f.txt").write_text("ok")
    out = dispatch(
        "list_directory",
        {"path": str(tmp_path)},
        store=MemoryStore(":memory:"),
        hebbian=HebbianMatrix(":memory:"),
        persona_dir=tmp_path,
    )
    assert "entries" in out
