"""Tests for `nell migrate --json` flag."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def test_json_flag_emits_machine_readable_report(tmp_path: Path, monkeypatch, capsys) -> None:
    # Build a minimal emergence-kit source.
    fixture = tmp_path / "kit"
    fixture.mkdir()
    (fixture / "memories.json").write_text(json.dumps([
        {
            "id": "m1",
            "content": "hello",
            "memory_type": "fact",
            "domain": "test",
            "emotions": {},
            "importance": 5,
            "tags": [],
            "active": True,
            "created_at": "2026-01-01T00:00:00Z",
        }
    ]))

    install_target = tmp_path / "kindled_home"
    monkeypatch.setenv("KINDLED_HOME", str(install_target))

    from brain.cli import main

    monkeypatch.setattr(sys, "argv", [
        "nell", "migrate", "--source", "emergence-kit",
        "--input", str(fixture), "--install-as", "phoebe", "--json",
    ])
    main()

    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["kind"] == "MigrationReport"
    assert payload["memories_migrated"] == 1
    assert payload["source_kind"] == "emergence-kit"
