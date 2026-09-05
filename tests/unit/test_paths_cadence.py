"""#178: persona-scoped cadence state files live under <persona>/cadence/,
resolved through one function that migrates a legacy root-level file."""

from __future__ import annotations

from pathlib import Path

from brain.paths import cadence_state_path


def test_resolves_under_cadence_subdir(tmp_path: Path) -> None:
    p = cadence_state_path(tmp_path, "finalize_cadence.json")
    assert p == tmp_path / "cadence" / "finalize_cadence.json"


def test_migrates_legacy_root_file_once(tmp_path: Path) -> None:
    legacy = tmp_path / "finalize_cadence.json"
    legacy.write_text('{"next_at": "2026-09-04T00:00:00+00:00"}', encoding="utf-8")

    p = cadence_state_path(tmp_path, "finalize_cadence.json")

    assert p.read_text(encoding="utf-8") == '{"next_at": "2026-09-04T00:00:00+00:00"}'
    assert not legacy.exists()


def test_does_not_clobber_existing_new_file_with_legacy(tmp_path: Path) -> None:
    (tmp_path / "cadence").mkdir()
    new = tmp_path / "cadence" / "x.json"
    new.write_text("new", encoding="utf-8")
    legacy = tmp_path / "x.json"
    legacy.write_text("old", encoding="utf-8")

    p = cadence_state_path(tmp_path, "x.json")

    assert p.read_text(encoding="utf-8") == "new"
    assert legacy.exists()  # left for the sidecar sweep / a human, not silently lost
