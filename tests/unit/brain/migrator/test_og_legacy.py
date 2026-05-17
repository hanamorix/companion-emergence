"""Tests for brain.migrator.og_legacy — verbatim preservation of biographical OG files."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.migrator.og_legacy import LEGACY_FILES, migrate_legacy_files


def test_legacy_copies_present_files(tmp_path: Path) -> None:
    """OG with a subset of LEGACY_FILES → those land in persona_dir/legacy/."""
    og_data = tmp_path / "og_data"
    og_data.mkdir()
    (og_data / "nell_journal.json").write_text('{"entries": []}')
    (og_data / "nell_gifts.json").write_text('{"gifts": []}')
    (og_data / "nell_personality.json").write_text('{"version": "1.0"}')

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    preserved, missing, integrity_issues = migrate_legacy_files(
        og_data_dir=og_data, persona_dir=persona_dir
    )
    assert set(preserved) == {"nell_journal.json", "nell_gifts.json", "nell_personality.json"}
    assert "nell_journal.json" not in missing
    assert "nell_outbox.json" in missing  # in LEGACY_FILES, not seeded → reported missing
    assert integrity_issues == []  # clean copies, no byte mismatch
    assert (persona_dir / "legacy" / "nell_journal.json").read_text() == '{"entries": []}'
    assert (persona_dir / "legacy" / "nell_gifts.json").read_text() == '{"gifts": []}'
    assert (persona_dir / "legacy" / "nell_personality.json").read_text() == '{"version": "1.0"}'


def test_legacy_handles_jsonl_files(tmp_path: Path) -> None:
    """Non-JSON content (behavioral_log.jsonl) preserves byte-for-byte."""
    og_data = tmp_path / "og_data"
    og_data.mkdir()
    raw_jsonl = b'{"event": "a"}\n{"event": "b"}\n'
    (og_data / "behavioral_log.jsonl").write_bytes(raw_jsonl)

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    preserved, _, _ = migrate_legacy_files(og_data_dir=og_data, persona_dir=persona_dir)
    assert "behavioral_log.jsonl" in preserved
    assert (persona_dir / "legacy" / "behavioral_log.jsonl").read_bytes() == raw_jsonl


def test_legacy_overwrites_on_rerun(tmp_path: Path) -> None:
    """Calling twice produces the same final state with the latest content; no error."""
    og_data = tmp_path / "og_data"
    og_data.mkdir()
    (og_data / "nell_journal.json").write_text("v1")
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    migrate_legacy_files(og_data_dir=og_data, persona_dir=persona_dir)
    assert (persona_dir / "legacy" / "nell_journal.json").read_text() == "v1"

    (og_data / "nell_journal.json").write_text("v2")
    preserved, _, _ = migrate_legacy_files(og_data_dir=og_data, persona_dir=persona_dir)
    assert "nell_journal.json" in preserved
    assert (persona_dir / "legacy" / "nell_journal.json").read_text() == "v2"


def test_legacy_empty_og_returns_all_missing(tmp_path: Path) -> None:
    """Fresh OG dir with none of the legacy files → all missing, none preserved.

    The legacy/ subdir is still created (empty) so subsequent file-listing
    assertions are stable.
    """
    og_data = tmp_path / "og_data"
    og_data.mkdir()
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    preserved, missing, integrity_issues = migrate_legacy_files(
        og_data_dir=og_data, persona_dir=persona_dir
    )
    assert preserved == []
    assert len(missing) == len(LEGACY_FILES)
    assert integrity_issues == []
    assert (persona_dir / "legacy").exists()
    assert (persona_dir / "legacy").is_dir()


def test_legacy_integrity_diagnostic_surfaces_byte_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the dest file ends up byte-mismatched from the source (truncated
    write, FS short-read, etc.), surface it as a LegacyIntegrityIssue.
    Non-fatal: the design intent is to preserve broken bytes; the diagnostic
    just tells the operator the COPY itself was lossy."""
    og_data = tmp_path / "og_data"
    og_data.mkdir()
    (og_data / "nell_journal.json").write_text("genuine source bytes")
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    # Force a corrupted write by patching write_bytes on the destination
    # path (the "new" temp file before os.replace) to truncate.
    real_write_bytes = Path.write_bytes

    def truncating_write_bytes(self, data, *a, **kw):
        if self.name == "nell_journal.json.new":
            return real_write_bytes(self, data[:5], *a, **kw)  # truncate
        return real_write_bytes(self, data, *a, **kw)

    monkeypatch.setattr(Path, "write_bytes", truncating_write_bytes)

    preserved, _, integrity_issues = migrate_legacy_files(
        og_data_dir=og_data, persona_dir=persona_dir
    )
    monkeypatch.undo()

    assert "nell_journal.json" in preserved  # still preserved (dest exists)
    assert len(integrity_issues) == 1
    issue = integrity_issues[0]
    assert issue.name == "nell_journal.json"
    assert issue.src_size == 20
    assert issue.dest_size == 5
    assert issue.src_sha256 != issue.dest_sha256
