"""Argparse wiring tests for `nell works`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain import cli, works
from brain.works.storage import write_markdown
from brain.works.store import WorksStore


def _seed(persona_dir: Path) -> str:
    """Insert one sample work; return its id."""
    w = works.Work(
        id="abc123def456",
        title="The Lighthouse",
        type="story",
        created_at=datetime(2026, 5, 4, tzinfo=UTC),
        session_id=None,
        word_count=4,
        summary="Sample story.",
    )
    write_markdown(persona_dir, w, content="A small story body.")
    WorksStore(persona_dir / "data" / "works.db").insert(w, content="A small story body.")
    return w.id


@pytest.fixture
def persona_with_one_work(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    persona_dir = home / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))
    _seed(persona_dir)
    return persona_dir


def test_works_list_prints_seeded_work(
    persona_with_one_work: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["works", "list", "--persona", "nell"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "abc123def456" in out
    assert "The Lighthouse" in out
    assert "story" in out


def test_works_list_filters_by_type(
    persona_with_one_work: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["works", "list", "--persona", "nell", "--type", "code"])
    assert rc == 0
    out = capsys.readouterr().out
    # No code-type works seeded → list should be empty
    assert "abc123def456" not in out


def test_works_list_respects_limit(
    persona_with_one_work: Path,
) -> None:
    rc = cli.main(["works", "list", "--persona", "nell", "--limit", "5"])
    assert rc == 0


def test_works_search_finds_seeded_work(
    persona_with_one_work: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["works", "search", "--persona", "nell", "--query", "lighthouse"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "abc123def456" in out


def test_works_search_returns_empty_on_no_match(
    persona_with_one_work: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["works", "search", "--persona", "nell", "--query", "zzzzzzzzz"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "abc123def456" not in out


def test_works_read_prints_full_content(
    persona_with_one_work: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["works", "read", "--persona", "nell", "--id", "abc123def456"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "A small story body." in out


def test_works_read_unknown_id_returns_1(
    persona_with_one_work: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["works", "read", "--persona", "nell", "--id", "zzzzzzzzzzzz"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown" in err.lower() or "not found" in err.lower()


@pytest.mark.parametrize("action", ["list", "search", "read"])
def test_works_action_requires_persona(action: str, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["works", action]
    if action == "search":
        args += ["--query", "x"]
    if action == "read":
        args += ["--id", "x"]
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2


def test_works_search_requires_query(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["works", "search", "--persona", "nell"])
    assert exc.value.code == 2


def test_works_read_requires_id(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["works", "read", "--persona", "nell"])
    assert exc.value.code == 2
