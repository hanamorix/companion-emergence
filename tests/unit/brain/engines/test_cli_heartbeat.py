"""Tests for the `nell heartbeat` CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def nell_persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "persona_root"
    root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(root))

    from brain.paths import get_persona_dir

    persona = get_persona_dir("nell")
    persona.mkdir(parents=True)

    store = MemoryStore(db_path=persona / "memories.db")
    seed = Memory.create_new(
        content="seed for heartbeat",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0},
    )
    seed.importance = 8.0
    store.create(seed)
    store.close()

    h = HebbianMatrix(db_path=persona / "hebbian.db")
    h.close()
    return persona


def test_nell_heartbeat_first_tick_initializes(
    nell_persona: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """First `nell heartbeat --trigger open` creates state + log; defers work."""
    from brain.cli import main

    rc = main(["heartbeat", "--persona", "nell", "--trigger", "open", "--provider", "fake"])
    assert rc == 0
    assert (nell_persona / "heartbeat_state.json").exists()
    assert (nell_persona / "heartbeats.log.jsonl").exists()
    log_line = (nell_persona / "heartbeats.log.jsonl").read_text().strip()
    assert json.loads(log_line)["initialized"] is True


def test_nell_heartbeat_second_tick_does_work(nell_persona: Path) -> None:
    """Second invocation does real work and writes updated state."""
    from brain.cli import main

    main(["heartbeat", "--persona", "nell", "--trigger", "open", "--provider", "fake"])  # init
    rc = main(["heartbeat", "--persona", "nell", "--trigger", "close", "--provider", "fake"])
    assert rc == 0

    state = json.loads((nell_persona / "heartbeat_state.json").read_text())
    assert state["tick_count"] == 1
    assert state["last_trigger"] == "close"


def test_nell_heartbeat_dry_run_no_writes(nell_persona: Path) -> None:
    """--dry-run doesn't create state file or log entry."""
    from brain.cli import main

    rc = main(
        ["heartbeat", "--persona", "nell", "--trigger", "manual", "--provider", "fake", "--dry-run"]
    )
    assert rc == 0
    assert not (nell_persona / "heartbeat_state.json").exists()
    assert not (nell_persona / "heartbeats.log.jsonl").exists()


def test_nell_heartbeat_unknown_persona_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing persona dir → FileNotFoundError."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path / "empty"))
    from brain.cli import main

    with pytest.raises(FileNotFoundError, match="persona"):
        main(
            [
                "heartbeat",
                "--trigger",
                "manual",
                "--persona",
                "ghost",
                "--provider",
                "fake",
            ]
        )


def test_nell_heartbeat_unknown_trigger_rejected(nell_persona: Path) -> None:
    """Argparse rejects --trigger values outside the enum."""
    from brain.cli import main

    with pytest.raises(SystemExit):
        main(["heartbeat", "--persona", "nell", "--trigger", "frobnicate", "--provider", "fake"])


def test_nell_heartbeat_compact_output_suppresses_not_due(
    nell_persona: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default output suppresses 'dream gated: not_due' and similar non-events."""
    from brain.cli import main

    main(["heartbeat", "--persona", "nell", "--trigger", "open", "--provider", "fake"])  # init
    main(["heartbeat", "--persona", "nell", "--trigger", "close", "--provider", "fake"])
    out = capsys.readouterr().out

    # The active tick should NOT print "dream gated: not_due" by default
    assert "dream gated: not_due" not in out
    # And should NOT print "research gated: reflex_won_tie" or "research gated: not_due"
    assert "research gated: not_due" not in out
    assert "research gated: reflex_won_tie" not in out
    # And should NOT print "interests bumped: 0"
    assert "interests bumped: 0" not in out
    # But the basic heartbeat lines should still be there
    assert "Heartbeat tick complete" in out
    assert "decayed:" in out


def test_nell_heartbeat_verbose_shows_all_gated_reasons(
    nell_persona: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--verbose flag re-enables non-event lines (dream gated: not_due, etc)."""
    from brain.cli import main

    main(["heartbeat", "--persona", "nell", "--trigger", "open", "--provider", "fake"])  # init
    main(
        [
            "heartbeat",
            "--persona",
            "nell",
            "--trigger",
            "close",
            "--provider",
            "fake",
            "--verbose",
        ]
    )
    out = capsys.readouterr().out

    # Verbose mode shows dream gated even for not_due
    assert "dream gated:" in out
    # Verbose mode shows interests bumped: 0
    assert "interests bumped: 0" in out
