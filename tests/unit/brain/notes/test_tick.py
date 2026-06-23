import logging
from datetime import UTC, datetime, timedelta

from brain.bridge import cli_throttle
from brain.notes import run_notes_tick


class _Cfg:
    def __init__(self, enabled, folder):
        self.notes_enabled = enabled
        self.notes_folder = folder


def test_tick_noop_when_disabled(tmp_path):
    made = []
    run_notes_tick(tmp_path, config=_Cfg(False, None), provider=None,
                   silence_hours=100.0, make_fn=lambda **k: made.append(1), now=datetime.now(UTC))
    assert made == []


def test_tick_noop_when_not_away_enough(tmp_path):
    made = []
    run_notes_tick(tmp_path, config=_Cfg(True, str(tmp_path)), provider=None,
                   silence_hours=3.0, make_fn=lambda **k: made.append(1),
                   now=datetime.now(UTC), away_hours=12.0)
    assert made == []


def test_tick_fires_when_away_and_enabled(tmp_path):
    folder = tmp_path / "Notes"
    folder.mkdir()
    made = []
    run_notes_tick(tmp_path, config=_Cfg(True, str(folder)), provider=None,
                   silence_hours=20.0, make_fn=lambda **k: made.append(1),
                   now=datetime.now(UTC), away_hours=12.0, cooldown_hours=24.0, daily_cap=1)
    assert made == [1]


def test_tick_throttle_unavailable_spends_no_budget_or_cooldown(tmp_path, caplog):
    # A throttle defer must cost nothing: no make_fn, no budget spend, cooldown
    # NOT advanced, no ERROR log (Windows v0.0.38 report — same shape as maker).
    from brain.notes.state import load_notes_state
    folder = tmp_path / "Notes"
    folder.mkdir()
    made = []
    now = datetime(2026, 6, 15, 12, tzinfo=UTC)
    cli_throttle.mark_interactive_active()  # chat "recent" → slot unavailable
    with caplog.at_level(logging.ERROR):
        run_notes_tick(tmp_path, config=_Cfg(True, str(folder)), provider=None,
                       silence_hours=20.0, make_fn=lambda **k: made.append(1),
                       now=now, away_hours=12.0, cooldown_hours=24.0, daily_cap=1)
    assert made == []  # note not attempted
    assert load_notes_state(tmp_path).last_note_at is None  # cooldown NOT advanced
    # budget untouched → still available today
    from brain.notes.state import consume_budget
    assert consume_budget(tmp_path, now=now, cap=1) is True
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)  # no crash log


def test_tick_respects_cooldown(tmp_path):
    from brain.notes.state import NotesState, save_notes_state
    folder = tmp_path / "Notes"
    folder.mkdir()
    now = datetime(2026, 6, 15, 12, tzinfo=UTC)
    save_notes_state(tmp_path, NotesState(last_note_at=(now - timedelta(hours=2)).isoformat()))
    made = []
    run_notes_tick(tmp_path, config=_Cfg(True, str(folder)), provider=None,
                   silence_hours=20.0, make_fn=lambda **k: made.append(1),
                   now=now, away_hours=12.0, cooldown_hours=24.0, daily_cap=5)
    assert made == []  # cooldown blocks
