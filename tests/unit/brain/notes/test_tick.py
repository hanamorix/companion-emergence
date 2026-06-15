from datetime import UTC, datetime, timedelta

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
