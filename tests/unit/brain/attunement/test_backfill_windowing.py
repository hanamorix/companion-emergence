"""Tests for backfill.window_buffer — chunks turns into rolling windows."""
from brain.attunement.backfill import window_buffer
from brain.attunement.store import BufferTurn


def test_window_buffer_empty_returns_empty():
    assert window_buffer([]) == []


def test_window_buffer_single_window_when_short():
    turns = [BufferTurn(id=f"t{i}", content=str(i)) for i in range(5)]
    windows = window_buffer(turns, size=20, stride=10)
    assert len(windows) == 1
    assert len(windows[0].turns) == 5
    assert windows[0].id == "window-0"


def test_window_buffer_overlapping_windows():
    turns = [BufferTurn(id=f"t{i}", content=str(i)) for i in range(40)]
    windows = window_buffer(turns, size=20, stride=10)
    # 40 turns: windows at 0-19, 10-29, 20-39 → 3 windows
    assert len(windows) == 3
    assert windows[0].id == "window-0"
    assert windows[1].id == "window-1"
    assert windows[2].id == "window-2"
    assert windows[1].turns[0].id == "t10"
    assert windows[2].turns[0].id == "t20"
    assert windows[2].turns[-1].id == "t39"


def test_window_buffer_exact_size_returns_one_window():
    turns = [BufferTurn(id=f"t{i}", content=str(i)) for i in range(20)]
    windows = window_buffer(turns, size=20, stride=10)
    assert len(windows) == 1
    assert windows[0].turns[0].id == "t0"
    assert windows[0].turns[-1].id == "t19"


def test_window_buffer_handles_partial_final_window():
    turns = [BufferTurn(id=f"t{i}", content=str(i)) for i in range(25)]
    windows = window_buffer(turns, size=20, stride=10)
    # 25 turns: window-0 0-19, window-1 10-24 (partial, size=15)
    assert len(windows) == 2
    assert len(windows[1].turns) == 15
    assert windows[1].turns[-1].id == "t24"
