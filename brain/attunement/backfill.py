"""One-time backfill migration over historical chat buffer.

Chunks the buffer into rolling windows, applies a stratified sample (Task 15),
runs the detector on each sampled window (Task 16), and emits a single
backfill_complete event (Task 17).

This file currently exposes only the windowing primitives; subsequent tasks
add the sampling, state-resume, trigger, and orchestration logic.
"""
from __future__ import annotations

from dataclasses import dataclass

from brain.attunement.store import BufferTurn


@dataclass(frozen=True)
class Window:
    id: str           # stable cursor ("window-0", "window-1", ...)
    turns: tuple[BufferTurn, ...]  # immutable view of the window's turns


def window_buffer(
    turns: list[BufferTurn], *, size: int = 20, stride: int = 10
) -> list[Window]:
    """Chunk `turns` into rolling windows of `size` turns with `stride` offset.

    For an empty input, returns []. For a list shorter than `size`,
    returns a single window with all the turns. Otherwise, slides a
    window of `size` along the list in steps of `stride`, stopping when
    the next window would start past the end of the list.
    """
    if not turns:
        return []
    windows: list[Window] = []
    i = 0
    while i < len(turns):
        end = min(i + size, len(turns))
        windows.append(Window(id=f"window-{len(windows)}", turns=tuple(turns[i:end])))
        if end == len(turns):
            break
        i += stride
    return windows
