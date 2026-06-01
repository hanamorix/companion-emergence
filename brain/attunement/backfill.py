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


def _window_tokens(window: Window) -> frozenset[str]:
    """Lowercased word tokens across all turns in the window."""
    tokens: set[str] = set()
    for turn in window.turns:
        for word in turn.content.lower().split():
            tokens.add(word)
    return frozenset(tokens)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


_CLUSTER_THRESHOLD = 0.5


def _cluster_windows(windows: list[Window]) -> list[list[Window]]:
    """Group windows by lexical similarity (jaccard >= 0.5).

    Greedy single-linkage clustering: walk windows in order, assign each
    to the first existing cluster whose representative has jaccard >= 0.5,
    otherwise start a new cluster. Deterministic by input order.
    """
    clusters: list[list[Window]] = []
    fingerprints: list[frozenset[str]] = []
    for w in windows:
        tokens = _window_tokens(w)
        placed = False
        for idx, rep in enumerate(fingerprints):
            if _jaccard(tokens, rep) >= _CLUSTER_THRESHOLD:
                clusters[idx].append(w)
                placed = True
                break
        if not placed:
            clusters.append([w])
            fingerprints.append(tokens)
    return clusters


def select_sample(windows: list[Window], *, rate: float = 0.2) -> list[Window]:
    """Stratified sample with topic-diversity weighting.

    Clusters windows by jaccard token overlap; picks one window per cluster
    before doubling up within any cluster. Returns at least one window when
    input is non-empty (rounds rate * count up to 1 if needed).

    Deterministic: same input → same output. No randomness.
    """
    if not windows:
        return []

    clusters = _cluster_windows(windows)
    target = max(1, round(len(windows) * rate))

    sample: list[Window] = []
    # First pass: one window per cluster (the first one — deterministic by order)
    for cluster in clusters:
        if len(sample) >= target:
            break
        sample.append(cluster[0])

    # Second pass: fill remainder by walking clusters round-robin and picking
    # the next un-picked window from each. Deterministic order.
    if len(sample) < target:
        picked = {id(w) for w in sample}
        round_idx = 1
        while len(sample) < target:
            added = False
            for cluster in clusters:
                if round_idx < len(cluster) and id(cluster[round_idx]) not in picked:
                    sample.append(cluster[round_idx])
                    picked.add(id(cluster[round_idx]))
                    added = True
                    if len(sample) >= target:
                        break
            if not added:
                break  # exhausted all clusters
            round_idx += 1

    return sample


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
