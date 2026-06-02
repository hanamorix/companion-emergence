"""One-time backfill migration over historical chat buffer.

Chunks the buffer into rolling windows, applies a stratified sample (Task 15),
runs the detector on each sampled window (Task 16), and emits a single
backfill_complete event (Task 17).

This file currently exposes only the windowing primitives; subsequent tasks
add the sampling, state-resume, trigger, and orchestration logic.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from datetime import UTC
from datetime import datetime as _datetime
from pathlib import Path

from brain.attunement.store import BufferTurn
from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

_log = logging.getLogger(__name__)
_STATE_FILE = "backfill_state.json"


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


# ---------------------------------------------------------------------------
# run_backfill orchestrator (Task 16)
# ---------------------------------------------------------------------------

def _state_path(persona_dir: Path) -> Path:
    return persona_dir / "attunement" / _STATE_FILE


def _now_iso_str(now: _datetime) -> str:
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_state(persona_dir: Path):  # -> BackfillState | None
    from brain.attunement.schemas import BackfillState

    p = _state_path(persona_dir)
    if not p.exists():
        return None
    try:
        return BackfillState(**json.loads(p.read_text()))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        _log.warning("attunement backfill: corrupt state file: %s", exc)
        return None


def _save_state(persona_dir: Path, state) -> None:  # state: BackfillState
    p = _state_path(persona_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))
    tmp.replace(p)


def _read_buffer_turns(persona_dir: Path) -> list[BufferTurn]:
    """Collect all user-role turns from active_conversations/*.jsonl, sorted by ts."""
    convs = persona_dir / "active_conversations"
    if not convs.exists():
        return []
    rows: list[tuple[str, BufferTurn]] = []
    for jsonl_file in sorted(convs.glob("*.jsonl")):
        for row in read_jsonl_skipping_corrupt(jsonl_file):
            if row.get("role") != "user":
                continue
            tid = str(row.get("id") or row.get("ts") or "")
            content = str(row.get("content") or "")
            ts = str(row.get("ts") or "")
            if tid and content:
                rows.append((ts, BufferTurn(id=tid, content=content)))
    rows.sort(key=lambda r: r[0])
    return [t for _, t in rows]


def _cursor_index(windows: list[Window], cursor: str) -> int:
    """Return the index AFTER the cursor window. 0 if cursor empty or not found."""
    if not cursor:
        return 0
    for i, w in enumerate(windows):
        if w.id == cursor:
            return i + 1
    return 0


_MIN_TURNS_FOR_BACKFILL = 10

# Categories added in schema v0.0.28-alpha.2 that need a supplementary bootstrap
# when an existing persona's backfill completed at an older schema version.
# tone + cadence are already learned and must NOT be double-counted.
_NEW_CATEGORIES: frozenset[str] = frozenset({"topic_affinity", "response_shape", "relational"})


def should_run_backfill(persona_dir: Path) -> bool:
    """Detect whether backfill should run at startup.

    Returns True iff: persona has >= 10 inbound user turns AND no
    completed backfill_state.json exists (missing OR status != 'complete').
    """
    turns = _read_buffer_turns(persona_dir)
    if len(turns) < _MIN_TURNS_FOR_BACKFILL:
        return False
    existing = _load_state(persona_dir)
    if existing is not None and existing.status == "complete":
        return False
    return True


def should_run_supplementary_backfill(persona_dir: Path) -> bool:
    """True iff a prior backfill completed at an OLDER schema version.

    The new category dimensions (topic_affinity, response_shape, relational)
    need a supplementary bootstrap from history.  Returns False when there is
    no prior completed state or when the completed state is already at the
    current SCHEMA_VERSION.
    """
    from brain.attunement.schemas import SCHEMA_VERSION

    existing = _load_state(persona_dir)
    if existing is None or existing.status != "complete":
        return False
    return existing.schema_version != SCHEMA_VERSION


def run_backfill(
    persona_dir: Path,
    *,
    detector_fn=None,
    now_dt: _datetime | None = None,
    cap: int | None = None,
    only_categories: frozenset[str] | None = None,
    supplementary: bool = False,
):
    """Run (or resume) the one-time backfill migration.

    Reads existing backfill_state.json if present; resumes from last_cursor.
    Halts at daily cap with status='deferred_to_next_day'.
    Returns the final BackfillState.

    When *supplementary=True* the existing completed state is bypassed and a
    fresh pass is started from the beginning (used by run_supplementary_backfill
    to bootstrap new category dimensions after a schema upgrade).

    When *only_categories* is set and *detector_fn* is None, the default
    detector is wrapped to restrict extraction to those categories so existing
    tone/cadence patterns are not double-counted.
    """
    from brain.attunement.budget import consume_call as _attunement_consume_call
    from brain.attunement.crystallise import check_crystallisations
    from brain.attunement.schemas import (
        DAILY_BUDGET_DEFAULT,
        SCHEMA_VERSION,
        BackfillState,
    )
    from brain.attunement.store import merge_into_learned

    if cap is None:
        cap = DAILY_BUDGET_DEFAULT

    if detector_fn is None:
        from brain.attunement.detector import run_detector as _default_detector
        if only_categories is not None:
            _cats = only_categories  # capture for closure

            def detector_fn(*, buffer_slice, reply_text):  # noqa: ANN001
                return _default_detector(
                    buffer_slice=buffer_slice,
                    reply_text=reply_text,
                    only_categories=_cats,
                )
        else:
            detector_fn = _default_detector

    now_dt = now_dt or _datetime.now(UTC)
    now_iso = _now_iso_str(now_dt)

    existing = _load_state(persona_dir)
    # Normal path: short-circuit if already complete.
    # Supplementary path: bypass the early-return — the whole point is to
    # re-run after a prior completion, starting fresh at cursor=0.
    if not supplementary and existing is not None and existing.status == "complete":
        return existing

    turns = _read_buffer_turns(persona_dir)
    all_windows = window_buffer(turns)
    sampled = select_sample(all_windows)

    if supplementary:
        # Always start fresh — ignore any cursor from the previous completed state
        start_idx = 0
        processed_so_far = 0
        started_at = now_iso
        patterns_emitted_so_far = 0
    else:
        start_idx = _cursor_index(sampled, existing.last_cursor) if existing else 0
        processed_so_far = existing.processed_windows if existing else 0
        started_at = existing.started_at if existing else now_iso
        patterns_emitted_so_far = existing.patterns_emitted if existing else 0

    state = BackfillState(
        started_at=started_at,
        total_windows=len(all_windows),
        sampled_windows=len(sampled),
        processed_windows=processed_so_far,
        patterns_emitted=patterns_emitted_so_far,
        status="running",
        last_cursor=existing.last_cursor if existing else "",
        schema_version=SCHEMA_VERSION,
    )
    _save_state(persona_dir, state)

    for i in range(start_idx, len(sampled)):
        window = sampled[i]
        if not _attunement_consume_call(persona_dir, now=now_dt, cap=cap):
            state = replace(state, status="deferred_to_next_day")
            _save_state(persona_dir, state)
            return state

        try:
            output = detector_fn(buffer_slice=list(window.turns), reply_text="")
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "attunement backfill: detector error on %s: %s", window.id, exc
            )
            continue

        merge_into_learned(
            persona_dir,
            output.pattern_candidates,
            list(window.turns),
            now_iso=now_iso,
        )
        patterns_emitted_so_far += len(output.pattern_candidates)
        processed_so_far += 1
        state = replace(
            state,
            processed_windows=processed_so_far,
            patterns_emitted=patterns_emitted_so_far,
            last_cursor=window.id,
        )
        _save_state(persona_dir, state)

    check_crystallisations(persona_dir, now_iso=now_iso)
    state = replace(state, status="complete")
    _save_state(persona_dir, state)
    return state


def run_supplementary_backfill(
    persona_dir: Path,
    *,
    detector_fn=None,
    now_dt: _datetime | None = None,
    cap: int | None = None,
):
    """One-time supplementary pass after a schema upgrade.

    Extracts ONLY the new category dimensions (topic_affinity, response_shape,
    relational) from history. tone + cadence are already learned — no
    double-counting. Writes a fresh state at the current SCHEMA_VERSION so the
    feed-source will not fire another backfill_complete for the same completion.
    """
    return run_backfill(
        persona_dir,
        detector_fn=detector_fn,
        now_dt=now_dt,
        cap=cap,
        only_categories=_NEW_CATEGORIES,
        supplementary=True,
    )


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
