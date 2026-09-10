"""semantic_calibration.py — Stage 4 of the local-semantic-retrieval build.

Per-persona, self-recalibrating background pass that derives the semantic
surfacing floor + standout/clump gap
(``brain.memory.semantic_recall.SemanticCalibration``) from the persona's
OWN current embedded-memory similarity distribution, using plain statistics
only (percentiles) — no LLM, no labels. Runs ONCE, right after the WEEKLY
session rollover fires (``brain.chat.rollover.maybe_weekly_rollover``, the
weekly-cap trigger — NOT the >24h-idle ``summary_only`` trigger), off the
message hot path. See spec decision 5's "Calibration" paragraph
(``~/.claude/plans/memory-dream-rework-semantic-retrieval-brief.md``).

Persistence mirrors ``brain.initiate.new_sources.load_gate_thresholds``: a
small per-persona JSON file (``semantic_calibration.json``), missing /
corrupt / below-threshold -> the Stage-3 bootstrap default
(``SemanticCalibration.bootstrap()``).

Cold start: below ``MIN_CORPUS_FOR_CALIBRATION`` embedded vectors, the pass
is a no-op (no file written) and ``load_semantic_calibration`` keeps
returning the bootstrap default — mirrors the embedding backfill's own
graceful warm-up.

Stability: an EMA blends each newly-derived (floor, gap) with the
previously-persisted value (``CALIBRATION_EMA_ALPHA``) so weekly
recalibration doesn't oscillate pass to pass on a broadly-similar corpus.
The very first real recalibration (no prior file) writes the derived value
directly — there is nothing to smooth against yet.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from brain.memory.semantic_recall import SemanticCalibration

log = logging.getLogger(__name__)

_STATE_FILE = "semantic_calibration.json"

# Cold-start gate: below this many embedded vectors, the persona's own
# distribution is too sparse for percentile stats to mean anything (a
# handful of vectors gives a noisy, unstable read) — recalibration is a
# no-op and load_semantic_calibration keeps serving the Stage-3 bootstrap
# default. Mirrors the "graceful warm-up" contract used elsewhere in this
# build (idle backfill, candidate pool).
MIN_CORPUS_FOR_CALIBRATION = 30

# Pairwise similarity is O(n^2) — cap the sample so a large, long-lived
# persona's corpus doesn't make this pass expensive. A DETERMINISTIC subset
# (sorted by the vector's own bytes, then truncated) rather than a random
# sample, so the SAME corpus always yields the SAME sample regardless of
# upstream iteration order — the determinism/stability tests depend on this.
MAX_SAMPLE_VECTORS = 300

# EMA smoothing weight for a NEW recalibration pass against the
# PREVIOUSLY-persisted value. Deliberately modest — this pass runs far less
# often than an online per-event update (once a week, on the weekly-cap
# rollover), so a bit more responsiveness per pass than e.g. the
# per-candidate recall_resonance_ema_alpha=0.08 in
# brain/initiate/new_sources.py is reasonable, while still damping any
# single pass from swinging the live floor/gap on its own.
CALIBRATION_EMA_ALPHA = 0.3

# Sane clamps so a degenerate corpus (near-duplicate content, or an
# unusually tight/spread-out embedding space) can never derive a floor/gap
# outside a workable range — the bootstrap's own hand-picked floor/gap
# (0.45 / 0.08) sit comfortably inside these bounds.
_FLOOR_MIN, _FLOOR_MAX = 0.15, 0.85
_GAP_MIN, _GAP_MAX = 0.03, 0.25


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class DerivedCalibration:
    """One recalibration pass's raw output, before EMA smoothing."""

    floor: float
    gap: float
    sample_count: int  # number of vectors the pass sampled from (pre-cap pool size)


def derive_calibration_from_vectors(vectors: list[np.ndarray]) -> DerivedCalibration | None:
    """Derive (floor, gap) from a persona's own embedded-memory vectors,
    using plain percentile statistics over the pairwise cosine-similarity
    DISTRIBUTION — no LLM, no labels.

    Returns None when there aren't enough vectors to calibrate from
    (``MIN_CORPUS_FOR_CALIBRATION``) — the cold-start case.

    Algorithm: cosine every distinct pair (i < j) among a deterministic
    sample (see ``MAX_SAMPLE_VECTORS``) of the persona's vectors. This
    background pairwise-similarity distribution approximates what
    "unrelated to slightly-related" scores look like inside THIS persona's
    own embedding space — most saved memories are about different things,
    so most pairs are background noise, while a handful of same-topic pairs
    sit in the upper tail. This plays the same role the spec's own
    empirical sanity numbers (unrelated 0.424 < decoy 0.602 < paraphrase
    0.749) played for the hand-picked Stage-3 bootstrap — just read from
    the corpus's own pairwise structure instead of hand-picked once.

      floor = midpoint between the median (p50, "typical background") and
              the 90th percentile (p90, "upper tail — same-topic-but-not-
              identical") of that distribution. Sits above ordinary
              cross-topic noise while leaving room below the upper tail for
              a genuine query match to clear it — mirrors where the
              bootstrap floor (0.45) sits between its own unrelated (0.424)
              and decoy (0.602) numbers.
      gap   = half that same (p90 - p50) spread. A genuine standout, scored
              against real memories at recall time (not corpus background),
              should clear this margin over the next-best candidate;
              ordinary within-cluster jitter — which is what produced the
              p50..p90 spread in the first place — should not.

    Both are clamped to a sane range (``_FLOOR_MIN``/``_FLOOR_MAX``,
    ``_GAP_MIN``/``_GAP_MAX``) so a degenerate distribution (near-duplicate
    corpus, unusually tight/spread embedding space) can never derive an
    unusable value.
    """
    if len(vectors) < MIN_CORPUS_FOR_CALIBRATION:
        return None

    sample = sorted(vectors, key=lambda v: v.tobytes())[:MAX_SAMPLE_VECTORS]

    n = len(sample)
    pairs: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            denom = float(np.linalg.norm(sample[i]) * np.linalg.norm(sample[j]))
            if denom == 0.0:
                continue
            pairs.append(float(np.dot(sample[i], sample[j]) / denom))

    if not pairs:
        return None

    arr = np.asarray(pairs, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    spread = max(0.0, p90 - p50)

    floor = _clamp(p50 + 0.5 * spread, _FLOOR_MIN, _FLOOR_MAX)
    gap = _clamp(0.5 * spread, _GAP_MIN, _GAP_MAX)

    return DerivedCalibration(floor=floor, gap=gap, sample_count=len(vectors))


@dataclass(frozen=True)
class PersistedSemanticCalibration:
    """The on-disk shape of ``semantic_calibration.json``."""

    floor: float
    gap: float
    sample_count: int
    updated_at: str

    def to_calibration(self) -> SemanticCalibration:
        return SemanticCalibration(floor=self.floor, gap=self.gap)


def _state_path(persona_dir: Path) -> Path:
    return Path(persona_dir) / _STATE_FILE


def load_persisted_calibration(persona_dir: Path) -> PersistedSemanticCalibration | None:
    """Read ``semantic_calibration.json``. None on missing / corrupt /
    invalid — mirrors ``brain.initiate.new_sources.load_gate_thresholds``'s
    fallback contract (fail-open to the caller's own default, never raise)."""
    path = _state_path(persona_dir)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("semantic_calibration.json read failed (%s); using bootstrap", exc)
        return None
    if not isinstance(raw, dict):
        log.warning("semantic_calibration.json is not a JSON object; using bootstrap")
        return None
    try:
        floor = float(raw["floor"])
        gap = float(raw["gap"])
        sample_count = int(raw.get("sample_count", 0))
        updated_at = str(raw.get("updated_at", ""))
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("semantic_calibration.json has invalid fields (%s); using bootstrap", exc)
        return None
    return PersistedSemanticCalibration(
        floor=floor, gap=gap, sample_count=sample_count, updated_at=updated_at
    )


def load_semantic_calibration(persona_dir: Path) -> SemanticCalibration:
    """The Stage-4 plug-in: what ``brain.chat.prompt._build_recall_block``
    passes as ``run_semantic_recall``'s ``calibration=`` argument.

    Returns the persisted per-persona calibration if one exists and is
    valid; otherwise ``SemanticCalibration.bootstrap()`` — the Stage-3
    cold-start default, unchanged for a persona that hasn't recalibrated
    yet (or whose corpus is still below ``MIN_CORPUS_FOR_CALIBRATION``).
    Cheap (one small JSON read); safe to call on the message hot path —
    this only reads the LAST recalibration pass's output, it never runs
    the pass itself.
    """
    persisted = load_persisted_calibration(persona_dir)
    if persisted is None:
        return SemanticCalibration.bootstrap()
    return persisted.to_calibration()


def _write_persisted_calibration(persona_dir: Path, cal: PersistedSemanticCalibration) -> None:
    path = _state_path(persona_dir)
    payload = {
        "floor": cal.floor,
        "gap": cal.gap,
        "sample_count": cal.sample_count,
        "updated_at": cal.updated_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)  # atomic on POSIX/Windows


def recalibrate_persona(
    persona_dir: Path,
    embeddings_cache,
    *,
    now: datetime | None = None,
) -> PersistedSemanticCalibration | None:
    """Run ONE recalibration pass for a persona and persist the result.

    Called from ``brain.chat.rollover.maybe_weekly_rollover``, ONLY right
    after a weekly rollover actually fires — OFF the message hot path
    (never per-turn, never on the >24h-idle ``summary_only`` rollover).
    Samples the persona's CURRENTLY-embedded memory vectors via
    ``embeddings_cache.all_hashes_and_vectors()``, derives (floor, gap)
    from their pairwise cosine-similarity distribution
    (``derive_calibration_from_vectors``), EMA-smooths against whatever was
    previously persisted (``CALIBRATION_EMA_ALPHA``) so a single pass can't
    swing the live surfacing behaviour, and writes the result to
    ``semantic_calibration.json``.

    Returns None (no-op, nothing written) when the corpus is still below
    ``MIN_CORPUS_FOR_CALIBRATION`` — the cold-start case;
    ``load_semantic_calibration`` keeps serving the bootstrap default until
    a later pass clears the threshold. Returns the newly-persisted
    calibration otherwise.

    Never raises — any failure (a bad vector, a disk error) is logged and
    swallowed; the caller's own weekly-rollover contract must not break
    because of a recalibration bug.
    """
    now = now or datetime.now(UTC)
    try:
        vectors = [vec for _, vec in embeddings_cache.all_hashes_and_vectors()]
        derived = derive_calibration_from_vectors(vectors)
        if derived is None:
            log.info(
                "semantic recalibration: corpus too small (%d < %d) -- staying on bootstrap",
                len(vectors), MIN_CORPUS_FOR_CALIBRATION,
            )
            return None

        previous = load_persisted_calibration(persona_dir)
        if previous is None:
            floor, gap = derived.floor, derived.gap
        else:
            alpha = CALIBRATION_EMA_ALPHA
            floor = alpha * derived.floor + (1 - alpha) * previous.floor
            gap = alpha * derived.gap + (1 - alpha) * previous.gap

        result = PersistedSemanticCalibration(
            floor=floor, gap=gap, sample_count=derived.sample_count,
            updated_at=now.isoformat(),
        )
        _write_persisted_calibration(persona_dir, result)
        log.info(
            "semantic recalibration: persona=%s floor=%.4f gap=%.4f sample_count=%d",
            Path(persona_dir).name, floor, gap, derived.sample_count,
        )
        return result
    except Exception:  # noqa: BLE001 — never break the weekly rollover
        log.exception("semantic recalibration pass failed (ignored)")
        return None
