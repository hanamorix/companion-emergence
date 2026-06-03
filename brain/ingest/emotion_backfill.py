"""One-time historical emotion backfill.

Re-tags active memories that have ``emotions == {}`` so existing personas
benefit from the A2 forward-only emotion seeding.  Mirrors the pattern in
``brain/attunement/backfill.py`` (cursor-resume, daily budget cap, stratified
sampling, fault-isolated supervisor wiring).

Public surface
--------------
should_run_emotion_backfill(persona_dir) -> bool
run_emotion_backfill(persona_dir, *, tagger_fn, cap, now_dt) -> EmotionBackfillState
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC
from datetime import datetime as _datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STATE_FILE = "emotion_backfill_state.json"
_SCHEMA_VERSION = "v1"

# Per-run sample size: tag at most this many memories per invocation so we
# don't lock the store for a long time on large personas.
_SAMPLE_BATCH = 50


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmotionBackfillState:
    started_at: str
    total_memories: int
    tagged_memories: int
    last_cursor: str          # last-tagged memory id (for resume ordering)
    status: str               # "running" | "complete" | "deferred_to_next_day"
    schema_version: str


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------

def _state_path(persona_dir: Path) -> Path:
    return persona_dir / _STATE_FILE


def _load_state(persona_dir: Path) -> EmotionBackfillState | None:
    p = _state_path(persona_dir)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return EmotionBackfillState(**raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("emotion_backfill: corrupt state file: %s", exc)
        return None


def _save_state(persona_dir: Path, state: EmotionBackfillState) -> None:
    p = _state_path(persona_dir)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------------------
# Budget helpers (own counter so as not to share the attunement budget)
# ---------------------------------------------------------------------------

_BUDGET_FILE = "emotion_backfill_budget.json"


def _budget_path(persona_dir: Path) -> Path:
    return persona_dir / _BUDGET_FILE


def _today_str(now: _datetime) -> str:
    return now.astimezone().strftime("%Y-%m-%d")


def _consume_budget(persona_dir: Path, *, now: _datetime, cap: int) -> bool:
    """Return True and decrement if under cap; False if exhausted."""
    path = _budget_path(persona_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (json.JSONDecodeError, ValueError):
        raw = {}
    today = _today_str(now)
    if raw.get("date") != today:
        raw = {"date": today, "count": 0}
    if int(raw.get("count", 0)) >= cap:
        return False
    raw["count"] = int(raw.get("count", 0)) + 1
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(raw), encoding="utf-8")
    tmp.replace(path)
    return True


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def _load_vocab() -> frozenset[str]:
    try:
        from brain.emotion.vocabulary import list_all
        return frozenset(e.name for e in list_all())
    except Exception:  # noqa: BLE001
        logger.warning("emotion_backfill: failed to load vocab; all names allowed")
        return frozenset()


def _normalize(raw: dict[str, Any], vocab: frozenset[str]) -> dict[str, float]:
    """Keep only registered names with intensity in (0, 10]; clamp to 10."""
    result: dict[str, float] = {}
    for name, intensity in raw.items():
        if vocab and name not in vocab:
            continue
        try:
            v = float(intensity)
        except (TypeError, ValueError):
            continue
        if v <= 0.0:
            continue
        result[name] = min(v, 10.0)
    return result


# ---------------------------------------------------------------------------
# Default tagger (one Haiku call per memory)
# ---------------------------------------------------------------------------

def _default_tagger(memory) -> dict[str, float]:  # noqa: ANN001
    """Ask Haiku to emit an emotion vector for this memory's content."""
    try:
        from brain.bridge.provider import get_default_provider

        vocab = _load_vocab()
        vocab_str = ", ".join(sorted(vocab)) if vocab else "(any emotion name)"
        prompt = (
            f"Return a JSON object mapping emotion names to intensities (0-10) "
            f"for the following memory.  Use ONLY names from this list: {vocab_str}. "
            f"Omit emotions with intensity 0.  Return ONLY the JSON object, no other text.\n\n"
            f"Memory: {memory.content}"
        )
        provider = get_default_provider()
        raw_text = provider.generate(prompt, model="haiku")
        raw = json.loads(raw_text)
        return _normalize(raw, vocab)
    except Exception as exc:  # noqa: BLE001
        logger.warning("emotion_backfill: default tagger error: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def should_run_emotion_backfill(persona_dir: Path) -> bool:
    """Return True iff there are active memories with empty emotion vectors
    and no completed backfill state file.

    Mirrors ``brain.attunement.backfill.should_run_backfill``.
    """
    existing = _load_state(persona_dir)
    if existing is not None and existing.status == "complete":
        return False

    db_path = persona_dir / "memories.db"
    if not db_path.exists():
        return False

    from brain.memory.store import MemoryStore
    store = MemoryStore(str(db_path), integrity_check=False)
    try:
        memories = store.list_active()
        return any(not m.emotions for m in memories)
    finally:
        store.close()


def run_emotion_backfill(
    persona_dir: Path,
    *,
    tagger_fn: Callable | None = None,
    cap: int = 200,
    now_dt: _datetime | None = None,
) -> EmotionBackfillState:
    """Run (or resume) the one-time emotion backfill.

    - Selects active memories with ``emotions == {}`` ordered by ``created_at``
      (deterministic cursor).
    - Calls ``tagger_fn(memory)`` → ``dict[str, float]``; filters to registered
      vocab; writes back via ``store.update(id, emotions=...)``.
    - Respects a DAILY BUDGET CAP; persists cursor to resume across ticks.
    - Returns ``EmotionBackfillState`` with ``status`` in
      ``{"complete", "deferred_to_next_day", "running"}``.

    Mirrors ``brain.attunement.backfill.run_backfill``.
    """
    if tagger_fn is None:
        tagger_fn = _default_tagger

    now_dt = now_dt or _datetime.now(UTC)
    now_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Short-circuit if already complete.
    existing = _load_state(persona_dir)
    if existing is not None and existing.status == "complete":
        return existing

    vocab = _load_vocab()

    db_path = persona_dir / "memories.db"
    from brain.memory.store import MemoryStore
    store = MemoryStore(str(db_path), integrity_check=False)
    try:
        all_active = store.list_active()
    finally:
        store.close()

    # Filter to emotion-less only, sorted deterministically by id (stable cursor).
    candidates = sorted(
        [m for m in all_active if not m.emotions],
        key=lambda m: m.id,
    )

    total = len(all_active)
    tagged_so_far = existing.tagged_memories if existing else 0
    started_at = existing.started_at if existing else now_iso
    last_cursor = existing.last_cursor if existing else ""

    # Resume: skip memories we have already tagged (cursor = last-tagged id).
    if last_cursor:
        resume_idx = 0
        for i, m in enumerate(candidates):
            if m.id == last_cursor:
                resume_idx = i + 1
                break
        candidates = candidates[resume_idx:]

    state = EmotionBackfillState(
        started_at=started_at,
        total_memories=total,
        tagged_memories=tagged_so_far,
        last_cursor=last_cursor,
        status="running",
        schema_version=_SCHEMA_VERSION,
    )
    _save_state(persona_dir, state)

    for memory in candidates:
        if not _consume_budget(persona_dir, now=now_dt, cap=cap):
            state = replace(state, status="deferred_to_next_day")
            _save_state(persona_dir, state)
            logger.info(
                "emotion_backfill: daily cap (%d) reached; deferred. "
                "tagged_so_far=%d cursor=%s",
                cap, tagged_so_far, last_cursor,
            )
            return state

        try:
            raw_emotions = tagger_fn(memory)
        except Exception as exc:  # noqa: BLE001
            logger.warning("emotion_backfill: tagger error on %s: %s", memory.id, exc)
            continue

        filtered = _normalize(raw_emotions, vocab)
        if not filtered:
            continue

        try:
            store2 = MemoryStore(str(db_path), integrity_check=False)
            try:
                store2.update(memory.id, emotions=filtered)
            finally:
                store2.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "emotion_backfill: write-back error on %s: %s", memory.id, exc
            )
            continue

        tagged_so_far += 1
        last_cursor = memory.id
        state = replace(
            state,
            tagged_memories=tagged_so_far,
            last_cursor=last_cursor,
        )
        _save_state(persona_dir, state)

    state = replace(state, status="complete")
    _save_state(persona_dir, state)
    logger.info(
        "emotion_backfill: complete. tagged=%d total_active=%d",
        tagged_so_far, total,
    )
    return state
