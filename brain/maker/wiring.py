"""brain.maker.wiring — the three wire-backs of a completed making."""
from __future__ import annotations

import logging

from brain.maker.maker import Making

logger = logging.getLogger(__name__)

# base + per-source accent (mirrors the W8 reach_emotion map shape)
_BASE = {"satisfaction": 0.12, "tenderness": 0.08}
_SOURCE_ACCENT = {
    "grief": {"tenderness": 0.10},
    "joy": {"satisfaction": 0.10},
    "dream": {"wonder": 0.10},
    "soul": {"pride": 0.10},
}


def making_emotion_delta(making: Making, *, dominant_source: str) -> dict[str, float]:
    """Small vocab-filtered emotion delta for having made. Decay-subordinate."""
    from brain.chat.extractor import _filter_to_registered
    raw = dict(_BASE)
    for name, v in _SOURCE_ACCENT.get(dominant_source, {}).items():
        raw[name] = raw.get(name, 0.0) + v
    return _filter_to_registered(raw)


def flip_ready_shares(persona_dir, *, now, delay_hours: float) -> int:
    """eventual_share makings older than delay → mark shared_at (surfaces to feed). Returns count flipped."""
    from datetime import datetime

    from brain.works.store import WorksStore
    db = persona_dir / "works.db"
    if not db.exists():
        return 0
    store = WorksStore(db)
    flipped = 0
    try:
        conn = store._conn
        rows = conn.execute(
            "SELECT id, created_at FROM works WHERE disposition='eventual_share' AND shared_at IS NULL"
        ).fetchall()
        for r in rows:
            try:
                age_h = (now - datetime.fromisoformat(r["created_at"])).total_seconds() / 3600.0
            except ValueError:
                continue
            if age_h >= delay_hours:
                conn.execute("UPDATE works SET shared_at=? WHERE id=?", (now.isoformat(), r["id"]))
                flipped += 1
        conn.commit()
    except Exception:
        logger.exception("maker: flip_ready_shares failed")
    finally:
        store.close()
    return flipped


def write_making_memory(store, making: Making, *, emotions: dict[str, float]) -> None:
    """Episodic memory of the ACT of making (distinct from the artifact)."""
    from brain.memory.store import Memory
    content = f"I made something — \"{making.title}\" ({making.type}). It came from what's been moving in me."
    mem = Memory.create_new(
        content=content, memory_type="making", domain="interior",
        tags=["making", making.disposition], emotions=emotions or None,
    )
    try:
        store.create(mem)
    except Exception:
        logger.exception("maker: act-memory write failed")
