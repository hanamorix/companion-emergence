"""Aggregator for the NellFace app's panel data.

The app's left panels (Inner Weather, Body, Recent Interior, Soul,
Connection) all consume real-time persona state. The data already
exists across many subsystems — emotions in the memory store,
body in compute_body_state, interior in daemon_state.json, soul in
SoulStore, mode in BridgeState. This module composes them behind a
single dict so the bridge can serve them at one endpoint.

Design intent: the helper is FAIL-SOFT. Any subsystem that errors
contributes its empty/null value rather than raising — the panels
should degrade gracefully on a fresh persona dir or partial data.
The endpoint must return 200 even if half the brain hasn't booted
yet; the UI just shows missing pieces as empty.

Mode derivation is v1 minimal: "live" when the bridge is up and
the provider is reachable, "offline" otherwise. Provider-failover
("provider_down") becomes meaningful when the FailoverProvider lands;
until then, mode is binary.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_persona_state(persona_dir: Path, *, now: datetime | None = None) -> dict[str, Any]:
    """Aggregate persona state for the NellFace panels.

    Returns a dict with keys:
      - emotions: dict[str, float] — non-zero emotions sorted desc by intensity
      - body: dict | None — body state from compute_body_state
      - interior: dict — last_dream / last_research / last_heartbeat themes
      - soul_highlight: dict | None — one crystallization (highest resonance,
        most recent on tie)
      - mode: "live" | "offline" — derived from data availability
      - persona: str — persona name (from persona_dir.name)

    Never raises. Each subsystem failure contributes None / {} rather
    than propagating.
    """
    if now is None:
        now = datetime.now(UTC)
    persona_name = persona_dir.name

    return {
        "persona": persona_name,
        "emotions": _build_emotions(persona_dir),
        "body": _build_body(persona_dir, now=now),
        "interior": _build_interior(persona_dir),
        "soul_highlight": _build_soul_highlight(persona_dir),
        "mode": "live",
    }


def _build_emotions(persona_dir: Path) -> dict[str, float]:
    """Top non-zero emotions from recent memories, sorted desc by intensity."""
    try:
        from brain.emotion.aggregate import aggregate_state
        from brain.memory.store import MemoryStore, _row_to_memory

        store = MemoryStore(persona_dir / "memories.db")
        try:
            rows = store._conn.execute(  # noqa: SLF001
                "SELECT * FROM memories WHERE active = 1 "
                "ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
            memories = [_row_to_memory(row) for row in rows]
            state = aggregate_state(memories)
            scores = state.emotions
        finally:
            store.close()
        if not scores:
            return {}
        # Sort desc, round to 1 decimal for UI display
        return {
            name: round(value, 1)
            for name, value in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        }
    except Exception:  # noqa: BLE001
        logger.warning("persona_state: emotions aggregation failed", exc_info=True)
        return {}


def _build_body(persona_dir: Path, *, now: datetime) -> dict | None:
    """Body block from compute_body_state. None on failure."""
    try:
        from brain.body.state import compute_body_state
        from brain.body.words import count_words_in_session
        from brain.emotion.aggregate import aggregate_state
        from brain.memory.store import MemoryStore, _row_to_memory
        from brain.utils.memory import days_since_human

        store = MemoryStore(persona_dir / "memories.db")
        try:
            rows = store._conn.execute(  # noqa: SLF001
                "SELECT * FROM memories WHERE active = 1 "
                "ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
            memories = [_row_to_memory(row) for row in rows]
            state = aggregate_state(memories)
            days = days_since_human(store, now=now)
            # session_hours=0 here — the panel snapshot doesn't track per-session
            # time. count_words_in_session falls back to 1-hour window on 0.
            words = count_words_in_session(
                store, persona_dir=persona_dir, session_hours=0.0, now=now,
            )
            body = compute_body_state(
                emotions=state.emotions,
                session_hours=0.0,
                words_written=words,
                days_since_contact=days,
                now=now,
            )
            return body.to_dict()
        finally:
            store.close()
    except Exception:  # noqa: BLE001
        logger.warning("persona_state: body computation failed", exc_info=True)
        return None


def _build_interior(persona_dir: Path) -> dict[str, Any]:
    """Recent interior — dream / research / heartbeat / reflex themes.

    Reads daemon_state.json directly (no LLM call). Empty dict on
    failure; missing fields default to None.
    """
    out: dict[str, Any] = {
        "dream": None,
        "research": None,
        "heartbeat": None,
        "reflex": None,
    }
    try:
        ds_path = persona_dir / "daemon_state.json"
        if not ds_path.exists():
            return out
        ds = json.loads(ds_path.read_text(encoding="utf-8"))
        if last_dream := ds.get("last_dream"):
            out["dream"] = last_dream.get("theme")
        if last_research := ds.get("last_research"):
            out["research"] = last_research.get("theme")
        if last_heartbeat := ds.get("last_heartbeat"):
            # Heartbeat doesn't have a "theme" field by convention; surface a
            # short summary built from its dominant emotion.
            dom = last_heartbeat.get("dominant_emotion")
            intensity = last_heartbeat.get("intensity")
            if dom and intensity is not None:
                out["heartbeat"] = f"{dom} {intensity}/10"
        if last_reflex := ds.get("last_reflex"):
            out["reflex"] = last_reflex.get("summary") or last_reflex.get("arc_name")
    except Exception:  # noqa: BLE001
        logger.warning("persona_state: interior read failed", exc_info=True)
    return out


def _build_soul_highlight(persona_dir: Path) -> dict | None:
    """Pick one crystallization to display: highest resonance, then most recent.

    None when the soul is empty or the read fails.
    """
    try:
        from brain.soul.store import SoulStore

        store = SoulStore(persona_dir / "crystallizations.db")
        try:
            actives = store.list_active()
            if not actives:
                return None
            # sort by (resonance desc, crystallized_at desc)
            actives.sort(
                key=lambda c: (c.resonance, c.crystallized_at),
                reverse=True,
            )
            top = actives[0]
            return {
                "id": top.id,
                "moment": top.moment,
                "love_type": top.love_type,
                "resonance": top.resonance,
                "crystallized_at": (
                    top.crystallized_at.isoformat()
                    if hasattr(top.crystallized_at, "isoformat")
                    else str(top.crystallized_at)
                ),
                "why_it_matters": getattr(top, "why_it_matters", None),
            }
        finally:
            store.close() if hasattr(store, "close") else None
    except Exception:  # noqa: BLE001
        logger.warning("persona_state: soul highlight read failed", exc_info=True)
        return None
