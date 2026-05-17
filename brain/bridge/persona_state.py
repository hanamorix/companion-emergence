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
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Re-export the canonical helper so existing call sites in this module
# (and any external imports of the private name) keep working. The
# function moved to brain.body.session_hours so brain.tools.dispatch can
# import it without crossing into the bridge layer. See the new module's
# docstring for the migration context.
from brain.body.session_hours import (
    compute_active_session_hours as _active_session_hours,  # noqa: F401
)

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
        "connection": _build_connection(persona_dir),
        "mode": "live",
        "recovering": _is_recovering(persona_dir),
    }


def _is_recovering(persona_dir: Path) -> bool:
    """True iff orphan session buffers are present that aren't live
    in-memory sessions.

    Phase 3.A of the autonomous-memory work. Surfaces the dirty-shutdown
    recovery path to the UI so the user can see "your previous chat is
    still being saved" instead of "Nell forgot what we said." Without
    this signal, a hard-quit ingest failure is invisible until the
    next clean shutdown clears it.

    Detection: walks ``active_conversations/`` and returns True if any
    session-buffer dir's id is *not* in the live in-memory session
    registry. Live chats are buffered in the same place but their ids
    are tracked in ``brain.chat.session``; the difference is what tells
    us "this buffer is from a previous run, not the current one."

    Edges:
      - Fresh persona dir (no ``active_conversations/``) → False.
      - Bridge just started, recovery still in flight → True until
        supervisor's first tick drains the orphans.
      - User in the middle of a chat with no orphans → False
        (the live session is in the registry).

    Fail-soft: any exception (registry import error, disk read error)
    returns False so a half-booted brain doesn't pin the banner on.
    """
    try:
        from brain.chat.session import all_sessions

        ac_dir = persona_dir / "active_conversations"
        if not ac_dir.is_dir():
            return False
        live_sids = {s.session_id for s in all_sessions()}
        for entry in ac_dir.iterdir():
            if not entry.is_dir():
                continue
            if not (entry / "turns.jsonl").exists():
                continue
            if entry.name not in live_sids:
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


def _build_connection(persona_dir: Path) -> dict[str, Any]:
    """Provider + model + last-heartbeat — drives the Connection panel.

    Provider comes from persona_config (the operator-tier choice).
    Model is provider-specific; for v1 we surface a sensible default
    per provider rather than read it from a separate config (the
    runner-side model isn't currently persisted).
    Last-heartbeat-at comes from heartbeat_state.json — null when the
    heartbeat has never run.
    """
    out: dict[str, Any] = {
        "provider": None,
        "model": None,
        "last_heartbeat_at": None,
    }
    try:
        from brain.persona_config import PersonaConfig

        cfg = PersonaConfig.load(persona_dir / "persona_config.json")
        out["provider"] = cfg.provider
        out["model"] = _default_model_for(cfg.provider)
    except Exception:  # noqa: BLE001
        logger.warning("persona_state: persona_config read failed", exc_info=True)
    try:
        hb_path = persona_dir / "heartbeat_state.json"
        if hb_path.exists():
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
            out["last_heartbeat_at"] = hb.get("last_tick_at")
    except Exception:  # noqa: BLE001
        logger.warning("persona_state: heartbeat_state read failed", exc_info=True)
    return out


def _default_model_for(provider: str) -> str | None:
    """v1 default model per provider. Replace with a real config field
    when callers need provider-specific overrides."""
    if provider == "claude-cli":
        return "sonnet"
    if provider == "ollama":
        return "huihui_ai/qwen2.5-abliterated:7b"
    if provider == "fake":
        return "fake"
    return None


def _build_emotions(persona_dir: Path) -> dict[str, float]:
    """Top non-zero emotions from recent emotion-carrying memories.

    The previous query took the last 50 memories by ``created_at``
    regardless of whether they actually carried an emotion vector.
    On a steady-state brain the most recent 50 are almost all
    heartbeats, observations, and facts — engine-internal records
    with ``emotions_json = '{}'`` — so the aggregator returned 2-3
    emotions even when 16+ were live in the underlying memory
    store. Filtering to non-empty emotion rows fixes the surface
    Inner Weather panel without aggregating over every active row
    on each /state poll.
    """
    try:
        from brain.emotion.aggregate import aggregate_state
        from brain.memory.store import MemoryStore, _row_to_memory

        store = MemoryStore(persona_dir / "memories.db")
        try:
            rows = store._conn.execute(  # noqa: SLF001
                "SELECT * FROM memories "
                "WHERE active = 1 "
                "AND emotions_json IS NOT NULL "
                "AND emotions_json != '{}' "
                "ORDER BY created_at DESC LIMIT 200"
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
            # Same fix as _build_emotions: filter to memories that
            # actually carry an emotion vector. The naive last-50-by-date
            # slice was almost all heartbeats / observations / facts
            # (empty emotions_json), so body_emotions stayed at zero
            # even when the underlying brain had strong signal.
            rows = store._conn.execute(  # noqa: SLF001
                "SELECT * FROM memories "
                "WHERE active = 1 "
                "AND emotions_json IS NOT NULL "
                "AND emotions_json != '{}' "
                "ORDER BY created_at DESC LIMIT 200"
            ).fetchall()
            memories = [_row_to_memory(row) for row in rows]
            state = aggregate_state(memories)
            days = days_since_human(store, now=now, persona_dir=persona_dir)
            # session_hours from the active conversation buffer's
            # earliest entry — 0 when no session is open.
            session_hours = _active_session_hours(persona_dir, now=now)
            words = count_words_in_session(
                store,
                persona_dir=persona_dir,
                session_hours=session_hours,
                now=now,
            )
            body = compute_body_state(
                emotions=state.emotions,
                session_hours=session_hours,
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


def _strip_label(text: str | None, label: str) -> str | None:
    """Strip a leading section-label prefix from interior text.

    Dream/reflex/research themes are written by a model that sometimes
    self-narrates with a header (``"DREAM: I was back in..."``). The UI
    renders the section name above the text already, so the in-body
    prefix shows up twice. Strip it once at the data boundary so every
    consumer benefits.
    """
    if not text:
        return text
    pattern = re.compile(rf"^\s*{re.escape(label)}\s*[:\-—]\s*", re.IGNORECASE)
    return pattern.sub("", text, count=1).lstrip() or None


def _entry_iso_timestamp(entry: dict) -> str | None:
    """Pull the iso timestamp off a daemon_state ``last_*`` entry.

    Different writers use different field names — accept any of
    ``timestamp`` / ``ts`` / ``written_at`` / ``fired_at`` and
    return None if none parses.
    """
    for key in ("timestamp", "ts", "written_at", "fired_at"):
        raw = entry.get(key)
        if isinstance(raw, str) and raw:
            return raw
    return None


def _build_interior(persona_dir: Path) -> dict[str, Any]:
    """Recent interior — dream / research / heartbeat / reflex themes
    plus per-entry timestamps so the UI can render "X ago" badges.

    Reads daemon_state.json directly (no LLM call). Empty dict on
    failure; missing fields default to None.

    Output shape:
        {
          "dream":     {"summary": "...", "ts": "2026-...Z"} | null,
          "research":  {"summary": "...", "ts": "..."}        | null,
          "heartbeat": {"summary": "love 9/10", "ts": "..."}  | null,
          "reflex":    {"summary": "...", "ts": "..."}        | null,
        }
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
            # Prefer the full summary over the 80-char theme slice; theme is
            # a hard-cut headline (``mem.content[:80]``) that lands
            # mid-sentence ("DREAM: I was back in every conversation —
            # not like" was the screenshot symptom). Reflex already used
            # ``summary``; dream is brought into line.
            dream_text = last_dream.get("summary") or last_dream.get("theme")
            summary = _strip_label(dream_text, "dream")
            if summary:
                out["dream"] = {"summary": summary, "ts": _entry_iso_timestamp(last_dream)}
        if last_research := ds.get("last_research"):
            research_text = last_research.get("summary") or last_research.get("theme")
            summary = _strip_label(research_text, "research")
            if summary:
                out["research"] = {"summary": summary, "ts": _entry_iso_timestamp(last_research)}
        if last_heartbeat := ds.get("last_heartbeat"):
            dom = last_heartbeat.get("dominant_emotion")
            intensity = last_heartbeat.get("intensity")
            if dom and intensity is not None:
                out["heartbeat"] = {
                    "summary": f"{dom} {intensity}/10",
                    "ts": _entry_iso_timestamp(last_heartbeat),
                }
        if last_reflex := ds.get("last_reflex"):
            summary = _strip_label(
                last_reflex.get("summary") or last_reflex.get("arc_name"),
                "reflex",
            )
            if summary:
                out["reflex"] = {"summary": summary, "ts": _entry_iso_timestamp(last_reflex)}
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
