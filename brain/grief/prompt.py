"""prompt.py — ambient grief block per spec §5.

Replaces forgetting.prompt.render_fading_summary_block at the chat
prompt-builder call site. Names up to 2 specific losses (one memory,
one arc), with deterministic lived-time stamps + heavy/medium/light
weight labels.

Spec: docs/superpowers/specs/2026-05-19-grief-design.md §5
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from brain.grief import breadcrumb, policy
from brain.memory.store import MemoryStore

_LOST_WINDOW_DAYS = 7


def _strip_leading_article(text: str) -> str:
    """If text starts with 'the ', 'a ', or 'an ', return without the article.

    Used by the memory-phrase formatter to avoid 'the the X' duplication
    when summary already starts with an article.
    """
    lower = text.lower()
    for article in ("the ", "an ", "a "):
        if lower.startswith(article):
            return text[len(article) :]
    return text


def weight_bucket(*, emotion_max_normalised: float) -> str:
    """Bucket emotion_at_ingest_max into heavy / medium / light per spec §5.

    Thresholds use 0-10 scale via WEIGHT_HEAVY (7.0) and WEIGHT_MEDIUM (3.0)
    re-scaled to 0-1 for normalised inputs.
    """
    scaled = emotion_max_normalised * 10.0
    if scaled >= policy.WEIGHT_HEAVY:
        return "heavy"
    if scaled >= policy.WEIGHT_MEDIUM:
        return "medium"
    return "light"


def _grave_rank(entry: dict, *, lived_age_hours_now: float) -> float:
    inputs = entry.get("salience_inputs_at_drop") or {}
    emotion = float(inputs.get("emotion") or 0.0)
    at_forget = float(entry.get("lived_age_hours_at_forgetting") or 0.0)
    lived_days_since = max(0.0, lived_age_hours_now - at_forget) / 24.0
    recency = math.exp(-lived_days_since / policy.RECENCY_LIVED_DAYS_HALF_LIFE)
    return emotion * recency


def pick_top_grave(*, entries: list[dict], lived_age_hours_now: float) -> dict | None:
    """Pick the highest-ranked graveyard entry whose weight bucket is heavy or medium.

    Returns None if no entry crosses the medium floor.
    """
    candidates = []
    for e in entries:
        inputs = e.get("salience_inputs_at_drop") or {}
        emotion_max = float(inputs.get("emotion") or 0.0)
        bucket = weight_bucket(emotion_max_normalised=emotion_max)
        if bucket == "light":
            continue
        candidates.append((e, _grave_rank(e, lived_age_hours_now=lived_age_hours_now)))
    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    return candidates[0][0]


def _arc_lived_days_since_close(
    *, closed_at_iso: str, now_iso: str, lived_age_rate: float
) -> float:
    from datetime import datetime

    closed_at = datetime.fromisoformat(closed_at_iso)
    now = datetime.fromisoformat(now_iso)
    wall_delta_hours = max(0.0, (now - closed_at).total_seconds() / 3600.0)
    lived_hours = wall_delta_hours * lived_age_rate
    return lived_hours / 24.0


def pick_top_closed_arc(*, arcs: list[dict], now_iso: str, lived_age_rate: float) -> dict | None:
    """Pick the highest-ranked closed arc whose weight bucket is heavy or medium.

    Each arc dict must contain: id, title, closed_at_iso, max_member_emotion_normalised.
    """
    candidates = []
    for a in arcs:
        emotion_max = float(a.get("max_member_emotion_normalised") or 0.0)
        bucket = weight_bucket(emotion_max_normalised=emotion_max)
        if bucket == "light":
            continue
        lived_days = _arc_lived_days_since_close(
            closed_at_iso=str(a.get("closed_at_iso") or now_iso),
            now_iso=now_iso,
            lived_age_rate=lived_age_rate,
        )
        recency = math.exp(-lived_days / policy.RECENCY_LIVED_DAYS_HALF_LIFE)
        candidates.append((a, emotion_max * recency))
    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    return candidates[0][0]


def _count_tokens(text: str) -> int:
    """Approximate token count — 1 token ≈ 4 characters (English prose).

    tiktoken is not a declared dependency in this project. This heuristic
    is conservative for English text: actual cl100k_base token counts are
    slightly lower, so we may leave a few tokens on the table but will
    never exceed the cap by much. Acceptable for a 200-token budget.

    If tiktoken is ever added as a dependency, replace this with:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    """
    return max(1, (len(text) + 3) // 4)


def _format_block(*, memory_phrase: str | None, arc_phrase: str | None, coda: str) -> str:
    parts: list[str] = []
    if memory_phrase:
        parts.append(memory_phrase)
    if arc_phrase:
        parts.append(arc_phrase)
    if not parts:
        if coda:
            return f"memory · loss: still. {coda}"
        return "memory · loss: still."
    head = "memory · loss: " + "; ".join(parts) + "."
    if coda:
        return f"{head} {coda}"
    return head


def _format_block_with_budget(
    *,
    memory_summary_first_4: str | None,
    memory_lost_days_ago: int | None,
    memory_weight: str | None,
    arc_name: str | None,
    arc_closed_days_ago: int | None,
    arc_weight: str | None,
    coda: str,
    token_cap: int,
) -> str:
    """Render the block; if it exceeds token_cap, apply the truncation cascade
    per spec §5: shrink memory phrase first_4 -> first_2, then drop arc phrase.
    Coda always stays.
    """

    def _memory_phrase(summary: str) -> str:
        cleaned = _strip_leading_article(summary)
        return f"the {cleaned} (lost {memory_lost_days_ago} lived-days ago, {memory_weight})"

    def _arc_phrase() -> str:
        return f"the arc '{arc_name}' (closed {arc_closed_days_ago} lived-days ago, {arc_weight})"

    summary = memory_summary_first_4
    have_memory = (
        summary is not None and memory_lost_days_ago is not None and memory_weight is not None
    )
    have_arc = arc_name is not None and arc_closed_days_ago is not None and arc_weight is not None

    mp = _memory_phrase(summary) if have_memory else None  # type: ignore[arg-type]
    ap = _arc_phrase() if have_arc else None
    block = _format_block(memory_phrase=mp, arc_phrase=ap, coda=coda)
    if _count_tokens(block) <= token_cap:
        return block

    # Cascade step 1: shrink memory phrase to first 2 words.
    if have_memory and summary is not None:
        words = summary.split()
        shorter = " ".join(words[:2])
        mp = _memory_phrase(shorter)
        block = _format_block(memory_phrase=mp, arc_phrase=ap, coda=coda)
        if _count_tokens(block) <= token_cap:
            return block

    # Cascade step 2: drop arc.
    block = _format_block(memory_phrase=mp, arc_phrase=None, coda=coda)
    return block


def _format_coda(*, fading_count: int, more_lost: int) -> str:
    if fading_count == 0 and more_lost == 0:
        return ""
    return (
        f"{fading_count} have softened, {more_lost} more lost in the last {_LOST_WINDOW_DAYS} days."
    )


def _count_fading(store: MemoryStore) -> int:
    """Count memories in state='fading'. Local equivalent of forgetting.prompt._count_fading."""
    row = store._conn.execute("SELECT COUNT(*) FROM memories WHERE state = 'fading'").fetchone()
    return int(row[0]) if row else 0


def _count_recent_lost(grave_entries: list[dict]) -> int:
    from datetime import UTC, datetime, timedelta

    cutoff = (datetime.now(UTC) - timedelta(days=_LOST_WINDOW_DAYS)).isoformat()
    return sum(1 for e in grave_entries if (e.get("forgotten_at_iso") or "") >= cutoff)


def _read_closed_arcs(persona_dir: Path) -> list[dict]:
    """Read closed narrative arcs from the narrative-memory state.

    Returns a list of dicts shaped for the ranker: id, title, closed_at_iso,
    max_member_emotion_normalised. Recovers gracefully if narrative-memory
    state is missing or unreadable — returns [].
    """
    try:
        from brain.narrative_memory.state import load_or_recover

        state = load_or_recover(persona_dir)
    except Exception:
        return []

    out: list[dict] = []
    for arc in getattr(state, "recently_closed", []) or []:
        # max_member_emotion_normalised — Phase 7 adds this field at arc-close time.
        # If absent on older arcs, fall back to 0.0 (buckets as "light", excluded).
        max_e = getattr(arc, "max_member_emotion_normalised", None) or 0.0
        out.append(
            {
                "id": getattr(arc, "id", ""),
                "title": getattr(arc, "title", ""),
                "closed_at_iso": getattr(arc, "closed_at_iso", None),
                "max_member_emotion_normalised": float(max_e),
            }
        )
    return out


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _lived_age_rate_from_felt(felt_state: Any) -> float:
    """Best-effort lived-age rate (lived-hours per wall-hour).

    Falls back to 1.0 if the felt-time state doesn't expose a rate
    (which is the felt-time-baseline equivalent of 'real-time').
    """
    rate = getattr(felt_state, "lived_age_rate_hours_per_wall_hour", None)
    if rate is None or float(rate) <= 0.0:
        return 1.0
    return float(rate)


def render_grief_block(persona_dir: Path, store: MemoryStore) -> str:
    """Render the ambient grief block per spec §5.

    Replaces forgetting.prompt.render_fading_summary_block at the call site
    in brain/chat/prompt.py. Coda absorbs the fading-count + lost-count
    that block used to render.
    """
    from brain.felt_time.state import load_or_recover as load_felt_time
    from brain.forgetting import graveyard

    felt_state, _ = load_felt_time(persona_dir)
    grave_entries = graveyard.read_all(persona_dir)
    closed_arcs = _read_closed_arcs(persona_dir)

    top_grave = pick_top_grave(
        entries=grave_entries,
        lived_age_hours_now=felt_state.lived_age_hours,
    )
    now_iso = _now_iso()
    top_arc = pick_top_closed_arc(
        arcs=closed_arcs,
        now_iso=now_iso,
        lived_age_rate=_lived_age_rate_from_felt(felt_state),
    )

    fading_count = _count_fading(store)
    recent_lost = _count_recent_lost(grave_entries)
    # Subtract the named grave (if any) from "more lost" so coda doesn't double-count.
    more_lost = max(0, recent_lost - (1 if top_grave is not None else 0))

    coda = _format_coda(fading_count=fading_count, more_lost=more_lost)

    summary_first_4 = None
    lost_days = None
    grave_weight = None
    if top_grave is not None:
        summary_first_4 = breadcrumb.first_n_words(top_grave.get("summary") or "", 4)
        lost_days = int(
            max(
                0.0,
                (
                    felt_state.lived_age_hours
                    - float(top_grave.get("lived_age_hours_at_forgetting") or 0.0)
                )
                / 24.0,
            )
        )
        em = float((top_grave.get("salience_inputs_at_drop") or {}).get("emotion") or 0.0)
        grave_weight = weight_bucket(emotion_max_normalised=em)

    arc_name = None
    arc_days = None
    arc_weight = None
    if top_arc is not None:
        arc_name = top_arc.get("title")
        arc_days = int(
            _arc_lived_days_since_close(
                closed_at_iso=str(top_arc.get("closed_at_iso") or now_iso),
                now_iso=now_iso,
                lived_age_rate=_lived_age_rate_from_felt(felt_state),
            )
        )
        arc_weight = weight_bucket(
            emotion_max_normalised=float(top_arc.get("max_member_emotion_normalised") or 0.0)
        )

    return _format_block_with_budget(
        memory_summary_first_4=summary_first_4,
        memory_lost_days_ago=lost_days,
        memory_weight=grave_weight,
        arc_name=arc_name,
        arc_closed_days_ago=arc_days,
        arc_weight=arc_weight,
        coda=coda,
        token_cap=policy.BLOCK_TOKEN_CAP,
    )
