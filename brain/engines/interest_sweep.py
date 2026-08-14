"""Weekly interest sweep — new interests from lived life + retire dead threads.

Safety net behind the per-turn extractor inlet (spec §6.3). Caller owns
cadence (supervisor) + throttle (background_slot).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

from brain.engines._interests import InterestSet, spawn_interest
from brain.utils.llm_output import extract_json_object
from brain.utils.memory import list_conversation_memories

logger = logging.getLogger(__name__)

SWEEP_CADENCE_FILE = "interest_sweep_cadence.json"
SWEEP_INTERVAL_HOURS = 168.0
_CAP = 3

_SYSTEM = """\
You maintain a companion's interest list. Given her current interests and a
sample of her recent lived life (conversations, dreams, inner monologue),
propose at most 3 NEW research interests that clearly recur in the lived
material but are missing from the list, and at most 3 existing interest ids
to retire (nothing in the lived material touches them and their thread feels
done). Be conservative — empty lists are a fine answer. Return ONLY:
{"new": [{"topic": "...", "keywords": ["..."], "why": "..."}], "retire": ["<id>"]}"""


def _lived_sample(store) -> str:
    lines: list[str] = []
    try:
        for m in list_conversation_memories(store, active_only=True, limit=15):
            lines.append(f"- (conversation) {m.content[:120]}")
    except Exception:  # noqa: BLE001
        pass
    for mtype in ("dream", "monologue_trace"):
        try:
            for m in store.list_by_type(mtype, active_only=True, limit=5):
                lines.append(f"- ({mtype}) {m.content[:120]}")
        except Exception:  # noqa: BLE001
            pass
    return "\n".join(lines) or "(no recent lived material)"


def run_sweep_tick(
    *,
    store,
    provider,
    interests_path: Path,
    default_interests_path: Path,
    now: datetime,
) -> dict:
    """One sweep: propose <=3 new interests (origin='sweep') + <=3 retirements.

    Returns {"spawned": int, "retired": int, "error": str | None}. Never raises.
    Caller owns cadence + throttle.
    """
    result = {"spawned": 0, "retired": 0, "error": None}
    try:
        interests = InterestSet.load(interests_path, default_path=default_interests_path)
        listing = "\n".join(
            f"- id={i.id} topic={i.topic!r} status={i.status} pull={i.pull_score:.1f}"
            for i in interests.interests
        ) or "(empty)"
        prompt = (
            f"=== Current interests ===\n{listing}\n\n"
            f"=== Recent lived life ===\n{_lived_sample(store)}\n\n"
            'Return: {"new": [...], "retire": [...]}'
        )
        raw = provider.generate(prompt, system=_SYSTEM)
        data = json.loads(extract_json_object(raw))

        for item in list(data.get("new", []))[:_CAP]:
            topic = str(item.get("topic", "")).strip()
            if not topic:
                continue
            interests, created = spawn_interest(
                interests,
                topic=topic,
                keywords=tuple(str(k) for k in item.get("keywords", [])),
                why=str(item.get("why", "")),
                origin="sweep",
                now=now,
            )
            result["spawned"] += int(created)

        retire_ids = {str(r) for r in list(data.get("retire", []))[:_CAP]}
        if retire_ids:
            updated = []
            for i in interests.interests:
                if i.id in retire_ids and i.status == "active":
                    updated.append(dataclasses.replace(i, status="dormant"))
                    result["retired"] += 1
                else:
                    updated.append(i)
            interests = InterestSet(interests=tuple(updated))

        if result["spawned"] or result["retired"]:
            interests.save(interests_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("interest sweep failed: %s", exc)
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
