"""Research — autonomous exploration of developed interests.

See docs/superpowers/specs/2026-04-24-week-4-research-engine-design.md.
This module ships the types + engine scaffold. run_tick body lands in Task 3.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.memory.store import MemoryStore
from brain.search.base import WebSearcher
from brain.utils.time import iso_utc, parse_iso_utc

logger = logging.getLogger(__name__)


# ---------- Types ----------


@dataclass(frozen=True)
class ResearchFire:
    """Record of one research firing."""

    interest_id: str
    topic: str
    fired_at: datetime  # tz-aware UTC
    trigger: str  # "manual" | "emotion_high" | "days_since_human"
    web_used: bool
    web_result_count: int
    output_memory_id: str | None  # None in dry-run

    def to_dict(self) -> dict:
        return {
            "interest_id": self.interest_id,
            "topic": self.topic,
            "fired_at": iso_utc(self.fired_at),
            "trigger": self.trigger,
            "web_used": self.web_used,
            "web_result_count": self.web_result_count,
            "output_memory_id": self.output_memory_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResearchFire:
        return cls(
            interest_id=str(data["interest_id"]),
            topic=str(data["topic"]),
            fired_at=parse_iso_utc(data["fired_at"]),
            trigger=str(data["trigger"]),
            web_used=bool(data["web_used"]),
            web_result_count=int(data["web_result_count"]),
            output_memory_id=data.get("output_memory_id"),
        )


@dataclass(frozen=True)
class ResearchResult:
    """Outcome of a single research evaluation."""

    fired: ResearchFire | None
    would_fire: str | None  # dry-run only — topic that would fire
    reason: (
        str | None
    )  # "not_due"|"no_eligible_interest"|"no_interests_defined"|"research_raised"|"reflex_won_tie"
    dry_run: bool
    evaluated_at: datetime  # tz-aware UTC


# ---------- Engine scaffold ----------


@dataclass
class ResearchEngine:
    """Autonomous exploration of developed interests.

    run_tick() implementation lands in Task 3; scaffold ships here.
    """

    store: MemoryStore
    provider: LLMProvider
    searcher: WebSearcher
    persona_name: str
    persona_system_prompt: str
    interests_path: Path
    research_log_path: Path
    default_interests_path: Path

    def run_tick(
        self,
        *,
        trigger: str = "manual",
        dry_run: bool = False,
        forced_interest_topic: str | None = None,
        emotion_state_override=None,
        days_since_human_override: float | None = None,
    ) -> ResearchResult:
        raise NotImplementedError("run_tick body lands in Task 3")


# ---------- Research log ----------


@dataclass(frozen=True)
class ResearchLog:
    """Fire-history log for one persona."""

    fires: tuple[ResearchFire, ...] = ()

    @classmethod
    def load(cls, path: Path) -> ResearchLog:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return cls()
            fires_raw = data.get("fires", [])
            if not isinstance(fires_raw, list):
                return cls()
            return cls(
                fires=tuple(ResearchFire.from_dict(f) for f in fires_raw if isinstance(f, dict))
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return cls()

    def save(self, path: Path) -> None:
        payload = {"version": 1, "fires": [f.to_dict() for f in self.fires]}
        tmp = path.with_suffix(path.suffix + ".new")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def appended(self, fire: ResearchFire) -> ResearchLog:
        return ResearchLog(fires=self.fires + (fire,))
