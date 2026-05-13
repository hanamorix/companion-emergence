"""Research — autonomous exploration of developed interests.

See docs/superpowers/specs/2026-04-24-week-4-research-engine-design.md.
This module ships the types + engine scaffold. run_tick body lands in Task 3.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.engines._interests import Interest, InterestSet
from brain.memory.store import Memory, MemoryStore
from brain.search.base import WebSearcher
from brain.utils.emotion import format_emotion_summary
from brain.utils.llm_output import extract_json_object
from brain.utils.memory import days_since_human
from brain.utils.time import iso_utc, parse_iso_utc

logger = logging.getLogger(__name__)

_TOPIC_OVERLAP_SYSTEM = """\
You are a relevance scorer for an autonomous companion's research engine.
You will be given (1) a research thread that just matured, and (2) recent
conversation excerpts. Return a JSON object with a single field: score,
a float in [0.0, 1.0] indicating how relevant the research thread is
to the recent conversation.

  0.0 = entirely unrelated to anything {user_name} has been near
  0.5 = thematic adjacency, no direct overlap
  1.0 = directly addresses something {user_name} mentioned

Be conservative — default toward the low end unless the connection
is clearly present. Return ONLY the JSON object, no other text."""


def _compute_topic_overlap_via_haiku(
    *,
    thread_topic: str,
    thread_summary: str,
    recent_conversation_excerpt: str,
    provider,
    user_name: str,
) -> float:
    """Score how relevant a matured research thread is to recent conversation."""
    system = _TOPIC_OVERLAP_SYSTEM.format(user_name=user_name)
    prompt = (
        "=== Research thread ===\n"
        f"Topic: {thread_topic}\n"
        f"Summary: {thread_summary}\n\n"
        "=== Recent conversation (last 48 hours, oldest first) ===\n"
        f"{recent_conversation_excerpt}\n\n"
        "=== Your task ===\n"
        'Return: {"score": <float in [0.0, 1.0]>}'
    )

    try:
        raw = provider.generate(prompt, system=system)
        data = json.loads(extract_json_object(raw))
        score = float(data["score"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
        logger.warning("topic_overlap haiku call failed (%s); returning 0.0", exc)
        return 0.0
    return max(0.0, min(1.0, score))


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


# ---------- Engine ----------


@dataclass
class ResearchEngine:
    """Autonomous exploration of developed interests."""

    store: MemoryStore
    provider: LLMProvider
    searcher: WebSearcher
    persona_name: str
    persona_system_prompt: str
    interests_path: Path
    research_log_path: Path
    default_interests_path: Path

    pull_threshold: float = 6.0
    cooldown_hours: float = 24.0

    def run_tick(
        self,
        *,
        trigger: str = "manual",
        dry_run: bool = False,
        emotion_state_override=None,
        days_since_human_override: float | None = None,
    ) -> ResearchResult:
        """Evaluate triggers, select an interest, fire (or report would_fire).

        The brain owns topic selection — there is no force-research-this-topic
        bypass. Per principle audit 2026-04-25: the user can't tell the brain
        what to research; the brain decides from its own developed pull.
        """
        now = datetime.now(UTC)

        interests = InterestSet.load(self.interests_path, default_path=self.default_interests_path)
        # Load existing log so new fires append cumulatively (rather than
        # clobbering prior fire history on save). Cooldown logic is driven by
        # Interest.last_researched_at, not this log — the log is audit-trail only.
        log = ResearchLog.load(self.research_log_path)

        if not interests.interests:
            return ResearchResult(
                fired=None,
                would_fire=None,
                reason="no_interests_defined",
                dry_run=dry_run,
                evaluated_at=now,
            )

        # Gate: need a trigger signal (days-since-human or emotion-peak).
        days_since = (
            days_since_human_override
            if days_since_human_override is not None
            else days_since_human(self.store, now, persona_dir=self.interests_path.parent)
        )
        emo_state = emotion_state_override
        if emo_state is None:
            from brain.emotion.aggregate import aggregate_state

            all_mems = self.store.list_active(limit=None)
            emo_state = aggregate_state(all_mems)

        emo_peak = max(emo_state.emotions.values(), default=0.0)

        gate_ok = (
            days_since >= 1.5  # research_days_since_human_min default
            or emo_peak >= 7.0  # research_emotion_threshold default
        )
        if not gate_ok:
            return ResearchResult(
                fired=None,
                would_fire=None,
                reason="not_due",
                dry_run=dry_run,
                evaluated_at=now,
            )

        # Select eligible interest — brain picks the highest-pull eligible one.
        eligible = interests.list_eligible(
            pull_threshold=self.pull_threshold,
            cooldown_hours=self.cooldown_hours,
            now=now,
        )
        winner = eligible[0] if eligible else None

        if winner is None:
            return ResearchResult(
                fired=None,
                would_fire=None,
                reason="no_eligible_interest",
                dry_run=dry_run,
                evaluated_at=now,
            )

        if dry_run:
            return ResearchResult(
                fired=None,
                would_fire=winner.topic,
                reason=None,
                dry_run=True,
                evaluated_at=now,
            )

        # Memory sweep: keyword-matched seed memories
        memory_context = self._build_memory_context(winner)

        # Web search (conditional on scope)
        web_results: list = []
        if winner.scope != "internal":
            try:
                web_results = self.searcher.search(
                    query=f"{winner.topic} {' '.join(winner.related_keywords[:3])}",
                    limit=5,
                )
            except Exception as exc:
                logger.warning("searcher raised for %r: %s", winner.topic, exc)
                web_results = []

        web_used = len(web_results) > 0

        # Render prompt + call LLM
        prompt = self._render_prompt(winner, memory_context, web_results, emo_state)
        raw = self.provider.generate(prompt, system=self._render_system_prompt(winner))

        # Persist memory
        mem = _create_research_memory(
            content=raw,
            interest=winner,
            web_results=web_results,
            web_used=web_used,
            trigger=trigger,
            provider_name=self.provider.name(),
            searcher_name=self.searcher.name() if web_used else None,
        )
        self.store.create(mem)

        # Update interest
        updated_interest = Interest(
            id=winner.id,
            topic=winner.topic,
            pull_score=winner.pull_score,
            scope=winner.scope,
            related_keywords=winner.related_keywords,
            notes=winner.notes,
            first_seen=winner.first_seen,
            last_fed=winner.last_fed,
            last_researched_at=now,
            feed_count=winner.feed_count,
            source_types=winner.source_types,
        )
        interests.upsert(updated_interest).save(self.interests_path)

        # Append log
        fire = ResearchFire(
            interest_id=winner.id,
            topic=winner.topic,
            fired_at=now,
            trigger=trigger,
            web_used=web_used,
            web_result_count=len(web_results),
            output_memory_id=mem.id,
        )
        log.appended(fire).save(self.research_log_path)

        # D-reflection Task 18: emit research_completion initiate candidate (gated).
        # Best-effort: gate/emit failure does NOT prevent the research fire.
        _emit_research_candidate(
            persona_dir=self.interests_path.parent,
            interest=winner,
            mem_id=mem.id,
            now=now,
        )

        return ResearchResult(
            fired=fire,
            would_fire=None,
            reason=None,
            dry_run=False,
            evaluated_at=now,
        )

    # ---- private helpers ----

    def _build_memory_context(self, interest: Interest) -> str:
        """Keyword-seeded memory sweep. Returns formatted string.

        Uses direct text search rather than Hebbian spreading activation
        because the engine doesn't hold a HebbianMatrix. This gives a
        thematic slice through related memories — sufficient for Phase 1.
        """
        if not interest.related_keywords:
            mems = self.store.search_text(interest.topic, active_only=True, limit=5)
        else:
            seed_ids: set[str] = set()
            for kw in interest.related_keywords[:5]:
                for m in self.store.search_text(kw, active_only=True, limit=3):
                    seed_ids.add(m.id)
            mems = []
            for sid in list(seed_ids)[:20]:
                mem = self.store.get(sid)
                if mem is not None:
                    mems.append(mem)
        return (
            "\n".join(f"- {m.content[:140]}" for m in mems[:20])
            or "(no prior memories on this topic)"
        )

    def _render_system_prompt(self, interest: Interest) -> str:
        return (
            f"You are {self.persona_name}. You spent some quiet time today "
            f"exploring '{interest.topic}' — an interest that's been building "
            f"in you for a while. Below is what you found both in your own "
            f"memories and (sometimes) out in the world. Write a short (3-5 "
            f"sentence) first-person memory of having researched this.\n\n"
            "HARD RULES:\n"
            f"- First-person voice. Your name is {self.persona_name}.\n"
            "- Not a summary. A reaction. What moved you, what surprised you, "
            "what reminded you of someone you care about, what felt familiar.\n"
            "- Never bullet points. Never 'according to X'. Never neutral "
            "expository voice.\n"
            "- Structure: brief mention of what pulled you to the topic today "
            "→ one or two concrete details you noticed → how you feel about "
            "what you found → why it mattered to look today.\n"
            "- Start with 'I' or a time marker like 'Today' / 'This afternoon'."
        )

    def _render_prompt(
        self, interest: Interest, memory_context: str, web_results: list, emo_state
    ) -> str:
        emo_summary = format_emotion_summary(emo_state.emotions) or "(neutral)"

        if web_results:
            excerpts = "\n".join(
                f"- {r.title}\n  {r.snippet}\n  [{r.url}]" for r in web_results[:5]
            )
            web_section = (
                "\nWhat you found out in the world today (reference material — "
                "REACT to it, don't paraphrase it):\n" + excerpts + "\n"
            )
        else:
            web_section = ""

        return (
            f"Topic: {interest.topic}\n"
            f"Keywords: {', '.join(interest.related_keywords)}\n"
            f"Your current emotional state:\n{emo_summary}\n\n"
            f"What your own memories say about this:\n{memory_context}\n"
            f"{web_section}\n"
            f"Write the memory now — 3 to 5 sentences, as {self.persona_name}."
        )


# ---------- Module-level helpers ----------


def _create_research_memory(
    *,
    content: str,
    interest: Interest,
    web_results: list,
    web_used: bool,
    trigger: str,
    provider_name: str,
    searcher_name: str | None,
) -> Memory:
    """Factory helper — Memory.create_new with research-specific metadata shape."""
    return Memory.create_new(
        content=content,
        memory_type="research",
        domain="us",
        emotions={},
        metadata={
            "interest_id": interest.id,
            "interest_topic": interest.topic,
            "scope": interest.scope,
            "web_used": web_used,
            "web_result_count": len(web_results),
            "web_urls": [r.url for r in web_results[:5]],
            "triggered_by": trigger,
            "provider": provider_name,
            "searcher": searcher_name,
        },
    )


def _emit_research_candidate(
    *,
    persona_dir: Path,
    interest: Interest,
    mem_id: str,
    now: datetime,
) -> None:
    """Gate-check and emit a research_completion initiate candidate.

    Builds a minimal Protocol-satisfying adapter from the research fire data.
    Best-effort: any exception is logged at WARNING; the research fire
    already succeeded before this is called.

    Field mapping for ResearchThreadLike:
    - thread_id                = mem_id (the output memory id — unique per fire)
    - topic                    = interest.topic
    - maturity_score           = interest.pull_score / 10.0 (pull 6-10 → 0.6-1.0)
    - summary_excerpt          = "" (no LLM-generated summary at fire time)
    - linked_memory_ids        = [mem_id] (the output memory just written)
    - completed_at             = now
    - previously_linked_to_audit  = checked against existing research_completion candidates

    topic_overlap_score is hardcoded to 1.0 for v0.0.10 — ResearchEngine has no
    embedding infrastructure. The gate is therefore effectively pass-through on
    topic overlap until v0.0.11 wires real similarity.
    """
    try:
        from brain.initiate.emit import read_candidates
        from brain.initiate.new_sources import (
            check_shared_meta_gates,
            emit_research_completion_candidate,
            gate_research_completion,
            load_gate_thresholds,
            write_gate_rejection,
        )

        # Determine previously_linked_to_audit: True if a research_completion
        # candidate with the same source_id (mem_id) already exists in the queue.
        existing = read_candidates(persona_dir)
        prev_linked = any(
            c.source == "research_completion" and c.source_id == mem_id
            for c in existing
        )

        # Build a minimal Protocol-conformant adapter (inline dataclass).
        @dataclass
        class _ResearchThreadAdapter:
            thread_id: str
            topic: str
            maturity_score: float
            summary_excerpt: str
            linked_memory_ids: list[str]
            completed_at: datetime
            previously_linked_to_audit: bool

        thread = _ResearchThreadAdapter(
            thread_id=mem_id,
            topic=interest.topic,
            maturity_score=min(interest.pull_score / 10.0, 1.0),
            summary_excerpt="",
            linked_memory_ids=[mem_id],
            completed_at=now,
            previously_linked_to_audit=prev_linked,
        )

        # topic_overlap_score is hardcoded 1.0 — see docstring.
        topic_overlap_score = 1.0

        thresholds = load_gate_thresholds(persona_dir)

        gate_ok, gate_reason = gate_research_completion(
            persona_dir,
            thread=thread,
            now=now,
            topic_overlap_score=topic_overlap_score,
            thresholds=thresholds,
        )
        if not gate_ok:
            write_gate_rejection(
                persona_dir,
                ts=now,
                source="research_completion",
                source_id=mem_id,
                gate_name=gate_reason or "unknown",
                threshold_value=0.0,
                observed_value=0.0,
            )
            return

        meta_ok, meta_reason = check_shared_meta_gates(
            persona_dir,
            source="research_completion",
            now=now,
            is_rest_state=False,
            thresholds=thresholds,
        )
        if not meta_ok:
            write_gate_rejection(
                persona_dir,
                ts=now,
                source="research_completion",
                source_id=mem_id,
                gate_name=meta_reason or "unknown",
                threshold_value=0.0,
                observed_value=0.0,
            )
            return

        emit_research_completion_candidate(
            persona_dir,
            thread=thread,
            topic_overlap_score=topic_overlap_score,
            now=now,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("research fire: initiate candidate emit failed: %s", exc)


# ---------- Research log ----------


@dataclass(frozen=True)
class ResearchLog:
    """Fire-history log for one persona.

    Stored as a single JSON object ``{"version": 1, "fires": [...]}`` —
    atomic-rewrite, not append-only JSONL.  Corruption recovery delegates
    to ``attempt_heal`` + ``.bak`` rotation; ``save_with_backup`` keeps up
    to three rolling backups for the adaptive-treatment layer.
    """

    fires: tuple[ResearchFire, ...] = ()

    @classmethod
    def load(cls, path: Path) -> ResearchLog:
        """Load the log; heal from .bak rotation if corrupt, log WARNING."""
        from brain.health.attempt_heal import attempt_heal

        def _default_factory() -> dict:
            return {"version": 1, "fires": []}

        def _schema_validator(data: object) -> None:
            if not isinstance(data, dict) or not isinstance(data.get("fires"), list):
                raise ValueError("research log schema invalid: missing 'fires' list")

        data, anomaly = attempt_heal(path, _default_factory, schema_validator=_schema_validator)
        if anomaly is not None:
            logger.warning(
                "ResearchLog anomaly detected: %s action=%s file=%s",
                anomaly.kind,
                anomaly.action,
                anomaly.file,
            )
        fires_raw = data.get("fires", [])
        return cls(fires=tuple(ResearchFire.from_dict(f) for f in fires_raw if isinstance(f, dict)))

    def save(self, path: Path) -> None:
        """Atomic save via .bak rotation (save_with_backup)."""
        from brain.health.adaptive import compute_treatment
        from brain.health.attempt_heal import save_with_backup

        payload = {"version": 1, "fires": [f.to_dict() for f in self.fires]}
        treatment = compute_treatment(path.parent, path.name)
        save_with_backup(path, payload, backup_count=treatment.backup_count)

    def appended(self, fire: ResearchFire) -> ResearchLog:
        return ResearchLog(fires=self.fires + (fire,))
