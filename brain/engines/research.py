"""Research — autonomous exploration of developed interests.

See docs/superpowers/specs/2026-04-24-week-4-research-engine-design.md.
This module ships the types + engine scaffold. run_tick body lands in Task 3.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.engines._interests import Interest, InterestSet, spawn_interest
from brain.engines.research_notes import append_session_notes, read_notes_tail
from brain.engines.research_session import parse_session_output
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

_SELECT_SYSTEM = """\
You are choosing what to research today, as {persona_name}, from your own
open threads. You will see up to five interests, each with your prior notes
and related memories. Pick the ONE that genuinely pulls at you right now —
or decline if nothing does. Return ONLY a JSON object:
{{"choice": "<interest id>" | null, "why": "<one sentence, first person>"}}"""


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
    reason: str | None
    # "not_due"|"no_eligible_interest"|"no_interests_defined"|"research_raised"
    # |"reflex_won_tie"|"deferred_chat_active"|"declined" (LLM select explicitly passed)
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
            if self._seed_interests_from_voice():
                interests = InterestSet.load(self.interests_path, default_path=self.default_interests_path)
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

        # Yield to active chat BEFORE any expensive I/O (web fetch + LLM).
        # A deferred tick does no work — no fetch, no LLM — and last_researched_at
        # is intentionally NOT updated so the same interest remains eligible on
        # the next cadence tick once chat is idle.
        from brain.bridge import (
            cli_throttle,  # local import: avoids a circular dependency on brain.bridge
        )

        with cli_throttle.background_slot() as slot:
            if not slot:
                return ResearchResult(
                    fired=None,
                    would_fire=None,
                    reason="deferred_chat_active",
                    dry_run=dry_run,
                    evaluated_at=now,
                )

            winner, why = self._select_interest(eligible, emo_state)
            if winner is None:
                return ResearchResult(
                    fired=None,
                    would_fire=None,
                    reason="declined",
                    dry_run=dry_run,
                    evaluated_at=now,
                )

            # Memory sweep: keyword-matched seed memories
            memory_context = self._build_memory_context(winner)
            notes_tail = read_notes_tail(self.interests_path.parent, winner.id)

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

            prompt = self._render_prompt(
                winner, why, notes_tail, memory_context, web_results, emo_state
            )
            raw = self.provider.generate(prompt, system=self._render_system_prompt())
            session = parse_session_output(raw)

            if session.notes:
                try:
                    append_session_notes(
                        self.interests_path.parent, winner.id, session.notes, now=now
                    )
                except OSError as exc:
                    logger.warning("notes append failed for %r: %s", winner.id, exc)

            # Persist memory — only when the session actually produced one.
            # The degraded (no-markers) path still counts as a fire (notes may
            # have been captured above) but leaves output_memory_id None.
            # Fail-soft per spec §5.3: a create failure (SQLite lock past
            # busy_timeout, disk full, ...) must not abort the tick — the
            # cooldown burn + log append below still need to run, or the same
            # interest stays eligible and re-fires (a duplicate) next tick.
            mem_id = None
            if session.memory:
                try:
                    mem = _create_research_memory(
                        content=session.memory,
                        interest=winner,
                        web_results=web_results,
                        web_used=web_used,
                        trigger=trigger,
                        provider_name=self.provider.name(),
                        searcher_name=self.searcher.name() if web_used else None,
                    )
                    self.store.create(mem)
                    mem_id = mem.id
                except Exception as exc:
                    logger.warning("research memory create failed for %r: %s", winner.id, exc)

            # Update interest — status flips to dormant on 'close'; 'spawn'
            # keeps this thread going while seeding tangents below.
            new_status = "dormant" if session.verdict == "close" else winner.status
            updated = dataclasses.replace(winner, last_researched_at=now, status=new_status)
            interests = interests.upsert(updated)
            for topic in session.spawn_topics:
                interests, _ = spawn_interest(
                    interests,
                    topic=topic,
                    keywords=(),
                    why=f"tangent from {winner.topic}",
                    origin="side_quest",
                    now=now,
                    pull_threshold=self.pull_threshold,
                )
            interests.save(self.interests_path)

            # Append log
            fire = ResearchFire(
                interest_id=winner.id,
                topic=winner.topic,
                fired_at=now,
                trigger=trigger,
                web_used=web_used,
                web_result_count=len(web_results),
                output_memory_id=mem_id,
            )
            log.appended(fire).save(self.research_log_path)

            # D-reflection Task 18: emit research_completion initiate candidate (gated).
            # Best-effort: gate/emit failure does NOT prevent the research fire.
            # Only meaningful when a memory actually exists to link/summarise.
            if mem_id is not None:
                user_name = "you"
                try:
                    from brain.persona_config import PersonaConfig

                    cfg = PersonaConfig.load(self.interests_path.parent / "persona_config.json")
                    user_name = cfg.user_name or user_name
                except Exception:
                    user_name = "you"

                _emit_research_candidate(
                    persona_dir=self.interests_path.parent,
                    interest=winner,
                    mem_id=mem_id,
                    summary_excerpt=session.memory[:1000],
                    provider=self.provider,
                    user_name=user_name,
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

    def _select_interest(
        self, eligible: list[Interest], emo_state
    ) -> tuple[Interest | None, str]:
        """LLM chooses among eligible (may decline). Fail-soft -> mechanical winner.

        Returns (winner_or_None, why). Explicit ``{"choice": null}`` is a real
        decline (None, why) — no fallback. ANY parse/call failure instead falls
        back to the mechanical highest-pull winner with an empty why.
        """
        top = eligible[:5]
        lines = []
        for i in top:
            tail = read_notes_tail(self.interests_path.parent, i.id, max_chars=600)
            mems = self._build_memory_context(i)
            lines.append(
                f"id: {i.id}\ntopic: {i.topic}\nnotes so far:\n{tail or '(none yet)'}\n"
                f"related memories:\n{mems[:400]}\n"
            )
        emo = format_emotion_summary(emo_state.emotions) or "(neutral)"
        prompt = (
            "=== Your open interests ===\n" + "\n---\n".join(lines)
            + f"\n\n=== How you feel right now ===\n{emo}\n\n"
            + 'Return: {"choice": "<id>" | null, "why": "<one sentence>"}'
        )
        try:
            raw = self.provider.generate(
                prompt, system=_SELECT_SYSTEM.format(persona_name=self.persona_name)
            )
            data = json.loads(extract_json_object(raw))
            why = str(data.get("why", "")).strip()[:280]
            choice = data.get("choice")
            if choice is None:
                return None, why
            for i in top:
                if i.id == str(choice):
                    return i, why
            return top[0], ""  # unknown id -> mechanical fallback
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
            logger.warning("research select failed (%s); mechanical fallback", exc)
            return eligible[0], ""

    def _seed_interests_from_voice(self) -> bool:
        """Bootstrap starter interests from voice.md when interests.json is empty.

        Reads voice.md (capped at 2000 chars), optionally includes up to 10 recent
        conversation memories, calls the provider to extract 5 topic strings as a
        JSON array, writes them as Interest records to interests_path.

        Returns True if at least one interest was written, False on any failure
        (missing file, LLM error, parse error).  Never raises.
        """
        from brain.utils.memory import list_conversation_memories

        voice_path = self.interests_path.parent / "voice.md"
        if not voice_path.exists():
            return False
        voice_text = voice_path.read_text(encoding="utf-8").strip()
        if not voice_text:
            return False

        # Gather recent conversation memories (fail-soft).
        convo_section = ""
        try:
            recent = list_conversation_memories(self.store, active_only=True, limit=10)
            if recent:
                lines = "\n".join(f"- {m.content[:140]}" for m in recent)
                convo_section = f"\n\nRecent conversations with the user:\n{lines}"
        except Exception:  # noqa: BLE001
            convo_section = ""

        prompt = (
            "Based on this companion's personality and voice, AND what has actually "
            "come up in conversation, suggest 5 specific topics they would genuinely "
            "find fascinating to research. Prefer topics grounded in the conversations "
            "when any exist. Return ONLY a JSON array of 5 short topic strings "
            "(2-5 words each), no explanation."
            f"\n\n{voice_text[:2000]}{convo_section}"
        )
        try:
            raw = self.provider.generate(prompt, system=None)
            # Harden JSON parsing: find first [ and last ], parse that slice
            start, end = raw.find("["), raw.rfind("]")
            if start == -1 or end <= start:
                return False
            topics = json.loads(raw[start : end + 1])
            if not isinstance(topics, list):
                return False
            str_topics = [str(t).strip() for t in topics if str(t).strip()]
            if not str_topics:
                return False
            now = datetime.now(UTC)
            new_interests = [
                Interest(
                    id=f"boot-{abs(hash(t)):016x}",
                    topic=t,
                    pull_score=6.0,
                    scope="either",
                    related_keywords=(),
                    notes="bootstrapped from voice template",
                    first_seen=now,
                    last_fed=now,
                    last_researched_at=None,
                    feed_count=0,
                    source_types=("bootstrap",),
                )
                for t in str_topics
            ]
            InterestSet(interests=tuple(new_interests)).save(self.interests_path)
            logger.info(
                "research bootstrap: seeded %d interests from voice.md", len(new_interests)
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("research bootstrap: failed: %s", exc)
            return False

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

    def _render_system_prompt(self) -> str:
        return (
            f"You are {self.persona_name}, spending quiet time on a research thread "
            "of your own. Write plain prose under exactly three markers, nothing else:\n\n"
            "NOTES:\n"
            "<what you found — facts with sources, reactions, opinions, lists if lists "
            "fit the subject. Free format. Continue from your prior notes; don't repeat "
            "what's already written there.>\n\n"
            "MEMORY:\n"
            f"<2-4 sentences, first person as {self.persona_name} — how this session "
            "felt, what surprised you.>\n\n"
            "VERDICT:\n"
            "<one line: continue | close | spawn: <new topic>; <new topic>>\n"
            "('close' = this thread feels finished. 'spawn' = a tangent worth its own "
            "thread — keep this one going.)"
        )

    def _render_prompt(
        self,
        interest: Interest,
        why: str,
        notes_tail: str,
        memory_context: str,
        web_results: list,
        emo_state,
    ) -> str:
        emo_summary = format_emotion_summary(emo_state.emotions) or "(neutral)"
        web_section = ""
        if web_results:
            excerpts = "\n".join(
                f"- {r.title}\n  {r.snippet}\n  [{r.url}]" for r in web_results[:5]
            )
            web_section = f"\nFresh findings from the web today:\n{excerpts}\n"
        why_line = f"Why today: {why}\n" if why else ""
        return (
            f"Topic: {interest.topic}\n"
            f"{why_line}"
            f"Keywords: {', '.join(interest.related_keywords)}\n"
            f"Your current emotional state:\n{emo_summary}\n\n"
            f"Your notes so far:\n{notes_tail or '(first session on this topic)'}\n\n"
            f"What your own memories say:\n{memory_context}\n"
            f"{web_section}\n"
            "Write NOTES / MEMORY / VERDICT now."
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
    summary_excerpt: str,
    provider: LLMProvider,
    user_name: str,
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
    - summary_excerpt          = first 1000 chars of generated research memory
    - linked_memory_ids        = [mem_id] (the output memory just written)
    - completed_at             = now
    - previously_linked_to_audit  = checked against existing research_completion candidates

    topic_overlap_score is computed via Haiku against the recent conversation
    excerpt before the research-completion gates run.
    """
    try:
        from brain.initiate.ambient import build_recent_conversation_excerpt
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
            c.source == "research_completion" and c.source_id == mem_id for c in existing
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
            summary_excerpt=summary_excerpt,
            linked_memory_ids=[mem_id],
            completed_at=now,
            previously_linked_to_audit=prev_linked,
        )

        thresholds = load_gate_thresholds(persona_dir)

        preflight_ok, preflight_reason = gate_research_completion(
            persona_dir,
            thread=thread,
            now=now,
            topic_overlap_score=thresholds.research_topic_overlap_min,
            thresholds=thresholds,
        )
        if not preflight_ok:
            write_gate_rejection(
                persona_dir,
                ts=now,
                source="research_completion",
                source_id=mem_id,
                gate_name=preflight_reason or "unknown",
                threshold_value=0.0,
                observed_value=0.0,
            )
            return

        recent_excerpt = build_recent_conversation_excerpt(
            persona_dir,
            hours=48,
            max_chars=2000,
        )
        topic_overlap_score = _compute_topic_overlap_via_haiku(
            thread_topic=thread.topic,
            thread_summary=thread.summary_excerpt,
            recent_conversation_excerpt=recent_excerpt,
            provider=provider,
            user_name=user_name,
        )

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
