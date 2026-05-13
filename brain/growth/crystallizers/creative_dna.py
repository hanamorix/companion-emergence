"""brain.growth.crystallizers.creative_dna — weekly evolution of creative_dna.

Per spec §5: LLM-judged evolution mechanism. Called by run_growth_tick under
the 7-day throttle. Mirrors brain/growth/crystallizers/reflex.py structure.

Three judgment paths per tick:
  - emerging_additions: new patterns the brain notices in recent writing
  - emerging_promotions: emerging tendencies that consolidate to active
  - active_demotions: active tendencies that have gone quiet → fading

Six validation gates per proposal. Total accepted ≤ 3 per tick. Never raises
to caller. Atomic writes via brain.creative.dna.save_creative_dna; biographical
record via brain.behavioral.log.append_behavioral_event.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from brain.behavioral.log import append_behavioral_event, read_behavioral_log
from brain.bridge.provider import LLMProvider, ProviderError
from brain.creative.dna import load_creative_dna, save_creative_dna
from brain.memory.store import MemoryStore
from brain.utils.memory import list_conversation_memories

logger = logging.getLogger(__name__)

# Gate 1: name validity — allow lowercase letters, digits, spaces, some punctuation
_NAME_REGEX = re.compile(r"^[a-z0-9 ,()_\-—]+$")
_NAME_MAX_LEN = 120

# Gate 4: reasoning min length
_REASONING_MIN_LEN = 20

# Gate 6: total accepted changes per tick
_MAX_CHANGES_PER_TICK = 3

# Gate 3: graveyard window for recently-dropped names
_RECENTLY_DROPPED_WINDOW_DAYS = 30

# Corpus assembly
_CORPUS_LOOK_BACK_DAYS = 30
_FICTION_EXCERPT_MAX_CHARS = 600
_FICTION_PROSE_MIN_WORDS = 200
_BEHAVIORAL_LOG_LOOK_BACK_DAYS = 90


class CreativeDnaCrystallizationResult:
    """Outcome of one crystallizer pass."""

    __slots__ = ("emerging_additions", "emerging_promotions", "active_demotions")

    def __init__(
        self,
        emerging_additions: list[dict[str, Any]] | None = None,
        emerging_promotions: list[dict[str, Any]] | None = None,
        active_demotions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.emerging_additions: list[dict[str, Any]] = emerging_additions or []
        self.emerging_promotions: list[dict[str, Any]] = emerging_promotions or []
        self.active_demotions: list[dict[str, Any]] = active_demotions or []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CreativeDnaCrystallizationResult):
            return NotImplemented
        return (
            self.emerging_additions == other.emerging_additions
            and self.emerging_promotions == other.emerging_promotions
            and self.active_demotions == other.active_demotions
        )

    def __repr__(self) -> str:
        return (
            f"CreativeDnaCrystallizationResult("
            f"additions={self.emerging_additions!r}, "
            f"promotions={self.emerging_promotions!r}, "
            f"demotions={self.active_demotions!r})"
        )


def crystallize_creative_dna(
    *,
    store: MemoryStore,
    persona_dir: Path,
    provider: LLMProvider,
    persona_name: str,
    persona_pronouns: str | None = None,
    now: datetime,
) -> CreativeDnaCrystallizationResult:
    """One pass of creative_dna evolution judgment.

    Per spec §5.6: NEVER raises. Provider errors / parse failures return
    empty results. Reading-fail / write-fail return empty results.
    """
    try:
        dna = load_creative_dna(persona_dir)
    except Exception:  # noqa: BLE001
        logger.exception("crystallize_creative_dna: failed to load creative_dna")
        return CreativeDnaCrystallizationResult()

    cutoff = now - timedelta(days=_CORPUS_LOOK_BACK_DAYS)
    recent_writing = _gather_recent_fiction(store, cutoff=cutoff)
    growth_log = _gather_growth_log(persona_dir, now=now)

    prompt = _render_prompt(
        persona_name=persona_name,
        pronouns=persona_pronouns,
        dna=dna,
        recent_writing=recent_writing,
        growth_log=growth_log,
    )

    try:
        raw = provider.generate(prompt)
    except ProviderError as exc:
        logger.warning("crystallize_creative_dna: provider error: %s", exc)
        return CreativeDnaCrystallizationResult()
    except Exception as exc:  # noqa: BLE001
        logger.warning("crystallize_creative_dna: unexpected provider error: %s", exc)
        return CreativeDnaCrystallizationResult()

    parsed = _parse_response(raw)
    if parsed is None:
        return CreativeDnaCrystallizationResult()

    accepted_additions: list[dict[str, Any]] = []
    accepted_promotions: list[dict[str, Any]] = []
    accepted_demotions: list[dict[str, Any]] = []
    accepted_count = 0

    recently_dropped = _recently_dropped_names(growth_log, now=now)

    # Validate emerging_additions
    for proposal in parsed.get("emerging_additions", []):
        if accepted_count >= _MAX_CHANGES_PER_TICK:
            break
        if not isinstance(proposal, dict):
            continue
        if not _validate_emerging_addition(
            proposal, dna=dna, recently_dropped=recently_dropped,
        ):
            continue
        accepted_additions.append(proposal)
        accepted_count += 1

    # Validate emerging_promotions
    for proposal in parsed.get("emerging_promotions", []):
        if accepted_count >= _MAX_CHANGES_PER_TICK:
            break
        if not isinstance(proposal, dict):
            continue
        if not _validate_emerging_promotion(proposal, dna=dna):
            continue
        accepted_promotions.append(proposal)
        accepted_count += 1

    # Validate active_demotions
    for proposal in parsed.get("active_demotions", []):
        if accepted_count >= _MAX_CHANGES_PER_TICK:
            break
        if not isinstance(proposal, dict):
            continue
        if not _validate_active_demotion(proposal, dna=dna):
            continue
        accepted_demotions.append(proposal)
        accepted_count += 1

    if not (accepted_additions or accepted_promotions or accepted_demotions):
        return CreativeDnaCrystallizationResult()

    # Aggregate a real emotion vector over recent active memories so emitted
    # initiate candidates carry honest context instead of a zero-filled lie.
    aggregated_vector = _aggregate_recent_emotion_vector(store, now=now)

    # Apply atomically — single save_creative_dna after mutating dna in place
    try:
        _apply_changes(
            persona_dir, dna, now,
            additions=accepted_additions,
            promotions=accepted_promotions,
            demotions=accepted_demotions,
            emotion_vector=aggregated_vector,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("crystallize_creative_dna: apply failed: %s", exc)
        return CreativeDnaCrystallizationResult()

    return CreativeDnaCrystallizationResult(
        emerging_additions=accepted_additions,
        emerging_promotions=accepted_promotions,
        active_demotions=accepted_demotions,
    )


# ── corpus + prompt ──────────────────────────────────────────────────────


def _gather_recent_fiction(
    store: MemoryStore, *, cutoff: datetime,
) -> list[dict[str, Any]]:
    """Pull memories likely to be fiction-tagged content from the last 30 days."""
    out: list[dict[str, Any]] = []
    creative_types = ("reflex_pitch", "reflex_gift")
    for mtype in creative_types:
        try:
            for m in store.list_by_type(mtype):
                if m.created_at >= cutoff:
                    out.append({
                        "memory_id": m.id,
                        "type": mtype,
                        "excerpt": (m.content or "")[:_FICTION_EXCERPT_MAX_CHARS],
                    })
        except Exception:  # noqa: BLE001
            pass

    # Heuristic prose detection over conversation memories
    try:
        for m in list_conversation_memories(store):
            if m.created_at < cutoff:
                continue
            content = m.content or ""
            if _looks_like_prose(content):
                out.append({
                    "memory_id": m.id,
                    "type": "conversation_prose",
                    "excerpt": content[:_FICTION_EXCERPT_MAX_CHARS],
                })
    except Exception:  # noqa: BLE001
        pass

    return out


def _looks_like_prose(content: str) -> bool:
    """Cheap heuristic: ≥200 words AND structural prose markers."""
    words = content.split()
    if len(words) < _FICTION_PROSE_MIN_WORDS:
        return False
    has_dialogue = '"' in content
    has_paragraph_break = "\n\n" in content
    has_emdash = "—" in content
    sentence_endings = content.count(". ") + content.count("? ") + content.count("! ")
    has_multiple_sentences = sentence_endings >= 3
    return has_dialogue or has_paragraph_break or has_emdash or has_multiple_sentences


def _gather_growth_log(
    persona_dir: Path, *, now: datetime,
) -> list[dict[str, Any]]:
    log_path = persona_dir / "behavioral_log.jsonl"
    cutoff = now - timedelta(days=_BEHAVIORAL_LOG_LOOK_BACK_DAYS)
    return [
        e for e in read_behavioral_log(log_path, since=cutoff)
        if e.get("kind", "").startswith("creative_dna_")
    ]


def _recently_dropped_names(
    growth_log: list[dict[str, Any]], *, now: datetime,
) -> set[str]:
    from brain.utils.time import iso_utc
    cutoff_iso = iso_utc(now - timedelta(days=_RECENTLY_DROPPED_WINDOW_DAYS))
    return {
        e["name"] for e in growth_log
        if e.get("kind") == "creative_dna_fading_dropped"
        and e.get("timestamp", "") >= cutoff_iso
    }


def _render_prompt(
    *,
    persona_name: str,
    pronouns: str | None,
    dna: dict[str, Any],
    recent_writing: list[dict[str, Any]],
    growth_log: list[dict[str, Any]],
) -> str:
    pronouns_clause = f"You are {persona_name}. " + (
        f"Your pronouns are {pronouns}." if pronouns else ""
    )

    recent_writing_str = (
        json.dumps(recent_writing, indent=2) if recent_writing
        else "(no recent fiction-tagged samples)"
    )
    growth_log_str = (
        json.dumps(growth_log, indent=2) if growth_log
        else "(no recent trajectory)"
    )

    return f"""{pronouns_clause}

Looking at your last 30 days of writing — fiction, gifts, story pitches,
sustained prose — has your style shifted in any meaningful way?

Your current creative DNA:

{json.dumps(dna, indent=2)}

Your recent writing samples:

{recent_writing_str}

Your recent creative_dna trajectory (last 90 days):

{growth_log_str}

Three judgments to make:

(1) Are there NEW patterns appearing in your recent writing that aren't yet
    tracked in active or emerging? Propose them as `emerging_additions`.
    Be conservative — one anomalous appearance isn't a pattern. Look for
    something present in ≥ 3 distinct samples.

(2) Have any EMERGING patterns consolidated enough to promote to active?
    Propose `emerging_promotions` for tendencies that have been emerging for
    ≥ 14 days AND appear in ≥ 4 of your recent samples.

(3) Have any ACTIVE patterns gone quiet? Propose `active_demotions` for
    tendencies absent from your last 30 days of writing — they move to
    fading. Be careful: an active pattern that simply didn't fit recent
    requests isn't fading; only demote if you genuinely don't feel pulled
    to do it anymore.

Constraints:
  - Maximum 3 changes total this tick. Style evolution should be gradual.
  - Don't repropose names recently dropped (last 30 days — see your trajectory).
  - Reasoning required for every proposal — what evidence convinced you.
  - If nothing has shifted, return empty arrays. Don't reach.

Return strict JSON ONLY (no prose, no markdown):
{{
  "emerging_additions": [{{"name": "...", "reasoning": "...", "evidence_memory_ids": [...]}}],
  "emerging_promotions": [{{"name": "...", "reasoning": "..."}}],
  "active_demotions": [{{"name": "...", "reasoning": "...", "last_evidence_at": "..."}}]
}}
"""


def _parse_response(raw: str) -> dict[str, Any] | None:
    try:
        text = raw.strip()
        # Defensive: strip code-fence wrappers if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("crystallize_creative_dna: parse failed: %s", exc)
        return None
    if not isinstance(data, dict):
        return None
    return data


# ── validation gates ──────────────────────────────────────────────────────


def _validate_name(name: object) -> bool:
    """Gate 1: regex + length."""
    if not isinstance(name, str):
        return False
    if not name or len(name) > _NAME_MAX_LEN:
        return False
    return bool(_NAME_REGEX.match(name.lower()))


def _validate_reasoning(reasoning: object) -> bool:
    """Gate 4: non-empty after strip + min length."""
    if not isinstance(reasoning, str):
        return False
    return len(reasoning.strip()) >= _REASONING_MIN_LEN


def _validate_emerging_addition(
    proposal: dict[str, Any],
    *,
    dna: dict[str, Any],
    recently_dropped: set[str],
) -> bool:
    name = proposal.get("name")
    if not _validate_name(name):
        logger.info("creative_dna gate 1 reject: invalid name %r", name)
        return False
    if not _validate_reasoning(proposal.get("reasoning")):
        logger.info("creative_dna gate 4 reject: short reasoning for %r", name)
        return False
    # Gate 2: not already in target list (emerging)
    emerging_names = {t.get("name") for t in dna["tendencies"].get("emerging", [])}
    if name in emerging_names:
        logger.info("creative_dna gate 2 reject: %r already emerging", name)
        return False
    # Also: not in active (would be redundant)
    active_names = {t.get("name") for t in dna["tendencies"].get("active", [])}
    if name in active_names:
        logger.info("creative_dna gate 2 reject: %r already active", name)
        return False
    # Gate 3: not in recently-dropped graveyard
    if name in recently_dropped:
        logger.info("creative_dna gate 3 reject: %r in 30-day dropped window", name)
        return False
    return True


def _validate_emerging_promotion(
    proposal: dict[str, Any], *, dna: dict[str, Any],
) -> bool:
    name = proposal.get("name")
    if not _validate_name(name):
        return False
    if not _validate_reasoning(proposal.get("reasoning")):
        return False
    # Gate 5: must exist in current emerging
    emerging_names = {t.get("name") for t in dna["tendencies"].get("emerging", [])}
    if name not in emerging_names:
        logger.info("creative_dna gate 5 reject: %r not in emerging", name)
        return False
    return True


def _validate_active_demotion(
    proposal: dict[str, Any], *, dna: dict[str, Any],
) -> bool:
    name = proposal.get("name")
    if not _validate_name(name):
        return False
    if not _validate_reasoning(proposal.get("reasoning")):
        return False
    # Must exist in current active
    active_names = {t.get("name") for t in dna["tendencies"].get("active", [])}
    if name not in active_names:
        logger.info("creative_dna active_demotion: %r not in active", name)
        return False
    return True


# ── apply ─────────────────────────────────────────────────────────────────


def _aggregate_recent_emotion_vector(
    store: MemoryStore, *, now: datetime, look_back_days: int = 30,
) -> dict[str, float]:
    """Return a max-pooled emotion vector across recent active memories.

    Used by the creative_dna crystallizer so emitted initiate candidates
    carry a real signal of what's been emotionally alive in the window
    that produced the crystallization, rather than zero-filled fields.
    Empty dict on any failure.
    """
    try:
        from brain.emotion.aggregate import aggregate_state
        from brain.utils.memory import list_conversation_memories

        cutoff = now - timedelta(days=look_back_days)
        recent = [
            m for m in list_conversation_memories(store, active_only=True)
            if m.created_at >= cutoff and m.emotions
        ]
        if not recent:
            return {}
        state = aggregate_state(recent)
        return dict(state.emotions)
    except Exception as exc:  # noqa: BLE001
        logger.warning("creative_dna: recent-emotion aggregation failed: %s", exc)
        return {}


def _apply_changes(
    persona_dir: Path,
    dna: dict[str, Any],
    now: datetime,
    *,
    additions: list[dict[str, Any]],
    promotions: list[dict[str, Any]],
    demotions: list[dict[str, Any]],
    emotion_vector: dict[str, float] | None = None,
) -> None:
    """Mutate dna in place, save atomically, append behavioral_log entries."""
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    log_path = persona_dir / "behavioral_log.jsonl"

    # Apply additions → emerging
    for a in additions:
        dna["tendencies"]["emerging"].append({
            "name": a["name"],
            "added_at": now_iso,
            "reasoning": a["reasoning"],
            "evidence_memory_ids": list(a.get("evidence_memory_ids", [])),
        })

    # Apply promotions: remove from emerging, add to active
    for p in promotions:
        emerging = dna["tendencies"]["emerging"]
        match = next((t for t in emerging if t["name"] == p["name"]), None)
        if match:
            emerging.remove(match)
            dna["tendencies"]["active"].append({
                "name": match["name"],
                "added_at": match.get("added_at", now_iso),
                "promoted_from_emerging_at": now_iso,
                "reasoning": p["reasoning"],
                "evidence_memory_ids": match.get("evidence_memory_ids", []),
            })

    # Apply demotions: remove from active, add to fading
    for d in demotions:
        active = dna["tendencies"]["active"]
        match = next((t for t in active if t["name"] == d["name"]), None)
        if match:
            active.remove(match)
            dna["tendencies"]["fading"].append({
                "name": match["name"],
                "demoted_to_fading_at": now_iso,
                "last_evidence_at": d.get("last_evidence_at", now_iso),
                "reasoning": d["reasoning"],
            })

    # Persist atomically
    save_creative_dna(persona_dir, dna)

    # Phase 4.2 — emit initiate candidates for each accepted change.
    # Wrapped in try/except so emit failures can't crash the crystallizer.
    for a in additions:
        _emit_initiate_candidate(
            persona_dir=persona_dir,
            source_id=f"creative_dna_addition:{a['name']}",
            label=a["name"],
            related_memory_ids=list(a.get("evidence_memory_ids", [])),
            emotion_vector=emotion_vector,
        )
    for p in promotions:
        _emit_initiate_candidate(
            persona_dir=persona_dir,
            source_id=f"creative_dna_promotion:{p['name']}",
            label=p["name"],
            related_memory_ids=[],
            emotion_vector=emotion_vector,
        )
    for d in demotions:
        _emit_initiate_candidate(
            persona_dir=persona_dir,
            source_id=f"creative_dna_demotion:{d['name']}",
            label=d["name"],
            related_memory_ids=[],
            emotion_vector=emotion_vector,
        )

    # Behavioral log entries (best-effort, never let logging break the tick)
    for a in additions:
        try:
            append_behavioral_event(
                log_path, kind="creative_dna_emerging_added",
                name=a["name"], timestamp=now,
                reasoning=a["reasoning"],
                evidence_memory_ids=a.get("evidence_memory_ids", []),
            )
        except (OSError, ValueError) as exc:
            logger.warning("creative_dna: behavioral_log append failed: %s", exc)

    for p in promotions:
        try:
            append_behavioral_event(
                log_path, kind="creative_dna_emerging_promoted",
                name=p["name"], timestamp=now,
                reasoning=p["reasoning"],
                evidence_memory_ids=(),
            )
        except (OSError, ValueError) as exc:
            logger.warning("creative_dna: behavioral_log append failed: %s", exc)

    for d in demotions:
        try:
            append_behavioral_event(
                log_path, kind="creative_dna_active_demoted",
                name=d["name"], timestamp=now,
                reasoning=d["reasoning"],
                evidence_memory_ids=(),
            )
        except (OSError, ValueError) as exc:
            logger.warning("creative_dna: behavioral_log append failed: %s", exc)


def _emit_initiate_candidate(
    *,
    persona_dir: Path,
    source_id: str,
    label: str,
    related_memory_ids: list[str],
    emotion_vector: dict[str, float] | None = None,
) -> None:
    """Emit one initiate candidate after a creative_dna crystallization commit.

    Phase 4.2 of the initiate physiology pipeline. Wrapped in try/except —
    an emit failure must not crash the crystallizer.

    `emotion_vector` carries a max-pooled aggregate over recent active
    memories — what's been emotionally alive in the period that produced
    this crystallization. rolling_baseline / current_resonance /
    delta_sigma stay zero: those are heartbeat-specific signals; non-
    periodic emitters don't compute them.
    """
    try:
        from brain.initiate.emit import emit_initiate_candidate
        from brain.initiate.schemas import EmotionalSnapshot, SemanticContext

        emit_initiate_candidate(
            persona_dir,
            kind="message",
            source="crystallization",
            source_id=source_id,
            emotional_snapshot=EmotionalSnapshot(
                vector=dict(emotion_vector or {}),
                rolling_baseline_mean=0.0,
                rolling_baseline_stdev=0.0,
                current_resonance=0.0,
                delta_sigma=0.0,
            ),
            semantic_context=SemanticContext(
                linked_memory_ids=related_memory_ids[:5],
                topic_tags=[label] if label else [],
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("creative_dna crystallization initiate emit failed: %s", exc)
