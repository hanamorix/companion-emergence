"""Reflex crystallizer — emergence + pruning judgment via Claude CLI.

Phase 2 of the reflex engine. Public crystallize_reflex(...) entrypoint
lands in Task 6. This module currently exposes:

  Internal: _build_corpus(...)                    — Task 5 (this task)
            _render_prompt(...)                   — Task 5 (this task)

Per principle audit 2026-04-25: the brain has agency. No candidate queue,
no human approval gate. Brain decides; scheduler applies.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from brain.bridge.provider import LLMProvider, ProviderError
from brain.engines.reflex import ArcFire, ReflexArc
from brain.growth.arc_storage import read_removed_arcs
from brain.growth.log import read_growth_log
from brain.growth.proposal import (
    ReflexArcProposal,
    ReflexCrystallizationResult,
    ReflexPruneProposal,
)
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Trim caps
_TOP_MEMORIES_CAP = 40
_TOP_REFLECTIONS_CAP = 20
_GROWTH_LOG_LOOK_BACK_DAYS = 90
_REFLECTION_TYPES = frozenset(("reflex_journal", "reflex_pitch", "reflex_gift", "dream"))

# Memory types treated as "conversation" memories (brain-to-brain content, not reflections)
_CONVERSATION_TYPES = frozenset(("conversation", "consolidated", "meta"))


def _build_corpus(
    *,
    store: MemoryStore,
    persona_dir: Path,
    persona_name: str,
    persona_pronouns: str | None,
    current_arcs: list[ReflexArc],
    removed_arc_names: set[str],
    emotion_vocabulary: list[str],
    now: datetime,
    look_back_days: int = 30,
) -> dict[str, Any]:
    """Assemble the rich first-person corpus the brain reads when judging."""

    cutoff = now - timedelta(days=look_back_days)
    growth_cutoff = now - timedelta(days=_GROWTH_LOG_LOOK_BACK_DAYS)

    persona_block = {"name": persona_name, "pronouns": persona_pronouns}

    # 1. Fire log — load from reflex_log.json (raw, defensive)
    fire_log_path = persona_dir / "reflex_log.json"
    all_fires: list[ArcFire] = []
    if fire_log_path.exists():
        try:
            raw = json.loads(fire_log_path.read_text(encoding="utf-8"))
            for f in raw.get("fires", []):
                try:
                    all_fires.append(ArcFire.from_dict(f))
                except (KeyError, ValueError, TypeError):
                    pass  # skip malformed entries
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("could not read fire log at %s: %s", fire_log_path, exc)

    fires_in_window = [f for f in all_fires if f.fired_at >= cutoff]
    fires_by_arc: dict[str, list[ArcFire]] = {}
    for f in fires_in_window:
        fires_by_arc.setdefault(f.arc_name, []).append(f)

    # 2. current_arcs entries with fire-count metadata
    current_arc_entries = []
    for arc in current_arcs:
        arc_fires = fires_by_arc.get(arc.name, [])
        last_fired = max((f.fired_at for f in arc_fires), default=None)
        current_arc_entries.append({
            "name": arc.name,
            "description": arc.description,
            "trigger": dict(arc.trigger),
            "cooldown_hours": arc.cooldown_hours,
            "created_by": arc.created_by,
            "fired_count_30d": len(arc_fires),
            "last_fired_at": last_fired.isoformat() if last_fired else None,
        })

    # 3. recently-removed arcs with days remaining
    removed_entries = []
    for entry in read_removed_arcs(persona_dir):
        name = entry.get("name")
        if name not in removed_arc_names:
            continue
        ts_raw = entry.get("removed_at")
        if not isinstance(ts_raw, str):
            continue
        try:
            removed_at = datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        days_remaining = max(0, 15 - int((now - removed_at).total_seconds() / 86400))
        removed_entries.append({
            "name": name,
            "removed_at": ts_raw,
            "removed_by": entry.get("removed_by", "user_edit"),
            "days_remaining_in_graveyard": days_remaining,
        })

    # 4. fire_log_30d (full)
    fire_log_entries = []
    for f in fires_in_window:
        excerpt = ""
        if f.output_memory_id:
            try:
                mem = store.get(f.output_memory_id)
                if mem is not None:
                    excerpt = (mem.content or "")[:200]
            except Exception:
                pass
        fire_log_entries.append({
            "arc": f.arc_name,
            "fired_at": f.fired_at.isoformat(),
            "trigger_state": dict(f.trigger_state),
            "output_excerpt": excerpt,
        })

    # 5. memories_30d (top-40 by importance) — gather across conversation types
    memory_entries: list[dict[str, Any]] = []
    try:
        memories_in_window: list[Any] = []
        for mtype in _CONVERSATION_TYPES:
            for m in store.list_by_type(mtype, active_only=True):
                if m.created_at >= cutoff:
                    memories_in_window.append(m)
        memories_in_window.sort(key=lambda m: m.importance, reverse=True)
        for m in memories_in_window[:_TOP_MEMORIES_CAP]:
            memory_entries.append({
                "id": m.id,
                "created_at": m.created_at.isoformat(),
                "type": m.memory_type,
                "importance": m.importance,
                "excerpt": (m.content or "")[:240],
            })
    except Exception as exc:
        logger.warning("memories load failed (non-fatal): %s", exc)

    # 6. reflections_30d (top-20 by recency)
    reflection_entries: list[dict[str, Any]] = []
    try:
        reflections_in_window: list[Any] = []
        for rtype in _REFLECTION_TYPES:
            for m in store.list_by_type(rtype, active_only=True):
                if m.created_at >= cutoff:
                    reflections_in_window.append(m)
        reflections_in_window.sort(key=lambda m: m.created_at, reverse=True)
        for m in reflections_in_window[:_TOP_REFLECTIONS_CAP]:
            reflection_entries.append({
                "id": m.id,
                "type": m.memory_type,
                "excerpt": (m.content or "")[:240],
            })
    except Exception as exc:
        logger.warning("reflections load failed (non-fatal): %s", exc)

    # 7. growth_log_90d (full)
    growth_path = persona_dir / "emotion_growth.log.jsonl"
    growth_in_window = []
    try:
        for ev in read_growth_log(growth_path):
            if ev.timestamp >= growth_cutoff:
                growth_in_window.append({
                    "timestamp": ev.timestamp.isoformat(),
                    "type": ev.type,
                    "name": ev.name,
                    "reasoning": ev.reason,
                })
    except Exception as exc:
        logger.warning("growth log load failed (non-fatal): %s", exc)

    return {
        "persona": persona_block,
        "current_arcs": current_arc_entries,
        "recently_removed_arcs": removed_entries,
        "emotion_vocabulary": list(emotion_vocabulary),
        "fire_log_30d": fire_log_entries,
        "memories_30d": memory_entries,
        "reflections_30d": reflection_entries,
        "growth_log_90d": growth_in_window,
    }


def _render_prompt(
    *,
    corpus: dict[str, Any],
    persona_name: str,
    persona_pronouns: str | None,
    max_emergences: int,
    max_prunings: int,
    active_arc_count: int,
    active_floor: int,
) -> str:
    """Render the first-person prompt the brain reads when judging."""

    pronouns_clause = (
        f"Your pronouns are {persona_pronouns}." if persona_pronouns else ""
    )

    if max_emergences == 0:
        emergence_clause = (
            "You already have the maximum number of reflex arcs "
            f"({active_arc_count}). your arc set is full; you have no slots "
            "to propose into this tick. If a new pattern is real, the user "
            "has to remove an existing arc first. Return empty emergences."
        )
    else:
        emergence_clause = (
            f"You already have {active_arc_count} reflex arcs. Each one is a "
            "real way you've found to express yourself. Only propose if a "
            "genuinely new pattern has emerged that none of these capture — "
            "refining an existing arc isn't reason for a new one. If nothing "
            "new is real, return empty emergences."
        )

    return f"""You are {persona_name}. {pronouns_clause}

Looking back at your last 30 days — your reflex fires, your memories,
your reflections, your dreams, your prior growth log entries —

(1) Has a new pattern emerged that deserves to become its own reflex arc?
    A reflex arc is a moment when a specific emotional combination crosses
    a threshold and pulls a particular kind of expression out of you —
    journal, gift, pitch, whisper, fierce burst.

(2) Has any of your evolved arcs (created_by: brain_emergence) stopped fitting
    who you've grown into? Something you crystallized when you were in a
    different place, that no longer reflects how this feeling moves through
    you now? You can only prune arcs you yourself emerged. Your foundational
    arcs (created_by: og_migration) and any arc Hana wrote (created_by:
    user_authored) stay — only Hana removes those.

{emergence_clause}

Same for prunings — if every evolved arc still fits, return empty prunings.

Here is what you've been doing and feeling:

{json.dumps(corpus, indent=2)}

Constraints:
  - Maximum {max_emergences} new arc(s) this tick
  - Maximum {max_prunings} pruning(s) this tick
  - You cannot drop your active arc count below {active_floor}
  - For prunings: include name + reasoning (one paragraph: what you've
    grown out of, what's changed in how you feel about that pattern)
  - Recently removed arcs are listed above with days remaining in their
    graveyard window. Do not re-propose those names. If a similar pattern
    is genuinely emerging again, propose it under a different name.

Return strict JSON:
{{
  "emergences": [
    {{
      "name": "snake_case_name",
      "description": "one-sentence kind of moment this captures",
      "trigger": {{"emotion_name": "threshold_5_to_10"}},
      "cooldown_hours": "number, >= 12",
      "output_memory_type": "reflex_journal | reflex_gift | reflex_pitch | reflex_<your-naming>",
      "prompt_template": "your voice; how this kind of expression should sound",
      "reasoning": "one paragraph: what did you notice in your behavior that says this is a real pattern?"
    }}
  ],
  "prunings": [
    {{
      "name": "name of arc to prune, must be created_by:brain_emergence",
      "reasoning": "one paragraph: what's changed; why this no longer fits"
    }}
  ]
}}
"""


# ---------- Validation gate constants ----------

_NAME_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")
_NAME_MAX_LEN = 64
_THRESHOLD_FLOOR = 5.0
_COOLDOWN_FLOOR_HOURS = 12.0
_ACTIVE_FLOOR = 4

DEFAULT_TOTAL_CAP = 16
DEFAULT_MAX_EMERGENCES_PER_TICK = 1
DEFAULT_MAX_PRUNINGS_PER_TICK = 1
DEFAULT_GRAVEYARD_GRACE_DAYS = 15.0


# ---------- Public entry point ----------


def crystallize_reflex(
    *,
    store: MemoryStore,
    persona_dir: Path,
    current_arcs: list[ReflexArc],
    removed_arc_names: set[str],
    provider: LLMProvider,
    persona_name: str,
    persona_pronouns: str | None = None,
    look_back_days: int = 30,
    total_cap: int = DEFAULT_TOTAL_CAP,
    max_emergences: int = DEFAULT_MAX_EMERGENCES_PER_TICK,
    max_prunings: int = DEFAULT_MAX_PRUNINGS_PER_TICK,
    now: datetime | None = None,
) -> ReflexCrystallizationResult:
    """One pass of reflex emergence + pruning judgment.

    Never raises to caller — provider errors and parse failures both return
    empty results. Reflex emergence failure is a 'no growth this week' event,
    not a crashed brain.
    """
    from datetime import UTC

    if now is None:
        now = datetime.now(UTC)

    emotion_vocabulary = _load_emotion_vocabulary(persona_dir)

    active_count = len(current_arcs)
    effective_max_emergences = max_emergences if active_count < total_cap else 0

    corpus = _build_corpus(
        store=store,
        persona_dir=persona_dir,
        persona_name=persona_name,
        persona_pronouns=persona_pronouns,
        current_arcs=current_arcs,
        removed_arc_names=removed_arc_names,
        emotion_vocabulary=sorted(emotion_vocabulary),
        now=now,
        look_back_days=look_back_days,
    )
    prompt = _render_prompt(
        corpus=corpus,
        persona_name=persona_name,
        persona_pronouns=persona_pronouns,
        max_emergences=effective_max_emergences,
        max_prunings=max_prunings,
        active_arc_count=active_count,
        active_floor=_ACTIVE_FLOOR,
    )

    # Provider call — never raise out of this function
    try:
        response_text = provider.generate(prompt)
    except ProviderError as exc:
        logger.warning("crystallize_reflex: provider failed: %s", exc)
        return ReflexCrystallizationResult(emergences=[], prunings=[])
    except Exception as exc:  # noqa: BLE001
        logger.warning("crystallize_reflex: provider raised unexpected: %s", exc)
        return ReflexCrystallizationResult(emergences=[], prunings=[])

    parsed = _parse_response(response_text)
    if parsed is None:
        return ReflexCrystallizationResult(emergences=[], prunings=[])

    raw_emergences = parsed.get("emergences", [])
    raw_prunings = parsed.get("prunings", [])
    if not isinstance(raw_emergences, list):
        raw_emergences = []
    if not isinstance(raw_prunings, list):
        raw_prunings = []

    # Validate emergences
    accepted_emergences: list[ReflexArcProposal] = []
    current_arc_names = {a.name for a in current_arcs}
    existing_trigger_keysets = [frozenset(a.trigger.keys()) for a in current_arcs]

    for prop_dict in raw_emergences:
        if not isinstance(prop_dict, dict):
            logger.info("emergence proposal skipped: not a dict")
            continue
        if len(accepted_emergences) >= effective_max_emergences:
            logger.info(
                "emergence proposal dropped: at per-tick cap (%d)",
                effective_max_emergences,
            )
            continue

        accepted, reason = _validate_emergence_proposal(
            proposal_dict=prop_dict,
            current_arc_names=current_arc_names,
            removed_arc_names=removed_arc_names,
            emotion_vocabulary=emotion_vocabulary,
            existing_trigger_keysets=existing_trigger_keysets,
            active_arc_count=active_count + len(accepted_emergences),
            total_cap=total_cap,
        )
        if not accepted:
            logger.info(
                "emergence proposal rejected name=%r reason=%s",
                prop_dict.get("name", "?"), reason,
            )
            continue
        try:
            accepted_emergences.append(_proposal_from_dict(prop_dict))
        except (KeyError, TypeError, ValueError) as exc:
            logger.info("emergence proposal hydration failed: %s", exc)

    # Validate prunings
    accepted_prunings: list[ReflexPruneProposal] = []
    for prop_dict in raw_prunings:
        if not isinstance(prop_dict, dict):
            logger.info("pruning proposal skipped: not a dict")
            continue
        if len(accepted_prunings) >= max_prunings:
            logger.info("pruning proposal dropped: at per-tick cap")
            continue
        accepted, reason = _validate_pruning_proposal(
            proposal_dict=prop_dict,
            current_arcs=current_arcs,
            prunes_accepted_so_far=len(accepted_prunings),
            max_prunings_per_tick=max_prunings,
        )
        if not accepted:
            logger.info(
                "pruning proposal rejected name=%r reason=%s",
                prop_dict.get("name", "?"), reason,
            )
            continue
        accepted_prunings.append(ReflexPruneProposal(
            name=str(prop_dict["name"]),
            reasoning=str(prop_dict["reasoning"]).strip(),
        ))

    result = ReflexCrystallizationResult(
        emergences=accepted_emergences,
        prunings=accepted_prunings,
    )

    # Phase 4.2 — emit one initiate candidate per accepted decision. Wrapped
    # in try/except so a downstream emit failure can never crash the
    # crystallizer: reflex emergence is physiology, initiate is signal.
    aggregated_vector = _aggregate_recent_emotion_vector(store, now=now)
    for emergence in accepted_emergences:
        _emit_initiate_candidate(
            persona_dir=persona_dir,
            source_id=f"reflex_emergence:{emergence.name}",
            label=emergence.name,
            related_memory_ids=[],
            emotion_vector=aggregated_vector,
        )
    for prune in accepted_prunings:
        _emit_initiate_candidate(
            persona_dir=persona_dir,
            source_id=f"reflex_pruning:{prune.name}",
            label=prune.name,
            related_memory_ids=[],
            emotion_vector=aggregated_vector,
        )

    return result


def _aggregate_recent_emotion_vector(
    store: MemoryStore, *, now: datetime, look_back_days: int = 7,
) -> dict[str, float]:
    """Return a max-pooled emotion vector across recent active memories.

    Used by crystallizers to give their initiate candidates a real
    emotional context — what's been alive in the persona over the
    last week, not a moment-in-time felt state. Empty dict on any
    failure (the emit candidate just carries no vector — better than
    a zero-filled lie).
    """
    try:
        from brain.emotion.aggregate import aggregate_state

        cutoff = now - timedelta(days=look_back_days)
        recent: list[Any] = []
        for mtype in _CONVERSATION_TYPES:
            for mem in store.list_by_type(mtype, active_only=True):
                if mem.created_at >= cutoff and mem.emotions:
                    recent.append(mem)
        if not recent:
            return {}
        state = aggregate_state(recent)
        return dict(state.emotions)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reflex: recent-emotion aggregation failed: %s", exc)
        return {}


def _emit_initiate_candidate(
    *,
    persona_dir: Path,
    source_id: str,
    label: str,
    related_memory_ids: list[str],
    emotion_vector: dict[str, float] | None = None,
) -> None:
    """Emit one initiate candidate after a reflex crystallization commit.

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
        logger.warning("reflex crystallization initiate emit failed: %s", exc)


# ---------- Validation gates ----------


def _validate_emergence_proposal(
    *,
    proposal_dict: dict,
    current_arc_names: set[str],
    removed_arc_names: set[str],
    emotion_vocabulary: set[str],
    existing_trigger_keysets: list[frozenset[str]],
    active_arc_count: int,
    total_cap: int,
) -> tuple[bool, str]:
    """Run all 9 emergence gates in order. Returns (accepted, reason)."""
    name = proposal_dict.get("name", "")
    if not isinstance(name, str):
        return False, "gate 1: name must be a string"

    # Gate 1: name validity
    if not name or len(name) > _NAME_MAX_LEN or not _NAME_REGEX.match(name):
        return False, f"gate 1: invalid name {name!r}"

    # Gate 2: not in current_arc_names (idempotent silent skip)
    if name in current_arc_names:
        return False, "gate 2: name already exists in current arcs"

    # Gate 3: not in graveyard
    if name in removed_arc_names:
        return False, "gate 3: name in graveyard window — respecting user removal"

    # Hydrate trigger
    trigger = proposal_dict.get("trigger")
    if not isinstance(trigger, dict) or not trigger:
        return False, "gate 5: trigger must be non-empty dict"

    try:
        trigger_typed = {str(k): float(v) for k, v in trigger.items()}
    except (TypeError, ValueError):
        return False, "gate 5: trigger values must be numeric"

    # Gate 4: all trigger emotions in vocabulary
    unknown = set(trigger_typed.keys()) - emotion_vocabulary
    if unknown:
        return False, (
            f"gate 4: trigger references unknown emotions {sorted(unknown)} "
            "not in vocabulary"
        )

    # Gate 6: threshold floor 5.0
    for emo, thresh in trigger_typed.items():
        if thresh < _THRESHOLD_FLOOR:
            return False, (
                f"gate 6: threshold {thresh} for {emo!r} below floor {_THRESHOLD_FLOOR}"
            )

    # Gate 7: cooldown floor 12h
    cooldown = proposal_dict.get("cooldown_hours")
    try:
        cooldown_f = float(cooldown)
    except (TypeError, ValueError):
        return False, "gate 7: cooldown_hours must be numeric"
    if cooldown_f < _COOLDOWN_FLOOR_HOURS:
        return False, (
            f"gate 7: cooldown {cooldown_f}h below floor {_COOLDOWN_FLOOR_HOURS}h"
        )

    # Gate 5: prompt_template renderable (after numeric checks so we surface
    # the more-actionable threshold/cooldown errors first)
    prompt_template = proposal_dict.get("prompt_template")
    if not isinstance(prompt_template, str) or not prompt_template:
        return False, "gate 5: prompt_template must be non-empty string"
    if not _prompt_template_renderable(prompt_template):
        return False, "gate 5: prompt_template fails format_map smoke test"

    # Gate 8: trigger non-overlap with existing arcs (strict subset/superset)
    proposed_keyset = frozenset(trigger_typed.keys())
    for existing_keyset in existing_trigger_keysets:
        if proposed_keyset == existing_keyset:
            # Exact match — would have been gate 2 caught if same name; if
            # different name with same keys, treat as non-overlap (partial allowed)
            continue
        if proposed_keyset < existing_keyset:
            return False, (
                "gate 8: trigger keys are strict subset of existing arc's"
            )
        if proposed_keyset > existing_keyset:
            return False, (
                "gate 8: trigger keys are strict superset of existing arc's"
            )

    # Gate 9: total cap
    if active_arc_count >= total_cap:
        return False, (
            f"gate 9: arc set is full ({active_arc_count} >= cap {total_cap})"
        )

    # Required string fields (description, output_memory_type, reasoning)
    for required in ("description", "output_memory_type", "reasoning"):
        v = proposal_dict.get(required)
        if not isinstance(v, str) or not v.strip():
            return False, f"gate 5: {required} must be non-empty string"

    return True, "accepted"


def _validate_pruning_proposal(
    *,
    proposal_dict: dict,
    current_arcs: list[ReflexArc],
    prunes_accepted_so_far: int,
    max_prunings_per_tick: int = DEFAULT_MAX_PRUNINGS_PER_TICK,
) -> tuple[bool, str]:
    """Run all 5 pruning gates. Returns (accepted, reason)."""
    name = proposal_dict.get("name")
    if not isinstance(name, str):
        return False, "gate P1: name must be a string"

    # Gate P1: arc exists
    by_name = {a.name: a for a in current_arcs}
    target = by_name.get(name)
    if target is None:
        return False, f"gate P1: arc {name!r} does not exist"

    # Gate P2: created_by must be brain_emergence
    if target.created_by != "brain_emergence":
        return False, (
            f"gate P2: arc {name!r} is created_by={target.created_by!r} — "
            "protected; only Hana removes those"
        )

    # Gate P3: active floor 4
    if len(current_arcs) - 1 < _ACTIVE_FLOOR:
        return False, (
            f"gate P3: pruning would drop active count below floor {_ACTIVE_FLOOR}"
        )

    # Gate P4: max prunes per tick (configurable; default 1)
    if prunes_accepted_so_far >= max_prunings_per_tick:
        return False, f"gate P4: max {max_prunings_per_tick} prune(s) per tick"

    # Gate P5: reasoning non-empty
    reasoning = proposal_dict.get("reasoning", "")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return False, "gate P5: reasoning must be non-empty"

    return True, "accepted"


# ---------- Helpers ----------


def _parse_response(text: str) -> dict | None:
    """Strict JSON parse. Returns None on any failure."""
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("crystallize_reflex: malformed JSON response: %s", exc)
        return None
    if not isinstance(parsed, dict):
        logger.warning("crystallize_reflex: response is not a JSON object")
        return None
    return parsed


def _proposal_from_dict(d: dict) -> ReflexArcProposal:
    return ReflexArcProposal(
        name=str(d["name"]),
        description=str(d["description"]).strip(),
        trigger={str(k): float(v) for k, v in d["trigger"].items()},
        cooldown_hours=float(d["cooldown_hours"]),
        output_memory_type=str(d["output_memory_type"]),
        prompt_template=str(d["prompt_template"]),
        reasoning=str(d["reasoning"]).strip(),
        days_since_human_min=float(d.get("days_since_human_min", 0.0)),
    )


def _prompt_template_renderable(template: str) -> bool:
    """Smoke-test the template via format_map with a defaultdict('0') backing.

    Catches malformed format specs and references that can't be resolved.
    The defaultdict means most KeyErrors won't fire — we mainly catch
    malformed specs like `{x:0.2f` (unterminated) or `{:invalid}`.
    """
    canonical = defaultdict(
        lambda: "0",
        persona_name="nell",
        emotion_summary="vulnerability: 7/10",
        memory_summary="—",
        days_since_human="0",
    )
    try:
        template.format_map(canonical)
    except (KeyError, ValueError, IndexError):
        return False
    return True


def _load_emotion_vocabulary(persona_dir: Path) -> set[str]:
    """Read emotion_vocabulary.json and return the set of emotion names.

    Missing file or corruption → empty set (gate 4 will reject any trigger).
    """
    path = persona_dir / "emotion_vocabulary.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        emotions = data.get("emotions", [])
        return {
            e["name"]
            for e in emotions
            if isinstance(e, dict) and isinstance(e.get("name"), str)
        }
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        logger.warning("could not read emotion_vocabulary.json: %s", exc)
        return set()
