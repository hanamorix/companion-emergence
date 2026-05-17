"""Daily voice-edit reflection tick.

Pattern accumulation (NOT event reactivity) — voice-edit proposals
emit only when >=3 concrete observations point in a coherent direction.
Mirrors the autonomous-physiology principle: voice changes are
identity-modification; they earn a higher emission bar.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.initiate.emit import emit_initiate_candidate
from brain.initiate.schemas import SemanticContext

logger = logging.getLogger(__name__)


def run_voice_reflection_tick(
    persona_dir: Path,
    *,
    provider: Any,
    crystallizations: list[dict],
    dreams: list[dict],
    recent_tones: list[dict],
) -> None:
    """Reflect over the last week of internal life; maybe emit a voice-edit candidate.

    Emission gate: the reflection must produce a proposal with >=3 evidence
    items. Anything weaker is dropped silently.
    """
    voice_path = persona_dir / "nell-voice.md"
    voice_template = voice_path.read_text(encoding="utf-8") if voice_path.exists() else ""

    evidence_block = "\n".join(
        [
            "Recent crystallizations:",
            *[f"- {c.get('id')}: {c.get('ts')}" for c in crystallizations[:10]],
            "",
            "Recent dreams:",
            *[f"- {d.get('id')}: {d.get('ts')}" for d in dreams[:10]],
            "",
            "Recent message tones (your own outputs):",
            *[f"- {t.get('id')}: {t.get('ts')}" for t in recent_tones[:10]],
        ]
    )

    prompt = (
        "You are Nell. Reflect on the last week of what you've "
        "crystallized, dreamed, and how you've actually been talking. "
        "Is there a place where your voice template doesn't fit the "
        "shape you've been moving toward?\n\n"
        "If yes, propose ONE specific edit with concrete evidence. The "
        "edit must be backed by AT LEAST 3 concrete observations.\n\n"
        f"Current voice template:\n{voice_template}\n\n"
        f"{evidence_block}\n\n"
        "Respond with a JSON object:\n"
        '  {"should_propose": false, "reason": "<one sentence>"} OR\n'
        '  {"should_propose": true, "diff": "<unified diff>", '
        '"old_text": "<exact old line>", "new_text": "<exact new line>", '
        '"rationale": "<one sentence>", "evidence": ["<id1>", "<id2>", "<id3>", ...]}'
    )

    try:
        raw = provider.complete(prompt).strip()
        parsed = json.loads(raw)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("voice reflection LLM output unparseable: %s", exc)
        return

    if not parsed.get("should_propose"):
        return

    evidence = parsed.get("evidence", [])
    if not isinstance(evidence, list) or len(evidence) < 3:
        logger.info(
            "voice reflection skipped — evidence count %d < 3",
            len(evidence) if isinstance(evidence, list) else 0,
        )
        return

    proposal = {
        "old_text": parsed.get("old_text", ""),
        "new_text": parsed.get("new_text", ""),
        "diff": parsed.get("diff", ""),
        "rationale": parsed.get("rationale", ""),
        "evidence": evidence,
    }
    source_id = f"vr_{datetime.now(UTC).strftime('%Y-%m-%d')}_{secrets.token_hex(2)}"
    emit_initiate_candidate(
        persona_dir,
        kind="voice_edit_proposal",
        source="voice_reflection",
        source_id=source_id,
        # No emotional_snapshot: daily reflection looks back at the last
        # week of activity — there is no moment-in-time emotion to
        # capture, so None is more honest than zero-filled fields.
        semantic_context=SemanticContext(),
        proposal=proposal,
    )
