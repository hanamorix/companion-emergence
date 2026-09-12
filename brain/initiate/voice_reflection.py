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

from brain import prompt_strings
from brain.initiate.emit import emit_initiate_candidate
from brain.initiate.schemas import SemanticContext

logger = logging.getLogger(__name__)

# Text externalized to prompt_strings.toml [initiate.voice_reflection] (issue #129 stage 2a).
_TICK_PROMPT_SEGMENTS = prompt_strings.register_segments("initiate.voice_reflection.tick_prompt_segments")


def run_voice_reflection_tick(
    persona_dir: Path,
    *,
    provider: Any,
    crystallizations: list[dict],
    dreams: list[dict],
    recent_tones: list[dict],
    companion_name: str = "Nell",
) -> None:
    """Reflect over the last week of internal life; maybe emit a voice-edit candidate.

    Emission gate: the reflection must produce a proposal with >=3 evidence
    items. Anything weaker is dropped silently.
    """
    voice_path = persona_dir / "voice.md"
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

    seg = _TICK_PROMPT_SEGMENTS
    prompt = seg[0] + companion_name + seg[1] + voice_template + seg[2] + evidence_block + seg[3]

    from brain.bridge import (
        cli_throttle,  # local import: avoids a circular dependency on brain.bridge
    )

    with cli_throttle.background_slot() as slot:
        if not slot:
            return  # deferred — daily reflection cadence re-fires next tick

        try:
            raw = provider.complete(prompt).strip()
            parsed = json.loads(raw)
        except cli_throttle.ThrottleDeferred as exc:
            logger.info("voice reflection deferred: %s", exc)  # #246: not a failure
            return
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
