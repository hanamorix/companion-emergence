"""voice.md loader — attempt_heal_text pattern for plain-text identity files.

voice.md is the persona's authored identity document. It is a plain-text
Markdown file, not JSON. We extend the health module's healing pattern
via a new attempt_heal_text helper that handles text files the same way
attempt_heal handles JSON files.

OG reference: NellBrain stored voice via Modelfile SYSTEM block (frozen
into model weights by regenerate_modelfile.py). In companion-emergence
we keep it editable and hot-loadable per turn — the brain's voice can
evolve without a model rebuild.
"""

from __future__ import annotations

from pathlib import Path

from brain import prompt_strings
from brain.health.anomaly import BrainAnomaly
from brain.health.attempt_heal import attempt_heal_text

# Text externalized to prompt_strings.toml [chat.voice] (issue #129 stage 1).
DEFAULT_VOICE_TEMPLATE = prompt_strings.register("chat.voice.default_template")


def load_voice(persona_dir: Path) -> tuple[str, BrainAnomaly | None]:
    """Load voice.md as plain text. Auto-heal via attempt_heal_text pattern.

    voice.md is treated as an atomic-rewrite identity file:
      - Missing → write + return DEFAULT_VOICE_TEMPLATE (no anomaly)
      - Empty/corrupt → quarantine, walk .bak rotation, restore freshest
      - All baks missing → write default template + return anomaly

    Returns (voice_md_content_as_string, anomaly_or_None).
    """
    path = persona_dir / "voice.md"
    persona_name = persona_dir.name

    def _default() -> str:
        return DEFAULT_VOICE_TEMPLATE.format(persona_name=persona_name)

    return attempt_heal_text(path, default_factory=_default)
