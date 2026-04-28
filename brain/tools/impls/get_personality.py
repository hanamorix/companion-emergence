"""get_personality tool — superseded by SP-6 voice.md.

Status as of 2026-04-28: SP-6 (chat engine) shipped and uses voice.md as
the persona's identity surface. This tool exists for tool-loop schema
compatibility (the brain-tools MCP catalogue has 9 entries, dropping one
would break clients that enumerate them) but returns a marker indicating
voice.md is the authoritative source.
"""

from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


def get_personality(
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Returns a marker pointing at voice.md as the authoritative personality.

    The brain-tools schema preserves this entry for tool-loop compatibility,
    but the actual persona identity now lives in voice.md (loaded into the
    chat system message by SP-6's `engine.respond`). Callers interpreting
    this output should consult voice.md instead.
    """
    voice_path = persona_dir / "voice.md"
    return {
        "loaded": False,
        "superseded_by": "voice.md (SP-6 chat engine)",
        "voice_md_present": voice_path.exists(),
        "note": (
            "Personality is now expressed via voice.md, loaded into the chat "
            "system message at run time. This tool returns a marker rather "
            "than a JSON personality blob to preserve tool-loop schema "
            "compatibility without re-deriving what SP-6 ships."
        ),
    }
