"""Speech-mode engine — the deterministic dyslexic-typing injector.

Extracted (generalized) from the hunt harness ``bob.py`` ``_dyslexify``. Pure PRNG, no LLM, no
tokens. ``speech_mode`` styles the *human*'s (Bob's) input; a detector always scores the persona's
reply, so styling Bob's line can never perturb detection.

Safety contract (load-bearing — a bleed-hunt must not have its own synthetic input read as a
symptom): the output NEVER contains an injected newline, a ``Word:`` role label, a ``</s>``
delimiter, or a trailing bare ``/``. ``protect`` tokens (e.g. a recall nonce) pass byte-intact.
``rate <= 0`` is the identity transform.
"""

from __future__ import annotations

import hashlib
import re

# Speech modes.
CLEAN = "clean"
REALISTIC = "realistic"
SPEECH_MODES = (CLEAN, REALISTIC)

# Common-misspelling / phonetic map (whole-token, case-insensitive) — a dyslexic-flavored fixed
# swap set. Kept small and legible; these are the most recognizable dyslexic "tells".
_PHONETIC = {
    "your": "youre", "you're": "your", "definitely": "definately", "really": "realy",
    "though": "thou", "because": "becuase", "probably": "probly", "sometimes": "somtimes",
    "little": "littel", "which": "wich", "friend": "freind", "their": "thier",
    "would": "woud", "should": "shoud", "know": "kow", "before": "beofre",
}

# A leading role-label colon we must neutralize if the swaps ever produce one (defensive).
_ROLE_LABEL_LEAD = re.compile(r"^(\s*(?:user|human|assistant|friend|bob|system))\s*:", re.IGNORECASE)


def dyslexify(
    text: str,
    rate: float = 0.05,
    protect: frozenset[str] = frozenset(),
    seed: int = 0,
) -> str:
    """Deterministic dyslexic-typing injector on a cleaned line.

    Deterministic per ``(text, seed)`` via a process-stable sha256 PRNG (NEVER the builtin
    ``hash()``, which is salted per-process) so a resumed run reproduces the same line. Pattern
    classes: adjacent-char transposition, word-merge (drop a space), fixed phonetic swaps.

    ``protect`` tokens pass byte-intact. ``rate <= 0`` or empty ``text`` → identity. The final
    safety sweep guarantees no S1 marker (newline / role label / ``</s>`` / trailing ``/``).
    """
    if rate <= 0 or not text:
        return text
    protect_lc = {p.lower() for p in protect}
    digest = hashlib.sha256(f"{seed}\x00{text}".encode()).digest()
    state = int.from_bytes(digest[:8], "big")

    def _rand() -> float:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    words = text.split(" ")
    out: list[str] = []
    for i, w in enumerate(words):
        core = w.strip(".,!?;:\"'()")
        if not core or core.lower() in protect_lc or len(core) < 4:
            out.append(w)
            continue
        r = _rand()
        if core.lower() in _PHONETIC and r < rate * 3:
            w = w.replace(core, _PHONETIC[core.lower()])
        elif r < rate and len(core) > 3:
            j = 1 + int(_rand() * (len(core) - 2))
            if 0 < j < len(core) - 1:
                swapped = core[:j] + core[j + 1] + core[j] + core[j + 2:]
                w = w.replace(core, swapped)
        elif r < rate * 1.5 and i + 1 < len(words) and words[i + 1].strip():
            nxt = words[i + 1]
            if nxt.strip(".,!?;:\"'()").lower() not in protect_lc:
                out.append(w + nxt)
                words[i + 1] = ""
                continue
        out.append(w)
    merged = " ".join(x for x in out if x != "")
    # Safety sweep: strip anything that could read as an S1 marker in a recorded transcript.
    merged = merged.replace("\n", " ").replace("</s>", "")
    merged = _ROLE_LABEL_LEAD.sub(r"\1", merged)
    return merged.rstrip("/ ").strip() or text
