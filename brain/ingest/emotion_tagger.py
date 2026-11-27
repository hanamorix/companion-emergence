"""Tag a memory that has no feelings but should: the temp gate's emotion tagging.

A memory about to be written through the consolidation gate (a promoted candidate, a merge target, a recalled row) that carries
no stored emotions is sent to one background-tier model call, and the emotions it returns are written onto the row (the write checks each name against her vocabulary)
(`tag_if_feelingless`). This replaces the idle emotion backfill job with its cursor, daily cap, pacing and state files: a row is
tagged when the gate touches it, so the database is fixed in time by the rows that are written and recalled.

Public surface
--------------
make_emotion_tagger(provider) -> Callable[[Memory], dict[str, float]]
tag_if_feelingless(store, tagger, memory) -> bool
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from brain import prompt_strings
from brain.emotion import vocabulary
from brain.utils.llm_output import find_json_object_block
from brain.utils.log_throttle import warn_throttled

if TYPE_CHECKING:
    from brain.bridge.provider import LLMProvider
    from brain.memory.store import Memory, MemoryStore

logger = logging.getLogger(__name__)

EmotionTagger = Callable[["Memory"], dict[str, float]]


def _load_vocab() -> frozenset[str]:
    try:
        return frozenset(e.name for e in vocabulary.list_all())
    except Exception:  # noqa: BLE001
        logger.warning("emotion_tagger: failed to load vocab; the prompt asks for any emotion name")
        return frozenset()


def _normalize(raw: dict[str, Any]) -> dict[str, float]:
    """Keep the emotions with a finite intensity above 0; clamp to 10. NaN and inf are dropped.

    The names are NOT filtered here: the store's write checks each against her vocabulary (`MemoryStore.update`), which adds a
    name that passes the gate and discards one that cannot be held (spec section 6). The stored key is the tidied name
    (`vocabulary.canonical_name`); spellings of one emotion in one reply are averaged, as `resolve_emotions` averages two
    spellings on a memory. A name that tidies to nothing is dropped.

    If a value cannot be read as a number it is skipped; nothing here raises for a bad reply.
    """
    values: dict[str, list[float]] = {}
    for name, intensity in raw.items():
        key = vocabulary.canonical_name(name) if isinstance(name, str) else ""
        if not key:
            continue
        try:
            v = float(intensity)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or v <= 0.0:
            continue
        values.setdefault(key, []).append(min(v, 10.0))
    return {key: sum(v) / len(v) for key, v in values.items()}


# Text externalized to prompt_strings.toml [ingest.emotion_tagger] (issue #129 stage 2a).
_TAGGER_SYSTEM_PROMPT = prompt_strings.register("ingest.emotion_tagger.tagger_system_prompt")
# Text externalized to prompt_strings.toml [ingest.emotion_tagger] (issue #129 stage 2b).
_TAGGER_PROMPT_SEGMENTS = prompt_strings.register_segments("ingest.emotion_tagger.tagger_prompt_segments")


def make_emotion_tagger(provider: LLMProvider) -> EmotionTagger:
    """Return a tagger closure that makes one ``provider.generate`` call per memory.

    ``provider`` is the background-classifier tier provider the consolidation gate already holds. The reply is read with the
    shared `find_json_object_block` (a fenced reply or a reply with prose around the object is read), then `_normalize`.

    A reply with no JSON object tags nothing and logs one throttled warning naming the kind: an empty (or whitespace-only) reply, or
    a reply that held no object (its length is logged, never its text). Each kind has its own throttle key. A valid `{}` logs nothing.
    A failed call (an exception from ``provider.generate``, or an invalid object) logs the `tagger error` line instead.
    """

    def _tagger(memory: Memory) -> dict[str, float]:
        try:
            vocab = _load_vocab()
            vocab_str = ", ".join(sorted(vocab)) if vocab else "(any emotion name)"
            seg = _TAGGER_PROMPT_SEGMENTS
            prompt = seg[0] + vocab_str + seg[1] + memory.content + seg[2]
            raw_text = provider.generate(prompt, system=_TAGGER_SYSTEM_PROMPT)
            block = find_json_object_block(raw_text)
            if block is None:
                # No object at all: a reply that is empty or only prose points at something else going wrong (a model that
                # answered nothing or only prose; a failed call raises instead), so it is logged, one throttle key per kind. A valid `{}` is not here: it is the model's answer.
                if not raw_text.strip():
                    warn_throttled(logger, "emotion_tagger.empty_reply", "emotion_tagger: empty reply for memory %s", memory.id, exc_info=False)
                else:
                    warn_throttled(
                        logger,
                        "emotion_tagger.no_json_object",
                        "emotion_tagger: reply held no JSON object for memory %s (%d characters)",
                        memory.id,
                        len(raw_text),
                        exc_info=False,
                    )
                return {}
            return _normalize(json.loads(block))
        except Exception as exc:  # noqa: BLE001
            from brain.bridge.cli_throttle import ThrottleDeferred

            if isinstance(exc, ThrottleDeferred):
                raise  # #456/#473: "not now": the row stays untagged and the caller decides
            # The same throttled line as the gate's: with no daily cap left, a provider that fails on every call must not log on every write.
            warn_throttled(logger, "emotion_tagger.tagger_error", "emotion_tagger: tagger error: %s", exc, exc_info=False)
            return {}

    return _tagger


def tag_if_feelingless(store: MemoryStore, tagger: EmotionTagger | None, memory: Memory) -> bool:
    """Write the tagger's feelings onto ``memory`` when it has none. True when a tag was written.

    One helper for every gate write. Nothing happens without a tagger, or for a row that already has stored emotions (the
    tagger is not called). The tag is written with ``store.update(memory.id, emotions=...)``, after the row exists, so the
    engine folds it (`on_emotions_assigned`) and the peak is raised; the stored-feelings fields are private to the engine and
    the store. The tagger call is slow, so the row is read once more before the write and a row that is gone, or that has
    feelings by then, is left as it is: a tag never replaces feelings written in the meantime.

    Raises whatever the tagger or the store raises (a deferred model call included): the gate's caller makes it fail-soft.

    Accepted cost (P4): a memory the tagger returns nothing for stays untagged and is sent to the tagger again at each recall.

    Accepted cost (P4): a memory long enough to time out the tagging call at every recall is accepted as it is (no length cap, no "tried" mark).

    Accepted edge (P4): this does not filter on ``active``. An inactive row reaches the helper only if a gate path writes onto
    it (a row opened by id after a deactivation, or a merge verdict that names an inactive id: the merge branch in
    consolidation.py checks no active flag). Nothing in brain/ deactivates a row today (``store.deactivate`` has no production
    caller). Such a row is tagged.
    """
    if tagger is None or store.has_stored_emotions(memory):
        return False
    tags = tagger(memory)
    if not tags:
        return False
    current = store.get(memory.id, bump=False)  # a maintenance read, not an access
    if current is None or store.has_stored_emotions(current):
        return False
    store.update(memory.id, emotions=tags)
    return True
