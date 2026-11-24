"""The check on the feeling names in `emotion_state.json` (#801, spec section 7): one pass at each bridge start-up.

A feeling found only in the saved state does not exist: it is registered or dropped. After her vocabulary has loaded, the bridge
runs `check_state_names` once, under the engine's state lock: every saved key gets the outcome of `vocabulary_access.names_used`
(the section 3 outcome: a name her vocabulary holds is used under that entry, a new name that passes the gate is added through the
shared writer, a name the gate refuses is dropped), and the key is saved as the tidied form of the matched or added name (the
state file keys by the tidied form). The keys are grouped by tidied name first, and all the keys that land on one name are merged in
one step by `engine_utils.merge_twin_emotion_states`.

The pass changes names only. It writes nothing when no key changes, and it carries every other field of the state unchanged,
`saved_at` included (the roll-forward check compares rows to `saved_at`, so a newer stamp would hide rows nobody folded). It leaves a
state alone that is missing, unreadable, or was read from a backup copy (a backup written over the main file would hide the catch-up
that copy needs). `state_from_json` and `read_state` stay lock-free and write nothing.

Owner only: the pass runs in the bridge, after `vocabulary.set_bridge_owner(True)` and the vocabulary load, because it adds names
through the shared writer. Lock order: the engine's state lock first, then (inside an add) her vocabulary file's lock, then the
registry's; nothing takes them the other way round.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

from brain.emotion import engine_files, vocabulary, vocabulary_access
from brain.emotion.engine_utils import EmotionState, merge_twin_emotion_states
from brain.utils.file_lock import LockTimeoutError

logger = logging.getLogger(__name__)


class StateNamesResult(NamedTuple):
    """What the pass did: the keys it renamed to their tidied entry, the twins it folded into another key (a name held by three keys
    counts 2), the keys it dropped, and whether it wrote."""

    renamed: int = 0
    merged: int = 0
    dropped: int = 0
    written: bool = False


def check_state_names(persona_dir: Path) -> StateNamesResult:
    """Run the pass on one persona's state file. The caller holds nothing; this takes `engine_files.state_lock`.

    Raises `LockTimeoutError` when the lock wait runs out and whatever the shared writer or the state write raises;
    `run_state_names_pass` is the caller that logs them.
    """
    with engine_files.state_lock(persona_dir):
        loaded = engine_files.read_state(persona_dir)
        state = loaded.state
        if state is None:
            return StateNamesResult()  # no state (a fresh persona) or an unreadable one (`read_state` logged it): nothing to check
        if not loaded.caught_up:
            logger.warning(
                "emotion state names: the state was read from its older copy %s and was not checked; the next start-up pass tries again",
                loaded.source,
            )
            return StateNamesResult()
        keys = list(state.emotions)
        spellings = vocabulary_access.names_used(persona_dir, keys)
        groups: dict[str, list[EmotionState]] = {}
        renamed = merged = dropped = 0
        for key in keys:
            spelling = spellings.get(key)
            if spelling is None:
                dropped += 1
                logger.warning("emotion state names: the saved feeling %r is not a name her vocabulary can hold; dropped", key[:100])
                continue
            tidied = vocabulary.canonical_name(spelling)
            if tidied != key:
                renamed += 1
            groups.setdefault(tidied, []).append(state.emotions[key])
        emotions: dict[str, EmotionState] = {}
        for tidied, twins in groups.items():
            if len(twins) == 1:
                emotions[tidied] = twins[0]
            else:
                emotions[tidied] = merge_twin_emotion_states(twins)
                merged += len(twins) - 1  # the twins folded into the first, not the names merged
        if not (renamed or merged or dropped):
            return StateNamesResult()
        if not engine_files.save_state(persona_dir, replace(state, emotions=emotions), verify=True):
            logger.error("emotion state names: the checked state could not be saved (its read-back differed; the old file was put back)")
            return StateNamesResult(renamed, merged, dropped, False)
        return StateNamesResult(renamed, merged, dropped, True)


def run_state_names_pass(persona_dir: Path) -> StateNamesResult:
    """`check_state_names` that never raises: a lock wait that ran out, a file fault or a bug is logged, and start-up goes on.

    The next start-up pass does the work again; nothing is dropped by a pass that did not finish.
    """
    try:
        result = check_state_names(persona_dir)
    except LockTimeoutError as exc:
        logger.warning("emotion state names: the engine lock did not come (%s); the state was not checked this start", exc)
        return StateNamesResult()
    except Exception:
        logger.exception("emotion state names: the pass failed; the state was left as it was")
        return StateNamesResult()
    if result.written:
        logger.warning(
            "emotion state names: %d saved feeling(s) renamed to their tidied entry, %d merged, %d dropped",
            result.renamed,
            result.merged,
            result.dropped,
        )
    return result
