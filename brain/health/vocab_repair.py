"""One-time startup migration repairing already-stubbed emotion_vocabulary.json.

Step 1 (sync, provider-free): entries with the OLD bad decay_half_life_days==1.0
AND the placeholder description → bumped to 14.0. Written through `brain/emotion/vocabulary_file.py`
(every change to the file in this module is). Stops ongoing damage immediately.

Step 2 (Haiku, budget-capped, fail-soft): for repaired stub names, derive
one-line descriptions and replace the placeholder. Batch into ONE prompt per
≤_DESCRIBE_BATCH names (name + up to 3 memory excerpts each). On any provider
failure/parse failure: keep placeholders — Step 1 already landed.

State file ``<persona_dir>/vocab_repair_state.json`` → idempotent skip.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from brain import prompt_strings
from brain.emotion import vocabulary_file
from brain.emotion.vocabulary import canonical_name
from brain.health.json_io import Recovery, read_json, write_json
from brain.health.parse_failures import record_parse_failure
from brain.memory.store import MemoryStore
from brain.utils.llm_output import extract_json_object

if TYPE_CHECKING:
    from brain.bridge.provider import LLMProvider

logger = logging.getLogger(__name__)

_STATE_FILE = "vocab_repair_state.json"

_OLD_STUB_DECAY_DAYS: float = 1.0
_REPAIRED_DECAY_DAYS: float = 14.0
_DESCRIBE_BATCH: int = 20
_EXCERPT_CHAR_LIMIT: int = 140
_MAX_EXCERPTS_PER_NAME: int = 3

# Text externalized to prompt_strings.toml [health.vocab_repair] (issue #129 stage 2c).
_DESCRIBE_SYSTEM = prompt_strings.register("health.vocab_repair.describe_system")


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class RepairReport:
    repaired: int       # number of entries with half-life bumped
    described: int      # number of placeholder descriptions replaced
    status: str         # "complete" | "skipped"
    completed_at: str   # ISO timestamp


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------

def _state_path(persona_dir: Path) -> Path:
    return persona_dir / _STATE_FILE


def _load_state(persona_dir: Path) -> dict | None:
    # CONTENT: an unreadable file raises (as it always did); a missing or corrupt one is "no state".
    return read_json(
        _state_path(persona_dir), default=lambda: None, recover=Recovery.CONTENT, expect=dict
    )


def _save_state(persona_dir: Path, state: dict) -> None:
    write_json(_state_path(persona_dir), state, indent=2, fsync=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def should_run_vocab_repair(persona_dir: Path) -> bool:
    """Return True iff the vocab file has any placeholder-description entry.

    #173: the completed-state file only retires Step 1 (the one-time 1.0 → 14.0
    half-life bump). Placeholders keep being minted afterwards by
    reconstruct.py / persona_loader.py (already at 14.0), so Step 2 (describe)
    must be allowed to run again whenever any placeholder exists. The
    supervisor only asks at startup, so the retry cadence is once per bridge
    start, bounded by _DESCRIBE_BATCH per run.
    """
    data = read_json(
        persona_dir / "emotion_vocabulary.json",
        default=lambda: None,
        recover=Recovery.CONTENT,
        expect=dict,
    )
    if data is None:  # missing or corrupt
        return False

    from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION

    for entry in data.get("emotions", []):
        if isinstance(entry, dict) and entry.get("description") == PLACEHOLDER_DESCRIPTION:
            return True
    return False


def run_vocab_repair(
    persona_dir: Path,
    *,
    store: MemoryStore,
    provider: LLMProvider | None,
) -> RepairReport:
    """Run the vocab repair.

    Step 1 (bump legacy 1.0 half-life stubs to 14.0) is one-time, retired by
    the completed-state file. Step 2 (derive a description for every
    placeholder entry) runs whenever placeholders exist and a provider is
    available (#173). Counts in the state file accumulate across runs.

    Returns a RepairReport whether or not work was done.
    """
    from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION

    existing = _load_state(persona_dir)
    prior_complete = existing is not None and existing.get("status") == "complete"
    prior_repaired = int(existing.get("repaired", 0)) if prior_complete else 0
    prior_described = int(existing.get("described", 0)) if prior_complete else 0

    vocab_path = persona_dir / "emotion_vocabulary.json"
    data = read_json(vocab_path, default=lambda: None, recover=Recovery.CONTENT, expect=dict)
    if data is None:  # missing, or corrupt (the module logged it): skip, as before
        return _write_and_return(persona_dir, repaired=0, described=0)

    # ------------------------------------------------------------------
    # Step 0 (#174, provider-free, every run): collapse spelling-variant twins
    # (love_grief_blend / grief_love_blend, anticipatory-grief / Anticipatory_Grief)
    # into one entry under the spelling that comes first in the file. A twin with a real
    # description gives the fields (it beats a placeholder). Every change to the file below
    # goes through `vocabulary_file`; a change it could not write raises, so the state file
    # is not rewritten and the repair runs again at the next start.
    # ------------------------------------------------------------------
    merge_result = vocabulary_file.merge(vocab_path)
    if merge_result.cannot_write:
        raise OSError(f"vocab_repair: the twin merge could not be written to {vocab_path.name}")
    if merge_result.pairs:
        logger.info("vocab_repair: merged %d variant twin(s): %s", len(merge_result.pairs), merge_result.pairs)
        # The file changed under us: the steps below work from what it holds now.
        data = read_json(vocab_path, default=lambda: None, recover=Recovery.CONTENT, expect=dict)
        if data is None:
            return _write_and_return(persona_dir, repaired=0, described=0)

    if prior_complete and provider is None:
        # Step 1 already done and Step 2 needs a provider: nothing more to do.
        return RepairReport(
            repaired=prior_repaired,
            described=prior_described,
            status="complete",
            completed_at=existing.get("completed_at", ""),
        )

    emotions = data.get("emotions", [])

    # ------------------------------------------------------------------
    # Step 1: bump half-life for stub entries (sync, provider-free)
    # ------------------------------------------------------------------
    stub_names: list[str] = []
    bumps: dict[str, dict] = {}
    for entry in emotions:
        if not isinstance(entry, dict):
            continue
        if entry.get("description") != PLACEHOLDER_DESCRIPTION:
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue  # a nameless entry has nothing to describe or bump
        # Step 2 target: every placeholder, whatever its half-life (#173).
        stub_names.append(name)
        if not prior_complete and entry.get("decay_half_life_days") == _OLD_STUB_DECAY_DAYS:
            bumps[name] = {"decay_half_life_days": _REPAIRED_DECAY_DAYS}

    repaired = 0
    if bumps:
        results = vocabulary_file.replace_fields(vocab_path, bumps, only_if=_is_old_stub)
        if any(r.outcome == "cannot_write" for r in results):
            raise OSError(f"vocab_repair: the half-life bump could not be written to {vocab_path.name}")
        repaired = sum(1 for r in results if r.changed)

    if repaired > 0:
        logger.info(
            "vocab_repair: bumped half-life to %.1f for %d stub entries: %s",
            _REPAIRED_DECAY_DAYS,
            repaired,
            stub_names,
        )

    # ------------------------------------------------------------------
    # Step 2: derive descriptions via provider (fail-soft). Re-runnable: any
    # placeholder still present (crash here, provider failure, or a newly
    # minted one) is picked up on the next startup pass (#173).
    # ------------------------------------------------------------------
    described = 0
    if provider is not None and stub_names:
        try:
            described = _describe_stubs(
                persona_dir=persona_dir,
                vocab_path=vocab_path,
                stub_names=stub_names,
                store=store,
                provider=provider,
            )
        except Exception as exc:  # noqa: BLE001
            from brain.bridge.cli_throttle import ThrottleDeferred

            if isinstance(exc, ThrottleDeferred):
                raise
            logger.warning(
                "vocab_repair: Step 2 description derivation failed (kept placeholders): %s", exc
            )

    return _write_and_return(
        persona_dir,
        repaired=prior_repaired + repaired,
        described=prior_described + described,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_and_return(persona_dir: Path, *, repaired: int, described: int) -> RepairReport:
    completed_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    state = {
        "status": "complete",
        "repaired": repaired,
        "described": described,
        "completed_at": completed_at,
    }
    _save_state(persona_dir, state)
    return RepairReport(
        repaired=repaired,
        described=described,
        status="complete",
        completed_at=completed_at,
    )


def _is_old_stub(entry: dict) -> bool:
    """The entry is still a placeholder with the old bad half-life (so the bump has not been made by anyone since)."""
    from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION

    return (
        entry.get("description") == PLACEHOLDER_DESCRIPTION and entry.get("decay_half_life_days") == _OLD_STUB_DECAY_DAYS
    )


def _is_placeholder(entry: dict) -> bool:
    """The entry still carries the placeholder description (so no one has described it since)."""
    from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION

    return entry.get("description") == PLACEHOLDER_DESCRIPTION


def _excerpts_for_name(name: str, store: MemoryStore) -> list[str]:
    """Return up to _MAX_EXCERPTS_PER_NAME memory excerpts mentioning name."""
    try:
        # F3: cap the search so a high-frequency name doesn't pull all rows.
        candidates = store.search_text(name, limit=10)
        excerpts = []
        wanted = canonical_name(name)
        for mem in candidates:
            # Names only: a stored emotion whose value is not finite still counts as evidence for its name.
            if wanted in {canonical_name(n) for n in MemoryStore.stored_emotion_names(mem)}:
                text = mem.content[:_EXCERPT_CHAR_LIMIT]
                excerpts.append(text)
                if len(excerpts) >= _MAX_EXCERPTS_PER_NAME:
                    break
        return excerpts
    except Exception as exc:  # noqa: BLE001
        logger.warning("vocab_repair: excerpt retrieval for %r failed: %s", name, exc)
        return []


def _describe_stubs(
    *,
    persona_dir: Path,
    vocab_path: Path,
    stub_names: list[str],
    store: MemoryStore,
    provider: LLMProvider,
) -> int:
    """Call provider in batches to derive descriptions; write them through `vocabulary_file.replace_fields`.

    Returns the number of descriptions successfully replaced. A batch's descriptions that could not be written raise OSError
    (the caller keeps the placeholders and the next start retries).
    """
    described = 0
    # F2: wrap each batch in a
    # background slot so interactive chat is never displaced.
    from brain.bridge import cli_throttle as _cli_throttle  # noqa: PLC0415

    # Batch into groups of _DESCRIBE_BATCH
    for batch_start in range(0, len(stub_names), _DESCRIBE_BATCH):
        batch = stub_names[batch_start : batch_start + _DESCRIBE_BATCH]

        # Build user message: name + excerpts
        lines = []
        for name in batch:
            excerpts = _excerpts_for_name(name, store)
            line = f"- {name}"
            if excerpts:
                short = " | ".join(excerpts)
                line += f"  (examples: {short})"
            lines.append(line)

        user_prompt = (
            "Emotion names to describe:\n"
            + "\n".join(lines)
            + "\n\nReturn ONLY a JSON object, e.g.: "
            + '{"name1": "one sentence", "name2": "one sentence"}'
        )

        with _cli_throttle.background_slot() as _slot:
            if not _slot:
                logger.info(
                    "vocab_repair: throttle slot unavailable — "
                    "deferring remaining batches; Step 1 already landed"
                )
                break

            try:
                # F1: guard the persona_dir kwarg — the LLMProvider ABC only
                # declares (prompt, *, system); ClaudeCliProvider adds persona_dir
                # as an extension.  A non-Cli provider raises TypeError on the
                # extra kwarg, so retry without it.
                try:
                    raw = provider.generate(
                        user_prompt,
                        system=_DESCRIBE_SYSTEM,
                        persona_dir=persona_dir,
                    )
                except TypeError:
                    raw = provider.generate(user_prompt, system=_DESCRIBE_SYSTEM)
                # #173: Haiku may fence its reply; use the shared tolerant extractor.
                try:
                    mapping: dict = json.loads(extract_json_object(raw))
                except ValueError as exc:
                    record_parse_failure(persona_dir, site="vocab_repair", raw=raw, error=str(exc))
                    raise
            except _cli_throttle.ThrottleDeferred:
                raise  # #456/#473: "not now": nothing more is written this run (no state rewrite either)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "vocab_repair: provider call or parse failed for batch %r: %s — "
                    "keeping placeholders for this batch",
                    batch,
                    exc,
                )
                continue

            # Match the reply to the stubs by tidied name (a provider may echo a name as tidied, not as spelled in the file) and
            # write the descriptions through the shared writer, only into an entry that is still a placeholder.
            reply_by_key = {canonical_name(k): v for k, v in mapping.items() if isinstance(k, str)}
            changes: dict[str, dict] = {}
            for name in batch:
                desc = reply_by_key.get(canonical_name(name))
                if isinstance(desc, str) and desc.strip():
                    changes[name] = {"description": desc.strip()}
            patched = 0
            if changes:
                results = vocabulary_file.replace_fields(vocab_path, changes, only_if=_is_placeholder)
                if any(r.outcome == "cannot_write" for r in results):
                    raise OSError(f"vocab_repair: the descriptions could not be written to {vocab_path.name}")
                patched = sum(1 for r in results if r.changed)

            if patched == 0:
                logger.warning(
                    "vocab_repair: provider reply parsed but matched no stub name "
                    "(reply keys=%s, batch=%s) — keeping placeholders for this batch",
                    sorted(mapping)[:10],
                    batch,
                )
            if patched > 0:
                described += patched
                logger.info(
                    "vocab_repair: described %d stubs in batch starting at index %d",
                    patched,
                    batch_start,
                )

    return described
