"""One-time startup migration repairing already-stubbed emotion_vocabulary.json.

Step 1 (sync, provider-free): entries with the OLD bad decay_half_life_days==1.0
AND the placeholder description → bumped to 14.0. Atomic write. Stops ongoing
damage immediately.

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

if TYPE_CHECKING:
    from brain.bridge.provider import LLMProvider
    from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)

_STATE_FILE = "vocab_repair_state.json"

_OLD_STUB_DECAY_DAYS: float = 1.0
_REPAIRED_DECAY_DAYS: float = 14.0
_DESCRIBE_BATCH: int = 20
_EXCERPT_CHAR_LIMIT: int = 140
_MAX_EXCERPTS_PER_NAME: int = 3

_DESCRIBE_SYSTEM = (
    "You write one-line emotion descriptions for an AI companion's emotion vocabulary. "
    "Return ONLY a JSON object mapping each name to a single-sentence description."
)


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
    p = _state_path(persona_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("vocab_repair: corrupt state file: %s", exc)
        return None


def _save_state(persona_dir: Path, state: dict) -> None:
    p = _state_path(persona_dir)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def should_run_vocab_repair(persona_dir: Path) -> bool:
    """Return True iff vocab_repair has not completed AND the vocab file has stubs."""
    existing = _load_state(persona_dir)
    if existing is not None and existing.get("status") == "complete":
        return False

    vocab_path = persona_dir / "emotion_vocabulary.json"
    if not vocab_path.exists():
        return False

    try:
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return False

    from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION

    for entry in data.get("emotions", []):
        if (
            isinstance(entry, dict)
            and entry.get("description") == PLACEHOLDER_DESCRIPTION
            and entry.get("decay_half_life_days") == _OLD_STUB_DECAY_DAYS
        ):
            return True
    return False


def run_vocab_repair(
    persona_dir: Path,
    *,
    store: MemoryStore,
    provider: LLMProvider | None,
) -> RepairReport:
    """Run the one-time vocab repair. Idempotent — skips if state==complete.

    Returns a RepairReport whether or not work was done.
    """
    from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION

    existing = _load_state(persona_dir)
    if existing is not None and existing.get("status") == "complete":
        return RepairReport(
            repaired=existing.get("repaired", 0),
            described=existing.get("described", 0),
            status="complete",
            completed_at=existing.get("completed_at", ""),
        )

    vocab_path = persona_dir / "emotion_vocabulary.json"
    if not vocab_path.exists():
        return _write_and_return(persona_dir, repaired=0, described=0)

    try:
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("vocab_repair: cannot read vocab: %s — skipping", exc)
        return _write_and_return(persona_dir, repaired=0, described=0)

    emotions = data.get("emotions", [])

    # ------------------------------------------------------------------
    # Step 1: bump half-life for stub entries (sync, provider-free)
    # ------------------------------------------------------------------
    stub_names: list[str] = []
    repaired = 0
    for entry in emotions:
        if not isinstance(entry, dict):
            continue
        if (
            entry.get("description") == PLACEHOLDER_DESCRIPTION
            and entry.get("decay_half_life_days") == _OLD_STUB_DECAY_DAYS
        ):
            entry["decay_half_life_days"] = _REPAIRED_DECAY_DAYS
            stub_names.append(entry["name"])
            repaired += 1

    if repaired > 0:
        _atomic_write_vocab(vocab_path, data)
        logger.info(
            "vocab_repair: bumped half-life to %.1f for %d stub entries: %s",
            _REPAIRED_DECAY_DAYS,
            repaired,
            stub_names,
        )

    # ------------------------------------------------------------------
    # Step 2: derive descriptions via provider (fail-soft)
    # ------------------------------------------------------------------
    described = 0
    if provider is not None and stub_names:
        try:
            described = _describe_stubs(
                persona_dir=persona_dir,
                vocab_path=vocab_path,
                data=data,
                stub_names=stub_names,
                store=store,
                provider=provider,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "vocab_repair: Step 2 description derivation failed (kept placeholders): %s", exc
            )

    return _write_and_return(persona_dir, repaired=repaired, described=described)


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


def _atomic_write_vocab(vocab_path: Path, data: dict) -> None:
    """Atomic JSON write with .bak rotation via save_with_backup."""
    from brain.health.attempt_heal import save_with_backup
    save_with_backup(vocab_path, data)


def _excerpts_for_name(name: str, store: MemoryStore) -> list[str]:
    """Return up to _MAX_EXCERPTS_PER_NAME memory excerpts mentioning name."""
    try:
        candidates = store.search_text(name)
        excerpts = []
        for mem in candidates:
            if name in (mem.emotions or {}):
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
    data: dict,
    stub_names: list[str],
    store: MemoryStore,
    provider: LLMProvider,
) -> int:
    """Call provider in batches to derive descriptions; patch data in-place.

    Returns the number of descriptions successfully replaced.
    """
    from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION

    described = 0
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

        try:
            raw = provider.generate(
                user_prompt,
                system=_DESCRIBE_SYSTEM,
                persona_dir=persona_dir,
            )
            mapping: dict = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "vocab_repair: provider call or parse failed for batch %r: %s — "
                "keeping placeholders for this batch",
                batch,
                exc,
            )
            continue

        # Patch entries in data
        batch_set = set(batch)
        patched = 0
        for entry in data.get("emotions", []):
            if not isinstance(entry, dict):
                continue
            n = entry.get("name")
            if n not in batch_set:
                continue
            desc = mapping.get(n)
            if not desc or not isinstance(desc, str):
                continue
            if entry.get("description") == PLACEHOLDER_DESCRIPTION:
                entry["description"] = desc.strip()
                patched += 1

        if patched > 0:
            _atomic_write_vocab(vocab_path, data)
            described += patched
            logger.info(
                "vocab_repair: described %d stubs in batch starting at index %d",
                patched,
                batch_start,
            )

    return described
