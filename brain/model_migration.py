"""Folding the retired per-persona model settings into the install-wide model settings in tunables.toml (#529). Permanent backward compatibility.

Before this change each persona's ``persona_config.json`` held a chat ``model`` and optional ``tier_models`` /
``derived_unavailable``. The model is now one install-wide setting per job in ``tunables.toml``
(``brain.bridge.model_tier``). There is no way to know which version an install is upgrading from, now or later, so
this module stays: whenever a persona config in the old format is found, it is brought forward the first time it is
seen, and an old file that turns up later (an imported or restored persona) is handled the same way.

``migrate_persona_models(home, persona_dir)`` runs (a) in the bridge lifespan after the tunables defaults rewrite and
before ``touch_last_opened``, and (b) at the start of the CLI write commands. It has two steps.

1. DECISION (only while the marker ``<home>/model_migration.done`` is absent). Every persona's config is read
   read-only BEFORE the migration lock is taken (a damaged file is skipped with a log, never healed). Candidates are
   personas on the Claude provider kind with a valid saved ``model`` or valid ``tier_models`` (a persona on another
   provider kind never chose a Claude model, so its saved value is never applied). With no candidate nothing is written
   and NO marker is written, so a persona imported later still goes through the decision. Otherwise the candidate with
   the newest ``last_opened_at`` wins (the file's mtime only when that field is absent or unusable; a tie goes to the
   first name alphabetically) and its values are written with ``set_job_models(..., only_if_default=True)`` (a job whose
   value already differs from its shipped default was changed by the user and is never overwritten); a value equal to the job's shipped default is not written, so a saved
   ``sonnet`` follows the shipped default. "Most recently used" is not "model most recently changed"; the decision
   runs before the launching persona's ``last_opened_at`` is touched, so a more recently used OTHER persona can win.
   That is the stated rule.
2. CLEARING (for the launching persona, on EVERY start until its legacy keys are gone, marker or not). A persona the
   decision above did not consider (its config appeared after the decision, or it was skipped by the scan) has its valid
   Claude values applied the same way: written with ``only_if_default=True``, so an install whose value for the job still
   equals the shipped default takes the persona's value, and an install that has changed it keeps it and the persona's value is discarded
   with a log line. If that write cannot be made nothing is cleared (the next start retries). Then the persona's own
   legacy keys are removed from its own file, and every saved model text removed without being applied is logged (an
   unusable text at WARNING, a valid one at INFO), so no saved value ever disappears without a trace. The cost per
   start is one read of the launching persona's own file and a key-presence test. Other personas' files are never
   rewritten by this process; each clears on its own start.

Lock order: this module's lock (``model_migration.done.lock``) is taken first and only around the short write section;
inside it ``set_job_models`` takes the tunables lock (the late write of step 2 takes only the tunables lock). Both waits
are bounded; on a timeout the decision logs and returns (no marker, no clearing), and the next start retries. It never
raises into startup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain import dev_constants, paths
from brain.bridge.model_tier import (
    CLAUDE_TIERS,
    is_valid_model,
    resolve_choice,
    set_job_models,
    shipped_default,
    unsupported_display,
)
from brain.health.json_io import JsonFileError, Recovery, read_json, read_json_result, write_json
from brain.persona_config import DEFAULT_PROVIDER, KNOWN_PROVIDERS, LEGACY_MODEL_KEYS
from brain.tunables import TunablesWriteError
from brain.utils.file_lock import LockTimeoutError, locked_within
from brain.utils.time import lenient_utc

logger = logging.getLogger(__name__)

MARKER_NAME = "model_migration.done"
_STALE_STAMP = "derived_model_recheck.stamp"
_CHAT = "interactive-chat"
_CLAUDE_PROVIDER = "claude-cli"


@dataclass(frozen=True)
class _Candidate:
    name: str
    chat: str | None  # a valid saved chat model
    tiers: dict[str, str]  # valid saved per-job models
    stamp: datetime
    signal: str  # "last_opened_at" or "file mtime"


def _valid_legacy(settings: dict[str, Any]) -> tuple[str | None, dict[str, str]]:
    """The valid values among a persona's legacy keys: (chat model, {job: model})."""
    raw_chat = settings.get("model")
    chat = raw_chat if is_valid_model(raw_chat) else None
    tiers: dict[str, str] = {}
    raw_tiers = settings.get("tier_models")
    if isinstance(raw_tiers, dict):
        for job, value in raw_tiers.items():
            if job in CLAUDE_TIERS and job != _CHAT and is_valid_model(value):
                tiers[job] = value
    return chat, tiers


def _provider_kind(data: dict[str, Any]) -> str:
    """The persona's provider kind as ``PersonaConfig`` reads it: an unknown or absent value is the default kind."""
    raw = data.get("provider")
    return raw if isinstance(raw, str) and raw in KNOWN_PROVIDERS else DEFAULT_PROVIDER


def _scan_candidates(skipped: list[str]) -> list[_Candidate]:
    """Read every persona's config read-only; the personas worth considering, newest first."""
    found: list[_Candidate] = []
    for name in paths.list_persona_names():
        path = paths.get_persona_dir(name) / "persona_config.json"
        try:
            res = read_json_result(path, default=dict, recover=Recovery.CONTENT, expect=dict)
        except JsonFileError as exc:  # an I/O fault that persisted
            logger.warning("model migration: could not read persona %s's config (%s); skipped", name, exc)
            skipped.append(name)
            continue
        if res.outcome == "defaulted":
            logger.info("model migration: persona %s's config is damaged (%s); skipped", name, res.kind)
            skipped.append(name)
            continue
        if res.outcome != "ok":
            continue  # no config file: nothing to migrate
        if _provider_kind(res.data) != _CLAUDE_PROVIDER:
            continue  # its saved model was never a Claude choice
        chat, tiers = _valid_legacy({k: res.data[k] for k in LEGACY_MODEL_KEYS if k in res.data})
        if chat is None and not tiers:
            continue
        stamp = lenient_utc(res.data.get("last_opened_at"))
        signal = "last_opened_at"
        if stamp is None:
            signal = "file mtime"
            try:
                stamp = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            except OSError:
                stamp = datetime.fromtimestamp(0, UTC)
        found.append(_Candidate(name, chat, tiers, stamp, signal))
    return sorted(found, key=lambda c: (-c.stamp.timestamp(), c.name))


def _winner_updates(candidates: list[_Candidate]) -> dict[str, str]:
    """{job: id to write}: per job the newest candidate that holds a value, mapped, default-equal values dropped."""
    updates: dict[str, str] = {}
    chosen: dict[str, tuple[str, str]] = {}  # job -> (persona name, saved value)
    for cand in candidates:  # newest first
        if cand.chat is not None and _CHAT not in chosen:
            chosen[_CHAT] = (cand.name, cand.chat)
        for job, value in cand.tiers.items():
            chosen.setdefault(job, (cand.name, value))
    for job, (name, value) in chosen.items():
        resolved = resolve_choice(value)
        if resolved is None or resolved == shipped_default(job):
            logger.info("model migration: %s: persona %s's %r equals the shipped default, nothing to write", job, name, value)
            continue
        updates[job] = resolved
    return updates


def _marker_path(home: Path) -> Path:
    return home / MARKER_NAME


def _decide(home: Path) -> tuple[bool, frozenset[str]]:
    """The decision step: (a decision exists now, the personas THIS call's decision considered).

    The second item is empty when the marker was already there (an earlier call decided) or nothing was decided.
    """
    marker = _marker_path(home)
    if marker.exists():
        return True, frozenset()
    skipped: list[str] = []
    candidates = _scan_candidates(skipped)  # BEFORE the lock: the hold below stays short by construction
    if not candidates:
        return False, frozenset()  # nothing to decide; no marker, so a persona imported later is still migrated
    for cand in candidates:
        logger.info(
            "model migration: candidate %s: chat=%r tier_models=%r (%s %s)",
            cand.name, cand.chat, cand.tiers, cand.signal, cand.stamp.isoformat(),
        )
    updates = _winner_updates(candidates)
    considered = frozenset(c.name for c in candidates)
    try:
        with locked_within(marker, dev_constants.SETTINGS_FILE_LOCK_WAIT_S):
            if marker.exists():
                return True, frozenset()  # another process decided while this one was reading
            written = set_job_models(dict(updates), only_if_default=True) if updates else {}
            skipped_existing = sorted(set(updates) - set(written))
            if skipped_existing:
                logger.info("model migration: %s already hold a setting; left as they are", skipped_existing)
            write_json(
                marker,
                {
                    "decided_at": datetime.now(UTC).isoformat(),
                    "source_persona": candidates[0].name,
                    "keys_written": sorted(written),
                    "personas_skipped": sorted(skipped),
                },
                indent=2,
                fsync=False,
            )
            logger.info("model migration: decided; wrote %s", sorted(written) or "nothing (all equal the defaults)")
            return True, considered
    except LockTimeoutError:
        logger.warning("model migration: another process holds the migration lock; retried at the next start")
    except TunablesWriteError as exc:
        logger.warning("model migration: could not write the model settings (%s); retried at the next start", exc)
    except OSError as exc:
        logger.warning("model migration: could not record the decision (%s); retried at the next start", exc)
    return False, frozenset()


def _apply_late(name: str, chat: str | None, tiers: dict[str, str]) -> bool:
    """Apply a persona's valid saved values that the decision did not consider: written only where the install's value still equals the
    shipped default, otherwise discarded with a log. False when the write could not be made (nothing may be cleared)."""
    cand = _Candidate(name, chat, tiers, datetime.fromtimestamp(0, UTC), "late")
    updates = _winner_updates([cand])
    try:
        written = set_job_models(dict(updates), only_if_default=True) if updates else {}
    except (TunablesWriteError, LockTimeoutError, OSError) as exc:
        logger.warning("model migration: could not apply persona %s's saved model (%s); retried at the next start", name, exc)
        return False
    if written:
        logger.info("model migration: persona %s's saved model applied to %s (the install had no setting)", name, sorted(written))
    for job in sorted(set(updates) - set(written)):
        logger.info(
            "model migration: persona %s's saved model %s for %s was not applied (the install's value was already changed) and is discarded",
            name, updates[job], job,
        )
    return True


def _clear_persona(persona_dir: Path, decision_exists: bool, considered: frozenset[str]) -> None:
    """The clearing step for ONE persona's own file: its legacy keys come out, every other key stays exactly as it is
    (the file is edited as parsed JSON, not rebuilt from the dataclass, so nothing else is normalised). A Claude persona
    the decision did not consider has its valid values applied first (``_apply_late``)."""
    from brain.health.adaptive import compute_treatment

    path = persona_dir / "persona_config.json"
    if not path.exists():
        return
    data = read_json(path, recover=Recovery.RAISE, expect=dict)
    legacy = {k: data[k] for k in LEGACY_MODEL_KEYS if k in data}
    if not legacy:
        return
    chat, tiers = _valid_legacy(legacy)
    has_valid = chat is not None or bool(tiers)
    applicable = has_valid and _provider_kind(data) == _CLAUDE_PROVIDER
    if applicable and not decision_exists:
        return  # the decision failed: nothing is lost, the next start retries
    applied = False
    if applicable and persona_dir.name not in considered:
        if not _apply_late(persona_dir.name, chat, tiers):
            return
        applied = True
    if not applied and ("model" in legacy or "tier_models" in legacy):
        # An unusable text (a typo), a value the decision above already dealt with (its winner was written, a loser is
        # logged there), or a persona on another provider kind: the text is removed from the file, so the log is where
        # it stays visible.
        shown = ", ".join(f"{key}={unsupported_display(legacy[key])}" for key in ("model", "tier_models") if key in legacy)
        logger.log(
            logging.INFO if has_valid else logging.WARNING,
            "model migration: removed the retired saved model settings from persona %s's file (%s)%s",
            persona_dir.name, shown, "" if has_valid else "; the text was not a usable model and is not applied",
        )
    for key in legacy:
        del data[key]
    treatment = compute_treatment(path.parent, path.name)
    write_json(path, data, indent=2, backups=treatment.backup_count, verify=treatment.verify_after_write)
    try:
        paths.cadence_state_path(persona_dir, _STALE_STAMP).unlink(missing_ok=True)
    except OSError:
        pass  # best effort: a stale stamp file is harmless


def migrate_persona_models(home: Path, persona_dir: Path | None) -> None:
    """Run the decision step and, for ``persona_dir`` (None for a CLI call), the clearing step. Never raises."""
    try:
        decision_exists, considered = _decide(home)
        if persona_dir is not None:
            _clear_persona(Path(persona_dir), decision_exists, considered)
    except Exception:  # noqa: BLE001 - startup and the CLI must never fail on this
        logger.warning("model migration failed (ignored; retried at the next start)", exc_info=True)


def pending_legacy() -> bool:
    """Read-only: True while some Claude persona's config still holds a valid saved model (applied or discarded when
    that persona next starts, or by the first decision)."""
    try:
        return bool(_scan_candidates([]))
    except Exception:  # noqa: BLE001
        return False
