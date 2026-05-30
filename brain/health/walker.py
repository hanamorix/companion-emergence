"""Proactive walk over every persona file — used by `nell health check`
and triggered automatically when a heartbeat tick produces >=2 anomalies."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from brain.health.anomaly import BrainAnomaly, BrainIntegrityError
from brain.health.attempt_heal import attempt_heal

# Atomic-rewrite files this walker checks. Each entry: filename -> default dict.
#
# When the soul module lands as a Phase 2a-extension, add `soul.json` here
# with default `{"version": 1, "crystallizations": []}` (or whatever the
# soul module's schema settles on). See spec §9.1 for the full plan.
_DEFAULTS: dict[str, dict] = {
    "user_preferences.json": {"dream_every_hours": 24.0},
    "persona_config.json": {"provider": "claude-cli", "searcher": "ddgs"},
    "heartbeat_config.json": {},  # all-default HeartbeatConfig
    "heartbeat_state.json": {},  # triggers fresh-init via load fallback
    "interests.json": {"version": 1, "interests": []},
    "reflex_arcs.json": {"version": 1, "arcs": []},
    "emotion_vocabulary.json": {"version": 1, "emotions": []},
}


# Text identity files checked separately (plain-text, not JSON).
# Two semantics:
#   _AUTO_CREATE_TEXT_FILES — auto-generated from default template if missing.
#     Currently only voice.md (chat engine relies on it being present after
#     first call). Walker delegates to load_voice() which owns the template.
#   _OPTIONAL_TEXT_FILES — optional reference docs. Skip silently if missing,
#     heal-from-bak if corrupt (empty), no auto-create. Used for files like
#     voicecraft.md that exist only when the persona's author chose to write
#     them.
_AUTO_CREATE_TEXT_FILES: tuple[str, ...] = ("voice.md",)
_OPTIONAL_TEXT_FILES: tuple[str, ...] = ("voicecraft.md",)


def walk_persona(persona_dir: Path) -> list[BrainAnomaly]:
    """Check every persona file. Heal what's healable; report anomalies.

    Iterates atomic-rewrite files via attempt_heal (captures corruption +
    repairs in one shot) then opens each SQLite store so the constructor's
    PRAGMA integrity_check runs. Returns an empty list when everything is
    healthy.
    """
    anomalies: list[BrainAnomaly] = []

    # Atomic-rewrite file scan — d=default captures the dict at iteration time
    # to avoid Python's late-binding closure gotcha.
    for name, default in _DEFAULTS.items():
        path = persona_dir / name
        _, anomaly = attempt_heal(path, default_factory=lambda d=default: d)
        if anomaly is not None:
            anomalies.append(anomaly)

    # Auto-creating text identity scan (voice.md). load_voice() owns the
    # default template and the attempt_heal_text invocation.
    for name in _AUTO_CREATE_TEXT_FILES:
        path = persona_dir / name
        if not path.exists():
            continue  # Missing voice.md is fine — created on first chat turn.
        from brain.chat.voice import load_voice

        _, anomaly = load_voice(persona_dir)
        if anomaly is not None:
            anomalies.append(anomaly)
        break  # load_voice checks voice.md by name; only one auto-create file

    # Optional text reference scan (voicecraft.md, future doc files).
    # Skip silently if missing. Heal from .bak if empty/corrupt; no auto-create.
    from brain.health.attempt_heal import attempt_heal_text

    for name in _OPTIONAL_TEXT_FILES:
        path = persona_dir / name
        if not path.exists():
            continue  # Optional file; absence is not an anomaly.
        # Pass an empty-string default — when all baks are corrupt, the heal
        # rewrites an empty file rather than fabricating content. The anomaly
        # signals the loss; the user can restore the original from VCS.
        _, anomaly = attempt_heal_text(path, default_factory=lambda: "")
        if anomaly is not None:
            anomalies.append(anomaly)

    # SQLite integrity — constructors run PRAGMA integrity_check where available;
    # works.db is checked explicitly because WorksStore is optimized for hot
    # request paths and does not deep-scan on every init.
    for db_name in ("memories.db", "hebbian.db", "crystallizations.db", "data/works.db"):
        db_path = persona_dir / db_name
        if not db_path.exists():
            continue
        try:
            if db_name == "memories.db":
                from brain.memory.store import MemoryStore

                MemoryStore(db_path=db_path).close()
            elif db_name == "hebbian.db":
                from brain.memory.hebbian import HebbianMatrix

                HebbianMatrix(db_path=db_path).close()
            else:  # crystallizations.db
                if db_name == "crystallizations.db":
                    from brain.soul.store import SoulStore

                    SoulStore(db_path=db_path).close()
                else:  # data/works.db
                    with sqlite3.connect(str(db_path)) as conn:
                        result = conn.execute("PRAGMA integrity_check").fetchall()
                    if result != [("ok",)]:
                        detail = "; ".join(str(row[0]) for row in result)
                        raise BrainIntegrityError(str(db_path), detail)
                    from brain.works.store import WorksStore

                    WorksStore(db_path).schema_version()
        except (BrainIntegrityError, sqlite3.DatabaseError) as exc:
            detail = exc.detail if isinstance(exc, BrainIntegrityError) else str(exc)
            anomalies.append(
                BrainAnomaly(
                    timestamp=datetime.now(UTC),
                    file=db_name,
                    kind="sqlite_integrity_fail",
                    action="alarmed_unrecoverable",
                    quarantine_path=None,
                    likely_cause="disk",
                    detail=detail[:500],
                )
            )

    return anomalies
