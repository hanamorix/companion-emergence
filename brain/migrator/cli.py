"""`nell migrate` subcommand — safety checks + orchestration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.state_file import pid_is_alive
from brain.emotion import vocabulary as _vocabulary
from brain.health.attempt_heal import save_with_backup_text
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.migrator.og import FileManifest, OGReader
from brain.migrator.og_interests import extract_interests_from_og, extract_soul_names_best_effort
from brain.migrator.og_journal_dna import migrate_creative_dna, migrate_journal_memories
from brain.migrator.og_legacy import migrate_legacy_files
from brain.migrator.og_reflex import migrate_reflex_arcs
from brain.migrator.og_reflex_log import migrate_reflex_log
from brain.migrator.og_soul import extract_crystallizations_from_og
from brain.migrator.og_soul_candidates import migrate_soul_candidates
from brain.migrator.og_vocabulary import extract_persona_vocabulary
from brain.migrator.report import MigrationReport, format_report, write_source_manifest
from brain.migrator.transform import SkippedMemory, transform_memory
from brain.paths import get_persona_dir

_MIGRATOR_MARKER_FILES = frozenset({"migration-report.md", "source-manifest.json", "memories.db"})


def _migrate_lock_path(target: Path) -> Path:
    """Lockfile path for a migration target.

    Lives as a sibling of the target so rmtree of the target dir (the
    --install-as flow blows away <name>.new before re-creating it) doesn't
    take the lock with it.
    """
    return target.parent / f".{target.name}.migrate.lock"


def _acquire_migrate_lock(lock_path: Path) -> int:
    """Atomically acquire the migrate lock for a target. Returns open fd.

    Raises RuntimeError if another live process holds the lock. Stale locks
    (PID is dead, or contents are malformed) are taken over silently.

    Mirrors the brain.bridge.daemon.acquire_lock pattern: O_CREAT | O_EXCL,
    PID-liveness check on conflict.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing_pid: int | None = None
        try:
            first_line = lock_path.read_text(encoding="utf-8").splitlines()[0]
            existing_pid = int(first_line.strip())
        except (OSError, ValueError, IndexError):
            existing_pid = None
        if existing_pid is None or not pid_is_alive(existing_pid):
            # Stale or unreadable — take over.
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            return _acquire_migrate_lock(lock_path)
        raise RuntimeError(
            f"another migrate is in progress: pid={existing_pid} holds {lock_path}. "
            f"If you're sure no other migrate is running, remove the lockfile."
        ) from None
    payload = f"{os.getpid()}\n{datetime.now(UTC).isoformat()}\n"
    os.write(fd, payload.encode("utf-8"))
    return fd


def _release_migrate_lock(lock_path: Path, fd: int) -> None:
    """Release a migrate lock acquired via _acquire_migrate_lock.

    Idempotent against double-release; tolerates the file having been removed
    out from under us.
    """
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


@dataclass(frozen=True)
class MigrateArgs:
    """Validated argument bundle for run_migrate."""

    input_dir: Path
    output_dir: Path | None
    install_as: str | None
    force: bool

    def __post_init__(self) -> None:
        if (self.output_dir is None) == (self.install_as is None):
            raise ValueError("Exactly one of --output or --install-as must be provided.")


def run_migrate(args: MigrateArgs) -> MigrationReport:
    """Execute a full migration. Returns the MigrationReport."""
    reader = OGReader(args.input_dir)
    reader.check_preflight()

    # Determine the lock target. For --output the target is the output dir
    # itself; for --install-as it's the final persona dir (NOT the .new
    # working dir, since that gets rmtree'd inside the lock-held region).
    if args.output_dir is not None:
        lock_target = args.output_dir
    else:
        assert args.install_as is not None
        lock_target = get_persona_dir(args.install_as)
    lock_path = _migrate_lock_path(lock_target)
    lock_fd = _acquire_migrate_lock(lock_path)
    try:
        return _run_migrate_locked(args, reader)
    finally:
        _release_migrate_lock(lock_path, lock_fd)


def _run_migrate_locked(args: MigrateArgs, reader: OGReader) -> MigrationReport:
    """Migration body — runs under the migrate lock."""
    # Determine the write directory.
    if args.output_dir is not None:
        work_dir = args.output_dir
        _ensure_clobber_safe(work_dir, args.force, kind="output directory")
        work_dir.mkdir(parents=True, exist_ok=True)
        finalise_target: Path | None = None
    else:
        assert args.install_as is not None
        final_dir = get_persona_dir(args.install_as)
        if final_dir.exists() and not args.force:
            raise FileExistsError(
                f"Persona directory already exists: {final_dir}. "
                "Pass --force to back up and overwrite."
            )
        # Write to <name>.new alongside the final dir for atomic rename.
        work_dir = final_dir.with_name(f"{args.install_as}.new")
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        finalise_target = final_dir

    started = time.monotonic()

    # ---- memories ----
    # try/finally around store.close() so a mid-loop exception on disk-backed
    # DBs doesn't leak a .db-wal file for the next run to recover from.
    og_memories = reader.read_memories()
    store = MemoryStore(db_path=work_dir / "memories.db")
    migrated_count = 0
    skipped: list[SkippedMemory] = []
    seen_ids: set[str] = set()
    try:
        for og_mem in og_memories:
            mem, sk = transform_memory(og_mem)
            if sk is not None:
                skipped.append(sk)
                continue
            assert mem is not None
            if mem.id in seen_ids:
                skipped.append(
                    SkippedMemory(
                        id=mem.id,
                        reason="duplicate_id",
                        field="id",
                        raw_snippet=mem.content[:120],
                    )
                )
                continue
            seen_ids.add(mem.id)
            store.create(mem)
            migrated_count += 1
    finally:
        store.close()

    # ---- hebbian ----
    hebbian = HebbianMatrix(db_path=work_dir / "hebbian.db")
    edges_migrated = 0
    try:
        for a, b, w in reader.iter_nonzero_upper_edges():
            hebbian.strengthen(a, b, delta=float(w))
            edges_migrated += 1
    finally:
        hebbian.close()

    # ---- vocabulary ----
    vocab_target = work_dir / "emotion_vocabulary.json"
    vocabulary_emotions_migrated = 0
    vocabulary_skipped_reason: str | None = None

    if vocab_target.exists() and not args.force:
        vocabulary_skipped_reason = "existing_file_not_overwritten"
    else:
        try:
            framework_baseline_names = {e.name for e in _vocabulary._BASELINE}
            og_memories_for_vocab = reader.read_memories()
            vocab_entries = extract_persona_vocabulary(
                og_memories_for_vocab,
                framework_baseline_names=framework_baseline_names,
            )
            _vocab_tmp = vocab_target.with_suffix(vocab_target.suffix + ".new")
            _vocab_tmp.write_text(
                json.dumps({"version": 1, "emotions": vocab_entries}, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(_vocab_tmp, vocab_target)
            vocabulary_emotions_migrated = len(vocab_entries)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            vocabulary_skipped_reason = f"migrate_error: {exc}"

    # ---- reflex arcs ----
    # --input points at OG's data/ dir, but reflex_engine.py sits one level
    # up at NellBrain root. Try both locations so the migrator works whether
    # the user points at NellBrain/ or NellBrain/data/.
    _candidate_reflex_paths = [
        args.input_dir / "reflex_engine.py",
        args.input_dir.parent / "reflex_engine.py",
    ]
    og_reflex_path = next((p for p in _candidate_reflex_paths if p.exists()), None)

    reflex_arcs_target = work_dir / "reflex_arcs.json"
    reflex_arcs_migrated = 0
    reflex_arcs_skipped_reason: str | None = None

    if og_reflex_path is not None:
        if reflex_arcs_target.exists() and not args.force:
            reflex_arcs_skipped_reason = "existing_file_not_overwritten"
        else:
            try:
                og_arcs = migrate_reflex_arcs(
                    persona_dir=work_dir,
                    og_reflex_engine_path=og_reflex_path,
                    force=args.force,
                )
                reflex_arcs_migrated = len(og_arcs)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                reflex_arcs_skipped_reason = f"extract_error: {exc}"
    else:
        reflex_arcs_skipped_reason = "og_reflex_engine_py_not_found"

    # ---- interests ----
    _candidate_interests_paths = [
        args.input_dir / "nell_interests.json",
        args.input_dir / "data" / "nell_interests.json",
        args.input_dir.parent / "nell_interests.json",
    ]
    og_interests_path = next((p for p in _candidate_interests_paths if p.exists()), None)
    interests_target = work_dir / "interests.json"
    interests_migrated = 0
    interests_skipped_reason: str | None = None

    if og_interests_path is not None:
        if interests_target.exists() and not args.force:
            interests_skipped_reason = "existing_file_not_overwritten"
        else:
            try:
                soul_names = extract_soul_names_best_effort(args.input_dir)
                og_interests = extract_interests_from_og(og_interests_path, soul_names=soul_names)
                _interests_tmp = interests_target.with_suffix(interests_target.suffix + ".new")
                _interests_tmp.write_text(
                    json.dumps({"version": 1, "interests": og_interests}, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(_interests_tmp, interests_target)
                interests_migrated = len(og_interests)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                interests_skipped_reason = f"migrate_error: {exc}"
    else:
        interests_skipped_reason = "og_nell_interests_json_not_found"

    # ---- soul crystallizations ----
    # nell_soul.json lives in the OG data/ dir (same dir as memories_v2.json).
    # Unlike reflex/interests we don't need a parent-dir fallback — the OG
    # soul file has a single canonical location.
    from brain.soul.store import SoulStore

    soul_db_path = work_dir / "crystallizations.db"
    crystallizations_migrated = 0
    crystallizations_skipped_reason: str | None = None

    if soul_db_path.exists() and not args.force:
        crystallizations_skipped_reason = "existing_file_not_overwritten"
    else:
        soul_crystals, soul_skipped = extract_crystallizations_from_og(args.input_dir)
        soul_store = SoulStore(db_path=soul_db_path)
        try:
            for crystal in soul_crystals:
                soul_store.create(crystal)
            crystallizations_migrated = len(soul_crystals)
        finally:
            soul_store.close()
        # soul_skipped entries are logged by og_soul's module-level logger if desired

    # ---- creative_dna ----
    creative_dna_migrated = False
    creative_dna_skipped_reason: str | None = None
    try:
        creative_dna_migrated = migrate_creative_dna(
            persona_dir=work_dir,
            og_root=args.input_dir.parent,
        )
        if not creative_dna_migrated:
            creative_dna_skipped_reason = "og file not present"
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        creative_dna_skipped_reason = f"migrate_error: {exc}"

    # ---- journal memories (reflex_journal -> journal_entry retag) ----
    journal_memories_retagged = 0
    journal_memories_skipped_reason: str | None = None
    journal_store = MemoryStore(db_path=work_dir / "memories.db")
    try:
        journal_memories_retagged = migrate_journal_memories(
            persona_dir=work_dir,
            store=journal_store,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        journal_memories_skipped_reason = f"migrate_error: {exc}"
    finally:
        journal_store.close()

    # ---- legacy preservation ----
    legacy_files_preserved = 0
    legacy_files_missing = 0
    legacy_integrity_issues: list[str] = []
    legacy_skipped_reason: str | None = None
    try:
        preserved, missing, integrity_issues = migrate_legacy_files(
            og_data_dir=args.input_dir,
            persona_dir=work_dir,
        )
        legacy_files_preserved = len(preserved)
        legacy_files_missing = len(missing)
        # Format integrity issues as report-ready strings. Non-fatal: the
        # legacy preserve-broken-bytes design intent stands; this is a
        # diagnostic surfacing the rare COPY-time truncation case.
        legacy_integrity_issues = [
            f"{i.name}: src {i.src_size}B sha256={i.src_sha256[:12]} != "
            f"dest {i.dest_size}B sha256={i.dest_sha256[:12]}"
            for i in integrity_issues
        ]
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        legacy_skipped_reason = f"copy_error: {exc}"

    # ---- soul_candidates schema migration ----
    soul_candidates_migrated = 0
    soul_candidates_skipped_missing_memory_id = 0
    soul_candidates_skipped_reason: str | None = None
    try:
        sc_migrated, sc_skipped = migrate_soul_candidates(
            og_data_dir=args.input_dir,
            persona_dir=work_dir,
        )
        soul_candidates_migrated = sc_migrated
        soul_candidates_skipped_missing_memory_id = sc_skipped
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        soul_candidates_skipped_reason = f"migrate_error: {exc}"

    # ---- reflex_log schema migration ----
    reflex_log_fires_migrated = 0
    reflex_log_skipped_reason: str | None = None
    try:
        reflex_log_fires_migrated = migrate_reflex_log(
            og_data_dir=args.input_dir,
            persona_dir=work_dir,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        reflex_log_skipped_reason = f"migrate_error: {exc}"

    elapsed = time.monotonic() - started

    # ---- post-run source re-stat (detect OG mutation during the run) ----
    manifest = reader.manifest()
    _verify_sources_unchanged(args.input_dir, manifest)

    # ---- report + manifest artefacts ----
    inspect_cmds = _inspect_cmds(work_dir) if args.output_dir is not None else []
    install_cmd = _install_cmd(args.input_dir) if args.output_dir is not None else ""
    report = MigrationReport(
        memories_migrated=migrated_count,
        memories_skipped=skipped,
        edges_migrated=edges_migrated,
        edges_skipped=0,
        elapsed_seconds=elapsed,
        source_manifest=manifest,
        next_steps_inspect_cmds=inspect_cmds,
        next_steps_install_cmd=install_cmd,
        reflex_arcs_migrated=reflex_arcs_migrated,
        reflex_arcs_skipped_reason=reflex_arcs_skipped_reason,
        interests_migrated=interests_migrated,
        interests_skipped_reason=interests_skipped_reason,
        vocabulary_emotions_migrated=vocabulary_emotions_migrated,
        vocabulary_skipped_reason=vocabulary_skipped_reason,
        crystallizations_migrated=crystallizations_migrated,
        crystallizations_skipped_reason=crystallizations_skipped_reason,
        creative_dna_migrated=creative_dna_migrated,
        creative_dna_skipped_reason=creative_dna_skipped_reason,
        journal_memories_retagged=journal_memories_retagged,
        journal_memories_skipped_reason=journal_memories_skipped_reason,
        legacy_files_preserved=legacy_files_preserved,
        legacy_files_missing=legacy_files_missing,
        legacy_skipped_reason=legacy_skipped_reason,
        legacy_integrity_issues=legacy_integrity_issues,
        soul_candidates_migrated=soul_candidates_migrated,
        soul_candidates_skipped_missing_memory_id=soul_candidates_skipped_missing_memory_id,
        soul_candidates_skipped_reason=soul_candidates_skipped_reason,
        reflex_log_fires_migrated=reflex_log_fires_migrated,
        reflex_log_skipped_reason=reflex_log_skipped_reason,
    )
    write_source_manifest(work_dir / "source-manifest.json", manifest)
    report_text = format_report(report)
    # Atomic write: a SIGINT between print() and a torn write_text would leave
    # a half-written migration-report.md and the operator wouldn't know the
    # SQLite DBs already committed successfully.
    save_with_backup_text(work_dir / "migration-report.md", report_text)
    print(report_text)

    # ---- finalise install-as with backup + atomic rename ----
    if finalise_target is not None:
        if finalise_target.exists():
            ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")
            backup = finalise_target.with_name(f"{finalise_target.name}.backup-{ts}")
            os.rename(finalise_target, backup)
        finalise_target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(work_dir, finalise_target)

    return report


def _ensure_clobber_safe(path: Path, force: bool, kind: str) -> None:
    """Refuse to clobber a non-empty dir without --force, and even with --force
    refuse to clobber a directory that doesn't look like a prior migration
    target (no migration-report.md / source-manifest.json / memories.db).

    `kind` is a short noun phrase used in the error message (e.g.
    "output directory"). Lowercase by contract — the function uppercases
    the first character only, so mid-word capitalization is preserved.
    """
    label = kind[:1].upper() + kind[1:] if kind else kind
    if not path.exists():
        return
    if not any(path.iterdir()):
        return  # empty dir is safe to use
    if not force:
        raise FileExistsError(
            f"{label} is non-empty: {path}. Pass --force to overwrite."
        )
    # --force: only clobber if the directory looks like a prior migration target
    has_marker = any((path / m).exists() for m in _MIGRATOR_MARKER_FILES)
    if not has_marker:
        raise FileExistsError(
            f"{label} {path} is not empty and does not contain any of "
            f"{sorted(_MIGRATOR_MARKER_FILES)} — refusing to clobber an "
            f"unrelated directory even with --force. Choose a different "
            f"output path or remove the directory first."
        )
    # --force + has marker: safe to clobber
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _verify_sources_unchanged(og_dir: Path, manifest: list[FileManifest]) -> None:
    """Re-stat + re-hash each source file; abort if size or SHA-256 differs.

    Size mismatch catches the obvious case (file truncated/grown). SHA-256
    catches the harder case where same-byte-length content was mutated
    during the migration window (e.g., a stale OG bridge writing fresh
    JSON of identical length). The hash was computed by OGReader during
    the initial manifest pass; we just compare it.
    """
    import hashlib

    for m in manifest:
        path = og_dir / m.relative_path
        st = path.stat()
        if st.st_size != m.size_bytes:
            raise RuntimeError(
                f"Source file {path} changed size during migration "
                f"(was {m.size_bytes}, now {st.st_size}). Aborting."
            )
        current_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if current_sha != m.sha256:
            raise RuntimeError(
                f"Source file {path} changed content during migration "
                f"(SHA-256 mismatch: was {m.sha256[:12]}..., now "
                f"{current_sha[:12]}...). Aborting."
            )


def _inspect_cmds(work_dir: Path) -> list[str]:
    p = work_dir
    return [
        f'sqlite3 "{p / "memories.db"}" "SELECT COUNT(*) FROM memories;"',
        f'sqlite3 "{p / "memories.db"}" "SELECT domain, COUNT(*) FROM memories GROUP BY domain;"',
        f'sqlite3 "{p / "hebbian.db"}" "SELECT COUNT(*) FROM hebbian_edges;"',
        f'cat "{p / "migration-report.md"}"',
    ]


def _install_cmd(input_dir: Path) -> str:
    return f"uv run nell migrate --input {input_dir} --install-as <persona-name>"


def build_parser(subparsers: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    """Build the `migrate` argparse subparser.

    If `subparsers` is provided, adds to it; otherwise builds a standalone parser.
    """
    if subparsers is not None:
        p = subparsers.add_parser(
            "migrate",
            help="Port OG NellBrain data into a new persona.",
        )
    else:
        p = argparse.ArgumentParser(
            prog="nell migrate",
            description="Port OG NellBrain data into a new persona.",
        )
    p.add_argument(
        "--input",
        dest="input_dir",
        type=Path,
        required=True,
        help="Path to the OG NellBrain data/ directory.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--output",
        dest="output_dir",
        type=Path,
        help="Write migrated artefacts to this directory for inspection.",
    )
    g.add_argument(
        "--install-as",
        dest="install_as",
        type=str,
        help="Install migrated data as this persona name.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite non-empty output dir / existing persona (with backup).",
    )
    p.set_defaults(func=_dispatch)
    return p


def _dispatch(args: argparse.Namespace) -> int:
    """Argparse-compatible handler: convert Namespace → MigrateArgs → run_migrate."""
    migrate_args = MigrateArgs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        install_as=args.install_as,
        force=args.force,
    )
    run_migrate(migrate_args)
    return 0
