"""`nell migrate` subcommand — safety checks + orchestration."""

from __future__ import annotations

import argparse
import json as _json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.migrator.og import FileManifest, OGReader
from brain.migrator.og_interests import extract_interests_from_og, extract_soul_names_best_effort
from brain.migrator.og_reflex import extract_arcs_from_og
from brain.migrator.report import MigrationReport, format_report, write_source_manifest
from brain.migrator.transform import SkippedMemory, transform_memory
from brain.paths import get_persona_dir


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
                og_arcs = extract_arcs_from_og(og_reflex_path)
                reflex_arcs_target.write_text(
                    _json.dumps({"version": 1, "arcs": og_arcs}, indent=2) + "\n",
                    encoding="utf-8",
                )
                reflex_arcs_migrated = len(og_arcs)
            except (ValueError, OSError) as exc:
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
                interests_target.write_text(
                    _json.dumps({"version": 1, "interests": og_interests}, indent=2) + "\n",
                    encoding="utf-8",
                )
                interests_migrated = len(og_interests)
            except (ValueError, FileNotFoundError, OSError) as exc:
                interests_skipped_reason = f"migrate_error: {exc}"
    else:
        interests_skipped_reason = "og_nell_interests_json_not_found"

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
    )
    write_source_manifest(work_dir / "source-manifest.json", manifest)
    report_text = format_report(report)
    (work_dir / "migration-report.md").write_text(report_text, encoding="utf-8")
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
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(
                f"{kind.capitalize()} is non-empty: {path}. Pass --force to overwrite."
            )
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def _verify_sources_unchanged(og_dir: Path, manifest: list[FileManifest]) -> None:
    """Re-stat each source file; abort if any size differs from manifest."""
    for m in manifest:
        path = og_dir / m.relative_path
        st = path.stat()
        if st.st_size != m.size_bytes:
            raise RuntimeError(
                f"Source file {path} changed size during migration "
                f"(was {m.size_bytes}, now {st.st_size}). Aborting."
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
