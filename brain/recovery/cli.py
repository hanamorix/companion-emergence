"""`nell recover` — restore wrongly-forgotten memories."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from brain.paths import get_persona_dir
from brain.recovery.engine import run_recovery


@dataclass(frozen=True)
class RecoverArgs:
    persona: str
    source_dir: Path | None
    force: bool
    dry_run: bool
    json_out: bool


def run_recover_cli(args: RecoverArgs) -> int:
    persona_dir = get_persona_dir(args.persona)
    if not (persona_dir / "memories.db").is_file():
        sys.exit(f"No companion-emergence persona named {args.persona!r} at {persona_dir}")
    report = run_recovery(persona_dir, source_dir=args.source_dir, dry_run=args.dry_run)
    if args.json_out:
        print(report.to_json())
    return 0


def build_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("recover", help="Restore memories wrongly forgotten after a migration.")
    p.add_argument("--persona", required=True, help="Persona to recover.")
    p.add_argument("--from", dest="source_dir", type=Path, default=None,
                   help="Original source persona dir.")
    p.add_argument("--force", action="store_true", help="Proceed even if a bridge looks live.")
    p.add_argument("--dry-run", action="store_true", help="Report what would change; write nothing.")
    p.add_argument("--json", dest="json_out", action="store_true",
                   help="Emit RecoveryReport JSON.")
    p.set_defaults(func=_dispatch)


def _dispatch(args: argparse.Namespace) -> int:
    return run_recover_cli(RecoverArgs(
        persona=args.persona,
        source_dir=args.source_dir,
        force=args.force,
        dry_run=args.dry_run,
        json_out=getattr(args, "json_out", False),
    ))
