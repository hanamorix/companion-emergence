"""`nell recover` — restore wrongly-forgotten memories."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from brain.bridge.state_file import pid_is_alive
from brain.paths import get_persona_dir
from brain.recovery.engine import run_recovery
from brain.recovery.report import format_report


@dataclass(frozen=True)
class RecoverArgs:
    persona: str
    source_dir: Path | None
    force: bool
    dry_run: bool
    json_out: bool


def _bridge_is_live(persona_dir: Path) -> bool:
    """True if a bridge process appears to be running for this persona.
    Defensive: a missing or malformed bridge.json reads as not-live."""
    bridge = persona_dir / "bridge.json"
    if not bridge.is_file():
        return False
    try:
        pid = int(json.loads(bridge.read_text()).get("pid", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return pid_is_alive(pid)


def run_recover_cli(args: RecoverArgs) -> int:
    persona_dir = get_persona_dir(args.persona)
    if not (persona_dir / "memories.db").is_file():
        sys.exit(f"No companion-emergence persona named {args.persona!r} at {persona_dir}")
    if not args.dry_run and not args.force and _bridge_is_live(persona_dir):
        sys.exit(
            "Bridge appears to be running — stop it or pass --force "
            "(recovery must not race the forgetting pass)."
        )
    report = run_recovery(persona_dir, source_dir=args.source_dir, dry_run=args.dry_run)
    if args.json_out or not sys.stdout.isatty():
        print(report.to_json())
    else:
        print(format_report(report))
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
