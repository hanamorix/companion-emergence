"""Entry point CLI for companion-emergence.

Invoked as `nell <subcommand> [options]`. Week 1 ships `--version`, help,
and a set of stub subcommands that print "not implemented yet" so the CLI
surface is stable while subsequent weeks fill in functionality.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from brain import __version__
from brain.bridge.provider import get_provider
from brain.engines.dream import DreamEngine
from brain.engines.heartbeat import HeartbeatEngine
from brain.engines.reflex import ReflexEngine
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.migrator.cli import build_parser as _build_migrate_parser
from brain.paths import get_persona_dir

# Subcommands the framework plans to ship. Each is a stub in Week 1;
# filled in across Weeks 2-8 as respective modules come online.
_STUB_COMMANDS: tuple[str, ...] = (
    "supervisor",
    "status",
    "rest",
    "soul",
    "memory",
    "works",
)


def _make_stub(name: str) -> Callable[[argparse.Namespace], int]:
    """Factory: build a stub command handler that prints + returns 0.

    The returned handler accepts `args: argparse.Namespace` as required by
    the `args.func(args)` dispatch protocol — stubs don't read it, but the
    signature shape is load-bearing and should not be "cleaned up" to `_args`.
    """

    def _handler(args: argparse.Namespace) -> int:
        print(
            f"nell {name} — not implemented yet. "
            "This subcommand is wired in a future week per the implementation plan."
        )
        return 0

    return _handler


def _dream_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell dream` to the DreamEngine."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir} — "
            f"run `nell migrate --install-as {args.persona}` first."
        )
    # Nested try/finally so a HebbianMatrix open failure still closes the
    # already-open MemoryStore connection. Inline contextmanager would be
    # prettier but stores don't implement __enter__/__exit__ yet.
    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        hebbian = HebbianMatrix(db_path=persona_dir / "hebbian.db")
        try:
            provider = get_provider(args.provider)
            engine = DreamEngine(
                store=store,
                hebbian=hebbian,
                embeddings=None,
                provider=provider,
                log_path=persona_dir / "dreams.log.jsonl",
            )
            result = engine.run_cycle(
                seed_id=args.seed,
                lookback_hours=args.lookback,
                depth=args.depth,
                decay_per_hop=args.decay,
                neighbour_limit=args.limit,
                dry_run=args.dry_run,
            )
        finally:
            hebbian.close()
    finally:
        store.close()

    if args.dry_run:
        print("Dry run — no writes.")
        print(f"Seed: {result.seed.id}  ({result.seed.content[:80]})")
        print(f"Neighbours: {len(result.neighbours)}")
        print(f"Prompt preview:\n{result.prompt[:400]}")
    else:
        print(result.dream_text or "")
    return 0


def _heartbeat_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell heartbeat` to the HeartbeatEngine."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir} — "
            f"run `nell migrate --install-as {args.persona}` first."
        )
    default_arcs_path = Path(__file__).parent / "engines" / "default_reflex_arcs.json"

    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        hebbian = HebbianMatrix(db_path=persona_dir / "hebbian.db")
        try:
            provider = get_provider(args.provider)
            engine = HeartbeatEngine(
                store=store,
                hebbian=hebbian,
                provider=provider,
                state_path=persona_dir / "heartbeat_state.json",
                config_path=persona_dir / "heartbeat_config.json",
                dream_log_path=persona_dir / "dreams.log.jsonl",
                heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
                reflex_arcs_path=persona_dir / "reflex_arcs.json",
                reflex_log_path=persona_dir / "reflex_log.json",
                reflex_default_arcs_path=default_arcs_path,
                persona_name=args.persona,
                persona_system_prompt=f"You are {args.persona}.",
            )
            result = engine.run_tick(trigger=args.trigger, dry_run=args.dry_run)
        finally:
            hebbian.close()
    finally:
        store.close()

    if result.initialized:
        print("Heartbeat initialized — work deferred until next tick.")
    elif args.dry_run:
        print("Heartbeat dry-run — no writes.")
        print(f"  elapsed: {result.elapsed_seconds / 3600:.2f}h")
        print(f"  would decay: {result.memories_decayed} memories")
        print(f"  would prune: {result.edges_pruned} edges")
        # `or "gated"` defends against any future engine refactor that could
        # leave dream_gated_reason=None — prevents literal "dream: None" output.
        print(
            f"  dream: {'would fire' if result.dream_id else (result.dream_gated_reason or 'gated')}"
        )
    else:
        print(f"Heartbeat tick complete ({args.trigger}).")
        print(f"  elapsed: {result.elapsed_seconds / 3600:.2f}h")
        print(f"  decayed: {result.memories_decayed} memories, pruned {result.edges_pruned} edges")
        if result.dream_id:
            print(f"  dream fired: {result.dream_id}")
        else:
            print(f"  dream gated: {result.dream_gated_reason or 'gated'}")
        if result.reflex_fired:
            print(f"  reflex fired: {', '.join(result.reflex_fired)}")
        elif result.reflex_skipped_count > 0:
            print(f"  reflex evaluated ({result.reflex_skipped_count} arc(s) skipped)")
    return 0


def _reflex_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell reflex` to the ReflexEngine."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir} — "
            f"run `nell migrate --install-as {args.persona}` first."
        )

    default_arcs_path = Path(__file__).parent / "engines" / "default_reflex_arcs.json"

    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        provider = get_provider(args.provider)
        engine = ReflexEngine(
            store=store,
            provider=provider,
            persona_name=args.persona,
            persona_system_prompt=f"You are {args.persona}.",
            arcs_path=persona_dir / "reflex_arcs.json",
            log_path=persona_dir / "reflex_log.json",
            default_arcs_path=default_arcs_path,
        )
        result = engine.run_tick(trigger=args.trigger, dry_run=args.dry_run)
    finally:
        store.close()

    if result.dry_run:
        if result.would_fire is not None:
            print(f"Reflex dry-run — would fire: {result.would_fire}.")
        else:
            print("Reflex dry-run — no arc eligible.")
    elif result.arcs_fired:
        fired = result.arcs_fired[0]
        print(f"Reflex fired: {fired.arc_name}")
        print(f"  Memory id: {fired.output_memory_id}")
    else:
        print("Reflex evaluated — no arc fired.")

    if result.arcs_skipped:
        skip_strs = [f"{s.arc_name} ({s.reason})" for s in result.arcs_skipped if s.arc_name]
        if skip_strs:
            print(f"  Skipped: {', '.join(skip_strs)}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser with all stub subcommands."""
    parser = argparse.ArgumentParser(
        prog="nell",
        description=("companion-emergence — CLI for building emotionally aware AI companions"),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"companion-emergence {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", title="subcommands")

    for name in _STUB_COMMANDS:
        sub = subparsers.add_parser(
            name,
            help=f"(stub) {name} — wired in a later week",
        )
        sub.set_defaults(func=_make_stub(name))

    _build_migrate_parser(subparsers)

    dream_sub = subparsers.add_parser(
        "dream",
        help="Run one dream cycle against a persona's memory store.",
    )
    dream_sub.add_argument("--persona", default="nell", help="Persona name (default: nell).")
    dream_sub.add_argument(
        "--seed", default=None, help="Explicit seed memory id (default: auto-select)."
    )
    dream_sub.add_argument(
        "--provider",
        default="claude-cli",
        help="LLM provider: claude-cli (default), fake, ollama.",
    )
    dream_sub.add_argument("--dry-run", action="store_true", help="Skip LLM call and store writes.")
    dream_sub.add_argument(
        "--lookback", type=int, default=24, help="Hours of history to consider (default: 24)."
    )
    dream_sub.add_argument(
        "--depth", type=int, default=2, help="Spreading-activation depth (default: 2)."
    )
    dream_sub.add_argument("--decay", type=float, default=0.5, help="Per-hop decay (default: 0.5).")
    dream_sub.add_argument(
        "--limit", type=int, default=8, help="Max neighbours in prompt (default: 8)."
    )
    dream_sub.set_defaults(func=_dream_handler)

    hb_sub = subparsers.add_parser(
        "heartbeat",
        help="Run one heartbeat orchestrator tick against a persona.",
    )
    hb_sub.add_argument("--persona", default="nell")
    hb_sub.add_argument(
        "--trigger",
        choices=["open", "close", "manual"],
        default="manual",
    )
    hb_sub.add_argument(
        "--provider",
        default="claude-cli",
        help="LLM provider: claude-cli (default), fake, ollama.",
    )
    hb_sub.add_argument("--dry-run", action="store_true")
    hb_sub.set_defaults(func=_heartbeat_handler)

    rf_sub = subparsers.add_parser(
        "reflex",
        help="Run one reflex evaluation tick against a persona.",
    )
    rf_sub.add_argument("--persona", default="nell")
    rf_sub.add_argument(
        "--trigger",
        choices=["open", "close", "manual"],
        default="manual",
    )
    rf_sub.add_argument(
        "--provider",
        default="claude-cli",
        help="LLM provider: claude-cli (default), fake, ollama.",
    )
    rf_sub.add_argument("--dry-run", action="store_true")
    rf_sub.set_defaults(func=_reflex_handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns shell exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
