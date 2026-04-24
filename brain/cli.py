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
from brain.engines._interests import Interest, InterestSet
from brain.engines.dream import DreamEngine
from brain.engines.heartbeat import HeartbeatEngine
from brain.engines.reflex import ReflexEngine
from brain.engines.research import ResearchEngine
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.migrator.cli import build_parser as _build_migrate_parser
from brain.paths import get_persona_dir
from brain.search.factory import get_searcher

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
    searcher = get_searcher(getattr(args, "searcher", "ddgs"))

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
                searcher=searcher,
                interests_path=persona_dir / "interests.json",
                research_log_path=persona_dir / "research_log.json",
                default_interests_path=_default_interests_path(),
                persona_name=args.persona,
                persona_system_prompt=f"You are {args.persona}.",
            )
            result = engine.run_tick(trigger=args.trigger, dry_run=args.dry_run)
        finally:
            hebbian.close()
    finally:
        store.close()

    if result.initialized:
        if args.dry_run:
            print("Heartbeat would initialize on first real tick — work deferred.")
        else:
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
        if result.research_fired:
            print(f"  research fired: {result.research_fired}")
        elif result.research_gated_reason and result.research_gated_reason != "not_due":
            print(f"  research gated: {result.research_gated_reason}")
        if result.interests_bumped > 0:
            print(f"  interests bumped: {result.interests_bumped}")
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


def _default_interests_path() -> Path:
    return Path(__file__).parent / "engines" / "default_interests.json"


def _research_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell research` to the ResearchEngine."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir} — "
            f"run `nell migrate --install-as {args.persona}` first."
        )

    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        provider = get_provider(args.provider)
        searcher = get_searcher(args.searcher)
        engine = ResearchEngine(
            store=store,
            provider=provider,
            searcher=searcher,
            persona_name=args.persona,
            persona_system_prompt=f"You are {args.persona}.",
            interests_path=persona_dir / "interests.json",
            research_log_path=persona_dir / "research_log.json",
            default_interests_path=_default_interests_path(),
        )
        result = engine.run_tick(
            trigger=args.trigger,
            dry_run=args.dry_run,
            forced_interest_topic=args.interest,
        )
    finally:
        store.close()

    if result.dry_run:
        if result.would_fire is not None:
            print(f"Research dry-run — would fire: {result.would_fire}.")
        else:
            print(f"Research dry-run — {result.reason or 'no eligible interest'}.")
    elif result.fired is not None:
        print(f"Research fired: {result.fired.topic}")
        print(f"  Memory id: {result.fired.output_memory_id}")
        print(
            f"  Web: {result.fired.web_result_count} results via "
            f"{searcher.name() if result.fired.web_used else 'memory-only'}"
        )
    else:
        print(f"Research evaluated — {result.reason or 'no fire'}.")
    return 0


def _interest_list_handler(args: argparse.Namespace) -> int:
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(f"No persona directory at {persona_dir}")
    interests = InterestSet.load(
        persona_dir / "interests.json", default_path=_default_interests_path()
    )
    print(f"Interests for persona {args.persona!r} ({len(interests.interests)}):")
    for i in interests.interests:
        last = (
            i.last_researched_at.isoformat().replace("+00:00", "Z")
            if i.last_researched_at
            else "never"
        )
        print(
            f"  - {i.topic:<40} pull={i.pull_score:.1f}  scope={i.scope:<8}  last_researched={last}"
        )
        print(f"    keywords: {', '.join(i.related_keywords)}")
    return 0


def _interest_add_handler(args: argparse.Namespace) -> int:
    import uuid
    from datetime import UTC, datetime

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(f"No persona directory at {persona_dir}")
    interests_path = persona_dir / "interests.json"
    interests = InterestSet.load(interests_path, default_path=_default_interests_path())
    now = datetime.now(UTC)
    new_interest = Interest(
        id=str(uuid.uuid4()),
        topic=args.topic,
        pull_score=5.0,
        scope=args.scope,
        related_keywords=tuple(k.strip() for k in args.keywords.split(",") if k.strip()),
        notes=args.notes or "",
        first_seen=now,
        last_fed=now,
        last_researched_at=None,
        feed_count=0,
        source_types=("manual",),
    )
    interests.upsert(new_interest).save(interests_path)
    print(f"Added interest: {new_interest.topic} (pull_score=5.0, scope={new_interest.scope})")
    return 0


def _interest_bump_handler(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(f"No persona directory at {persona_dir}")
    interests_path = persona_dir / "interests.json"
    interests = InterestSet.load(interests_path, default_path=_default_interests_path())
    if interests.find_by_topic(args.topic) is None:
        print(f"Interest not found: {args.topic!r}")
        return 1
    updated = interests.bump(args.topic, amount=args.amount, now=datetime.now(UTC))
    updated.save(interests_path)
    bumped = updated.find_by_topic(args.topic)
    assert bumped is not None
    print(
        f"Bumped {args.topic!r}: pull_score={bumped.pull_score:.1f}, feed_count={bumped.feed_count}"
    )
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
    hb_sub.add_argument(
        "--searcher",
        default="ddgs",
        choices=["ddgs", "noop", "claude-tool"],
        help="Web searcher for research engine: ddgs (default), noop, claude-tool.",
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

    # nell research
    r_sub = subparsers.add_parser(
        "research",
        help="Run one research evaluation tick against a persona.",
    )
    r_sub.add_argument("--persona", default="nell")
    r_sub.add_argument(
        "--trigger",
        choices=["manual", "emotion_high", "days_since_human", "open", "close"],
        default="manual",
    )
    r_sub.add_argument("--provider", default="claude-cli")
    r_sub.add_argument("--searcher", default="ddgs", choices=["ddgs", "noop", "claude-tool"])
    r_sub.add_argument(
        "--interest", default=None, help="Force-research this topic, bypassing gates."
    )
    r_sub.add_argument("--dry-run", action="store_true")
    r_sub.set_defaults(func=_research_handler)

    # nell interest <list|add|bump>
    i_sub = subparsers.add_parser("interest", help="Manage persona interests.")
    i_actions = i_sub.add_subparsers(dest="action", required=True)

    i_list = i_actions.add_parser("list", help="List current interests.")
    i_list.add_argument("--persona", default="nell")
    i_list.set_defaults(func=_interest_list_handler)

    i_add = i_actions.add_parser("add", help="Add a new interest.")
    i_add.add_argument("topic")
    i_add.add_argument("--keywords", default="")
    i_add.add_argument("--scope", choices=["internal", "external", "either"], default="either")
    i_add.add_argument("--notes", default=None)
    i_add.add_argument("--persona", default="nell")
    i_add.set_defaults(func=_interest_add_handler)

    i_bump = i_actions.add_parser("bump", help="Nudge an interest's pull_score.")
    i_bump.add_argument("topic")
    i_bump.add_argument("--amount", type=float, default=1.0)
    i_bump.add_argument("--persona", default="nell")
    i_bump.set_defaults(func=_interest_bump_handler)

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
