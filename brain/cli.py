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
from brain.emotion.persona_loader import load_persona_vocabulary
from brain.engines._interests import InterestSet
from brain.engines.dream import DreamEngine
from brain.engines.heartbeat import HeartbeatEngine
from brain.engines.reflex import ReflexEngine
from brain.engines.research import ResearchEngine
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.migrator.cli import build_parser as _build_migrate_parser
from brain.paths import get_persona_dir
from brain.persona_config import PersonaConfig
from brain.search.factory import get_searcher


def _resolve_routing(persona_dir: Path, args: argparse.Namespace) -> tuple[str, str]:
    """Resolve provider + searcher: CLI flag overrides persona file overrides default.

    The brain owns provider/searcher; CLI flags are developer overrides only,
    never written back. Per principle audit 2026-04-25 (PR-B).
    """
    config = PersonaConfig.load(persona_dir / "persona_config.json")
    provider = getattr(args, "provider", None) or config.provider
    searcher = getattr(args, "searcher", None) or config.searcher
    return provider, searcher

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
    """Dispatch `nell dream` to the DreamEngine.

    Developer-only entry point — production dreams fire from the heartbeat.
    Mechanism knobs (seed, depth, decay, limit, lookback) are constructor-level
    calibration, not CLI flags. Per principle audit 2026-04-25.
    """
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir}. "
            "If you're porting existing OG NellBrain data, run `nell migrate "
            f"--input /path/to/og/data --install-as {args.persona}`. "
            f"Otherwise create {persona_dir} manually to start a fresh persona."
        )
    # Nested try/finally so a HebbianMatrix open failure still closes the
    # already-open MemoryStore connection. Inline contextmanager would be
    # prettier but stores don't implement __enter__/__exit__ yet.
    provider_name, _ = _resolve_routing(persona_dir, args)
    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=store)
        hebbian = HebbianMatrix(db_path=persona_dir / "hebbian.db")
        try:
            provider = get_provider(provider_name)
            engine = DreamEngine(
                store=store,
                hebbian=hebbian,
                embeddings=None,
                provider=provider,
                log_path=persona_dir / "dreams.log.jsonl",
                persona_name=args.persona,
                persona_system_prompt=(
                    f"You are {args.persona}. You just woke from a dream about "
                    "interconnected memories. Reflect in first person, 2-3 sentences, "
                    "starting with 'DREAM: '. Be honest and specific, not abstract."
                ),
            )
            result = engine.run_cycle(dry_run=args.dry_run)
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
            f"No persona directory at {persona_dir}. "
            "If you're porting existing OG NellBrain data, run `nell migrate "
            f"--input /path/to/og/data --install-as {args.persona}`. "
            f"Otherwise create {persona_dir} manually to start a fresh persona."
        )
    default_arcs_path = Path(__file__).parent / "engines" / "default_reflex_arcs.json"
    provider_name, searcher_name = _resolve_routing(persona_dir, args)
    searcher = get_searcher(searcher_name)

    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=store)
        hebbian = HebbianMatrix(db_path=persona_dir / "hebbian.db")
        try:
            provider = get_provider(provider_name)
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
        verbose = getattr(args, "verbose", False)
        print(f"Heartbeat tick complete ({args.trigger}).")
        print(f"  elapsed: {result.elapsed_seconds / 3600:.2f}h")
        print(f"  decayed: {result.memories_decayed} memories, pruned {result.edges_pruned} edges")

        # Dream: show fires + interesting gates. Suppress "not_due" by default.
        if result.dream_id:
            print(f"  dream fired: {result.dream_id}")
        elif verbose or (result.dream_gated_reason and result.dream_gated_reason != "not_due"):
            print(f"  dream gated: {result.dream_gated_reason or 'gated'}")

        # Reflex: show fires. Suppress "evaluated, nothing fired" unless --verbose.
        if result.reflex_fired:
            print(f"  reflex fired: {', '.join(result.reflex_fired)}")
        elif verbose and result.reflex_skipped_count > 0:
            print(f"  reflex evaluated ({result.reflex_skipped_count} arc(s) skipped)")

        # Research: show fires + interesting gates (no_eligible_interest,
        # no_interests_defined, research_raised). Suppress not_due + reflex_won_tie
        # by default.
        if result.research_fired:
            print(f"  research fired: {result.research_fired}")
        elif result.research_gated_reason and (
            verbose or result.research_gated_reason not in ("not_due", "reflex_won_tie")
        ):
            print(f"  research gated: {result.research_gated_reason}")

        # Interest bumps: show only if > 0 (already compact). Verbose adds zero.
        if result.interests_bumped > 0:
            print(f"  interests bumped: {result.interests_bumped}")
        elif verbose:
            print("  interests bumped: 0")
    return 0


def _reflex_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell reflex` to the ReflexEngine."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir}. "
            "If you're porting existing OG NellBrain data, run `nell migrate "
            f"--input /path/to/og/data --install-as {args.persona}`. "
            f"Otherwise create {persona_dir} manually to start a fresh persona."
        )

    default_arcs_path = Path(__file__).parent / "engines" / "default_reflex_arcs.json"
    provider_name, _ = _resolve_routing(persona_dir, args)

    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=store)
        provider = get_provider(provider_name)
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
            f"No persona directory at {persona_dir}. "
            "If you're porting existing OG NellBrain data, run `nell migrate "
            f"--input /path/to/og/data --install-as {args.persona}`. "
            f"Otherwise create {persona_dir} manually to start a fresh persona."
        )

    provider_name, searcher_name = _resolve_routing(persona_dir, args)
    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=store)
        provider = get_provider(provider_name)
        searcher = get_searcher(searcher_name)
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
    load_persona_vocabulary(persona_dir / "emotion_vocabulary.json")
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


def _growth_log_handler(args: argparse.Namespace) -> int:
    """`nell growth log` — read-only inspection of the brain's growth biography.

    Per Phase 2a §8: read-only. No add/approve/reject/force. The user
    reads what the brain decided; if they want to override, they edit
    `emotion_vocabulary.json` directly.
    """
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir}. "
            f"Persona {args.persona!r} does not exist."
        )

    from brain.growth.log import read_growth_log

    log_path = persona_dir / "emotion_growth.log.jsonl"
    events = read_growth_log(log_path, limit=args.limit)
    if args.type:
        events = [e for e in events if e.type == args.type]

    print(f"Growth log for persona {args.persona!r} ({len(events)} events shown):")
    if not events:
        print("  (empty)")
        return 0

    for e in events:
        ts = e.timestamp.isoformat().replace("+00:00", "Z")
        print(f"\n  {ts}  {e.type:<20} {e.name}")
        print(f'    "{e.description}"')
        decay = (
            "identity-level (no decay)"
            if e.decay_half_life_days is None
            else f"{e.decay_half_life_days:.1f} days"
        )
        print(f"    decay: {decay}  score: {e.score:.2f}")
        print(f"    reason: {e.reason}")
        if e.relational_context:
            print(f"    relational: {e.relational_context}")
        if e.evidence_memory_ids:
            preview = ", ".join(e.evidence_memory_ids[:3])
            extra = (
                f", ... ({len(e.evidence_memory_ids)} total)"
                if len(e.evidence_memory_ids) > 3
                else ""
            )
            print(f"    evidence: {preview}{extra}")
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
        help=(
            "(developer) Run one dream cycle against a persona's memory store. "
            "Production dreams fire from the heartbeat — this is for debugging."
        ),
    )
    dream_sub.add_argument(
        "--persona",
        required=True,
        help=(
            "Persona name (required). "
            "To port existing OG NellBrain data: `nell migrate --input /path/to/og/data --install-as <name>`. "
            "To start fresh: create personas/<name>/ manually."
        ),
    )
    dream_sub.add_argument(
        "--provider",
        default=None,
        help=(
            "(developer override) LLM provider — claude-cli, fake, ollama. "
            "Defaults to the value in {persona}/persona_config.json."
        ),
    )
    dream_sub.add_argument("--dry-run", action="store_true", help="Skip LLM call and store writes.")
    dream_sub.set_defaults(func=_dream_handler)

    hb_sub = subparsers.add_parser(
        "heartbeat",
        help="Run one heartbeat orchestrator tick against a persona.",
    )
    hb_sub.add_argument(
        "--persona",
        required=True,
        help=(
            "Persona name (required). "
            "To port existing OG NellBrain data: `nell migrate --input /path/to/og/data --install-as <name>`. "
            "To start fresh: create personas/<name>/ manually."
        ),
    )
    hb_sub.add_argument(
        "--trigger",
        choices=["open", "close", "manual"],
        default="manual",
    )
    hb_sub.add_argument(
        "--provider",
        default=None,
        help=(
            "(developer override) LLM provider — claude-cli, fake, ollama. "
            "Defaults to the value in {persona}/persona_config.json."
        ),
    )
    hb_sub.add_argument(
        "--searcher",
        default=None,
        choices=["ddgs", "noop", "claude-tool"],
        help=(
            "(developer override) Web searcher for research engine — ddgs, noop, "
            "claude-tool. Defaults to the value in {persona}/persona_config.json."
        ),
    )
    hb_sub.add_argument("--dry-run", action="store_true")
    hb_sub.add_argument(
        "--verbose",
        action="store_true",
        help="Show all engine outcomes including gated reasons + zero-count engines. "
        "Default output is compact — events shown, non-events hidden.",
    )
    hb_sub.set_defaults(func=_heartbeat_handler)

    rf_sub = subparsers.add_parser(
        "reflex",
        help="Run one reflex evaluation tick against a persona.",
    )
    rf_sub.add_argument(
        "--persona",
        required=True,
        help=(
            "Persona name (required). "
            "To port existing OG NellBrain data: `nell migrate --input /path/to/og/data --install-as <name>`. "
            "To start fresh: create personas/<name>/ manually."
        ),
    )
    rf_sub.add_argument(
        "--trigger",
        choices=["open", "close", "manual"],
        default="manual",
    )
    rf_sub.add_argument(
        "--provider",
        default=None,
        help=(
            "(developer override) LLM provider — claude-cli, fake, ollama. "
            "Defaults to the value in {persona}/persona_config.json."
        ),
    )
    rf_sub.add_argument("--dry-run", action="store_true")
    rf_sub.set_defaults(func=_reflex_handler)

    # nell research
    r_sub = subparsers.add_parser(
        "research",
        help="Run one research evaluation tick against a persona.",
    )
    r_sub.add_argument(
        "--persona",
        required=True,
        help=(
            "Persona name (required). "
            "To port existing OG NellBrain data: `nell migrate --input /path/to/og/data --install-as <name>`. "
            "To start fresh: create personas/<name>/ manually."
        ),
    )
    r_sub.add_argument(
        "--trigger",
        choices=["manual", "emotion_high", "days_since_human", "open", "close"],
        default="manual",
    )
    r_sub.add_argument(
        "--provider",
        default=None,
        help=(
            "(developer override) LLM provider — claude-cli, fake, ollama. "
            "Defaults to the value in {persona}/persona_config.json."
        ),
    )
    r_sub.add_argument(
        "--searcher",
        default=None,
        choices=["ddgs", "noop", "claude-tool"],
        help=(
            "(developer override) Web searcher — ddgs, noop, claude-tool. "
            "Defaults to the value in {persona}/persona_config.json."
        ),
    )
    r_sub.add_argument("--dry-run", action="store_true")
    r_sub.set_defaults(func=_research_handler)

    # nell interest list — read-only inspection. The brain develops its own
    # interests; the user does not add or bump them. Per principle audit
    # 2026-04-25.
    i_sub = subparsers.add_parser(
        "interest",
        help="Inspect persona interests (read-only).",
    )
    i_actions = i_sub.add_subparsers(dest="action", required=True)

    i_list = i_actions.add_parser("list", help="List current interests.")
    i_list.add_argument(
        "--persona",
        required=True,
        help=(
            "Persona name (required). "
            "To port existing OG NellBrain data: `nell migrate --input /path/to/og/data --install-as <name>`. "
            "To start fresh: create personas/<name>/ manually."
        ),
    )
    i_list.set_defaults(func=_interest_list_handler)

    # nell growth log — read-only inspection of brain growth biography.
    # Per Phase 2a §8: only `log` action ships; no add/approve/reject/force.
    g_sub = subparsers.add_parser(
        "growth",
        help="Inspect the brain's autonomous growth biography (read-only).",
    )
    g_actions = g_sub.add_subparsers(dest="action", required=True)

    g_log = g_actions.add_parser("log", help="Print the growth log.")
    g_log.add_argument(
        "--persona",
        required=True,
        help="Persona name (required).",
    )
    g_log.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Show only the most-recent N events.",
    )
    g_log.add_argument(
        "--type",
        default=None,
        help="Filter by event type (e.g. 'emotion_added').",
    )
    g_log.set_defaults(func=_growth_log_handler)

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
