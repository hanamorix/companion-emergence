"""Entry point CLI for companion-emergence.

Invoked as `nell <subcommand> [options]`. Week 1 ships `--version`, help,
and a set of stub subcommands that print "not implemented yet" so the CLI
surface is stable while subsequent weeks fill in functionality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from brain import __version__
from brain.bridge import state_file
from brain.bridge.provider import get_provider
from brain.emotion.persona_loader import load_persona_vocabulary
from brain.engines._interests import InterestSet
from brain.engines.dream import DreamEngine
from brain.engines.heartbeat import HeartbeatEngine
from brain.engines.reflex import ReflexEngine
from brain.engines.research import ResearchEngine
from brain.health.alarm import compute_pending_alarms
from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.health.walker import walk_persona
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.migrator.cli import build_parser as _build_migrate_parser
from brain.paths import get_home, get_persona_dir
from brain.persona_config import PersonaConfig
from brain.search.factory import get_searcher
from brain.setup import (
    VOICE_TEMPLATES,
    install_voice_template,
    validate_persona_name,
    write_persona_config,
)
from brain.utils.time import iso_utc


def _resolve_routing(persona_dir: Path, args: argparse.Namespace) -> tuple[str, str]:
    """Resolve provider + searcher: CLI flag overrides persona file overrides default.

    The brain owns provider/searcher; CLI flags are developer overrides only,
    never written back. Per principle audit 2026-04-25 (PR-B).
    """
    config = PersonaConfig.load(persona_dir / "persona_config.json")
    provider = getattr(args, "provider", None) or config.provider
    searcher = getattr(args, "searcher", None) or config.searcher
    return provider, searcher


# All Week-1 framework stubs have been resolved:
# - `nell supervisor` shipped 2026-05-04
# - `nell rest` removed 2026-05-04 (rest is body-state physiology, not a command)
# - `nell works` shipped 2026-05-04


def _status_handler(args: argparse.Namespace) -> int:
    """Print local persona/bridge status without making provider calls or writes."""
    persona_dir = get_persona_dir(args.persona)
    persona_exists = persona_dir.exists()

    print(f"companion-emergence {__version__}")
    print(f"home: {get_home()}")
    print(f"persona: {args.persona}")
    print(f"persona_dir: {persona_dir}")
    print(f"persona_exists: {'yes' if persona_exists else 'no'}")

    state = state_file.read(persona_dir)
    if not persona_exists:
        _print_bridge_status(state)
        return 1

    config = PersonaConfig.load(persona_dir / "persona_config.json")
    print(f"provider: {config.provider}")
    print(f"searcher: {config.searcher}")
    print(f"mcp_audit_log_level: {config.mcp_audit_log_level}")

    memory_path = persona_dir / "memories.db"
    if memory_path.exists():
        store = MemoryStore(memory_path, integrity_check=False)
        try:
            print(f"memories_active: {store.count(active_only=True)}")
        finally:
            store.close()
    else:
        print("memories_active: missing")

    _print_bridge_status(state)
    return 0


def _print_bridge_status(state: state_file.BridgeState | None) -> None:
    """Print bridge status while never exposing bearer tokens."""
    if state is None or state.pid is None:
        print("bridge: not running")
        return

    if state_file.pid_is_alive(state.pid):
        print("bridge: running")
        print(f"pid: {state.pid}")
        print(f"port: {state.port or 'unknown'}")
        print(f"started_at: {state.started_at}")
        print(f"client_origin: {state.client_origin}")
        return

    if not state.shutdown_clean:
        print("bridge: crashed-dirty")
        print(f"pid: {state.pid}")
        print("recovery: needed on next bridge start")
        return

    print("bridge: stopped")
    print(f"stopped_at: {state.stopped_at or 'unknown'}")


def _service_unsupported_message(action: str) -> int:
    """Print the dispatcher's UnsupportedPlatformError message and return 1.

    Surfaces the friendly "this OS isn't wired yet" hint instead of
    letting users hit the launchctl-not-found / SCM-missing failure
    deeper in the call stack. Returns 1 so non-mac shells signal
    failure cleanly.
    """
    from brain.service import UnsupportedPlatformError, current_backend

    try:
        current_backend()
    except UnsupportedPlatformError as exc:
        print(f"service {action}: {exc}", file=sys.stderr)
        return 1
    return 0


def _service_print_plist_handler(args: argparse.Namespace) -> int:
    """Print the macOS LaunchAgent plist for a persona without installing it."""
    if (rc := _service_unsupported_message("print-plist")) != 0:
        return rc
    from brain.service.launchd import (
        LaunchdConfigError,
        build_launchd_plist_xml,
        resolve_nell_path,
    )

    try:
        nell_path = resolve_nell_path(args.nell_path)
        xml = build_launchd_plist_xml(
            persona=args.persona,
            nell_path=nell_path,
            env_path=args.env_path,
            nellbrain_home=args.nellbrain_home,
        )
    except LaunchdConfigError as exc:
        print(f"service print-plist: {exc}", file=sys.stderr)
        return 1
    print(xml, end="")
    return 0


def _service_doctor_handler(args: argparse.Namespace) -> int:
    """Run non-mutating preflight checks for launchd service install."""
    if (rc := _service_unsupported_message("doctor")) != 0:
        return rc
    from brain.service.launchd import doctor_checks

    checks = doctor_checks(
        persona=args.persona,
        nell_path=args.nell_path,
        env_path=args.env_path,
    )
    for check in checks:
        status = "ok" if check.ok else "FAIL"
        print(f"{status:4} {check.name}: {check.detail}")
    return 0 if all(check.ok for check in checks) else 1


def _service_install_handler(args: argparse.Namespace) -> int:
    """Install/bootstrap the macOS LaunchAgent for one persona."""
    if (rc := _service_unsupported_message("install")) != 0:
        return rc
    from brain.service.launchd import (
        LaunchdCommandError,
        LaunchdConfigError,
        build_launchd_plist_xml,
        install_service,
        resolve_nell_path,
    )

    try:
        nell_path = resolve_nell_path(args.nell_path)
        if args.dry_run:
            print(
                build_launchd_plist_xml(
                    persona=args.persona,
                    nell_path=nell_path,
                    env_path=args.env_path,
                    nellbrain_home=args.nellbrain_home,
                ),
                end="",
            )
            return 0
        plist_path = install_service(
            persona=args.persona,
            nell_path=nell_path,
            env_path=args.env_path,
            nellbrain_home=args.nellbrain_home,
        )
    except (LaunchdConfigError, LaunchdCommandError) as exc:
        print(f"service install: {exc}", file=sys.stderr)
        return 1
    print(f"service installed: {plist_path}")
    return 0


def _service_uninstall_handler(args: argparse.Namespace) -> int:
    """Boot out the persona LaunchAgent and remove the plist."""
    if (rc := _service_unsupported_message("uninstall")) != 0:
        return rc
    from brain.service.launchd import LaunchdConfigError, uninstall_service

    try:
        plist_path = uninstall_service(persona=args.persona, keep_plist=args.keep_plist)
    except LaunchdConfigError as exc:
        print(f"service uninstall: {exc}", file=sys.stderr)
        return 1
    action = "kept" if args.keep_plist else "removed"
    print(f"service uninstalled: plist {action} at {plist_path}")
    return 0


def _service_status_handler(args: argparse.Namespace) -> int:
    """Print launchd service status without changing service state."""
    if (rc := _service_unsupported_message("status")) != 0:
        return rc
    from brain.service.launchd import LaunchdConfigError, service_status

    try:
        status = service_status(persona=args.persona)
    except LaunchdConfigError as exc:
        print(f"service status: {exc}", file=sys.stderr)
        return 1
    print(f"label: {status.label}")
    print(f"plist: {status.plist_path}")
    print(f"installed: {'yes' if status.installed else 'no'}")
    print(f"loaded: {'yes' if status.loaded else 'no'}")
    if status.detail:
        print(f"detail: {status.detail}")
    return 0


def _daemon_state_refresh_handler(args: argparse.Namespace) -> int:
    """Rewrite daemon_state.json's last_<type> entries from active memories.

    Useful when a constant change (e.g. summary cap bump) means existing
    daemon_state.json entries were truncated under the old rules but
    the underlying memory in memories.db is intact. Walks each daemon
    type, finds the most recent active memory of the matching type,
    and overwrites the corresponding ``last_<type>.summary`` field.

    Idempotent: re-running with the same memories produces the same
    output. Engine fires write fresh entries on next tick anyway, so
    this is a recovery tool, not part of the normal flow.
    """
    from brain.engines.daemon_state import load_daemon_state, update_daemon_state

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        print(f"No persona directory at {persona_dir}.", file=sys.stderr)
        return 1

    store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
    try:
        state, _ = load_daemon_state(persona_dir)
        # Memory-type keyed by daemon_type. The engine's writer functions
        # use the same names; keep this in sync if a new daemon_type lands.
        type_map = {
            "dream": "dream",
            "reflex": "reflex_journal",
            "research": "research",
            # Heartbeat memories exist but the daemon_state writer for
            # heartbeat synthesizes its summary from dominant_emotion +
            # intensity rather than copying the memory content, so
            # there's nothing to refresh from a memory row.
            "heartbeat": None,
        }
        refreshed = 0
        for daemon_type, memory_type in type_map.items():
            existing = getattr(state, f"last_{daemon_type}", None)
            if existing is None or memory_type is None:
                continue
            mems = store.list_by_type(memory_type, active_only=True, limit=1)
            if not mems:
                continue
            mem = mems[0]
            old_len = len(existing.summary)
            new_len = len(mem.content)
            if old_len == new_len and existing.summary == mem.content:
                continue  # Already in sync — nothing to refresh.
            update_daemon_state(
                persona_dir,
                daemon_type=daemon_type,  # type: ignore[arg-type]
                dominant_emotion=existing.dominant_emotion,
                intensity=existing.intensity,
                theme=existing.theme,
                summary=mem.content,
                trigger=existing.trigger,
            )
            print(f"refreshed last_{daemon_type}: {old_len} → {new_len} chars")
            refreshed += 1
        if refreshed == 0:
            print("daemon_state already in sync with memories")
        return 0
    finally:
        store.close()


def _open_memory_store_for_cli(persona: str) -> tuple[MemoryStore | None, int]:
    """Open a persona memory store for read-only CLI inspection."""
    persona_dir = get_persona_dir(persona)
    if not persona_dir.exists():
        print(f"No persona directory at {persona_dir}.", file=sys.stderr)
        return None, 1

    memory_path = persona_dir / "memories.db"
    if not memory_path.exists():
        print(f"No memory store at {memory_path}.", file=sys.stderr)
        return None, 1

    return MemoryStore(memory_path, integrity_check=False), 0


def _memory_preview(content: str, *, limit: int = 72) -> str:
    """Return a compact single-line memory preview."""
    single_line = " ".join(content.split())
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 1].rstrip() + "…"


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _print_memory_rows(title: str, memories: list[Memory]) -> None:
    """Print compact memory rows for list/search output."""
    print(title)
    if not memories:
        print("(none)")
        return

    for memory in memories:
        print(
            f"{memory.id[:8]}  {memory.created_at.date().isoformat()}  "
            f"{memory.memory_type}/{memory.domain}  "
            f"importance={memory.importance:.2f}  {_memory_preview(memory.content)}"
        )


def _format_mapping(mapping: dict[str, object]) -> str:
    """Format small dicts deterministically for CLI output."""
    if not mapping:
        return "(none)"
    return ", ".join(f"{key}={float(value):.2f}" for key, value in sorted(mapping.items()))


def _memory_list_handler(args: argparse.Namespace) -> int:
    """List recent active memories for a persona."""
    store, rc = _open_memory_store_for_cli(args.persona)
    if store is None:
        return rc
    try:
        memories = store.list_active(limit=args.limit)
    finally:
        store.close()

    _print_memory_rows(f"active memories for {args.persona}", memories)
    return 0


def _memory_search_handler(args: argparse.Namespace) -> int:
    """Search active memories by non-empty text query."""
    query = args.query.strip()
    if not query:
        print("query must not be empty", file=sys.stderr)
        return 2

    store, rc = _open_memory_store_for_cli(args.persona)
    if store is None:
        return rc
    try:
        memories = store.search_text(query, limit=args.limit)
    finally:
        store.close()

    _print_memory_rows(f"memory search for {args.persona}: {query}", memories)
    return 0


def _memory_show_handler(args: argparse.Namespace) -> int:
    """Show one full memory record by id."""
    store, rc = _open_memory_store_for_cli(args.persona)
    if store is None:
        return rc
    try:
        memory = store.get(args.memory_id)
    finally:
        store.close()

    if memory is None:
        print(f"unknown memory id: {args.memory_id}", file=sys.stderr)
        return 1

    print(f"id: {memory.id}")
    print(f"type: {memory.memory_type}")
    print(f"domain: {memory.domain}")
    print(f"created_at: {memory.created_at.isoformat()}")
    print(f"last_accessed_at: {memory.last_accessed_at.isoformat() if memory.last_accessed_at else '(never)'}")
    print(f"importance: {memory.importance:.2f}")
    print(f"score: {memory.score:.2f}")
    print(f"active: {'yes' if memory.active else 'no'}")
    print(f"protected: {'yes' if memory.protected else 'no'}")
    print(f"tags: {', '.join(memory.tags) if memory.tags else '(none)'}")
    print(f"emotions: {_format_mapping(memory.emotions)}")
    print(f"metadata: {json.dumps(memory.metadata, sort_keys=True)}")
    print("content:")
    print(memory.content)
    return 0


def _works_list_handler(args: argparse.Namespace) -> int:
    """List recent brain-authored creative artifacts."""
    from brain.paths import get_persona_dir
    from brain.tools.impls.list_works import list_works

    persona_dir = get_persona_dir(args.persona)
    works_list = list_works(type=args.type, limit=args.limit, persona_dir=persona_dir)
    if not works_list:
        print("(no works)")
        return 0
    for w in works_list:
        summary = f" — {w['summary']}" if w["summary"] else ""
        print(f"{w['id']} | {w['type']:<8} | {w['created_at']} | {w['title']}{summary}")
    return 0


def _works_search_handler(args: argparse.Namespace) -> int:
    """Search brain-authored creative artifacts by title/summary/content."""
    from brain.paths import get_persona_dir
    from brain.tools.impls.search_works import search_works

    persona_dir = get_persona_dir(args.persona)
    matches = search_works(
        query=args.query, type=args.type, limit=args.limit, persona_dir=persona_dir
    )
    if not matches:
        print("(no matches)")
        return 0
    for w in matches:
        summary = f" — {w['summary']}" if w["summary"] else ""
        print(f"{w['id']} | {w['type']:<8} | {w['created_at']} | {w['title']}{summary}")
    return 0


def _works_read_handler(args: argparse.Namespace) -> int:
    """Print one work's frontmatter header followed by its full content."""
    from brain.paths import get_persona_dir
    from brain.tools.impls.read_work import read_work

    persona_dir = get_persona_dir(args.persona)
    result = read_work(id=args.id, persona_dir=persona_dir)
    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1
    print("---")
    print(f"id: {result['id']}")
    print(f"title: {result['title']}")
    print(f"type: {result['type']}")
    print(f"created_at: {result['created_at']}")
    if result.get("summary"):
        print(f"summary: {result['summary']}")
    print(f"word_count: {result['word_count']}")
    print("---")
    print()
    print(result["content"])
    return 0


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

        # Health banner — printed BEFORE engine status lines so it's visible
        # at the top. Only shown when there are unacknowledged alarms.
        if result.pending_alarms_count > 0:
            print(
                f"⚠️  Brain alarm — needs your attention. "
                f"Run `nell health show --persona {args.persona}` for details."
            )

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

        # Self-treatment line — shown when anomalies were healed but no pending alarms.
        # Appears after engine status, before the trailing newline.
        if result.anomalies and result.pending_alarms_count == 0:
            distinct_files = list(dict.fromkeys(a.file for a in result.anomalies))
            count = len(distinct_files)
            if count <= 2:
                files_desc = ", ".join(distinct_files)
            else:
                files_desc = ", ".join(distinct_files[:2]) + ", ..."
            print(
                f"  \U0001fa79 brain self-treated {count} file"
                f"{'s' if count != 1 else ''} ({files_desc})"
                f" — see `nell health show` for details"
            )
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
            f"No persona directory at {persona_dir}. Persona {args.persona!r} does not exist."
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


def _health_show_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell health show` — pending alarms + recent self-treatments (read-only)."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir}. Persona {args.persona!r} does not exist."
        )

    from datetime import UTC, datetime, timedelta

    audit_path = persona_dir / "heartbeats.log.jsonl"
    cutoff = datetime.now(UTC) - timedelta(days=7)

    # Collect recent anomaly records from the audit log.
    recent_treatments: list[dict] = []
    for entry in read_jsonl_skipping_corrupt(audit_path):
        try:
            from brain.utils.time import parse_iso_utc

            ts = parse_iso_utc(entry["timestamp"])
        except (KeyError, ValueError, TypeError):
            continue
        if ts < cutoff:
            continue
        for a in entry.get("anomalies") or []:
            if isinstance(a, dict):
                recent_treatments.append(a)

    alarms = compute_pending_alarms(persona_dir)

    print(f"Health for persona {args.persona!r}:\n")

    print(f"  Pending alarms: {len(alarms)}")
    for alarm in alarms:
        date_str = alarm.first_seen_at.strftime("%Y-%m-%d")
        print(
            f"    {alarm.file}: {alarm.kind} {date_str} "
            f"({alarm.occurrences_in_window} occurrences in window)"
        )

    print(f"  Recent self-treatments (last 7 days): {len(recent_treatments)}")
    for t in recent_treatments:
        try:
            from brain.utils.time import parse_iso_utc

            ts = parse_iso_utc(t["timestamp"])
            ts_str = iso_utc(ts)
        except (KeyError, ValueError, TypeError):
            ts_str = "unknown"
        f = t.get("file", "?")
        action = t.get("action", "?")
        cause = t.get("likely_cause", "unknown")
        qpath = t.get("quarantine_path")
        print(f"    {ts_str}  {f}  {action}  (cause: {cause})")
        if qpath:
            print(f"             forensic: {qpath}")

    return 0


def _health_check_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell health check` — run walk_persona, print per-file status."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir}. Persona {args.persona!r} does not exist."
        )

    anomalies = walk_persona(persona_dir)

    # Classify anomalies: unhealable vs self-treated.
    unhealable = [a for a in anomalies if a.action == "alarmed_unrecoverable"]
    healed = [a for a in anomalies if a.action != "alarmed_unrecoverable"]

    # Gather all checked file names (from anomalies only — healthy files print OK below).
    anomaly_files = {a.file for a in anomalies}

    # Print per-file status for anomalies.
    for a in healed:
        print(f"⚠️  {a.file}: {a.action} ({a.kind})")
    for a in unhealable:
        print(f"❌  {a.file}: {a.kind} — unrecoverable")

    # Healthy files: any JSON file in the walker's default set not in anomalies.
    walker_files = [
        "user_preferences.json",
        "persona_config.json",
        "heartbeat_config.json",
        "heartbeat_state.json",
        "interests.json",
        "reflex_arcs.json",
        "emotion_vocabulary.json",
        "memories.db",
        "hebbian.db",
    ]
    for fname in walker_files:
        if fname not in anomaly_files and (persona_dir / fname).exists():
            print(f"✅  {fname}: OK")

    n_healed = len(healed)
    n_unhealable = len(unhealable)
    is_alarming = n_unhealable > 0
    state = "alarming" if is_alarming else "healthy"
    print(f"\n{n_healed} file(s) healed, {n_unhealable} unhealable. Brain is {state}.")

    return 2 if is_alarming else 0


def _health_acknowledge_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell health acknowledge` — append user_acknowledged entry to audit log."""
    from datetime import UTC, datetime

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir}. Persona {args.persona!r} does not exist."
        )

    # Validate: at most one of --file / --all may be set. Default to --all.
    has_file = getattr(args, "ack_file", None) is not None
    has_all = getattr(args, "ack_all", False)

    if has_file and has_all:
        print("Error: --file and --all are mutually exclusive.", file=sys.stderr)
        return 1

    if has_file:
        files_to_ack = [args.ack_file]
    else:
        # Default: --all — acknowledge every pending alarm.
        alarms = compute_pending_alarms(persona_dir)
        files_to_ack = [alarm.file for alarm in alarms]

    import json

    entry = {
        "timestamp": iso_utc(datetime.now(UTC)),
        "user_acknowledged": files_to_ack,
    }
    audit_path = persona_dir / "heartbeats.log.jsonl"
    with audit_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    n = len(files_to_ack)
    if n == 0:
        print("No pending alarms to acknowledge.")
    else:
        desc = ", ".join(files_to_ack[:3])
        if n > 3:
            desc += f", ... ({n} total)"
        print(f"Acknowledged {n} alarm(s): {desc}")
    return 0


def _soul_list_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell soul list` — list active crystallizations."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir}. Persona {args.persona!r} does not exist."
        )

    from brain.soul.store import SoulStore

    soul_store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        active = soul_store.list_active()
    finally:
        soul_store.close()

    limit = getattr(args, "limit", None)
    if limit is not None:
        active = active[-limit:]

    print(f"Soul crystallizations for persona {args.persona!r} ({len(active)} active):")
    if not active:
        print("  (none yet)")
        return 0

    for c in active:
        ts = c.crystallized_at.isoformat().replace("+00:00", "Z")
        moment_preview = c.moment[:80].replace("\n", " ")
        print(f"\n  {c.id[:8]}…  [{c.love_type}]  resonance={c.resonance}  {ts}")
        print(f"    {moment_preview}")
    return 0


def _soul_revoke_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell soul revoke` — revoke a crystallization."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir}. Persona {args.persona!r} does not exist."
        )

    from brain.soul.revoke import revoke_crystallization
    from brain.soul.store import SoulStore

    soul_store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        result = revoke_crystallization(soul_store, args.id, args.reason)
    finally:
        soul_store.close()

    if result is None:
        print(f"Crystallization {args.id!r} not found.", file=sys.stderr)
        return 1

    print(f"Revoked: {result.id}")
    print(f"  moment: {result.moment[:80]}")
    print(f"  reason: {result.revoked_reason}")
    return 0


def _soul_candidates_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell soul candidates` — list pending soul_candidates.jsonl entries."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir}. Persona {args.persona!r} does not exist."
        )

    from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

    candidates_path = persona_dir / "soul_candidates.jsonl"
    records = read_jsonl_skipping_corrupt(candidates_path)
    pending = [r for r in records if r.get("status", "auto_pending") == "auto_pending"]

    limit = getattr(args, "limit", None)
    if limit is not None:
        pending = pending[-limit:]

    print(f"Pending soul candidates for persona {args.persona!r} ({len(pending)}):")
    if not pending:
        print("  (none)")
        return 0

    for c in pending:
        text_preview = str(c.get("text", ""))[:80].replace("\n", " ")
        queued = str(c.get("queued_at", "?"))[:19].replace("T", " ")
        label = c.get("label", "?")
        print(f"\n  {c.get('id', '?')[:8]}…  [{label}]  queued={queued}")
        print(f"    {text_preview}")
    return 0


def _soul_audit_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell soul audit` — tail soul_audit.jsonl."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir}. Persona {args.persona!r} does not exist."
        )

    from brain.soul.audit import read_audit_log

    limit = getattr(args, "limit", 20)
    entries = read_audit_log(persona_dir, limit=limit)

    print(f"Soul audit log for persona {args.persona!r} (last {len(entries)} entries):")
    if not entries:
        print("  (empty)")
        return 0

    for e in entries:
        ts = str(e.get("ts", "?"))[:19].replace("T", " ")
        decision = e.get("decision", "?")
        cid = str(e.get("candidate_id", "?"))[:8]
        confidence = e.get("confidence", "?")
        love_type = e.get("love_type", "?")
        dry = " (dry-run)" if e.get("dry_run") else ""
        print(
            f"\n  {ts}  {decision:<8}  cid={cid}…  confidence={confidence}  type={love_type}{dry}"
        )
        reasoning = str(e.get("reasoning", ""))[:120]
        if reasoning:
            print(f"    {reasoning}")
        if e.get("parse_error"):
            print(f"    parse_error: {e['parse_error']}")
        if e.get("forced_defer_reason"):
            print(f"    forced_defer: {e['forced_defer_reason']}")
    return 0


def _soul_review_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell soul review` — run autonomous soul review pass."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir}. Persona {args.persona!r} does not exist."
        )

    provider_name, _ = _resolve_routing(persona_dir, args)
    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=store)
        provider = get_provider(provider_name)

        from brain.soul.review import review_pending_candidates
        from brain.soul.store import SoulStore

        soul_store = SoulStore(str(persona_dir / "crystallizations.db"))
        try:
            report = review_pending_candidates(
                persona_dir,
                store=store,
                soul_store=soul_store,
                provider=provider,
                max_decisions=getattr(args, "max", 5),
                confidence_threshold=getattr(args, "confidence_threshold", 7),
                dry_run=args.dry_run,
            )
        finally:
            soul_store.close()
    finally:
        store.close()

    if args.dry_run:
        print("Soul review dry-run — no writes.")
    print(
        f"Soul review complete: {report.pending_at_start} pending, "
        f"{report.examined} examined, {report.accepted} accepted, "
        f"{report.rejected} rejected, {report.deferred} deferred, "
        f"{report.parse_failures} parse failures."
    )
    if report.crystallization_ids:
        print(f"  New crystallizations: {', '.join(report.crystallization_ids)}")
    return 0


def _chat_direct_mode(args: argparse.Namespace) -> int:
    """Dispatch `nell chat` to the chat engine (in-process, no bridge).

    One-shot mode (message provided): `nell chat --persona X "message"`
    Interactive REPL mode: `nell chat --persona X`

    The REPL keeps a single SessionState across turns.  On exit (EOF / 'exit' /
    'quit') it flushes the conversation through the ingest pipeline best-effort
    and prints a summary.
    """
    from brain.chat.engine import respond
    from brain.chat.session import create_session
    from brain.ingest.pipeline import close_session

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir}. "
            "If you're porting existing OG NellBrain data, run `nell migrate "
            f"--input /path/to/og/data --install-as {args.persona}`. "
            f"Otherwise create {persona_dir} manually to start a fresh persona."
        )

    provider_name, _ = _resolve_routing(persona_dir, args)

    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        from brain.emotion.persona_loader import load_persona_vocabulary

        load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=store)
        hebbian = HebbianMatrix(db_path=persona_dir / "hebbian.db")
        try:
            provider = get_provider(provider_name)

            # One-shot mode
            message = getattr(args, "message", None)
            if message:
                result = respond(
                    persona_dir,
                    message,
                    store=store,
                    hebbian=hebbian,
                    provider=provider,
                )
                print(result.content)
                # Flush conversation through ingest pipeline (best-effort).
                # Mirrors the REPL's finally block — without this, every
                # one-shot reply orphans its buffer file and no memories
                # ever commit (live-exercise 2026-04-27 surfaced the bug).
                try:
                    close_session(
                        persona_dir,
                        result.session_id,
                        store=store,
                        hebbian=hebbian,
                        provider=provider,
                    )
                except Exception as exc:  # noqa: BLE001
                    import warnings

                    warnings.warn(
                        f"Session ingest flush failed: {exc}",
                        RuntimeWarning,
                        stacklevel=1,
                    )
                return 0

            # Interactive REPL mode
            session = create_session(args.persona)
            total_tool_calls = 0
            try:
                while True:
                    try:
                        user_input = input("> ")
                    except EOFError:
                        break
                    user_input = user_input.strip()
                    if user_input.lower() in ("exit", "quit"):
                        break
                    if not user_input:
                        continue
                    result = respond(
                        persona_dir,
                        user_input,
                        store=store,
                        hebbian=hebbian,
                        provider=provider,
                        session=session,
                    )
                    print(result.content)
                    print()
                    total_tool_calls += len(result.tool_invocations)
            finally:
                # Flush conversation through ingest pipeline (best-effort)
                try:
                    close_session(
                        persona_dir,
                        session.session_id,
                        store=store,
                        hebbian=hebbian,
                        provider=provider,
                    )
                except Exception as exc:  # noqa: BLE001
                    import warnings

                    warnings.warn(
                        f"Session ingest flush failed: {exc}",
                        RuntimeWarning,
                        stacklevel=1,
                    )
                # Summary
                print(
                    f"\nSession ended. {session.turns} turn(s), "
                    f"{total_tool_calls} tool call(s) used."
                )
        finally:
            hebbian.close()
    finally:
        store.close()

    return 0


def _chat_via_bridge(args: argparse.Namespace, persona_dir: Path, *, readiness=None) -> int:
    """Chat with a persona via the bridge daemon's WebSocket streaming API.

    `readiness` (BridgeReadiness): the verified port + auth_token captured
    by `_chat_handler` from cmd_start's out= dict. Using this instead of
    re-reading state_file closes the race window where state_file could
    rotate or the bridge could die between /health verification and our
    read. Falls back to state_file for legacy callers (e.g. tests that
    instantiate this directly).
    """
    import json
    import sys as _sys
    import time as _time

    import httpx
    from websockets.exceptions import InvalidStatus
    from websockets.sync.client import connect

    from brain.bridge import state_file

    if readiness is not None:
        port = readiness.port
        auth_token = readiness.auth_token
    else:
        s = state_file.read(persona_dir)
        port = s.port
        auth_token = s.auth_token
    base = f"http://127.0.0.1:{port}"
    # H-C: read the bridge's ephemeral auth token from bridge.json and
    # send it on every HTTP/WS request. None when running against a
    # legacy/dev bridge with auth disabled.
    http_headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    # WS auth is via Sec-WebSocket-Protocol: bearer, <token> — the only
    # auth path the server accepts. Same fix as cmd_tail (audit-2 I-1).
    ws_subprotocols = ["bearer", auth_token] if auth_token else None

    sid_arg = getattr(args, "session", None)
    if sid_arg:
        sid = sid_arg
    else:
        sid = httpx.post(
            f"{base}/session/new", json={"client": "cli"}, headers=http_headers,
        ).json()["session_id"]

    print(f"chat session {sid} (Ctrl-D to exit)")
    while True:
        try:
            line = input("you: ").strip()
        except EOFError:
            break
        if not line:
            continue
        # WS connect with retry-with-backoff. Defends against a transient
        # port-binding window after auto-spawn — uvicorn's port may close
        # and reopen during the bind handshake. Three attempts, 200ms
        # apart. ConnectionRefusedError is the typical symptom.
        ws_cm = None
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                ws_cm = connect(
                    f"ws://127.0.0.1:{port}/stream/{sid}",
                    subprotocols=ws_subprotocols,
                )
                break
            except (ConnectionRefusedError, OSError, InvalidStatus) as exc:
                last_exc = exc
                if attempt < 2:
                    _time.sleep(0.2)
        if ws_cm is None:
            print(
                f"\n[error: could not connect to bridge after 3 attempts: "
                f"{last_exc.__class__.__name__}: {last_exc}]",
                file=_sys.stderr,
            )
            return 1
        with ws_cm as ws:
            ws.send(json.dumps({"message": line}))
            print("nell: ", end="", flush=True)
            while True:
                msg = json.loads(ws.recv())
                if msg.get("type") == "reply_chunk":
                    print(msg["text"], end="", flush=True)
                elif msg.get("type") == "done":
                    print()
                    break
                elif msg.get("type") == "error":
                    print(
                        f"\n[error: {msg.get('detail', msg.get('code'))}]",
                        file=_sys.stderr,
                    )
                    return 1
    # /sessions/close runs the full ingest pipeline (extract via Claude CLI,
    # commit memories, dedupe, soul candidates). On long sessions this can
    # take 30s+. Default httpx timeout (5s) reliably trips. 120s leaves
    # plenty of headroom; if the server is slower than that, the bridge
    # has bigger problems than this client missing.
    try:
        httpx.post(
            f"{base}/sessions/close",
            json={"session_id": sid},
            headers=http_headers,
            timeout=120.0,
        )
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        # Bridge keeps running; ingest may still complete server-side. We
        # don't surface as an error to the user since the chat itself
        # succeeded — but flag it so they know.
        print(
            f"\n[note: session close call timed out ({exc.__class__.__name__}); "
            f"ingest may complete in the background]",
            file=_sys.stderr,
        )
    return 0


def _chat_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell chat` — auto-spawns bridge unless --no-bridge is set."""
    # Note: auto-spawn imports brain.bridge.daemon directly; it does
    # NOT shell out to `nell supervisor`, so this path is unaffected
    # by CLI surface changes.
    from brain.bridge import daemon, state_file
    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    if args.no_bridge:
        return _chat_direct_mode(args)

    # Bug B (2026-05-05 audit-3): we capture cmd_start's verified
    # BridgeReadiness via the out= dict pattern. _chat_via_bridge then
    # uses that directly instead of re-reading state_file — eliminates
    # the race where state_file could rotate or the bridge could die
    # between /health verification and the caller's read.
    readiness_out: dict = {}
    if not state_file.is_running(persona_dir):
        if args.bridge_only:
            print("bridge not running (--bridge-only set)", file=sys.stderr)
            return 1

        # Auto-spawn
        class _StartArgs:
            persona = args.persona
            idle_shutdown = 30
            client_origin = "cli"

        rc = daemon.cmd_start(_StartArgs(), out=readiness_out)
        if rc != 0:
            return rc
    else:
        # Bridge already running — capture its readiness so the chat REPL
        # uses the same handoff-not-re-read path. cmd_start populates the
        # readiness even when returning 2 (already-running) for this case.
        class _NoOpStartArgs:
            persona = args.persona
            idle_shutdown = 30
            client_origin = "cli"

        daemon.cmd_start(_NoOpStartArgs(), out=readiness_out)

    return _chat_via_bridge(args, persona_dir, readiness=readiness_out.get("readiness"))


def _prompt(question: str, default: str | None = None) -> str:
    """Interactive prompt with optional default. EOFError → default or "".

    Test seam: monkeypatched in unit tests via input() since this calls it
    directly. Behavior is intentionally minimal — no readline, no
    coloring; the wizard is short.
    """
    suffix = f" [{default}]" if default is not None else ""
    try:
        raw = input(f"{question}{suffix}: ").strip()
    except EOFError:
        return default if default is not None else ""
    return raw if raw else (default if default is not None else "")


def _prompt_yes_no(question: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = _prompt(f"{question} [{suffix}]").lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def _init_handler(args: argparse.Namespace) -> int:
    """Set up a new persona — interactive wizard or flag-driven.

    Three things this command guarantees:
      1. <NELLBRAIN_HOME>/personas/<name>/ exists
      2. persona_config.json is written with user_name set (closes Bug A
         attribution drift — extractor needs to know who the user is)
      3. voice.md is written from the chosen template (or absent, in
         which case the framework's DEFAULT_VOICE_TEMPLATE applies)

    Optionally migrates OG NellBrain data when --migrate-from is given.

    Non-interactive when all required flags are supplied; otherwise
    prompts for what's missing.
    """
    persona = args.persona
    user_name = args.user_name
    migrate_from = args.migrate_from
    voice_template = args.voice_template
    force = bool(getattr(args, "force", False))

    # ----- interactive fill-in for missing flags -----
    if not persona:
        persona = _prompt("persona name (e.g. 'nell', 'siren')", default="nell")
    try:
        validate_persona_name(persona)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if user_name is None:  # explicit empty string allowed (means "leave unset")
        user_name = _prompt(
            "your name (the human's — used so the brain knows who's "
            "talking to her)",
            default="",
        ) or None

    if migrate_from is None and not getattr(args, "fresh", False):
        if _prompt_yes_no("migrate from OG NellBrain data?", default=False):
            migrate_from = _prompt("path to OG data dir") or None
            if not migrate_from:
                print("no path given — skipping migration", file=sys.stderr)

    if voice_template is None:
        print()
        print("voice template options:")
        for key, desc in VOICE_TEMPLATES.items():
            print(f"  {key}: {desc}")
        voice_template = _prompt("voice template", default="default")
    if voice_template not in VOICE_TEMPLATES:
        print(
            f"error: unknown voice template {voice_template!r} — must be "
            f"one of {sorted(VOICE_TEMPLATES.keys())}",
            file=sys.stderr,
        )
        return 1

    persona_dir = get_persona_dir(persona)

    # ----- guard against clobbering an existing persona -----
    if persona_dir.exists() and any(persona_dir.iterdir()) and not force:
        if migrate_from:
            print(
                f"persona '{persona}' already exists at {persona_dir}.\n"
                f"to re-migrate over an existing persona, run:\n"
                f"  uv run nell migrate --input {migrate_from} "
                f"--install-as {persona} --force",
                file=sys.stderr,
            )
            return 1
        # Fresh-init over existing persona — refuse unless --force; we
        # don't want a wizard run to accidentally overwrite a live
        # voice.md or persona_config.
        print(
            f"persona '{persona}' already exists at {persona_dir}.\n"
            f"re-run with --force to overwrite the persona_config.json + "
            f"voice.md (existing memories/soul/etc are preserved).",
            file=sys.stderr,
        )
        return 1

    # ----- migration path -----
    if migrate_from:
        from brain.migrator.cli import MigrateArgs, run_migrate

        try:
            run_migrate(
                MigrateArgs(
                    input_dir=Path(migrate_from).expanduser(),
                    output_dir=None,
                    install_as=persona,
                    force=force,
                )
            )
        except Exception as exc:  # noqa: BLE001 — migration failure is operator-actionable
            print(f"migration failed: {exc}", file=sys.stderr)
            return 1

    # ----- always: write persona_config + voice.md -----
    persona_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_persona_config(persona_dir, user_name=user_name)
    voice_path = install_voice_template(persona_dir, voice_template)

    print()
    # Keep CLI status text encodable on Windows' legacy cp1252 consoles.
    # GitHub Actions Windows caught this: a leading "✓" raised
    # UnicodeEncodeError during bundled `nell init` smoke.
    print(f"OK persona '{persona}' ready at {persona_dir}")
    print(f"  - {config_path.name}: user_name={user_name!r}")
    if voice_path is not None:
        print(f"  - {voice_path.name}: copied from '{voice_template}' template")
        if voice_template == "nell-example":
            print(
                "    edit it before chatting — replace Nell-specific "
                "identity content with your own"
            )
    else:
        print(
            "  - voice.md: not written; framework's DEFAULT_VOICE_TEMPLATE "
            "applies on first chat"
        )
    if migrate_from:
        print(f"  - migrated from {migrate_from}")
    print()
    print(f"next: uv run nell chat --persona {persona}")
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

    # nell init — interactive setup wizard for new personas (both fresh and
    # OG-migration paths). Closes the user-experience gap where forkers
    # had to hand-edit persona_config.json + voice.md.
    init_sub = subparsers.add_parser(
        "init",
        help="Set up a new persona (interactive wizard, with non-interactive flags).",
    )
    init_sub.add_argument(
        "--persona",
        default=None,
        help="Persona name. Prompts if omitted.",
    )
    init_sub.add_argument(
        "--user-name",
        default=None,
        help=(
            "Your name (the human's). Used by the ingest extractor so the "
            "brain doesn't conflate you with historical figures referenced "
            "in soul context. Prompts if omitted; pass empty string ('') "
            "to leave unset."
        ),
    )
    src_group = init_sub.add_mutually_exclusive_group()
    src_group.add_argument(
        "--migrate-from",
        default=None,
        help="Path to OG NellBrain data dir to migrate from. Prompts if omitted.",
    )
    src_group.add_argument(
        "--fresh",
        action="store_true",
        help="Skip the migrate prompt — start from a clean persona dir.",
    )
    init_sub.add_argument(
        "--voice-template",
        default=None,
        choices=sorted(VOICE_TEMPLATES.keys()),
        help="Voice.md starter. Prompts if omitted.",
    )
    init_sub.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing persona's persona_config.json + voice.md.",
    )
    init_sub.set_defaults(func=_init_handler)

    status_sub = subparsers.add_parser(
        "status",
        help="Show local persona, memory, and bridge status without contacting providers.",
    )
    status_sub.add_argument(
        "--persona",
        default="nell",
        help="Persona name to inspect. Defaults to nell.",
    )
    status_sub.set_defaults(func=_status_handler)

    memory_sub = subparsers.add_parser(
        "memory",
        help="Inspect local persona memories safely.",
    )
    memory_actions = memory_sub.add_subparsers(dest="action", required=True)

    memory_list = memory_actions.add_parser("list", help="List recent active memories.")
    memory_list.add_argument("--persona", default="nell", help="Persona name. Defaults to nell.")
    memory_list.add_argument(
        "--limit",
        type=_positive_int,
        default=20,
        help="Maximum active memories to show (default 20).",
    )
    memory_list.set_defaults(func=_memory_list_handler)

    memory_search = memory_actions.add_parser("search", help="Search active memories by text.")
    memory_search.add_argument("query", help="Non-empty text to search for.")
    memory_search.add_argument("--persona", default="nell", help="Persona name. Defaults to nell.")
    memory_search.add_argument(
        "--limit",
        type=_positive_int,
        default=20,
        help="Maximum matching memories to show (default 20).",
    )
    memory_search.set_defaults(func=_memory_search_handler)

    memory_show = memory_actions.add_parser("show", help="Show one full memory by id.")
    memory_show.add_argument("memory_id", help="Memory UUID to inspect.")
    memory_show.add_argument("--persona", default="nell", help="Persona name. Defaults to nell.")
    memory_show.set_defaults(func=_memory_show_handler)

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
        choices=["ddgs", "noop"],
        help=(
            "(developer override) Web searcher for research engine — ddgs, noop, "
            "Defaults to the value in {persona}/persona_config.json."
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
        choices=["ddgs", "noop"],
        help=(
            "(developer override) Web searcher — ddgs, noop. "
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

    # Daemon handlers — shared by `supervisor` (canonical) and `bridge` (deprecated alias).
    from brain.bridge.daemon import (
        cmd_restart,
        cmd_run,
        cmd_start,
        cmd_status,
        cmd_stop,
        cmd_tail,
        cmd_tail_log,
    )

    # nell supervisor — canonical bridge lifecycle command.
    s_sub = subparsers.add_parser(
        "supervisor",
        help="Manage the per-persona bridge daemon — canonical lifecycle command.",
    )
    s_actions = s_sub.add_subparsers(dest="action", required=True)

    def _add_persona_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--persona", required=True)

    def _nonneg_int(v: str) -> int:
        iv = int(v)
        if iv < 0:
            raise argparse.ArgumentTypeError("must be a non-negative integer")
        return iv

    client_origin_choices = ["cli", "tauri", "tests", "launchd"]

    s_start = s_actions.add_parser("start", help="Start the bridge daemon.")
    _add_persona_arg(s_start)
    s_start.add_argument(
        "--idle-shutdown",
        type=float,
        default=30,
        help="Idle-shutdown threshold in minutes (0 = never).",
    )
    s_start.add_argument("--client-origin", default="cli", choices=client_origin_choices)
    s_start.set_defaults(func=cmd_start)

    s_run = s_actions.add_parser(
        "run",
        help="Run the bridge in the foreground for launchd/system service managers.",
    )
    _add_persona_arg(s_run)
    s_run.add_argument(
        "--idle-shutdown",
        type=float,
        default=0,
        help="Idle-shutdown threshold in minutes (0 = never, default for service mode).",
    )
    s_run.add_argument("--client-origin", default="launchd", choices=client_origin_choices)
    s_run.set_defaults(func=cmd_run)

    s_stop = s_actions.add_parser("stop", help="Stop the bridge daemon.")
    _add_persona_arg(s_stop)
    s_stop.add_argument("--timeout", type=float, default=180.0)
    s_stop.set_defaults(func=cmd_stop)

    s_status = s_actions.add_parser("status", help="Show bridge daemon status.")
    _add_persona_arg(s_status)
    s_status.set_defaults(func=cmd_status)

    s_restart = s_actions.add_parser(
        "restart", help="Stop and start — gated on stop success.",
    )
    _add_persona_arg(s_restart)
    s_restart.add_argument("--idle-shutdown", type=float, default=30)
    s_restart.add_argument("--client-origin", default="cli", choices=client_origin_choices)
    s_restart.add_argument("--timeout", type=float, default=180.0)
    s_restart.set_defaults(func=cmd_restart)

    s_tail_events = s_actions.add_parser(
        "tail-events", help="Tail /events as JSON lines."
    )
    _add_persona_arg(s_tail_events)
    s_tail_events.set_defaults(func=cmd_tail)

    s_tail_log = s_actions.add_parser(
        "tail-log", help="Tail the bridge log file (cross-platform).",
    )
    _add_persona_arg(s_tail_log)
    s_tail_log.add_argument(
        "-n",
        "--lines",
        dest="lines",
        type=_nonneg_int,
        default=50,
        help="Print the last N lines (default 50; 0 = none, useful with -f).",
    )
    s_tail_log.add_argument(
        "-f",
        "--follow",
        dest="follow",
        action="store_true",
        default=False,
        help="Follow the log: emit new lines as written. Ctrl-c to exit.",
    )
    s_tail_log.set_defaults(func=cmd_tail_log)

    # nell service — OS integration surface. First slice is deliberately
    # non-mutating: generate the LaunchAgent plist and run doctor checks
    # before install/bootstrap commands are exposed.
    svc_sub = subparsers.add_parser(
        "service",
        help="Inspect OS service integration for the per-persona supervisor.",
    )
    svc_actions = svc_sub.add_subparsers(dest="action", required=True)

    def _add_service_common(p: argparse.ArgumentParser) -> None:
        # Lazy-import so the CLI default tracks the launchd library's
        # current default — without this they drifted (the library
        # added ``~/.local/bin`` for claude lookup, the CLI kept the
        # OG path string, and ``nell service doctor`` reported
        # ``claude not found in launchd PATH`` even though the plist
        # builder would have included it).
        from brain.service.launchd import DEFAULT_LAUNCHD_PATH

        p.add_argument("--persona", required=True)
        p.add_argument(
            "--nell-path",
            default=None,
            help="Absolute path to the nell executable for launchd ProgramArguments[0].",
        )
        p.add_argument(
            "--env-path",
            default=DEFAULT_LAUNCHD_PATH,
            help="PATH value launchd should provide to the service.",
        )

    svc_print = svc_actions.add_parser(
        "print-plist",
        help="Print the LaunchAgent plist XML without installing it.",
    )
    _add_service_common(svc_print)
    svc_print.add_argument(
        "--nellbrain-home",
        default=None,
        help="Optional NELLBRAIN_HOME value to embed in the LaunchAgent environment.",
    )
    svc_print.set_defaults(func=_service_print_plist_handler)

    svc_install = svc_actions.add_parser(
        "install",
        help="Write and bootstrap the LaunchAgent for a persona.",
    )
    _add_service_common(svc_install)
    svc_install.add_argument(
        "--nellbrain-home",
        default=None,
        help="Optional NELLBRAIN_HOME value to embed in the LaunchAgent environment.",
    )
    svc_install.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plist that would be installed; do not write or bootstrap.",
    )
    svc_install.set_defaults(func=_service_install_handler)

    svc_uninstall = svc_actions.add_parser(
        "uninstall",
        help="Boot out the LaunchAgent and remove its plist.",
    )
    svc_uninstall.add_argument("--persona", required=True)
    svc_uninstall.add_argument(
        "--keep-plist",
        action="store_true",
        help="Boot out the service but leave the plist file in place.",
    )
    svc_uninstall.set_defaults(func=_service_uninstall_handler)

    svc_status = svc_actions.add_parser(
        "status",
        help="Show LaunchAgent installed/loaded state.",
    )
    svc_status.add_argument("--persona", required=True)
    svc_status.set_defaults(func=_service_status_handler)

    svc_doctor = svc_actions.add_parser(
        "doctor",
        help="Run non-mutating launchd service preflight checks.",
    )
    _add_service_common(svc_doctor)
    svc_doctor.set_defaults(func=_service_doctor_handler)

    # nell daemon-state — recovery + maintenance for daemon_state.json.
    # This file holds the cross-process residue (last_dream / last_reflex /
    # last_research / last_heartbeat) that the chat layer reads to colour
    # its responses. When a constant or format changes (e.g. summary cap
    # bump), existing entries can be stale-by-truncation; refresh rebuilds
    # them from active memories instead of waiting for the next engine fire.
    ds_sub = subparsers.add_parser(
        "daemon-state",
        help="Maintain daemon_state.json (residue cache for engine fires).",
    )
    ds_actions = ds_sub.add_subparsers(dest="action", required=True)

    ds_refresh = ds_actions.add_parser(
        "refresh",
        help=(
            "Rewrite last_dream / last_reflex / last_research summaries from "
            "the most recent active memory of each type. Useful after a "
            "summary-cap or format change so existing entries pick up the new "
            "cap without waiting for the next engine fire."
        ),
    )
    ds_refresh.add_argument("--persona", required=True, help="Persona name (required).")
    ds_refresh.set_defaults(func=_daemon_state_refresh_handler)

    # nell works — read-only inspection of brain-authored creative artifacts.
    # Saving is brain-territory via the save_work MCP tool, not a CLI command.
    w_sub = subparsers.add_parser(
        "works",
        help="Inspect brain-authored creative artifacts (read-only).",
    )
    w_actions = w_sub.add_subparsers(dest="action", required=True)

    w_list = w_actions.add_parser("list", help="List recent works.")
    w_list.add_argument("--persona", required=True)
    w_list.add_argument(
        "--type",
        default=None,
        help="Filter by type (story/code/planning/idea/role_play/letter/other).",
    )
    w_list.add_argument("--limit", type=int, default=20)
    w_list.set_defaults(func=_works_list_handler)

    w_search = w_actions.add_parser("search", help="Search works by title/summary/content.")
    w_search.add_argument("--persona", required=True)
    w_search.add_argument("--query", required=True)
    w_search.add_argument("--type", default=None)
    w_search.add_argument("--limit", type=int, default=20)
    w_search.set_defaults(func=_works_search_handler)

    w_read = w_actions.add_parser("read", help="Read one work's full content.")
    w_read.add_argument("--persona", required=True)
    w_read.add_argument("--id", required=True)
    w_read.set_defaults(func=_works_read_handler)

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

    # nell health show/check/acknowledge — read-only inspection + append-only audit.
    # Per Task 13: no restore/add/delete/approve/reject actions are wired.
    h_sub = subparsers.add_parser(
        "health",
        help="Inspect and acknowledge brain health alarms (read-only + audit-append).",
    )
    h_actions = h_sub.add_subparsers(dest="action", required=True)

    h_show = h_actions.add_parser("show", help="Print pending alarms + recent self-treatments.")
    h_show.add_argument("--persona", required=True, help="Persona name (required).")
    h_show.set_defaults(func=_health_show_handler)

    h_check = h_actions.add_parser(
        "check", help="Run a full file integrity walk; exit 2 if unhealable alarms exist."
    )
    h_check.add_argument("--persona", required=True, help="Persona name (required).")
    h_check.set_defaults(func=_health_check_handler)

    h_ack = h_actions.add_parser(
        "acknowledge",
        help="Acknowledge pending alarms (appends to audit log; no destructive changes).",
    )
    h_ack.add_argument("--persona", required=True, help="Persona name (required).")
    h_ack.add_argument(
        "--file",
        dest="ack_file",
        default=None,
        help="Acknowledge a specific file. Mutually exclusive with --all.",
    )
    h_ack.add_argument(
        "--all",
        dest="ack_all",
        action="store_true",
        default=False,
        help="Acknowledge all pending alarms (default if neither --file nor --all is given).",
    )
    h_ack.set_defaults(func=_health_acknowledge_handler)

    # nell soul — autonomous soul management (SP-5)
    soul_sub = subparsers.add_parser(
        "soul",
        help="Manage the persona's permanent soul (crystallizations).",
    )
    soul_actions = soul_sub.add_subparsers(dest="action", required=True)

    # nell soul list
    sl_list = soul_actions.add_parser("list", help="List active crystallizations.")
    sl_list.add_argument("--persona", required=True, help="Persona name (required).")
    sl_list.add_argument("--limit", type=int, default=None, help="Show only the last N.")
    sl_list.set_defaults(func=_soul_list_handler)

    # nell soul revoke
    sl_revoke = soul_actions.add_parser("revoke", help="Revoke a crystallization.")
    sl_revoke.add_argument("--persona", required=True, help="Persona name (required).")
    sl_revoke.add_argument("--id", required=True, help="Crystallization UUID to revoke.")
    sl_revoke.add_argument("--reason", required=True, help="Reason for revocation.")
    sl_revoke.set_defaults(func=_soul_revoke_handler)

    # nell soul candidates
    sl_cands = soul_actions.add_parser(
        "candidates", help="List pending soul_candidates.jsonl entries."
    )
    sl_cands.add_argument("--persona", required=True, help="Persona name (required).")
    sl_cands.add_argument("--limit", type=int, default=None, help="Show only the last N.")
    sl_cands.set_defaults(func=_soul_candidates_handler)

    # nell soul audit
    sl_audit = soul_actions.add_parser("audit", help="Tail soul_audit.jsonl entries.")
    sl_audit.add_argument("--persona", required=True, help="Persona name (required).")
    sl_audit.add_argument(
        "--limit", type=int, default=20, help="Show only the last N entries (default 20)."
    )
    sl_audit.set_defaults(func=_soul_audit_handler)

    # nell chat — keystone chat engine (SP-6)
    chat_sub = subparsers.add_parser(
        "chat",
        help="Chat with a persona. One-shot or interactive REPL.",
    )
    chat_sub.add_argument(
        "--persona",
        required=True,
        help=(
            "Persona name (required). "
            "To port existing OG NellBrain data: `nell migrate --input /path/to/og/data "
            "--install-as <name>`. To start fresh: create personas/<name>/ manually."
        ),
    )
    chat_sub.add_argument(
        "--provider",
        default=None,
        help=(
            "(developer override) LLM provider — claude-cli, fake, ollama. "
            "Defaults to the value in {persona}/persona_config.json."
        ),
    )
    chat_sub.add_argument(
        "message",
        nargs="?",
        default=None,
        help="Single message for one-shot mode. Omit for interactive REPL.",
    )
    chat_sub.add_argument(
        "--no-bridge",
        action="store_true",
        default=False,
        help="Bypass the bridge daemon and call engine.respond() in-process.",
    )
    chat_sub.add_argument(
        "--bridge-only",
        action="store_true",
        default=False,
        help="Error out if bridge isn't running; do not auto-spawn.",
    )
    chat_sub.set_defaults(func=_chat_handler)

    # nell soul review
    sl_review = soul_actions.add_parser(
        "review",
        help="Run an autonomous soul review pass — brain decides on pending candidates.",
    )
    sl_review.add_argument("--persona", required=True, help="Persona name (required).")
    sl_review.add_argument(
        "--max", type=int, default=5, help="Maximum candidates to evaluate (default 5)."
    )
    sl_review.add_argument(
        "--confidence-threshold",
        type=int,
        default=7,
        dest="confidence_threshold",
        help="Minimum confidence to accept/reject; below this → defer (default 7).",
    )
    sl_review.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate + audit log but skip all writes.",
    )
    sl_review.add_argument(
        "--provider",
        default=None,
        help="(developer override) LLM provider — claude-cli, fake, ollama.",
    )
    sl_review.set_defaults(func=_soul_review_handler)

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
