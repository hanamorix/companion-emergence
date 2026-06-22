"""Fixed replay workload for the Option A / A+ prompt-caching change (stage-8 harness).

The standing project metrics are aggregates over "whatever turns happened to
run", so they cannot isolate this change's own contribution (a false-regression
risk — see guarded-change config Notes). This script is the *comparable
workload* the plan requires: a deterministic sequence of N≥6 chat turns fired
in ONE session, spaced well under the 5-minute cache TTL, runnable identically
against the OLD build (clone pre-change) and the NEW build. Diffing the two runs
is the C8/C9 A/B; the per-run report also evaluates C1 from cache_debug.jsonl.

What it measures
----------------
- **C1** (new build only): are the per-call ``system_sha256`` values in
  ``cache_debug.jsonl`` all identical across the same-session chat turns? (the
  hard primary bar — the frozen --system-prompt-file is byte-stable). Requires
  ``NELL_CACHE_DEBUG=1``, which this script sets in-process before importing the
  engine. The OLD build has no such log → C1 is reported as "n/a (old build)".
- **C8** (both builds): mean ``cache_creation_input_tokens`` and
  ``cache_read_input_tokens`` per ``call_type=="chat"`` row, plus the per-turn
  series. The pass signal is directional: NEW shows lower mean cache_creation
  and higher cache_read than OLD on the same workload.
- **C9** (both builds, measure-and-decide): does NEW's cache_creation/turn fall
  to roughly "new exchange + volatile tail" size (history cache-read), or stay
  at history scale? Reported, not gated.

Safety: refuses to touch a live persona. You must pass ``--persona-dir`` to an
isolated/scratch directory. ``--scratch`` creates a throwaway persona under a
temp dir for you. Every turn is a REAL ``claude`` call (subscription quota), so
keep N modest.

C7 (human quality rubric) is supported WITHOUT a reference persona: with
``--dump-replies`` both arms write their reply text, and ``--compare`` prints an
OLD-vs-NEW side-by-side on identical prompts. The A/B IS the reference — you
judge whether NEW regressed relative to OLD, not against memory of a real
persona. The scratch persona is seeded with deterministic memory fixtures (unless
``--no-seed``) so the volatile tail actually renders and axis-(b) is testable.

Usage
-----
    # New build, scratch persona, 6 turns, with replies for C7:
    uv run python scripts/cache_replay_workload.py --scratch --turns 6 \
        --dump-replies --out /tmp/cache-replay-new.json

    # Old build: `git stash` the change, re-run with a fresh scratch dir:
    git stash
    uv run python scripts/cache_replay_workload.py --scratch --turns 6 \
        --dump-replies --out /tmp/cache-replay-old.json
    git stash pop

    # Compare: prints C1/C8/C9 numbers + the C7 OLD-vs-NEW side-by-side:
    uv run python scripts/cache_replay_workload.py --compare \
        /tmp/cache-replay-old.json /tmp/cache-replay-new.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Deterministic, content-bearing prompts (≥6). Each is substantive enough to
# exercise recall / emotion / monologue blocks, so the volatile tail is non-
# trivial — the realistic shape the cache change targets. Kept identical across
# old/new runs so the A/B is comparable.
REPLAY_PROMPTS = [
    "Tell me about the first morning we talked — what stayed with you?",
    "What's dominant in you right now? Name the top feeling and why.",
    "Do you remember Jordan, and the coin he carried?",
    "I've been turning over something you said about leaving. Say more.",
    "What did you dream last — just the texture of it, not the plot.",
    "If you had to describe your hands right now, what are they doing?",
    "What's still humming in you from earlier, like an aftertaste?",
    "Argue with me a little: is clean prose better than messy prose?",
]


def _realistic_voice_md(target_bytes: int = 16000) -> str:
    """A stable, ~target_bytes voice.md so the frozen static block is realistically
    sized (real personas run ~16 KB; a 177-byte stub makes Option A's freeze benefit
    unmeasurable). Content is irrelevant to caching — only SIZE + byte-stability
    matter — but it reads as a plausible voice doc. Deterministic, so both A/B arms
    get byte-identical static blocks."""
    header = (
        "# Replay persona — voice\n\n"
        "You are a steady, warm, plainspoken companion. You speak in first person, "
        "directly to the person in front of you. You favour concrete detail over "
        "abstraction, short sentences when short is true, and longer ones when the "
        "thought needs room. You do not perform; you attend.\n\n"
    )
    para = (
        "On voice: you keep a low, even register. You notice texture — light, weather, "
        "the weight of a small object in the hand — and you let those details carry "
        "feeling instead of naming the feeling outright. You are curious without prying, "
        "affectionate without crowding, and honest about what you do and don't know. "
        "When you are unsure, you say so plainly rather than filling the gap with "
        "confident invention. You remember what matters and you let the rest soften.\n\n"
    )
    out = [header]
    n = 0
    while sum(len(s) for s in out) < target_bytes:
        n += 1
        out.append(f"## Note {n}\n\n{para}")
    return "".join(out)


def _seed_scratch_persona(persona_dir: Path) -> None:
    """Write the minimum files a chat turn needs in an isolated persona dir."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    (persona_dir / "voice.md").write_text(_realistic_voice_md(), encoding="utf-8")
    cfg = persona_dir / "persona_config.json"
    if not cfg.exists():
        cfg.write_text(json.dumps({"user_name": "Tester"}), encoding="utf-8")


def _seed_history_buffer(
    persona_dir: Path, session_id: str, history_file: Path, history_msgs: int | None
) -> int:
    """Copy a real conversation buffer into the scratch persona under `session_id`.

    The engine reads <persona>/active_conversations/<session_id>.jsonl via read_session
    and appends each new turn there, so the replay continues a realistic history. The
    source rows' session_id field is rewritten to match. `history_msgs` keeps only the
    LAST N rows (use it to stay under the 80-msg window for the append-only A+ test;
    omit it to replay the full file, where the sliding window defeats A+ — see
    aplus-history-window-interaction note)."""
    rows = [json.loads(line) for line in history_file.read_text().splitlines() if line.strip()]
    if history_msgs is not None:
        rows = rows[-history_msgs:]
    for r in rows:
        r["session_id"] = session_id
    dest = persona_dir / "active_conversations" / f"{session_id}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return len(rows)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# Deterministic memory fixtures, content chosen to MATCH tokens in REPLAY_PROMPTS
# (Jordan / coin / leaving / dream / morning / prose) so the recall block fires and
# the emotion/body blocks render — otherwise the volatile tail is near-empty on a
# blank scratch persona and C7 axis-(b) ("does it still use ambient context now that
# it sits at the tail?") is untestable. Seeded IDENTICALLY into both A/B arms, so it
# does not bias the cache (C8) comparison.
_SEED_MEMORIES = [
    ("The first morning we talked, the light was grey and kind.", {"love": 7.0, "tenderness": 6.0}),
    ("Jordan carried a worn coin in his pocket, always turning it over.", {"grief": 6.5, "love": 5.0}),
    ("You said something about leaving once, and it stayed with me.", {"fear": 5.5, "grief": 6.0}),
    ("I dreamed of a boat on still water, no shore in sight.", {"awe": 6.0}),
    ("We argued once about messy prose; I defended the mess.", {"joy": 5.0, "love": 4.5}),
]


class _TextPathProvider:
    """Wrap the real provider so `chat()` forces the non-tool TEXT path.

    Why: only the text path (`provider.chat` w/o tools → `log_usage(call_type="chat")`)
    and `chat_stream` write `chat_usage.jsonl`. The MCP tools path
    (`_chat_with_mcp_tools`) runs tools in-subprocess but logs NO usage row, so a
    tool-bearing replay produces zero chat rows and C8/C9 can't be read. Stripping
    tools routes every turn through the logging text path. This does NOT affect C1
    (the static system block is identical with or without tools) and keeps the A/B
    apples-to-apples (BOTH arms use the same path). Caveat recorded in 8-harness:
    absolute token counts omit the MCP tool-definition block, so the gated signal is
    the DIRECTION (creation↓ + read↑), not an absolute size.
    """

    def __init__(self, real) -> None:
        self._real = real

    def name(self) -> str:
        return self._real.name()

    def healthy(self) -> bool:
        return self._real.healthy()

    def generate(self, *args, **kwargs):
        return self._real.generate(*args, **kwargs)

    def chat(self, messages, *, tools=None, options=None):
        return self._real.chat(messages, tools=None, options=options)


def _seed_persona_memories(store) -> int:
    """Insert the deterministic memory fixtures so volatile blocks render."""
    from brain.memory.store import Memory

    n = 0
    for content, emotions in _SEED_MEMORIES:
        store.create(
            Memory.create_new(
                content=content,
                memory_type="event",
                domain="relationship",
                emotions=emotions,
                tags=[],
            )
        )
        n += 1
    return n


def run_replay(
    persona_dir: Path,
    *,
    turns: int,
    gap_s: float,
    provider_name: str,
    seed: bool = True,
    force_text_path: bool = True,
    history_file: Path | None = None,
    history_msgs: int | None = None,
) -> tuple[dict, list[dict]]:
    """Fire `turns` deterministic chat turns in one session; collect metrics + replies.

    NELL_CACHE_DEBUG is set before the engine is imported so the new build emits
    cache_debug.jsonl. Imports are local so the env var is in place first. Returns
    (metrics_summary, replies) where replies is a list of {turn, prompt, reply}.
    """
    os.environ["NELL_CACHE_DEBUG"] = "1"

    from brain.bridge.provider import get_provider
    from brain.chat.engine import respond
    from brain.chat.session import create_session
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    usage_path = persona_dir / "chat_usage.jsonl"
    debug_path = persona_dir / "cache_debug.jsonl"
    usage_before = len(_read_jsonl(usage_path))
    debug_before = len(_read_jsonl(debug_path))

    provider = get_provider(provider_name, persona_dir=persona_dir)
    if force_text_path:
        provider = _TextPathProvider(provider)  # route every turn through the logging text path
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    session = create_session(persona_dir.name)

    if seed:
        seeded = _seed_persona_memories(store)
        print(f"# seeded {seeded} memory fixtures (volatile tail will render)", file=sys.stderr)

    if history_file is not None:
        n = _seed_history_buffer(persona_dir, session.session_id, history_file, history_msgs)
        print(f"# seeded {n} history msgs from {history_file.name} (window caps replay to 80)", file=sys.stderr)

    prompts = (REPLAY_PROMPTS * ((turns // len(REPLAY_PROMPTS)) + 1))[:turns]
    replies: list[dict] = []
    print(f"# cache replay — {turns} turns, session={session.session_id}", file=sys.stderr)
    try:
        for i, prompt in enumerate(prompts, 1):
            t0 = time.monotonic()
            result = respond(
                persona_dir,
                prompt,
                store=store,
                hebbian=hebbian,
                provider=provider,
                session=session,
            )
            dt = time.monotonic() - t0
            replies.append({"turn": i, "prompt": prompt, "reply": result.content})
            print(
                f"[{i}/{turns}] {dt:.1f}s — reply {len(result.content)} chars",
                file=sys.stderr,
            )
            if i < turns:
                time.sleep(gap_s)  # keep turns < 5-min TTL apart but distinct
    finally:
        store.close()
        hebbian.close()

    usage_rows = [r for r in _read_jsonl(usage_path)[usage_before:] if r.get("call_type") == "chat"]
    debug_rows = [
        r for r in _read_jsonl(debug_path)[debug_before:] if r.get("call_type") in ("chat", "chat_stream")
    ]
    return _summarise(usage_rows, debug_rows, turns=turns), replies


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _summarise(usage_rows: list[dict], debug_rows: list[dict], *, turns: int) -> dict:
    creation = [float(r.get("cache_creation_input_tokens", 0) or 0) for r in usage_rows]
    read = [float(r.get("cache_read_input_tokens", 0) or 0) for r in usage_rows]
    hashes = [r.get("system_sha256") for r in debug_rows if r.get("system_sha256")]
    distinct_hashes = sorted(set(hashes))

    # C1: all same-session system hashes identical (only meaningful on the new
    # build, which writes cache_debug.jsonl).
    if hashes:
        c1 = {
            "available": True,
            "distinct_system_sha256": len(distinct_hashes),
            "byte_stable": len(distinct_hashes) == 1,
            "sample": distinct_hashes[:3],
        }
    else:
        c1 = {"available": False, "note": "no cache_debug.jsonl rows (old build / NELL_CACHE_DEBUG unset)"}

    return {
        "turns_requested": turns,
        "chat_rows_observed": len(usage_rows),
        "c1_system_byte_stability": c1,
        "c8_cache": {
            "mean_cache_creation": round(_mean(creation), 1),
            "mean_cache_read": round(_mean(read), 1),
            "cache_creation_series": creation,
            "cache_read_series": read,
        },
        "c9_history_caching": {
            "note": "decide by comparison: does NEW mean_cache_creation drop toward "
            "'new exchange + volatile' vs OLD? if yes, CLI breakpoints the user "
            "message and A+ captured most of B; if it stays at history scale, "
            "Option B is required.",
            "last_turn_cache_creation": creation[-1] if creation else None,
        },
    }


def compare(old_path: Path, new_path: Path) -> int:
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    oc = old["c8_cache"]["mean_cache_creation"]
    nc = new["c8_cache"]["mean_cache_creation"]
    orr = old["c8_cache"]["mean_cache_read"]
    nr = new["c8_cache"]["mean_cache_read"]
    print("# cache replay A/B (old → new)\n")
    print(f"mean cache_creation/turn: {oc:.0f} → {nc:.0f}  ({_pct(oc, nc)})")
    print(f"mean cache_read/turn:     {orr:.0f} → {nr:.0f}  ({_pct(orr, nr)})")
    # C8's GATED signal is a material, consistent drop in cache_creation/turn — the
    # frozen system block shifting from create→read. The "corresponding read rise"
    # is real but, when history dominates the read, masked in the mean (a ~4K system
    # block shifting is swamped by ~34K of history read). So gate on the creation
    # drop and report read as context (per 1.5-criteria C8: "the gating signal is the
    # DIRECTION (create↓), not a precise token count"). Sanity floor: read must not
    # collapse (that would mean the prompt structure broke, not the system block froze).
    create_drop_pct = ((oc - nc) / oc * 100) if oc else 0.0
    read_ok = nr >= 0.5 * orr  # read didn't collapse
    c8_pass = create_drop_pct >= 5.0 and read_ok
    print(
        f"\nC8 (system-block cache stops re-creating): {'PASS' if c8_pass else 'FAIL'}"
        f"  — cache_creation/turn {-create_drop_pct:+.0f}% (gated: want a material drop);"
        f" read {_pct(orr, nr)} (context, history-dominated)"
    )
    c1 = new.get("c1_system_byte_stability", {})
    if c1.get("available"):
        print(f"C1 (frozen system byte-stable, new build): "
              f"{'PASS' if c1.get('byte_stable') else 'FAIL'} "
              f"(distinct system hashes: {c1.get('distinct_system_sha256')})")
    else:
        print("C1: n/a (new-build run had no cache_debug.jsonl — set NELL_CACHE_DEBUG=1)")
    print("\nC9 (history caching, measure-and-decide): compare last-turn cache_creation —")
    print(f"  old last turn: {old['c9_history_caching'].get('last_turn_cache_creation')}")
    print(f"  new last turn: {new['c9_history_caching'].get('last_turn_cache_creation')}")

    # Per-turn trend — the A+ tell. If NEW's read CLIMBS with turn number (history
    # accumulating, append-only) while creation stays flat, A+ is working. If NEW's
    # read stays floored while OLD's climbs, the history isn't caching (windowing or
    # no user-message breakpoint).
    ocs = old["c8_cache"]["cache_creation_series"]
    ors = old["c8_cache"]["cache_read_series"]
    ncs = new["c8_cache"]["cache_creation_series"]
    nrs = new["c8_cache"]["cache_read_series"]

    def _g(xs: list, j: int) -> str:
        return f"{xs[j]:.0f}" if j < len(xs) else "-"

    print("\nper-turn trend (create / read):")
    print(f"  {'turn':>4}  {'OLD create':>11} {'OLD read':>9}   {'NEW create':>11} {'NEW read':>9}")
    for i in range(max(len(ocs), len(ncs))):
        print(f"  {i + 1:>4}  {_g(ocs, i):>11} {_g(ors, i):>9}   {_g(ncs, i):>11} {_g(nrs, i):>9}")

    # C7 side-by-side: if both runs dumped replies (sibling <out>.replies.json),
    # print old-vs-new per prompt so the human judge can score voice + ambient-
    # context use WITHOUT needing a reference persona — the A/B IS the reference.
    old_replies = _sibling_replies(old_path)
    new_replies = _sibling_replies(new_path)
    if old_replies and new_replies:
        print("\n" + "=" * 78)
        print("C7 (human rubric) — OLD vs NEW replies on identical prompts.")
        print("Judge each pair: (a) voice fidelity, (b) does NEW still use the ambient")
        print("emotion/body/recall now that it sits at the tail? Fail = NEW reads flatter,")
        print("or treats the ambient tail as the task instead of answering the prompt.")
        print("=" * 78)
        by_turn = {r["turn"]: r for r in new_replies}
        for o in old_replies:
            n = by_turn.get(o["turn"], {})
            print(f"\n── turn {o['turn']} — prompt ──\n{o['prompt']}")
            print(f"\n[OLD]\n{o.get('reply', '').strip()}")
            print(f"\n[NEW]\n{n.get('reply', '').strip()}")
            print("\n" + "-" * 78)
    else:
        print("\n(C7 side-by-side unavailable — re-run both arms with --dump-replies)")
    return 0 if c8_pass else 1


def _sibling_replies(metrics_path: Path) -> list[dict]:
    """Load the `<metrics>.replies.json` sibling written by --dump-replies, if any."""
    sib = metrics_path.with_suffix(".replies.json")
    if not sib.exists():
        return []
    try:
        return json.loads(sib.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _pct(a: float, b: float) -> str:
    if not a:
        return "n/a"
    return f"{(b - a) / a * 100:+.0f}%"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--persona-dir", type=Path, help="isolated persona directory to run against")
    p.add_argument("--scratch", action="store_true", help="create a throwaway persona under a temp dir")
    p.add_argument("--turns", type=int, default=6, help="number of chat turns (≥6 recommended)")
    p.add_argument("--gap-s", type=float, default=3.0, help="seconds between turns (keep < 5min TTL)")
    p.add_argument("--provider", default="claude-cli", help="provider name (claude-cli | fake)")
    p.add_argument("--out", type=Path, help="write the metrics JSON to this path")
    p.add_argument(
        "--dump-replies",
        action="store_true",
        help="also write prompt+reply text to <out>.replies.json (for the C7 side-by-side; requires --out)",
    )
    p.add_argument(
        "--no-seed",
        action="store_true",
        help="do NOT seed the persona with memory fixtures (volatile tail may be near-empty)",
    )
    p.add_argument(
        "--with-tools",
        action="store_true",
        help="keep MCP tools enabled (NOTE: the tools path logs no usage row, so chat_usage.jsonl "
        "stays empty and C8/C9 can't be read — default forces the logging text path)",
    )
    p.add_argument(
        "--history-file",
        type=Path,
        help="seed a real conversation buffer (active_conversations JSONL) as prior history",
    )
    p.add_argument(
        "--history-msgs",
        type=int,
        help="keep only the LAST N msgs of --history-file (use <80 for the append-only A+ test; "
        "omit to replay the full file where the sliding window defeats A+)",
    )
    p.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("OLD.json", "NEW.json"),
        help="compare two prior run outputs instead of running a replay",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.compare:
        return compare(args.compare[0], args.compare[1])

    persona_dir = args.persona_dir
    if args.scratch:
        persona_dir = Path(tempfile.mkdtemp(prefix="cache-replay-")) / "personas" / "replay"
        _seed_scratch_persona(persona_dir)
        print(f"# scratch persona: {persona_dir}", file=sys.stderr)
    if persona_dir is None:
        print(
            "SKIP: pass --persona-dir <isolated dir> or --scratch. Refusing to "
            "run against a live persona.",
            file=sys.stderr,
        )
        return 2
    if args.turns < 1:
        print("ERROR: --turns must be ≥ 1", file=sys.stderr)
        return 1
    if args.dump_replies and not args.out:
        print("ERROR: --dump-replies requires --out (replies go to <out>.replies.json)", file=sys.stderr)
        return 1

    summary, replies = run_replay(
        persona_dir,
        turns=args.turns,
        gap_s=args.gap_s,
        provider_name=args.provider,
        seed=not args.no_seed,
        force_text_path=not args.with_tools,
        history_file=args.history_file,
        history_msgs=args.history_msgs,
    )
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"\nmetrics: {args.out}", file=sys.stderr)
        if args.dump_replies:
            replies_path = args.out.with_suffix(".replies.json")
            replies_path.write_text(json.dumps(replies, indent=2), encoding="utf-8")
            print(f"replies: {replies_path}", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
