#!/usr/bin/env python3
"""Pick the brain tools the CLI must never defer behind ToolSearch (#525) — read-only over a persona's logs.

    uv run python scripts/select_always_loaded_tools.py <persona_dir> [--body captured.json]
        [--from 2026-06-08] [--to 2026-09-12]

Rule (frozen in the #525 change record): keep a brain tool up front (MCP `_meta` anthropic/alwaysLoad) iff the
Wilson 95% LOWER bound of its per-chat-turn use rate >= its break-even rate r*:
    r* = U / H,  U = schema tokens * mean CLI turns per chat call (the schema is re-read every CLI turn),
                 H = median cache_read tokens per CLI turn (one extra CLI turn = one ToolSearch hop).
Unit = chat turns that used the tool (a hop is paid once per CLI invocation, not per call). The tool log and
chat_usage share no key (guarded-change.companion.md Notes), so a call is assigned to the first chat row logged
within 600 s after it — a heuristic, reported as such. Schema size: from a captured /v1/messages request body
(`--body`) when you have one, else from the brain's own tool schemas (close to the wire size; chars / 4 = tokens).

Re-run when the monologue directive changes (#417, #418) or a new tool shows up in the logs, and update
`ALWAYS_LOADED_TOOLS` in brain/mcp_server/tools.py from the printed set. Never point it at a persona you do not own.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _wilson_lo(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def _schema_chars(persona: Path, body: Path | None) -> dict[str, int]:
    if body is not None:
        tools = json.loads(body.read_text(encoding="utf-8"))["tools"]
        return {
            t["name"].removeprefix("mcp__brain-tools__"): len(json.dumps(t, separators=(",", ":")))
            for t in tools
            if t.get("name", "").startswith("mcp__brain-tools__")
        }
    from brain.tools import NELL_TOOL_NAMES
    from brain.tools.schemas import build_schemas

    schemas = build_schemas(persona.name)
    return {
        name: len(json.dumps(
            {"name": f"mcp__brain-tools__{name}", "description": schemas[name].get("description", ""),
             "input_schema": schemas[name].get("parameters", {"type": "object"})},
            separators=(",", ":"),
        ))
        for name in NELL_TOOL_NAMES
        if name in schemas
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("persona", type=Path)
    ap.add_argument("--body", type=Path, help="captured /v1/messages request body (exact wire schema sizes)")
    ap.add_argument("--from", dest="lo", default="2026-06-08", help="first day of the window (inclusive)")
    ap.add_argument("--to", dest="hi", default="2026-09-12", help="last day of the window (exclusive)")
    args = ap.parse_args(argv)
    lo, hi = args.lo, args.hi
    chars = _schema_chars(args.persona, args.body)
    use = _jsonl(args.persona / "chat_usage.jsonl")
    inv = _jsonl(args.persona / "tool_invocations.log.jsonl")
    chats = [r for r in use if r.get("call_type") == "chat" and lo <= r["ts"][:10] < hi]
    calls = [r for r in inv if lo <= r["timestamp"][:10] < hi and r.get("origin") is None]
    n = len(chats)
    if not n:
        print(f"no chat turns in [{lo}, {hi})", file=sys.stderr)
        return 1
    mean_turns = statistics.mean(r["num_turns"] for r in chats if r.get("num_turns"))
    hop = statistics.median(
        r["cache_read_input_tokens"] / r["num_turns"]
        for r in chats
        if r.get("num_turns") and r.get("cache_read_input_tokens")
    )
    ends = sorted(_t(r["ts"]) for r in chats)
    used: dict[str, set[int]] = collections.defaultdict(set)
    unassigned = 0
    for r in calls:
        i = bisect.bisect_left(ends, _t(r["timestamp"]))
        if i < len(ends) and (ends[i] - _t(r["timestamp"])).total_seconds() < 600:
            used[r["name"]].add(i)
        else:
            unassigned += 1
    print(f"window [{lo}, {hi}): chat turns n={n}, brain calls={len(calls)} (unassigned {unassigned}), "
          f"mean CLI turns/chat={mean_turns:.2f}, hop H={hop:.0f} tok, chars/4 = tokens, "
          f"schemas from {'--body' if args.body else 'brain.tools.schemas'}")
    keep = []
    for name, c in sorted(chars.items(), key=lambda kv: -len(used.get(kv[0], ()))):
        k = len(used.get(name, ()))
        rstar = (c / 4) * mean_turns / hop
        lo95 = _wilson_lo(k, n)
        verdict = "KEEP UP FRONT" if lo95 >= rstar else "defer"
        if verdict != "defer":
            keep.append(name)
        print(f"{name:22} turns {k:3} rate {k / n:6.1%} wilson95_lo {lo95:6.1%} r* {rstar:5.1%} -> {verdict}")
    print("ALWAYS_LOAD =", keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
