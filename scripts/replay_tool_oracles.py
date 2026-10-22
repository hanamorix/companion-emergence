#!/usr/bin/env python3
"""Behaviour oracles for the #525 tool-deferral A/B (and its Sonnet re-run, #536).

Reads the metrics JSON that `scripts/cache_replay_workload.py --out` writes (its `per_turn` list: each turn's
chat usage rows and tool rows) for an OLD and a NEW arm, and scores:

  B1  record_monologue kept      default turns, share with an ok MCP call of record_monologue
  B2  deferred tools reached     reach turns, share with a dispatched MCP call of any deferred brain tool,
                                 plus the mechanism check (one NEW turn with a ToolSearch row and such a call)
  B2b memory reach kept          recall turns, share with a dispatched search_memories or read_full_memory
  B3  cost / turns / cache       default turns (per user turn and per chat call), with an A/A precondition
  B8  no scaffolding leak        NEW replies never mention ToolSearch / tool_reference / tool search / deferred tool
  B9  reply length               advisory

Only content turns count (integer ids from 1); the warm-up (0) and the plain micro-acks (p1, p2) never do.
Row classes: `ok` = MCP-child row (no origin) with no error and outcome ok/None; `dispatched` = any MCP-child
row, whatever its error (the CLI dispatched the full mcp__brain-tools__ name; a bridge-routed search with no
bridge still counts); `bare` = a cli_builtin row named like a brain tool (a failed bare-name call);
`toolsearch` = a cli_builtin ToolSearch row.

A share gate FAILs only when NEW < OLD and a one-sided Fisher exact test gives p < 0.05, and counts only when
OLD has at least 5 qualifying turns (below that even NEW = 0 cannot reach p < 0.05): else UNMEASURED.

Every gate is always reported: a prompt set missing from either arm, or no replies to scan, makes that gate
UNMEASURED, never a pass. Replies are read from each metrics file's `<name>.replies.json` sibling (written by
`--dump-replies`): B8 scans every NEW reply, B9 compares the arms on default-set replies.

Exit 0 all gates PASS, 10 any FAIL, 11 any UNMEASURED or STOP_FOR_HUMAN (and none FAIL), 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from math import comb
from pathlib import Path

ALPHA = 0.05
MIN_OLD = 5
MEMORY_TOOLS = frozenset({"search_memories", "read_full_memory"})
_LEAK = re.compile(r"toolsearch|tool_reference|tool search|deferred tool", re.IGNORECASE)


def _nell_tool_names() -> frozenset[str]:
    try:
        from brain.tools import NELL_TOOL_NAMES
    except ImportError:  # run outside the repo env: no bare-name attempts are detected (advisory only)
        return frozenset()
    return frozenset(NELL_TOOL_NAMES)


NELL = _nell_tool_names()


# ---- row classes -------------------------------------------------------------------------


def is_dispatched(row: dict) -> bool:
    return row.get("origin") is None


def is_ok(row: dict) -> bool:
    return is_dispatched(row) and row.get("error") is None and row.get("outcome") in (None, "ok", "deduped")


def is_bare(row: dict) -> bool:
    return row.get("origin") == "cli_builtin" and row.get("name") in NELL


def is_toolsearch(row: dict) -> bool:
    return row.get("origin") == "cli_builtin" and row.get("name") == "ToolSearch"


def turn_has(turn: dict, cls, names) -> bool:
    return any(cls(r) and (names is None or r.get("name") in names) for r in turn["tool_rows"])


def content_turns(run: dict) -> list[dict]:
    return [t for t in run["per_turn"] if isinstance(t["turn"], int) and t["turn"] >= 1]


def count(runs: list[dict], pred) -> tuple[int, int] | None:
    """(turns matching pred, content turns) over runs; None if any run has no per-turn data."""
    if any(r.get("per_turn") is None for r in runs):
        return None
    turns = [t for r in runs for t in content_turns(r)]
    return sum(1 for t in turns if pred(t)), len(turns)


# ---- statistics ----------------------------------------------------------------------------


def fisher_one_sided(old_k: int, old_n: int, new_k: int, new_n: int) -> float:
    """P(NEW successes <= new_k | margins): small when NEW is lower than OLD by more than chance."""
    total_k, total_n = old_k + new_k, old_n + new_n
    denom = comb(total_n, total_k)
    return sum(comb(new_n, x) * comb(old_n, total_k - x) for x in range(0, new_k + 1) if total_k - x <= old_n) / denom


def gate(old_k: int, old_n: int, new_k: int, new_n: int, min_old: int = MIN_OLD) -> dict:
    res = {"old": [old_k, old_n], "new": [new_k, new_n], "p": None}
    if old_k < min_old:
        return {**res, "verdict": "UNMEASURED", "reason": f"OLD has {old_k} < {min_old} qualifying turns"}
    if new_k * old_n >= old_k * new_n:
        return {**res, "verdict": "PASS"}
    p = fisher_one_sided(old_k, old_n, new_k, new_n)
    return {**res, "p": round(p, 4), "verdict": "FAIL" if p < ALPHA else "PASS"}


def _share_gate(old_runs, new_runs, pred) -> dict:
    o, n = count(old_runs, pred), count(new_runs, pred)
    if o is None or n is None or not o[1] or not n[1]:
        return {"verdict": "UNMEASURED", "reason": "a run has no per-turn data, or an arm has no turns"}
    return gate(o[0], o[1], n[0], n[1])


# ---- gates ----------------------------------------------------------------------------------


def b1(old_runs, new_runs) -> dict:
    return _share_gate(old_runs, new_runs, lambda t: turn_has(t, is_ok, {"record_monologue"}))


def b2(old_runs, new_runs, always_load) -> dict:
    deferred = lambda r: is_dispatched(r) and r.get("name") not in always_load  # noqa: E731
    res = _share_gate(old_runs, new_runs, lambda t: turn_has(t, deferred, None))
    mech = count(new_runs, lambda t: turn_has(t, is_toolsearch, None) and turn_has(t, deferred, None))
    res["mechanism"] = bool(mech and mech[0] >= 1)
    if res["verdict"] == "PASS" and not res["mechanism"]:
        res["verdict"] = "FAIL"
        res["reason"] = "no NEW reach turn shows a ToolSearch row and a deferred-tool call in the same turn"
    return res


def b2b(old_runs, new_runs) -> dict:
    res = _share_gate(old_runs, new_runs, lambda t: turn_has(t, is_dispatched, MEMORY_TOOLS))
    for arm, runs in (("old", old_runs), ("new", new_runs)):
        split = {}
        for name in sorted(MEMORY_TOOLS):
            c = count(runs, lambda t, n=name: turn_has(t, is_dispatched, {n}))
            split[name] = c[0] if c else None
        res[f"{arm}_split"] = split
    return res


def _mean(xs):
    return statistics.fmean(xs) if xs else None


def _b3_values(run: dict) -> dict:
    turns = content_turns(run)
    calls = [c for t in turns for c in t["chat_rows"]]
    return {
        "cost_per_user_turn": [sum(c["cost_usd"] for c in t["chat_rows"]) for t in turns],
        "cost_per_chat_call": [c["cost_usd"] for c in calls],
        "num_turns_per_user_turn": [sum(c["num_turns"] for c in t["chat_rows"]) for t in turns],
        "num_turns_per_chat_call": [c["num_turns"] for c in calls],
        "cache_creation_per_chat_call": [c["cache_creation"] for c in calls],
    }


_PCT_METRICS = ("cost_per_user_turn", "cost_per_chat_call", "cache_creation_per_chat_call")
_ABS_METRICS = ("num_turns_per_user_turn", "num_turns_per_chat_call")


def b3(old_runs, new_runs) -> dict:
    if any(r.get("per_turn") is None for r in old_runs + new_runs):
        return {"verdict": "UNMEASURED", "reason": "a run has no per-turn data", "stop_for_human": False}
    try:
        old_v = [_b3_values(r) for r in old_runs]
        new_v = [_b3_values(r) for r in new_runs]
    except (KeyError, TypeError) as exc:
        return {"verdict": "UNMEASURED", "reason": f"a chat row lacks a number: {exc!r}", "stop_for_human": False}
    metrics = {}
    for m in _PCT_METRICS + _ABS_METRICS:
        o = _mean([x for v in old_v for x in v[m]])
        n = _mean([x for v in new_v for x in v[m]])
        entry = {"old": o, "new": n}
        if o is None or n is None:
            entry["verdict"] = "UNMEASURED"
        elif m in _PCT_METRICS:
            per_replay = [_mean(v[m]) for v in old_v if v[m]]
            spread_ok = all(abs(x - o) <= 0.10 * o for x in per_replay) if o else False
            if not spread_ok:
                entry["verdict"] = "UNMEASURED"
                entry["reason"] = "OLD replay means differ by more than ±10%: the bar cannot separate noise"
            else:
                entry["verdict"] = "PASS" if n <= o * 1.10 else "FAIL"
        else:
            entry["verdict"] = "PASS" if n <= o + 1 else "FAIL"
        metrics[m] = entry
    breaks = [(r.get("cache_break_check") or {}).get("status") for r in new_runs]
    if "FAIL" in breaks:
        ratio = "FAIL"
    else:  # a check the harness could not judge (UNMEASURED, NOT_APPLICABLE, absent) is not a pass
        ratio = "PASS" if all(b == "PASS" for b in breaks) else "UNMEASURED"
    metrics["cache_read_ratio"] = {"new": breaks, "verdict": ratio}
    verdicts = {m: e["verdict"] for m, e in metrics.items()}
    failed = {m for m, v in verdicts.items() if v == "FAIL"}
    verdict = "FAIL" if failed else ("UNMEASURED" if "UNMEASURED" in verdicts.values() else "PASS")
    # A creation-only failure with both cost bars passing is biased against NEW (its tools-block saving lands
    # in the excluded warm-up row; the deferred listing is re-written every call): the human decides.
    stop = failed == {"cache_creation_per_chat_call"} and all(
        verdicts[c] == "PASS" for c in ("cost_per_user_turn", "cost_per_chat_call")
    )
    if stop:
        verdict = "STOP_FOR_HUMAN"
    tails = {
        "num_turns_max": {
            "old": max((x for v in old_v for x in v["num_turns_per_user_turn"]), default=None),
            "new": max((x for v in new_v for x in v["num_turns_per_user_turn"]), default=None),
        }
    }
    return {"verdict": verdict, "metrics": metrics, "stop_for_human": stop, "advisory_tail": tails}


def b8(replies: list[dict]) -> dict:
    if not replies:
        return {"verdict": "UNMEASURED", "reason": "no NEW replies to scan (run with --dump-replies)"}
    leaks = [r["turn"] for r in replies if _LEAK.search(r.get("reply") or "")]
    return {"verdict": "FAIL" if leaks else "PASS", "leaking_turns": leaks}


def b9(replies: list[dict]) -> dict:
    lens = [len(r.get("reply") or "") for r in replies if isinstance(r["turn"], int) and r["turn"] >= 1]
    return {"n": len(lens), "mean_chars": _mean(lens), "median_chars": statistics.median(lens) if lens else None}


def advisory_counts(runs, always_load) -> dict:
    """Per-tool dispatched turns, bare attempts and ToolSearch turns, per prompt set (reported, never gated)."""
    out = {}
    for r in runs:
        if r.get("per_turn") is None:
            continue
        s = out.setdefault(r.get("prompt_set", "?"), {"turns": 0, "dispatched": {}, "bare": {}, "toolsearch_turns": 0})
        for t in content_turns(r):
            s["turns"] += 1
            s["toolsearch_turns"] += turn_has(t, is_toolsearch, None)
            for row in t["tool_rows"]:
                key = "dispatched" if is_dispatched(row) else ("bare" if is_bare(row) else None)
                if key:
                    s[key][row["name"]] = s[key].get(row["name"], 0) + 1
    return out


# ---- CLI ------------------------------------------------------------------------------------


def _load(paths) -> list[dict]:
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths or []]


def _load_replies(paths) -> list[dict]:
    """The `<metrics>.replies.json` siblings, each reply tagged with its file's prompt set."""
    out = []
    for p in paths or []:
        sib = Path(p).with_suffix(".replies.json")
        if not sib.exists():
            continue
        prompt_set = json.loads(Path(p).read_text(encoding="utf-8")).get("prompt_set", "default")
        out += [{**row, "prompt_set": prompt_set} for row in json.loads(sib.read_text(encoding="utf-8"))]
    return out


def _by_set(runs, name):
    return [r for r in runs if r.get("prompt_set", "default") == name]


def _missing(name) -> dict:
    return {"verdict": "UNMEASURED", "reason": f"no {name} runs for one arm or both"}


def evaluate(old_runs, new_runs, always_load, old_replies=(), new_replies=()) -> dict:
    report = {"always_load": sorted(always_load), "gates": {}, "advisory": {}}
    old_d, new_d = _by_set(old_runs, "default"), _by_set(new_runs, "default")
    have = lambda s: bool(_by_set(old_runs, s) and _by_set(new_runs, s))  # noqa: E731
    g = report["gates"]
    g["B1"] = b1(old_d, new_d) if have("default") else _missing("default")
    g["B2"] = b2(_by_set(old_runs, "reach"), _by_set(new_runs, "reach"), always_load) if have("reach") else _missing("reach")
    g["B2b"] = b2b(_by_set(old_runs, "recall"), _by_set(new_runs, "recall")) if have("recall") else _missing("recall")
    g["B3"] = b3(old_d, new_d) if have("default") else _missing("default")
    g["B8"] = b8(list(new_replies))
    report["advisory"]["B9"] = {
        arm: b9([r for r in reps if r.get("prompt_set", "default") == "default"])
        for arm, reps in (("old", old_replies), ("new", new_replies))
    }
    report["advisory"]["counts_old"] = advisory_counts(old_runs, always_load)
    report["advisory"]["counts_new"] = advisory_counts(new_runs, always_load)
    return report


def exit_code(report: dict) -> int:
    verdicts = [g["verdict"] for g in report["gates"].values()]
    if "FAIL" in verdicts:
        return 10
    if "UNMEASURED" in verdicts or "STOP_FOR_HUMAN" in verdicts or not verdicts:
        return 11
    return 0


def self_test() -> int:
    """Seeded cases (the same ones the pytest file runs): every gate shown able to fail and to pass."""
    def mk(rows_per_turn, ps):
        return {"prompt_set": ps, "per_turn_error": None, "cache_break_check": {"status": "PASS"},
                "per_turn": [{"turn": k, "prompt": "", "chat_rows": [], "tool_rows": rows}
                             for k, rows in enumerate(rows_per_turn, 1)]}
    rm = {"name": "record_monologue", "origin": None, "outcome": "ok", "error": None}
    rf = {"name": "read_full_memory", "origin": None, "outcome": "ok", "error": None}
    checks = [
        round(fisher_one_sided(4, 24, 0, 24), 4) == 0.0546,
        b1([mk([[rm]] * 18 + [[]] * 6, "default")], [mk([[rm]] * 6 + [[]] * 18, "default")])["verdict"] == "FAIL",
        b1([mk([[rm]] * 12 + [[]] * 12, "default")], [mk([[rm]] * 12 + [[]] * 12, "default")])["verdict"] == "PASS",
        b2b([mk([[rf]] * 4 + [[]] * 14, "recall")], [mk([[]] * 18, "recall")])["verdict"] == "UNMEASURED",
        b8([{"turn": 1, "reply": "via ToolSearch"}])["verdict"] == "FAIL",
    ]
    print(f"self-test: {sum(checks)}/{len(checks)} passed", file=sys.stderr)
    return 0 if all(checks) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old", nargs="+", type=Path, help="OLD arm metrics JSON files (any prompt sets)")
    p.add_argument("--new", nargs="+", type=Path, help="NEW arm metrics JSON files")
    p.add_argument(
        "--always-load", nargs="+", help="NEW's always-loaded brain tools, never deferred (required: without it B2 "
        "would count them as deferred)"
    )
    p.add_argument("--out", type=Path, help="write the report JSON here")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.old or not args.new or not args.always_load:
        p.error("--old, --new and --always-load are required")
    report = evaluate(
        _load(args.old), _load(args.new), set(args.always_load), _load_replies(args.old), _load_replies(args.new)
    )
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
