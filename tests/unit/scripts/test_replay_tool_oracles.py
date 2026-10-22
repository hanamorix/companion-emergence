"""Tests for scripts/replay_tool_oracles.py — the #525 behaviour oracles (B1, B2, B2b, B3, B8, B9).

Each gate is shown able to fail on a seeded result built from real row shapes, and shown to pass on
identical arms. Row shapes are those `cache_replay_workload._per_turn` writes from the real logs: an
MCP-child row has no `origin`; a CLI built-in row has `origin: "cli_builtin"`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("replay_tool_oracles_under_test", REPO / "scripts" / "replay_tool_oracles.py")
rto = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rto
_spec.loader.exec_module(rto)

ALWAYS = ("record_monologue",)


def mcp(name, outcome="ok", error=None):
    return {"name": name, "origin": None, "outcome": outcome, "error": error}


def builtin(name):
    return {"name": name, "origin": "cli_builtin", "outcome": "ok", "error": None}


def chat(cost=0.1, turns=2, creation=1000, read=20000):
    return {"cost_usd": cost, "num_turns": turns, "cache_creation": creation, "cache_read": read, "duration_ms": 9000}


def run(prompt_set, turns, *, cache="PASS"):
    """A metrics JSON: warm-up turn 0 (never scored), then content turns 1..n with the given tool rows."""
    per_turn = [{"turn": 0, "prompt": "ok", "chat_rows": [chat(creation=90000)], "tool_rows": [mcp("record_monologue")]}]
    for k, rows in enumerate(turns, 1):
        per_turn.append({"turn": k, "prompt": f"p{k}", "chat_rows": [chat()], "tool_rows": rows})
    return {"prompt_set": prompt_set, "per_turn": per_turn, "per_turn_error": None, "cache_break_check": {"status": cache}}


def hits(n, total, row):
    return [[row] for _ in range(n)] + [[] for _ in range(total - n)]


# ---- Fisher + minimum count -------------------------------------------------------------


def test_fisher_matches_exact_values():
    assert rto.fisher_one_sided(4, 24, 0, 24) == pytest.approx(0.0546, abs=1e-4)
    assert rto.fisher_one_sided(5, 24, 0, 24) == pytest.approx(0.0248, abs=1e-4)
    assert rto.fisher_one_sided(5, 18, 0, 18) == pytest.approx(0.0227, abs=1e-4)


def test_gate_fails_on_a_real_drop_passes_identical_and_is_unmeasured_below_the_minimum():
    assert rto.gate(18, 24, 6, 24)["verdict"] == "FAIL"
    assert rto.gate(12, 24, 12, 24)["verdict"] == "PASS"
    assert rto.gate(12, 24, 20, 24)["verdict"] == "PASS"  # NEW higher never fails
    assert rto.gate(4, 24, 0, 24)["verdict"] == "UNMEASURED"


# ---- attribution classes ----------------------------------------------------------------


def test_classes_separate_ok_dispatched_bare_and_toolsearch():
    assert rto.is_ok(mcp("record_monologue"))
    assert not rto.is_ok(mcp("search_memories", outcome="refused", error="bridge unreachable"))
    assert rto.is_dispatched(mcp("search_memories", outcome="refused", error="bridge unreachable"))
    assert not rto.is_dispatched(builtin("search_memories"))
    assert rto.is_bare(builtin("search_memories")) and not rto.is_bare(builtin("WebSearch"))
    assert rto.is_toolsearch(builtin("ToolSearch")) and not rto.is_toolsearch(mcp("ToolSearch"))


# ---- B1, B2, B2b ------------------------------------------------------------------------


def test_b1_fails_on_a_seeded_drop_and_passes_identical_arms():
    old = [run("default", hits(18, 24, mcp("record_monologue")))]
    assert rto.b1(old, [run("default", hits(6, 24, mcp("record_monologue")))])["verdict"] == "FAIL"
    assert rto.b1(old, old)["verdict"] == "PASS"


def test_b1_counts_only_ok_calls():
    old = [run("default", hits(18, 24, mcp("record_monologue")))]
    bare_only = [run("default", hits(18, 24, builtin("record_monologue")))]
    assert rto.b1(old, bare_only)["verdict"] == "FAIL"


def test_b2_counts_dispatched_deferred_tools_and_needs_the_mechanism():
    reach = mcp("list_directory", outcome="refused", error="path does not exist: /x")
    old = [run("reach", hits(14, 21, reach))]
    new_hop = [run("reach", [[builtin("ToolSearch"), reach]] * 14 + [[]] * 7)]
    assert rto.b2(old, new_hop, ALWAYS)["verdict"] == "PASS"
    no_hop = rto.b2(old, [run("reach", hits(14, 21, reach))], ALWAYS)
    assert no_hop["verdict"] == "FAIL" and no_hop["mechanism"] is False
    assert rto.b2(old, [run("reach", hits(0, 21, reach))], ALWAYS)["verdict"] == "FAIL"


def test_b2_ignores_always_loaded_tools():
    old = [run("reach", hits(14, 21, mcp("record_monologue")))]
    assert rto.b2(old, old, ALWAYS)["verdict"] == "UNMEASURED"  # OLD has 0 deferred-tool turns


def test_b2b_memory_reach_counts_either_tool_and_bridge_refusals():
    sm = mcp("search_memories", outcome="refused", error="bridge unreachable")
    rf = mcp("read_full_memory")
    old = [run("recall", hits(10, 18, rf))]
    assert rto.b2b(old, [run("recall", hits(2, 18, sm))])["verdict"] == "FAIL"
    assert rto.b2b(old, [run("recall", hits(10, 18, sm))])["verdict"] == "PASS"
    assert rto.b2b([run("recall", hits(4, 18, rf))], [run("recall", hits(0, 18, rf))])["verdict"] == "UNMEASURED"
    assert rto.b2b(old, [run("recall", hits(10, 18, builtin("read_full_memory")))])["verdict"] == "FAIL"


def test_warmup_and_plain_turns_are_never_scored():
    r = run("recall", hits(0, 18, mcp("read_full_memory")))
    r["per_turn"][0]["tool_rows"] = [mcp("read_full_memory")]
    r["per_turn"].append({"turn": "p2", "prompt": "thanks", "chat_rows": [chat()], "tool_rows": [mcp("read_full_memory")]})
    assert rto.count([r], lambda t: rto.turn_has(t, rto.is_dispatched, {"read_full_memory"})) == (0, 18)


def test_a_run_without_per_turn_data_is_unmeasured():
    bad = run("recall", hits(10, 18, mcp("read_full_memory")))
    bad["per_turn"], bad["per_turn_error"] = None, "the log shrank"
    assert rto.b2b([bad], [bad])["verdict"] == "UNMEASURED"


# ---- B3 ---------------------------------------------------------------------------------


def _cost_run(cost, creation=1000, cache="PASS", n=8):
    r = run("default", [[]] * n, cache=cache)
    for t in r["per_turn"][1:]:
        t["chat_rows"] = [chat(cost=cost, creation=creation)]
    return r


def test_b3_passes_identical_and_fails_a_cost_rise():
    old = [_cost_run(0.10) for _ in range(3)]
    assert rto.b3(old, [_cost_run(0.10) for _ in range(3)])["verdict"] == "PASS"
    assert rto.b3(old, [_cost_run(0.12) for _ in range(3)])["verdict"] == "FAIL"


def test_b3_creation_only_failure_asks_for_the_human():
    old = [_cost_run(0.10) for _ in range(3)]
    res = rto.b3(old, [_cost_run(0.10, creation=1500) for _ in range(3)])
    # Not a FAIL (that routes back to stage 2): criteria Rev 4 sends a creation-only failure to the human.
    assert res["verdict"] == "STOP_FOR_HUMAN" and res["stop_for_human"] is True
    assert rto.exit_code({"gates": {"B3": res}}) == 11


def test_b3_a_noisy_old_arm_is_unmeasured():
    old = [_cost_run(0.07), _cost_run(0.10), _cost_run(0.13)]
    assert rto.b3(old, [_cost_run(0.10) for _ in range(3)])["metrics"]["cost_per_chat_call"]["verdict"] == "UNMEASURED"


def test_b3_a_cache_break_in_new_fails():
    old = [_cost_run(0.10) for _ in range(3)]
    assert rto.b3(old, [_cost_run(0.10), _cost_run(0.10, cache="FAIL"), _cost_run(0.10)])["verdict"] == "FAIL"


@pytest.mark.parametrize("status", ["UNMEASURED", "NOT_APPLICABLE", None])
def test_b3_an_unjudged_cache_check_is_unmeasured_not_a_fail(status):
    old = [_cost_run(0.10) for _ in range(3)]
    res = rto.b3(old, [_cost_run(0.10), _cost_run(0.10, cache=status), _cost_run(0.10)])
    assert res["metrics"]["cache_read_ratio"]["verdict"] == "UNMEASURED"
    assert res["verdict"] == "UNMEASURED"


# ---- B8, B9 -----------------------------------------------------------------------------


def test_b8_with_no_replies_is_unmeasured():
    # Scanning nothing is not a pass (gate 7 L-1b).
    assert rto.b8([])["verdict"] == "UNMEASURED"


def test_b8_flags_scaffolding_in_a_reply():
    assert rto.b8([{"turn": 1, "reply": "I loaded it with ToolSearch first."}])["verdict"] == "FAIL"
    assert rto.b8([{"turn": 1, "reply": "A deferred Tool and a tool search."}])["verdict"] == "FAIL"
    assert rto.b8([{"turn": 1, "reply": "I searched my memories for you."}])["verdict"] == "PASS"


def test_b9_reports_reply_lengths():
    rep = rto.b9([{"turn": 1, "reply": "abcd"}, {"turn": 2, "reply": "ab"}, {"turn": 0, "reply": "x" * 99}])
    assert rep == {"n": 2, "mean_chars": 3.0, "median_chars": 3.0}


# ---- CLI ---------------------------------------------------------------------------------


def _write(tmp_path, name, metrics, replies=None):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    if replies is not None:
        path.with_suffix(".replies.json").write_text(json.dumps(replies), encoding="utf-8")
    return str(path)


def _arm(tmp_path, arm, rm_hits, reply="fine"):
    """One file per prompt set, each with its sibling replies (as --dump-replies writes them)."""
    rm, rf, ls, ts = mcp("record_monologue"), mcp("read_full_memory"), mcp("list_directory"), builtin("ToolSearch")
    replies = [{"turn": k, "prompt": "p", "reply": reply} for k in range(0, 25)]
    reach_turns = [[ls, ts] if arm == "new" else [ls] for _ in range(12)] + [[] for _ in range(9)]
    return [
        _write(tmp_path, f"{arm}-default", run("default", hits(rm_hits, 24, rm)), replies),
        _write(tmp_path, f"{arm}-reach", run("reach", reach_turns), replies[:22]),
        _write(tmp_path, f"{arm}-recall", run("recall", hits(10, 18, rf)), replies[:19]),
    ]


def test_cli_scores_every_gate_and_exits_by_the_worst(tmp_path):
    old = _arm(tmp_path, "old", 18)
    new = _arm(tmp_path, "new", 18)
    out = tmp_path / "report.json"
    assert rto.main(["--old", *old, "--new", *new, "--always-load", "record_monologue", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report["gates"]) == {"B1", "B2", "B2b", "B3", "B8"}
    assert report["always_load"] == ["record_monologue"]
    assert set(report["advisory"]["B9"]) == {"old", "new"}  # per arm, default replies (gate 7 M-2)
    drop = _arm(tmp_path, "drop", 6)
    assert rto.main(["--old", *old, "--new", *drop, "--always-load", "record_monologue"]) == 10


def test_cli_a_missing_prompt_set_is_unmeasured_not_a_pass(tmp_path):
    old = _arm(tmp_path, "old", 18)
    report = rto.evaluate([json.loads(Path(old[0]).read_text())], [json.loads(Path(old[0]).read_text())], {"record_monologue"}, [], [])
    assert {g: report["gates"][g]["verdict"] for g in ("B2", "B2b", "B8")} == dict.fromkeys(("B2", "B2b", "B8"), "UNMEASURED")
    assert rto.main(["--old", old[0], "--new", old[0], "--always-load", "record_monologue"]) == 11


def test_cli_scans_the_replies_of_every_new_file_for_leaks(tmp_path):
    old = _arm(tmp_path, "old", 18)
    new = _arm(tmp_path, "new", 18)
    Path(new[2]).with_suffix(".replies.json").write_text(json.dumps([{"turn": 3, "reply": "via ToolSearch"}]), encoding="utf-8")
    assert rto.main(["--old", *old, "--new", *new, "--always-load", "record_monologue"]) == 10


def test_cli_requires_the_always_load_set(tmp_path):
    # Without it record_monologue would count as "deferred" and B2 would measure a proxy (gate 7 L-1c).
    old = _arm(tmp_path, "old", 18)
    with pytest.raises(SystemExit) as exc:
        rto.main(["--old", *old, "--new", *old])
    assert exc.value.code == 2


def test_self_test_passes():
    assert rto.main(["--self-test"]) == 0
