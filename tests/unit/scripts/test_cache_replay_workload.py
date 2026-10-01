"""scripts/cache_replay_workload.py - the per-row cache-break check (#339).

`check_cache_break` is a pure function over `chat_usage.jsonl` rows. Within ONE run (one slice of
chat rows) the FIRST chat row writes the fixed part of the prompt fresh: its cache_creation is F.
A later row whose cache_creation reaches F within the config threshold is a cache break. No call
counts, no model calls, no OLD-vs-NEW comparison: a shrunken prompt cannot trip it because each
run is measured against its own first row.

Row shapes here are SYNTHETIC (owner ruling: a mechanical check, tested without model calls),
plus a few REAL-shaped rows copied from recorded CLI output (named below). Real-row validity of
the first-row rules is unverified by that ruling.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = Path(os.environ.get("CRW_SCRIPT", REPO / "scripts" / "cache_replay_workload.py"))
CONFIG = REPO / "guarded-change.companion.md"

_spec = importlib.util.spec_from_file_location("cache_replay_workload_under_test", SCRIPT)
crw = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = crw
_spec.loader.exec_module(crw)

T = 10.0  # the config's threshold today; tests that read the real file assert it separately


def row(creation, read, *, turns=1, model="m", t=0, inp=10, call_type="chat"):
    return {
        "ts": f"2026-10-01T12:00:{t:02d}+00:00",
        "call_type": call_type,
        "model": model,
        "input_tokens": inp,
        "output_tokens": 50,
        "cache_creation_input_tokens": creation,
        "cache_read_input_tokens": read,
        "total_cost_usd": 0.01,
        "num_turns": turns,
        "duration_ms": 1000,
        "session_id": "s",
    }


def healthy(f=16984, n_later=5, later_creation=1200):
    rows = [row(f, 0, t=0)]
    rows += [row(later_creation, f + 500 * k, t=k) for k in range(1, n_later + 1)]
    return rows


def check(rows, thr=T):
    return crw.check_cache_break(rows, threshold_pct=thr)


# --------------------------------------------------------------------------------------
# C9 characterization: written and run against the PRE-change script first
# --------------------------------------------------------------------------------------

_OLD_SUMMARY_KEYS = {
    "turns_requested",
    "chat_rows_observed",
    "c1_system_byte_stability",
    "c8_cache",
    "c9_history_caching",
}


def test_characterization_summarise_existing_keys_unchanged():
    usage = [row(1000, 0), row(200, 1000), row(300, 1500)]
    dbg = [{"call_type": "chat", "system_sha256": "abc"}, {"call_type": "chat", "system_sha256": "abc"}]
    s = crw._summarise(usage, dbg, turns=3)
    assert _OLD_SUMMARY_KEYS <= set(s)
    assert s["turns_requested"] == 3
    assert s["chat_rows_observed"] == 3
    assert s["c1_system_byte_stability"] == {
        "available": True,
        "distinct_system_sha256": 1,
        "byte_stable": True,
        "sample": ["abc"],
    }
    assert s["c8_cache"] == {
        "mean_cache_creation": 500.0,
        "mean_cache_read": 833.3,
        "cache_creation_series": [1000.0, 200.0, 300.0],
        "cache_read_series": [0.0, 1000.0, 1500.0],
    }
    assert s["c9_history_caching"]["last_turn_cache_creation"] == 300.0


def _write_summary(path, creation, read):
    path.write_text(
        json.dumps(
            {
                "turns_requested": len(creation),
                "chat_rows_observed": len(creation),
                "c1_system_byte_stability": {"available": False, "note": "n"},
                "c8_cache": {
                    "mean_cache_creation": sum(creation) / len(creation),
                    "mean_cache_read": sum(read) / len(read),
                    "cache_creation_series": creation,
                    "cache_read_series": read,
                },
                "c9_history_caching": {"note": "n", "last_turn_cache_creation": creation[-1]},
            }
        )
    )


def test_characterization_compare_output_and_exit(tmp_path, capsys):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    _write_summary(old, [1000.0, 1000.0], [2000.0, 2000.0])
    _write_summary(new, [500.0, 500.0], [2500.0, 2500.0])
    rc = crw.compare(old, new)
    out = capsys.readouterr().out
    assert rc == 0
    assert "mean cache_creation/turn: 1000 → 500  (-50%)" in out
    assert "mean cache_read/turn:     2000 → 2500  (+25%)" in out
    assert "C8 (system-block cache stops re-creating): PASS" in out
    assert "C1: n/a" in out


def test_characterization_compare_fails_when_creation_does_not_drop(tmp_path, capsys):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    _write_summary(old, [1000.0], [2000.0])
    _write_summary(new, [1000.0], [2000.0])
    assert crw.compare(old, new) == 1
    assert "C8 (system-block cache stops re-creating): FAIL" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# C1 break rule
# --------------------------------------------------------------------------------------


def _with_later(creation, f=1000):
    # first row: F=f cold write; one later row with the given creation
    return [row(f, 0, t=0), row(creation, f, t=1)]


def test_boundary_exactly_at_threshold_breaks_one_below_passes():
    assert check(_with_later(900))["status"] == "FAIL"  # 900*100 >= 1000*90
    assert check(_with_later(899))["status"] == "PASS"


def test_break_reports_row_creation_share_ts_and_gap():
    rows = [row(1000, 0, t=0), row(100, 1000, t=5), row(950, 100, t=59)]
    res = check(rows)
    assert res["status"] == "FAIL"
    assert res["fixed_prompt_tokens"] == 1000
    assert res["breaks"] == [
        {"row": 3, "creation": 950, "share_of_F": 0.95, "ts": "2026-10-01T12:00:59+00:00", "gap_s": 54.0}
    ]


def test_gap_absent_not_an_error_when_ts_unparsable():
    rows = [row(1000, 0), row(100, 1000), row(990, 100)]
    rows[2]["ts"] = "not a time"
    res = check(rows)
    assert res["status"] == "FAIL"
    assert res["breaks"][0]["gap_s"] is None


def test_check_is_per_row_not_cumulative():
    # five healthy later rows of ~300 each: sum 1500 > 900 but no single row reaches 900
    rows = [row(1000, 0)] + [row(300, 1000 + 100 * k) for k in range(5)]
    assert check(rows)["status"] == "PASS"


def test_compares_to_first_row_not_previous_row():
    # later rows equal to EACH OTHER at 500 (>= 0.9 * previous would break); against F=1000 they are fine
    rows = [row(1000, 0), row(500, 1000), row(500, 1000), row(500, 1000)]
    assert check(rows)["status"] == "PASS"


# --------------------------------------------------------------------------------------
# C2 healthy passes / C3 break fails
# --------------------------------------------------------------------------------------


def test_healthy_run_passes_with_real_shaped_cold_first_row():
    # first row copied from the recorded spike: text path, "ok": input 10, creation 16984, read 0, num_turns 1
    res = check(healthy())
    assert res["status"] == "PASS"
    assert res["fixed_prompt_tokens"] == 16984
    assert res["breaks"] == []
    assert res["max_share_of_F"] == round(1200 / 16984, 3)
    assert res["rows_checked"] == 5


@pytest.mark.parametrize("break_at", [2, 4, 6])
def test_break_row_fails_wherever_it_is(break_at):
    rows = healthy()
    rows[break_at - 1] = row(16900, 90, t=break_at)  # re-wrote ~F, reads collapsed
    res = check(rows)
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == [break_at]


def test_several_breaks_and_break_larger_than_f():
    rows = healthy()
    rows[1] = row(20000, 0)
    rows[3] = row(17000, 0)
    res = check(rows)
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == [2, 4]
    assert res["breaks"][0]["share_of_F"] > 1


def test_fail_wins_over_a_malformed_row():
    rows = healthy()
    rows[1]["cache_creation_input_tokens"] = None
    rows[3] = row(16900, 90)
    assert check(rows)["status"] == "FAIL"


# --------------------------------------------------------------------------------------
# C4 #332-shaped shrink passes (CHOSEN parameters; the recorded #333 numbers cannot be reproduced)
# --------------------------------------------------------------------------------------


def _run(f):
    # 5 later rows, creation 3000 each, reads F + 500*k
    return [row(f, 0, t=0)] + [row(3000, f + 500 * k, t=k) for k in range(1, 6)]


def _legacy_ratio(rows):
    return sum(r["cache_read_input_tokens"] for r in rows) / sum(r["cache_creation_input_tokens"] for r in rows)


def test_shrunken_prompt_passes_and_the_old_ratio_would_have_misfired():
    old_run, new_run = _run(26000), _run(26000 - 9300)
    # precondition (oracle shown able to fail): the OLD sum(read)/sum(creation) check drops > 10% on this pair
    drop = (_legacy_ratio(new_run) - _legacy_ratio(old_run)) / _legacy_ratio(old_run)
    assert drop < -0.10
    # the per-run check passes BOTH runs: each is measured against its own first row
    assert check(old_run)["status"] == "PASS"
    assert check(new_run)["status"] == "PASS"


# --------------------------------------------------------------------------------------
# C5 unmeasured is never a pass
# --------------------------------------------------------------------------------------


def _unmeasured(rows, thr=T, contains=None):
    res = check(rows, thr)
    assert res["status"] == "UNMEASURED", res
    if contains:
        assert contains in res["reason"], res["reason"]
    return res


def test_needs_two_chat_rows():
    _unmeasured([], contains="V1")
    _unmeasured([row(1000, 0)], contains="V1")
    _unmeasured([row(1000, 0), row(5, 5, call_type="generate")], contains="V1")


@pytest.mark.parametrize("bad", [0, None, "1000", True, 1000.0, -5])
def test_first_row_creation_must_be_a_positive_int(bad):
    rows = healthy()
    rows[0]["cache_creation_input_tokens"] = bad
    _unmeasured(rows, contains="V2")


@pytest.mark.parametrize("field", ["input_tokens", "cache_read_input_tokens"])
def test_first_row_other_token_fields_must_be_ints(field):
    rows = healthy()
    rows[0][field] = None
    _unmeasured(rows, contains="V2")


@pytest.mark.parametrize("turns", [2, 4, None, True, "1", 0])
def test_first_row_must_be_single_call(turns):
    rows = healthy()
    rows[0]["num_turns"] = turns
    _unmeasured(rows, contains="V3")
    rows[0].pop("num_turns")
    _unmeasured(rows, contains="V3")


def test_first_row_v4_boundary_read_vs_threshold():
    # T=10: read*100 >= creation*10 is not a cold write. creation 1000: read 100 -> UNMEASURED, 99 -> measurable
    ok = [row(1000, 99), row(100, 1000)]
    assert check(ok)["status"] == "PASS"
    _unmeasured([row(1000, 100), row(100, 1000)], contains="V4")


def test_real_shaped_warm_first_row_is_unmeasured():
    # recorded spike repeat call: creation 0, read 16984 (V2 fails on creation 0)
    _unmeasured([row(0, 16984), row(1200, 17000)], contains="V2")
    # a warm row that still wrote something: V4
    _unmeasured([row(1200, 16984), row(1200, 17000)], contains="V4")


def test_real_recorded_tool_using_result_row_as_first_row_is_unmeasured():
    # result frame of tests/bridge/fixtures/cli_2_1_284_web_tools.ndjson: input 27, creation 38235,
    # read 72930, num_turns 4 (three API calls summed)
    first = row(38235, 72930, turns=4, inp=27)
    _unmeasured([first, row(900, 40000)], contains="V3")


def test_malformed_later_row_without_a_break_is_unmeasured():
    rows = healthy()
    rows[2]["cache_creation_input_tokens"] = "x"
    res = _unmeasured(rows, contains="row 3")
    assert res["fixed_prompt_tokens"] == 16984


def test_model_mismatch_rule():
    rows = healthy()
    rows[2]["model"] = "other"
    _unmeasured(rows, contains="model")
    # either side missing a model => comparable
    rows[2].pop("model")
    assert check(rows)["status"] == "PASS"
    rows = healthy()
    rows[0].pop("model")
    rows[2]["model"] = "other"
    assert check(rows)["status"] == "PASS"
    # a break on a different-model row is not counted (not comparable); another real break still fails
    rows = healthy()
    rows[1] = row(16900, 0, model="other")
    _unmeasured(rows, contains="model")
    rows[3] = row(16900, 0)
    assert check(rows)["status"] == "FAIL"


def test_non_chat_rows_are_ignored_even_when_first():
    gen = row(999999, 0, call_type="generate")
    assert check([gen] + healthy())["status"] == "PASS"
    broken = healthy()
    broken[2] = row(16900, 0)
    res = check([gen] + broken)
    assert res["status"] == "FAIL" and res["fixed_prompt_tokens"] == 16984


def test_first_valid_row_selection_is_rejected():
    # a bad first row (warm, num_turns 1, small creation) must not be rescued by a later row as reference
    rows = [row(500, 17000), row(16984, 0), row(1200, 17000)]
    _unmeasured(rows, contains="V4")


@pytest.mark.parametrize("thr", [0, -1, 100, 150, None, "10", True])
def test_threshold_must_be_a_percentage(thr):
    _unmeasured(healthy(), thr=thr, contains="threshold")


# --------------------------------------------------------------------------------------
# C11 declared blind spots are pinned (PASS by design; see the module docstring / config note)
# --------------------------------------------------------------------------------------


def test_declared_blind_spot_system_only_break_behind_intact_tools_block():
    # CHOSEN numbers: 9.3k tools block of F=26000; a system-only break re-writes 16.7k = 64% of F
    rows = [row(26000, 0)] + [row(1000, 26000)] * 2 + [row(16700, 9300)] + [row(1000, 26000)]
    assert check(rows)["status"] == "PASS"


def test_declared_blind_spot_history_only_break_in_a_short_run():
    rows = [row(26000, 0), row(1000, 26000), row(2500, 26000), row(1000, 28000)]
    assert check(rows)["status"] == "PASS"


def test_declared_blind_spot_large_first_message_inflates_f():
    # F = 17000 fixed + 15000 first message = 32000; a full fixed-part re-write (17000 + 800) stays under 28800
    rows = [row(32000, 0), row(1000, 32000), row(17800, 15000)]
    assert check(rows)["status"] == "PASS"


# --------------------------------------------------------------------------------------
# C6 threshold has no constant
# --------------------------------------------------------------------------------------


def _config(tmp_path, threshold, gating="true"):
    p = tmp_path / "cfg.md"
    p.write_text(
        "metrics:\n"
        "  - name: cache_creation_per_chat_call\n"
        '    regression_threshold: "+10%"\n'
        "  - name: cache_read_ratio\n"
        "    source: x\n"
        f'    regression_threshold: "{threshold}"\n'
        f"    gating: {gating}\n"
        "  - name: other\n"
        '    regression_threshold: "+99%"\n',
        encoding="utf-8",
    )
    return p


def test_threshold_parsed_from_the_entry_only(tmp_path):
    assert crw.read_regression_threshold(_config(tmp_path, "-10%")) == 10.0
    assert crw.read_regression_threshold(_config(tmp_path, "-25.5%")) == 25.5


def test_threshold_unparsable_raises(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text("  - name: cache_read_ratio\n    gating: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        crw.read_regression_threshold(p)
    with pytest.raises(ValueError):
        crw.read_regression_threshold(tmp_path / "missing.md")
    q = tmp_path / "noentry.md"
    q.write_text("  - name: other\n    regression_threshold: \"-10%\"\n", encoding="utf-8")
    with pytest.raises(ValueError):
        crw.read_regression_threshold(q)


def test_a_different_config_threshold_flips_a_borderline_verdict(tmp_path):
    rows = _with_later(800)  # 80% of F
    assert check(rows, crw.read_regression_threshold(_config(tmp_path, "-10%")))["status"] == "PASS"
    assert check(rows, crw.read_regression_threshold(_config(tmp_path, "-25%")))["status"] == "FAIL"


def test_real_repo_config_entry_parses_and_is_gating():
    assert crw.read_regression_threshold(CONFIG) == 10.0
    entry = _entry_lines(CONFIG.read_text(encoding="utf-8"))
    assert "    gating: true" in entry
    assert "    direction: lower_is_better" in entry


def _entry_lines(text):
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "- name: cache_read_ratio")
    out = [lines[start]]
    for ln in lines[start + 1 :]:
        if ln.strip().startswith("- name:") or ln.startswith("  # --- BLOCKED"):
            break
        out.append(ln)
    return out


def test_config_gating_oracle_can_fail(tmp_path):
    flipped = _config(tmp_path, "-10%", gating="false")
    assert "    gating: true" not in _entry_lines(flipped.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# C8 stale text / help
# --------------------------------------------------------------------------------------


def test_stale_tools_path_text_is_gone_and_replaced():
    src = SCRIPT.read_text(encoding="utf-8")
    for stale in ("logs no usage row", "logs NO usage row", "stays empty", "can't be read"):
        assert stale not in src, stale
    helps = {a.option_strings[0]: a.help for a in _parser()._actions if a.option_strings}
    assert "logs usage rows" in helps["--with-tools"]
    assert "logs a usage" in " ".join((crw._TextPathProvider.__doc__ or "").split())


def _parser():
    import argparse

    captured = {}
    orig = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        captured["p"] = self
        return orig(self, *a, **k)

    argparse.ArgumentParser.parse_args = grab
    try:
        crw.parse_args([])
    finally:
        argparse.ArgumentParser.parse_args = orig
    return captured["p"]


# --------------------------------------------------------------------------------------
# C10 wiring: _summarise + CLI
# --------------------------------------------------------------------------------------


def test_summarise_carries_the_check_and_survives_a_bad_config(monkeypatch):
    s = crw._summarise(healthy(), [], turns=5)
    assert s["cache_break_check"]["status"] == "PASS"
    assert _OLD_SUMMARY_KEYS <= set(s)

    def boom(*a, **k):
        raise ValueError("no entry")

    monkeypatch.setattr(crw, "read_regression_threshold", boom)
    s = crw._summarise(healthy(), [], turns=5)
    assert s["cache_break_check"]["status"] == "UNMEASURED"
    assert "threshold unreadable" in s["cache_break_check"]["reason"]
    assert crw.format_cache_break_line(s["cache_break_check"]).startswith("cache_break_check: UNMEASURED (")


def _write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _cli(path, *extra):
    return crw.main(["--cache-break-check", str(path), *extra])


def test_cli_exit_codes_and_skips_non_chat_rows(tmp_path, capsys):
    f = tmp_path / "u.jsonl"
    rows = healthy()
    rows.insert(2, row(77777, 0, call_type="generate"))
    _write_rows(f, rows)
    assert _cli(f) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    bad = healthy()
    bad[3] = row(16900, 0)
    _write_rows(f, bad)
    assert _cli(f) == 1
    _write_rows(f, [row(1200, 16984), row(1200, 17000)])
    assert _cli(f) == 2


def test_cli_from_row_is_one_based_over_chat_rows(tmp_path):
    f = tmp_path / "u.jsonl"
    # a previous run (cold row + small rows), then a second run starting at chat row 4
    prev = [row(16984, 0), row(1200, 17000), row(1200, 17500)]
    cur = [row(26000, 0), row(1000, 26000), row(25500, 100)]
    rows = [row(5, 5, call_type="generate")] + prev + [row(5, 5, call_type="generate")] + cur
    _write_rows(f, rows)
    assert _cli(f, "--from-row", "4") == 1  # current run: row 3 of it is a break
    assert _cli(f, "--from-row", "1") == 1  # whole file: the 25500 row is a break against F=16984
    assert _cli(f, "--from-row", "2") == 2  # reference = second chat row (warm): UNMEASURED
    for bad in ("0", "-3"):
        assert _cli(f, "--from-row", bad) == 2
    with pytest.raises(SystemExit) as exc:
        _cli(f, "--from-row", "x")
    assert exc.value.code == 2


def test_cli_from_row_counts_chat_rows_not_file_lines(tmp_path):
    f = tmp_path / "u.jsonl"
    # file lines: 1 generate, 2 chat(cold), 3 generate, 4 chat, 5 chat(break). --from-row 1 -> reference is chat row 1 (line 2)
    _write_rows(
        f,
        [row(1, 1, call_type="generate"), row(1000, 0), row(1, 1, call_type="generate"), row(100, 1000), row(950, 1000)],
    )
    assert _cli(f, "--from-row", "1") == 1
    # chat row 2 (file line 4) as reference is warm (read 1000 >= 10% of 100) -> UNMEASURED; file-line indexing would give another verdict
    assert _cli(f, "--from-row", "2") == 2
    assert _cli(f, "--from-row", "3") == 2  # a single chat row left: V1


def test_cli_refuses_corrupt_lines_instead_of_skipping_them(tmp_path, capsys):
    f = tmp_path / "u.jsonl"
    good = [json.dumps(r) for r in healthy()]
    # a torn line where the break row would be: skipping it would yield PASS
    f.write_text("\n".join(good[:3] + ['{"call_type": "chat", "cache_creation_input_t'] + good[3:]) + "\n", encoding="utf-8")
    assert _cli(f) == 2
    assert "line 4 is not valid JSON" in capsys.readouterr().err
    # a torn FIRST chat row must not silently move the reference
    f.write_text('{"call_type": "chat", "cache_cr\n' + "\n".join(good[1:]) + "\n", encoding="utf-8")
    assert _cli(f) == 2
    # valid JSON that is not an object
    f.write_text("[1, 2]\n" + "\n".join(good) + "\n", encoding="utf-8")
    assert _cli(f) == 2
    assert "not a JSON object" in capsys.readouterr().err
    # blank lines are fine
    f.write_text("\n\n" + "\n".join(good) + "\n\n", encoding="utf-8")
    assert _cli(f) == 0


def test_cli_any_unexpected_exception_is_exit_2(tmp_path, monkeypatch, capsys):
    f = tmp_path / "u.jsonl"
    _write_rows(f, healthy())

    def boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(crw, "check_cache_break", boom)
    assert _cli(f) == 2
    assert "unexpected" in capsys.readouterr().err


def test_cli_errors_are_exit_2_never_1(tmp_path, capsys):
    assert _cli(tmp_path / "missing.jsonl") == 2
    f = tmp_path / "u.jsonl"
    f.write_text("not json\n", encoding="utf-8")
    assert _cli(f) == 2
    capsys.readouterr()
    _write_rows(f, healthy())
    assert _cli(f, "--regression-threshold", "150") == 2


def test_cli_threshold_override(tmp_path):
    f = tmp_path / "u.jsonl"
    _write_rows(f, _with_later(800))
    assert _cli(f) == 0
    assert _cli(f, "--regression-threshold", "25") == 1


# --------------------------------------------------------------------------------------
# Warm-up turn: wiring through run_replay on the text path and the tools path
# --------------------------------------------------------------------------------------


class _Scripted:
    """FakeProvider that also writes a usage row per chat call, through the real log_usage."""

    def __init__(self, persona_dir, script):
        from brain.bridge.provider import FakeProvider

        self._fake = FakeProvider()
        self.persona_dir = persona_dir
        self.script = script
        self.calls = []

    def name(self):
        return "scripted"

    def healthy(self):
        return True

    def generate(self, *a, **k):
        return self._fake.generate(*a, **k)

    def chat(self, messages, *, tools=None, options=None):
        from brain.bridge.usage_log import log_usage

        user_text = messages[-1].content_text()
        self.calls.append({"tools": tools is not None, "user": user_text})
        creation, read, turns = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        log_usage(
            self.persona_dir,
            call_type="chat",
            model="m",
            frame={
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": creation,
                    "cache_read_input_tokens": read,
                },
                "num_turns": turns,
                "total_cost_usd": 0.0,
                "session_id": "s",
            },
        )
        return self._fake.chat(messages, tools=tools, options=options)


def _replay(tmp_path, monkeypatch, script, *, with_tools, warmup=True, turns=3):
    import brain.bridge.provider as provider_mod

    monkeypatch.setenv("NELL_CACHE_DEBUG", "0")  # run_replay sets it to 1; monkeypatch restores the original
    pd = tmp_path / "personas" / "replay"
    crw._seed_scratch_persona(pd)
    holder = {}

    def fake_get_provider(name, *, persona_dir=None, model_override=None):
        holder["p"] = _Scripted(persona_dir, script)
        return holder["p"]

    monkeypatch.setattr(provider_mod, "get_provider", fake_get_provider)
    summary, replies = crw.run_replay(
        pd,
        turns=turns,
        gap_s=0,
        provider_name="claude-cli",
        seed=False,
        force_text_path=not with_tools,
        warmup=warmup,
    )
    return summary, replies, holder["p"]


# first call = warm-up: cold single-call write; later calls read it. Tools-path rows can be multi-call.
_HEALTHY = [(16984, 0, 1)] + [(1200, 17000 + 500 * k, 3) for k in range(1, 6)]
_BREAK = [(16984, 0, 1), (1200, 17500, 3), (16900, 90, 3), (1200, 18000, 3)]


@pytest.mark.parametrize("with_tools", [False, True])
def test_warmup_is_the_first_turn_on_both_paths_and_the_run_is_measurable(tmp_path, monkeypatch, with_tools):
    summary, replies, prov = _replay(tmp_path, monkeypatch, _HEALTHY, with_tools=with_tools)
    assert replies[0]["turn"] == 0 and replies[0]["prompt"] == crw.WARMUP_PROMPT
    assert [r["turn"] for r in replies] == [0, 1, 2, 3]
    assert prov.calls[0]["user"].strip().endswith(crw.WARMUP_PROMPT)
    # tools were offered to the provider on the tools path only (the text-path wrapper strips them)
    assert all(c["tools"] is with_tools for c in prov.calls)
    assert summary["warmup_turn"] is True
    assert summary["cache_break_check"]["status"] == "PASS"
    assert summary["cache_break_check"]["fixed_prompt_tokens"] == 16984
    assert summary["chat_rows_observed"] == 4 and summary["turns_requested"] == 3


@pytest.mark.parametrize("with_tools", [False, True])
def test_a_break_in_a_warmed_run_fails_on_both_paths(tmp_path, monkeypatch, with_tools):
    summary, _, _ = _replay(tmp_path, monkeypatch, _BREAK, with_tools=with_tools)
    res = summary["cache_break_check"]
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == [3]


def test_without_warmup_a_multi_call_first_row_is_unmeasured_never_a_pass(tmp_path, monkeypatch):
    multi_first = [(38235, 0, 4)] + _HEALTHY[1:]
    summary, replies, _ = _replay(tmp_path, monkeypatch, multi_first, with_tools=True, warmup=False)
    assert summary["warmup_turn"] is False
    assert replies[0]["turn"] == 1
    assert summary["cache_break_check"]["status"] == "UNMEASURED"
    assert "V3" in summary["cache_break_check"]["reason"]


def test_no_warmup_flag_parses():
    assert crw.parse_args(["--no-warmup"]).no_warmup is True
    assert crw.parse_args([]).no_warmup is False


def test_warmup_prompt_is_the_micro_ack():
    # "ok" is exempt from the record_monologue directive (brain/chat/monologue_prompts.py); pin it
    assert crw.WARMUP_PROMPT == "ok"


def test_compare_warns_when_only_one_arm_had_the_warmup_row(tmp_path, capsys):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    _write_summary(old, [1000.0, 1000.0], [2000.0, 2000.0])
    _write_summary(new, [500.0, 500.0], [2500.0, 2500.0])
    nj = json.loads(new.read_text())
    nj["warmup_turn"] = True
    new.write_text(json.dumps(nj))
    crw.compare(old, new)
    assert "WARNING: warmup_turn differs (old=False, new=True)" in capsys.readouterr().out
    # same setting on both arms: no warning (old-format JSON on both sides is the characterization case)
    oj = json.loads(old.read_text())
    oj["warmup_turn"] = True
    old.write_text(json.dumps(oj))
    crw.compare(old, new)
    assert "WARNING" not in capsys.readouterr().out


def test_replay_prints_the_verdict_line_on_stderr(tmp_path, monkeypatch, capsys):
    summary, _, _ = _replay(tmp_path, monkeypatch, _HEALTHY, with_tools=False)
    err = capsys.readouterr().err
    assert "cache_break_check: PASS (" in err
    assert crw.format_cache_break_line(summary["cache_break_check"]) in err


def test_compare_does_not_warn_when_neither_arm_has_a_warmup_row(tmp_path, capsys):
    # an old-format JSON (no key) against a --no-warmup JSON (False): neither arm has the warm-up row
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    _write_summary(old, [1000.0, 1000.0], [2000.0, 2000.0])
    _write_summary(new, [500.0, 500.0], [2500.0, 2500.0])
    nj = json.loads(new.read_text())
    nj["warmup_turn"] = False
    new.write_text(json.dumps(nj))
    crw.compare(old, new)
    assert "WARNING" not in capsys.readouterr().out
