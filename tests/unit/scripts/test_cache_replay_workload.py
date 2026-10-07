"""scripts/cache_replay_workload.py - the per-run cache-break check (#339, reworked for #340).

`check_cache_break` is a pure function over `chat_usage.jsonl` rows. A cache break is a later PLAIN
(num_turns == 1) chat row whose cache_read is 0 or below S, the minimum of the earlier nonzero judged reads
(T = 0 in the config). The first chat row (the cold write) is never judged; multi-call rows are not judged and
impose no rule; the run needs at least MIN_JUDGED_ROWS (2) judged rows, else UNMEASURED. No call counts, no model
calls, no OLD-vs-NEW comparison. The definition, bases and limits live in the config entry (one home).

Row shapes here are SYNTHETIC (owner ruling: a mechanical check, tested without model calls). Rows are
written (creation, read) as in the #340 review. Shapes taken from a real measurement are named in comments.
Read lists such as [6449, 2000, 2000] are the judged reads of rows 2.. with a cold row 1 prepended; row numbers
in assertions are chat-row numbers (the cold row is row 1).
Every check test asserts `status` FIRST so a failure against another version of the script is a verdict
mismatch, not a KeyError (the bite harness runs this file against the old script and against mutants).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = Path(os.environ.get("CRW_SCRIPT", REPO / "scripts" / "cache_replay_workload.py"))
CONFIG = Path(os.environ.get("CRW_CONFIG", REPO / "guarded-change.companion.md"))  # the bite harness points it at a mutated copy
LOCKROOT = Path(os.environ.get("CRW_LOCKROOT", REPO))  # where pyproject.toml / uv.lock are read from (the bite harness)

_spec = importlib.util.spec_from_file_location("cache_replay_workload_under_test", SCRIPT)
crw = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = crw
_spec.loader.exec_module(crw)

# The config's threshold today is 0 (tests that read the real file assert it separately). The bite harness
# sets CRW_TEST_T=10 to run the OLD script, which rejects T = 0 (that rejection would be a proxy kill).
T = float(os.environ.get("CRW_TEST_T", "0"))


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


def run(*items, first=(17385, 0), later_creation=12000):
    """A cold first row, then one row per item: an int read (single call) or (read, num_turns)."""
    rows = [row(first[0], first[1], t=0)]
    for k, it in enumerate(items, 1):
        read, turns = it if isinstance(it, tuple) else (it, 1)
        rows.append(row(later_creation, read, turns=turns, t=k))
    return rows


def check(rows, thr=None):
    return crw.check_cache_break(rows, threshold_pct=T if thr is None else thr)


def safe_check(rows, thr=None):
    """check(), but a crash of the check is reported as a failed assertion (a verdict-bearing failure)."""
    try:
        return check(rows, thr)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"check_cache_break crashed: {exc!r}") from exc


def _old_rule_fails(rows, t=10.0):
    """The signal this rework replaced: a later row's creation reaching the first row's creation F."""
    f = rows[0]["cache_creation_input_tokens"]
    return any(r["cache_creation_input_tokens"] * 100 >= f * (100 - t) for r in rows[1:])


# --------------------------------------------------------------------------------------
# Characterization: written and run against the PRE-change script first
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
    c8 = s["c8_cache"]  # the pre-existing keys keep their values (the labels are additive)
    assert {k: c8[k] for k in ("mean_cache_creation", "mean_cache_read", "cache_creation_series", "cache_read_series")} == {
        "mean_cache_creation": 500.0,
        "mean_cache_read": 833.3,
        "cache_creation_series": [1000.0, 200.0, 300.0],
        "cache_read_series": [0.0, 1000.0, 1500.0],
    }
    assert s["c9_history_caching"]["last_turn_cache_creation"] == 300.0


def _write_summary(path, creation, read, **extra):
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
                **extra,
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
    assert crw.compare(old, new) == crw.EXIT_FAIL
    assert "C8 (system-block cache stops re-creating): FAIL" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# C1 / C2: the review's counterexample and the #332 shrink pass
# --------------------------------------------------------------------------------------


def test_c1_review_counterexample_passes():
    # (creation/read) rows from the #340 review: the 6000-token prefix is read every turn.
    rows = [row(17000, 0, t=0), row(11500, 6000, t=1), row(16000, 6000, t=2)]
    assert _old_rule_fails(rows)  # precondition: the replaced rule FAILed this healthy run (16000/17000)
    res = check(rows)
    assert res["status"] == "PASS"
    assert res["rows_checked"] == 2
    assert res["stable_read_tokens"] == 6000


@pytest.mark.parametrize("prefix", [15476, 6176])
def test_c2_shrunk_prefix_passes_per_run(prefix):
    # #332 shape: run A has the full prefix (15476, the review's real constant), run B 9.3k less (6176),
    # same volatile tail (creation 11-20k). Each run is measured against its own reads.
    creations = [11000, 14000, 17000, 20000, 15000]
    rows = [row(prefix + 11000, 0, t=0)] + [row(c, prefix, t=k) for k, c in enumerate(creations, 1)]
    res = check(rows)
    assert res["status"] == "PASS"
    assert res["rows_checked"] == 5


def test_c2_preconditions_the_legacy_signals_misfire_on_the_shrunk_run():
    def build(prefix):
        creations = [11000, 14000, 17000, 20000, 15000]
        return [row(prefix + 11000, 0, t=0)] + [row(c, prefix, t=k) for k, c in enumerate(creations, 1)]

    a, b = build(15476), build(6176)

    def ratio(rows):
        return sum(r["cache_read_input_tokens"] for r in rows) / sum(r["cache_creation_input_tokens"] for r in rows)

    assert (ratio(b) - ratio(a)) / ratio(a) < -0.10  # the legacy sum(read)/sum(creation) ratio dropped > 10%
    assert _old_rule_fails(b)  # and the replaced per-row signal fails run B's healthy row (20000 vs F=17176)


# --------------------------------------------------------------------------------------
# C3: healthy shapes pass
# --------------------------------------------------------------------------------------


def test_c3_constant_reads_pass_the_measured_shape():
    # measured on a healthy replay: every plain row read exactly 6449 (rows 9-11 of the 2026-10-06 run)
    res = check(run(6449, 6449, 6449))
    assert res["status"] == "PASS"
    assert (res["rows_checked"], res["stable_read_tokens"], res["min_share_of_S"]) == (3, 6449, 1.0)


def test_c3_growing_reads_pass():
    assert check(run(6000, 9000, 12000))["status"] == "PASS"


def test_c3_row_one_is_never_judged_whatever_it_is():
    assert check(run(6449, 6449, first=(5000, 17000)))["status"] == "PASS"  # warm re-run: row 1 read a lot
    rows = run(6449, 6449)
    rows[0]["num_turns"] = 4  # row 1 multi-call
    assert check(rows)["status"] == "PASS"
    rows[0]["cache_read_input_tokens"] = None  # row 1 malformed: not used at all
    assert check(rows)["status"] == "PASS"


def test_c3_boundary_at_zero_tolerance_equal_passes_one_below_breaks():
    assert check(run(1000, 1000, 1000), 0)["status"] == "PASS"
    res = check(run(1000, 999, 1000), 0)
    assert res["status"] == "FAIL" and [b["row"] for b in res["breaks"]] == [3]


def test_c3_the_threshold_is_honoured_when_the_config_value_changes():
    assert check(run(1000, 951), 10)["status"] == "PASS"  # 4.9% below S, tolerated at T=10
    assert check(run(1000, 899, 1000), 10)["status"] == "FAIL"  # 10.1% below
    assert check(run(1000, 900, 1000), 10)["status"] == "PASS"  # exactly 10% below is within T


def _highest_read_rule_fails(reads, t=0):
    """The replaced rule (aaac1b22): S = the highest earlier judged read; a read below S breaks."""
    s = None
    for rd in reads:
        if rd == 0 or (s is not None and rd * 100 < s * (100 - t)):
            return True
        s = rd if s is None else max(s, rd)
    return False


def test_c3_documented_limit_with_a_positive_threshold_small_steps_lower_s_and_pass():
    # config limit (4): with T > 0 each step within T% of the current minimum lowers S; T is 0 today.
    # The replaced highest-read reference FAILed this erosion (a 43% fall), the minimum passes it.
    reads = [10000, 9100, 8281, 7536, 6858, 6240, 5679]
    assert _highest_read_rule_fails(reads, 10)  # precondition: the replaced rule fails it
    res = check(run(*reads), 10)
    assert res["status"] == "PASS" and res["stable_read_tokens"] == 5679


def test_c3_non_chat_rows_are_ignored_even_when_first():
    rows = run(6449, 6449)
    rows.insert(0, row(99999, 99999, call_type="generate"))
    rows.insert(2, row(1, 1, call_type="generate"))
    assert check(rows)["status"] == "PASS"


# --------------------------------------------------------------------------------------
# C4: a genuine break fails
# --------------------------------------------------------------------------------------


def test_c4_a_read_drop_fails_and_names_the_row():
    res = check(run(6449, 6449, 2000, 6449))
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == [4]
    assert res["breaks"][0]["read"] == 2000 and res["breaks"][0]["reference_read"] == 6449
    assert res["breaks"][0]["share_of_S"] == round(2000 / 6449, 3)


def test_c4_break_at_the_last_row_and_break_detail_fields():
    rows = run(6449, 6449, 0)
    rows[3]["ts"] = "2026-10-01T12:00:59+00:00"
    rows[2]["ts"] = "2026-10-01T12:00:05+00:00"
    res = check(rows)
    assert res["status"] == "FAIL"
    assert res["breaks"] == [
        {
            "row": 4,
            "read": 0,
            "reference_read": 6449,
            "share_of_S": 0.0,
            "creation": 12000,
            "num_turns": 1,
            "ts": "2026-10-01T12:00:59+00:00",
            "gap_s": 54.0,
        }
    ]


def test_c4_gap_is_none_when_ts_is_unparsable():
    rows = run(6449, 2000)
    rows[2]["ts"] = "not a time"
    assert check(rows)["breaks"][0]["gap_s"] is None


def test_c4_persistent_zero_reads_fail_even_before_s_exists():
    res = check(run(0, 0, 0))
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == [2, 3, 4]
    assert res["breaks"][0]["reference_read"] is None and res["breaks"][0]["share_of_S"] is None


def test_c4_a_zero_among_healthy_rows_fails():
    assert check(run(6449, 0, 6449))["status"] == "FAIL"


def test_c4_a_persisting_lower_level_is_one_break_because_the_lower_read_becomes_s():
    res = check(run(6449, 3000, 3000, 3000))
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == [3]  # row 3 broke against 6449; later rows are not below S = 3000
    assert res["stable_read_tokens"] == 3000


def test_c4_fail_wins_over_a_malformed_row():
    rows = run(6449, 6449, 2000)
    rows.insert(2, row(12000, None, t=9))
    assert check(rows)["status"] == "FAIL"


def test_c4_a_slice_spanning_a_prompt_change_fails_naming_row_and_gap():
    # documented limit: a slice must be ONE run; S from the old prefix, every row after the change breaks
    res = check(run(15476, 15476, 6176, 6176))
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == [4]
    assert all(b["ts"] and b["gap_s"] is not None for b in res["breaks"])


def test_c4_documented_limit_a_persistent_partial_break_passes_by_construction():
    # later plain reads constant at 3000 from the first judged row (a tools block still readable behind
    # a churning system block): S follows the level. Left to cache_creation / cost metrics (config, limit 1).
    res = check(run(3000, 3000, 3000))
    assert res["status"] == "PASS" and res["stable_read_tokens"] == 3000


# --------------------------------------------------------------------------------------
# K1: the minimum of the earlier judged reads (main's ruling A)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reads",
    [
        [6449, 15476, 6449, 6449],  # a high read in the middle
        [6449, 20000, 30000, 6449],  # history window slide: reads climb, then fall back to the prefix
        [6449, 12000, 18000, 6449, 12000],  # compaction shape
    ],
)
def test_k1_a_high_read_after_the_first_judged_row_does_not_fail_later_rows(reads):
    assert _highest_read_rule_fails(reads)  # precondition: the replaced highest-read rule FAILed these
    res = check(run(*reads))
    assert res["status"] == "PASS"
    assert res["stable_read_tokens"] == min(reads)


def test_k1_documented_limit_a_high_read_on_the_first_judged_row_fails_later_normal_reads():
    # The review's exact rows (creation/read, all single-call): the first judged read is 15476, so the later 6449
    # reads break at row 3. Known limit of the minimum, see guarded-change.companion.md, metric cache_read_ratio,
    # "Limits (1)": not observed in either measurement; the FAIL names the row, the read and S.
    rows = [row(17000, 0, t=0), row(9000, 15476, t=1), row(9000, 6449, t=2), row(9000, 6449, t=3)]
    res = check(rows)
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == [3]
    assert res["breaks"][0]["read"] == 6449 and res["breaks"][0]["reference_read"] == 15476


def test_k1_a_persistent_break_from_the_second_judged_row_fails_at_that_row():
    res = check(run(6449, 2000, 2000))
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == [3]  # row 4 is not below S = 2000, the break is reported once
    assert res["stable_read_tokens"] == 2000


@pytest.mark.parametrize(
    "reads,rows",
    [([0, 6449, 6449], [2]), ([5000, 0, 6449], [3]), ([0, 0, 0], [2, 3, 4]), ([6449, 0, 6449, 6449], [3]), ([6449, 6449, 6449, 0], [5])],
)
def test_k1_a_zero_read_is_a_break_never_a_crash_and_never_enters_s(reads, rows):
    res = safe_check(run(*reads))  # a zero in S would divide by zero in the share and let every later row pass
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == rows
    if reads == [6449, 0, 6449, 6449]:
        assert res["stable_read_tokens"] == 6449


def test_k1_every_short_list_of_reads_gets_the_verdict_of_an_independent_oracle():
    import itertools

    for n in range(0, 6):
        for reads in itertools.product((0, 1, 2000, 6449), repeat=n):
            res = safe_check(run(*reads))
            prior_min, fail = None, False
            for rd in reads:
                if rd == 0 or (prior_min is not None and rd < prior_min):
                    fail = True
                if rd > 0:
                    prior_min = rd if prior_min is None else min(prior_min, rd)
            expected = "FAIL" if fail else ("PASS" if len(reads) >= 2 else "UNMEASURED")
            assert res["status"] == expected, reads


def test_k1_the_text_path_shape_of_two_judged_rows_is_one_comparison_with_zero_tolerance():
    assert check(run(8403, 8403))["status"] == "PASS"
    assert check(run(8403, 8402))["status"] == "FAIL"
    assert check(run(8403, 9000))["status"] == "PASS"  # a rise is fine


def test_k1_precondition_the_replaced_rule_fails_what_min_passes_and_one_establishing_row_fails_the_limit():
    assert _highest_read_rule_fails([6449, 15476, 6449, 6449])
    first_only = [15476, 6449, 6449]  # min-of-earlier with one establishing row: row 2 is below 15476
    assert any(rd < min(first_only[:i]) for i, rd in enumerate(first_only) if i >= 1)


def test_k1_a_negative_read_is_unusable_not_a_break_or_a_reference():
    res = check(run(-1, 6449, 6449))
    assert res["status"] == "UNMEASURED" and "row 2" in res["reason"] and res["stable_read_tokens"] == 6449


# Real healthy rows (creation, read, num_turns), numbers copied from two measurement runs on throwaway synthetic
# personas: 2026-10-06 haiku, tools path (11 rows) and 2026-10-07 sonnet, text path (8 rows).
_REAL_HAIKU_TOOLS = [
    (17385, 0, 1), (35488, 77663, 5), (23760, 42332, 3), (23170, 18543, 2), (24998, 115856, 6), (24171, 42643, 3),
    (24883, 18543, 2), (25646, 18543, 2), (12455, 6449, 1), (12703, 6449, 1), (12919, 6449, 1),
]
_REAL_SONNET_TEXT = [
    (22423, 0, 1), (14188, 8403, 1), (14994, 54499, 4), (16019, 55138, 3), (16256, 79827, 4), (18365, 83479, 4),
    (16977, 57852, 3), (16238, 8403, 1),
]


def test_k7_the_two_measured_healthy_runs_pass():
    for real, judged, s in ((_REAL_HAIKU_TOOLS, 3, 6449), (_REAL_SONNET_TEXT, 2, 8403)):
        rows = [row(c, rd, turns=n, t=k) for k, (c, rd, n) in enumerate(real)]
        res = check(rows)
        assert res["status"] == "PASS" and res["rows_checked"] == judged and res["stable_read_tokens"] == s


# --------------------------------------------------------------------------------------
# C5: multi-call rows are not judged and impose no rule
# --------------------------------------------------------------------------------------

_PLAIN = [6449, 6449]


@pytest.mark.parametrize("multi_read", [0, 5, 6449, 99999])
def test_c5_interleaved_multi_call_rows_change_nothing(multi_read):
    base = check(run(*_PLAIN))
    mixed = check(run(6449, (multi_read, 3), 6449, (multi_read, 2)))
    assert base["status"] == mixed["status"] == "PASS"
    assert mixed["rows_checked"] == base["rows_checked"] == 2
    assert mixed["stable_read_tokens"] == base["stable_read_tokens"]
    assert mixed["multicall_rows"] == [3, 5]
    broken = check(run(6449, (multi_read, 3), 2000, (multi_read, 2)))
    assert broken["status"] == "FAIL" and [b["row"] for b in broken["breaks"]] == [4]


def test_c5_a_trailing_tail_of_multi_call_rows_imposes_no_rule():
    res = check(run(6449, 6449, (0, 3), (1, 5), (0, 2)))
    assert res["status"] == "PASS" and res["multicall_rows"] == [4, 5, 6]


@pytest.mark.parametrize("turns", [2, 3, 4, None, True])
def test_c5_anything_but_an_int_one_is_multi_call(turns):
    rows = run(6449, 6449, 6449)
    rows[3]["num_turns"] = turns
    if turns is None:
        del rows[3]["num_turns"]
    res = check(rows)
    assert res["status"] == "PASS"
    assert res["multicall_rows"] == [4] and res["rows_checked"] == 2


def test_c5_fixture_shaped_row_is_not_judged_and_not_a_break():
    # numbers of the recorded real CLI result frame (tests/bridge/fixtures/cli_2_1_284_web_tools.ndjson):
    # creation 38235, read 72930, num_turns 4 (3 API calls; its last call read 37309)
    rows = [row(17385, 0), row(12000, 37309), row(12000, 37309), row(38235, 72930, turns=4)]
    res = check(rows)
    assert res["status"] == "PASS" and res["multicall_rows"] == [4] and res["rows_checked"] == 2


def test_c5_the_verdict_text_states_the_coverage():
    res = check(run(6449, (1, 3), 6449, (1, 3), (1, 3)))
    assert "judged 2 of 5 later rows; 3 multi-call rows not judged" in res["reason"]


def test_c5_a_run_of_only_multi_call_rows_is_unmeasured_never_a_pass():
    res = check(run((6449, 3), (6449, 2), (6449, 4)))
    assert res["status"] == "UNMEASURED" and res["rows_checked"] == 0


# --------------------------------------------------------------------------------------
# C6: UNMEASURED is never a pass
# --------------------------------------------------------------------------------------


def test_c6_fewer_than_two_chat_rows():
    assert check([])["status"] == "UNMEASURED"
    assert check([row(17385, 0)])["status"] == "UNMEASURED"
    assert check([row(17385, 0), row(1, 1, call_type="generate")])["status"] == "UNMEASURED"


def test_c6_minimum_judged_rows_boundary_is_two():
    assert crw.MIN_JUDGED_ROWS == 2  # derived: one row establishes S, one is compared against it
    one = check(run(6449))
    assert one["status"] == "UNMEASURED" and one["rows_checked"] == 1
    assert check(run(6449, 6449))["status"] == "PASS"
    assert check(run(6449, (6449, 3), (6449, 3)))["status"] == "UNMEASURED"


@pytest.mark.parametrize("bad", [None, "6449", 6449.0, True, "missing"])
def test_c6_a_later_row_without_an_int_read_is_unmeasured(bad):
    rows = run(6449, 6449, 6449)
    if bad == "missing":
        del rows[2]["cache_read_input_tokens"]
    else:
        rows[2]["cache_read_input_tokens"] = bad
    res = check(rows)
    assert res["status"] == "UNMEASURED" and "row 3" in res["reason"]


def test_c6_no_cache_activity_at_all_is_unusable_not_a_break():
    rows = run(6449, 6449)
    rows.append(row(0, 0, t=9))
    res = check(rows)
    assert res["status"] == "UNMEASURED" and "no cache activity" in res["reason"]


def test_c6_a_different_model_is_not_comparable_unless_another_row_breaks():
    rows = run(6449, 6449, 6449)
    rows[3]["model"] = "other"
    assert check(rows)["status"] == "UNMEASURED"
    rows[2]["cache_read_input_tokens"] = 100
    assert check(rows)["status"] == "FAIL"
    rows = run(6449, 6449, 6449)  # a missing model on either side is comparable
    rows[3].pop("model")
    assert check(rows)["status"] == "PASS"


def test_c6_the_runs_model_is_the_first_usable_single_call_later_row_not_row_one():
    rows = run((6449, 3), 6449, 6449, 6449)
    rows[0]["model"] = "first-row-model"
    rows[1]["model"] = "multi-call-model"
    assert check(rows)["status"] == "PASS"  # neither row 1 nor a multi-call row anchors the model


@pytest.mark.parametrize("thr", [-1, 100, 150, True, "10", None, float("nan")])
def test_c6_threshold_must_be_a_percentage_in_zero_to_hundred(thr):
    assert crw.check_cache_break(run(6449, 6449), threshold_pct=thr)["status"] == "UNMEASURED"


def test_c6_rows_checked_counts_only_judged_rows():
    rows = run(6449, (1, 3), 6449, 7000, (2, 2), 7000)  # 6 later rows
    rows.append(row(0, 0, t=9))  # unusable
    res = check(rows)
    assert res["status"] == "UNMEASURED"  # the unusable row
    assert res["rows_checked"] == 4  # rows 2, 4, 5, 7 judged; rows 3 and 6 multi-call; row 8 unusable


def test_c6_rows_checked_in_a_passing_mixed_run():
    res = check(run(6449, (1, 3), 6449, 7000, (2, 2), 7000))
    assert res["status"] == "PASS" and res["rows_checked"] == 4 and res["multicall_rows"] == [3, 6]


# --------------------------------------------------------------------------------------
# C7: the threshold is read from the config's yaml fence; no constant in the check
# --------------------------------------------------------------------------------------


def _config(tmp_path, threshold, *, fence="yaml", extra_before="", extra_in_entry=""):
    p = tmp_path / "cfg.md"
    p.write_text(
        "".join(
            [
                extra_before,
                f"```{fence}\n",
                "metrics:\n",
                "  - name: cache_creation_per_chat_call\n",
                '    regression_threshold: "+10%"\n',
                "  - name: cache_read_ratio\n",
                "    source: >\n",
                "      folded text that mentions the regression_threshold below\n",
                f"    regression_threshold: {threshold}\n",
                "    gating: true\n",
                extra_in_entry,
                "  - name: other\n",
                '    regression_threshold: "+99%"\n',
                "```\n",
            ]
        ),
        encoding="utf-8",
    )
    return p


@pytest.mark.parametrize(
    "literal,expected",
    [('"0%"', 0.0), ('"-10%"', 10.0), ("'-25.5%'", 25.5), ("0%", 0.0), ('"-10%"   # trailing comment', 10.0)],
)
def test_c7_threshold_is_parsed_from_the_entry_in_any_yaml_spelling(tmp_path, literal, expected):
    assert crw.read_regression_threshold(_config(tmp_path, literal)) == expected


def test_c7_key_order_and_the_entry_only(tmp_path):
    p = tmp_path / "c.md"
    p.write_text(
        "```yaml\nmetrics:\n  - regression_threshold: \"+99%\"\n    name: other\n"
        "  - gating: true\n    regression_threshold: '-7%'\n    name: cache_read_ratio\n```\n",
        encoding="utf-8",
    )
    assert crw.read_regression_threshold(p) == 7.0


def test_c7_the_same_key_elsewhere_is_not_picked_up(tmp_path):
    # in another metric, in prose outside the fence, in a comment, and inside the entry's folded source
    p = _config(
        tmp_path,
        '"-3%"',
        extra_before='prose regression_threshold: "-90%"\n- name: cache_read_ratio\n',
        extra_in_entry='    # regression_threshold: "-91%"\n',
    )
    assert crw.read_regression_threshold(p) == 3.0
    q = tmp_path / "q.md"
    q.write_text(
        "```yaml\nmetrics:\n  - name: cache_read_ratio\n    source: >\n"
        '      regression_threshold: "-92%"\n    regression_threshold: "-4%"\n```\n',
        encoding="utf-8",
    )
    assert crw.read_regression_threshold(q) == 4.0


def test_c7_only_fences_tagged_yaml_are_parsed(tmp_path):
    p = _config(tmp_path, '"-3%"', fence="")  # a bare fence is prose
    with pytest.raises(ValueError):
        crw.read_regression_threshold(p)
    q = tmp_path / "q.md"
    q.write_text(_config(tmp_path, '"-5%"').read_text() + "\n```\nstray: [unclosed\n", encoding="utf-8")
    assert crw.read_regression_threshold(q) == 5.0  # a stray bare fence after the yaml fence is harmless


@pytest.mark.parametrize("literal", ['"fast"', '"10"', '"+"', "10", "null"])
def test_c7_a_non_percent_value_raises(tmp_path, literal):
    with pytest.raises(ValueError):
        crw.read_regression_threshold(_config(tmp_path, literal))


def test_c7_missing_file_entry_key_or_broken_yaml_raise(tmp_path):
    with pytest.raises(ValueError):
        crw.read_regression_threshold(tmp_path / "missing.md")
    q = tmp_path / "noentry.md"
    q.write_text('```yaml\nmetrics:\n  - name: other\n    regression_threshold: "-10%"\n```\n', encoding="utf-8")
    with pytest.raises(ValueError):
        crw.read_regression_threshold(q)
    k = tmp_path / "nokey.md"
    k.write_text("```yaml\nmetrics:\n  - name: cache_read_ratio\n    gating: true\n```\n", encoding="utf-8")
    with pytest.raises(ValueError):
        crw.read_regression_threshold(k)
    b = tmp_path / "broken.md"
    b.write_text("```yaml\nmetrics: [unclosed\n```\n", encoding="utf-8")
    with pytest.raises(ValueError):
        crw.read_regression_threshold(b)


def test_c7_a_different_config_value_flips_a_borderline_verdict(tmp_path):
    rows = run(1000, 940)
    assert check(rows, crw.read_regression_threshold(_config(tmp_path, '"-10%"')))["status"] == "PASS"
    assert check(rows, crw.read_regression_threshold(_config(tmp_path, '"0%"')))["status"] == "FAIL"


def test_c7_real_repo_config_entry_parses_to_zero_percent_and_is_gating_stable():
    assert crw.read_regression_threshold(CONFIG) == 0.0
    import yaml

    body = re.search(r"^```yaml[ \t]*\n(.*?)^```[ \t]*$", CONFIG.read_text(encoding="utf-8"), re.M | re.S).group(1)
    entry = next(m for m in yaml.safe_load(body)["metrics"] if m["name"] == "cache_read_ratio")
    assert entry["gating"] is True
    assert entry["direction"] == "stable"
    assert entry["regression_threshold"] == "0%"


def test_c7_gating_oracle_can_fail(tmp_path):
    import yaml

    p = _config(tmp_path, '"0%"')
    p.write_text(p.read_text().replace("gating: true", "gating: false"), encoding="utf-8")
    body = re.search(r"^```yaml[ \t]*\n(.*?)^```[ \t]*$", p.read_text(), re.M | re.S).group(1)
    entry = next(m for m in yaml.safe_load(body)["metrics"] if m["name"] == "cache_read_ratio")
    assert entry["gating"] is not True


def test_c7_no_threshold_literal_in_the_check_code():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in ("check_cache_break", "read_regression_threshold"):
            consts = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))]
            assert 10 not in consts and 10.0 not in consts and 0.1 not in consts, (fn.name, consts)
            assert 0.9 not in consts


# --------------------------------------------------------------------------------------
# C9: reader reuse, fail-closed; CLI
# --------------------------------------------------------------------------------------


def _write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _cli(path, *extra):
    return crw.main(["--cache-break-check", str(path), *extra])


def _good_lines(*items):
    return [json.dumps(r) for r in run(*items)]


def test_cli_exit_codes_and_skips_non_chat_rows(tmp_path, capsys):
    f = tmp_path / "u.jsonl"
    rows = run(6449, 6449, 6449)
    rows.insert(2, row(77777, 0, call_type="generate"))
    _write_rows(f, rows)
    assert _cli(f, "--regression-threshold", "0") == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    _write_rows(f, run(6449, 6449, 2000))
    assert _cli(f, "--regression-threshold", "0") == 10
    _write_rows(f, run(6449))  # one judged row only
    assert _cli(f, "--regression-threshold", "0") == 11


def test_cli_reads_the_threshold_from_the_config_by_default(tmp_path, capsys):
    f = tmp_path / "u.jsonl"
    _write_rows(f, run(1000, 999))  # a 0.1% drop: a break only at T = 0 (the real config's value)
    assert _cli(f) == 10
    assert _cli(f, "--regression-threshold", "10") == 0
    capsys.readouterr()


def test_cli_from_row_is_one_based_over_chat_rows(tmp_path):
    f = tmp_path / "u.jsonl"
    prev = run(6449, 6449)  # a previous run
    cur = run(15476, 15476, 3000)  # the run under test: its last row breaks
    rows = [row(5, 5, call_type="generate")] + prev + [row(5, 5, call_type="generate")] + cur
    _write_rows(f, rows)
    assert _cli(f, "--from-row", "4", "--regression-threshold", "0") == 10  # the current run alone
    assert _cli(f, "--from-row", "1", "--regression-threshold", "0") == 10  # whole file: 15476 then 3000 breaks too
    assert _cli(f, "--from-row", "6", "--regression-threshold", "0") == 11  # two chat rows left: 1 judged row
    for bad in ("0", "-3"):
        assert _cli(f, "--from-row", bad) == 2
    with pytest.raises(SystemExit) as exc:
        _cli(f, "--from-row", "x")
    assert exc.value.code == 2


def test_cli_from_row_counts_chat_rows_not_file_lines(tmp_path):
    f = tmp_path / "u.jsonl"
    # file lines: 1 generate, 2 chat(cold), 3 generate, 4 chat, 5 chat, 6 chat(break)
    rows = [row(1, 1, call_type="generate"), row(17385, 0), row(1, 1, call_type="generate")]
    rows += [row(12000, 6449), row(12000, 6449), row(12000, 100)]
    _write_rows(f, rows)
    assert _cli(f, "--from-row", "1", "--regression-threshold", "0") == 10
    # chat row 2 as the run's first row leaves judged rows 3 and 4 (6449, 100): still a break
    assert _cli(f, "--from-row", "2", "--regression-threshold", "0") == 10
    # chat row 3 as first row leaves one later row: UNMEASURED (file-line indexing would give another verdict)
    assert _cli(f, "--from-row", "3", "--regression-threshold", "0") == 11


def test_cli_refuses_corrupt_lines_instead_of_skipping_them(tmp_path, capsys):
    f = tmp_path / "u.jsonl"
    good = _good_lines(6449, 6449, 2000)  # the break is the last row
    torn = '{"call_type": "chat", "cache_creation_input_t'
    for lines in (good[:3] + [torn] + good[3:], good[:3] + [torn], [torn] + good):
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert _cli(f, "--regression-threshold", "0") == 11
        assert "skipped 1 corrupt or non-object line" in capsys.readouterr().err
    # valid JSON that is not an object
    f.write_text("[1, 2]\n" + "\n".join(good) + "\n", encoding="utf-8")
    assert _cli(f, "--regression-threshold", "0") == 11
    # the break row torn instead of present: skipping it would give a PASS from the remaining rows
    f.write_text("\n".join(good[:3] + ['{"call_type": "chat", "cache_read_inp']) + "\n", encoding="utf-8")
    assert _cli(f, "--regression-threshold", "0") == 11
    capsys.readouterr()
    # blank lines are fine
    f.write_text("\n\n" + "\n".join(_good_lines(6449, 6449)) + "\n\n", encoding="utf-8")
    assert _cli(f, "--regression-threshold", "0") == 0


@pytest.mark.parametrize("silencer", ["disable", "disabled", "filter"])
def test_cli_fail_closed_does_not_depend_on_logging_configuration(tmp_path, silencer):
    f = tmp_path / "u.jsonl"
    f.write_text("\n".join(_good_lines(6449, 6449) + ["{torn"]) + "\n", encoding="utf-8")
    log = logging.getLogger("brain.health.jsonl_reader")
    flt = logging.Filter()
    flt.filter = lambda record: False
    try:
        if silencer == "disable":
            logging.disable(logging.CRITICAL)
        elif silencer == "disabled":
            log.disabled = True
        else:
            log.addFilter(flt)
        assert _cli(f, "--regression-threshold", "0") == 11
    finally:
        logging.disable(logging.NOTSET)
        log.disabled = False
        log.removeFilter(flt)


def test_cli_uses_the_shared_jsonl_reader(tmp_path, monkeypatch):
    import brain.health.jsonl_reader as reader

    calls = []
    real = reader.iter_jsonl_skipping_corrupt

    def spy(path):
        calls.append(Path(path))
        return real(path)

    monkeypatch.setattr(reader, "iter_jsonl_skipping_corrupt", spy)
    f = tmp_path / "u.jsonl"
    _write_rows(f, run(6449, 6449))
    assert _cli(f, "--regression-threshold", "0") == 0
    assert calls == [f]
    assert not hasattr(crw, "_read_jsonl_strict") and not hasattr(crw, "_read_jsonl")  # no private reader


def test_cli_an_unexpected_exception_is_not_a_verdict_it_propagates(tmp_path, monkeypatch):
    f = tmp_path / "u.jsonl"
    _write_rows(f, run(6449, 6449))

    def boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(crw, "check_cache_break", boom)
    with pytest.raises(RuntimeError):  # as a process this is exit 1, the crash code (see the subprocess test)
        _cli(f, "--regression-threshold", "0")


def test_cli_bad_arguments_are_exit_2_and_could_not_measure_is_exit_11(tmp_path, capsys, monkeypatch):
    assert _cli(tmp_path / "missing.jsonl") == 2
    assert _cli(tmp_path) == 2  # a directory is not a log file
    f = tmp_path / "u.jsonl"
    _write_rows(f, run(6449, 6449))
    for bad in ("150", "-1", "nan", "100"):
        assert _cli(f, "--regression-threshold", bad) == 2, bad
    assert _cli(f, "--from-row", "0") == 2
    f.write_text("not json\n", encoding="utf-8")
    assert _cli(f) == 11  # a corrupt line: could not measure, not a verdict of break
    _write_rows(f, run(6449, 6449))

    def bad_config(*a, **k):
        raise ValueError("no entry")

    monkeypatch.setattr(crw, "read_regression_threshold", bad_config)
    assert _cli(f) == 11  # an unreadable config
    capsys.readouterr()


def _script_run(*args, cwd=None):
    import subprocess

    env = dict(os.environ, PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd or REPO, env=env, capture_output=True, text=True, timeout=120)


def test_k3_exit_codes_of_the_script_as_a_process(tmp_path):
    f = tmp_path / "u.jsonl"
    cases = [(run(6449, 6449), 0), (run(6449, 6449, 2000), 10), (run(6449), 11)]
    for rows, code in cases:
        _write_rows(f, rows)
        assert _script_run("--cache-break-check", str(f), "--regression-threshold", "0").returncode == code
    assert _script_run("--cache-break-check", str(f), "--from-row", "x").returncode == 2  # argparse
    assert _script_run("--cache-break-check", str(tmp_path / "nope.jsonl")).returncode == 2
    missing = tmp_path / "missing.json"
    crash = _script_run("--compare", str(missing), str(missing))  # an uncaught FileNotFoundError
    assert crash.returncode == 1 and "FileNotFoundError" in crash.stderr
    assert {0, 10, 11}.isdisjoint({1, 2})


# --------------------------------------------------------------------------------------
# Config text (C12) and stale text
# --------------------------------------------------------------------------------------


def _entry_text(text):
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "- name: cache_read_ratio")
    out = [lines[start]]
    for ln in lines[start + 1 :]:
        if ln.strip().startswith("- name:") or ln.startswith("  # --- BLOCKED"):
            break
        out.append(ln)
    flat = " ".join(re.sub(r"^\s*#\s?", "", ln).strip() for ln in out)
    return " ".join(flat.split())


_PHRASES = [
    "never judged",
    "Only single-call rows (num_turns == 1) are judged",
    "The first judged row establishes S",
    "the minimum of the earlier nonzero judged reads",
    "read is 0 or below S",
    "at least 2 judged rows (one establishes S, one is compared against it)",
    "Multi-call rows (num_turns != 1, tool turns) are not judged and impose no rule",
    "same cached prefix (system prompt and tool block) as a plain message's",
    "the model decides on tools only after that call",
    "so a prefix break shows on plain rows too",
    "direction: stable",
    'regression_threshold: "0%"',
    "0% tolerance",
    "not an OLD->NEW delta",
    "Basis of S = the minimum of the earlier judged reads",
    "reads 6449, 2000, 2000 give S = 2000",
    "while the minimum catches it at the second row",
    "Every nonzero judged read enters the minimum, a zero read is a break and never enters it",
    "A high read on the FIRST judged row followed by normal reads FAILs",
    "reads 15476, 6449, 6449 break at the second",
    "It has not been observed",
    "The FAIL names the row, the read and S",
    "partial break on the first judged row lowers the floor",
    "one comparison of two reads with zero tolerance",
    "a --history-file replay is unmeasured",
    "lower S step by step and pass",
    "two small runs on two models",
    "5 direct single-call reads and 3 inferred first calls",
    "6449",
    "8403",
    "15,476",
    "OBSERVED, not bounded",
    "a transient miss included",
    "re-measure",
    "Basis of the minimum of 2 judged rows (derived, not tuned)",
    "UNMEASURED (exit 11) = an unverified gating criterion",
    "a log file that got smaller during the run (rotated or truncated)",
    "Exit codes of a verdict: 0 PASS, 10 FAIL, 11 UNMEASURED",
    "1 = a crash",
    "2 = argparse usage errors",
    "status NOT_APPLICABLE, exit 0 and one stderr line saying so",
    "`--compare` returns 0 / 10 / 11",
    "parses chat_usage.jsonl once, and only the bytes this run appended",
    "NOT COMPARABLE",
    "must be ONE run",
    "num_turns == 1 meaning one API call is an assumption",
    "uv run python scripts/cache_replay_workload.py --cache-break-check",
    "no OLD/NEW baseline, call count or logging change is needed",
    "complementary to the creation and cost metrics",
    "all of its chat rows, warm-up and plain turns included",
    "held two more chat rows than the nine replay turns, of unidentified origin",
    "so only the first is certainly a replay turn",
    "these three are the inferred first calls counted below",
    "were only estimated from an assumed per-call read",
]


@pytest.mark.parametrize("phrase", _PHRASES)
def test_c12_config_entry_states_the_new_meaning(phrase):
    entry = _entry_text(CONFIG.read_text(encoding="utf-8"))
    assert " ".join(phrase.split()) in entry, phrase


def test_k6_the_fields_are_self_describing_at_the_field_itself():
    lines = CONFIG.read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(lines) if ln == "    direction: stable")
    j = next(k for k, ln in enumerate(lines) if ln == '    regression_threshold: "0%"')
    above_direction = " ".join(lines[i - 2 : i])
    above_threshold = lines[j - 1]
    assert above_direction.count("#") == 2 and "Not an OLD->NEW delta" in above_direction
    assert "run's own stable read S" in above_direction
    assert above_threshold.strip().startswith("#") and "Tolerance below S in percent" in above_threshold
    assert "Not a regression threshold against a baseline" in above_threshold


_STALE = [
    r"\bstrict-first\b",
    r"\bfirst row is valid\b",
    r"\bV[1-4]\b",
    r"\bfixed_prompt_tokens\b",
    r"\bmax_share_of_F\b",
    r"\bF\b",
    r"within 10% of F",
    r"cache_creation_input_tokens \* 100 >= F",
    r"cold arm",
]

# Statements that stopped being true with the minimum / exit codes 0-10-11 / the declared dependency; swept over
# the whole config text and the script (the config entry says "highest earlier read" only in its history).
_STALE_V3 = [
    r"running max",
    r"highest read among",
    r"S = max",
    r"S is built from reads",
    r"exit 0 PASS / 1 FAIL / 2 UNMEASURED",
    r"UNMEASURED \(exit 2\)",
    r"legitimately exit 1 or 2",
    r"no pyproject change",
    r"locked transitive dependency",
    r"chat_rows_observed = turns \+ 3",
    r"warns on stderr when they differ",
    r"exits 0 PASS / 1 FAIL / 2 UNMEASURED",
    r"0 PASS / 1 FAIL / 2 UNMEASURED",
]


@pytest.mark.parametrize("pattern", _STALE)
def test_c12_stale_first_row_cold_text_is_gone(pattern):
    entry = _entry_text(CONFIG.read_text(encoding="utf-8"))
    script = " ".join(SCRIPT.read_text(encoding="utf-8").split())
    assert not re.search(pattern, entry), ("config", pattern)
    assert not re.search(pattern, script), ("script", pattern)


@pytest.mark.parametrize("pattern", _STALE_V3)
def test_k6_text_made_false_by_this_change_is_gone(pattern):
    config = " ".join(re.sub(r"^\s*#\s?", "", ln).strip() for ln in CONFIG.read_text(encoding="utf-8").splitlines())
    script = " ".join(SCRIPT.read_text(encoding="utf-8").split())
    assert not re.search(pattern, " ".join(config.split())), ("config", pattern)
    assert not re.search(pattern, script), ("script", pattern)


def test_k6_the_other_test_text_carries_no_stale_phrase_either():
    text = Path(__file__).read_text(encoding="utf-8")
    text = re.sub(r"^_STALE(_V3)? = \[.*?^\]\n", "", text, flags=re.M | re.S)  # the pattern lists name the phrases
    flat = " ".join(text.split())
    for pattern in _STALE_V3:
        assert not re.search(pattern, flat), pattern


def test_k6_the_rationale_has_one_home_and_the_docstrings_point_to_it():
    script = SCRIPT.read_text(encoding="utf-8")
    flat = " ".join(script.split())
    for rationale in ("catches it at the second row", "8403", "6449, 2000, 2000", "OBSERVED", "lower S step by step"):
        assert rationale not in flat, rationale  # the bases and limits are in the config only
    tree = ast.parse(script)
    mod_doc = " ".join((ast.get_docstring(tree) or "").split())
    assert "guarded-change.companion.md" in mod_doc and "cache_read_ratio" in mod_doc
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check_cache_break")
    fn_doc = " ".join((ast.get_docstring(fn) or "").split())
    assert "guarded-change.companion.md, metric cache_read_ratio" in fn_doc
    assert len((ast.get_docstring(fn) or "").splitlines()) <= 7  # a one-line summary, a pointer and the result keys


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
# Wiring: _summarise
# --------------------------------------------------------------------------------------


def test_summarise_carries_the_check_and_survives_a_bad_config(monkeypatch):
    rows = run(6449, 6449, 6449)
    s = crw._summarise(rows, [], turns=3)
    assert s["cache_break_check"]["status"] == "PASS"
    assert _OLD_SUMMARY_KEYS <= set(s)

    def boom(*a, **k):
        raise ValueError("no entry")

    monkeypatch.setattr(crw, "read_regression_threshold", boom)
    s = crw._summarise(rows, [], turns=3)
    assert s["cache_break_check"]["status"] == "UNMEASURED"
    assert "threshold unreadable" in s["cache_break_check"]["reason"]
    assert crw.format_cache_break_line(s["cache_break_check"]).startswith("cache_break_check: UNMEASURED (")


def test_summarise_rows_error_makes_the_verdict_unmeasured():
    s = crw._summarise(run(6449, 6449, 6449), [], turns=3, rows_error="2 lines skipped")
    assert s["cache_break_check"]["status"] == "UNMEASURED" and "2 lines skipped" in s["cache_break_check"]["reason"]


# --------------------------------------------------------------------------------------
# Replay: wiring through run_replay (scripted provider, real log_usage, no model call)
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

    def after_call(self, n):
        """Hook for tests: called after the n-th (1-based) call's rows were logged."""

    def chat(self, messages, *, tools=None, options=None):
        from brain.bridge.usage_log import log_usage

        user_text = messages[-1].content_text()
        self.calls.append({"tools": tools is not None, "user": user_text})
        entry = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        for creation, read, turns, *rest in entry if isinstance(entry, list) else [entry]:
            log_usage(
                self.persona_dir,
                call_type=rest[0] if rest else "chat",
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
        self.after_call(len(self.calls))
        return self._fake.chat(messages, tools=tools, options=options)


def _patch_provider(monkeypatch, tmp_path, script, cls=None):
    import brain.bridge.provider as provider_mod

    monkeypatch.setenv("NELL_CACHE_DEBUG", "0")  # run_replay sets it to 1; monkeypatch restores the original
    pd = tmp_path / "personas" / "replay"
    crw._seed_scratch_persona(pd)
    holder = {}

    def fake_get_provider(name, *, persona_dir=None, model_override=None):
        holder["p"] = (cls or _Scripted)(persona_dir, script)
        return holder["p"]

    monkeypatch.setattr(provider_mod, "get_provider", fake_get_provider)
    return pd, holder


def _replay(tmp_path, monkeypatch, script, *, with_tools, warmup=True, plain=True, turns=3, cls=None, provider_name="claude-cli", before=None):
    pd, holder = _patch_provider(monkeypatch, tmp_path, script, cls)
    if before:
        before(pd)
    try:
        summary, replies = crw.run_replay(
            pd,
            turns=turns,
            gap_s=0,
            provider_name=provider_name,
            seed=False,
            force_text_path=not with_tools,
            warmup=warmup,
            plain_turns=plain,
        )
    except Exception as exc:  # noqa: BLE001 - a crashed replay is a failed assertion, not an apparatus error
        raise AssertionError(f"run_replay crashed: {exc!r}") from exc
    return summary, replies, holder["p"]


# calls in order for turns=3: warm-up, plain "hi", 3 content turns, plain "thanks".
# warm-up = cold single-call write; content turns on the tools path are multi-call (reads are sums).
_HEALTHY_TOOLS = [(17385, 0, 1), (12000, 6449, 1), (12500, 18543, 3), (12500, 18543, 3), (12500, 18543, 3), (12000, 6449, 1)]
_HEALTHY_TEXT = [(17385, 0, 1)] + [(12000, 6449, 1)] * 5
_BREAK_IN_CONTENT = [(17385, 0, 1), (12000, 6449, 1), (17000, 0, 3), (17000, 0, 3), (17000, 0, 3), (17000, 0, 1)]


@pytest.mark.parametrize("with_tools", [False, True])
def test_c17_turn_order_the_warmup_then_a_plain_turn_then_content_then_a_plain_turn(tmp_path, monkeypatch, with_tools):
    script = _HEALTHY_TOOLS if with_tools else _HEALTHY_TEXT
    summary, replies, prov = _replay(tmp_path, monkeypatch, script, with_tools=with_tools)
    assert [r["turn"] for r in replies] == [0, "p1", 1, 2, 3, "p2"]
    assert replies[0]["prompt"] == crw.WARMUP_PROMPT
    assert replies[1]["prompt"] == crw.PLAIN_LEAD_PROMPT and replies[5]["prompt"] == crw.PLAIN_TRAIL_PROMPT
    assert prov.calls[1]["user"].strip().endswith(crw.PLAIN_LEAD_PROMPT)
    assert prov.calls[5]["user"].strip().endswith(crw.PLAIN_TRAIL_PROMPT)
    # tools were offered to the provider on the tools path only (the text-path wrapper strips them)
    assert all(c["tools"] is with_tools for c in prov.calls)
    assert summary["warmup_turn"] is True and summary["plain_turns"] is True
    assert summary["chat_rows_observed"] == 6 and summary["turns_requested"] == 3
    res = summary["cache_break_check"]
    # text path: all 5 later rows are plain; tools path: only the two plain micro-acks are single-call
    assert res["status"] == "PASS" and res["rows_checked"] == (2 if with_tools else 5)
    assert res["stable_read_tokens"] == 6449


def test_c17_no_plain_turns_and_no_warmup_flags(tmp_path, monkeypatch):
    summary, replies, _ = _replay(tmp_path, monkeypatch, _HEALTHY_TEXT, with_tools=False, plain=False)
    assert [r["turn"] for r in replies] == [0, 1, 2, 3]
    assert summary["plain_turns"] is False and summary["chat_rows_observed"] == 4
    summary, replies, _ = _replay(tmp_path, monkeypatch, _HEALTHY_TEXT, with_tools=False, warmup=False, plain=False)
    assert [r["turn"] for r in replies] == [1, 2, 3]
    assert crw.parse_args(["--no-plain-turns"]).no_plain_turns is True
    assert crw.parse_args([]).no_plain_turns is False
    assert crw.parse_args(["--no-warmup"]).no_warmup is True


def test_c17_the_plain_prompts_are_micro_acks_exempt_from_the_monologue_directive():
    # 'hi', 'thanks', 'ok' are the micro-acks named in brain/chat/monologue_prompts.py; pin them
    assert (crw.WARMUP_PROMPT, crw.PLAIN_LEAD_PROMPT, crw.PLAIN_TRAIL_PROMPT) == ("ok", "hi", "thanks")
    text = (REPO / "brain" / "chat" / "monologue_prompts.py").read_text(encoding="utf-8")
    for word in ("'hi'", "'thanks'", "'ok'"):
        assert word in text


def test_c17_a_break_that_begins_in_the_content_turns_is_seen_by_the_trailing_plain_turn(tmp_path, monkeypatch):
    summary, _, _ = _replay(tmp_path, monkeypatch, _BREAK_IN_CONTENT, with_tools=True)
    res = summary["cache_break_check"]
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == [6]  # the trailing plain row; content rows are multi-call


def test_c17_plain_turns_that_come_back_multi_call_are_unmeasured_never_a_pass(tmp_path, monkeypatch):
    script = [(17385, 0, 1), (12000, 6449, 3), (12500, 18543, 3), (12500, 18543, 3), (12500, 18543, 3), (12000, 6449, 2)]
    summary, _, _ = _replay(tmp_path, monkeypatch, script, with_tools=True)
    assert summary["cache_break_check"]["status"] == "UNMEASURED"


def test_c17_without_plain_turns_a_tools_path_run_is_unmeasured(tmp_path, monkeypatch):
    summary, _, _ = _replay(tmp_path, monkeypatch, _HEALTHY_TOOLS, with_tools=True, plain=False)
    assert summary["cache_break_check"]["status"] == "UNMEASURED"


def test_replay_prints_the_verdict_line_on_stderr(tmp_path, monkeypatch, capsys):
    summary, _, _ = _replay(tmp_path, monkeypatch, _HEALTHY_TEXT, with_tools=False)
    err = capsys.readouterr().err
    assert "cache_break_check: PASS (" in err
    assert crw.format_cache_break_line(summary["cache_break_check"]) in err


def _usage_file(pd):
    return pd / "chat_usage.jsonl"


def test_k2_an_old_corrupt_line_before_the_run_is_never_parsed_and_changes_nothing(tmp_path, monkeypatch, caplog):
    def before(pd):
        _usage_file(pd).write_text("{torn\n" + json.dumps(row(5, 5, call_type="generate")) + "\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="brain.health.jsonl_reader"):
        summary, _, _ = _replay(tmp_path, monkeypatch, _HEALTHY_TEXT, with_tools=False, before=before)
    assert summary["cache_break_check"]["status"] == "PASS"
    assert summary["chat_rows_observed"] == 6
    assert [r for r in caplog.records if "skipping" in r.getMessage()] == []  # the old line was never parsed


class _CorruptsAt3(_Scripted):
    def after_call(self, n):
        if n == 3:  # a torn line in the middle of the run's own rows
            with _usage_file(self.persona_dir).open("a", encoding="utf-8") as fh:
                fh.write('{"call_type": "chat", "cache_rea\n')


def test_k2_a_corrupt_line_inside_the_runs_rows_is_unmeasured_with_one_warning_from_one_parse(tmp_path, monkeypatch, caplog):
    import brain.health.jsonl_reader as reader

    calls = []
    real = reader.iter_jsonl_skipping_corrupt
    monkeypatch.setattr(reader, "iter_jsonl_skipping_corrupt", lambda path: (calls.append(Path(path)), real(path))[1])
    with caplog.at_level(logging.WARNING, logger="brain.health.jsonl_reader"):
        summary, _, _ = _replay(tmp_path, monkeypatch, _HEALTHY_TEXT, with_tools=False, cls=_CorruptsAt3)
    res = summary["cache_break_check"]
    assert res["status"] == "UNMEASURED"
    assert "skipped 1 corrupt or non-object line" in res["reason"]
    assert "byte offset 0" in res["reason"] and "number lines from 1 at that offset" in res["reason"]
    assert summary["chat_rows_observed"] == 6  # the means still come from the surviving rows
    assert len([r for r in caplog.records if "skipping malformed" in r.getMessage()]) == 1  # one bad line, one warning
    # the usage log was parsed once (from the run's bytes; the debug log had no new bytes); the engine's own
    # readers of other logs are not ours to count
    assert len([c for c in calls if c.name == "slice.jsonl"]) == 1
    assert not any(c.name == "chat_usage.jsonl" for c in calls)
    assert summary["c9_history_caching"]["last_turn_cache_creation"] is None  # attribution failed, run ends with a plain turn


def test_k2_the_old_file_ending_without_a_newline_does_not_hide_the_runs_first_row(tmp_path, monkeypatch):
    def before(pd):
        _usage_file(pd).write_text("{torn", encoding="utf-8")  # no newline: the first run row is glued to it in the file

    summary, _, _ = _replay(tmp_path, monkeypatch, _HEALTHY_TEXT, with_tools=False, before=before)
    assert summary["cache_break_check"]["status"] == "PASS" and summary["chat_rows_observed"] == 6


class _DebugWriter(_Scripted):
    def after_call(self, n):
        with (self.persona_dir / "cache_debug.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"call_type": "chat", "system_sha256": "run"}) + "\n")


def test_k2_the_debug_log_uses_the_same_slice_and_never_crashes_the_replay(tmp_path, monkeypatch):
    def before(pd):
        (pd / "cache_debug.jsonl").write_text(json.dumps({"call_type": "chat", "system_sha256": "old"}) + "\n", encoding="utf-8")

    summary, _, _ = _replay(tmp_path, monkeypatch, _HEALTHY_TEXT, with_tools=False, cls=_DebugWriter, before=before)
    assert summary["c1_system_byte_stability"] == {"available": True, "distinct_system_sha256": 1, "byte_stable": True, "sample": ["run"]}

    class _BadBytes(_Scripted):
        def after_call(self, n):
            with (self.persona_dir / "cache_debug.jsonl").open("ab") as fh:
                fh.write(b"\xff\xfe\n")

    summary, _, _ = _replay(tmp_path / "second", monkeypatch, _HEALTHY_TEXT, with_tools=False, cls=_BadBytes)
    c1 = summary["c1_system_byte_stability"]  # invalid UTF-8 in the debug slice: no debug rows, no crash, and C1 says why
    assert c1["available"] is False and "could not be read during the run" in c1["note"]
    assert summary["cache_break_check"]["status"] == "PASS"


class _TruncatesAt1(_Scripted):
    def after_call(self, n):
        if n == 1:
            _usage_file(self.persona_dir).write_text("", encoding="utf-8")


def test_k2_a_log_that_shrinks_during_the_run_is_unmeasured_for_every_provider(tmp_path, monkeypatch):
    big_old = "".join(json.dumps(row(5, 5, call_type="generate")) + "\n" for _ in range(60))  # far bigger than the run
    summary, _, _ = _replay(
        tmp_path, monkeypatch, _HEALTHY_TEXT, with_tools=False, cls=_TruncatesAt1,
        before=lambda pd: _usage_file(pd).write_text(big_old, encoding="utf-8"),
    )
    assert summary["cache_break_check"]["status"] == "UNMEASURED" and "shrank" in summary["cache_break_check"]["reason"]
    summary, _, _ = _replay(
        tmp_path / "fake", monkeypatch, _HEALTHY_TEXT, with_tools=False, cls=_TruncatesAt1, provider_name="fake",
        before=lambda pd: _usage_file(pd).write_text(big_old, encoding="utf-8"),
    )
    assert summary["cache_break_check"]["status"] == "UNMEASURED"  # never NOT_APPLICABLE


class _ShrinksBetweenTurns(_Scripted):
    def after_call(self, n):
        if n == 3:  # the file falls below the previous turn's boundary (rotation), then regrows
            _usage_file(self.persona_dir).write_text(json.dumps(row(1, 1)) + "\n", encoding="utf-8")


def test_k2_a_decreasing_turn_boundary_is_a_shrink(tmp_path, monkeypatch):
    summary, _, _ = _replay(tmp_path, monkeypatch, _HEALTHY_TEXT, with_tools=False, cls=_ShrinksBetweenTurns)
    assert summary["cache_break_check"]["status"] == "UNMEASURED" and "shrank" in summary["cache_break_check"]["reason"]


class _WritesOnlyATornLine(_Scripted):
    def chat(self, messages, *, tools=None, options=None):
        with _usage_file(self.persona_dir).open("a", encoding="utf-8") as fh:
            fh.write("{torn\n")
        return self._fake.chat(messages, tools=tools, options=options)


def test_k2_a_skipped_line_beats_not_applicable(tmp_path, monkeypatch):
    for provider in ("fake", "ollama"):
        summary, _, _ = _replay(
            tmp_path / provider, monkeypatch, _HEALTHY_TEXT, with_tools=False, cls=_WritesOnlyATornLine, provider_name=provider
        )
        assert summary["cache_break_check"]["status"] == "UNMEASURED", provider


def test_k2_read_run_slice_unit_cases(tmp_path):
    f = tmp_path / "u.jsonl"
    lines = [json.dumps(row(100 + k, 6449, t=k)) + "\n" for k in range(3)]
    old = "caf\u00e9 \u2713 old part\r\n".encode("utf-8") + b"\xff\xfe not utf-8\n"  # multibyte, CRLF, invalid UTF-8: never parsed
    f.write_bytes(old + "".join(lines).encode())
    off = len(old)
    sizes = [off + len(lines[0]), off + len(lines[0]) + len(lines[1]), off + sum(len(x) for x in lines)]
    rows, skipped, counts, consistent = crw._read_run_slice(f, off, sizes)
    assert (len(rows), skipped, counts, consistent) == (3, 0, [1, 1, 1], True)
    # a slice starting in the middle of a line: that line is skipped, never judged
    rows, skipped, _, consistent = crw._read_run_slice(f, off + 5, sizes)
    assert (len(rows), skipped, consistent) == (2, 1, False)
    # a turn boundary inside a line, bytes after the last boundary, counts that do not add up: not attributable
    assert crw._read_run_slice(f, off, [off + 10, sizes[1], sizes[2]])[3] is False
    assert crw._read_run_slice(f, off, sizes[:2])[3] is False
    # a decreasing boundary or a size below the offset is a shrink; a missing file with an offset too
    with pytest.raises(ValueError):
        crw._read_run_slice(f, off, [sizes[1], sizes[0]])
    with pytest.raises(ValueError):
        crw._read_run_slice(f, off + 10_000, [])
    with pytest.raises(ValueError):
        crw._read_run_slice(tmp_path / "gone.jsonl", 5, [])
    assert crw._read_run_slice(tmp_path / "none.jsonl", 0, [0]) == ([], 0, [0], True)  # an empty slice is consistent
    # invalid UTF-8 inside the run's own bytes is a ValueError (the replay turns it into UNMEASURED)
    g = tmp_path / "g.jsonl"
    g.write_bytes(b"\xff\xfe\n")
    with pytest.raises(ValueError):
        crw._read_run_slice(g, 0, [3])


class _BadBytesInRun(_Scripted):
    def after_call(self, n):
        if n == 2:
            with _usage_file(self.persona_dir).open("ab") as fh:
                fh.write(b"\xff\xfe\n")


def test_k2_invalid_utf8_inside_the_runs_rows_is_unmeasured_and_the_replay_still_returns(tmp_path, monkeypatch):
    summary, _, _ = _replay(tmp_path, monkeypatch, _HEALTHY_TEXT, with_tools=False, cls=_BadBytesInRun)
    assert summary["cache_break_check"]["status"] == "UNMEASURED"
    assert summary["chat_rows_observed"] == 0  # nothing could be parsed from the slice


def test_replay_with_a_real_provider_that_logs_no_usage_is_unmeasured_not_a_crash(tmp_path, monkeypatch):
    # a claude-cli replay whose provider wrote no usage rows (e.g. a broken log path): UNMEASURED, never N/A
    import brain.bridge.provider as provider_mod

    monkeypatch.setenv("NELL_CACHE_DEBUG", "0")
    pd = tmp_path / "personas" / "replay"
    crw._seed_scratch_persona(pd)
    monkeypatch.setattr(provider_mod, "get_provider", lambda name, *, persona_dir=None, model_override=None: provider_mod.FakeProvider())
    summary, _ = crw.run_replay(pd, turns=2, gap_s=0, provider_name="claude-cli", seed=False, force_text_path=True)
    assert summary["cache_break_check"]["status"] == "UNMEASURED" and summary["chat_rows_observed"] == 0
    assert crw._read_rows_fail_closed(tmp_path / "missing.jsonl") == []


@pytest.mark.parametrize("provider", ["fake", "ollama"])
def test_k3_a_provider_that_never_logs_usage_is_not_applicable_and_exits_0(tmp_path, monkeypatch, capsys, provider):
    import brain.bridge.provider as provider_mod

    monkeypatch.setenv("NELL_CACHE_DEBUG", "0")
    _scratch_under(monkeypatch, tmp_path)
    monkeypatch.setattr(provider_mod, "get_provider", lambda name, *, persona_dir=None, model_override=None: provider_mod.FakeProvider())
    out = tmp_path / "metrics.json"
    rc = crw.main(["--scratch", "--turns", "2", "--gap-s", "0", "--provider", provider, "--out", str(out)])
    cap = capsys.readouterr()
    assert rc == 0
    written = json.loads(out.read_text(encoding="utf-8"))  # the JSON is written before the exit code is decided
    assert written["cache_break_check"]["status"] == "NOT_APPLICABLE"
    lines = [ln for ln in cap.err.splitlines() if ln.startswith("cache_break_check:")]
    assert len(lines) == 1 and "not applicable" in lines[0] and repr(provider) in lines[0]


def test_k3_a_real_provider_with_zero_usage_rows_exits_11(tmp_path, monkeypatch, capsys):
    import brain.bridge.provider as provider_mod

    monkeypatch.setenv("NELL_CACHE_DEBUG", "0")
    _scratch_under(monkeypatch, tmp_path)
    monkeypatch.setattr(provider_mod, "get_provider", lambda name, *, persona_dir=None, model_override=None: provider_mod.FakeProvider())
    rc = crw.main(["--scratch", "--turns", "2", "--gap-s", "0", "--out", str(tmp_path / "m.json")])
    assert rc == 11
    assert json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))["cache_break_check"]["status"] == "UNMEASURED"
    capsys.readouterr()


def test_k3_the_non_logging_provider_set_matches_the_provider_source():
    src = (REPO / "brain" / "bridge" / "provider.py").read_text(encoding="utf-8")
    classes = {n.name: n for n in ast.parse(src).body if isinstance(n, ast.ClassDef)}

    def logs(cls):
        return any(isinstance(n, ast.Name) and n.id == "log_usage" for n in ast.walk(classes[cls]))

    assert logs("ClaudeCliProvider")
    assert not logs("FakeProvider") and not logs("OllamaProvider")
    assert crw._NON_LOGGING_PROVIDERS == frozenset({"fake", "ollama"})


# --------------------------------------------------------------------------------------
# K4: exact turn attribution (last content turn) while the means stay over all rows
# --------------------------------------------------------------------------------------


def _layout_script(warm=True, plain=True, content_entries=None, p2=None):
    """One script entry per call in order: warm-up, 'hi', 3 content turns, 'thanks' (flags drop the extras)."""
    content = content_entries or [(12001, 6449, 1), (12002, 6449, 1), (12003, 6449, 1)]
    out = []
    if warm:
        out.append((17385, 0, 1))
    if plain:
        out.append((11000, 6449, 1))
    out += content
    if plain:
        out.append((11999, 6449, 1) if p2 is None else p2)
    return out


@pytest.mark.parametrize("warm", [True, False])
@pytest.mark.parametrize("plain", [True, False])
def test_k4_last_turn_cache_creation_is_the_last_content_turn_for_every_flag_combination(tmp_path, monkeypatch, warm, plain):
    script = _layout_script(warm, plain)
    summary, replies, _ = _replay(tmp_path, monkeypatch, script, with_tools=False, warmup=warm, plain=plain)
    expected_ids = ([0] if warm else []) + (["p1"] if plain else []) + [1, 2, 3] + (["p2"] if plain else [])
    assert [r["turn"] for r in replies] == expected_ids
    assert summary["c8_cache"]["row_turns"] == expected_ids
    assert summary["c9_history_caching"]["last_turn_cache_creation"] == 12003.0
    creation = [float(c) for c, _r, _n in script]
    assert summary["c8_cache"]["cache_creation_series"] == creation  # means/series stay over ALL chat rows
    assert summary["c8_cache"]["mean_cache_creation"] == round(sum(creation) / len(creation), 1)
    assert "all chat rows of the run" in summary["c8_cache"].get("rows_include", "")
    assert summary["chat_rows_observed"] == len(script)


def test_k4_attribution_follows_the_byte_boundaries_not_positions(tmp_path, monkeypatch):
    # a content turn logging two chat rows: the last chat row of the last content turn counts
    two_rows = [(12003, 6449, 1), (12099, 6449, 1)]
    summary, _, _ = _replay(tmp_path / "a", monkeypatch, _layout_script(content_entries=[(12001, 6449, 1), (12002, 6449, 1), two_rows]), with_tools=False)
    assert summary["c9_history_caching"]["last_turn_cache_creation"] == 12099.0
    assert summary["c8_cache"]["row_turns"] == [0, "p1", 1, 2, 3, 3, "p2"]
    # a trailing plain turn logging two rows, or none: a position-based guess would be wrong in both
    summary, _, _ = _replay(tmp_path / "b", monkeypatch, _layout_script(p2=[(11999, 6449, 1), (11998, 6449, 1)]), with_tools=False)
    assert summary["c9_history_caching"]["last_turn_cache_creation"] == 12003.0
    assert summary["c8_cache"]["row_turns"] == [0, "p1", 1, 2, 3, "p2", "p2"]
    summary, _, _ = _replay(tmp_path / "c", monkeypatch, _layout_script(p2=[]), with_tools=False)
    assert summary["c9_history_caching"]["last_turn_cache_creation"] == 12003.0
    assert summary["c8_cache"]["row_turns"] == [0, "p1", 1, 2, 3]
    # a content turn that also logs a non-chat row: the row is attributed to its turn and dropped as non-chat
    mixed = [(12003, 6449, 1), (1, 1, 1, "generate")]
    summary, _, _ = _replay(tmp_path / "d", monkeypatch, _layout_script(content_entries=[(12001, 6449, 1), mixed, (12002, 6449, 1)]), with_tools=False)
    assert summary["c9_history_caching"]["last_turn_cache_creation"] == 12002.0
    assert summary["c8_cache"]["row_turns"] == [0, "p1", 1, 2, 3, "p2"]
    assert summary["chat_rows_observed"] == 6


def test_k4_without_attribution_the_last_turn_is_unknown_when_a_plain_turn_ends_the_run():
    rows = run(6449, 6449, 6449)
    s = crw._summarise(rows, [], turns=3, row_turns=None, trailing_plain=True)
    assert s["c9_history_caching"]["last_turn_cache_creation"] is None and "unavailable" in s["c9_history_caching"]["last_turn_note"]
    s = crw._summarise(rows, [], turns=3, row_turns=None, trailing_plain=False)
    assert s["c9_history_caching"]["last_turn_cache_creation"] == 12000.0
    s = crw._summarise(rows, [], turns=3, row_turns=[0, 1, 2, 3])
    assert s["c9_history_caching"]["last_turn_cache_creation"] == 12000.0 and s["c8_cache"]["row_turns"] == [0, 1, 2, 3]


def test_k4_the_summary_records_which_path_it_measured(tmp_path, monkeypatch):
    s1, _, _ = _replay(tmp_path / "t", monkeypatch, _HEALTHY_TEXT, with_tools=False)
    s2, _, _ = _replay(tmp_path / "w", monkeypatch, _HEALTHY_TOOLS, with_tools=True)
    assert s1["with_tools"] is False and s2["with_tools"] is True


def test_documented_limit_a_transient_miss_inside_the_content_turns_is_not_seen(tmp_path, monkeypatch):
    # the cache is cold for two content turns and healthy again by the trailing plain turn: PASS, judged 2
    script = [(17385, 0, 1), (12000, 6449, 1), (17000, 0, 3), (17000, 0, 3), (12500, 18543, 3), (12000, 6449, 1)]
    summary, _, _ = _replay(tmp_path, monkeypatch, script, with_tools=True)
    assert summary["cache_break_check"]["status"] == "PASS" and summary["cache_break_check"]["rows_checked"] == 2


def test_no_warmup_with_plain_turns_on_the_tools_path_is_unmeasured_and_the_help_says_so(tmp_path, monkeypatch):
    # the lead micro-ack becomes the never-judged first row: one judgeable row is left
    script = [(17385, 0, 1), (12500, 18543, 3), (12500, 18543, 3), (12500, 18543, 3), (12000, 6449, 1)]
    summary, _, _ = _replay(tmp_path, monkeypatch, script, with_tools=True, warmup=False)
    assert summary["cache_break_check"]["status"] == "UNMEASURED" and summary["cache_break_check"]["rows_checked"] == 1
    helps = {a.option_strings[0]: a.help for a in _parser()._actions if a.option_strings}
    assert "use together with --no-plain-turns" in " ".join(helps["--no-warmup"].split())


def test_the_fail_reason_names_each_break_against_the_s_it_was_compared_with():
    res = check(run(6449, 3000, 9000, 2000))
    assert res["status"] == "FAIL"
    assert "row 3 read 3000 vs S=6449" in res["reason"] and "row 5 read 2000 vs S=3000" in res["reason"]
    assert "S=None" not in check(run(0, 0, 0))["reason"] and "with no S yet" in check(run(0, 0, 0))["reason"]


def test_c7_a_missing_pyyaml_is_a_value_error_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)  # `import yaml` raises ImportError
    with pytest.raises(ValueError):
        crw.read_regression_threshold(CONFIG)


def test_warmup_prompt_is_the_micro_ack():
    assert crw.WARMUP_PROMPT == "ok"


# --------------------------------------------------------------------------------------
# C8: replay exit codes (0 PASS / 10 FAIL / 11 UNMEASURED), through main()
# --------------------------------------------------------------------------------------


def _scratch_under(monkeypatch, tmp_path):
    import tempfile

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # tempfile caches the temp dir; TMPDIR alone is too late


def _main_replay(tmp_path, monkeypatch, script, capsys):
    _patch_provider(monkeypatch, tmp_path, script)
    _scratch_under(monkeypatch, tmp_path)
    out = tmp_path / "metrics.json"
    rc = crw.main(["--scratch", "--turns", "3", "--gap-s", "0", "--out", str(out)])
    cap = capsys.readouterr()
    return rc, out, cap


@pytest.mark.parametrize(
    "script,expected,status",
    [(_HEALTHY_TOOLS, 0, "PASS"), (_BREAK_IN_CONTENT, 10, "FAIL"), ([(17385, 0, 1)] + [(12000, 6449, 3)] * 5, 11, "UNMEASURED")],
)
def test_c8_replay_main_exit_code_follows_the_verdict_after_the_json_is_written(
    tmp_path, monkeypatch, capsys, script, expected, status
):
    rc, out, cap = _main_replay(tmp_path, monkeypatch, script, capsys)
    assert rc == expected
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["cache_break_check"]["status"] == status
    assert json.loads(cap.out)["cache_break_check"]["status"] == status  # printed as well
    assert f"cache_break_check: {status}" in cap.err


def test_c8_a_summary_without_the_key_or_with_an_unknown_status_is_exit_11(tmp_path, monkeypatch):
    for summary in ({}, {"cache_break_check": {"status": "WEIRD"}}, {"cache_break_check": None}):
        monkeypatch.setattr(crw, "run_replay", lambda *a, _s=summary, **k: (_s, []))
        assert crw.main(["--persona-dir", str(tmp_path)]) == 11


def test_c8_exit_code_mapping():
    assert (crw.EXIT_PASS, crw.EXIT_FAIL, crw.EXIT_UNMEASURED) == (0, 10, 11)
    assert [crw._exit_code_for({"status": s}) for s in ("PASS", "NOT_APPLICABLE", "FAIL", "UNMEASURED")] == [0, 0, 10, 11]
    assert crw._exit_code_for(None) == 11 and crw._exit_code_for("PASS") == 11
    # no verdict ever collides with the crash code (1) or the usage-error code (2)
    assert all(crw._exit_code_for({"status": s}) not in (1, 2) for s in ("PASS", "NOT_APPLICABLE", "FAIL", "UNMEASURED", "X"))


def test_c8_help_states_the_exit_codes_and_the_error_codes():
    doc = " ".join((crw.__doc__ or "").split())
    assert "0 PASS (also NOT_APPLICABLE" in doc and "10 FAIL, 11 UNMEASURED" in doc
    assert "Errors never use 10 or 11: 1 = an uncaught crash, ``--turns`` below 1, ``--dump-replies`` without ``--out``" in doc
    assert "2 = argparse usage errors, no persona dir given, and bad ``--cache-break-check`` arguments" in doc
    assert "``--compare`` returns 0 / 10 / 11" in doc
    assert "uv run python scripts/cache_replay_workload.py" in doc
    helps = {a.option_strings[0]: a.help for a in _parser()._actions if a.option_strings}
    assert "exit 0 PASS / 10 FAIL / 11 UNMEASURED" in helps["--cache-break-check"] and "2 bad arguments" in helps["--cache-break-check"]


# --------------------------------------------------------------------------------------
# C11 / K4: compare() diagnostics go to stderr, stdout says "not comparable", verdicts are shown
# --------------------------------------------------------------------------------------


def _pair(tmp_path, **new_extra):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    _write_summary(old, [1000.0, 1000.0], [2000.0, 2000.0])
    _write_summary(new, [500.0, 500.0], [2500.0, 2500.0], **new_extra)
    return old, new


def test_c11_the_warmup_warning_is_on_stderr_not_stdout_and_names_both_verdicts(tmp_path, capsys):
    old, new = _pair(
        tmp_path,
        warmup_turn=True,
        cache_break_check={"status": "FAIL", "reason": "1 judged row(s) read below the run's stable read S=6449"},
    )
    crw.compare(old, new)
    cap = capsys.readouterr()
    assert "WARNING" not in cap.out and "warmup_turn" not in cap.out
    assert "WARNING: warmup_turn differs (old=False, new=True)" in cap.err
    assert "cache_break_check: old n/a, new FAIL" in cap.err
    assert "new cache_break_check: FAIL (1 judged row(s) read below" in cap.err
    assert "old cache_break_check: n/a (run predates the check)" in cap.err


def test_c11_the_plain_turns_difference_warns_on_stderr(tmp_path, capsys):
    old, new = _pair(tmp_path, warmup_turn=False, plain_turns=True)
    crw.compare(old, new)
    cap = capsys.readouterr()
    assert "WARNING: plain_turns differs (old=False, new=True)" in cap.err and "WARNING" not in cap.out


def test_c11_no_warning_when_the_settings_match_but_each_verdict_is_still_shown(tmp_path, capsys):
    old, new = _pair(tmp_path, warmup_turn=False, plain_turns=False)
    assert crw.compare(old, new) == 0
    cap = capsys.readouterr()
    assert "WARNING" not in cap.err and "WARNING" not in cap.out
    assert "old cache_break_check: n/a" in cap.err and "new cache_break_check: n/a" in cap.err
    assert "mean cache_creation/turn: 1000 → 500  (-50%)" in cap.out  # stdout is still the A/B report only
    assert "NOT COMPARABLE" not in cap.out


@pytest.mark.parametrize(
    "new_extra",
    [
        {"warmup_turn": True},
        {"plain_turns": True},
        {"warmup_turn": True, "plain_turns": True},
        {"warmup_turn": False, "plain_turns": False, "turns_requested": 9},  # old arm records 2 (len of the series)
    ],
)
def test_k4_arms_that_differ_are_not_comparable_on_stdout_with_no_deltas_and_no_c8_verdict(tmp_path, capsys, new_extra):
    old, new = _pair(tmp_path, **new_extra)
    rc = crw.compare(old, new)
    cap = capsys.readouterr()
    assert rc == 11
    assert "NOT COMPARABLE" in cap.out
    assert not re.search(r"\([+-]\d+%\)", cap.out)  # no percentage delta
    assert not re.search(r"\d+ → \d+", cap.out)  # and no old → new delta lines
    assert "C8 (system-block cache stops re-creating): not comparable (no verdict)" in cap.out
    assert "PASS" not in cap.out.split("C1")[0] and "FAIL" not in cap.out.split("C1")[0]
    assert "mean cache_creation/row:  old 1000, new 500  (no delta: not comparable)" in cap.out
    assert "old arm per-row series" in cap.out and "new arm per-row series" in cap.out  # each arm on its own, not paired


def test_k4_with_tools_difference_makes_arms_not_comparable_and_an_absent_key_is_ignored(tmp_path, capsys):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    _write_summary(old, [1000.0, 1000.0], [2000.0, 2000.0], with_tools=False, turns_requested=2)
    _write_summary(new, [500.0, 500.0], [2500.0, 2500.0], with_tools=True, turns_requested=2)
    assert crw.compare(old, new) == 11 and "with_tools differs" in capsys.readouterr().out
    _write_summary(old, [1000.0, 1000.0], [2000.0, 2000.0])  # an older arm: no with_tools key
    assert crw.compare(old, new) == 0
    capsys.readouterr()


def test_k4_an_arm_from_an_older_script_is_comparable_only_with_a_new_arm_that_sent_no_extra_turns(tmp_path, capsys):
    old, new = _pair(tmp_path, warmup_turn=False, plain_turns=False, cache_break_check={"status": "PASS", "reason": "r"})
    assert crw.compare(old, new) == 0
    out = capsys.readouterr().out
    assert "the old arm predates the cache-break check" in out and "NOT COMPARABLE" not in out
    old, new = _pair(tmp_path, warmup_turn=True, plain_turns=True, cache_break_check={"status": "PASS", "reason": "r"})
    assert crw.compare(old, new) == 11  # the new arm used the extra turns, the old script never sent them
    assert "NOT COMPARABLE" in capsys.readouterr().out


def test_k4_the_c9_line_labels_an_arm_whose_last_row_may_be_the_trailing_micro_ack(tmp_path, capsys):
    old, new = _pair(tmp_path, warmup_turn=True, plain_turns=True)
    _write_summary(old, [1000.0, 1000.0], [2000.0, 2000.0], warmup_turn=True, plain_turns=True)  # no row_turns key
    new_js = json.loads(new.read_text())
    new_js["c8_cache"]["row_turns"] = [0, "p1", 1, "p2"]
    new.write_text(json.dumps(new_js))
    assert crw.compare(old, new) == 0
    out = capsys.readouterr().out
    assert "old last turn: 1000.0 (last row: includes the trailing micro-ack)" in out
    assert "new last turn: 500.0\n" in out


def test_k4_a_not_applicable_arm_is_shown_and_compared_by_its_flags(tmp_path, capsys):
    old, new = _pair(tmp_path, warmup_turn=False, plain_turns=False, cache_break_check={"status": "NOT_APPLICABLE", "reason": "provider 'fake' never logs usage rows"})
    assert crw.compare(old, new) == 0
    assert "new cache_break_check: NOT_APPLICABLE (provider 'fake'" in capsys.readouterr().err


def test_k4_the_c9_lines_of_a_not_comparable_pair_are_marked_reference_only(tmp_path, capsys):
    old, new = _pair(tmp_path, warmup_turn=True)
    crw.compare(old, new)
    out = capsys.readouterr().out
    assert "last-turn cache_creation of each arm, for reference only (arms not comparable):" in out
    old, new = _pair(tmp_path, warmup_turn=False)
    crw.compare(old, new)
    assert "compare last-turn cache_creation:" in capsys.readouterr().out


def test_k4_the_not_comparable_output_labels_each_arms_rows_by_turn_id_when_known(tmp_path, capsys):
    old, new = _pair(tmp_path, warmup_turn=True)
    new_js = json.loads(new.read_text())
    new_js["c8_cache"]["row_turns"] = [0, 1]
    new.write_text(json.dumps(new_js))
    crw.compare(old, new)
    out = capsys.readouterr().out
    assert "new arm per-row series (create / read), rows labelled by turn id when known:\n  0:500/2500  1:500/2500" in out
    assert "old arm per-row series (create / read), rows labelled by turn id when known:\n  1:1000/2000  2:1000/2000" in out


# --------------------------------------------------------------------------------------
# Error exits, I/O failures, newline handling in the run's own bytes
# --------------------------------------------------------------------------------------


def test_k3_the_pre_run_error_codes_are_not_verdict_codes(tmp_path, capsys):
    assert crw.main(["--persona-dir", str(tmp_path), "--turns", "0"]) == 1
    assert crw.main(["--persona-dir", str(tmp_path), "--dump-replies"]) == 1  # --dump-replies needs --out
    assert crw.main([]) == 2  # no persona dir: refusing to run against a live persona
    capsys.readouterr()


def test_k3_a_malformed_compare_input_is_a_crash_not_a_verdict(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    res = _script_run("--compare", str(bad), str(bad))
    assert res.returncode == 1 and "JSONDecodeError" in res.stderr


def test_k2_an_unwritable_temp_dir_is_unmeasured_not_a_late_crash(tmp_path, monkeypatch):
    import tempfile

    def boom(*a, **k):
        raise PermissionError("no space for the slice copy")

    monkeypatch.setattr(tempfile, "TemporaryDirectory", boom)
    summary, _, _ = _replay(tmp_path, monkeypatch, _HEALTHY_TEXT, with_tools=False)  # a crash here fails the assertion
    res = summary["cache_break_check"]
    assert res["status"] == "UNMEASURED" and "no space for the slice copy" in res["reason"]


def test_k2_a_final_size_below_the_last_turn_boundary_is_a_shrink(tmp_path):
    f = tmp_path / "u.jsonl"
    line = json.dumps(row(1, 1)) + "\n"
    f.write_text(line * 3, encoding="utf-8")
    n = len(line)
    with pytest.raises(ValueError):
        crw._read_run_slice(f, n, [2 * n, 5 * n])  # the file is 3 lines long but a boundary says 5 were there


def test_k2_a_mid_line_boundary_cannot_cancel_out_bytes_after_the_last_boundary(tmp_path):
    f = tmp_path / "u.jsonl"
    lines = [json.dumps(row(100 + k, 6449, t=k)) + "\n" for k in range(3)]
    f.write_text("".join(lines), encoding="utf-8")
    boundaries = [4, len(lines[0]) + len(lines[1])]  # the first boundary is inside line 0, line 2 lies after the last one
    rows, skipped, counts, consistent = crw._read_run_slice(f, 0, boundaries)
    assert (len(rows), skipped, counts) == (3, 0, [1, 2])  # the counts add up to the rows ...
    assert consistent is False  # ... but the last boundary is not the end of the file, so no attribution


@pytest.mark.parametrize("eol", ["\r\n", "\r", "\n"])
def test_k2_line_endings_inside_the_runs_own_bytes_are_counted_like_the_reader_does(tmp_path, eol):
    f = tmp_path / "u.jsonl"
    lines = [json.dumps(row(100 + k, 6449, t=k)) + eol for k in range(3)]
    blob = "".join(lines).encode()
    f.write_bytes(blob)
    sizes = [len(lines[0]), len(lines[0]) + len(lines[1]), len(blob)]
    rows, skipped, counts, consistent = crw._read_run_slice(f, 0, sizes)
    assert (len(rows), skipped, counts, consistent) == (3, 0, [1, 1, 1], True)


# --------------------------------------------------------------------------------------
# K5: PyYAML is a declared dependency
# --------------------------------------------------------------------------------------


def test_k5_pyyaml_is_declared_and_locked_as_a_direct_dependency():
    import tomllib

    py = tomllib.loads((LOCKROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert any(re.match(r"(?i)pyyaml\s*>=\s*6", d) for d in py["project"]["dependencies"])
    lock = tomllib.loads((LOCKROOT / "uv.lock").read_text(encoding="utf-8"))
    pkg = next(p for p in lock["package"] if p["name"] == py["project"]["name"])
    assert {"name": "pyyaml"} in pkg["dependencies"]
    assert any(r["name"] == "pyyaml" and r["specifier"] == ">=6.0" for r in pkg["metadata"]["requires-dist"])
