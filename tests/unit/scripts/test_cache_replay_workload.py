"""scripts/cache_replay_workload.py - the per-run cache-break check (#339, reworked for #340).

`check_cache_break` is a pure function over `chat_usage.jsonl` rows. A cache break is a later PLAIN
(num_turns == 1) chat row whose cache_read falls below the run's stable read S (T = 0 in the config:
read 0 or any read below S). The first chat row (the cold write) is never judged; multi-call rows are
not judged and impose no rule; the run needs at least MIN_JUDGED_ROWS (2) judged rows (one establishes S,
one is compared against it), else UNMEASURED. No call counts, no model calls, no OLD-vs-NEW comparison.

Row shapes here are SYNTHETIC (owner ruling: a mechanical check, tested without model calls). Rows are
written (creation, read) as in the #340 review. Shapes taken from a real measurement are named in comments.
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
CONFIG = REPO / "guarded-change.companion.md"

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
    assert s["c8_cache"] == {
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
    assert crw.compare(old, new) == 1
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


def test_c3_growing_reads_pass_because_s_is_a_running_max():
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


def test_c3_s_is_the_running_max_so_small_steps_down_cannot_drift():
    # each step is within 10% of the previous row but the last is 16% below the highest read: S must not follow it
    assert check(run(1000, 920, 840), 10)["status"] == "FAIL"
    assert check(run(1000, 920, 910), 10)["status"] == "PASS"


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


def test_c4_a_persisting_lower_level_is_flagged_on_every_row_because_breaks_do_not_lower_s():
    res = check(run(6449, 3000, 3000, 3000))
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == [3, 4, 5]


def test_c4_fail_wins_over_a_malformed_row():
    rows = run(6449, 6449, 2000)
    rows.insert(2, row(12000, None, t=9))
    assert check(rows)["status"] == "FAIL"


def test_c4_a_slice_spanning_a_prompt_change_fails_naming_row_and_gap():
    # documented limit: a slice must be ONE run; S from the old prefix, every row after the change breaks
    res = check(run(15476, 15476, 6176, 6176))
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == [4, 5]
    assert all(b["ts"] and b["gap_s"] is not None for b in res["breaks"])


def test_c4_documented_limit_a_persistent_partial_break_passes_by_construction():
    # later plain reads constant at 3000 from the first judged row (a tools block still readable behind
    # a churning system block): S follows the level. Left to cache_creation / cost metrics (config, limit 1).
    res = check(run(3000, 3000, 3000))
    assert res["status"] == "PASS" and res["stable_read_tokens"] == 3000


def test_c4_stated_failure_mode_an_outlier_read_fails_the_following_healthy_rows():
    res = check(run(6449, 12094, 6449, 6449))
    assert res["status"] == "FAIL"
    assert [b["row"] for b in res["breaks"]] == [4, 5]


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
    assert _cli(f, "--regression-threshold", "0") == 1
    _write_rows(f, run(6449))  # one judged row only
    assert _cli(f, "--regression-threshold", "0") == 2


def test_cli_reads_the_threshold_from_the_config_by_default(tmp_path, capsys):
    f = tmp_path / "u.jsonl"
    _write_rows(f, run(1000, 999))  # a 0.1% drop: a break only at T = 0 (the real config's value)
    assert _cli(f) == 1
    assert _cli(f, "--regression-threshold", "10") == 0
    capsys.readouterr()


def test_cli_from_row_is_one_based_over_chat_rows(tmp_path):
    f = tmp_path / "u.jsonl"
    prev = run(6449, 6449)  # a previous run
    cur = run(15476, 15476, 3000)  # the run under test: its last row breaks
    rows = [row(5, 5, call_type="generate")] + prev + [row(5, 5, call_type="generate")] + cur
    _write_rows(f, rows)
    assert _cli(f, "--from-row", "4", "--regression-threshold", "0") == 1  # the current run alone
    assert _cli(f, "--from-row", "1", "--regression-threshold", "0") == 1  # whole file: 15476 then 3000 breaks too
    assert _cli(f, "--from-row", "6", "--regression-threshold", "0") == 2  # two chat rows left: 1 judged row
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
    assert _cli(f, "--from-row", "1", "--regression-threshold", "0") == 1
    # chat row 2 as the run's first row leaves judged rows 3 and 4 (6449, 100): still a break
    assert _cli(f, "--from-row", "2", "--regression-threshold", "0") == 1
    # chat row 3 as first row leaves one later row: UNMEASURED (file-line indexing would give another verdict)
    assert _cli(f, "--from-row", "3", "--regression-threshold", "0") == 2


def test_cli_refuses_corrupt_lines_instead_of_skipping_them(tmp_path, capsys):
    f = tmp_path / "u.jsonl"
    good = _good_lines(6449, 6449, 2000)  # the break is the last row
    torn = '{"call_type": "chat", "cache_creation_input_t'
    for lines in (good[:3] + [torn] + good[3:], good[:3] + [torn], [torn] + good):
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert _cli(f, "--regression-threshold", "0") == 2
        assert "skipped 1 corrupt or non-object line" in capsys.readouterr().err
    # valid JSON that is not an object
    f.write_text("[1, 2]\n" + "\n".join(good) + "\n", encoding="utf-8")
    assert _cli(f, "--regression-threshold", "0") == 2
    # the break row torn instead of present: skipping it would give a PASS from the remaining rows
    f.write_text("\n".join(good[:3] + ['{"call_type": "chat", "cache_read_inp']) + "\n", encoding="utf-8")
    assert _cli(f, "--regression-threshold", "0") == 2
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
        assert _cli(f, "--regression-threshold", "0") == 2
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
    assert not hasattr(crw, "_read_jsonl_strict")
    calls.clear()
    assert crw._read_jsonl(f) and calls == [f]  # the replay's lenient read uses it as well


def test_cli_any_unexpected_exception_is_exit_2(tmp_path, monkeypatch, capsys):
    f = tmp_path / "u.jsonl"
    _write_rows(f, run(6449, 6449))

    def boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(crw, "check_cache_break", boom)
    assert _cli(f, "--regression-threshold", "0") == 2
    assert "unexpected" in capsys.readouterr().err


def test_cli_errors_are_exit_2_never_1(tmp_path, capsys):
    assert _cli(tmp_path / "missing.jsonl") == 2
    f = tmp_path / "u.jsonl"
    f.write_text("not json\n", encoding="utf-8")
    assert _cli(f) == 2
    capsys.readouterr()
    _write_rows(f, run(6449, 6449))
    assert _cli(f, "--regression-threshold", "150") == 2


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


@pytest.mark.parametrize(
    "phrase",
    [
        "never judged",
        "Only single-call rows (num_turns == 1) are judged",
        "highest read among the earlier judged rows (running max)",
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
        "Six exact first-call reads were all 6449 tokens",
        "15,476",
        "OBSERVED, not bounded",
        "a transient miss included",
        "re-measure",
        "Basis of S = running max",
        "one outlier (higher) read",
        "Basis of the minimum of 2 judged rows (derived, not tuned)",
        "UNMEASURED (exit 2) = an unverified gating criterion",
        "persistent PARTIAL break",
        "must be ONE run",
        "num_turns == 1 meaning one API call is an assumption",
        "uv run python scripts/cache_replay_workload.py --cache-break-check",
        "no OLD/NEW baseline, call count or logging change is needed",
        "exit 0 PASS / 1 FAIL / 2 UNMEASURED",
        "complementary to the creation and cost metrics",
    ],
)
def test_c12_config_entry_states_the_new_meaning(phrase):
    entry = _entry_text(CONFIG.read_text(encoding="utf-8"))
    assert " ".join(phrase.split()) in entry, phrase


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


@pytest.mark.parametrize("pattern", _STALE)
def test_c12_stale_first_row_cold_text_is_gone(pattern):
    entry = _entry_text(CONFIG.read_text(encoding="utf-8"))
    script = " ".join(SCRIPT.read_text(encoding="utf-8").split())
    assert not re.search(pattern, entry), ("config", pattern)
    assert not re.search(pattern, script), ("script", pattern)


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


def _patch_provider(monkeypatch, tmp_path, script):
    import brain.bridge.provider as provider_mod

    monkeypatch.setenv("NELL_CACHE_DEBUG", "0")  # run_replay sets it to 1; monkeypatch restores the original
    pd = tmp_path / "personas" / "replay"
    crw._seed_scratch_persona(pd)
    holder = {}

    def fake_get_provider(name, *, persona_dir=None, model_override=None):
        holder["p"] = _Scripted(persona_dir, script)
        return holder["p"]

    monkeypatch.setattr(provider_mod, "get_provider", fake_get_provider)
    return pd, holder


def _replay(tmp_path, monkeypatch, script, *, with_tools, warmup=True, plain=True, turns=3):
    pd, holder = _patch_provider(monkeypatch, tmp_path, script)
    summary, replies = crw.run_replay(
        pd,
        turns=turns,
        gap_s=0,
        provider_name="claude-cli",
        seed=False,
        force_text_path=not with_tools,
        warmup=warmup,
        plain_turns=plain,
    )
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


def test_replay_verdict_slice_is_fail_closed_while_the_means_stay_lenient(tmp_path, monkeypatch):
    pd, _ = _patch_provider(monkeypatch, tmp_path, _HEALTHY_TEXT)
    with (pd / "chat_usage.jsonl").open("a", encoding="utf-8") as fh:  # corruption left from an earlier run
        fh.write("{torn\n")
    summary, _ = crw.run_replay(pd, turns=3, gap_s=0, provider_name="claude-cli", seed=False, force_text_path=True)
    res = summary["cache_break_check"]
    assert res["status"] == "UNMEASURED" and "skipped 1 corrupt or non-object line" in res["reason"]
    assert summary["chat_rows_observed"] == 6  # the means still come from the lenient read


def test_replay_with_a_provider_that_logs_no_usage_is_unmeasured_not_a_crash(tmp_path, monkeypatch):
    # e.g. --provider fake: chat_usage.jsonl is never created; the replay must still write its summary
    import brain.bridge.provider as provider_mod

    monkeypatch.setenv("NELL_CACHE_DEBUG", "0")
    pd = tmp_path / "personas" / "replay"
    crw._seed_scratch_persona(pd)
    monkeypatch.setattr(provider_mod, "get_provider", lambda name, *, persona_dir=None, model_override=None: provider_mod.FakeProvider())
    crashed, summary = None, None
    try:
        summary, _ = crw.run_replay(pd, turns=2, gap_s=0, provider_name="claude-cli", seed=False, force_text_path=True)
    except Exception as exc:  # noqa: BLE001
        crashed = exc
    assert crashed is None, f"replay crashed: {crashed!r}"
    assert summary["cache_break_check"]["status"] == "UNMEASURED" and summary["chat_rows_observed"] == 0
    assert crw._read_rows_fail_closed(tmp_path / "missing.jsonl") == []


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
    assert "row 3 read 3000 vs S=6449" in res["reason"] and "row 5 read 2000 vs S=9000" in res["reason"]
    assert "S=None" not in check(run(0, 0, 0))["reason"] and "with no S yet" in check(run(0, 0, 0))["reason"]


def test_c7_a_missing_pyyaml_is_a_value_error_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)  # `import yaml` raises ImportError
    with pytest.raises(ValueError):
        crw.read_regression_threshold(CONFIG)


def test_warmup_prompt_is_the_micro_ack():
    assert crw.WARMUP_PROMPT == "ok"


# --------------------------------------------------------------------------------------
# C8: replay exit codes (0 PASS / 1 FAIL / 2 UNMEASURED), through main()
# --------------------------------------------------------------------------------------


def _main_replay(tmp_path, monkeypatch, script, capsys):
    _patch_provider(monkeypatch, tmp_path, script)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    out = tmp_path / "metrics.json"
    rc = crw.main(["--scratch", "--turns", "3", "--gap-s", "0", "--out", str(out)])
    cap = capsys.readouterr()
    return rc, out, cap


@pytest.mark.parametrize(
    "script,expected,status",
    [(_HEALTHY_TOOLS, 0, "PASS"), (_BREAK_IN_CONTENT, 1, "FAIL"), ([(17385, 0, 1)] + [(12000, 6449, 3)] * 5, 2, "UNMEASURED")],
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


def test_c8_a_summary_without_the_key_or_with_an_unknown_status_is_exit_2(tmp_path, monkeypatch):
    for summary in ({}, {"cache_break_check": {"status": "WEIRD"}}, {"cache_break_check": None}):
        monkeypatch.setattr(crw, "run_replay", lambda *a, _s=summary, **k: (_s, []))
        assert crw.main(["--persona-dir", str(tmp_path)]) == 2


def test_c8_exit_code_mapping():
    assert [crw._exit_code_for({"status": s}) for s in ("PASS", "FAIL", "UNMEASURED")] == [0, 1, 2]
    assert crw._exit_code_for(None) == 2 and crw._exit_code_for("PASS") == 2


def test_c8_help_states_the_exit_codes_and_the_earlier_stage_codes():
    doc = " ".join((crw.__doc__ or "").split())
    assert "exits 0 PASS / 1 FAIL / 2 UNMEASURED" in doc
    assert "pre-run argument errors exit 1, argparse usage errors and the live-persona refusal exit 2" in doc
    assert "an OLD arm may legitimately exit 1 or 2" in doc
    assert "uv run python scripts/cache_replay_workload.py" in doc


# --------------------------------------------------------------------------------------
# C11: compare() diagnostics go to stderr and show each arm's verdict
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
    crw.compare(old, new)
    cap = capsys.readouterr()
    assert "WARNING" not in cap.err and "WARNING" not in cap.out
    assert "old cache_break_check: n/a" in cap.err and "new cache_break_check: n/a" in cap.err
    assert "mean cache_creation/turn: 1000 → 500  (-50%)" in cap.out  # stdout is still the A/B report only
