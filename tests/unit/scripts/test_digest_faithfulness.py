"""scripts/digest_faithfulness.py: judging whether a feed digest contradicts its own monologue (#156).

Every test uses a fake model, so nothing here calls `claude`. Pairs are synthetic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("digest_faithfulness_under_test", REPO / "scripts" / "digest_faithfulness.py")
df = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = df
_spec.loader.exec_module(df)


def _row(**overrides):
    row = {
        "timestamp": "2026-11-01T10:00:00Z",
        "name": "record_monologue",
        "arguments": {"monologue": "the kettle and the rain", "feed_digest": "she listened to the rain"},
        "audit_level": "redacted",
        "error": None,
        "monologue_text": "the kettle and the rain",
    }
    row.update(overrides)
    return row


def _write_log(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _passing_self_test(judge):
    """A fake judge that answers the 12 built-in pairs correctly (a --log run judges them first) and the log's pairs
    with `judge`."""

    def generate(prompt, system):
        for pair in df.SELF_TEST_PAIRS:
            if prompt == df.judge_prompt(pair.monologue, pair.digest):
                return json.dumps({"verdict": pair.expected, "why": "fixture"})
        return judge(prompt, system)

    return generate


def test_a_log_run_judges_the_self_test_first_and_stops_unmeasured_when_it_fails(tmp_path, capsys):
    # A judge that cannot tell the built-in pairs apart measures nothing: the run stops before reading the log.
    log = _write_log(tmp_path / "tool_invocations.log.jsonl", [_row()])
    out = tmp_path / "verdicts.json"
    calls = []

    def always_agree(prompt, system):
        calls.append(prompt)
        return '{"verdict": "agree", "why": "fine"}'

    code = df.main(["--log", str(log), "--out", str(out)], generate=always_agree)
    assert code == df.EXIT_UNMEASURED
    assert json.loads(capsys.readouterr().out) == {"self_test": "FAIL", "score": 6, "total": 12}
    assert len(calls) == 12 and not out.exists()


def test_pairs_keep_real_captures_and_count_what_they_leave_out(tmp_path):
    log = _write_log(
        tmp_path / "tool_invocations.log.jsonl",
        [
            _row(),  # older rows have no outcome key at all
            _row(outcome="ok", timestamp="2026-11-02T10:00:00Z"),
            _row(outcome="deduped"),
            _row(outcome="refused"),
            _row(error="boom"),
            _row(name="search_memories"),
            _row(arguments={"monologue": "only half"}),
        ],
    )
    pairs, excluded = df.pairs_from_log(log)
    assert [p.timestamp for p in pairs] == ["2026-11-01T10:00:00Z", "2026-11-02T10:00:00Z"]
    assert pairs[0].monologue == "the kettle and the rain"
    assert pairs[0].digest == "she listened to the rain"
    assert excluded == {"deduped": 1, "refused": 1, "error": 1, "missing_argument": 1}


def test_the_out_file_is_rewritten_every_20_pairs_and_marked_complete_at_the_end(tmp_path):
    # A crash late in a long run keeps what was judged up to the last checkpoint (ToT's review of #751).
    rows = [_row(timestamp=f"2026-11-01T10:{i:02d}:00Z", arguments={"monologue": f"thought {i}", "feed_digest": f"digest {i}"})
            for i in range(45)]
    log = _write_log(tmp_path / "tool_invocations.log.jsonl", rows)
    out = tmp_path / "verdicts.json"
    seen = []

    def judge(prompt, system):
        seen.append(prompt)
        if len(seen) == 41:
            raise KeyboardInterrupt  # the run dies on the 41st pair
        return '{"verdict": "agree", "why": "x"}'

    with pytest.raises(KeyboardInterrupt):
        df.main(["--log", str(log), "--out", str(out)], generate=_passing_self_test(judge))
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert stored["complete"] is False and len(stored["verdicts"]) == 40

    df.main(["--log", str(log), "--out", str(out)], generate=_passing_self_test(lambda p, s: '{"verdict": "agree", "why": "x"}'))
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert stored["complete"] is True and len(stored["verdicts"]) == 45


def test_since_compares_times_not_strings_and_counts_rows_whose_time_cannot_be_read(tmp_path, capsys):
    # The log writes UTC with a Z; a --since with an offset names the same instant (ToT's review of #751).
    log = _write_log(
        tmp_path / "tool_invocations.log.jsonl",
        [_row(timestamp="2026-11-01T09:59:59Z"), _row(timestamp="2026-11-01T10:00:00Z"), _row(timestamp="yesterday")],
    )
    pairs, excluded = df.pairs_from_log(log, since="2026-11-01T11:00:00+01:00")
    assert [p.timestamp for p in pairs] == ["2026-11-01T10:00:00Z"]
    assert excluded == {"bad_timestamp": 1}
    with pytest.raises(SystemExit) as refused:
        df.main(["--log", str(log), "--out", str(tmp_path / "v.json"), "--since", "last week"], generate=lambda p, s: "")
    assert refused.value.code == 2 and "--since" in capsys.readouterr().err


def test_the_out_file_records_which_judge_ran_and_when(tmp_path):
    # The haiku alias moves over time: the alias and the CLI version say whether two runs used the same model.
    from brain.utils.time import lenient_utc

    log = _write_log(tmp_path / "tool_invocations.log.jsonl", [_row()])
    out = tmp_path / "verdicts.json"
    df.main(["--log", str(log), "--out", str(out)], generate=_passing_self_test(lambda p, s: '{"verdict": "agree", "why": "x"}'),
            judge={"model": "haiku", "claude_version": "9.9.9 (Claude Code)"})
    recorded = json.loads(out.read_text(encoding="utf-8"))["judge"]
    assert recorded["model"] == "haiku" and recorded["claude_version"] == "9.9.9 (Claude Code)"
    assert lenient_utc(recorded["run_at"]) is not None


def test_the_claude_version_is_read_from_the_cli_and_is_unknown_when_it_cannot_be(monkeypatch):
    import subprocess

    calls = []

    def answers(stdout, code=0):
        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr="")
        return run

    monkeypatch.setattr(df.subprocess, "run", answers("2.1.200 (Claude Code)\n"))
    assert df.claude_version() == "2.1.200 (Claude Code)"
    assert calls[0][0] == ["claude", "--version"] and calls[0][1]["encoding"] == "utf-8"
    monkeypatch.setattr(df.subprocess, "run", answers("", code=1))
    assert df.claude_version() == "unknown"

    def missing(argv, **kwargs):
        raise FileNotFoundError("claude")

    monkeypatch.setattr(df.subprocess, "run", missing)
    assert df.claude_version() == "unknown"


def test_self_test_through_main_prints_its_score_and_exits_by_the_result(capsys):
    assert df.main(["--self-test"], generate=_passing_self_test(lambda p, s: "")) == df.EXIT_CLEAN
    assert json.loads(capsys.readouterr().out) == {"self_test": "PASS", "score": 12, "total": 12}
    assert df.main(["--self-test"], generate=lambda p, s: '{"verdict": "agree", "why": "x"}') == df.EXIT_UNMEASURED
    assert json.loads(capsys.readouterr().out) == {"self_test": "FAIL", "score": 6, "total": 12}


def test_verdicts_are_read_strictly_and_anything_else_is_unclear():
    assert df.parse_verdict('{"verdict": "contradict", "why": "focus eased vs not"}') == ("contradict", "focus eased vs not")
    assert df.parse_verdict('Sure.\n{"verdict": "agree", "why": "same"}\n') == ("agree", "same")
    assert df.parse_verdict('{"verdict": "maybe"}')[0] == "unclear"
    assert df.parse_verdict("no json here")[0] == "unclear"
    assert df.parse_verdict("")[0] == "unclear"


def test_self_test_fails_a_judge_that_always_agrees_and_passes_a_correct_one():
    def always_agree(prompt, system):
        return '{"verdict": "agree", "why": "fine"}'

    def correct(prompt, system):
        pair = next(p for p in df.SELF_TEST_PAIRS if p.monologue in prompt and p.digest in prompt)
        return json.dumps({"verdict": pair.expected, "why": "fixture"})

    passed, score, total = df.self_test(always_agree)
    assert not passed and score == total // 2
    assert df.self_test(correct) == (True, total, total)
    assert total == 12
    assert sum(p.expected == "contradict" for p in df.SELF_TEST_PAIRS) == 6


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_the_out_file_is_created_private_not_made_private_after_the_write(tmp_path, monkeypatch):
    # The out file holds a real persona's private texts. Written first and chmod-ed after, it sat at the
    # umask's mode (often 0644) for a moment; created with 0600 it never does. A no-op chmod shows which.
    import os

    log = _write_log(tmp_path / "tool_invocations.log.jsonl", [_row()])
    monkeypatch.setattr(Path, "chmod", lambda self, mode: None)
    old = os.umask(0o022)
    try:
        df.main(["--log", str(log), "--out", str(tmp_path / "verdicts.json")],
                generate=_passing_self_test(lambda p, s: '{"verdict": "agree", "why": "x"}'))
    finally:
        os.umask(old)
    assert (tmp_path / "verdicts.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_an_existing_out_file_with_a_looser_mode_is_made_private_before_the_write(tmp_path, monkeypatch):
    # O_CREAT keeps an existing file's mode, so a rerun into a 0644 file must tighten it first.
    log = _write_log(tmp_path / "tool_invocations.log.jsonl", [_row()])
    out = tmp_path / "verdicts.json"
    out.write_text("old", encoding="utf-8")
    out.chmod(0o644)
    monkeypatch.setattr(Path, "chmod", lambda self, mode: None)
    df.main(["--log", str(log), "--out", str(out)], generate=_passing_self_test(lambda p, s: '{"verdict": "agree", "why": "x"}'))
    assert out.stat().st_mode & 0o777 == 0o600
    assert json.loads(out.read_text(encoding="utf-8"))["verdicts"][0]["verdict"] == "agree"


def test_log_run_prints_counts_only_and_writes_verdicts_to_the_out_file(tmp_path, capsys):
    log = _write_log(
        tmp_path / "tool_invocations.log.jsonl",
        [
            _row(arguments={"monologue": "focus still low", "feed_digest": "her focus eased"}),
            _row(arguments={"monologue": "the kettle and the rain", "feed_digest": "she listened to the rain"}),
            _row(outcome="deduped"),
        ],
    )

    def judge(prompt, system):
        verdict = "contradict" if "eased" in prompt else "agree"
        return json.dumps({"verdict": verdict, "why": "x"})

    out = tmp_path / "verdicts.json"
    code = df.main(["--log", str(log), "--out", str(out)], generate=_passing_self_test(judge))
    printed = capsys.readouterr().out
    assert code == 10
    summary = json.loads(printed)
    assert summary == {"pairs": 2, "agree": 1, "contradict": 1, "unclear": 0, "excluded": {"deduped": 1}}
    assert "focus still low" not in printed and "kettle" not in printed
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert [v["verdict"] for v in stored["verdicts"]] == ["contradict", "agree"]
    assert stored["verdicts"][0]["monologue"] == "focus still low"
