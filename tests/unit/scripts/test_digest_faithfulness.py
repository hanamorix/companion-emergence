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
                generate=lambda p, s: '{"verdict": "agree", "why": "x"}')
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
    df.main(["--log", str(log), "--out", str(out)], generate=lambda p, s: '{"verdict": "agree", "why": "x"}')
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
    code = df.main(["--log", str(log), "--out", str(out)], generate=judge)
    printed = capsys.readouterr().out
    assert code == 10
    summary = json.loads(printed)
    assert summary == {"pairs": 2, "agree": 1, "contradict": 1, "unclear": 0, "excluded": {"deduped": 1}}
    assert "focus still low" not in printed and "kettle" not in printed
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert [v["verdict"] for v in stored["verdicts"]] == ["contradict", "agree"]
    assert stored["verdicts"][0]["monologue"] == "focus still low"
