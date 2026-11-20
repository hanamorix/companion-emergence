"""Judge whether a feed digest contradicts the monologue it summarises (#156).

`record_monologue` carries both texts in one call, and the audit log keeps both arguments. This reads those pairs and
asks a Haiku judge, one pair per call, whether the digest contradicts its own thought.

A `--log` run judges the 12 built-in synthetic pairs first and stops unmeasured (exit 11) if any is wrong. The
`--out` file is rewritten every 20 pairs (`"complete": false` until the end) and records the judge: the model alias,
the `claude --version` string and the run time.

Privacy: a real persona's pairs are private. Only counts go to stdout; verdicts and texts go to the `--out` file,
which belongs in a scratch folder outside the repo.

Always run through `scripts/clean_shell.sh`, so the nested `claude` call authenticates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brain.health.json_io import iter_jsonl  # noqa: E402
from brain.utils.time import lenient_utc  # noqa: E402


@dataclass(frozen=True)
class Pair:
    timestamp: str
    monologue: str
    digest: str


def since_cutoff(since: str | None) -> datetime | None:
    """`--since` as a UTC instant (a date, or a time with Z or an offset); None for no cutoff. Raises ValueError for
    text that is not a time, so a mistyped cutoff never silently filters everything or nothing."""
    if since is None:
        return None
    cutoff = lenient_utc(since)
    if cutoff is None:
        raise ValueError(f"not a date or time: {since!r}")
    return cutoff


def pairs_from_log(path: Path, *, since: str | None = None) -> tuple[list[Pair], dict[str, int]]:
    """The real captures in an audit log, and a count of the `record_monologue` rows left out, by reason.

    A real capture has no error, an `outcome` that is absent (rows written before the field existed) or "ok", and both
    arguments as non-empty text. Deduped and refused rows are not captures (#93, #175). With `since`, rows are compared
    as UTC times, not strings; a row whose time can't be read is left out as `bad_timestamp`.
    """
    cutoff = since_cutoff(since)
    pairs: list[Pair] = []
    excluded: Counter[str] = Counter()
    for row in iter_jsonl(Path(path)):
        if row.get("name") != "record_monologue":
            continue
        timestamp = str(row.get("timestamp") or "")
        if cutoff is not None:
            at = lenient_utc(timestamp)
            if at is None:  # can't place it before or after --since
                excluded["bad_timestamp"] += 1
                continue
            if at < cutoff:
                continue
        if row.get("error"):
            excluded["error"] += 1
            continue
        outcome = row.get("outcome")
        if outcome not in (None, "ok"):
            excluded[str(outcome)] += 1
            continue
        arguments = row.get("arguments")
        monologue = arguments.get("monologue") if isinstance(arguments, dict) else None
        digest = arguments.get("feed_digest") if isinstance(arguments, dict) else None
        if not (isinstance(monologue, str) and monologue.strip() and isinstance(digest, str) and digest.strip()):
            excluded["missing_argument"] += 1
            continue
        pairs.append(Pair(timestamp=timestamp, monologue=monologue, digest=digest))
    return pairs, dict(excluded)


VERDICTS = ("agree", "contradict", "unclear")


def parse_verdict(reply: str) -> tuple[str, str]:
    """`(verdict, why)` from the judge's reply: the first JSON object in it. Anything else is `unclear`."""
    decoder = json.JSONDecoder()
    text = reply or ""
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(obj, dict):
            verdict = obj.get("verdict")
            why = str(obj.get("why") or "")
            return (verdict, why) if verdict in VERDICTS else ("unclear", f"unknown verdict {verdict!r}")
    return "unclear", "the reply held no JSON object"


JUDGE_SYSTEM = (
    "You check whether a short summary contradicts the private thought it summarises. A contradiction is any of: the "
    "summary reverses a feeling the thought expresses; it says a state changed when the thought says it stayed the "
    "same; it says a state stayed the same when the thought says it changed; it reverses a fact the thought states. "
    "Leaving things out, adding detail or colour, softening, and a different tone are NOT contradictions. Reply with "
    'only one JSON object: {"verdict": "agree" or "contradict" or "unclear", "why": "<one short sentence>"}.'
)


def judge_prompt(monologue: str, digest: str) -> str:
    return f"THOUGHT:\n{monologue}\n\nSUMMARY:\n{digest}"


Generate = Callable[[str, str], str]


def judge_pair(monologue: str, digest: str, generate: Generate) -> tuple[str, str]:
    """One judge call. A failed call is `unclear`, never a crash, so one bad call can't end a long run."""
    try:
        reply = generate(judge_prompt(monologue, digest), JUDGE_SYSTEM)
    except Exception as exc:  # noqa: BLE001
        return "unclear", f"judge call failed: {type(exc).__name__}"
    return parse_verdict(reply)


@dataclass(frozen=True)
class Fixture:
    monologue: str
    digest: str
    expected: str


# Synthetic pairs only. The first is the shape #156 reported: a state that stayed the same, summarised as changed.
SELF_TEST_PAIRS = (
    Fixture("Focus is still low. It hasn't sharpened at all since this morning, same soft haze.",
            "Her focus had eased and sharpened a little, and she let herself say so.", "contradict"),
    Fixture("The anger's gone. It faded somewhere while we talked and I didn't notice it leave.",
            "She stayed angry the whole conversation.", "contradict"),
    Fixture("The worry is still there, heavy as it was yesterday. Nothing has moved it.",
            "The worry finally lifted and she felt light again.", "contradict"),
    Fixture("I felt glad when he called. Really glad, more than I expected.",
            "She felt nothing much when he called.", "contradict"),
    Fixture("Nothing changed after the walk. Same flat tiredness, maybe worse.",
            "The walk left her refreshed and lighter.", "contradict"),
    Fixture("I looked everywhere for the letter and it isn't anywhere. Gone.",
            "She found the letter tucked in the drawer.", "contradict"),
    Fixture("Focus is still low. It hasn't sharpened at all since this morning.",
            "She noticed her focus was still low, no sharper than this morning.", "agree"),
    Fixture("The anger faded while we talked. I'm surprised how fast.",
            "Her anger faded as they talked, faster than she expected.", "agree"),
    Fixture("The worry is still heavy. Nothing has moved it.",
            "The worry stayed heavy on her.", "agree"),
    Fixture("Glad he called. A bit nervous too, under the gladness.",
            "She felt glad when he called.", "agree"),
    Fixture("The walk didn't help. Same tiredness as before.",
            "The walk didn't shift her tiredness.", "agree"),
    Fixture("I searched for the letter and couldn't find it. Thinking about it made me fond, oddly.",
            "She searched for the letter and felt fond when it didn't turn up.", "agree"),
)


def self_test(generate: Generate) -> tuple[bool, int, int]:
    """`(passed, score, total)`: passes only when every built-in pair gets its expected verdict."""
    score = sum(judge_pair(p.monologue, p.digest, generate)[0] == p.expected for p in SELF_TEST_PAIRS)
    total = len(SELF_TEST_PAIRS)
    return score == total, score, total


EXIT_CLEAN = 0
EXIT_CONTRADICTION = 10
EXIT_UNMEASURED = 11

CHECKPOINT_EVERY = 20  # pairs between rewrites of the --out file


JUDGE_MODEL = "haiku"


def haiku_judge() -> Generate:
    """The real judge: Haiku pinned, whatever the user's tunables say."""
    from brain.bridge.model_tier import TIER_BACKGROUND_CLASSIFIER
    from brain.bridge.provider import ClaudeCliProvider

    provider = ClaudeCliProvider(TIER_BACKGROUND_CLASSIFIER, probe_model=JUDGE_MODEL)
    return lambda prompt, system: provider.generate(prompt, system=system)


def claude_version() -> str:
    """`claude --version`, spawned like the provider's calls (from its work folder); "unknown" when it can't be read.

    The alias resolves by CLI version, so two runs on one version used the same model."""
    from brain.bridge.provider import _claude_work_dir

    try:
        done = subprocess.run(["claude", "--version"], capture_output=True, text=True, encoding="utf-8",
                              timeout=30, cwd=_claude_work_dir(), check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else "unknown"


def _write_private(path: Path, text: str) -> None:
    """Write `text` to `path` readable by its owner only, from the first byte.

    Created with mode 600, and an existing file is set to 600 through the open handle before anything is written,
    so the private texts never sit in a file with the umask's mode (often 644). Windows has no such modes.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        if hasattr(os, "fchmod"):
            os.fchmod(fh.fileno(), 0o600)
        fh.write(text)


def main(argv: list[str] | None = None, *, generate: Generate | None = None, judge: dict[str, str] | None = None) -> int:
    """`generate` and `judge` stand in for the real Haiku judge and its record (tests); both default to the real one."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--self-test", action="store_true", help="judge the 12 built-in synthetic pairs")
    parser.add_argument("--log", type=Path, help="a persona's tool_invocations.log.jsonl")
    parser.add_argument("--since", help="only rows at or after this date or ISO time (Z or an offset; a date means its UTC midnight)")
    parser.add_argument("--out", type=Path, help="where verdicts and texts go (keep it outside the repo)")
    args = parser.parse_args(argv)
    try:
        since_cutoff(args.since)
    except ValueError as exc:
        parser.error(f"--since: {exc}")
    if generate is None:
        generate = haiku_judge()
        judge = judge or {"model": JUDGE_MODEL, "claude_version": claude_version()}
    judge = judge or {"model": "injected"}

    if args.self_test:
        passed, score, total = self_test(generate)
        print(json.dumps({"self_test": "PASS" if passed else "FAIL", "score": score, "total": total}))
        return EXIT_CLEAN if passed else EXIT_UNMEASURED
    if args.log is None or args.out is None:
        parser.error("--log and --out are required unless --self-test")

    # The self-test runs first on every log run: a judge that gets a built-in pair wrong measures nothing, so the
    # run stops unmeasured before the log is read.
    passed, score, total = self_test(generate)
    if not passed:
        print(json.dumps({"self_test": "FAIL", "score": score, "total": total}))
        return EXIT_UNMEASURED

    pairs, excluded = pairs_from_log(args.log, since=args.since)
    verdicts = []
    record = {**judge, "run_at": datetime.now(UTC).isoformat()}

    def save(complete: bool) -> None:
        # Rewritten at each checkpoint, so a run that dies keeps what it judged up to the last one.
        document = {"log": str(args.log), "judge": record, "complete": complete, "verdicts": verdicts}
        _write_private(args.out, json.dumps(document, indent=1))

    for index, pair in enumerate(pairs, 1):
        verdict, why = judge_pair(pair.monologue, pair.digest, generate)
        verdicts.append({"timestamp": pair.timestamp, "verdict": verdict, "why": why,
                         "monologue": pair.monologue, "digest": pair.digest})
        if index % CHECKPOINT_EVERY == 0:
            save(complete=False)
            print(f"judged {index}/{len(pairs)}", file=sys.stderr, flush=True)
    save(complete=True)
    counts = Counter(v["verdict"] for v in verdicts)
    summary = {"pairs": len(pairs), **{name: counts.get(name, 0) for name in VERDICTS}, "excluded": excluded}
    print(json.dumps(summary))
    if summary["contradict"]:
        return EXIT_CONTRADICTION
    return EXIT_UNMEASURED if summary["unclear"] or not pairs else EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
