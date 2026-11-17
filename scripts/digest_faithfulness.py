"""Judge whether a feed digest contradicts the monologue it summarises (#156).

`record_monologue` carries both texts in one call, and the audit log keeps both arguments. This reads those pairs and
asks a Haiku judge, one pair per call, whether the digest contradicts its own thought.

Privacy: a real persona's pairs are private. Only counts go to stdout; verdicts and texts go to the `--out` file,
which belongs in a scratch folder outside the repo.

Always run through `scripts/clean_shell.sh`, so the nested `claude` call authenticates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brain.health.json_io import iter_jsonl  # noqa: E402


@dataclass(frozen=True)
class Pair:
    timestamp: str
    monologue: str
    digest: str


def pairs_from_log(path: Path, *, since: str | None = None) -> tuple[list[Pair], dict[str, int]]:
    """The real captures in an audit log, and a count of the `record_monologue` rows left out, by reason.

    A real capture has no error, an `outcome` that is absent (rows written before the field existed) or "ok", and both
    arguments as non-empty text. Deduped and refused rows are not captures (#93, #175).
    """
    pairs: list[Pair] = []
    excluded: Counter[str] = Counter()
    for row in iter_jsonl(Path(path)):
        if row.get("name") != "record_monologue":
            continue
        timestamp = str(row.get("timestamp") or "")
        if since is not None and timestamp < since:
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


def haiku_judge() -> Generate:
    """The real judge: Haiku pinned, whatever the user's tunables say."""
    from brain.bridge.model_tier import TIER_BACKGROUND_CLASSIFIER
    from brain.bridge.provider import ClaudeCliProvider

    provider = ClaudeCliProvider(TIER_BACKGROUND_CLASSIFIER, probe_model="haiku")
    return lambda prompt, system: provider.generate(prompt, system=system)


def main(argv: list[str] | None = None, *, generate: Generate | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--self-test", action="store_true", help="judge the 12 built-in synthetic pairs")
    parser.add_argument("--log", type=Path, help="a persona's tool_invocations.log.jsonl")
    parser.add_argument("--since", help="only rows at or after this ISO timestamp")
    parser.add_argument("--out", type=Path, help="where verdicts and texts go (keep it outside the repo)")
    args = parser.parse_args(argv)
    generate = generate or haiku_judge()

    if args.self_test:
        passed, score, total = self_test(generate)
        print(json.dumps({"self_test": "PASS" if passed else "FAIL", "score": score, "total": total}))
        return EXIT_CLEAN if passed else EXIT_UNMEASURED
    if args.log is None or args.out is None:
        parser.error("--log and --out are required unless --self-test")

    pairs, excluded = pairs_from_log(args.log, since=args.since)
    verdicts = []
    for index, pair in enumerate(pairs, 1):
        verdict, why = judge_pair(pair.monologue, pair.digest, generate)
        verdicts.append({"timestamp": pair.timestamp, "verdict": verdict, "why": why,
                         "monologue": pair.monologue, "digest": pair.digest})
        if index % 20 == 0:
            print(f"judged {index}/{len(pairs)}", file=sys.stderr, flush=True)
    args.out.write_text(json.dumps({"log": str(args.log), "verdicts": verdicts}, indent=1), encoding="utf-8")
    args.out.chmod(0o600)
    counts = Counter(v["verdict"] for v in verdicts)
    summary = {"pairs": len(pairs), **{name: counts.get(name, 0) for name in VERDICTS}, "excluded": excluded}
    print(json.dumps(summary))
    if summary["contradict"]:
        return EXIT_CONTRADICTION
    return EXIT_UNMEASURED if summary["unclear"] or not pairs else EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
