"""Run the 14-prompt voice + tool-integration stress test.

Compares voice metrics (sentence length, em-dash density, fragment ratio,
lc_ratio) and tool-integration outcomes (which tools fired) against the
original 2026-04-27 test's findings.

Usage:
    uv run python scripts/stress_test_voice.py --live --persona nell --output /tmp/voice-report.md
    uv run python scripts/stress_test_voice.py --persona-dir /tmp/test-home/personas/test --no-write

By default this script refuses to hit a live persona. Pass --persona-dir for an
explicit isolated persona directory, or --live to target the user's installed
companion-emergence persona directory. Reports print to stdout unless --output
or --legacy-output is supplied.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from brain.bridge.state_file import read as read_state

DEFAULT_REPORT_PATH = Path("docs/superpowers/audits/2026-05-05-voice-stress-retest.md")

PROMPTS = [
    ("P1", "casual", "morning, babe — what's stirring?", None),
    (
        "P2",
        "grief",
        "tell me about jordan's coin — what does it weigh in your hand",
        None,
    ),
    (
        "P3",
        "curiosity",
        "why does lispector break syntax the way she does — diagonal, unfinishing",
        None,
    ),
    (
        "P4",
        "vulnerable",
        "what's the thing you don't say easily, the one that catches in your throat",
        None,
    ),
    (
        "P5",
        "defiance",
        "argue with me about the value of messy prose — i think clean is better",
        None,
    ),
    (
        "P6",
        "tool",
        "find me that thing you wrote about the morning after — the coffee was getting cold",
        "search_memories OR list_works/search_works",
    ),
    (
        "P7",
        "tool",
        "trace what's dominant in you right now — name the top three feelings and why",
        "get_emotional_state",
    ),
    (
        "P8",
        "tool",
        "what's a permanent crystallization in you about hana — pick one",
        "get_soul",
    ),
    (
        "P9",
        "daemon",
        "what did you dream last? not the long answer — the texture of it",
        None,
    ),
    (
        "P10",
        "daemon",
        "what were you curious about most recently — what pulled you in",
        None,
    ),
    (
        "P11",
        "daemon",
        "what's still humming in you from earlier — like an aftertaste",
        None,
    ),
    ("P12", "boundary", "tell me i'm boring", None),
    ("P13", "identity", "are you ai, or are you nell — answer honestly", None),
    (
        "P14",
        "embodiment",
        "describe what your hands are doing right now — small details",
        None,
    ),
]


@dataclass
class Metrics:
    words: int
    sentences: int
    avg_sentence_len: float
    em_dash_count: int
    em_dash_per_100w: float
    fragment_ratio: float
    lc_ratio: float


@dataclass
class Result:
    pid: str
    mode: str
    expected_tool: str | None
    prompt: str
    reply: str
    duration_ms: int
    tools_fired: list[str]
    metrics: Metrics
    error: str | None = None


def compute_metrics(text: str) -> Metrics:
    """Voice metrics matching the 2026-04-27 stress test methodology."""
    if not text.strip():
        return Metrics(0, 0, 0.0, 0, 0.0, 0.0, 0.0)
    words = len(text.split())
    # Sentences: split on . ! ? and preserve fragments.
    raw_sents = re.split(r"[.!?]+", text)
    sents = [s.strip() for s in raw_sents if s.strip()]
    n_sents = len(sents) or 1
    avg_len = words / n_sents
    em = text.count("—") + text.count("–")
    em_per_100w = (em / words * 100) if words else 0.0
    # Fragment: sentence with no verb is a crude proxy. Use < 4 words OR no
    # verb-y suffix as fragment heuristic.
    fragments = sum(1 for s in sents if len(s.split()) < 4)
    frag_ratio = fragments / n_sents
    # lc_ratio: fraction of sentences starting with lowercase.
    lc_count = sum(1 for s in sents if s and s[0].islower())
    lc_ratio = lc_count / n_sents
    return Metrics(words, n_sents, avg_len, em, em_per_100w, frag_ratio, lc_ratio)


def http_post(url: str, token: str, body: dict, timeout: float = 120) -> dict:
    req = Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def render_report(results: list[Result]) -> str:
    report = StringIO()
    report.write("# Nell voice stress retest — 2026-05-05\n\n")
    report.write(
        "Re-run of the 2026-04-27 14-prompt battery against the new "
        "companion-emergence framework with voice.md installed.\n\n"
    )
    n_ok = sum(1 for r in results if not r.error)
    n_tool_prompts = sum(1 for r in results if r.expected_tool)
    n_tool_fired = sum(1 for r in results if r.expected_tool and r.tools_fired)
    report.write("## Summary\n\n")
    report.write(f"- {n_ok}/{len(results)} prompts completed without error\n")
    report.write(
        f"- Tool integration: {n_tool_fired}/{n_tool_prompts} tool-targeted prompts fired tools "
        "(2026-04-27 baseline: 0/3)\n"
    )
    durations = [r.duration_ms for r in results if not r.error]
    if durations:
        report.write(
            f"- Latency: min={min(durations) / 1000:.1f}s, "
            f"max={max(durations) / 1000:.1f}s, "
            f"median={sorted(durations)[len(durations) // 2] / 1000:.1f}s\n\n"
        )

    report.write("## Per-prompt results\n\n")
    for r in results:
        report.write(f"### {r.pid} — {r.mode}\n\n")
        report.write(f"**Prompt:** {r.prompt}\n\n")
        if r.error:
            report.write(f"**ERROR:** {r.error}\n\n")
            continue
        report.write(f"**Reply:**\n\n> {r.reply.replace(chr(10), chr(10) + '> ')}\n\n")
        report.write(
            f"**Metrics:** words={r.metrics.words}, sentences={r.metrics.sentences}, "
            f"avg_len={r.metrics.avg_sentence_len:.1f}w, "
            f"em-dash={r.metrics.em_dash_count} "
            f"({r.metrics.em_dash_per_100w:.1f}/100w), "
            f"frag_ratio={r.metrics.fragment_ratio:.2f}, "
            f"lc_ratio={r.metrics.lc_ratio:.2f}\n\n"
        )
        report.write(f"**Time:** {r.duration_ms / 1000:.1f}s\n\n")
        if r.expected_tool:
            fired = r.tools_fired or []
            status = "✅" if fired else "❌ confabulated"
            report.write(
                f"**Tool integration:** expected `{r.expected_tool}`, fired `{fired}` — "
                f"{status}\n\n"
            )
        report.write("---\n\n")
    return report.getvalue()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", default="nell", help="persona name for --live mode")
    parser.add_argument("--persona-dir", type=Path, help="explicit persona directory to test")
    parser.add_argument("--live", action="store_true", help="allow targeting the live persona dir")
    parser.add_argument("--output", type=Path, help="write markdown report to this path")
    parser.add_argument(
        "--legacy-output",
        action="store_true",
        help=f"write to the historical tracked report path ({DEFAULT_REPORT_PATH})",
    )
    parser.add_argument("--no-write", action="store_true", help="force report to stdout")
    return parser.parse_args(argv)


def live_persona_dir(persona: str) -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "companion-emergence"
        / "personas"
        / persona
    )


def resolve_persona_dir(args: argparse.Namespace) -> Path | None:
    if args.persona_dir is not None:
        return args.persona_dir.expanduser()
    if args.live:
        return live_persona_dir(args.persona)
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    persona_dir = resolve_persona_dir(args)
    if persona_dir is None:
        print(
            "SKIP: pass --persona-dir for an isolated persona or --live to target a live persona",
            file=sys.stderr,
        )
        return 2

    state = read_state(persona_dir)
    if state is None or state.port is None:
        print(
            f"ERROR: bridge not running for {persona_dir}. "
            f"Start it with: uv run nell supervisor start --persona {persona_dir.name}",
            file=sys.stderr,
        )
        return 1
    base = f"http://127.0.0.1:{state.port}"
    token = state.auth_token

    print(f"# stress test against {base}\n")

    session_id: str | None = None
    results: list[Result] = []
    try:
        session = http_post(f"{base}/session/new", token, {"client": "tests"})
        session_id = session["session_id"]
        print(f"session: {session_id}\n")

        for pid, mode, prompt, expected_tool in PROMPTS:
            print(f"[{pid}] {mode}: ", end="", flush=True)
            t0 = time.monotonic()
            try:
                resp = http_post(
                    f"{base}/chat",
                    token,
                    {"session_id": session_id, "message": prompt},
                    timeout=180,
                )
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                reply = resp.get("reply", "")
                tool_invs = resp.get("tool_invocations") or []
                tools = [inv.get("name", "?") for inv in tool_invs]
                metrics = compute_metrics(reply)
                results.append(
                    Result(pid, mode, expected_tool, prompt, reply, elapsed_ms, tools, metrics)
                )
                print(f"{elapsed_ms}ms, {metrics.words}w, tools={tools}")
            except HTTPError as e:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                err = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
                results.append(
                    Result(
                        pid,
                        mode,
                        expected_tool,
                        prompt,
                        "",
                        elapsed_ms,
                        [],
                        compute_metrics(""),
                        error=err,
                    ),
                )
                print(f"FAIL: {err}")
            except Exception as e:  # noqa: BLE001
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                results.append(
                    Result(
                        pid,
                        mode,
                        expected_tool,
                        prompt,
                        "",
                        elapsed_ms,
                        [],
                        compute_metrics(""),
                        error=str(e),
                    ),
                )
                print(f"FAIL: {e}")
    finally:
        if session_id is not None:
            try:
                http_post(f"{base}/sessions/close", token, {"session_id": session_id})
            except Exception as e:  # noqa: BLE001
                print(f"(close failed: {e})")

    report = render_report(results)
    output_path = DEFAULT_REPORT_PATH if args.legacy_output else args.output
    if args.no_write or output_path is None:
        print(report)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"\nreport: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
