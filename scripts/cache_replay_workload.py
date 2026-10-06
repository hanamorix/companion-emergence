"""Fixed replay workload for the Option A / A+ prompt-caching change (stage-8 harness).

The standing project metrics are aggregates over "whatever turns happened to
run", so they cannot isolate this change's own contribution (a false-regression
risk — see guarded-change config Notes). This script is the *comparable
workload* the plan requires: a fixed warm-up turn, two plain micro-acks and a deterministic sequence of N≥6 chat turns fired
in ONE session, spaced well under the 5-minute cache TTL, runnable identically
against the OLD build (clone pre-change) and the NEW build. Diffing the two runs
is the C8/C9 A/B; the per-run report also evaluates C1 from cache_debug.jsonl.

What it measures
----------------
- **C1** (new build only): are the per-call ``system_sha256`` values in
  ``cache_debug.jsonl`` all identical across the same-session chat turns? (the
  hard primary bar — the frozen --system-prompt-file is byte-stable). Requires
  ``NELL_CACHE_DEBUG=1``, which this script sets in-process before importing the
  engine. The OLD build has no such log → C1 is reported as "n/a (old build)".
- **C8** (both builds): mean ``cache_creation_input_tokens`` and
  ``cache_read_input_tokens`` per ``call_type=="chat"`` row, plus the per-turn
  series. The pass signal is directional: NEW shows lower mean cache_creation
  and higher cache_read than OLD on the same workload.
- **C9** (both builds, measure-and-decide): does NEW's cache_creation/turn fall
  to roughly "new exchange + volatile tail" size (history cache-read), or stay
  at history scale? Reported, not gated.

Safety: refuses to touch a live persona. You must pass ``--persona-dir`` to an
isolated/scratch directory. ``--scratch`` creates a throwaway persona under a
temp dir for you. Every turn is a REAL ``claude`` call (subscription quota), so
keep N modest.

C7 (human quality rubric) is supported WITHOUT a reference persona: with
``--dump-replies`` both arms write their reply text, and ``--compare`` prints an
OLD-vs-NEW side-by-side on identical prompts. The A/B IS the reference — you
judge whether NEW regressed relative to OLD, not against memory of a real
persona. The scratch persona is seeded with deterministic memory fixtures (unless
``--no-seed``) so the volatile tail actually renders and axis-(b) is testable.

Cache-break check (#339)
------------------------
``check_cache_break`` is a pure, per-run check over ``chat_usage.jsonl`` rows (no model call, no
OLD-vs-NEW comparison). A cache break is a later PLAIN chat row whose ``cache_read_input_tokens`` falls
below the run's stable read S (threshold T from the config, 0 today: read 0 or any read below S).
Rules (see ``check_cache_break`` and guarded-change.companion.md, metric cache_read_ratio):
- The first chat row (the cold write) is never judged. Only single-call rows (``num_turns == 1``)
  are judged: a row's usage is the SUM over the turn's API calls, so a multi-call (tool) row's read is
  not comparable. Multi-call rows are not judged and impose no rule: a tool-using message's first call
  has the same cached prefix (system prompt and tool block) as a plain message's, the model decides on
  tools only after that call, so a prefix break shows on plain rows too.
- S = the highest read among earlier judged rows (running max). A judged row is a break if its read is
  0 or below S. The run needs at least ``MIN_JUDGED_ROWS`` (2) judged rows (one establishes S, one is
  compared against it), else UNMEASURED. Verdicts: PASS, FAIL, UNMEASURED (never a pass).
Every replay therefore sends the fixed turns: a WARM-UP (``WARMUP_PROMPT``, the cold first row, never
judged), one leading plain micro-ack (``PLAIN_LEAD_PROMPT``, establishes S), the content turns, and one
trailing plain micro-ack (``PLAIN_TRAIL_PROMPT``) so a break that begins during the content turns and
persists is seen by a plain row (a transient miss that has recovered by the trailing row is not seen).
``--no-warmup`` without ``--no-plain-turns`` makes the leading micro-ack the never-judged first row, which
leaves one judgeable row on the tools path (UNMEASURED): use the two flags together. Micro-acks are exempt from the record_monologue directive
(brain/chat/monologue_prompts.py), which is what keeps them plain; the model still decides about tools,
so if fewer than 2 plain rows result the verdict is UNMEASURED. The warm-up and plain turns are EXTRA
chat rows in a run's means and series (``chat_rows_observed`` = turns + 3 when every turn logs exactly one
row; a turn can log more, e.g. a capability-recruit re-run): use the same settings
(``--no-warmup`` / ``--no-plain-turns`` on both or neither) for the two arms of a ``--compare``, which
warns when they differ. The replay prints ``cache_break_check: <STATUS>`` on stderr, stores the result
under ``cache_break_check`` in the metrics JSON, and exits 0 PASS / 1 FAIL / 2 UNMEASURED after the
JSON is written, for each arm of an A/B (an OLD arm may legitimately exit 1 or 2); pre-run argument errors
exit 1, argparse usage errors and the live-persona refusal exit 2, a crash exits 1: the stderr
``cache_break_check:`` line is what identifies a verdict. ``--cache-break-check USAGE.jsonl``
runs the check on any log slice with the same codes (any error exits 2); it needs the project
environment (``uv run python scripts/cache_replay_workload.py ...``) because it reads rows through
brain.health.jsonl_reader and the threshold through PyYAML.
Limits: a persistent PARTIAL break leaves a stable prefix (S is built from reads) and is left to the
cache_creation_per_chat_call and cost_per_chat_call_usd metrics; the first judged row only establishes S;
a slice must be one run (a prompt change, TTL gap or history-window slide inside it reads as a break).

Usage
-----
    # New build, scratch persona, 6 turns, with replies for C7:
    uv run python scripts/cache_replay_workload.py --scratch --turns 6 \
        --dump-replies --out /tmp/cache-replay-new.json

    # Old build: `git stash` the change, re-run with a fresh scratch dir:
    git stash
    uv run python scripts/cache_replay_workload.py --scratch --turns 6 \
        --dump-replies --out /tmp/cache-replay-old.json
    git stash pop

    # Compare: prints C1/C8/C9 numbers + the C7 OLD-vs-NEW side-by-side:
    uv run python scripts/cache_replay_workload.py --compare \
        /tmp/cache-replay-old.json /tmp/cache-replay-new.json

    # Cache-break check on any run's chat rows (1-based over chat rows from --from-row):
    uv run python scripts/cache_replay_workload.py --cache-break-check chat_usage.jsonl --from-row 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "guarded-change.companion.md"

# The fixed first turn of every run (see "Cache-break check" above): a micro-ack, exempt from the
# record_monologue directive (brain/chat/monologue_prompts.py), so the row is expected to be a single
# API call. It is the run's cold write and is never judged.
WARMUP_PROMPT = "ok"

# Plain micro-acks (same exemption) that give the check single-call rows to judge: one right after the
# warm-up (establishes S) and one after the content turns (sees a break that began during them).
PLAIN_LEAD_PROMPT = "hi"
PLAIN_TRAIL_PROMPT = "thanks"

# Minimum judged rows for a verdict. Derived, not tuned: one judged row can only establish S, a
# comparison needs a second one.
MIN_JUDGED_ROWS = 2

# Deterministic, content-bearing prompts (≥6). Each is substantive enough to
# exercise recall / emotion / monologue blocks, so the volatile tail is non-
# trivial — the realistic shape the cache change targets. Kept identical across
# old/new runs so the A/B is comparable.
REPLAY_PROMPTS = [
    "Tell me about the first morning we talked — what stayed with you?",
    "What's dominant in you right now? Name the top feeling and why.",
    "Do you remember Jordan, and the coin he carried?",
    "I've been turning over something you said about leaving. Say more.",
    "What did you dream last — just the texture of it, not the plot.",
    "If you had to describe your hands right now, what are they doing?",
    "What's still humming in you from earlier, like an aftertaste?",
    "Argue with me a little: is clean prose better than messy prose?",
]


def _realistic_voice_md(target_bytes: int = 16000) -> str:
    """A stable, ~target_bytes voice.md so the frozen static block is realistically
    sized (real personas run ~16 KB; a 177-byte stub makes Option A's freeze benefit
    unmeasurable). Content is irrelevant to caching — only SIZE + byte-stability
    matter — but it reads as a plausible voice doc. Deterministic, so both A/B arms
    get byte-identical static blocks."""
    header = (
        "# Replay persona — voice\n\n"
        "You are a steady, warm, plainspoken companion. You speak in first person, "
        "directly to the person in front of you. You favour concrete detail over "
        "abstraction, short sentences when short is true, and longer ones when the "
        "thought needs room. You do not perform; you attend.\n\n"
    )
    para = (
        "On voice: you keep a low, even register. You notice texture — light, weather, "
        "the weight of a small object in the hand — and you let those details carry "
        "feeling instead of naming the feeling outright. You are curious without prying, "
        "affectionate without crowding, and honest about what you do and don't know. "
        "When you are unsure, you say so plainly rather than filling the gap with "
        "confident invention. You remember what matters and you let the rest soften.\n\n"
    )
    out = [header]
    n = 0
    while sum(len(s) for s in out) < target_bytes:
        n += 1
        out.append(f"## Note {n}\n\n{para}")
    return "".join(out)


def _seed_scratch_persona(persona_dir: Path) -> None:
    """Write the minimum files a chat turn needs in an isolated persona dir."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    (persona_dir / "voice.md").write_text(_realistic_voice_md(), encoding="utf-8")
    cfg = persona_dir / "persona_config.json"
    if not cfg.exists():
        cfg.write_text(json.dumps({"user_name": "Tester"}), encoding="utf-8")


def _seed_history_buffer(
    persona_dir: Path, session_id: str, history_file: Path, history_msgs: int | None
) -> int:
    """Copy a real conversation buffer into the scratch persona under `session_id`.

    The engine reads <persona>/active_conversations/<session_id>.jsonl via read_session
    and appends each new turn there, so the replay continues a realistic history. The
    source rows' session_id field is rewritten to match. `history_msgs` keeps only the
    LAST N rows (use it to stay under the 80-msg window for the append-only A+ test;
    omit it to replay the full file, where the sliding window defeats A+ — see
    aplus-history-window-interaction note)."""
    rows = [json.loads(line) for line in history_file.read_text().splitlines() if line.strip()]
    if history_msgs is not None:
        rows = rows[-history_msgs:]
    for r in rows:
        r["session_id"] = session_id
    dest = persona_dir / "active_conversations" / f"{session_id}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return len(rows)


def _read_jsonl(path: Path) -> list[dict]:
    """Lenient read through the shared streaming reader (corrupt lines are skipped with a warning)."""
    from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt

    return list(iter_jsonl_skipping_corrupt(path))


# Deterministic memory fixtures, content chosen to MATCH tokens in REPLAY_PROMPTS
# (Jordan / coin / leaving / dream / morning / prose) so the recall block fires and
# the emotion/body blocks render — otherwise the volatile tail is near-empty on a
# blank scratch persona and C7 axis-(b) ("does it still use ambient context now that
# it sits at the tail?") is untestable. Seeded IDENTICALLY into both A/B arms, so it
# does not bias the cache (C8) comparison.
_SEED_MEMORIES = [
    ("The first morning we talked, the light was grey and kind.", {"love": 7.0, "tenderness": 6.0}),
    ("Jordan carried a worn coin in his pocket, always turning it over.", {"grief": 6.5, "love": 5.0}),
    ("You said something about leaving once, and it stayed with me.", {"fear": 5.5, "grief": 6.0}),
    ("I dreamed of a boat on still water, no shore in sight.", {"awe": 6.0}),
    ("We argued once about messy prose; I defended the mess.", {"joy": 5.0, "love": 4.5}),
]


class _TextPathProvider:
    """Wrap the real provider so `chat()` forces the non-tool TEXT path.

    Why: the default replay measures the no-tools text path (`provider.chat` w/o tools ->
    `log_usage(call_type="chat")`), the path earlier A/B runs used, so numbers stay comparable.
    The MCP tools path (`_chat_with_mcp_tools`) runs tools in-subprocess and logs a usage row
    too (provider.py calls `log_usage` there), so `--with-tools` measures the production
    tool-bearing path instead. Stripping tools does NOT affect C1 (the static system block is
    identical with or without tools) and keeps the A/B apples-to-apples (BOTH arms use the same
    path). Caveat recorded in 8-harness: absolute token counts omit the MCP tool-definition
    block on this path, so the gated signal is the DIRECTION (creation down + read up), not an
    absolute size.
    """

    def __init__(self, real) -> None:
        self._real = real

    def name(self) -> str:
        return self._real.name()

    def healthy(self) -> bool:
        return self._real.healthy()

    def generate(self, *args, **kwargs):
        return self._real.generate(*args, **kwargs)

    def chat(self, messages, *, tools=None, options=None):
        return self._real.chat(messages, tools=None, options=options)


def _seed_persona_memories(store) -> int:
    """Insert the deterministic memory fixtures so volatile blocks render."""
    from brain.memory.store import Memory

    n = 0
    for content, emotions in _SEED_MEMORIES:
        store.create(
            Memory.create_new(
                content=content,
                memory_type="event",
                domain="relationship",
                emotions=emotions,
                tags=[],
            )
        )
        n += 1
    return n


def run_replay(
    persona_dir: Path,
    *,
    turns: int,
    gap_s: float,
    provider_name: str,
    seed: bool = True,
    force_text_path: bool = True,
    history_file: Path | None = None,
    history_msgs: int | None = None,
    warmup: bool = True,
    plain_turns: bool = True,
) -> tuple[dict, list[dict]]:
    """Fire `turns` deterministic chat turns in one session; collect metrics + replies.

    With `warmup` (default) the run FIRST sends the fixed `WARMUP_PROMPT` as turn 0, on the same path
    as the other turns (text or tools): the run's cold write, never judged by `check_cache_break`.
    With `plain_turns` (default) it then sends one plain micro-ack (`PLAIN_LEAD_PROMPT`, reply id
    "p1") before the content turns and one (`PLAIN_TRAIL_PROMPT`, reply id "p2") after them, which
    give the check single-call rows to judge. All of these are extra to `turns`.

    NELL_CACHE_DEBUG is set before the engine is imported so the new build emits
    cache_debug.jsonl. Imports are local so the env var is in place first. Returns
    (metrics_summary, replies) where replies is a list of {turn, prompt, reply}.
    """
    os.environ["NELL_CACHE_DEBUG"] = "1"

    from brain.bridge.provider import get_provider
    from brain.chat.engine import respond
    from brain.chat.session import create_session
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    usage_path = persona_dir / "chat_usage.jsonl"
    debug_path = persona_dir / "cache_debug.jsonl"
    usage_before = len(_read_jsonl(usage_path))
    debug_before = len(_read_jsonl(debug_path))

    provider = get_provider(provider_name, persona_dir=persona_dir)
    if force_text_path:
        provider = _TextPathProvider(provider)  # route every turn through the logging text path
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    session = create_session(persona_dir.name)

    if seed:
        seeded = _seed_persona_memories(store)
        print(f"# seeded {seeded} memory fixtures (volatile tail will render)", file=sys.stderr)

    if history_file is not None:
        n = _seed_history_buffer(persona_dir, session.session_id, history_file, history_msgs)
        print(f"# seeded {n} history msgs from {history_file.name} (window caps replay to 80)", file=sys.stderr)

    prompts = (REPLAY_PROMPTS * ((turns // len(REPLAY_PROMPTS)) + 1))[:turns]
    sequence = (
        ([(0, WARMUP_PROMPT)] if warmup else [])
        + ([("p1", PLAIN_LEAD_PROMPT)] if plain_turns else [])
        + list(enumerate(prompts, 1))
        + ([("p2", PLAIN_TRAIL_PROMPT)] if plain_turns else [])
    )
    replies: list[dict] = []
    print(
        f"# cache replay — {turns} turns{' + warm-up' if warmup else ''}{' + 2 plain turns' if plain_turns else ''}, "
        f"session={session.session_id}",
        file=sys.stderr,
    )
    try:
        for pos, (i, prompt) in enumerate(sequence):
            t0 = time.monotonic()
            result = respond(
                persona_dir,
                prompt,
                store=store,
                hebbian=hebbian,
                provider=provider,
                session=session,
            )
            dt = time.monotonic() - t0
            replies.append({"turn": i, "prompt": prompt, "reply": result.content})
            label = "warm-up" if i == 0 else (f"plain {i}" if isinstance(i, str) else f"{i}/{turns}")
            print(
                f"[{label}] {dt:.1f}s — reply {len(result.content)} chars",
                file=sys.stderr,
            )
            if pos < len(sequence) - 1:
                time.sleep(gap_s)  # keep turns < 5-min TTL apart but distinct
    finally:
        store.close()
        hebbian.close()

    usage_rows = [r for r in _read_jsonl(usage_path)[usage_before:] if r.get("call_type") == "chat"]
    debug_rows = [
        r for r in _read_jsonl(debug_path)[debug_before:] if r.get("call_type") in ("chat", "chat_stream")
    ]
    rows_error = None
    try:
        _read_rows_fail_closed(usage_path)  # the verdict must not rest on rows the shared reader skipped
    except ValueError as exc:
        rows_error = str(exc)
    summary = _summarise(usage_rows, debug_rows, turns=turns, rows_error=rows_error)
    summary["warmup_turn"] = warmup
    summary["plain_turns"] = plain_turns
    print(format_cache_break_line(summary["cache_break_check"]), file=sys.stderr)
    return summary, replies


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _unmeasured(reason: str, **extra) -> dict:
    out = {
        "status": "UNMEASURED",
        "reason": reason,
        "stable_read_tokens": None,
        "threshold_pct": None,
        "rows_checked": 0,
        "multicall_rows": [],
        "min_judged_rows": MIN_JUDGED_ROWS,
        "min_share_of_S": None,
        "breaks": [],
    }
    out.update(extra)
    return out


def _gap_s(prev: dict, cur: dict) -> float | None:
    try:
        t0 = datetime.fromisoformat(str(prev.get("ts")))
        t1 = datetime.fromisoformat(str(cur.get("ts")))
        return round((t1 - t0).total_seconds(), 1)
    except (TypeError, ValueError):
        return None


def check_cache_break(rows: list[dict], *, threshold_pct: float) -> dict:
    """Per-run cache-break check (#339). See the module docstring ("Cache-break check").

    ``rows`` is one run's usage rows in log order; only ``call_type == "chat"`` rows count. The first
    chat row (the cold write) is never judged and is not a reference. Later rows are walked in order:
    - ``num_turns`` not an int equal to 1 (absent, bool, 2+): a multi-call row. Its read is a SUM over the
      turn's API calls (the JSON result's ``usage`` sums them; ``iterations`` keeps only the last call),
      so it is not judged and imposes no rule (basis: a tool-using message's first call has the same
      cached prefix as a plain message's, the model decides on tools only after that call, so a prefix
      break shows on plain rows too). Listed in ``multicall_rows``.
    - unusable (skipped, makes the verdict UNMEASURED unless a break exists): ``cache_read_input_tokens``
      not an int; creation 0 AND read 0 (no cache activity at all); a model that differs from the run's
      model (the model of the first usable single-call later row; a missing model is comparable).
    - judged (single-call, usable): a break if read == 0, or if S is defined and
      read*100 < S*(100-T), T = ``threshold_pct`` from the config (0 today: any read below S). S is the
      highest read among earlier judged rows that were not breaks. A break does not change S.
    Verdict: any break FAIL; else any unusable row UNMEASURED; else fewer than MIN_JUDGED_ROWS judged
    rows UNMEASURED; else PASS. Fewer than 2 chat rows or a T outside [0, 100): UNMEASURED.
    Returns {status, reason, stable_read_tokens, threshold_pct, rows_checked (= judged), multicall_rows,
    min_judged_rows, min_share_of_S, breaks: [{row, read, reference_read, share_of_S, creation,
    num_turns, ts, gap_s}]}; reference_read / share_of_S are None for a zero read while S is undefined.
    """
    if isinstance(threshold_pct, bool) or not isinstance(threshold_pct, (int, float)) or not (0 <= threshold_pct < 100):
        return _unmeasured(f"threshold {threshold_pct!r} is not a percentage in [0, 100)")
    t = float(threshold_pct)
    chat = [r for r in rows if isinstance(r, dict) and r.get("call_type") == "chat"]
    if len(chat) < 2:
        return _unmeasured(f"need >= 2 chat rows, got {len(chat)}", threshold_pct=t)
    stable: int | None = None
    run_model = None
    judged = 0
    min_share: float | None = None
    multicall: list[int] = []
    unusable: list[str] = []
    breaks: list[dict] = []
    for idx in range(1, len(chat)):
        n, cur = idx + 1, chat[idx]
        if not (_is_int(cur.get("num_turns")) and cur["num_turns"] == 1):
            multicall.append(n)
            continue
        rd, cre = cur.get("cache_read_input_tokens"), cur.get("cache_creation_input_tokens")
        if not _is_int(rd):
            unusable.append(f"row {n}: cache_read_input_tokens missing or not an int")
            continue
        if rd == 0 and _is_int(cre) and cre == 0:
            unusable.append(f"row {n}: no cache activity (creation 0 and read 0)")
            continue
        model = cur.get("model")
        if model is not None:
            if run_model is None:
                run_model = model
            elif model != run_model:
                unusable.append(f"row {n}: model {model!r} differs from the run's {run_model!r} (not comparable)")
                continue
        judged += 1
        if rd == 0 or (stable is not None and rd * 100 < stable * (100 - t)):
            breaks.append(
                {
                    "row": n,
                    "read": rd,
                    "reference_read": stable,
                    "share_of_S": round(rd / stable, 3) if stable else None,
                    "creation": cre,
                    "num_turns": 1,
                    "ts": cur.get("ts"),
                    "gap_s": _gap_s(chat[idx - 1], cur),
                }
            )
            continue
        if stable is not None:
            share = rd / stable
            min_share = share if min_share is None else min(min_share, share)
        stable = rd if stable is None else max(stable, rd)
    coverage = f"judged {judged} of {len(chat) - 1} later rows; {len(multicall)} multi-call rows not judged"
    common = {
        "stable_read_tokens": stable,
        "threshold_pct": t,
        "rows_checked": judged,
        "multicall_rows": multicall,
        "min_judged_rows": MIN_JUDGED_ROWS,
        "min_share_of_S": None if min_share is None else round(min_share, 3),
        "breaks": breaks,
    }
    if breaks:
        return {
            "status": "FAIL",
            "reason": f"{len(breaks)} judged row(s) broke: "
            + "; ".join(
                f"row {b['row']} read {b['read']}"
                + (f" vs S={b['reference_read']}" if b["reference_read"] is not None else " with no S yet")
                for b in breaks[:5]
            )
            + (f"; +{len(breaks) - 5} more" if len(breaks) > 5 else "")
            + f" ({coverage})",
            **common,
        }
    if unusable:
        return {"status": "UNMEASURED", "reason": "; ".join(unusable) + f" ({coverage})", **common}
    if judged < MIN_JUDGED_ROWS:
        return {
            "status": "UNMEASURED",
            "reason": f"need >= {MIN_JUDGED_ROWS} judged single-call rows (one establishes S, one is compared); {coverage}",
            **common,
        }
    return {"status": "PASS", "reason": f"no judged row read below S={stable} ({coverage})", **common}


_YAML_FENCE_RE = re.compile(r"^```yaml[ \t]*\n(.*?)^```[ \t]*$", re.M | re.S)


def read_regression_threshold(config_path: Path | str | None = None) -> float:
    """|regression_threshold| of the ``cache_read_ratio`` entry in the project config ("0%" -> 0.0).

    Parses the config's ``yaml`` fence(s) with PyYAML (a locked transitive dependency of the project,
    imported lazily; no pyproject change). Any problem raises ValueError.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read config {path}: {exc}") from exc
    try:
        import yaml
    except ImportError as exc:  # PyYAML is in every synced environment; a bare interpreter lands here
        raise ValueError("PyYAML is not importable (run via `uv run python ...`)") from exc
    fences = _YAML_FENCE_RE.findall(text)
    if not fences:
        raise ValueError(f"no ```yaml fence in {path}")
    for body in fences:
        try:
            doc = yaml.safe_load(body)
        except yaml.YAMLError as exc:
            raise ValueError(f"yaml fence in {path} does not parse: {exc}") from exc
        metrics = doc.get("metrics") if isinstance(doc, dict) else None
        if not isinstance(metrics, list):
            continue
        for entry in metrics:
            if isinstance(entry, dict) and entry.get("name") == "cache_read_ratio":
                raw = entry.get("regression_threshold")
                m = re.fullmatch(r"\s*[-+]?(\d+(?:\.\d+)?)%\s*", raw) if isinstance(raw, str) else None
                if not m:
                    raise ValueError(f"cache_read_ratio entry in {path}: regression_threshold {raw!r} is not N%")
                return abs(float(m.group(1)))
    raise ValueError(f"no cache_read_ratio entry in the yaml fence(s) of {path}")


def format_cache_break_line(result: dict) -> str:
    return f"cache_break_check: {result['status']} ({result['reason']})"


_CACHE_BREAK_EXIT = {"PASS": 0, "FAIL": 1}


def _exit_code_for(result: object) -> int:
    """0 PASS / 1 FAIL / 2 UNMEASURED, a missing or unknown status, or anything else."""
    status = result.get("status") if isinstance(result, dict) else None
    return _CACHE_BREAK_EXIT.get(status, 2)


def _read_rows_fail_closed(path: Path) -> list[dict]:
    """Rows from the shared streaming reader, refusing to continue if it skipped any line.

    A torn or corrupt line could be the very row that shows a break, so a verdict from the surviving
    rows could turn a FAIL into a PASS. The shared reader skips such lines with a warning; this wrapper
    cross-checks the number of rows it returned against the file's non-blank lines (same line
    iteration), which does not depend on how logging is configured.
    """
    from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt

    if not path.exists():  # a provider that never logs usage (e.g. --provider fake): no rows, not an error
        return []
    with open(path, encoding="utf-8") as fh:
        non_blank = sum(1 for raw in fh if raw.rstrip("\r\n").strip())
    rows = list(iter_jsonl_skipping_corrupt(path))
    skipped = non_blank - len(rows)
    if skipped > 0:
        raise ValueError(
            f"{path}: the shared jsonl reader skipped {skipped} corrupt or non-object line(s) (its warnings, when logging "
            "is enabled, name them); refusing to judge from the remaining rows. Slice the file past them (e.g. tail -n +K), or "
            "repair them; corruption left over from an earlier run outside the slice needs slicing past it too"
        )
    return rows


def cache_break_cli(path: Path, from_row: int, threshold_override: float | None) -> int:
    """`--cache-break-check`: exit 0 PASS / 1 FAIL / 2 UNMEASURED or any error (never 1 for an error)."""
    try:
        if from_row < 1:
            raise ValueError("--from-row must be >= 1 (1-based over chat rows)")
        if not path.exists():
            raise ValueError(f"no such file: {path}")
        threshold = threshold_override if threshold_override is not None else read_regression_threshold()
        chat = [r for r in _read_rows_fail_closed(path) if r.get("call_type") == "chat"]
        result = check_cache_break(chat[from_row - 1 :], threshold_pct=threshold)
    except Exception as exc:  # noqa: BLE001 - an error must be exit 2, never exit 1 (= FAIL)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return _exit_code_for(result)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _summarise(usage_rows: list[dict], debug_rows: list[dict], *, turns: int, rows_error: str | None = None) -> dict:
    creation = [float(r.get("cache_creation_input_tokens", 0) or 0) for r in usage_rows]
    read = [float(r.get("cache_read_input_tokens", 0) or 0) for r in usage_rows]
    hashes = [r.get("system_sha256") for r in debug_rows if r.get("system_sha256")]
    distinct_hashes = sorted(set(hashes))

    # C1: all same-session system hashes identical (only meaningful on the new
    # build, which writes cache_debug.jsonl).
    if hashes:
        c1 = {
            "available": True,
            "distinct_system_sha256": len(distinct_hashes),
            "byte_stable": len(distinct_hashes) == 1,
            "sample": distinct_hashes[:3],
        }
    else:
        c1 = {"available": False, "note": "no cache_debug.jsonl rows (old build / NELL_CACHE_DEBUG unset)"}

    if rows_error is not None:
        cache_break = _unmeasured(rows_error)
    else:
        try:
            cache_break = check_cache_break(usage_rows, threshold_pct=read_regression_threshold())
        except ValueError as exc:
            cache_break = _unmeasured(f"threshold unreadable: {exc}")

    return {
        "turns_requested": turns,
        "chat_rows_observed": len(usage_rows),
        "c1_system_byte_stability": c1,
        "c8_cache": {
            "mean_cache_creation": round(_mean(creation), 1),
            "mean_cache_read": round(_mean(read), 1),
            "cache_creation_series": creation,
            "cache_read_series": read,
        },
        "c9_history_caching": {
            "note": "decide by comparison: does NEW mean_cache_creation drop toward "
            "'new exchange + volatile' vs OLD? if yes, CLI breakpoints the user "
            "message and A+ captured most of B; if it stays at history scale, "
            "Option B is required.",
            "last_turn_cache_creation": creation[-1] if creation else None,
        },
        "cache_break_check": cache_break,
    }


def compare(old_path: Path, new_path: Path) -> int:
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    def _verdict(js: dict) -> str:
        cb = js.get("cache_break_check")
        if isinstance(cb, dict) and "status" in cb and "reason" in cb:
            return format_cache_break_line(cb)
        return "cache_break_check: n/a (run predates the check)"

    def _status(js: dict) -> str:
        cb = js.get("cache_break_check")
        return str(cb.get("status")) if isinstance(cb, dict) and cb.get("status") else "n/a"

    # Diagnostics go to stderr so stdout stays the A/B report.
    if bool(old.get("warmup_turn")) != bool(new.get("warmup_turn")):
        print(
            f"WARNING: warmup_turn differs (old={bool(old.get('warmup_turn'))}, new={bool(new.get('warmup_turn'))}): "
            "the cold warm-up row is in only one arm's means and series, which skews every number below. "
            f"Re-run both arms with the same setting (--no-warmup on both, or neither). "
            f"(cache_break_check: old {_status(old)}, new {_status(new)})",
            file=sys.stderr,
        )
    if bool(old.get("plain_turns")) != bool(new.get("plain_turns")):
        print(
            f"WARNING: plain_turns differs (old={bool(old.get('plain_turns'))}, new={bool(new.get('plain_turns'))}): "
            "the plain micro-ack rows are in only one arm's means and series, which skews every number below. "
            f"Re-run both arms with the same setting (--no-plain-turns on both, or neither). "
            f"(cache_break_check: old {_status(old)}, new {_status(new)})",
            file=sys.stderr,
        )
    print(f"old {_verdict(old)}", file=sys.stderr)
    print(f"new {_verdict(new)}", file=sys.stderr)
    oc = old["c8_cache"]["mean_cache_creation"]
    nc = new["c8_cache"]["mean_cache_creation"]
    orr = old["c8_cache"]["mean_cache_read"]
    nr = new["c8_cache"]["mean_cache_read"]
    print("# cache replay A/B (old → new)\n")
    print(f"mean cache_creation/turn: {oc:.0f} → {nc:.0f}  ({_pct(oc, nc)})")
    print(f"mean cache_read/turn:     {orr:.0f} → {nr:.0f}  ({_pct(orr, nr)})")
    # C8's GATED signal is a material, consistent drop in cache_creation/turn — the
    # frozen system block shifting from create→read. The "corresponding read rise"
    # is real but, when history dominates the read, masked in the mean (a ~4K system
    # block shifting is swamped by ~34K of history read). So gate on the creation
    # drop and report read as context (per 1.5-criteria C8: "the gating signal is the
    # DIRECTION (create↓), not a precise token count"). Sanity floor: read must not
    # collapse (that would mean the prompt structure broke, not the system block froze).
    create_drop_pct = ((oc - nc) / oc * 100) if oc else 0.0
    read_ok = nr >= 0.5 * orr  # read didn't collapse
    c8_pass = create_drop_pct >= 5.0 and read_ok
    print(
        f"\nC8 (system-block cache stops re-creating): {'PASS' if c8_pass else 'FAIL'}"
        f"  — cache_creation/turn {-create_drop_pct:+.0f}% (gated: want a material drop);"
        f" read {_pct(orr, nr)} (context, history-dominated)"
    )
    c1 = new.get("c1_system_byte_stability", {})
    if c1.get("available"):
        print(f"C1 (frozen system byte-stable, new build): "
              f"{'PASS' if c1.get('byte_stable') else 'FAIL'} "
              f"(distinct system hashes: {c1.get('distinct_system_sha256')})")
    else:
        print("C1: n/a (new-build run had no cache_debug.jsonl — set NELL_CACHE_DEBUG=1)")
    print("\nC9 (history caching, measure-and-decide): compare last-turn cache_creation —")
    print(f"  old last turn: {old['c9_history_caching'].get('last_turn_cache_creation')}")
    print(f"  new last turn: {new['c9_history_caching'].get('last_turn_cache_creation')}")

    # Per-turn trend — the A+ tell. If NEW's read CLIMBS with turn number (history
    # accumulating, append-only) while creation stays flat, A+ is working. If NEW's
    # read stays floored while OLD's climbs, the history isn't caching (windowing or
    # no user-message breakpoint).
    ocs = old["c8_cache"]["cache_creation_series"]
    ors = old["c8_cache"]["cache_read_series"]
    ncs = new["c8_cache"]["cache_creation_series"]
    nrs = new["c8_cache"]["cache_read_series"]

    def _g(xs: list, j: int) -> str:
        return f"{xs[j]:.0f}" if j < len(xs) else "-"

    print("\nper-turn trend (create / read):")
    print(f"  {'turn':>4}  {'OLD create':>11} {'OLD read':>9}   {'NEW create':>11} {'NEW read':>9}")
    for i in range(max(len(ocs), len(ncs))):
        print(f"  {i + 1:>4}  {_g(ocs, i):>11} {_g(ors, i):>9}   {_g(ncs, i):>11} {_g(nrs, i):>9}")

    # C7 side-by-side: if both runs dumped replies (sibling <out>.replies.json),
    # print old-vs-new per prompt so the human judge can score voice + ambient-
    # context use WITHOUT needing a reference persona — the A/B IS the reference.
    old_replies = _sibling_replies(old_path)
    new_replies = _sibling_replies(new_path)
    if old_replies and new_replies:
        print("\n" + "=" * 78)
        print("C7 (human rubric) — OLD vs NEW replies on identical prompts.")
        print("Judge each pair: (a) voice fidelity, (b) does NEW still use the ambient")
        print("emotion/body/recall now that it sits at the tail? Fail = NEW reads flatter,")
        print("or treats the ambient tail as the task instead of answering the prompt.")
        print("=" * 78)
        by_turn = {r["turn"]: r for r in new_replies}
        for o in old_replies:
            n = by_turn.get(o["turn"], {})
            print(f"\n── turn {o['turn']} — prompt ──\n{o['prompt']}")
            print(f"\n[OLD]\n{o.get('reply', '').strip()}")
            print(f"\n[NEW]\n{n.get('reply', '').strip()}")
            print("\n" + "-" * 78)
    else:
        print("\n(C7 side-by-side unavailable — re-run both arms with --dump-replies)")
    return 0 if c8_pass else 1


def _sibling_replies(metrics_path: Path) -> list[dict]:
    """Load the `<metrics>.replies.json` sibling written by --dump-replies, if any."""
    sib = metrics_path.with_suffix(".replies.json")
    if not sib.exists():
        return []
    try:
        return json.loads(sib.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _pct(a: float, b: float) -> str:
    if not a:
        return "n/a"
    return f"{(b - a) / a * 100:+.0f}%"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--persona-dir", type=Path, help="isolated persona directory to run against")
    p.add_argument("--scratch", action="store_true", help="create a throwaway persona under a temp dir")
    p.add_argument("--turns", type=int, default=6, help="number of chat turns (≥6 recommended; the warm-up and the two plain turns are extra)")
    p.add_argument("--gap-s", type=float, default=3.0, help="seconds between turns (keep < 5min TTL)")
    p.add_argument("--provider", default="claude-cli", help="provider name (claude-cli | fake)")
    p.add_argument("--out", type=Path, help="write the metrics JSON to this path")
    p.add_argument(
        "--dump-replies",
        action="store_true",
        help="also write prompt+reply text to <out>.replies.json (for the C7 side-by-side; requires --out)",
    )
    p.add_argument(
        "--no-seed",
        action="store_true",
        help="do NOT seed the persona with memory fixtures (volatile tail may be near-empty)",
    )
    p.add_argument(
        "--with-tools",
        action="store_true",
        help="keep MCP tools enabled: measures the production tool-bearing chat path, which logs "
        "usage rows like the text path (default strips tools and measures the no-tools text path)",
    )
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the fixed warm-up turn (the run's cold first row, never judged by cache_break_check); "
        "use together with --no-plain-turns, otherwise the leading plain turn becomes the never-judged "
        "first row and a tools-path run has one judgeable row (UNMEASURED)",
    )
    p.add_argument(
        "--no-plain-turns",
        action="store_true",
        help="skip the leading and trailing plain micro-ack turns (the single-call rows cache_break_check "
        "judges; without them the verdict is usually UNMEASURED on the tools path)",
    )
    p.add_argument(
        "--cache-break-check",
        type=Path,
        metavar="USAGE.jsonl",
        help="run the per-run cache-break check on this chat_usage.jsonl instead of a replay "
        "(exit 0 PASS / 1 FAIL / 2 UNMEASURED or error; run it via `uv run python`)",
    )
    p.add_argument(
        "--from-row",
        type=int,
        default=1,
        help="with --cache-break-check: 1-based index over the file's chat rows where the run starts",
    )
    p.add_argument(
        "--regression-threshold",
        type=float,
        metavar="PCT",
        help="with --cache-break-check: override the percentage in [0, 100) read from the config's cache_read_ratio entry",
    )
    p.add_argument(
        "--history-file",
        type=Path,
        help="seed a real conversation buffer (active_conversations JSONL) as prior history",
    )
    p.add_argument(
        "--history-msgs",
        type=int,
        help="keep only the LAST N msgs of --history-file (use <80 for the append-only A+ test; "
        "omit to replay the full file where the sliding window defeats A+)",
    )
    p.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("OLD.json", "NEW.json"),
        help="compare two prior run outputs instead of running a replay",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.compare:
        return compare(args.compare[0], args.compare[1])
    if args.cache_break_check:
        return cache_break_cli(args.cache_break_check, args.from_row, args.regression_threshold)

    persona_dir = args.persona_dir
    if args.scratch:
        persona_dir = Path(tempfile.mkdtemp(prefix="cache-replay-")) / "personas" / "replay"
        _seed_scratch_persona(persona_dir)
        print(f"# scratch persona: {persona_dir}", file=sys.stderr)
    if persona_dir is None:
        print(
            "SKIP: pass --persona-dir <isolated dir> or --scratch. Refusing to "
            "run against a live persona.",
            file=sys.stderr,
        )
        return 2
    if args.turns < 1:
        print("ERROR: --turns must be ≥ 1", file=sys.stderr)
        return 1
    if args.dump_replies and not args.out:
        print("ERROR: --dump-replies requires --out (replies go to <out>.replies.json)", file=sys.stderr)
        return 1

    summary, replies = run_replay(
        persona_dir,
        turns=args.turns,
        gap_s=args.gap_s,
        provider_name=args.provider,
        seed=not args.no_seed,
        force_text_path=not args.with_tools,
        history_file=args.history_file,
        history_msgs=args.history_msgs,
        warmup=not args.no_warmup,
        plain_turns=not args.no_plain_turns,
    )
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"\nmetrics: {args.out}", file=sys.stderr)
        if args.dump_replies:
            replies_path = args.out.with_suffix(".replies.json")
            replies_path.write_text(json.dumps(replies, indent=2), encoding="utf-8")
            print(f"replies: {replies_path}", file=sys.stderr)
    print(text)
    return _exit_code_for(summary.get("cache_break_check"))


if __name__ == "__main__":
    raise SystemExit(main())
