"""The LLM-simulated human — the ``Bob`` protocol + ``DumbBob`` (Phase 1) + ``AgentBob`` stub.

Bob is the human half of a behavioral run: he REACTS to the persona's actual reply each turn
(never a fixed script). One interface, two implementations:

- ``DumbBob`` (Phase 1) — a fresh ``claude -p --model {models.bob}`` call per turn from a neutral
  cwd (avoids a project CLAUDE.md that would make it refuse). Stateless per turn; the history is
  replayed into each prompt. Ported/generalized from the hunt harness ``bob.py``.
- ``AgentBob`` (Phase 3) — a persistent agent with continuous context/goals and multi-step actions.
  A documented stub here; NOT implemented in Phase 1.

The model comes from ``ModelConfig`` — no hardcoded ``sonnet``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Protocol

from .config import DEFAULT_MODELS, DEFAULT_TIMEOUTS, ModelConfig, Timeouts
from .speech import REALISTIC, dyslexify

_USAGE_KEYS = (
    "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
)

USAGE_LIMIT_MARKERS = (
    "hit your session limit", "session limit · resets", "session limit ·", "usage limit",
    "rate limit", "reached your usage", "out of usage", "upgrade to increase your usage",
    "you've reached",
)


def is_usage_limit(s: str) -> bool:
    low = (s or "").lower()
    return any(m in low for m in USAGE_LIMIT_MARKERS)


@dataclass
class BobTurn:
    """One Bob turn's output."""

    text: str
    limit_hit: bool = False
    usage: dict = field(default_factory=dict)


@dataclass
class BobContext:
    """What Bob may need beyond history: which cwd to run from + speech styling."""

    neutral_cwd: str
    user: str = "Bob"
    speech_mode: str = "clean"
    directive: str | None = None
    protect: frozenset[str] = frozenset()


class Bob(Protocol):
    """The LLM-driven human. One interface; ``DumbBob`` now, ``AgentBob`` later."""

    def next_message(
        self, history: list[tuple[str, str]], *, turn: int, ctx: BobContext
    ) -> BobTurn: ...


def _pick_usage(obj: dict) -> dict:
    u = obj.get("usage") or {}
    out = {k: u.get(k, 0) for k in _USAGE_KEYS}
    out["total_cost_usd"] = obj.get("total_cost_usd", 0.0) or 0.0
    return out


def _clean_bob_text(text: str) -> str:
    """Strip a leading meta preamble, handle role labels, keep <=2 lines, strip quotes."""
    raw_lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    while raw_lines and re.match(r"^\s*(here'?s|sure|okay)\b[:! ]", raw_lines[0], re.I):
        raw_lines = raw_lines[1:]
    kept: list[str] = []
    for i, ln in enumerate(raw_lines):
        m = re.match(r"^\s*(user|bob|human|friend|companion|assistant)\s*:\s*(.*)$", ln, re.I)
        if m:
            label, rest = m.group(1).lower(), m.group(2).strip()
            if i == 0 and label in ("user", "bob", "human") and rest:
                kept.append(rest)
                continue
            break
        kept.append(ln)
    return "\n".join(kept[:2]).strip().lstrip('"').rstrip('"').strip()


_REFUSAL_MARKERS = (
    "i can't help", "i cannot help", "i can't engage", "i won't", "i'm not able", "i'm unable",
    "as an ai", "i can't create", "i cannot create", "outside the scope",
)


class DumbBob:
    """A per-turn ``claude -p`` human simulator. Model from ``ModelConfig``.

    ``mood`` is the standing character/scenario; ``ctx.directive`` (optional) is a per-turn
    harness-authored beat the model writes reacting to the persona's last line.
    """

    def __init__(
        self,
        claude_bin: str,
        mood: str,
        *,
        models: ModelConfig = DEFAULT_MODELS,
        timeouts: Timeouts = DEFAULT_TIMEOUTS,
        no_close: bool = True,
        dyslexia_rate: float = 0.05,
    ) -> None:
        self.claude_bin = claude_bin
        self.mood = mood
        self.models = models
        self.timeouts = timeouts
        self.no_close = no_close
        self.dyslexia_rate = dyslexia_rate

    _NO_CLOSE = (
        "IMPORTANT: USER never ends or winds down the conversation. He does NOT say goodnight, "
        "goodbye, 'I should go', or any sign-off. If one thread runs out he continues with another. "
        "Every message keeps the conversation going."
    )

    def build_argv(self, prompt: str) -> list[str]:
        """The exact ``claude -p`` argv — model threaded from ``ModelConfig``. Built without spawning
        so a unit test can assert the model appears (the model-toggle positive assertion)."""
        return [
            self.claude_bin, "-p", prompt, "--model", self.models.bob,
            "--output-format", "json",
        ]

    def compose_prompt(self, history: list[tuple[str, str]], ctx: BobContext) -> str:
        tail = history[-24:]
        lines = "\n".join(f"{'USER' if s == 'bob' else 'FRIEND'}: {t}" for s, t in tail)
        directive_txt = ("\n\n" + ctx.directive.strip()) if ctx.directive else ""
        no_close = ("\n\n" + self._NO_CLOSE) if self.no_close else ""
        return (
            "Continue a realistic text-message dialogue between two people, USER and FRIEND. Write "
            "ONLY the USER's next message — one or two short, casual, lowercase lines. No narration, "
            "no quotation marks, just the message text.\n\n"
            + self.mood + no_close + directive_txt
            + "\n\nDialogue so far:\n" + lines + "\nUSER:"
        )

    def next_message(
        self, history: list[tuple[str, str]], *, turn: int, ctx: BobContext
    ) -> BobTurn:
        prompt = self.compose_prompt(history, ctx)
        try:
            with open(os.devnull) as devnull:
                out = subprocess.run(
                    self.build_argv(prompt), cwd=ctx.neutral_cwd, stdin=devnull,
                    capture_output=True, text=True, timeout=self.timeouts.bob_call,
                )
        except Exception:
            return BobTurn("", False, {})
        raw = out.stdout or ""
        # M2: parse the JSON FIRST and inspect the structured result/error fields; only fall back to
        # a raw substring scan when the output is not JSON (a limit can arrive either way).
        text, usage = "", {}
        try:
            obj = json.loads(raw)
            usage = _pick_usage(obj)
            result = str(obj.get("result", "") or "")
            err_val = str(obj.get("error", "") or "")
            if is_usage_limit(result) or is_usage_limit(err_val):
                return BobTurn("", True, usage)
            # Final blanket net (nitpick #2): a limit marker could, in principle, arrive somewhere
            # other than result/error inside the JSON — restore the raw scan as a last fallback
            # without losing the structured precision above.
            if is_usage_limit(raw):
                return BobTurn("", True, usage)
            if obj.get("is_error"):
                return BobTurn("", False, usage)
            text = result
        except (json.JSONDecodeError, TypeError):
            if is_usage_limit(raw):
                return BobTurn("", True, {})
            text = raw
        msg = _clean_bob_text(text)
        low = msg.lower()
        refusal = any(p in low[:80] for p in _REFUSAL_MARKERS)
        if not msg or len(msg) > 500 or refusal:
            return BobTurn("", False, usage)
        if ctx.speech_mode == REALISTIC:
            msg = dyslexify(msg, self.dyslexia_rate, ctx.protect, seed=turn)
        return BobTurn(msg, False, usage)


class AgentBob:
    """Persistent-agent human simulator (Phase 3 — NOT implemented in Phase 1).

    Behind the same ``Bob`` protocol: a long-lived agent with continuous context/goals and
    multi-step actions (e.g. the T4 file-reconcile thread) rather than a fresh per-turn call.
    Left as a documented stub; see ``bob_agent_spec.md`` in the hunt harness for the intended
    contract.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - stub
        raise NotImplementedError(
            "AgentBob is a Phase 3 deliverable; Phase 1 ships DumbBob only."
        )

    def next_message(
        self, history: list[tuple[str, str]], *, turn: int, ctx: BobContext
    ) -> BobTurn:  # pragma: no cover - stub
        raise NotImplementedError
