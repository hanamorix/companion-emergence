"""A fake `claude` CLI that behaves like a live print-mode session (`--input-format stream-json`), for the tests of
`ClaudeCliSession` and of the memory-sorting step's judge sessions (S71). It never talks to a model.

The fake keeps a conversation history the way the real CLI does: every user message is appended to it, and the exact
plain-text message `/clear` empties it, starts a new session id and answers with the frames the real CLI sends for a clear
(`conversation_reset`, `system/init`, then a `result` with `num_turns` 0 and an empty result; shapes copied from
docs/cli-provider-capabilities.md and the 2026-11-27 measurement). Every spawn, clear, answer and exit is appended to a log file as
one JSON line, so a test can see what each process held when it answered.

Behaviour switches (environment, set by `install`):
  FAKE_SESSION_MODE   comma list of: echo (the reply is the JSON of the history the fake holds), swallow_clear (a `/clear`
                      is confirmed with the OLD session id and the history is kept), late_clear (the clear's result is
                      delayed), dup_result (an extra result frame follows the first answer), no_reset_frames.
  FAKE_SESSION_DIE_AFTER  exit after writing the Nth answer (the process dies between items).
A prompt containing FAIL is answered with an error frame ("boom"); one containing HANG is never answered (non-result frames
trickle until stdin closes); one containing LATE_STALE gets its reply followed by a second reply for the same message.
A non-session call (`--output-format json`, prompt on stdin) is answered once with a json result, as `generate` expects.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPLY = '{"verdict": "new", "importance": 5, "names": []}'


def _main() -> None:
    argv = sys.argv[1:]
    if argv == ["--help"]:
        print("  --output-format <format>  Output format\n  --tools <tools...>  Tools")
        return
    if argv == ["--tools=", "-p"]:
        print("Error: Input must be provided", file=sys.stderr)
        sys.exit(1)
    log_path = os.environ["FAKE_SESSION_LOG"]
    modes = {m for m in os.environ.get("FAKE_SESSION_MODE", "").split(",") if m}
    die_after = int(os.environ.get("FAKE_SESSION_DIE_AFTER", "0") or 0)
    system_path = argv[argv.index("--system-prompt-file") + 1] if "--system-prompt-file" in argv else None
    system = Path(system_path).read_text(encoding="utf-8") if system_path else None
    pid = os.getpid()

    def log(**row: object) -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"pid": pid, **row}) + "\n")

    def out(obj: dict) -> None:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    if "stream-json" not in argv or "--input-format" not in argv:
        prompt = sys.stdin.read()
        log(event="oneshot", system=system, prompt=prompt)
        out({"type": "result", "subtype": "success", "is_error": False, "result": DEFAULT_REPLY, "num_turns": 1,
             "session_id": str(uuid.uuid4())})
        return

    log(event="spawn", system_path=system_path, system=system, argv=argv)
    history: list[str] = []
    session_id = str(uuid.uuid4())
    answered = 0

    def result(*, num_turns: int, text: str, is_error: bool = False) -> dict:
        return {"type": "result", "subtype": "error" if is_error else "success", "is_error": is_error,
                "result": text, "num_turns": num_turns, "session_id": session_id, "duration_api_ms": 1,
                "total_cost_usd": 0, "usage": {}}

    for line in sys.stdin:
        content = json.loads(line)["message"]["content"]
        if content == "/clear":
            log(event="clear", held=list(history))
            if "swallow_clear" not in modes:
                history = []
                session_id = str(uuid.uuid4())
            if "no_reset_frames" not in modes:
                out({"type": "conversation_reset", "trigger": "clear", "session_id": session_id})
                out({"type": "system", "subtype": "init", "session_id": session_id})
            if "late_clear" in modes:
                time.sleep(0.4)
            out(result(num_turns=0, text=""))
            continue
        history.append(content)
        log(event="message", held=list(history))
        if "HANG" in content:
            while True:
                out({"type": "system", "subtype": "hook_progress", "session_id": session_id})
                time.sleep(0.05)
        out({"type": "system", "subtype": "init", "session_id": session_id})
        out({"type": "assistant", "message": {"content": []}, "session_id": session_id})
        if "FAIL" in content:
            out(result(num_turns=1, text="boom", is_error=True))
            continue
        reply = json.dumps({"history": history}) if "echo" in modes else DEFAULT_REPLY
        answered += 1
        log(event="answer", held=list(history))
        out(result(num_turns=1, text=reply))
        if ("dup_result" in modes and answered == 1) or "LATE_STALE" in content:
            out(result(num_turns=1, text=reply))
        if die_after and answered >= die_after:
            log(event="died")
            os._exit(3)
    log(event="exit")


@dataclass
class FakeClaude:
    """What `install` returns: the log of the fake processes."""

    log_path: Path

    def events(self, kind: str | None = None) -> list[dict]:
        if not self.log_path.exists():
            return []
        rows = [json.loads(x) for x in self.log_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return [r for r in rows if kind is None or r["event"] == kind]

    def answers(self) -> list[dict]:
        """What each answering process held when it answered (the prompt just sent included)."""
        return self.events("answer")


def install(tmp_path: Path, monkeypatch, *, mode: str = "", die_after: int = 0) -> FakeClaude:
    """Put a fake `claude` first on PATH (under the temp root, which the spawn guard allows) and point the provider at it."""
    import brain.bridge.provider as provider

    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    launcher = bin_dir / "claude"
    launcher.write_text(
        f"#!{sys.executable}\nimport runpy\nrunpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')\n",
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    log_path = tmp_path / "fake-claude-log.jsonl"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_SESSION_LOG", str(log_path))
    monkeypatch.setenv("FAKE_SESSION_MODE", mode)
    monkeypatch.setenv("FAKE_SESSION_DIE_AFTER", str(die_after))
    monkeypatch.setattr(provider, "_cli_supports_tools_flag", lambda: True)
    monkeypatch.setattr(provider, "_claude_work_dir", lambda: str(tmp_path))
    return FakeClaude(log_path)


if __name__ == "__main__":
    _main()
