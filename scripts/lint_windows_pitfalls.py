#!/usr/bin/env python3
"""lint_windows_pitfalls.py — static check for the recurring Windows footguns.

The developer has no Windows host and the `#[cfg(target_os="windows")]` Rust arm
isn't even compiled on macOS, so Windows subprocess/boot regressions ship
test-verified-on-macOS-only and are caught in production. This greps brain/ for
the specific, hard-won footguns documented in CLAUDE.md so a re-introduction is
caught locally.

Checks:
  1. subprocess.run/Popen WITHOUT `encoding=`. Windows' default subprocess
     encoding is not UTF-8 → mojibake/crashes in the provider (v0.0.12-alpha.4).
  2. os.kill(...) — on Windows this is TerminateProcess (no cleanup, no
     shutdown_clean). Bridge shutdown must go through BridgeShutdownController
     (v0.0.33 cross-platform shutdown).

(The v0.0.37 "bare print under pythonw crashes" footgun is NOT checked: it is
neutralized globally by cli._harden_std_streams, which points a None stdout/
stderr at os.devnull at the top of main(). Re-add a targeted check only if that
hardening is ever removed.)

Advisory by default (exit 0). `--strict` exits 1 on any finding — wire into the
ruff-clean gate / release_preflight if you want it blocking.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAIN = ROOT / "brain"

_SPAWN_RE = re.compile(r"subprocess\.(?:run|Popen|call|check_output|check_call)\s*\(")


def _py_files():
    return [p for p in BRAIN.rglob("*.py") if "__pycache__" not in p.parts]


def scan() -> list[str]:
    findings: list[str] = []
    for path in _py_files():
        rel = str(path.relative_to(ROOT))
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # 1. subprocess spawn without encoding= (only for text=/universal_newlines
        #    calls — a bytes-mode spawn legitimately has no encoding).
        for m in _SPAWN_RE.finditer(text):
            start = m.start()
            span = text[start : start + 500]  # ~the call's arg list
            lineno = text.count("\n", 0, start) + 1
            uses_text = "text=True" in span or "universal_newlines=True" in span
            if uses_text and "encoding=" not in span:
                findings.append(
                    f"{rel}:{lineno}: text-mode subprocess spawn without encoding= — "
                    f"Windows default is not UTF-8. Pass encoding='utf-8'."
                )

        # 2. os.kill() with a real signal (not a `, 0)` liveness probe, which
        #    Python emulates cross-platform; and not a mention inside a string/
        #    comment).
        for i, line in enumerate(lines):
            idx = line.find("os.kill(")
            if idx == -1:
                continue
            before = line[:idx]
            if '"' in before or "'" in before or "#" in before:
                continue  # docstring / comment mention
            if re.search(r"os\.kill\([^,]+,\s*0\s*\)", line[idx:]):
                continue  # liveness probe (fine on Windows)
            findings.append(
                f"{rel}:{i + 1}: os.kill() with a signal — TerminateProcess on "
                f"Windows (no cleanup, no shutdown_clean). Route bridge shutdown "
                f"through BridgeShutdownController."
            )
    return findings


def main() -> int:
    findings = scan()
    print("== Windows-pitfall lint ==\n")
    if not findings:
        print("  ✅ no known Windows footguns found.")
        return 0
    for f in findings:
        print(f"  ⚠️  {f}")
    print(f"\n{len(findings)} finding(s). Advisory — review each against CLAUDE.md's "
          "Windows gotchas; some may be intentional/guarded.")
    return 1 if "--strict" in sys.argv else 0


if __name__ == "__main__":
    raise SystemExit(main())
