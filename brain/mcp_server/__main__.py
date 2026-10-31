"""Entry: `python -m brain.mcp_server --persona-dir <path> [--tool-prefix <prefix>]`."""

from __future__ import annotations

import argparse
from pathlib import Path

from brain.mcp_server import run_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m brain.mcp_server")
    parser.add_argument(
        "--persona-dir",
        required=True,
        type=Path,
        help="Path to the active persona directory (required).",
    )
    parser.add_argument(
        "--tool-prefix",
        default="",
        help="What the client calls these tools, e.g. mcp__brain-tools__ (default: bare names).",
    )
    args = parser.parse_args(argv)
    run_server(args.persona_dir, tool_prefix=args.tool_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
