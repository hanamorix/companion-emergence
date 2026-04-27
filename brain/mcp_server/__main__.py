"""Entry: `python -m brain.mcp_server --persona-dir <path>`."""

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
    args = parser.parse_args(argv)
    run_server(args.persona_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
