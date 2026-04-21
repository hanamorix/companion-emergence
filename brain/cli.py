"""Entry point CLI for companion-emergence.

Invoked as `nell <subcommand> [options]`. Week 1 wires up `--version` and
the subcommand skeleton; subcommands themselves are stubs until their
respective weeks land.
"""

from __future__ import annotations

import argparse
import sys

from brain import __version__


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser."""
    parser = argparse.ArgumentParser(
        prog="nell",
        description=(
            "companion-emergence — CLI for building emotionally aware "
            "AI companions"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"companion-emergence {__version__}",
    )
    parser.add_subparsers(dest="command", title="subcommands")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns shell exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
