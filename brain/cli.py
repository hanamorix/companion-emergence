"""Entry point CLI for companion-emergence.

Invoked as `nell <subcommand> [options]`. Week 1 ships `--version`, help,
and a set of stub subcommands that print "not implemented yet" so the CLI
surface is stable while subsequent weeks fill in functionality.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from brain import __version__
from brain.migrator.cli import build_parser as _build_migrate_parser

# Subcommands the framework plans to ship. Each is a stub in Week 1;
# filled in across Weeks 2-8 as respective modules come online.
_STUB_COMMANDS: tuple[str, ...] = (
    "supervisor",
    "dream",
    "heartbeat",
    "reflex",
    "status",
    "rest",
    "soul",
    "memory",
    "works",
)


def _make_stub(name: str) -> Callable[[argparse.Namespace], int]:
    """Factory: build a stub command handler that prints + returns 0.

    The returned handler accepts `args: argparse.Namespace` as required by
    the `args.func(args)` dispatch protocol — stubs don't read it, but the
    signature shape is load-bearing and should not be "cleaned up" to `_args`.
    """

    def _handler(args: argparse.Namespace) -> int:
        print(
            f"nell {name} — not implemented yet. "
            "This subcommand is wired in a future week per the implementation plan."
        )
        return 0

    return _handler


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser with all stub subcommands."""
    parser = argparse.ArgumentParser(
        prog="nell",
        description=("companion-emergence — CLI for building emotionally aware AI companions"),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"companion-emergence {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", title="subcommands")

    for name in _STUB_COMMANDS:
        sub = subparsers.add_parser(
            name,
            help=f"(stub) {name} — wired in a later week",
        )
        sub.set_defaults(func=_make_stub(name))

    _build_migrate_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns shell exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
