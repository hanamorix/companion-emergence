"""CLI: `nell kindled my-code` + `connect` build the argparser + dispatch to the bridge."""
from __future__ import annotations

from brain.cli import _build_parser


def test_my_code_subcommand_parses():
    parser = _build_parser()
    args = parser.parse_args(["kindled", "my-code", "--persona", "nell"])
    assert args.kindled_action == "my-code"
    assert args.persona == "nell"
    assert callable(args.func)


def test_connect_subcommand_parses():
    parser = _build_parser()
    args = parser.parse_args(["kindled", "connect", "--persona", "nell", "kindled1:abc"])
    assert args.kindled_action == "connect"
    assert args.persona == "nell"
    assert args.code == "kindled1:abc"
    assert callable(args.func)
