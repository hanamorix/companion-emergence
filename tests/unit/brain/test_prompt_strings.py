"""Canaries for the prompt-strings accessor (issue #129 stage 1).

Unlike ``tunables.py`` this module is fail-CLOSED by design: a missing key,
a missing/malformed file, or a wrong-type value at a registered key must
raise ``PromptStringError`` immediately, never fall back silently. These
tests cover the real happy-path keys in the shipped ``prompt_strings.toml``
plus every fail-closed branch the module implements.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def prompt_strings():
    import brain.prompt_strings as ps

    ps._reset_for_tests()
    yield ps
    ps._reset_for_tests()


def test_get_returns_exact_string_for_known_key(prompt_strings):
    assert prompt_strings.get("bridge.feed.type_opener.dream") == "I dreamed"


def test_register_returns_exact_string_and_records(prompt_strings):
    assert prompt_strings.register("bridge.feed.type_opener.dream") == "I dreamed"
    assert "bridge.feed.type_opener.dream" in prompt_strings._registry


def test_get_segments_returns_exact_list_for_known_key(prompt_strings):
    assert prompt_strings.get_segments(
        "attunement.prompts.category_restriction_suffix_segments"
    ) == [
        "\n\nFOR THIS PASS ONLY: extract candidates for these categories: ",
        ". Do NOT emit candidates for any other category.",
    ]


def test_register_segments_returns_exact_list_and_records(prompt_strings):
    key = "attunement.prompts.category_restriction_suffix_segments"
    assert prompt_strings.register_segments(key) == [
        "\n\nFOR THIS PASS ONLY: extract candidates for these categories: ",
        ". Do NOT emit candidates for any other category.",
    ]
    assert key in prompt_strings._registry


def test_get_missing_key_raises(prompt_strings):
    with pytest.raises(prompt_strings.PromptStringError):
        prompt_strings.get("nonexistent.totally_missing_key")


def test_register_missing_key_raises(prompt_strings):
    with pytest.raises(prompt_strings.PromptStringError):
        prompt_strings.register("nonexistent.totally_missing_key")


def test_get_segments_missing_key_raises(prompt_strings):
    with pytest.raises(prompt_strings.PromptStringError):
        prompt_strings.get_segments("nonexistent.totally_missing_key")


def test_register_segments_missing_key_raises(prompt_strings):
    with pytest.raises(prompt_strings.PromptStringError):
        prompt_strings.register_segments("nonexistent.totally_missing_key")


def test_get_on_list_value_raises_wrong_type(prompt_strings):
    # A real segments key is a list, not a plain string — get() must reject it.
    with pytest.raises(prompt_strings.PromptStringError):
        prompt_strings.get("attunement.prompts.category_restriction_suffix_segments")


def test_get_segments_on_string_value_raises_wrong_type(prompt_strings):
    # A real plain-string key is not a list — get_segments() must reject it.
    with pytest.raises(prompt_strings.PromptStringError):
        prompt_strings.get_segments("bridge.feed.type_opener.dream")


def test_get_segments_on_list_with_non_string_element_raises(prompt_strings, monkeypatch, tmp_path):
    toml_path = tmp_path / "prompt_strings.toml"
    toml_path.write_text(
        'bad = { segments = ["ok", 5] }\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(prompt_strings, "_file_path", lambda: toml_path)
    with pytest.raises(prompt_strings.PromptStringError):
        prompt_strings.get_segments("bad.segments")


def test_the_bridge_server_registers_no_prompt_it_does_not_use():
    """Review S14 (#456): the close heartbeat is gone, so its persona prompt must not stay registered (or in the
    TOML) as a dead string. Every prompt the bridge server registers is used in that module."""
    import ast
    from pathlib import Path

    import brain.bridge.server as server

    tree = ast.parse(Path(server.__file__).read_text(encoding="utf-8"))
    registered = [
        t.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr in ("register", "register_segments")
        for t in node.targets
        if isinstance(t, ast.Name)
    ]
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert [n for n in registered if n not in used] == []
    toml = (Path(server.__file__).parents[1] / "prompt_strings.toml").read_text(encoding="utf-8")
    assert "heartbeat_close_system_prompt_segments" not in toml
