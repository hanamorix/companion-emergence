"""Wizard's persona-name prompt should not pre-fill 'nell' as default."""

from __future__ import annotations

import inspect

from brain import cli


def test_wizard_persona_prompt_default_is_not_nell():
    """No _prompt call in cli.py should use default='nell'.

    Regression test (Bug 3 from the Kubuntu user's report): the wizard
    pre-filled 'nell' as the persona name, causing users to pick
    confusing names like 'sigrun' or accept the default without
    realising it was one of many possible names.
    """
    src = inspect.getsource(cli)
    bad = [
        line.strip()
        for line in src.splitlines()
        if "_prompt" in line and 'default="nell"' in line
    ]
    assert not bad, f"_prompt calls still using default='nell': {bad}"
