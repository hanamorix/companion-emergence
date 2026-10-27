"""dev_constants — INC-1 of the RAM-spike fix (C13, C27, C33).

Static checks that the self-tune gate is a DEV-level constant (S26/I7): a plain
module attribute in `brain.dev_constants`, with the owner's value, and that the
module never touches `brain.tunables`. Behavioral coverage (does the 30 s
timeout actually let a chat write through a long transaction, does the gate
fire at 201 not 200) lives in test_busy_timeout.py and test_judge_selftune.py
respectively. The values of every dev constant are held by
test_dev_constants_snapshot.py, which warns when one changes.
"""

from __future__ import annotations

import inspect

from brain import dev_constants


def test_selftune_gate_is_200() -> None:
    """S7 (owner 2026-09-26: "It should be 200!")."""
    assert dev_constants.JUDGE_SELFTUNE_GATE_HANDFUL_DECISIONS == 200


def test_dev_constants_module_never_touches_tunables() -> None:
    """I7 fence, the opposite half of tunables.py's own fence: this module
    must never IMPORT brain.tunables or CALL register()/get_tunable() — a
    dev-level constant that quietly grew a user override would defeat the
    owner's ruling (S26) that this is NOT user-facing. Checked on the code
    only (import/call statements), not the module's own prose docstring,
    which legitimately names `brain.tunables` as the sibling module it is
    NOT."""
    code_lines = [
        line
        for line in inspect.getsource(dev_constants).splitlines()
        if not line.lstrip().startswith("#")
    ]
    # Strip the module docstring (the triple-quoted block at the top) before
    # scanning for real code references.
    joined = "\n".join(code_lines)
    doc = dev_constants.__doc__ or ""
    code_only = joined.replace(doc, "", 1)
    assert "import tunables" not in code_only
    assert "from brain import tunables" not in code_only
    assert ".register(" not in code_only
    assert ".get_tunable(" not in code_only

