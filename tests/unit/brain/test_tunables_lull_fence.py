"""ram-spike-fix INC-6, criterion C27 (S47/S51/I7): the lull tunable key is
registered in tunables.py itself, and tunables.py's own module docstring
fence names it as the one allowed ops-timing key. (The dev_constants.py half
of C27 — the busy timeout / gate / decay budget living there, not here — is
covered by test_dev_constants.py, INC-1, frozen; not duplicated here.)"""
from __future__ import annotations

import ast

from brain import tunables


def test_lull_key_is_registered_in_tunables_py_itself() -> None:
    assert tunables.CHAT_IDLE_LULL_SECONDS == 600.0
    tunables._reset_for_tests()
    try:
        assert tunables.get_tunable("chat.idle_lull_seconds", tunables.CHAT_IDLE_LULL_SECONDS) == 600.0
    finally:
        tunables._reset_for_tests()


def test_registration_call_site_is_in_tunables_module() -> None:
    """AST check: the `register("chat.idle_lull_seconds", ...)` call itself
    lives in brain/tunables.py (not re-exported from elsewhere pretending to
    register it locally)."""
    src = tunables.__file__ and __import__("pathlib").Path(tunables.__file__).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else None
        )
        if name != "register":
            continue
        if any(
            isinstance(a, ast.Constant) and a.value == "chat.idle_lull_seconds"
            for a in node.args
        ):
            found = True
    assert found, "chat.idle_lull_seconds must be registered in tunables.py"


def test_module_docstring_fence_names_the_one_allowed_key() -> None:
    doc = tunables.__doc__ or ""
    assert "chat.idle_lull_seconds" in doc
    assert "ONE named exception" in doc or "one allowed" in doc.lower()


def test_no_second_ops_timing_key_registered_in_tunables_py() -> None:
    """The fence's own claim ("no other ops-timing key may be registered
    here") checked against the actual file: every tunables.register(...)
    call site in tunables.py itself is either the lull key or the pre-
    existing throttle.max_concurrent_background cap (a concurrency cap, not
    a TIMING value — explicitly carved out by 2-plan §3.1: "it is not an
    idle value"). Registrations made by OTHER modules (cli_throttle.py,
    pass2_queue.py before this increment, provider.py's unrelated streaming
    timeout, etc.) are out of scope for this specific fence, which only
    binds tunables.py's own module."""
    import inspect

    src = inspect.getsource(tunables)
    tree = ast.parse(src)
    registered_keys = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else None
        )
        if name == "register" and node.args and isinstance(node.args[0], ast.Constant):
            registered_keys.append(node.args[0].value)
    assert set(registered_keys) == {"chat.idle_lull_seconds"}, (
        f"unexpected registrations directly in tunables.py: {registered_keys}"
    )
