"""C-18 (Engine 6, #584): the modules S6 ties together import cleanly in a fresh interpreter, in any order.

``brain.utils.memory`` (``days_since_human``), ``brain.initiate.user_pattern`` (silence), ``brain.ingest.buffer``
(``last_user_turn_ts``) and ``brain.emotion.engine`` sit in one import neighbourhood: the engine imports
``brain.utils.memory``, which imports the memory store, which imports the engine lazily, and ``brain.ingest`` imports the
store and the provider stack. A top-level import in the wrong place makes a cycle that only shows when a particular module is
imported first, and an in-process test cannot see it because pytest has already imported everything. So each order runs in
its own interpreter.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULES = (
    "brain.utils.memory",
    "brain.initiate.user_pattern",
    "brain.ingest.buffer",
    "brain.emotion.engine",
)


def _orders() -> list[tuple[str, ...]]:
    """Each module first, in the listed order and in the reverse order: eight distinct orders."""
    forward, backward = list(_MODULES), list(reversed(_MODULES))
    found: list[tuple[str, ...]] = []
    for base in (forward, backward):
        for shift in range(len(base)):
            found.append(tuple(base[shift:] + base[:shift]))
    return found


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=300, check=False
    )


@pytest.mark.parametrize("order", _orders(), ids=lambda order: "-".join(m.rsplit(".", 1)[1] for m in order))
def test_the_s6_modules_import_in_a_fresh_interpreter_in_any_order(order: tuple[str, ...]) -> None:
    code = "\n".join(f"import {name}" for name in order)
    code += "\nfrom brain.ingest.buffer import last_user_turn_ts\nfrom brain.utils.memory import days_since_human\n"
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_the_ingest_stack_loads_when_days_since_human_is_first_called_not_when_the_engine_is_imported() -> None:
    """``days_since_human`` imports ``last_user_turn_ts`` inside its body, so a process that only imports the engine does not
    load the ingest pipeline (and through it the provider stack); the first call does, which shows the function reaches it."""
    code = (
        "import sys, tempfile\n"
        "from datetime import UTC, datetime\n"
        "from pathlib import Path\n"
        "import brain.emotion.engine, brain.utils.memory, brain.initiate.user_pattern\n"
        "def ingest():\n"
        "    return sorted(m for m in sys.modules if m == 'brain.ingest' or m.startswith('brain.ingest.'))\n"
        "before = ingest()\n"
        "assert not before, f'imported with the engine: {before}'\n"
        "with tempfile.TemporaryDirectory() as folder:\n"
        "    brain.utils.memory.days_since_human(None, datetime.now(UTC), persona_dir=Path(folder))\n"
        "assert 'brain.ingest.buffer' in sys.modules, 'days_since_human did not reach brain.ingest.buffer'\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr[-2000:]
