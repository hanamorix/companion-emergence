"""dev_constants — INC-1 of the RAM-spike fix (C13, C27, C33).

Static checks that the memories.db busy timeout and the self-tune gate are
DEV-level constants (S26/I7): plain module attributes in `brain.dev_constants`,
never `tunables.register()`-ed, and that the two memories.db connect sites
(store.py, embedding_matrix.py) actually source their timeout from this
module rather than a re-hardcoded literal. Behavioral coverage (does the
30 s timeout actually let a chat write through a long transaction, does the
gate fire at 201 not 200) lives in test_busy_timeout.py and
test_judge_selftune.py respectively — this file is the "where do these
numbers live" half of C27.
"""

from __future__ import annotations

import inspect

from brain import dev_constants


def test_memories_db_busy_timeout_is_30s() -> None:
    """S48/S58: sized above the largest measured memories.db write
    transaction (clustering's set_cluster_memberships, 6.7 s on F-bob20k,
    2-plan.md §4.3) with a Phoebe-class safety margin."""
    assert dev_constants.MEMORIES_DB_BUSY_TIMEOUT_S == 30.0


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


def test_store_py_sources_busy_timeout_from_dev_constants() -> None:
    """C27/C33: the memories.db connect site in store.py uses the shared
    constant, not a re-hardcoded literal — so raising it later is a
    one-file change, and the 2-plan §4.2 report's "every connect site"
    inventory stays accurate."""
    from brain.memory import store

    source = inspect.getsource(store)
    assert "dev_constants.MEMORIES_DB_BUSY_TIMEOUT_S" in source
    assert "busy_timeout = 5000" not in source, "memories.db must no longer use the old 5s literal"


def test_embedding_matrix_py_sources_busy_timeout_from_dev_constants() -> None:
    from brain.memory import embedding_matrix

    source = inspect.getsource(embedding_matrix)
    assert "dev_constants.MEMORIES_DB_BUSY_TIMEOUT_S" in source
    assert "busy_timeout = 5000" not in source
