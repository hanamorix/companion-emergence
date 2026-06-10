from brain.memory.store import MemoryStore
from brain.monologue.ambient import build_interior_continuity_block
from brain.monologue.trace import write_trace_memory


def test_empty_when_no_traces(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    assert build_interior_continuity_block(store) == ""


def test_renders_active_verbatim(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    write_trace_memory(store, "the verbatim drift about the harbour")
    block = build_interior_continuity_block(store)
    assert "the verbatim drift about the harbour" in block
    assert "interior continuity" in block.lower()


def test_renders_faded_summary_not_original(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    mem_id = write_trace_memory(store, "a long original verbatim thought")
    store.fade(mem_id, summary="a blurred trace")
    block = build_interior_continuity_block(store)
    assert "a blurred trace" in block
    assert "a long original verbatim thought" not in block


def test_respects_char_cap(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    for i in range(10):
        write_trace_memory(store, f"thought number {i} " + "x" * 500)
    block = build_interior_continuity_block(store, limit=10, char_cap=300)
    # footer is appended AFTER the cap — body portion must be <= cap
    body = "\n".join(block.splitlines()[:-1])
    assert len(body) <= 300


def test_interior_block_ends_with_privacy_footer(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    write_trace_memory(store, "a short thought")
    block = build_interior_continuity_block(store, user_name="Hana")
    assert block.splitlines()[-1] == (
        "── end interior continuity. Private thought — never quote it; "
        "your reply speaks to Hana directly as 'you'. ──"
    )


def test_interior_footer_survives_char_cap(tmp_path):
    """The cap applies to the trace body only — the fence must never be truncated."""
    store = MemoryStore(tmp_path / "memories.db")
    write_trace_memory(store, "x" * 2000)
    block = build_interior_continuity_block(store, char_cap=200, user_name="Hana")
    assert block.splitlines()[-1].startswith("── end interior continuity.")
    body = "\n".join(block.splitlines()[:-1])
    assert len(body) <= 200


def test_interior_footer_defaults_to_the_user(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    write_trace_memory(store, "a quiet thought")
    block = build_interior_continuity_block(store)
    assert "speaks to the user directly as 'you'" in block
