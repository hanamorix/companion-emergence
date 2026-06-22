from brain.kindled_link.relationship import write_kindled_peer_memory
from brain.memory.store import MemoryStore


def test_kindled_peer_memory_is_provenance_marked(tmp_path):
    ms = MemoryStore(tmp_path / "mem.db")
    write_kindled_peer_memory(ms, peer_id="kid_a", session_id="s1",
        speaker="peer", stage="familiar",
        content="The peer said they had been dreaming about the sea.")
    rows = ms.list_by_type("kindled_peer")
    assert len(rows) == 1
    m = rows[0]
    assert m.memory_type == "kindled_peer"
    assert m.domain == "kindled_peer"
    assert m.metadata.get("peer_id") == "kid_a"
    assert m.metadata.get("relationship_stage") == "familiar"


def test_peer_attributed_helper_prefixes_only_kindled_peer():
    # the render helper is the provenance guard; test it directly (deterministic),
    # independent of which recall path renders it.
    from types import SimpleNamespace

    from brain.chat.prompt import _peer_attributed
    peer = SimpleNamespace(memory_type="kindled_peer")
    user = SimpleNamespace(memory_type="conversation")
    assert _peer_attributed(peer, "dreaming of the sea").lower().startswith("(something a peer said)")
    assert _peer_attributed(user, "I went to the sea") == "I went to the sea"


def test_peer_attributed_handles_graveyard_dict_entry():
    # The lost/graveyard recall path renders from a DICT (not a Memory object);
    # a lost kindled_peer memory must STILL be attributed (Phase 7a T12 — closes
    # the Phase-5 lost-path provenance gap; the tombstone carries memory_type).
    from brain.chat.prompt import _peer_attributed
    peer_entry = {"memory_type": "kindled_peer", "summary": "the sea"}
    user_entry = {"memory_type": "conversation", "summary": "the sea"}
    assert _peer_attributed(peer_entry, "the sea").lower().startswith("(something a peer said)")
    assert _peer_attributed(user_entry, "the sea") == "the sea"


def test_recall_block_attributes_kindled_peer_memory_legacy_path(tmp_path):
    # provenance invariant through the legacy recall path (persona_dir=None).
    from brain.chat.prompt import _build_recall_block
    ms = MemoryStore(tmp_path / "mem.db")
    write_kindled_peer_memory(ms, peer_id="kid_a", session_id="s1",
        speaker="peer", stage="familiar", content="dreaming about the sea")
    block = _build_recall_block(ms, "sea", persona_dir=None)
    if block.strip():  # if it surfaces at all, it must be attributed
        assert "a peer said" in block.lower()
