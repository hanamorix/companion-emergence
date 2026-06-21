"""Piece 0 — persisted atomic outbound sequence counter (T3 prerequisite).

next_outbound_sequence(peer_id, session_id) -> int
  - first call → 1
  - second call → 2
  - independent per session
  - survives store reopen
"""

from brain.kindled_link.store import KindledLinkStore


def test_first_call_returns_one(tmp_path):
    s = KindledLinkStore(tmp_path / "kl.db")
    assert s.next_outbound_sequence("kid_p", "ks_1") == 1


def test_second_call_returns_two(tmp_path):
    s = KindledLinkStore(tmp_path / "kl.db")
    s.next_outbound_sequence("kid_p", "ks_1")
    assert s.next_outbound_sequence("kid_p", "ks_1") == 2


def test_independent_per_session(tmp_path):
    s = KindledLinkStore(tmp_path / "kl.db")
    s.next_outbound_sequence("kid_p", "ks_1")
    s.next_outbound_sequence("kid_p", "ks_1")
    # different session starts at 1
    assert s.next_outbound_sequence("kid_p", "ks_2") == 1


def test_survives_reopen(tmp_path):
    db = tmp_path / "kl.db"
    s = KindledLinkStore(db)
    s.next_outbound_sequence("kid_p", "ks_1")
    s.next_outbound_sequence("kid_p", "ks_1")
    s.close()
    s2 = KindledLinkStore(db)
    assert s2.next_outbound_sequence("kid_p", "ks_1") == 3
