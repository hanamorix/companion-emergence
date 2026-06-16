"""Persisted per-(peer, session) sequence high-water mark for replay rejection,
surviving a reopen (restart)."""

from brain.kindled_link.store import KindledLinkStore


def test_high_water_defaults_zero_and_advances(tmp_path):
    db = tmp_path / "kl.db"
    s = KindledLinkStore(db)
    assert s.get_seq_high_water("kid_p", "ks_1") == 0
    s.set_seq_high_water("kid_p", "ks_1", 5)
    assert s.get_seq_high_water("kid_p", "ks_1") == 5
    # different session is independent
    assert s.get_seq_high_water("kid_p", "ks_2") == 0


def test_high_water_survives_reopen(tmp_path):
    db = tmp_path / "kl.db"
    KindledLinkStore(db).set_seq_high_water("kid_p", "ks_1", 9)
    assert KindledLinkStore(db).get_seq_high_water("kid_p", "ks_1") == 9  # persisted
