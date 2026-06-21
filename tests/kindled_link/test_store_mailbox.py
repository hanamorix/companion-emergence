"""Tests for store.get_or_create_local_mailbox() — decoupled local mailbox
(Phase 7a T2.4, piece 1)."""
from brain.kindled_link.store import KindledLinkStore


def test_local_mailbox_starts_with_mbx(tmp_path) -> None:
    s = KindledLinkStore(tmp_path / "kl.db")
    mbx = s.get_or_create_local_mailbox()
    assert mbx.startswith("mbx_")


def test_local_mailbox_is_stable(tmp_path) -> None:
    """Two calls on the same store return the same id."""
    s = KindledLinkStore(tmp_path / "kl.db")
    assert s.get_or_create_local_mailbox() == s.get_or_create_local_mailbox()


def test_local_mailbox_persists_across_open_close(tmp_path) -> None:
    """Mailbox survives close + re-open of the same DB file."""
    db = tmp_path / "kl.db"
    s1 = KindledLinkStore(db)
    mbx = s1.get_or_create_local_mailbox()
    s1.close()
    s2 = KindledLinkStore(db)
    assert s2.get_or_create_local_mailbox() == mbx
    s2.close()


# --- piece 2: peers.relay_mailbox column ---


def _now():
    from datetime import UTC, datetime

    return datetime(2026, 6, 21, 12, 0, tzinfo=UTC)


def test_upsert_peer_persists_relay_mailbox(tmp_path) -> None:
    s = KindledLinkStore(tmp_path / "kl.db")
    s.upsert_peer(peer_id="kid_x", identity_pub_hex="aa", fingerprint="kid_x",
                  consent_state="pending_local", relay_url="https://r",
                  relay_mailbox="mbx_deadbeef", now=_now())
    assert s.get_peer("kid_x")["relay_mailbox"] == "mbx_deadbeef"


def test_existing_db_without_relay_mailbox_column_migrates(tmp_path) -> None:
    """A peers DB created before the column exists still opens + gains the column."""
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE peers (
            peer_id TEXT PRIMARY KEY, identity_pub TEXT NOT NULL,
            fingerprint TEXT NOT NULL, consent_state TEXT NOT NULL,
            relay_url TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        INSERT INTO peers VALUES ('kid_old','cc','kid_old','paired',NULL,'t','t');
        """
    )
    conn.commit()
    conn.close()

    s = KindledLinkStore(db)  # opening must run the migration, not raise
    assert s.get_peer("kid_old")["relay_mailbox"] is None
    s.upsert_peer(peer_id="kid_old", identity_pub_hex="cc", fingerprint="kid_old",
                  consent_state="paired", relay_url=None,
                  relay_mailbox="mbx_new", now=_now())
    assert s.get_peer("kid_old")["relay_mailbox"] == "mbx_new"
