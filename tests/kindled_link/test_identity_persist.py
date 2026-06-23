import os
import stat

import pytest

from brain.kindled_link.identity import KindledIdentity

_KEY_REL = "kindled_link/identity_ed25519.key"


def test_create_then_reload_same_key(tmp_path) -> None:
    a = KindledIdentity.load_or_create(tmp_path)
    assert (tmp_path / _KEY_REL).exists()
    b = KindledIdentity.load_or_create(tmp_path)  # second call loads, not regenerates
    assert a.key_id == b.key_id
    assert a.public_bytes == b.public_bytes


def test_key_id_and_pubkey_shape(tmp_path) -> None:
    idn = KindledIdentity.load_or_create(tmp_path)
    assert idn.key_id.startswith("kid_") and len(idn.key_id) == 20
    assert len(idn.public_bytes) == 32


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_key_file_is_0600(tmp_path) -> None:
    KindledIdentity.load_or_create(tmp_path)
    mode = stat.S_IMODE((tmp_path / _KEY_REL).stat().st_mode)
    assert mode == 0o600


def test_key_write_is_atomic_no_truncated_file_on_failure(tmp_path, monkeypatch) -> None:
    # A mid-write failure (e.g. full disc) must NOT leave a truncated key at the
    # final path that crashes the next load with from_private_bytes(b'') (#44).
    # Write-then-rename: a failed write orphans only the temp, leaving the final
    # path absent so the next load regenerates cleanly.
    import brain.kindled_link.identity as idmod

    real_write = os.write

    def _boom(fd, data):
        raise OSError("disk full")

    monkeypatch.setattr(idmod.os, "write", _boom)
    with pytest.raises(OSError):
        KindledIdentity.load_or_create(tmp_path)
    assert not (tmp_path / _KEY_REL).exists(), "no truncated key file may remain"

    monkeypatch.setattr(idmod.os, "write", real_write)
    idn = KindledIdentity.load_or_create(tmp_path)  # regenerates cleanly
    assert idn.key_id.startswith("kid_")
