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
