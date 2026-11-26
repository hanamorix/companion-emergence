"""#483: two simultaneous first starts must end with ONE identity: the one on disk.

`KindledIdentity.load_or_create` let both callers see no key file, each generate its own keypair and each `os.replace` it onto the
key path; the later replace won, and the caller whose replace landed first returned an identity (its key_id, its signatures) that
did not match the file the next start loads. The check-and-create now runs under `file_lock(key_path)`, and the loser reads the
winner's key. `rotate` overwrites on purpose and takes the same lock for the replace.
"""

from __future__ import annotations

import threading

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from brain.kindled_link.identity import _KEY_DIRNAME, _KEY_FILENAME, KindledIdentity
from tests.thread_support import finish_threads


def _race(persona_dir, monkeypatch, n=2):
    """Start `n` callers together; hold each one just after it has generated its keypair so they overlap."""
    real_generate = Ed25519PrivateKey.generate
    barrier = threading.Barrier(n)

    def generate():
        key = real_generate()
        try:
            barrier.wait(timeout=1.0)  # every caller that reaches here has generated before any of them writes
        except threading.BrokenBarrierError:
            pass  # a caller that is serialised behind a lock never arrives: that is the fix working
        return key

    monkeypatch.setattr(Ed25519PrivateKey, "generate", staticmethod(generate))
    results: list[KindledIdentity] = [None] * n  # type: ignore[list-item]

    def run(i):
        results[i] = KindledIdentity.load_or_create(persona_dir)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        finish_threads(t)
    monkeypatch.setattr(Ed25519PrivateKey, "generate", staticmethod(real_generate))
    return results


def test_simultaneous_first_starts_all_return_the_key_that_is_on_disk(tmp_path, monkeypatch):
    results = _race(tmp_path, monkeypatch, n=3)

    on_disk = KindledIdentity.load_or_create(tmp_path)
    assert all(r is not None for r in results)
    assert {r.key_id for r in results} == {on_disk.key_id}


def test_the_returned_identity_signs_for_the_key_a_later_start_loads(tmp_path, monkeypatch):
    first, second = _race(tmp_path, monkeypatch, n=2)

    later = KindledIdentity.load_or_create(tmp_path)
    for r in (first, second):
        assert r.public_bytes == later.public_bytes
        assert later.sign(b"hello") == r.sign(b"hello")  # Ed25519 signing is deterministic: same key, same signature


def test_a_second_start_loads_the_existing_key_unchanged(tmp_path):
    # PIN
    first = KindledIdentity.load_or_create(tmp_path)
    key_path = tmp_path / _KEY_DIRNAME / _KEY_FILENAME
    before = key_path.read_bytes()

    second = KindledIdentity.load_or_create(tmp_path)

    assert second.key_id == first.key_id and key_path.read_bytes() == before


def test_rotate_still_replaces_the_key_and_returns_new_and_old(tmp_path):
    # PIN
    old_loaded = KindledIdentity.load_or_create(tmp_path)

    new, old = KindledIdentity.rotate(tmp_path)

    assert old.key_id == old_loaded.key_id and new.key_id != old.key_id
    assert KindledIdentity.load_or_create(tmp_path).key_id == new.key_id
