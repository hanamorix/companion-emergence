"""#465 C2 (B13): POST /upload writes the blob off the event loop, as a counted request worker.

The blob write now fsyncs up to the 20 MB cap and scans a directory, so running it inline on the loop would stall every
other bridge request for the duration. The test enters ONE `with TestClient(app)` block: that gives both client threads
a single shared event loop (the `_client` helper in test_upload.py returns a client that is never entered, which starts
a separate loop per request, so a blocked handler would not delay a second request and the test could not fail).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.bridge import cli_throttle, shutdown_progress
from brain.bridge.server import build_app

_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63606060600000000400015e36b8c80000000049454e44ae426082"
)
_FILE = b"a plain text file, not an image\n"


def _post(client, results, key, which):
    if which == "image":
        files = {"file": ("photo.png", _PNG, "image/png")}
    else:
        files = {"file": ("notes.txt", _FILE, "text/plain")}
    results[key] = client.post("/upload", files=files)


@pytest.mark.parametrize("which", ["image", "file"])
def test_upload_does_not_block_the_event_loop_and_the_worker_is_counted(persona_dir: Path, monkeypatch, which):
    started, release = threading.Event(), threading.Event()
    target = "brain.images.save_image_bytes" if which == "image" else "brain.file_store.save_file_bytes"
    import importlib

    module_name, attr = target.rsplit(".", 1)
    real = getattr(importlib.import_module(module_name), attr)

    def slow(*args, **kwargs):
        started.set()
        assert release.wait(30), "the test never released the blocked save"
        return real(*args, **kwargs)

    monkeypatch.setattr(target, slow)
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=None)
    results: dict[str, object] = {}
    with TestClient(app) as client:
        uploader = threading.Thread(target=_post, args=(client, results, "up", which))
        uploader.start()
        try:
            assert started.wait(10), "the upload never reached the save"
            prober = threading.Thread(target=lambda: results.__setitem__("health", client.get("/health")))
            prober.start()
            prober.join(timeout=5)
            # (a) the second request completes while the first is still blocked in the save
            assert not prober.is_alive(), "the event loop is blocked by the upload's save"
            assert results["health"].status_code == 200
            # (b) the worker is a counted request worker while it is blocked
            assert shutdown_progress.registry.count() == 1
            assert shutdown_progress.registry.snapshot() == ["upload"]
            assert shutdown_progress.registry.seal() is False
        finally:
            release.set()
            uploader.join(timeout=30)
        # (c) released: the upload completes with its record and the worker is uncounted
        resp = results["up"]
        assert resp.status_code == 200, resp.text
        assert resp.json()["kind"] == which
        assert shutdown_progress.registry.count() == 0


def test_upload_after_the_registry_is_sealed_is_refused_and_writes_nothing(persona_dir: Path):
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=None)
    with TestClient(app) as client:
        assert shutdown_progress.registry.seal() is True
        with pytest.raises(cli_throttle.ShutdownDeferred):
            client.post("/upload", files={"file": ("photo.png", _PNG, "image/png")})
    assert not (persona_dir / "images").exists() or list((persona_dir / "images").iterdir()) == []
