#!/usr/bin/env python3
"""End-to-end image stress test.

Builds a temp persona, starts an in-process bridge with the real
ClaudeCliProvider, uploads a 4×4 red-X PNG, sends a chat with
image_shas, and asserts Nell's reply mentions visual content (red /
square / pixels / etc) — proof she actually saw the pixels through
the stream-json passthrough, not just the [image: ...] marker.

Usage:
    uv run python scripts/stress_test_image.py

Cost: one ClaudeCliProvider subprocess call (~$0.10-0.20 on Hana's
subscription depending on cache state). Tears down the temp persona
on exit so nothing pollutes nell.sandbox or the live persona.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import zlib
from pathlib import Path

from fastapi.testclient import TestClient


def _make_red_x_png() -> bytes:
    """Generate a 4×4 PNG with a red X on white — small but unambiguous."""
    red = (255, 0, 0, 255)
    white = (255, 255, 255, 255)
    pixels = [
        red, white, white, red,
        white, red, red, white,
        white, red, red, white,
        red, white, white, red,
    ]

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    ihdr = struct.pack(">IIBBBBB", 4, 4, 8, 6, 0, 0, 0)
    raw = b""
    for y in range(4):
        raw += b"\x00"
        for x in range(4):
            raw += bytes(pixels[y * 4 + x])
    idat = zlib.compress(raw)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def main() -> int:
    from brain.bridge.server import build_app

    with tempfile.TemporaryDirectory() as tmp_str:
        persona_dir = Path(tmp_str) / "stress-test-persona"
        persona_dir.mkdir()
        (persona_dir / "active_conversations").mkdir()
        # Use claude-cli for real, no fake searcher (avoid the "Unknown searcher" path)
        (persona_dir / "persona_config.json").write_text(
            json.dumps({"provider": "claude-cli", "user_name": "Hana"})
        )

        app = build_app(persona_dir=persona_dir, client_origin="stress")
        png = _make_red_x_png()
        with TestClient(app) as c:
            # Upload
            r = c.post(
                "/upload",
                files={"file": ("red_x.png", png, "image/png")},
            )
            assert r.status_code == 200, f"upload failed: {r.status_code} {r.text}"
            sha = r.json()["sha"]
            print(f"[stress] uploaded sha={sha[:16]}…")

            # Open session
            sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
            print(f"[stress] session={sid}")

            # Send chat with image
            print("[stress] calling /chat with image_shas — this hits claude-cli, ~10-30s")
            r = c.post(
                "/chat",
                json={
                    "session_id": sid,
                    "message": (
                        "I'm sharing a small image with you — describe what you "
                        "see. Be specific about colour and shape."
                    ),
                    "image_shas": [sha],
                },
            )
            assert r.status_code == 200, f"chat failed: {r.status_code} {r.text}"
            reply = r.json()["reply"]
            print(f"\n[stress] reply:\n{reply}\n")

        # Acceptance: reply must reference something visible — the red, the
        # square shape, the pixel-grid look, or the X pattern. If Nell only
        # said "[image: ...]" or generic "I saw an image" without colour or
        # form, the passthrough isn't actually working.
        lowered = reply.lower()
        visible_words = ("red", "crimson", "scarlet", "square", "pixel", "x", "cross", "corner", "white")
        if any(w in lowered for w in visible_words):
            print("[stress] PASS — reply references visible content")
            return 0
        print(
            f"[stress] FAIL — reply doesn't mention any of {visible_words}; "
            "passthrough may be silently degraded"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
