"""Pass-2 daemon threads should have unique names so logs disambiguate."""
from __future__ import annotations

import threading
import time
from pathlib import Path


def test_concurrent_pass2_threads_have_distinct_names(tmp_path: Path):
    from brain.chat.tool_loop import _spawn_pass2

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    names_seen: set[str] = set()

    class _FakeProvider:
        def generate(self, prompt, *, system=None):
            # Record the thread's name AT THE MOMENT the daemon runs.
            names_seen.add(threading.current_thread().name)
            time.sleep(0.1)  # hold the thread alive long enough to overlap
            return "{}"

    provider = _FakeProvider()
    for i in range(3):
        _spawn_pass2(
            provider=provider,
            monologue_text=f"monologue {i}",
            visible_reply="reply",
            recent_user_msgs=(),
            persona_dir=persona_dir,
        )

    # Wait for all daemon threads to register their names.
    deadline = time.time() + 5.0
    while time.time() < deadline and len(names_seen) < 3:
        time.sleep(0.05)

    assert len(names_seen) == 3, f"expected 3 distinct thread names, got {names_seen}"
    # All names should share the "monologue-extractor" prefix.
    assert all("monologue-extractor" in n for n in names_seen)
