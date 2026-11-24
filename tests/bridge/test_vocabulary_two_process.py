"""Two real processes: a real bridge (`brain.bridge.runner` through the test launcher, synthetic persona, fake provider) and a `nell`-style
command that keeps no copy of her vocabulary (#801 criterion 4; criterion 7 at a real bridge start).

Marked `integration` like the repo's other real-process bridge tests (a runner and a worker are real interpreters), so the default local gate
(`-m "not integration"`) leaves it to CI and to a deliberate local run. The bridge is the one in `tests/bridge/test_shutdown_e2e.py`
(`Bridge`: a real runner with a stub supervisor, no other background thread).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.bridge.test_shutdown_e2e import REPO, Bridge

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform == "win32", reason="the real bridge launcher uses POSIX signals and flock; Windows legs run in CI"),
]

FILE = "emotion_vocabulary.json"


class Command:
    """The second process: a long-lived worker that asks the bridge, one JSON line at a time."""

    def __init__(self, bridge: Bridge) -> None:
        env = dict(os.environ)
        env.update({"KINDLED_HOME": str(bridge.home), "PYTHONPATH": str(REPO)})
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "tests.bridge._vocabulary_client_worker", str(bridge.pd)],
            env=env,
            cwd=str(REPO),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def ask(self, request: dict) -> dict:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.wait(30)


@pytest.fixture
def bridges():
    """The bridges a test started; each is released and killed afterwards."""
    started: list[Bridge] = []
    yield started
    for b in started:
        b.release_all(8)
        b.kill()


@pytest.fixture
def command():
    started: list[Command] = []
    yield started
    for c in started:
        if c.proc.poll() is None:
            c.proc.kill()
        c.proc.wait(30)


def _entry(name: str, desc: str = "a real description") -> dict:
    return {"name": name, "description": desc, "category": "persona_extension", "decay_half_life_days": 6.0, "intensity_clamp": 10.0}


def test_a_name_added_through_the_bridge_is_seen_by_a_running_command_without_a_restart(tmp_path: Path, bridges: list, command: list) -> None:
    """Criterion 4: the command asks, the name is unknown; a name is added through the bridge; the SAME command process asks again and
    gets it under the spelling in her file, and it never holds a copy of her vocabulary."""
    b = Bridge(tmp_path)
    bridges.append(b)
    c = Command(b)
    command.append(c)

    assert c.ask({"op": "lookup", "names": ["Brand New Feeling"]}) == {"found": {"Brand New Feeling": None}, "registry_copy": []}

    added = b.post("/vocabulary/add", {"items": [{"name": "Brand New Feeling", "fields": {"description": "new"}}]})
    assert added.status_code == 200 and added.json()["results"][0]["spelling"] == "brand_new_feeling"

    after = c.ask({"op": "lookup", "names": ["Brand New Feeling", "joy"]})
    assert after["found"] == {"Brand New Feeling": "brand_new_feeling", "joy": "joy"}
    assert after["registry_copy"] == []  # no copy of her vocabulary in the command's process
    assert [e["name"] for e in json.loads((b.pd / FILE).read_text(encoding="utf-8"))["emotions"]] == ["brand_new_feeling"]


def test_a_command_adds_a_name_through_the_bridge_and_the_bridge_is_the_only_writer(tmp_path: Path, bridges: list, command: list) -> None:
    b = Bridge(tmp_path)
    bridges.append(b)
    c = Command(b)
    command.append(c)

    out = c.ask({"op": "check", "items": [["Command Word", {"description": "from a command"}], ["---", None]]})
    assert out["results"] == [["Command Word", "command_word", "used", True], ["---", None, "refused", False]]
    entries = json.loads((b.pd / FILE).read_text(encoding="utf-8"))["emotions"]
    assert [(e["name"], e["description"]) for e in entries] == [("command_word", "from a command")]
    assert c.ask({"op": "lookup", "names": ["command_word"]})["registry_copy"] == []


def test_a_real_bridge_start_merges_two_entries_of_one_name_under_the_first_spelling(tmp_path: Path, bridges: list) -> None:
    """Criterion 7 at a real bridge start: `comfort seeking` and `comfort_seeking` end as one entry under `comfort seeking`."""

    def prepare(pd: Path) -> None:
        (pd / FILE).write_text(
            json.dumps({"version": 1, "emotions": [_entry("comfort seeking", "first"), _entry("other_word"), _entry("comfort_seeking", "second")]}),
            encoding="utf-8",
        )

    b = Bridge(tmp_path, prepare=prepare)
    bridges.append(b)
    entries = json.loads((b.pd / FILE).read_text(encoding="utf-8"))["emotions"]
    assert [e["name"] for e in entries] == ["comfort seeking", "other_word"]
    assert entries[0]["description"] == "first"
