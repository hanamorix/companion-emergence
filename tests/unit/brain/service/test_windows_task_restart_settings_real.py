"""The Task Scheduler restart settings against the REAL scheduler (D26 j; Windows only, runs on the windows-latest CI leg).

The unit tests prove what our XML says (`RestartOnFailure` Interval PT1M, Count 255). Only the scheduler can say that it
accepts them: MS-TSCH documents Count as 1 to 255 and Interval as 1 minute to 31 days, but the Task Scheduler UI allows
999, so a count the scheduler refuses would fail the whole install. This registers a throwaway task from our generated
XML, reads it back and deletes it.
"""

from __future__ import annotations

import re
import sys
import uuid

import pytest

from brain.service import windows_service

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="needs the real Windows Task Scheduler")


def test_the_scheduler_accepts_our_restart_settings_and_reports_them_back(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    persona = f"ci-d26-{uuid.uuid4().hex[:8]}"
    name = windows_service.task_name(persona)
    xml_path = windows_service.write_task_xml(persona=persona, nell_path=sys.executable)
    created = windows_service.run_schtasks(["/Create", "/XML", str(xml_path), "/TN", name, "/F"])
    try:
        assert created.returncode == 0, f"the scheduler refused our task XML: {created.stdout} {created.stderr}"
        queried = windows_service.run_schtasks(["/Query", "/TN", name, "/XML"])
        assert queried.returncode == 0, queried.stderr
        # schtasks' console output may carry NULs (UTF-16) or replacement characters; only the ASCII tags matter here.
        text = queried.stdout.replace("\x00", "").replace("�", "")
        assert re.search(rf"<Interval>{windows_service.RESTART_INTERVAL}</Interval>", text), text
        assert re.search(rf"<Count>{windows_service.RESTART_COUNT}</Count>", text), text
    finally:
        windows_service.run_schtasks(["/Delete", "/TN", name, "/F"])
