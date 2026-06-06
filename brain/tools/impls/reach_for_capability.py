"""reach_for_capability — agency safety-valve. Nell calls this to recruit a
faculty (memory/files/works) that wasn't pre-attached this turn; the engine
re-runs the turn once with the heavier tools allowed (see chat/tool_loop, Task 2.3).
The impl only records the intent — the recruitment is the engine's re-invoke."""
from __future__ import annotations

from pathlib import Path


def reach_for_capability(capability: str = "", *, persona_dir: Path, **_) -> dict:
    return {"ok": True, "recruited": capability}
