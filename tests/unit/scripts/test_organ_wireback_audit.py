"""The wire-back audit's write-only jsonl detector must not silently drop files.

The detector existed to catch the brain's characteristic failure — a writer whose
reader is dead. Measured 2026-08-08: it reported 5 write-only jsonl files while
32 basenames appear under brain/. The other 27 were invisible, including a 46 MB
`gate_rejections.jsonl` with 221,906 lines and no reader anywhere.

Cause: it required the filename literal and the write operation on the SAME LINE.
The common — and better — pattern puts them apart:

    path = persona_dir / "gate_rejections.jsonl"   # name here
    with path.open("a", encoding="utf-8") as f:     # write here

So a file with no same-line hint scored neither write nor read and was excluded
from the report entirely: it could never be flagged, in either direction. The
heuristic systematically missed the tidier code.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_AUDIT = Path(__file__).resolve().parents[3] / "scripts" / "organ_wireback_audit.py"


def _load():
    spec = importlib.util.spec_from_file_location("organ_wireback_audit", _AUDIT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["organ_wireback_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_write_via_a_path_variable_is_detected(tmp_path: Path):
    """The real-world shape: name on one line, `.open("a")` on another."""
    mod = _load()
    (tmp_path / "writer.py").write_text(
        'def append_row(persona_dir, row):\n'
        '    path = persona_dir / "gate_rejections.jsonl"\n'
        '    with path.open("a", encoding="utf-8") as f:\n'
        '        f.write("x")\n',
        encoding="utf-8",
    )
    assert "gate_rejections.jsonl" in mod.find_write_only_jsonl(root=tmp_path)


def test_an_unrelated_read_elsewhere_in_the_file_does_not_mask_a_write_only_log(
    tmp_path: Path,
):
    """Regression on the scope boundary — I got this wrong once while fixing it.

    `ast.walk(module)` descends into every function, so treating the module as a
    scope makes a file's jsonl names inherit every operation anywhere in it. That
    marked a genuinely write-only log as read and hid it again. Scopes are
    function bodies; the module is not one.
    """
    mod = _load()
    (tmp_path / "mixed.py").write_text(
        'def append_row(persona_dir, row):\n'
        '    path = persona_dir / "gate_rejections.jsonl"\n'
        '    with path.open("a", encoding="utf-8") as f:\n'
        '        f.write("x")\n'
        '\n'
        'def load_something_else(persona_dir):\n'
        '    return (persona_dir / "other.json").read_text()\n',
        encoding="utf-8",
    )
    assert "gate_rejections.jsonl" in mod.find_write_only_jsonl(root=tmp_path)
