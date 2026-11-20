"""#746: scripts/refresh_shard_durations.py rebuilds tests/shard_durations.json from the Windows junit artifacts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "refresh_shard_durations.py"


def _load():
    spec = importlib.util.spec_from_file_location("refresh_shard_durations", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_junit_case_maps_to_the_node_id_the_shard_plugin_sees(tmp_path):
    script = _load()
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "test_m.py").write_text("", encoding="utf-8")
    cases = {
        ("tests.unit.test_m", "test_a"): "tests/unit/test_m.py::test_a",
        ("tests.unit.test_m", "test_p[x.y-1]@hf-cache"): "tests/unit/test_m.py::test_p[x.y-1]",
        ("tests.unit.test_m.TestK", "test_b"): "tests/unit/test_m.py::TestK::test_b",
        ("tests.unit.test_m", "test_q[a@b]"): "tests/unit/test_m.py::test_q[a@b]",
        ("", "tests.unit.test_m"): None,  # a module-level entry (collection error or skip), not a test
        ("tests.unit.gone", "test_c"): None,  # no such module any more
    }
    for (classname, name), want in cases.items():
        assert script.nodeid_of(classname, name, tmp_path) == want, (classname, name)


def test_only_tests_over_the_threshold_are_stored_and_the_rest_weigh_their_mean(tmp_path):
    script = _load()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_m.py").write_text("", encoding="utf-8")
    shard_a, shard_b = tmp_path / "a.xml", tmp_path / "b.xml"
    shard_a.write_text(
        '<testsuites><testsuite><testcase classname="tests.test_m" name="test_slow@hf-cache" time="2.5"/>'
        '<testcase classname="tests.test_m" name="test_fast" time="0.1"/>'
        '<testcase classname="" name="tests.test_gone" time="9.0"/></testsuite></testsuites>',
        encoding="utf-8",
    )
    shard_b.write_text(
        '<testsuite><testcase classname="tests.test_m" name="test_quick" time="0.3"/>'
        '<testcase classname="tests.test_m" name="test_edge" time="0.5"/></testsuite>',
        encoding="utf-8",
    )
    built = script.build([shard_a, shard_b], tmp_path, threshold=0.5)
    assert built["durations"] == {"tests/test_m.py::test_edge": 0.5, "tests/test_m.py::test_slow": 2.5}
    assert built["default_seconds"] == 0.2  # the mean of test_fast and test_quick


def test_the_command_writes_the_file_the_shard_plugin_reads(tmp_path):
    import json

    from tests.shard import load_durations

    script = _load()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_m.py").write_text("", encoding="utf-8")
    report = tmp_path / "r.xml"
    report.write_text('<testsuite><testcase classname="tests.test_m" name="test_slow" time="3.25"/>'
                      '<testcase classname="tests.test_m" name="test_fast" time="0.05"/></testsuite>', encoding="utf-8")
    out = tmp_path / "shard_durations.json"
    assert script.main([str(report), "--out", str(out), "--root", str(tmp_path)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["durations"] == {"tests/test_m.py::test_slow": 3.25}
    assert load_durations(out) == ({"tests/test_m.py::test_slow": 3.25}, 0.05)
