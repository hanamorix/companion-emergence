"""#746: `--shard i/N` splits the suite across CI runners: every test in exactly one shard, each xdist_group whole, the
split independent of the (random) order, and balanced by recorded Windows timings."""

from __future__ import annotations

import pytest

# The base check keeps tests/ at the PR head (HEAD_ONLY in scripts/new_tests_fail_on_base.py), and the plugin under
# test lives at tests/shard.py, so these pass there by construction. On main tests/shard.py does not exist and each of
# them fails on its import (each was seen failing that way before its code was written, #746).
_TEST_INFRA = pytest.mark.no_base_failure(
    reason="tests the plugin at tests/shard.py, which the base check keeps at the PR head; fails on import on main (#746)"
)

@_TEST_INFRA
def test_every_unit_lands_in_exactly_one_shard():
    from tests.shard import assign

    weights = {f"t{i}": float(i % 7) + 0.1 for i in range(50)}
    shards = assign(weights, 4)
    assert set(shards) == set(weights)
    assert set(shards.values()) == {1, 2, 3, 4}


@_TEST_INFRA
def test_the_shards_are_balanced_by_weight_and_a_unit_heavier_than_the_rest_is_alone():
    from tests.shard import assign

    weights = {"heavy": 100.0} | {f"t{i}": 1.0 for i in range(60)}
    shards = assign(weights, 3)
    loads = {s: sum(w for k, w in weights.items() if shards[k] == s) for s in (1, 2, 3)}
    assert [k for k in shards if shards[k] == shards["heavy"]] == ["heavy"]
    others = sorted(v for s, v in loads.items() if s != shards["heavy"])
    assert others == [30.0, 30.0]


@_TEST_INFRA
def test_a_shard_is_written_i_of_n_and_anything_else_is_refused():
    import pytest

    from tests.shard import parse_shard

    assert parse_shard("1/4") == (1, 4)
    assert parse_shard("4/4") == (4, 4)
    for bad in ("0/4", "5/4", "1/0", "-1/4", "1", "a/b", "1/4/2", "", " 1/4"):
        with pytest.raises(ValueError):
            parse_shard(bad)


def _suite() -> list[tuple[str, str | None]]:
    """(node id, xdist group) pairs: plain tests, a parametrized one, and two groups."""
    tests: list[tuple[str, str | None]] = [(f"tests/a/test_m{i}.py::test_{j}", None) for i in range(6) for j in range(9)]
    tests += [(f"tests/b/test_p.py::test_p[case{k}]", None) for k in range(12)]
    tests += [(f"tests/c/test_hf.py::test_h{k}@hf-cache", "hf-cache") for k in range(5)]
    tests += [(f"tests/c/test_net.py::TestNet::test_n{k}", "network-install") for k in range(3)]
    return tests


@_TEST_INFRA
def test_the_shards_together_select_every_test_once_and_keep_each_group_whole_whatever_the_order():
    import random

    from tests.shard import select

    tests = _suite()
    seen: dict[str, int] = {}
    for index in (1, 2, 3):
        for i in select(tests, {}, 0.3, index, 3):
            assert tests[i][0] not in seen, "a test is in two shards"
            seen[tests[i][0]] = index
    assert set(seen) == {t for t, _ in tests}
    for group in ("hf-cache", "network-install"):
        assert len({seen[t] for t, g in tests if g == group}) == 1, group

    shuffled = tests[:]
    random.Random(7).shuffle(shuffled)
    for index in (1, 2, 3):
        assert {shuffled[i][0] for i in select(shuffled, {}, 0.3, index, 3)} == {t for t, s in seen.items() if s == index}


@_TEST_INFRA
def test_run_through_pytest_the_shards_collect_every_test_once_with_the_group_whole(pytester):
    """Through the real hooks: `--shard i/N` on the command line, marker deselection and random order included."""
    pytester.makepyfile(
        test_plain="\n".join(f"def test_{i}(): pass" for i in range(20)),
        test_param="import pytest\n@pytest.mark.parametrize('n', range(10))\ndef test_p(n): pass\n",
        test_grouped="import pytest\npytestmark = pytest.mark.xdist_group('shared')\n"
        + "\n".join(f"def test_g{i}(): pass" for i in range(4)),
    )
    everything = _collected(pytester.runpytest("-p", "tests.shard", "--collect-only", "-q", "-p", "randomly"))
    assert len(everything) == 34
    seen: list[str] = []
    for index in (1, 2, 3):
        mine = _collected(pytester.runpytest("-p", "tests.shard", f"--shard={index}/3", "--collect-only", "-q",
                                              "-p", "randomly"))
        assert mine, f"shard {index} is empty"
        seen += mine
        grouped = [t for t in mine if "test_grouped.py" in t]
        assert grouped == [] or len(grouped) == 4, "the group was split"
    assert sorted(seen) == sorted(everything)


@_TEST_INFRA
def test_run_through_pytest_the_recorded_timings_balance_the_shards(pytester, monkeypatch, tmp_path):
    """A test recorded as heavier than all the others together gets a shard to itself."""
    import json

    import tests.shard

    pytester.makepyfile(test_plain="\n".join(f"def test_{i}(): pass" for i in range(12)))
    timings = tmp_path / "shard_durations.json"
    timings.write_text(json.dumps({"default_seconds": 0.1, "durations": {"test_plain.py::test_5": 30.0}}),
                       encoding="utf-8")
    monkeypatch.setattr(tests.shard, "DURATIONS_FILE", timings)
    shards = [_collected(pytester.runpytest("-p", "tests.shard", f"--shard={i}/2", "--collect-only", "-q"))
              for i in (1, 2)]
    assert ["test_plain.py::test_5"] in shards
    assert sorted(shards[0] + shards[1]) == sorted(f"test_plain.py::test_{i}" for i in range(12))


@_TEST_INFRA
def test_the_suite_itself_offers_the_shard_option(pytestconfig):
    """CI passes `--shard i/N` to the real suite, so tests/conftest.py must load the plugin. (Its value is the running
    shard in CI and None elsewhere; only an unregistered option gives back the default.)"""
    assert pytestconfig.getoption("--shard", default="missing") != "missing"


def _collected(result) -> list[str]:
    return [line.strip() for line in result.outlines if "::" in line and not line.startswith(("=", " "))]


@_TEST_INFRA
def test_a_group_weighs_the_same_whatever_order_its_tests_come_in():
    """Float sums depend on order: 0.1 + 0.2 + 0.3 is 0.6000000000000001, 0.3 + 0.2 + 0.1 is 0.6. Against a unit of exactly
    0.6 that flips the tie, so two shards with different random orders would split differently (a test run twice, another
    never)."""
    from tests.shard import select

    g = [(f"tests/t.py::test_g{k}", "b") for k in range(3)]
    durations = {"tests/t.py::test_g0": 0.1, "tests/t.py::test_g1": 0.2, "tests/t.py::test_g2": 0.3,
                 "tests/t.py::test_a": 0.6}
    one = [*g, ("tests/t.py::test_a", "a")]
    other = [*reversed(g), ("tests/t.py::test_a", "a")]
    for index in (1, 2):
        assert {one[i][0] for i in select(one, durations, 0.0, index, 2)} == {
            other[i][0] for i in select(other, durations, 0.0, index, 2)
        }
