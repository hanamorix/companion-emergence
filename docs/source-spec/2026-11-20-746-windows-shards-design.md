# Shard the Windows python job across runners (#746, part 1)

The `python windows-latest` job is the longest check on every PR. On recent green runs on main its pytest step took 32 to 50 minutes, against 4 to 13 on macOS and 6 to 9 on Ubuntu. This change splits it across runners. Part 2 of #746 (a guard against redundant tests) is a separate change.

## Measurements (main, 2026-11-20)

- The Windows tests add up to 80 minutes (14,784 tests, a mean of 324 ms each), against 24 on macOS (99 ms each).
- The time is spread out: the 100 slowest tests take 14 of the 80 minutes, and the longest single test is 134 s (`hf-cache` group).
- `-n auto` counts **physical** cores: Windows gets 2 workers on its 4 vCPUs, Ubuntu 2, macOS 3. `-n logical` would give Windows 4.
- Collection and setup cost about 1.5 minutes per runner.
- Main has no branch protection and no rulesets, so no required check name constrains the job names.

## Goals

- The Windows python check is no longer the longest python check on a normal PR, which is macOS's today at 10 to 12 minutes.
- The same tests run: every test in exactly one shard, and each `xdist_group` kept whole in one shard.
- Random order and the slow-test budget behave as now: each shard has its own fresh seed, and the budget stays on Ubuntu (unsharded).
- One check, `python windows-latest`, still says whether Windows passed.

## Non-goals

- Part 2 of #746 (the test-growth guard).
- Sharding Ubuntu or macOS, or changing manual (`workflow_dispatch`) runs, which stay unsharded for faithful reproduction.

## Approaches (picked: A)

| | What | Trade-off |
|---|---|---|
| **A, picked** | Our own `--shard i/N` plugin: assign by node id, keep groups whole, balance with recorded Windows timings | About 100 lines we maintain; the split doesn't depend on the random order, so each shard keeps a fresh seed |
| B | `pytest-split` | It splits by collection order, so every shard needs one shared seed; it needs a durations file for all 15,000 tests and doesn't know our groups |
| C | `-n logical` or a larger runner alone | `-n logical` likely gives 25 to 30 minutes, still the longest; larger runners are paid, standard ones are free here |

Hana picked A on 2026-11-20.

## Design

**`tests/shard.py`** (loaded from `tests/conftest.py` through `pytest_plugins`):

- **The option.** `--shard i/N` (1 ≤ i ≤ N); a malformed value is a usage error. Without it, nothing changes.
- **Units.** Each test is a unit keyed by its node id, with xdist's `@<group>` suffix stripped using xdist's own rule. Tests sharing an `xdist_group` name form one unit.
- **Weights.** A unit weighs the sum of its tests' recorded seconds from `tests/shard_durations.json`. A test with no entry weighs the file's `default_seconds`.
- **Assignment.** Units are sorted by weight, heaviest first, with ties broken by key. Each goes to the least-loaded shard, ties to the lowest index. The result depends only on the collected set and the file, never on the order, so every shard and every xdist worker computes the same split.
- **Selection.** In `pytest_collection_modifyitems` (`trylast`, so after marker deselection and the random shuffle), the plugin keeps this shard's items in their current order and reports the rest through `pytest_deselected`.

**`tests/shard_durations.json`:**

- **Contents.** `{"default_seconds": d, "durations": {nodeid: seconds}}`. Only tests of at least 0.5 s on Windows have an entry; `d` is the mean of the rest.
- **Refreshing.** `scripts/refresh_shard_durations.py <junit.xml>...` rebuilds it from the Windows junit artifacts. It maps a testcase's classname and name to a node id by finding the module file, and strips the group suffix.
- **When it goes stale.** Only the balance suffers; correctness doesn't.

**`.github/workflows/test.yml`:**

- **`python`** runs Ubuntu and macOS on push and pull request. A manual run is unchanged, Windows unsharded included.
- **`python-windows`** runs on push and pull request.
  - It has a matrix of N shards, named `python windows-latest (i/N)`.
  - Each shard runs the same pytest command plus `--shard i/N`.
  - Artifacts are named per shard: `pytest-junit-windows-latest-i-of-N` and `slow-budget-cpu-windows-latest-i-of-N`.
  - Ruff runs once, in `python`.
- **`python windows-latest`** needs `python-windows` and runs `if: always()`. It passes only when every shard passed.

N and the worker count (`-n auto` or `-n logical`) are chosen from a 4-worker measurement run (#746 work log), then checked on this PR's own CI.

## Wiring

No brain organ. It reads the junit timings CI already uploads, and feeds the Windows check every PR waits on.

## Tests (written first, one at a time)

- **Assignment:**
  - every test lands in exactly one shard;
  - a group's tests stay in one shard;
  - shuffling the input gives the same split;
  - weights balance the shards (the heaviest unit is alone when it outweighs the rest);
  - a malformed `--shard` is a usage error.
- **Through pytest (pytester):** the shards of a small sample suite, run separately, together collect every test exactly once, with a group kept whole.
- **The refresh script:** junit cases map to node ids (module, class, parameters, group suffix); only cases over the threshold are stored; the default is the mean of the rest.
- **The workflow:**
  - the shard matrix covers 1..N with `--shard`;
  - the aggregate job is named `python windows-latest`, needs the shards and fails on a failed or cancelled shard;
  - `python` no longer runs Windows on push and pull request;
  - artifact names are unique.
