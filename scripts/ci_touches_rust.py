"""Does a PR change anything the Rust build or its tests read? (#758)

test.yml's `rust-changes` job pipes the PR's changed paths in (one per line) and records `rust=true|false`; the macOS
`tauri (cargo check + test)` job runs only on `true`, because it waits on GitHub's short macOS arm64 runner pool.
Linux and Windows compile and test the workspace on every PR regardless, and every push to main runs the macOS job.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

# A trailing "/" is a folder. The two outside app/src-tauri are read by Rust tests (include_str! of
# app/src/brainRecovery.ts in brain_update.rs; the bridge_stop_tests fixtures in lib.rs) —
# tests/unit/scripts/test_ci_touches_rust.py fails if another include_str!/include_bytes! reaches outside unlisted.
RUST_INPUTS = (
    "app/src-tauri/",
    "app/src/brainRecovery.ts",
    "tests/fixtures/bridge_watcher_files/",
    ".github/workflows/test.yml",
)


def touches_rust(paths: Iterable[str]) -> bool:
    return any(path == entry or (entry.endswith("/") and path.startswith(entry)) for path in paths for entry in RUST_INPUTS)


def main() -> int:
    print(f"rust={'true' if touches_rust(line.strip() for line in sys.stdin if line.strip()) else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
