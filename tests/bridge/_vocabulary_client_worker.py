"""A second process for the two-process vocabulary test (#801 criterion 4): a `nell`-style command that keeps no copy of her vocabulary.

Usage: ``python -m tests.bridge._vocabulary_client_worker <persona_dir>``. It reads one JSON request per line on stdin and answers one JSON line
on stdout, staying alive between requests, so a test can change her vocabulary in the bridge and ask the SAME process again.

  {"op": "lookup", "names": [...]}   -> {"found": {raw: spelling or null}, "registry_copy": [raw names this process has registered]}
  {"op": "check", "items": [[name, fields_or_null], ...]} -> {"results": [[raw, spelling, outcome, changed], ...]}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from brain.emotion import vocabulary, vocabulary_access


def main(argv: list[str]) -> int:
    persona_dir = Path(argv[0])
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "lookup":
            found = vocabulary_access.lookup(persona_dir, request["names"])
            answer = {
                "found": {raw: (None if entry is None else entry.spelling) for raw, entry in found.items()},
                "registry_copy": [raw for raw in request["names"] if vocabulary.get(raw) is not None and not vocabulary.is_baseline(raw)],
            }
        else:
            results = vocabulary_access.check(persona_dir, [(name, fields) for name, fields in request["items"]])
            answer = {"results": [[r.raw, r.spelling, r.outcome, r.changed] for r in results]}
        sys.stdout.write(json.dumps(answer) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
