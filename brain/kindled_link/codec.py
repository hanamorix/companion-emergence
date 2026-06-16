"""Canonical JSON — the single encoder for every signed/derived/authenticated
byte string in kindled-link (protocol doc §1). Determinism here is load-bearing:
a mismatch breaks signatures across machines."""
from __future__ import annotations

import json
from typing import Any


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
