"""One-shot backfill: extract OG crystallizations + write to an existing persona's
crystallizations.db. Used after SP-5 added SoulStore — pre-existing personas
were migrated before SoulStore existed and have empty soul DBs.

Usage:
    python scripts/backfill_soul_for_persona.py \\
        --og /Users/hanamori/NellBrain/data \\
        --persona-name nell.sandbox

Idempotent: if a crystallization with the same id is already in the soul DB,
skips it (count goes in `already_present` bucket).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from brain.migrator.og_soul import extract_crystallizations_from_og
from brain.paths import get_persona_dir
from brain.soul.store import SoulStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill OG soul crystallizations into an existing persona's soul DB."
    )
    parser.add_argument(
        "--og",
        type=Path,
        required=True,
        help="Path to OG NellBrain data/ directory containing nell_soul.json",
    )
    parser.add_argument(
        "--persona-name",
        type=str,
        required=True,
        help="Persona name (e.g. nell.sandbox)",
    )
    args = parser.parse_args()

    persona_dir = get_persona_dir(args.persona_name)
    if not persona_dir.exists():
        print(f"persona dir not found: {persona_dir}", file=sys.stderr)
        return 1

    og_soul_path = args.og / "nell_soul.json"
    if not og_soul_path.exists():
        print(f"nell_soul.json not found at: {og_soul_path}", file=sys.stderr)
        return 1

    crystals, skipped = extract_crystallizations_from_og(args.og)
    print(f"extracted: {len(crystals)} active, {len(skipped)} skipped from OG")

    soul_db_path = persona_dir / "crystallizations.db"
    soul_store = SoulStore(db_path=soul_db_path)

    created = 0
    already_present = 0
    try:
        for crystal in crystals:
            if soul_store.get(crystal.id) is not None:
                already_present += 1
                continue
            soul_store.create(crystal)
            created += 1
    finally:
        soul_store.close()

    print(f"created: {created}")
    print(f"already_present (idempotent skip): {already_present}")
    print(f"skipped at extract: {len(skipped)}")
    if skipped:
        for s in skipped[:5]:
            print(f"  - {s}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
