#!/usr/bin/env python3
"""Augment the synthetic memories.db: add missing feelings (#654), give feelings real ranges (#664).

build_db.py cannot be re-run (its writer "parts" files are not committed), so this script
starts from the committed base database and writes a new one:

  python augment_db.py --input BASE_memories.db --output NEW_memories.db [--seed N] [--force]
                       [--manifest-in manifest.json] [--allow-other-base]

The base is the database as committed in git commit d8eedb76 (sha256 below):
  git show d8eedb76:test-data/synthetic-db/memories.db > /path/outside/the/repo/base.db

Two edits, both deterministic for a given seed (stdlib only, no network, nothing is read except
--input and --manifest-in):

1. #654: invented rows are added for five feelings that the body-state and reflex-arc readers
   use and the base has none of: rest_need, arousal, climax, touch_hunger and creative_hunger.
   Each row carries exactly one emotion key and the tags and metadata shape of a
   conversation-derived memory, because the emotion engine counts only such rows (and a few
   other kinds); rows without the conversation tag would be invisible to it. Each feeling is
   spread over the 109-day window so its stored value has a real range in every 30-day window
   (NEW_FEELINGS holds the plan; check_db.py checks it).
2. #664: the stored values of seven existing feelings are rescaled (row text and every other
   column untouched). NEAR_ZERO feelings are mapped monotonically into 0.02 to 0.30; SPIKY
   feelings keep their strongest conversation-tagged rows (about 6 percent of their rows) and
   map every other row into 0.10 to 1.00. All other feelings, including the spelling-variant
   keys comfort-seeking and memory-grief that tests rely on, are untouched. The README says
   why each feeling was chosen.

Before anything is published the script checks its own result (verify): every base row is
intact apart from the profile feelings' values, the added rows match the plan, and the window
table holds. A seed that fails the window table is rejected. It never opens the input for
writing, never overwrites an existing output (unless --force, and then only a file that is the
recorded base or already augmented), and publishes the database, then the manifest, with
os.replace.

Pure stdlib (sqlite3, json, random.Random, hashlib, argparse, tempfile, shutil, math, datetime).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

from augment_pools import POOLS
from build_db import SPAN_DAYS, START_DATE, TOPICAL_POOL, make_created_at

# the date this was written; any seed that passes verify() is as good
DEFAULT_SEED = 20261110
BASE_SHA256 = "f0a4def9fb77bd009e75ce2571914a4c964204cf246a799fba8ae6542f4c69da"
BASE_COMMIT = "d8eedb76"
# marks added rows in metadata_json; also how a second augmentation is refused
PROVIDER = "synthetic-aug-v1"

# the engine's snapshot length; "a real range" is measured over every run of this many days
WINDOW_DAYS = 30
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
# share of a feeling's rows (the highest wave values) that take a "strong" text line
STRONG_SHARE = 0.4
# last_accessed_at = created_at + exponential delay: the builder's rule (build_db.py build())
ACCESS_MEAN_HOURS = 20.0
ACCESS_CAP_HOURS = 500.0
# same floor as the base's values; 9.5 keeps clear of the 10.0 clamp
VALUE_MIN = 0.5
VALUE_MAX = 9.5
# uniform noise either side of the wave, so rows do not sit on a clean curve
NOISE = 0.5

MEMORY_TYPES = ("monologue_emotion", "observation", "note", "monologue", "feeling")
# domain per type, as the base database has it (check_db.py check 12)
TYPE_DOMAIN = {
    "monologue_emotion": "monologue",
    "monologue": "monologue",
    "observation": "brain",
    "note": "brain",
    "feeling": "brain",
}


class FeelingPlan(NamedTuple):
    """One added feeling: rows by memory type (order of MEMORY_TYPES), check demands, wave."""

    types: tuple[int, int, int, int, int]
    min_days: int  # distinct days with a row
    min_window_rows: int  # rows in every WINDOW_DAYS run
    min_window_range: float  # max - min stored value in every WINDOW_DAYS run
    period: float  # wave period in days; at most 26 so a 30-day window holds a full cycle
    low: float  # wave floor; high - low is at least 4.0 so the range target of 3.0 has slack
    high: float
    spike_share: float  # share of rows lifted to the old fixed level its reader used
    # rest_need/arousal/touch_hunger 7, climax 5, creative_hunger 8 (the old levels)
    spike_band: tuple[float, float]
    topic: str  # third tag, from build_db.TOPICAL_POOL

    @property
    def rows(self) -> int:
        return sum(self.types)


NEW_FEELINGS: dict[str, FeelingPlan] = {
    "rest_need": FeelingPlan(
        (18, 11, 7, 7, 5), 40, 6, 3.0, 21.0, 0.8, 5.2, 0.15, (7.0, 9.0), "rest"
    ),
    "arousal": FeelingPlan(
        (16, 9, 5, 6, 4), 34, 6, 3.0, 17.0, 0.7, 4.8, 0.15, (7.0, 9.0), "intimacy"
    ),
    "climax": FeelingPlan(
        (12, 7, 4, 4, 3), 26, 5, 3.0, 14.0, 0.6, 4.6, 0.20, (5.0, 9.0), "intimacy"
    ),
    "touch_hunger": FeelingPlan(
        (18, 11, 6, 6, 4), 38, 6, 3.0, 26.0, 0.8, 5.4, 0.15, (7.0, 9.0), "closeness"
    ),
    "creative_hunger": FeelingPlan(
        (16, 12, 8, 7, 2), 38, 6, 3.0, 24.0, 0.8, 5.4, 0.15, (8.0, 9.5), "craft"
    ),
}

# Rescale profiles (#664). Choice and reasons: README "Feeling profiles".
NEAR_ZERO = ("vulnerability", "amusement", "worry")
SPIKY = ("loneliness", "defiance", "curiosity", "joy")
ORIGINAL_RANGE = (0.5, 8.0)  # the base's stored-value range
NEAR_ZERO_RANGE = (0.02, 0.30)  # the brief's "about 0.02-0.3"
SPIKY_LOW_RANGE = (0.10, 1.00)
SPIKE_SHARE = (
    0.06  # of a spiky feeling's rows, the strongest conversation-tagged ones keep their value
)
SPIKE_MIN = 3
# spelling-variant keys tests rely on: never rescaled; check 19 holds their base counts
VARIANT_COUNTS = {
    "comfort_seeking": 36,
    "comfort-seeking": 29,
    "memory_grief": 141,
    "memory-grief": 12,
}
ABSENT_EMOTIONS = (
    "peace",
    "body_grief",
)  # #648 (unregistered) and #652 (being removed): never added

INSERT_SQL = """
INSERT INTO memories (
    id, content, memory_type, domain, emotions_json, tags_json, importance, score, created_at,
    last_accessed_at, active, protected, metadata_json, state, content_snapshot, recall_count,
    peak_emotion_intensity
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class AugmentError(Exception):
    """A named failure: one message, exit 1, nothing published."""


# ---------------------------------------------------------------------------
# shared with check_db.py (one copy of each)
# ---------------------------------------------------------------------------


def day_of(created_at: str) -> int:
    """Day number 1..109 of an ISO created_at string, counted from the database's first day."""
    y, m, d = (int(x) for x in created_at[:10].split("-"))
    return (date(y, m, d) - START_DATE.date()).days + 1


def window_of(day: int) -> int:
    """The 16 life windows, as check_db.py's coverage check maps them."""
    return 16 if day >= 106 else (day - 1) // 7 + 1


def window_problems(name: str, pairs: list[tuple[int, float]], plan: FeelingPlan) -> list[str]:
    """Messages for a feeling whose (day, value) rows miss its plan; empty when it holds.

    Distinct days, then for every run of WINDOW_DAYS days starting on day 1 to day 80: enough
    rows and a wide enough range.
    """
    problems = []
    days = {d for d, _ in pairs}
    if len(days) < plan.min_days:
        problems.append("%s: %d distinct days, plan %d" % (name, len(days), plan.min_days))
    row_gap = range_gap = None
    for start in range(1, SPAN_DAYS - WINDOW_DAYS + 2):
        values = [v for d, v in pairs if start <= d < start + WINDOW_DAYS]
        if len(values) < plan.min_window_rows:
            row_gap = row_gap or "%s: days %d-%d hold %d rows, plan %d" % (
                name,
                start,
                start + WINDOW_DAYS - 1,
                len(values),
                plan.min_window_rows,
            )
        elif max(values) - min(values) < plan.min_window_range:
            range_gap = range_gap or "%s: days %d-%d range %.1f, plan %.1f" % (
                name,
                start,
                start + WINDOW_DAYS - 1,
                max(values) - min(values),
                plan.min_window_range,
            )
    return problems + [p for p in (row_gap, range_gap) if p]


def db_summary(conn: sqlite3.Connection) -> dict:
    """The counts manifest.json holds, worked out from the database (check 20 shares it)."""
    types = dict(conn.execute("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"))
    windows: dict[int, int] = {}
    day_counts: dict[int, int] = {}
    for (created_at,) in conn.execute("SELECT created_at FROM memories"):
        day = day_of(created_at)
        day_counts[day] = day_counts.get(day, 0) + 1
        windows[window_of(day)] = windows.get(window_of(day), 0) + 1
    values = list(day_counts.values())
    return {
        "total_rows": sum(types.values()),
        "per_memory_type_counts": types,
        "per_window_counts": {str(k): windows[k] for k in sorted(windows)},
        "per_day_stats": {
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        },
        "active_count": conn.execute("SELECT COUNT(*) FROM memories WHERE active = 1").fetchone()[
            0
        ],
        "fading_count": conn.execute(
            "SELECT COUNT(*) FROM memories WHERE state = 'fading'"
        ).fetchone()[0],
        "inactive_count": conn.execute(
            "SELECT COUNT(*) FROM memories WHERE state = 'inactive'"
        ).fetchone()[0],
        "day_span": {"min": min(day_counts), "max": max(day_counts)},
    }


# ---------------------------------------------------------------------------
# #664: rescale
# ---------------------------------------------------------------------------


def _scale(value: float, lo: float, hi: float) -> float:
    o_lo, o_hi = ORIGINAL_RANGE
    return min(max(round(lo + (value - o_lo) * (hi - lo) / (o_hi - o_lo), 2), lo), hi)


def rescale_updates(rows: list[tuple[str, dict, list]]) -> dict[str, dict]:
    """New emotions dict for every row whose profile feelings change: {id: emotions}.

    rows: (id, emotions, tags) for rows that have emotions. NEAR_ZERO: every row mapped into
    NEAR_ZERO_RANGE, order kept. SPIKY: the max(SPIKE_MIN, round(SPIKE_SHARE * n)) rows with the
    highest original value among those carrying the conversation tag (the rows the engine reads)
    keep their value, ties by id; every other row is mapped into SPIKY_LOW_RANGE.
    """
    new_values: dict[tuple[str, str], float] = {}
    for name in NEAR_ZERO:
        for row_id, emotions, _tags in rows:
            if name in emotions:
                new_values[(row_id, name)] = _scale(emotions[name], *NEAR_ZERO_RANGE)
    for name in SPIKY:
        having = [
            (row_id, emotions[name], "conversation" in tags)
            for row_id, emotions, tags in rows
            if name in emotions
        ]
        want = max(SPIKE_MIN, round(SPIKE_SHARE * len(having)))
        counted = sorted((x for x in having if x[2]), key=lambda x: (-x[1], x[0]))
        if len(counted) < want:
            raise AugmentError(
                "%s: %d conversation-tagged rows, need %d spikes" % (name, len(counted), want)
            )
        spikes = {x[0] for x in counted[:want]}
        for row_id, value, _counted in having:
            new_values[(row_id, name)] = (
                value if row_id in spikes else _scale(value, *SPIKY_LOW_RANGE)
            )
    updates: dict[str, dict] = {}
    by_id = {row_id: emotions for row_id, emotions, _tags in rows}
    for (row_id, name), value in new_values.items():
        updates.setdefault(row_id, dict(by_id[row_id]))[name] = value
    return updates


# ---------------------------------------------------------------------------
# #654: new rows
# ---------------------------------------------------------------------------


def _pool_lines(
    name: str, plan: FeelingPlan
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(mild, strong) lists of (memory_type, text); refuses a pool that is not the planned size."""
    pool = POOLS.get(name)
    if pool is None:
        raise AugmentError("no text pool for %s" % name)
    mild: list[tuple[str, str]] = []
    strong: list[tuple[str, str]] = []
    for memory_type, planned in zip(MEMORY_TYPES, plan.types):
        lines = pool.get(memory_type, {"mild": [], "strong": []})
        have = len(lines["mild"]) + len(lines["strong"])
        if have != planned:
            raise AugmentError(
                "text pool %s/%s has %d lines, plan %d" % (name, memory_type, have, planned)
            )
        mild += [(memory_type, t) for t in lines["mild"]]
        strong += [(memory_type, t) for t in lines["strong"]]
    if len(strong) != round(STRONG_SHARE * plan.rows):
        raise AugmentError(
            "text pool %s has %d strong lines, plan %d"
            % (name, len(strong), round(STRONG_SHARE * plan.rows))
        )
    return mild, strong


def new_rows(
    donors: dict[str, list[tuple]], existing_ids: set[str], rng: random.Random
) -> list[tuple]:
    """The rows to insert, in INSERT_SQL column order."""
    out = []
    for name, plan in NEW_FEELINGS.items():
        n = plan.rows
        if plan.topic not in TOPICAL_POOL:
            raise AugmentError("%s: topic %r is not in build_db.TOPICAL_POOL" % (name, plan.topic))
        mild, strong = _pool_lines(name, plan)
        rng.shuffle(mild)
        rng.shuffle(strong)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        days = []
        waves = []
        for i in range(
            n
        ):  # one stratum of the 109 days per row, one jittered day in each: distinct, evenly spread
            lo = int(i * SPAN_DAYS / n) + 1
            hi = max(lo, int((i + 1) * SPAN_DAYS / n))
            day = rng.randint(lo, hi)
            wave = plan.low + (plan.high - plan.low) * (
                0.5 + 0.5 * math.sin(2.0 * math.pi * day / plan.period + phase)
            )
            days.append(day)
            waves.append(wave + rng.uniform(-NOISE, NOISE))
        order = sorted(range(n), key=lambda i: (-waves[i], i))
        spikes = set(order[: round(plan.spike_share * n)])
        strong_rows = set(order[: len(strong)])
        if not spikes <= strong_rows:
            raise AugmentError("%s: spikes would take mild text lines" % name)
        for i in range(n):
            memory_type, text = (strong if i in strong_rows else mild).pop()
            if i in spikes:
                value = round(rng.uniform(*plan.spike_band), 1)
            else:
                value = round(min(max(waves[i], VALUE_MIN), VALUE_MAX), 1)
            created = make_created_at(days[i], rng)
            hours = min(rng.expovariate(1.0 / ACCESS_MEAN_HOURS), ACCESS_CAP_HOURS)
            donor = rng.choice(donors[memory_type])
            tags = [memory_type, "conversation", plan.topic]
            metadata = {
                "source": "conversation",
                "importance_echo": donor[0],
                "tags_echo": list(tags),
                "seed_id": "%08x" % rng.getrandbits(32),
                "provider": PROVIDER,
                "schema_version": 1,
            }
            row_id = "%032x" % rng.getrandbits(128)
            if row_id in existing_ids:
                raise AugmentError("id collision %s" % row_id)
            existing_ids.add(row_id)
            out.append(
                (
                    row_id,
                    text,
                    memory_type,
                    TYPE_DOMAIN[memory_type],
                    json.dumps({name: value}),
                    json.dumps(tags),
                    donor[0],
                    donor[1],
                    created.strftime(DATE_FORMAT),
                    (created + timedelta(hours=hours)).strftime(DATE_FORMAT),
                    1,
                    0,
                    json.dumps(metadata),
                    "active",
                    None,
                    donor[2],
                    donor[3],
                )
            )
    return out


# ---------------------------------------------------------------------------
# postcondition
# ---------------------------------------------------------------------------


def verify(base_path: str, out_path: str) -> dict:
    """Check the new database against the base; raises AugmentError listing the first problems.

    Every base row present with every column but emotions_json equal; a base row's emotions
    differ only in value and only on the seven profile feelings; the added rows carry the
    marker, one emotion key each, match the plan, and pass the window table; no
    ABSENT_EMOTIONS key anywhere.
    """
    base = sqlite3.connect(Path(base_path).resolve().as_uri() + "?mode=ro", uri=True)
    out = sqlite3.connect(Path(out_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        columns = [r[1] for r in base.execute("PRAGMA table_info(memories)")]
        if columns != [r[1] for r in out.execute("PRAGMA table_info(memories)")]:
            raise AugmentError("verify: the memories columns changed")
        emo = columns.index("emotions_json")
        meta = columns.index("metadata_json")
        base_rows = {r[0]: r for r in base.execute("SELECT * FROM memories")}
        out_rows = {r[0]: r for r in out.execute("SELECT * FROM memories")}
        problems = []
        profile = set(NEAR_ZERO) | set(SPIKY)
        changed = 0
        for row_id, b in base_rows.items():
            o = out_rows.get(row_id)
            if o is None:
                problems.append("base row %s missing" % row_id)
                continue
            for i, col in enumerate(columns):
                if i != emo and b[i] != o[i]:
                    problems.append("row %s column %s changed" % (row_id, col))
            if b[emo] != o[emo]:
                be, oe = json.loads(b[emo]), json.loads(o[emo])
                if set(be) != set(oe) or any(be[k] != oe[k] and k not in profile for k in be):
                    problems.append("row %s emotions changed outside the profile feelings" % row_id)
                changed += 1
            if len(problems) > 5:
                break
        added: dict[str, list[tuple[int, float]]] = {name: [] for name in NEW_FEELINGS}
        for row_id, o in out_rows.items():
            emotions = json.loads(o[emo])
            if any(k in emotions for k in ABSENT_EMOTIONS):
                problems.append("row %s carries an absent emotion key" % row_id)
            if row_id in base_rows:
                continue
            if json.loads(o[meta]).get("provider") != PROVIDER:
                problems.append("added row %s lacks the marker" % row_id)
            if len(emotions) != 1 or next(iter(emotions)) not in NEW_FEELINGS:
                problems.append("added row %s emotions %r" % (row_id, emotions))
                continue
            name, value = next(iter(emotions.items()))
            added[name].append((day_of(o[columns.index("created_at")]), value))
        for name, plan in NEW_FEELINGS.items():
            if len(added[name]) != plan.rows:
                problems.append("%s: %d rows added, plan %d" % (name, len(added[name]), plan.rows))
            problems += window_problems(name, added[name], plan)
        if problems:
            raise AugmentError("verify failed: " + "; ".join(problems[:6]))
        return {
            "base_rows": len(base_rows),
            "added_rows": len(out_rows) - len(base_rows),
            "rows_rescaled": changed,
        }
    finally:
        base.close()
        out.close()


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _has_marker(path: str) -> bool:
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return (
            conn.execute(
                "SELECT 1 FROM memories WHERE metadata_json LIKE ? LIMIT 1",
                ("%" + PROVIDER + "%",),
            ).fetchone()
            is not None
        )
    except sqlite3.DatabaseError as exc:
        raise AugmentError("%s is not a readable memories database: %s" % (path, exc)) from exc
    finally:
        conn.close()


def _remove(path: str | None) -> None:
    if path and os.path.exists(path):
        os.remove(path)


def augment(args: argparse.Namespace) -> dict:
    src = os.path.abspath(args.input)
    dst = os.path.abspath(args.output)
    if not os.path.isfile(src):
        raise AugmentError("input is not a file: %s" % src)
    if os.path.exists(dst) and os.path.samefile(src, dst):
        raise AugmentError("output is the input path: %s" % dst)
    if not args.allow_other_base and _sha256(src) != BASE_SHA256:
        raise AugmentError(
            "input is not the recorded base (sha256 %s); recover it with: "
            "git show %s:test-data/synthetic-db/memories.db" % (BASE_SHA256[:12], BASE_COMMIT)
        )
    if _has_marker(src):
        raise AugmentError("input already carries augmented rows; start from the base database")
    if os.path.exists(dst):
        if not args.force:
            raise AugmentError(
                "output exists (use --force to replace the base or an augmented database): %s" % dst
            )
        if _sha256(dst) != BASE_SHA256 and not _has_marker(dst):
            raise AugmentError(
                "--force only replaces the base or an augmented database, not %s" % dst
            )
    out_dir = os.path.dirname(dst)
    manifest_in = args.manifest_in or os.path.join(os.path.dirname(src), "manifest.json")
    manifest_out = os.path.join(out_dir, "manifest.json")
    manifest: dict = {}
    if os.path.isfile(manifest_in):
        with open(manifest_in, encoding="utf-8") as fh:
            manifest = json.load(fh)

    tmp_db = tmp_manifest = None
    try:
        fd, tmp_db = tempfile.mkstemp(dir=out_dir, prefix=".augment-", suffix=".db")
        os.close(fd)
        shutil.copyfile(src, tmp_db)  # the input is only ever read
        rng = random.Random(args.seed)
        conn = sqlite3.connect(tmp_db)
        try:
            with conn:
                rows = [
                    (row_id, json.loads(ej), json.loads(tj))
                    for row_id, ej, tj in conn.execute(
                        "SELECT id, emotions_json, tags_json FROM memories "
                        "WHERE emotions_json != '{}'"
                    )
                ]
                for row_id, emotions in sorted(rescale_updates(rows).items()):
                    conn.execute(
                        "UPDATE memories SET emotions_json = ? WHERE id = ?",
                        (json.dumps(emotions), row_id),
                    )
                donors = {
                    t: conn.execute(
                        "SELECT importance, score, recall_count, peak_emotion_intensity "
                        "FROM memories WHERE memory_type = ? AND state = 'active' ORDER BY id",
                        (t,),
                    ).fetchall()
                    for t in MEMORY_TYPES
                }
                existing = {r[0] for r in conn.execute("SELECT id FROM memories")}
                conn.executemany(INSERT_SQL, new_rows(donors, existing, rng))
        finally:
            conn.close()
        os.chmod(tmp_db, 0o644)
        try:
            result = verify(src, tmp_db)
        except AugmentError as exc:
            raise AugmentError("seed %d: %s" % (args.seed, exc)) from exc

        check = sqlite3.connect(Path(tmp_db).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            summary = db_summary(check)
        finally:
            check.close()
        for key, value in summary.items():
            if key == "per_memory_type_counts":
                manifest[key] = {**manifest.get(key, {}), **value}
            else:
                manifest[key] = value
        manifest["augmentation"] = {
            "script": "augment_db.py",
            "seed": args.seed,
            "base_sha256": BASE_SHA256,
            "base_commit": BASE_COMMIT,
            "output_sha256": _sha256(tmp_db),
            "rows_added": {name: plan.rows for name, plan in NEW_FEELINGS.items()},
            "near_zero": list(NEAR_ZERO),
            "low_with_spikes": list(SPIKY),
        }
        fd, tmp_manifest = tempfile.mkstemp(dir=out_dir, prefix=".augment-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        os.chmod(tmp_manifest, 0o644)
        os.replace(
            tmp_db, dst
        )  # the database first, then its manifest; check 20 catches a crash between the two
        tmp_db = None
        os.replace(tmp_manifest, manifest_out)
        tmp_manifest = None
        result["output_sha256"] = manifest["augmentation"]["output_sha256"]
        return result
    finally:
        _remove(tmp_db)
        _remove(tmp_manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help="the base memories.db (read only)")
    parser.add_argument("--output", required=True, help="where to write the augmented database")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--manifest-in",
        help="manifest.json to update (default: beside --input); written beside --output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output that is the base or already augmented",
    )
    parser.add_argument(
        "--allow-other-base",
        action="store_true",
        help="accept an input that is not the recorded base",
    )
    args = parser.parse_args(argv)
    try:
        result = augment(args)
    except AugmentError as exc:
        print("augment_db: FAILED: %s" % exc, file=sys.stderr)
        return 1
    print("augment_db: OK seed=%d %s" % (args.seed, json.dumps(result, sort_keys=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
