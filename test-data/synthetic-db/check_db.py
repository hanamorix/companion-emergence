#!/usr/bin/env python3
"""Validate a synthetic memories.db against the target shape and house-style
rules.

Prints one PASS/FAIL line per check and exits nonzero if any check fails.
Pure stdlib only (sqlite3, json, re, argparse, subprocess, tempfile, shutil).

Checks:
  1. Total row count within +-5% of 19,452.
  2. All 19 memory_types present; each within +-25% of its budget, except
     insight/file_write/journal_entry/event (budget < 25, presence only).
  3. All 16 windows present; every day 1-109 has at least one row.
  4. Zero NULLs in any NOT NULL column (derived from PRAGMA table_info, not
     hardcoded); every JSON column parses.
  5. Zero em-dash (U+2014); banned-phrase rate <= 0.1% of rows.
  6. Zero banned names, word-boundary, case-insensitive. The deny list is supplied
     by the caller (environment variable SYNTHETIC_DB_DENY_NAMES, comma separated)
     and is not stored in the repository; with no list the check is skipped.
  7. Zero exact-duplicate content; near-duplicate (same first 60 chars) rate
     <= 2% of rows.
  8. Brain-open smoke test: real MemoryStore, real venv, on a scratch copy.
  9. Fading fraction within +-10% (relative) of 6669/19452.
 10. content_snapshot non-NULL on exactly the fading rows; on every fading
     row snapshot != content and len(snapshot) > len(content).
 11. decay_coherence: every fading row's decayed `content` reads as a
     coherent (if shortened) thought -- FAILs a row that ends on a dangling
     function word (determiner/preposition/conjunction/possessive/
     auxiliary/etc., see DANGLING_STOPWORDS) before its final punctuation,
     or has fewer than MIN_DECAY_WORDS words.
 12. domain per memory_type matches the SKELETON.md section 3 mapping
     (monologue types -> 'monologue', brain types -> 'brain' (with a small
     documented residual-domain exception for 'fact'), 'us' types -> 'us',
     journal_entry -> 'brain' or 'us', residual types -> a residual domain).
 13. tag-count shape: mode of tag-count among TAGGED rows is 3, >=60% of
     tagged rows have exactly 3 tags, tagged-fraction of ALL rows in
     [30%, 50%], store-wide mean tags/row in [0.8, 1.6].
 14. peak_emotion_intensity: median in [1.0, 2.0], mean in [2.5, 4.0].
 15. recall_count: median in [70, 150].
 16. id column: no NULLs, all values unique.
 17. FTS integrity, checked read-only on the ORIGINAL db file (never via
     MemoryStore, whose boot path self-heals/rebuilds FTS and would mask a
     genuinely broken file): count(memories_fts_docsize) == count(memories),
     and a MATCH query for a common term returns rows.
 18. new_feelings_present (#654) (checks 18-21 run before the FTS and smoke-test checks, which stay last): rest_need, arousal, climax, touch_hunger and
     creative_hunger each have the planned distinct days and, in every 30-day
     window, the planned rows and value range (plan: augment_db.NEW_FEELINGS);
     each such row carries one emotion key, the conversation tag (three tags)
     and a last_accessed_at not before created_at; peace and body_grief occur
     as no emotion key; emotion_vocabulary.json beside the database registers
     creative_hunger through the app's own loader, equal to its canonical entry.
 19. emotion_profiles_hold (#664): near-zero feelings stay in 0.02-0.30 with a
     30-day range under 0.5; low-with-spikes feelings have 3-12% spike rows (all
     conversation-tagged, on 4+ days, the largest 4.0+) and every other value in
     0.10-1.00; the spelling-variant keys keep their base counts and maxima.
 20. manifest_matches_database: manifest.json counts equal the database's.
 21. added_rows_peak_is_max_emotion: on the rows augment_db.py added (found by
     the synthetic-aug-v1 marker), peak_emotion_intensity equals the largest of
     the row's emotion values, the brain's own definition. Other rows' peaks were
     sampled independently of their emotions by build_db.py and are not checked.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile

from augment_db import (
    ABSENT_EMOTIONS,
    NEAR_ZERO,
    NEAR_ZERO_RANGE,
    NEW_FEELINGS,
    PROVIDER,
    SPIKY,
    SPIKY_LOW_RANGE,
    VARIANT_COUNTS,
    WINDOW_DAYS,
    day_of,
    db_summary,
    window_of,
    window_problems,
)
from build_db import (
    BRAIN_TYPES,
    MONOLOGUE_TYPES,
    RESIDUAL_DOMAINS,
    RESIDUAL_TYPES,
    US_TYPES,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(HERE, "memories.db")

FADING_TARGET_RATIO = 6669.0 / 19452.0
FADING_TOLERANCE = 0.10  # +-10% RELATIVE to the target ratio

TAG_MODE_COUNT = 3
TAG_MODE_MIN_SHARE = 0.60
TAGGED_FRACTION_RANGE = (0.30, 0.50)
MEAN_TAGS_PER_ROW_RANGE = (0.8, 1.6)

PEAK_EMOTION_MEDIAN_RANGE = (1.0, 2.0)
PEAK_EMOTION_MEAN_RANGE = (2.5, 4.0)

RECALL_COUNT_MEDIAN_RANGE = (70.0, 150.0)

TARGET_TOTAL = 19452
TOTAL_TOLERANCE = 0.05

TYPE_BUDGETS = {
    "monologue": 3674,
    "fact": 3383,
    "monologue_soul_candidate": 2742,
    "observation": 2663,
    "monologue_trace": 1935,
    "monologue_emotion": 1824,
    "decision": 782,
    "research": 676,
    "note": 635,
    "heartbeat": 612,
    "feeling": 188,
    "dream": 94,
    "question": 91,
    "making": 56,
    "initiate_outbound": 43,
    "event": 20,
    "journal_entry": 17,
    "file_write": 10,
    "insight": 7,
}
# Types whose budget is under 25: presence-only, no +-25% band check.
PRESENCE_ONLY_TYPES = {"insight", "file_write", "journal_entry", "event"}
TYPE_TOLERANCE = 0.25

NUM_WINDOWS = 16
SPAN_DAYS = 109

BANNED_PHRASES = ["delve", "tapestry", "testament to"]
BANNED_PHRASE_RATE_LIMIT = 0.001  # 0.1% of rows

BANNED_NAMES = [n.strip() for n in os.environ.get("SYNTHETIC_DB_DENY_NAMES", "").split(",") if n.strip()]

EM_DASH = "—"

# Function words a fading row's decayed `content` must never end on (before
# its final punctuation) -- the exact list from the decay-coherence fix
# spec: determiners, prepositions, conjunctions, possessives, auxiliaries,
# and a few other closers that can't stand alone as a clause end. Must stay
# a SUBSET of build_db.DANGLING_WORDS (which is a defense-in-depth superset
# of this exact list) so the generator and this check agree.
DANGLING_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "with",
    "and", "but", "or", "his", "her", "my", "their", "its",
    "was", "were", "is", "be", "been", "how", "that", "which", "who",
    "if", "as", "from", "by", "about", "into", "than", "so",
}
MIN_DECAY_WORDS = 4

BRAIN_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
VENV_PYTHON = os.environ.get("SYNTHETIC_DB_PYTHON", sys.executable)


class CheckResult:
    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        if self.detail:
            return "%s: %s -- %s" % (status, self.name, self.detail)
        return "%s: %s" % (status, self.name)


def fetch_all_text_values(conn: sqlite3.Connection):
    """Yield every string value that belongs to a "text column" surface:
    content, content_snapshot, and every string value found anywhere inside
    metadata_json / tags_json / emotions_json (recursively, since these are
    small generic JSON objects/arrays whose string leaves are the only place
    banned text could hide)."""
    cur = conn.execute(
        "SELECT content, content_snapshot, metadata_json, tags_json, emotions_json FROM memories"
    )
    for content, snapshot, metadata_json, tags_json, emotions_json in cur:
        if content is not None:
            yield content
        if snapshot is not None:
            yield snapshot
        for raw in (metadata_json, tags_json, emotions_json):
            if raw is None:
                continue
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            yield from _walk_strings(parsed)


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_strings(item)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_total_row_count(conn: sqlite3.Connection) -> CheckResult:
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    lo = TARGET_TOTAL * (1 - TOTAL_TOLERANCE)
    hi = TARGET_TOTAL * (1 + TOTAL_TOLERANCE)
    ok = lo <= total <= hi
    return CheckResult(
        "total_row_count",
        ok,
        "total=%d target=%d (+-%.0f%%) range=[%.0f, %.0f]" % (
            total, TARGET_TOTAL, TOTAL_TOLERANCE * 100, lo, hi
        ),
    )


def check_type_budgets(conn: sqlite3.Connection) -> CheckResult:
    rows = conn.execute(
        "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"
    ).fetchall()
    counts = {t: c for t, c in rows}
    problems = []
    missing_types = sorted(set(TYPE_BUDGETS) - set(counts))
    if missing_types:
        problems.append("missing types: %s" % missing_types)
    for mtype, budget in TYPE_BUDGETS.items():
        actual = counts.get(mtype, 0)
        if mtype in PRESENCE_ONLY_TYPES:
            if actual <= 0:
                problems.append("%s: absent (presence-only type)" % mtype)
            continue
        lo = budget * (1 - TYPE_TOLERANCE)
        hi = budget * (1 + TYPE_TOLERANCE)
        if not (lo <= actual <= hi):
            problems.append("%s: actual=%d budget=%d range=[%.0f, %.0f]" % (
                mtype, actual, budget, lo, hi
            ))
    unknown_types = sorted(set(counts) - set(TYPE_BUDGETS))
    if unknown_types:
        problems.append("unrecognized memory_type values: %s" % unknown_types)
    ok = not problems
    return CheckResult("memory_type_budgets", ok, "; ".join(problems) if problems else "all 19 types within band")


def check_window_and_day_coverage(conn: sqlite3.Connection) -> CheckResult:
    # window is not a stored column; reconstruct from the created_at day number
    # (augment_db.day_of / window_of, the one copy of that mapping).
    rows = conn.execute("SELECT created_at FROM memories WHERE created_at IS NOT NULL").fetchall()
    days_present = set()
    for (created_at,) in rows:
        try:
            day_num = day_of(created_at)
        except Exception:
            continue
        if 1 <= day_num <= SPAN_DAYS:
            days_present.add(day_num)

    missing_days = [d for d in range(1, SPAN_DAYS + 1) if d not in days_present]

    windows_present = {window_of(d) for d in days_present}
    missing_windows = [w for w in range(1, NUM_WINDOWS + 1) if w not in windows_present]

    problems = []
    if missing_windows:
        problems.append("missing windows: %s" % missing_windows)
    if missing_days:
        shown = missing_days[:15]
        more = "" if len(missing_days) <= 15 else " (+%d more)" % (len(missing_days) - 15)
        problems.append("missing days: %s%s" % (shown, more))
    ok = not problems
    return CheckResult(
        "window_and_day_coverage", ok,
        "; ".join(problems) if problems else "all 16 windows and all 109 days present",
    )


def check_not_null_and_json(conn: sqlite3.Connection) -> CheckResult:
    columns = conn.execute("PRAGMA table_info(memories)").fetchall()
    not_null_cols = [row[1] for row in columns if row[3] == 1]  # row: cid,name,type,notnull,dflt,pk
    problems = []
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if total == 0:
        return CheckResult("not_null_and_json", False, "table is empty, nothing to check")

    for col in not_null_cols:
        n_null = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE %s IS NULL" % col
        ).fetchone()[0]
        if n_null:
            problems.append("%s: %d NULLs" % (col, n_null))

    for col in ("emotions_json", "tags_json", "metadata_json"):
        rows = conn.execute("SELECT id, %s FROM memories" % col).fetchall()
        bad = 0
        bad_example = None
        for row_id, raw in rows:
            if raw is None:
                bad += 1
                bad_example = bad_example or (row_id, "NULL")
                continue
            try:
                json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                bad += 1
                bad_example = bad_example or (row_id, raw[:60])
        if bad:
            problems.append("%s: %d rows fail to parse as JSON (e.g. %s)" % (col, bad, bad_example))

    ok = not problems
    return CheckResult(
        "not_null_and_json", ok,
        "; ".join(problems) if problems else "no NULLs in %d NOT NULL columns; all JSON columns parse" % len(not_null_cols),
    )


def check_house_style(conn: sqlite3.Connection) -> CheckResult:
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if total == 0:
        return CheckResult("house_style_banned_phrases", False, "table is empty, nothing to check")

    em_dash_rows = 0
    phrase_hits = {p: 0 for p in BANNED_PHRASES}
    phrase_pattern = {
        p: re.compile(re.escape(p), re.IGNORECASE) for p in BANNED_PHRASES
    }

    for value in fetch_all_text_values(conn):
        if EM_DASH in value:
            em_dash_rows += 1
        for p, pattern in phrase_pattern.items():
            if pattern.search(value):
                phrase_hits[p] += 1

    total_phrase_hits = sum(phrase_hits.values())
    limit = total * BANNED_PHRASE_RATE_LIMIT
    problems = []
    if em_dash_rows:
        problems.append("em-dash found in %d text values" % em_dash_rows)
    if total_phrase_hits > limit:
        problems.append(
            "banned-phrase hits=%d exceeds 0.1%% of %d rows (%.2f) -- breakdown %s" % (
                total_phrase_hits, total, limit, phrase_hits
            )
        )
    ok = not problems
    return CheckResult(
        "house_style_banned_phrases", ok,
        "; ".join(problems) if problems else "no em-dash; banned-phrase hits=%d <= limit %.2f" % (total_phrase_hits, limit),
    )


def check_banned_names(conn: sqlite3.Connection) -> CheckResult:
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if total == 0:
        return CheckResult("banned_names", False, "table is empty, nothing to check")
    if not BANNED_NAMES:
        return CheckResult("banned_names", True, "skipped: no deny list supplied (set SYNTHETIC_DB_DENY_NAMES)")

    patterns = {name: re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE) for name in BANNED_NAMES}
    hits = {name: 0 for name in BANNED_NAMES}
    for value in fetch_all_text_values(conn):
        for name, pattern in patterns.items():
            if pattern.search(value):
                hits[name] += 1

    total_hits = sum(hits.values())
    ok = total_hits == 0
    return CheckResult(
        "banned_names", ok,
        "clean" if ok else "hits: %s" % {k: v for k, v in hits.items() if v},
    )


def check_duplicates(conn: sqlite3.Connection) -> CheckResult:
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if total == 0:
        return CheckResult("duplicates", False, "table is empty, nothing to check")

    exact_dupe_rows = conn.execute(
        """
        SELECT COALESCE(SUM(cnt - 1), 0) FROM (
            SELECT COUNT(*) AS cnt FROM memories GROUP BY content HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    prefix_rows = conn.execute(
        "SELECT substr(content, 1, 60) AS prefix, COUNT(*) AS cnt FROM memories GROUP BY prefix HAVING cnt > 1"
    ).fetchall()
    near_dupe_row_total = sum(cnt for _, cnt in prefix_rows)
    near_dupe_rate = near_dupe_row_total / total

    problems = []
    if exact_dupe_rows:
        problems.append("exact-duplicate content rows: %d" % exact_dupe_rows)
    if near_dupe_rate > 0.02:
        problems.append(
            "near-duplicate (first-60-char) rows=%d rate=%.4f exceeds 2%%" % (
                near_dupe_row_total, near_dupe_rate
            )
        )
    ok = not problems
    detail = "; ".join(problems) if problems else (
        "0 exact duplicates; near-dup rows=%d rate=%.4f" % (near_dupe_row_total, near_dupe_rate)
    )
    return CheckResult("duplicates", ok, detail)


def check_fading_fraction(conn: sqlite3.Connection) -> CheckResult:
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if total == 0:
        return CheckResult("fading_fraction", False, "table is empty, nothing to check")
    fading = conn.execute("SELECT COUNT(*) FROM memories WHERE state = 'fading'").fetchone()[0]
    ratio = fading / total
    lo = FADING_TARGET_RATIO * (1 - FADING_TOLERANCE)
    hi = FADING_TARGET_RATIO * (1 + FADING_TOLERANCE)
    ok = lo <= ratio <= hi
    return CheckResult(
        "fading_fraction", ok,
        "fading=%d/%d=%.4f target=%.4f range=[%.4f, %.4f]" % (
            fading, total, ratio, FADING_TARGET_RATIO, lo, hi
        ),
    )


def check_content_snapshot_semantics(conn: sqlite3.Connection) -> CheckResult:
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if total == 0:
        return CheckResult("content_snapshot_semantics", False, "table is empty, nothing to check")

    problems = []

    mismatched = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE (content_snapshot IS NOT NULL) != (state = 'fading')"
    ).fetchone()[0]
    if mismatched:
        problems.append(
            "%d rows have content_snapshot NULL-ness not matching state=='fading'" % mismatched
        )

    rows = conn.execute(
        "SELECT id, content, content_snapshot FROM memories WHERE state = 'fading'"
    ).fetchall()
    bad_equal = 0
    bad_not_longer = 0
    bad_example = None
    for row_id, content, snapshot in rows:
        if snapshot is None:
            continue  # already reported above
        if snapshot == content:
            bad_equal += 1
            bad_example = bad_example or (row_id, "snapshot == content")
        elif not (len(snapshot) > len(content)):
            bad_not_longer += 1
            bad_example = bad_example or (row_id, "len(snapshot) <= len(content)")
    if bad_equal:
        problems.append("%d fading rows have snapshot == content" % bad_equal)
    if bad_not_longer:
        problems.append("%d fading rows have len(snapshot) <= len(content)" % bad_not_longer)
    if bad_example:
        problems.append("e.g. id=%s (%s)" % bad_example)

    ok = not problems
    return CheckResult(
        "content_snapshot_semantics", ok,
        "; ".join(problems) if problems else (
            "content_snapshot non-NULL on exactly the %d fading rows; "
            "snapshot != content and strictly longer on all of them" % len(rows)
        ),
    )


def check_decay_coherence(conn: sqlite3.Connection) -> CheckResult:
    """Every fading row's decayed `content` must read as a coherent, if
    shortened, thought -- not a mid-phrase cut with a period slapped on
    (the "an old.", "were allowed to both." bug: VOICE-AUDIT-FINAL.md's
    fading-decay section). FAILs a row if its content, stripped of a
    trailing '.'/'!'/'?', ends on a dangling function word from
    DANGLING_STOPWORDS, or has fewer than MIN_DECAY_WORDS words."""
    rows = conn.execute(
        "SELECT id, content FROM memories WHERE state = 'fading'"
    ).fetchall()
    if not rows:
        return CheckResult("decay_coherence", False, "no fading rows to check")

    bad = []
    for row_id, content in rows:
        if content is None:
            bad.append((row_id, "content is NULL"))
            continue
        text = content.strip()
        words = text.split()
        if len(words) < MIN_DECAY_WORDS:
            bad.append((row_id, "only %d word(s): %r" % (len(words), text)))
            continue
        last = re.sub(r'[.,;:!?"\')\]]+$', "", words[-1]).lower()
        if last in DANGLING_STOPWORDS:
            bad.append((row_id, "ends on dangling word %r: %r" % (last, text)))

    ok = not bad
    if ok:
        detail = "%d/%d fading rows end coherently" % (len(rows), len(rows))
    else:
        examples = "; ".join("id=%s (%s)" % b for b in bad[:5])
        detail = "%d/%d fading rows FAIL coherence: %s" % (len(bad), len(rows), examples)
    return CheckResult("decay_coherence", ok, detail)


def check_domain_type_mapping(conn: sqlite3.Connection) -> CheckResult:
    """Domain per memory_type must match SKELETON.md section 3's mapping, as
    implemented by build_db.assign_domains: monologue types -> 'monologue';
    brain types -> 'brain', except a small documented residual-domain
    exception for 'fact' rows (SKELETON.md section 3's "a handful of facts"
    in the residual domains); shared-relationship types -> 'us';
    journal_entry -> 'brain' or 'us'; residual types -> a residual domain.
    """
    rows = conn.execute(
        "SELECT memory_type, domain, COUNT(*) FROM memories GROUP BY memory_type, domain"
    ).fetchall()
    problems = []
    for mtype, domain, count in rows:
        if mtype in MONOLOGUE_TYPES:
            if domain != "monologue":
                problems.append("%s: %d rows in domain=%r, expected 'monologue'" % (mtype, count, domain))
        elif mtype in BRAIN_TYPES:
            if domain != "brain" and not (mtype == "fact" and domain in RESIDUAL_DOMAINS):
                problems.append(
                    "%s: %d rows in domain=%r, expected 'brain' (or a residual domain for 'fact')"
                    % (mtype, count, domain)
                )
        elif mtype in US_TYPES:
            if domain != "us":
                problems.append("%s: %d rows in domain=%r, expected 'us'" % (mtype, count, domain))
        elif mtype == "journal_entry":
            if domain not in ("brain", "us"):
                problems.append(
                    "%s: %d rows in domain=%r, expected 'brain' or 'us'" % (mtype, count, domain)
                )
        elif mtype in RESIDUAL_TYPES:
            if domain not in RESIDUAL_DOMAINS:
                problems.append(
                    "%s: %d rows in domain=%r, expected one of %s" % (mtype, count, domain, RESIDUAL_DOMAINS)
                )
        else:
            problems.append("%s: unrecognized memory_type (domain=%r, %d rows)" % (mtype, domain, count))

    ok = not problems
    return CheckResult(
        "domain_type_mapping", ok,
        "; ".join(problems) if problems else "domain matches SKELETON.md section 3 for every memory_type",
    )


def check_tag_count_shape(conn: sqlite3.Connection) -> CheckResult:
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if total == 0:
        return CheckResult("tag_count_shape", False, "table is empty, nothing to check")

    rows = conn.execute("SELECT tags_json FROM memories").fetchall()
    counts = []
    for (raw,) in rows:
        try:
            tags = json.loads(raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            tags = []
        counts.append(len(tags) if isinstance(tags, list) else 0)

    tagged_counts = [c for c in counts if c > 0]
    tagged_fraction = len(tagged_counts) / total
    mean_tags_all = sum(counts) / total

    problems = []
    lo_tf, hi_tf = TAGGED_FRACTION_RANGE
    if not (lo_tf <= tagged_fraction <= hi_tf):
        problems.append("tagged_fraction=%.4f outside [%.2f, %.2f]" % (tagged_fraction, lo_tf, hi_tf))

    lo_mt, hi_mt = MEAN_TAGS_PER_ROW_RANGE
    if not (lo_mt <= mean_tags_all <= hi_mt):
        problems.append("mean_tags_per_row=%.4f outside [%.2f, %.2f]" % (mean_tags_all, lo_mt, hi_mt))

    mode_share = 0.0
    mode_val = None
    if tagged_counts:
        freq = {}
        for c in tagged_counts:
            freq[c] = freq.get(c, 0) + 1
        mode_val = max(freq, key=freq.get)
        mode_share = freq.get(TAG_MODE_COUNT, 0) / len(tagged_counts)
        if mode_val != TAG_MODE_COUNT:
            problems.append("mode tag-count among tagged rows is %r, expected %d" % (mode_val, TAG_MODE_COUNT))
        if mode_share < TAG_MODE_MIN_SHARE:
            problems.append(
                "share of tagged rows with exactly %d tags=%.4f below %.2f"
                % (TAG_MODE_COUNT, mode_share, TAG_MODE_MIN_SHARE)
            )
    else:
        problems.append("no tagged rows at all")

    ok = not problems
    return CheckResult(
        "tag_count_shape", ok,
        "; ".join(problems) if problems else (
            "tagged_fraction=%.4f mean_tags/row=%.4f mode=%r share_exactly_%d=%.4f"
            % (tagged_fraction, mean_tags_all, mode_val, TAG_MODE_COUNT, mode_share)
        ),
    )


def check_peak_emotion_intensity(conn: sqlite3.Connection) -> CheckResult:
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if total == 0:
        return CheckResult("peak_emotion_intensity", False, "table is empty, nothing to check")

    values = [r[0] for r in conn.execute("SELECT peak_emotion_intensity FROM memories").fetchall()]
    mean = statistics.mean(values)
    median = statistics.median(values)

    problems = []
    lo_med, hi_med = PEAK_EMOTION_MEDIAN_RANGE
    if not (lo_med <= median <= hi_med):
        problems.append("median=%.4f outside [%.2f, %.2f]" % (median, lo_med, hi_med))
    lo_mean, hi_mean = PEAK_EMOTION_MEAN_RANGE
    if not (lo_mean <= mean <= hi_mean):
        problems.append("mean=%.4f outside [%.2f, %.2f]" % (mean, lo_mean, hi_mean))

    ok = not problems
    return CheckResult(
        "peak_emotion_intensity", ok,
        "; ".join(problems) if problems else "median=%.4f mean=%.4f" % (median, mean),
    )


def check_recall_count(conn: sqlite3.Connection) -> CheckResult:
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if total == 0:
        return CheckResult("recall_count", False, "table is empty, nothing to check")

    values = [r[0] for r in conn.execute("SELECT recall_count FROM memories").fetchall()]
    median = statistics.median(values)

    lo, hi = RECALL_COUNT_MEDIAN_RANGE
    ok = lo <= median <= hi
    return CheckResult(
        "recall_count", ok,
        "median=%.4f range=[%.1f, %.1f]" % (median, lo, hi),
    )


def check_id_column(conn: sqlite3.Connection) -> CheckResult:
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if total == 0:
        return CheckResult("id_column", False, "table is empty, nothing to check")

    null_count = conn.execute("SELECT COUNT(*) FROM memories WHERE id IS NULL").fetchone()[0]
    # COUNT(DISTINCT id) excludes NULLs on its own, so compare against the
    # non-NULL row count -- otherwise a NULL id also misreports as a
    # "duplicate" (COUNT(DISTINCT ...) silently drops NULLs from the count).
    non_null_total = total - null_count
    distinct_count = conn.execute(
        "SELECT COUNT(DISTINCT id) FROM memories WHERE id IS NOT NULL"
    ).fetchone()[0]

    problems = []
    if null_count:
        problems.append("%d NULL ids" % null_count)
    if distinct_count != non_null_total:
        problems.append(
            "%d distinct non-NULL ids != %d non-NULL rows (duplicates present)"
            % (distinct_count, non_null_total)
        )

    ok = not problems
    return CheckResult(
        "id_column", ok,
        "; ".join(problems) if problems else "no NULLs, %d/%d unique" % (distinct_count, total),
    )


def check_fts_integrity(db_path: str) -> CheckResult:
    """Opened READ-ONLY via plain sqlite3 in uri mode=ro on the ORIGINAL
    output db file -- never via MemoryStore, whose boot path self-heals /
    rebuilds FTS and would mask a genuinely broken file."""
    if not os.path.isfile(db_path):
        return CheckResult("fts_integrity", False, "db not found at %s" % db_path)

    uri = "file:" + db_path + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        return CheckResult("fts_integrity", False, "could not open read-only: %r" % exc)

    try:
        mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        try:
            docsize_count = conn.execute("SELECT COUNT(*) FROM memories_fts_docsize").fetchone()[0]
        except sqlite3.Error as exc:
            return CheckResult("fts_integrity", False, "memories_fts_docsize unreadable: %r" % exc)

        problems = []
        if docsize_count != mem_count:
            problems.append(
                "memories_fts_docsize count=%d != memories count=%d" % (docsize_count, mem_count)
            )

        term_row = conn.execute(
            "SELECT term FROM memories_vocab ORDER BY doc DESC LIMIT 1"
        ).fetchone()
        match_count = 0
        term = None
        if term_row is None:
            problems.append("memories_vocab returned no terms (FTS index looks empty)")
        else:
            term = term_row[0]
            match_count = conn.execute(
                "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH ?", (term,)
            ).fetchone()[0]
            if match_count <= 0:
                problems.append("MATCH query for common term %r returned 0 rows" % term)

        ok = not problems
        detail = "; ".join(problems) if problems else (
            "docsize=%d == memories=%d; MATCH %r -> %d rows" % (docsize_count, mem_count, term, match_count)
        )
        return CheckResult("fts_integrity", ok, detail)
    except Exception as exc:
        return CheckResult("fts_integrity", False, "exception: %r" % exc)
    finally:
        conn.close()


def check_brain_open_smoke_test(db_path: str) -> CheckResult:
    if not os.path.isfile(VENV_PYTHON):
        return CheckResult(
            "brain_open_smoke_test", False,
            "venv python not found at %s" % VENV_PYTHON,
        )
    if not os.path.isfile(db_path):
        return CheckResult("brain_open_smoke_test", False, "db not found at %s" % db_path)

    tmpdir = tempfile.mkdtemp(prefix="check_db_smoke_")
    try:
        scratch_db = os.path.join(tmpdir, "memories.db")
        shutil.copy2(db_path, scratch_db)

        before_mtime = os.path.getmtime(db_path)
        before_size = os.path.getsize(db_path)

        script = (
            "from brain.memory.store import MemoryStore\n"
            "store = MemoryStore(%r)\n"
            "rows = store.list_active(limit=50)\n"
            "print('OK', len(rows))\n" % scratch_db
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = BRAIN_REPO + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [VENV_PYTHON, "-c", script],
            cwd=BRAIN_REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

        after_mtime = os.path.getmtime(db_path)
        after_size = os.path.getsize(db_path)
        if before_mtime != after_mtime or before_size != after_size:
            return CheckResult(
                "brain_open_smoke_test", False,
                "original DB file was modified by the smoke test (mtime/size changed) "
                "-- this must never happen, the check must operate on a copy only",
            )

        if proc.returncode != 0:
            return CheckResult(
                "brain_open_smoke_test", False,
                "subprocess exit=%d stderr(tail)=%r" % (proc.returncode, proc.stderr[-500:]),
            )
        return CheckResult(
            "brain_open_smoke_test", True,
            "MemoryStore(...).list_active() ran cleanly: %s" % proc.stdout.strip(),
        )
    except Exception as exc:
        return CheckResult("brain_open_smoke_test", False, "exception: %r" % exc)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# emotion checks (#654 new feelings, #664 profiles) and the manifest
# ---------------------------------------------------------------------------

# A near-zero feeling's 30-day range stays below this (the brief's wording). It follows from the
# 0.02-0.30 value band checked first; it is kept so the property is stated, not only implied.
NEAR_ZERO_MAX_RANGE = 0.5
SPIKE_LEVEL = 1.0  # a low-with-spikes value above this is a spike
SPIKE_SHARE_RANGE = (0.03, 0.12)  # augment_db rescales so about 6% of rows are spikes
SPIKE_MIN_DAYS = 4
SPIKE_MIN_PEAK = 4.0
VARIANT_MIN_PEAK = 7.0  # the spelling-variant keys must keep their strong rows (a low-profile rescale would remove them)
VALUE_EPS = 1e-9


def _scan_emotions(conn: sqlite3.Connection) -> dict:
    """{feeling: [(id, day, value, tags, created_at, last_accessed_at, number_of_keys_on_the_row)]}."""
    scan: dict = {}
    for row_id, emotions_json, tags_json, created, accessed in conn.execute(
        "SELECT id, emotions_json, tags_json, created_at, last_accessed_at FROM memories WHERE emotions_json != '{}'"
    ):
        emotions = json.loads(emotions_json)
        tags = json.loads(tags_json)
        for name, value in emotions.items():
            scan.setdefault(name, []).append((row_id, day_of(created), value, tags, created, accessed, len(emotions)))
    return scan


def _vocabulary_problems(db_path: str) -> list:
    """The shipped emotion_vocabulary.json must register creative_hunger through the app's loader, as its canonical entry."""
    vocab = os.path.join(os.path.dirname(os.path.abspath(db_path)), "emotion_vocabulary.json")
    if not os.path.isfile(vocab):
        return ["emotion_vocabulary.json missing beside the database"]
    if not os.path.isfile(VENV_PYTHON):
        return ["venv python not found at %s" % VENV_PYTHON]
    tmpdir = tempfile.mkdtemp(prefix="check_db_vocab_")
    try:
        scratch = os.path.join(tmpdir, "emotion_vocabulary.json")
        shutil.copy2(vocab, scratch)
        script = (
            "import json\n"
            "from pathlib import Path\n"
            "from brain.emotion import vocabulary\n"
            "from brain.emotion._canonical_personal_emotions import _CANONICAL\n"
            "from brain.emotion.persona_loader import load_persona_vocabulary\n"
            "registered = load_persona_vocabulary(Path(%r))\n"
            "got = vocabulary.get('creative_hunger')\n"
            "print(json.dumps([registered, got == _CANONICAL['creative_hunger'], got.decay_half_life_days, got.category]))\n"
            % scratch
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = BRAIN_REPO + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [VENV_PYTHON, "-c", script], cwd=BRAIN_REPO, env=env, capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            return ["vocabulary load failed: %r" % proc.stderr[-300:]]
        registered, canonical, half_life, category = json.loads(proc.stdout.strip().splitlines()[-1])
        if registered != 1 or not canonical or half_life != 5.0 or category != "persona_extension":
            return [
                "creative_hunger via load_persona_vocabulary: registered=%s equals_canonical=%s half_life=%s category=%s"
                % (registered, canonical, half_life, category)
            ]
        return []
    except Exception as exc:
        return ["vocabulary check raised %r" % exc]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def check_new_feelings_present(conn: sqlite3.Connection, db_path: str) -> CheckResult:
    scan = _scan_emotions(conn)
    problems = []
    for name, plan in NEW_FEELINGS.items():
        rows = scan.get(name, [])
        if not rows:
            problems.append("%s: no rows" % name)
            continue
        problems += window_problems(name, [(r[1], r[2]) for r in rows], plan)
        bad = [
            r[0] for r in rows
            if r[6] != 1 or "conversation" not in r[3] or len(r[3]) != 3 or r[5] is None or r[5] < r[4]
        ]
        if bad:
            problems.append(
                "%s: %d rows not (single emotion key, conversation tag, 3 tags, last_accessed_at >= created_at), e.g. %s"
                % (name, len(bad), bad[0])
            )
    for key in ABSENT_EMOTIONS:
        if key in scan:
            problems.append("%s occurs as an emotion key in %d rows" % (key, len(scan[key])))
    problems += _vocabulary_problems(db_path)
    detail = "; ".join(problems) if problems else (
        "rows/days: %s; every 30-day window meets the plan; creative_hunger registers at 5.0 days via the app loader"
        % ", ".join("%s %d/%d" % (n, len(scan[n]), len({r[1] for r in scan[n]})) for n in NEW_FEELINGS)
    )
    return CheckResult("new_feelings_present", not problems, detail)


def check_emotion_profiles_hold(conn: sqlite3.Connection) -> CheckResult:
    scan = _scan_emotions(conn)
    problems = []
    lo, hi = NEAR_ZERO_RANGE
    for name in NEAR_ZERO:
        rows = scan.get(name, [])
        if not rows:
            problems.append("%s: no rows" % name)
            continue
        outside = sum(1 for r in rows if not (lo - VALUE_EPS <= r[2] <= hi + VALUE_EPS))
        if outside:
            problems.append("%s: %d of %d values outside [%.2f, %.2f], max %.2f" % (name, outside, len(rows), lo, hi, max(r[2] for r in rows)))
            continue
        for start in range(1, SPAN_DAYS - WINDOW_DAYS + 2):
            values = [r[2] for r in rows if start <= r[1] < start + WINDOW_DAYS]
            if values and max(values) - min(values) >= NEAR_ZERO_MAX_RANGE:
                problems.append("%s: days %d-%d range %.2f" % (name, start, start + WINDOW_DAYS - 1, max(values) - min(values)))
                break
    low_lo, _low_hi = SPIKY_LOW_RANGE
    for name in SPIKY:
        rows = scan.get(name, [])
        if not rows:
            problems.append("%s: no rows" % name)
            continue
        spikes = [r for r in rows if r[2] > SPIKE_LEVEL + VALUE_EPS]
        share = len(spikes) / len(rows)
        if not (SPIKE_SHARE_RANGE[0] <= share <= SPIKE_SHARE_RANGE[1]):
            problems.append("%s: spike share %.3f outside %s" % (name, share, SPIKE_SHARE_RANGE))
        if len({r[1] for r in spikes}) < SPIKE_MIN_DAYS:
            problems.append("%s: spikes on %d days, need %d" % (name, len({r[1] for r in spikes}), SPIKE_MIN_DAYS))
        if any("conversation" not in r[3] for r in spikes):
            problems.append("%s: a spike row is not conversation-tagged (the engine would not read it)" % name)
        if not spikes or max(r[2] for r in spikes) < SPIKE_MIN_PEAK:
            problems.append("%s: largest spike %.1f, need %.1f" % (name, max((r[2] for r in spikes), default=0.0), SPIKE_MIN_PEAK))
        if any(r[2] < low_lo - VALUE_EPS for r in rows):
            problems.append("%s: a value below %.2f" % (name, low_lo))
    for key, count in VARIANT_COUNTS.items():
        rows = scan.get(key, [])
        if len(rows) != count or max((r[2] for r in rows), default=0.0) < VARIANT_MIN_PEAK:
            problems.append(
                "%s: %d rows (base %d), max %.1f (need %.1f)"
                % (key, len(rows), count, max((r[2] for r in rows), default=0.0), VARIANT_MIN_PEAK)
            )
    detail = "; ".join(problems) if problems else (
        "near-zero %s within [%.2f, %.2f]; low-with-spikes %s; spelling-variant keys unchanged"
        % (",".join(NEAR_ZERO), lo, hi, ",".join(SPIKY))
    )
    return CheckResult("emotion_profiles_hold", not problems, detail)


def check_manifest_matches_database(conn: sqlite3.Connection, db_path: str) -> CheckResult:
    path = os.path.join(os.path.dirname(os.path.abspath(db_path)), "manifest.json")
    if not os.path.isfile(path):
        return CheckResult("manifest_matches_database", False, "no manifest.json beside the database")
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    summary = db_summary(conn)
    problems = []
    for key in ("total_rows", "per_memory_type_counts", "per_window_counts", "active_count", "fading_count", "inactive_count", "day_span"):
        if manifest.get(key) != summary[key]:
            problems.append("%s: manifest %r, database %r" % (key, manifest.get(key), summary[key]))
    stats = manifest.get("per_day_stats", {})
    for key in ("min", "max"):
        if stats.get(key) != summary["per_day_stats"][key]:
            problems.append("per_day_stats.%s: manifest %r, database %r" % (key, stats.get(key), summary["per_day_stats"][key]))
    if abs(stats.get("mean", -1.0) - summary["per_day_stats"]["mean"]) > 1e-6:
        problems.append("per_day_stats.mean differs")
    return CheckResult(
        "manifest_matches_database", not problems,
        "; ".join(problems) if problems else "manifest counts equal the database (total=%d)" % summary["total_rows"],
    )


def check_added_rows_peak_is_max_emotion(conn: sqlite3.Connection) -> CheckResult:
    """An added row's peak is the largest of its emotion values (brain/memory/store.py, Memory.create_new)."""
    added = 0
    bad = []
    for row_id, emotions_json, metadata_json, peak in conn.execute(
        "SELECT id, emotions_json, metadata_json, peak_emotion_intensity FROM memories "
        "WHERE metadata_json LIKE ?",
        ("%" + PROVIDER + "%",),
    ):
        if json.loads(metadata_json).get("provider") != PROVIDER:
            continue
        added += 1
        expected = max((float(v) for v in json.loads(emotions_json).values()), default=0.0)
        if abs(peak - expected) > VALUE_EPS:
            bad.append("%s peak %r, max emotion %r" % (row_id, peak, expected))
    if not added:
        return CheckResult("added_rows_peak_is_max_emotion", False, "no added rows found (marker %s)" % PROVIDER)
    detail = (
        "%d of %d added rows differ, e.g. %s" % (len(bad), added, bad[0])
        if bad
        else "%d added rows, peak equals the largest emotion value" % added
    )
    return CheckResult("added_rows_peak_is_max_emotion", not bad, detail)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_all_checks(db_path: str):
    if not os.path.isfile(db_path):
        return [CheckResult("db_exists", False, "no file at %s" % db_path)]

    conn = sqlite3.connect(db_path)
    try:
        results = [
            check_total_row_count(conn),
            check_type_budgets(conn),
            check_window_and_day_coverage(conn),
            check_not_null_and_json(conn),
            check_house_style(conn),
            check_banned_names(conn),
            check_duplicates(conn),
            check_fading_fraction(conn),
            check_content_snapshot_semantics(conn),
            check_decay_coherence(conn),
            check_domain_type_mapping(conn),
            check_tag_count_shape(conn),
            check_peak_emotion_intensity(conn),
            check_recall_count(conn),
            check_id_column(conn),
            check_new_feelings_present(conn, db_path),
            check_emotion_profiles_hold(conn),
            check_manifest_matches_database(conn, db_path),
            check_added_rows_peak_is_max_emotion(conn),
        ]
    finally:
        conn.close()

    # FTS must be checked against the ORIGINAL file via a fresh read-only
    # connection, never through the connection above (and never through
    # MemoryStore -- see check_fts_integrity's docstring).
    results.append(check_fts_integrity(db_path))
    results.append(check_brain_open_smoke_test(db_path))
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate a synthetic memories.db.")
    parser.add_argument("db_path", nargs="?", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    results = run_all_checks(args.db_path)
    for r in results:
        print(r.line())

    all_passed = all(r.passed for r in results)
    print()
    print("RESULT: %s (%d/%d checks passed)" % (
        "PASS" if all_passed else "FAIL",
        sum(1 for r in results if r.passed),
        len(results),
    ))
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
