#!/usr/bin/env python3
"""Build a synthetic companion-memory memories.db from parts/*.jsonl.

Reads writer-generated JSONL "parts" files (each line a fully synthetic memory
row, no real content), derives every database column, and writes memories.db
plus manifest.json. Source-blind: never reads anything under the real
companion-emergence app directory, and never touches the real memories.db or
parts/ unless the caller explicitly points --parts-dir/--out-dir there.

Pure stdlib only (sqlite3, json, argparse, random.Random, re, uuid formatting,
datetime). No numpy, no embedding model, nothing that needs meaningful RAM.

All randomness is drawn from one seeded random.Random instance threaded
through every call site (never the bare `random` module), so a run with a
fixed --seed is reproducible: same input files, same output database, byte
for byte, across repeated runs.

Schema is copied verbatim from brain/memory/store.py's _SCHEMA string plus
its idempotent migrations (read directly from that file, not retyped from
memory), so a real MemoryStore opens the result with zero migration diffs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sqlite3
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DEFAULT_SEED = 20260517
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARTS_DIR = os.path.join(HERE, "parts")
DEFAULT_OUT_DIR = HERE

START_DATE = datetime(2026, 5, 17, 0, 0, 0)
SPAN_DAYS = 109
NUM_WINDOWS = 16

MEMORY_TYPES = {
    "monologue", "fact", "monologue_soul_candidate", "observation",
    "monologue_trace", "monologue_emotion", "decision", "research", "note",
    "heartbeat", "feeling", "dream", "question", "making",
    "initiate_outbound", "event", "journal_entry", "file_write", "insight",
}

MONOLOGUE_TYPES = {
    "monologue", "monologue_soul_candidate", "monologue_trace", "monologue_emotion",
}
BRAIN_TYPES = {"fact", "observation", "decision", "note", "feeling", "dream", "question"}
US_TYPES = {"research", "heartbeat", "initiate_outbound"}
RESIDUAL_TYPES = {"making", "file_write", "event", "insight"}
RESIDUAL_DOMAINS = ["interior", "relationship", "lo_personal", "self", "intimacy", "coding"]

# Small, undocumented-exact-fraction minorities called out in SKELETON.md as
# "a few journal entries" and "a handful of facts" without a precise count.
# These fractions are an explicit, disclosed fill-in (see plan step 5 in
# changes/synthetic-db-tooling/2-plan.md), not a re-derivation of a real value.
JOURNAL_TO_BRAIN_FRACTION = 0.10
FACT_TO_RESIDUAL_FRACTION = 0.02

# Importance bucket weights, body 0-10 (SKELETON.md section 4). Used only as
# a fallback when a source row's own importance is missing or unusable.
IMPORTANCE_BUCKETS = [
    (0, 1, 3388), (1, 2, 1634), (2, 3, 331), (3, 4, 670), (4, 5, 1315),
    (5, 6, 2230), (6, 7, 3187), (7, 8, 3158), (8, 9, 2304), (9, 10, 1072),
]
IMPORTANCE_TAIL_FRACTION = 33.0 / 19452.0  # thin power-law-ish tail above 10

# Diurnal shape: a quiet overnight gap roughly 2am-6am (SKELETON.md section
# 5), modeled as near-zero rather than a hard zero. 4 quiet hours share 3% of
# the daily probability mass; the remaining 20 waking hours (6am-2am wrapped)
# share 97%, so the quiet band's per-hour density is roughly 6-7x lower than
# the waking band's, without ever forcing it to literally zero.
QUIET_HOUR_START = 2
QUIET_HOUR_END = 6  # exclusive
QUIET_MASS = 0.03

FILLER_WORDS = ["really", "very", "just", "actually", "honestly", "kind of", "sort of", "basically"]
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Schema copied verbatim from brain/memory/store.py's _SCHEMA (lines 234-410
# at the time this was read) plus the idempotent idx_memories_state index
# created separately in MemoryStore.__init__ (store.py:600). Read directly
# from that file this session, not retyped from a paraphrase, so the FTS5
# syntax and every column/constraint matches exactly.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    domain TEXT NOT NULL,
    emotions_json TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.0,
    score REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    protected INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL DEFAULT 'active',
    content_snapshot TEXT,
    recall_count REAL NOT NULL DEFAULT 0,
    peak_emotion_intensity REAL NOT NULL DEFAULT 0.0,
    embedding BLOB,
    embedding_model_id TEXT,
    cluster_id INTEGER,
    cluster_model_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state);

CREATE TABLE IF NOT EXISTS cluster_centroids (
    model_id TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,
    centroid BLOB NOT NULL,
    dim INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS calibration_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    day_bucket TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d','now')),
    query TEXT NOT NULL,
    candidate_ids TEXT NOT NULL,
    reranker_scores TEXT NOT NULL,
    reranker_model_id TEXT NOT NULL,
    local_judge_label TEXT,
    haiku_label TEXT,
    score_scale TEXT NOT NULL DEFAULT 'raw'
);
CREATE INDEX IF NOT EXISTS idx_calibration_log_day_bucket ON calibration_log(day_bucket);

CREATE TABLE IF NOT EXISTS reranker_floor_calibration (
    reranker_model_id TEXT PRIMARY KEY,
    floor REAL NOT NULL,
    raw_fit_floor REAL NOT NULL,
    sample_pairs INTEGER NOT NULL,
    is_cold_start INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    score_scale TEXT NOT NULL DEFAULT 'raw'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, content='memories', content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_vocab USING fts5vocab('memories_fts', 'row');

CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""


# ---------------------------------------------------------------------------
# window/day helpers
# ---------------------------------------------------------------------------

def window_day_range(window: int) -> tuple[int, int]:
    """Return (first_day, last_day), 1-indexed, for a given window (1-16).

    Window 16 is the short 4-day tail (days 106-109); windows 1-15 each span
    7 days.
    """
    if window == NUM_WINDOWS:
        return (106, 109)
    start = (window - 1) * 7 + 1
    return (start, start + 6)


def coerce_int(v):
    """Return v as a plain int if it is an int or an integral float, else
    None. Rejects bool explicitly (JSON has no bool-as-int convention here
    and Python's bool is a confusing int subclass)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


def normalize_window(w):
    w = coerce_int(w)
    if w is not None and 1 <= w <= NUM_WINDOWS:
        return w
    return None


def normalize_day(d):
    d = coerce_int(d)
    if d is not None and 1 <= d <= SPAN_DAYS:
        return d
    return None


def valid_window(w) -> bool:
    return normalize_window(w) is not None


def valid_day(d) -> bool:
    return normalize_day(d) is not None


def sample_hour(rng) -> float:
    """Sample a fractional hour in [0, 24) with a thin overnight quiet gap."""
    if rng.random() < QUIET_MASS:
        return QUIET_HOUR_START + rng.random() * (QUIET_HOUR_END - QUIET_HOUR_START)
    waking_span = 24 - (QUIET_HOUR_END - QUIET_HOUR_START)  # 20 hours
    u = rng.random() * waking_span
    if u < (24 - QUIET_HOUR_END):
        return QUIET_HOUR_END + u
    return u - (24 - QUIET_HOUR_END)


def make_created_at(day: int, rng) -> datetime:
    hour_f = sample_hour(rng)
    hh = int(hour_f)
    mm = rng.randint(0, 59)
    ss = rng.randint(0, 59)
    return START_DATE + timedelta(days=day - 1, hours=hh, minutes=mm, seconds=ss)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def parse_parts(parts_dir: str):
    """Read and validate every parts/*.jsonl line.

    Returns (rows, skip_counts) where rows is a list of validated dicts and
    skip_counts is {"malformed_json": n, "missing_required_field": n}.
    A row is skipped (never patched with an invented value) when it is
    genuinely unusable: bad JSON, or missing/wrong-typed content/memory_type,
    or an unrecognized memory_type, or neither a valid day nor a valid window
    to place it in time.
    """
    rows = []
    skip_counts = {"malformed_json": 0, "missing_required_field": 0}
    pattern = os.path.join(parts_dir, "*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    skip_counts["malformed_json"] += 1
                    continue
                if not isinstance(d, dict):
                    skip_counts["malformed_json"] += 1
                    continue

                content = d.get("content")
                memory_type = d.get("memory_type")
                if not isinstance(content, str) or not content.strip():
                    skip_counts["missing_required_field"] += 1
                    continue
                if not isinstance(memory_type, str) or memory_type not in MEMORY_TYPES:
                    skip_counts["missing_required_field"] += 1
                    continue

                window = d.get("window")
                day = d.get("day")
                if not valid_day(day) and not valid_window(window):
                    skip_counts["missing_required_field"] += 1
                    continue

                rows.append(d)
    return rows, skip_counts


def coerce_importance(value) -> float | None:
    """Return a usable float importance, or None if the source value is
    absent/unusable (caller falls back to bucket sampling)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f < 0:
        f = 0.0
    return f


def coerce_emotions(value) -> dict:
    if not isinstance(value, dict):
        return {}
    out = {}
    for k, v in value.items():
        if not isinstance(k, str) or not k.strip():
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def coerce_tags(value) -> list:
    if not isinstance(value, list):
        return []
    return [t for t in value if isinstance(t, str) and t.strip()]


def sample_importance(rng) -> float:
    if rng.random() < IMPORTANCE_TAIL_FRACTION:
        # thin tail above 10, roughly power-law-ish toward the low end of it
        return 10.0 + rng.expovariate(1.0 / 6.0)
    total = sum(w for _, _, w in IMPORTANCE_BUCKETS)
    r = rng.uniform(0, total)
    acc = 0.0
    for lo, hi, w in IMPORTANCE_BUCKETS:
        acc += w
        if r <= acc:
            return rng.uniform(lo, hi)
    lo, hi, _ = IMPORTANCE_BUCKETS[-1]
    return rng.uniform(lo, hi)


# ---------------------------------------------------------------------------
# weighted sampling without replacement (Efraimidis-Spirakis), seeded
# ---------------------------------------------------------------------------

def weighted_priority_order(indices, weights, rng):
    """Return `indices` permuted into weighted-random priority order
    (highest-weight-biased item first), via the Efraimidis-Spirakis A-ES
    key. Consumes exactly len(indices) rng.random() draws -- the same as
    weighted_sample_without_replacement(indices, weights, k, rng) for any
    k, so slicing this list's first k is identical to that call."""
    keyed = []
    for i, w in zip(indices, weights):
        w = max(w, 1e-12)
        u = rng.random()
        key = u ** (1.0 / w)
        keyed.append((key, i))
    keyed.sort(reverse=True)
    return [i for _, i in keyed]


def weighted_sample_without_replacement(indices, weights, k, rng):
    return weighted_priority_order(indices, weights, rng)[:k]


# ---------------------------------------------------------------------------
# domain assignment
# ---------------------------------------------------------------------------

def assign_domains(rows, rng):
    n = len(rows)
    fact_indices = [i for i, d in enumerate(rows) if d["memory_type"] == "fact"]
    n_residual_fact = round(len(fact_indices) * FACT_TO_RESIDUAL_FRACTION)
    residual_fact_idx = set(rng.sample(fact_indices, n_residual_fact)) if n_residual_fact else set()

    domains = [None] * n
    for i, d in enumerate(rows):
        mtype = d["memory_type"]
        if mtype in MONOLOGUE_TYPES:
            domains[i] = "monologue"
        elif mtype in BRAIN_TYPES:
            if mtype == "fact" and i in residual_fact_idx:
                domains[i] = rng.choice(RESIDUAL_DOMAINS)
            else:
                domains[i] = "brain"
        elif mtype in US_TYPES:
            domains[i] = "us"
        elif mtype == "journal_entry":
            domains[i] = "brain" if rng.random() < JOURNAL_TO_BRAIN_FRACTION else "us"
        elif mtype in RESIDUAL_TYPES:
            domains[i] = rng.choice(RESIDUAL_DOMAINS)
        else:
            domains[i] = "us"
    return domains


# ---------------------------------------------------------------------------
# fading-row content decay
#
# Real MemoryStore.fade() (companion-emergence brain/memory/store.py, the
# UPDATE at ~line 1549) does:
#     content_snapshot = content   (the pre-fade, full original text)
#     content           = summary  (a shorter decayed version)
#     state             = 'fading'
# i.e. content_snapshot holds the ORIGINAL and content holds the SHORTER
# decayed text. This function produces that shorter `content` replacement;
# the caller assigns the untouched writer text straight to content_snapshot.
# ---------------------------------------------------------------------------

# Coordinating conjunctions ("FANBOYS") -- the ONLY conjunctions a
# single-sentence clause cut is allowed to land before. Subordinators like
# "because"/"while"/"that"/"which" are deliberately excluded: cutting before
# them is exactly how the old implementation produced dangling fragments
# ("...were allowed to both.", "...an old.").
COORD_CONJUNCTIONS = ("for", "and", "nor", "but", "or", "yet", "so")

# Function words a decayed snippet must never end on: determiners,
# prepositions, conjunctions, possessives, auxiliaries, and a few other
# closers that can't stand alone as a clause end. Superset of (and must stay
# in sync with) check_db.py's DANGLING_STOPWORDS, which encodes the exact
# list from the fix spec.
DANGLING_WORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "with",
    "and", "but", "or", "his", "her", "my", "their", "its",
    "was", "were", "is", "be", "been", "how", "that", "which", "who",
    "if", "as", "from", "by", "about", "into", "than", "so",
    # defense-in-depth extras beyond check_db's exact list
    "nor", "yet", "he", "she", "they", "him", "them", "our", "your",
    "do", "does", "did", "has", "have", "had", "will", "would",
    "can", "could", "should", "must", "might", "may", "not", "no",
    "when", "while", "because",
}

MIN_DECAY_WORDS = 5

# Titles/abbreviations whose period is NOT a sentence end -- without this,
# SENT_SPLIT_RE's bare "(?<=[.!?])\s+" treats "Mr." as a full sentence and a
# multi-sentence cut can land right after it, severing a title from the name
# it belongs to (e.g. "...fixing the porch step for Mr." with "Aldana" cut
# off). A single capital letter followed by a period (an initial, "J.") is
# treated the same way.
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc",
    "rev", "capt", "gen", "sgt", "col", "lt", "ave", "blvd", "inc",
    "corp", "co", "no", "vol", "ch", "mt", "ft", "approx", "dept", "fig",
}


def _split_sentences(text: str):
    """Sentence-split `text`, then re-merge any split that landed right
    after a title/initial abbreviation (see ABBREVIATIONS) rather than a
    real sentence end."""
    raw_parts = [s for s in SENT_SPLIT_RE.split(text) if s]
    sentences = []
    for part in raw_parts:
        if sentences:
            prev = sentences[-1]
            prev_words = prev.rstrip(".!?").split()
            prev_stem = prev_words[-1].lower() if prev_words else ""
            if prev.rstrip().endswith(".") and (
                prev_stem in ABBREVIATIONS or len(prev_stem) == 1
            ):
                sentences[-1] = prev.rstrip() + " " + part
                continue
        sentences.append(part)
    return sentences


def _clean_word(w: str) -> str:
    return w.strip(".,;:!?\"'()[]").lower()


MIN_TRAILING_CLAUSE_WORDS = 4


def _final_clause_ok(text: str) -> bool:
    """Reject a kept-text whose FINAL clause (the text after its last
    internal comma/semicolon/colon, or the whole text if it has none) opens
    with a coordinating conjunction but is too short to be its own clause --
    e.g. "...anymore, and tonight." (VOICE-AUDIT-FINAL.md-style dangle: "and
    tonight" sets up a clause that never arrives). The overall
    >=MIN_DECAY_WORDS check alone misses this because it counts words across
    the WHOLE snippet, not just the clause the cut actually lands in."""
    last_boundary = None
    for m in re.finditer(r"[,;:]", text):
        last_boundary = m
    tail = text[last_boundary.end():].strip() if last_boundary else text
    tail_words = tail.split()
    if tail_words and _clean_word(tail_words[0]) in COORD_CONJUNCTIONS:
        return len(tail_words) >= MIN_TRAILING_CLAUSE_WORDS
    return True


def _is_coherent_snippet(text: str) -> bool:
    """A candidate kept-text is coherent iff it has enough words, does not
    end on a dangling function word (determiner/preposition/conjunction/
    possessive/auxiliary/etc.), and its final clause isn't a coordinating
    conjunction left dangling with nothing after it (see _final_clause_ok)."""
    words = text.split()
    if len(words) < MIN_DECAY_WORDS:
        return False
    last = _clean_word(words[-1])
    if not last or last in DANGLING_WORDS:
        return False
    return _final_clause_ok(text)


def _finalize_snippet(kept: str) -> str:
    snippet = kept.rstrip(" \t,;:-")
    if snippet and not snippet.endswith((".", "!", "?")):
        snippet += "."
    return snippet


def _decay_candidates(content: str):
    """Return every clean-cut (length, kept_text) candidate for decaying
    `content`, or [] if none exist. `kept_text` is unfinalized (no trailing
    period yet) but already passes _is_coherent_snippet.

    Multi-sentence content is cut ONLY at sentence boundaries: a leading run
    of whole sentences. Single-sentence content is cut ONLY at a real clause
    boundary: after a comma/semicolon/colon, or immediately before a
    coordinating conjunction. There is no word/char-ratio fallback -- if no
    such boundary exists, the content has no candidates and is not eligible
    to fade (see can_decay)."""
    original = content.strip()
    sentences = _split_sentences(original)
    candidates = []

    if len(sentences) > 1:
        for k in range(1, len(sentences)):
            kept = " ".join(sentences[:k]).strip()
            if kept and kept != original and _is_coherent_snippet(kept):
                candidates.append((len(kept), kept))
        return candidates

    for m in re.finditer(r"[,;:]", original):
        kept = original[:m.start()].rstrip(" \t,;:-")
        if kept and kept != original and _is_coherent_snippet(kept):
            candidates.append((len(kept), kept))

    conj_pattern = re.compile(
        r"\s+(" + "|".join(COORD_CONJUNCTIONS) + r")\s+(\S+)", re.IGNORECASE
    )
    for m in conj_pattern.finditer(original):
        kept = original[:m.start()].rstrip(" \t,;:-")
        if not kept or kept == original or not _is_coherent_snippet(kept):
            continue
        # Reject a reduplication idiom split ("over and over", "again and
        # again", "round and round"): the conjunction's cut point looks
        # like a clean clause boundary, but the word right after it is the
        # SAME word the kept text already ends on, so the cut severs a
        # single repeated-word phrase rather than joining two clauses.
        last_kept_word = _clean_word(kept.split()[-1]) if kept.split() else ""
        next_word = _clean_word(m.group(2))
        if next_word and next_word == last_kept_word:
            continue
        candidates.append((len(kept), kept))

    return candidates


def can_decay(content: str) -> bool:
    """True iff make_decayed_content() can produce a coherent, strictly-
    shorter decayed version of `content` -- i.e. _decay_candidates() finds
    at least one real clause/sentence boundary to cut at. This is the sole
    gate on fading-eligibility (see build()'s eligible_fading_idx): a row
    with no clean cut is simply never chosen to fade, rather than being
    fading and getting a broken decayed content."""
    return bool(_decay_candidates(content))


def make_decayed_content(content: str, rng, avoid=None):
    """Return a strictly-shorter, cleanly-ending decayed version of content
    for a row transitioning to 'fading' state, or None if no coherent cut
    exists (callers must gate with can_decay() first -- see build()).

    Only ever cuts at a real clause boundary (see _decay_candidates): a
    leading run of whole sentences for multi-sentence content, or a
    comma/semicolon/colon/pre-conjunction cut for single-sentence content.
    The kept text is always >=MIN_DECAY_WORDS words and never ends on a
    dangling function word -- never a blind word- or char-count ratio, which
    is what previously produced fragments like "...an old." or "...were
    allowed to both.".

    Deterministic given rng: draws a target length ratio and picks the
    candidate closest to it, falling through the remaining candidates (in
    closeness order) to skip any whose finalized text is in `avoid` (used by
    build()'s uniqueness-retry pass). Returns None only if every candidate's
    finalized text is already in `avoid`."""
    original = content.strip()
    candidates = _decay_candidates(original)
    if not candidates:
        return None
    avoid = avoid or set()

    target_ratio = rng.uniform(0.45, 0.7)
    target_len = len(original) * target_ratio
    ordered = sorted(candidates, key=lambda c: abs(c[0] - target_len))

    for _, kept in ordered:
        snippet = _finalize_snippet(kept)
        if snippet and snippet != original and len(snippet) < len(original) and snippet not in avoid:
            return snippet
    return None


# ---------------------------------------------------------------------------
# tags_json derivation (defect 2: writers over-supply topical tags on too
# many rows; SKELETON.md section 4 wants ~40% of rows tagged at all, and a
# specific tag-count shape among the tagged rows -- see build_tags below).
# ---------------------------------------------------------------------------

TAG_FRACTION = 0.40  # fraction of ALL rows that get any tags at all
SOURCE_MARKERS = ["conversation", "auto-ingested"]
# Generic topical filler drawn only from SKELETON.md section 1's leak-free
# domain vocabulary, used to pad out a row's topical tags when the writer
# didn't supply enough of its own.
TOPICAL_POOL = [
    "bond", "closeness", "family", "loss", "grief", "mental-health",
    "medication", "rest", "animals", "craft", "repair", "household",
    "routine", "music", "science", "research", "identity", "memory",
    "relationship", "intimacy",
]


def build_tags(writer_topical: list, mtype: str, rng) -> list:
    """Return this row's tags_json array, deterministic given rng.

    ~TAG_FRACTION of rows get no tags at all. Of the tagged rows: ~78%
    get exactly 3 tags (memory_type + a source marker + 1 topical tag,
    matching the "mode is 3" target), ~15% get exactly 2 (memory_type +
    source marker, no topical), ~5% get a 4-8 tag tail, and ~2% get a rare
    9-13 tag tail -- topical tags beyond what the writer supplied are
    padded from TOPICAL_POOL.
    """
    if rng.random() >= TAG_FRACTION:
        return []

    source = rng.choice(SOURCE_MARKERS)
    r = rng.random()
    if r < 0.78:
        target_topical = 1
    elif r < 0.93:
        target_topical = 0
    elif r < 0.98:
        target_topical = rng.randint(2, 6)  # total tag count 4-8
    else:
        target_topical = rng.randint(7, 11)  # total tag count 9-13

    available = list(writer_topical)
    rng.shuffle(available)
    topical = []
    for t in available:
        if len(topical) >= target_topical:
            break
        if t not in topical and t != mtype and t not in SOURCE_MARKERS:
            topical.append(t)
    while len(topical) < target_topical:
        candidate = rng.choice(TOPICAL_POOL)
        if candidate not in topical and candidate != mtype:
            topical.append(candidate)

    return [mtype, source] + topical


# ---------------------------------------------------------------------------
# metadata_json generation
# ---------------------------------------------------------------------------

def make_metadata(importance: float, tags: list, rng) -> dict:
    meta = {
        "source": rng.choice(["auto-ingested", "conversation"]),
        "importance_echo": importance,
        "tags_echo": list(tags),
        "seed_id": "%08x" % rng.getrandbits(32),
        "provider": "synthetic-gen-v1",
        "schema_version": 1,
    }
    flavor = rng.random()
    if flavor < 0.25:
        meta["decay_bookkeeping"] = {
            "elapsed_seconds": rng.randint(60, 500000),
            "memories_decayed": rng.randint(0, 40),
            "edges_pruned": rng.randint(0, 20),
        }
    elif flavor < 0.40:
        meta["dream_id"] = "%08x" % rng.getrandbits(32)
    elif flavor < 0.55:
        meta["interest_topic"] = rng.choice(
            ["music", "science", "craft", "family", "animals", "routine"]
        )
        meta["interest_scope"] = rng.choice(["narrow", "broad"])
    elif flavor < 0.70:
        meta["outbound_audit"] = {
            "subject": rng.choice(["check-in", "follow-up", "reminder"]),
            "state": rng.choice(["sent", "queued", "skipped"]),
            "timestamp": None,
        }
    elif flavor < 0.85:
        meta["web_usage"] = {
            "used": True,
            "result_count": rng.randint(1, 8),
            "urls": ["synthetic://example/%d" % rng.randint(1, 9999)],
        }
    elif flavor < 0.92:
        meta["correction_of"] = "%08x" % rng.getrandbits(32)
    return meta


# ---------------------------------------------------------------------------
# main assembly
# ---------------------------------------------------------------------------

def build(parts_dir: str, out_dir: str, seed: int):
    rng = random.Random(seed)

    rows, skip_counts = parse_parts(parts_dir)
    n_before_dedup = len(rows)

    seen_content = set()
    deduped_rows = []
    dedup_dropped = 0
    for d in rows:
        c = d["content"]
        if c in seen_content:
            dedup_dropped += 1
            continue
        seen_content.add(c)
        deduped_rows.append(d)
    rows = deduped_rows
    n = len(rows)

    if n == 0:
        raise SystemExit(
            "No usable rows parsed from %s (skipped: %s). Nothing to build."
            % (parts_dir, skip_counts)
        )

    domains = assign_domains(rows, rng)

    days = [None] * n
    for i, d in enumerate(rows):
        day = normalize_day(d.get("day"))
        window = normalize_window(d.get("window"))
        if day is not None:
            days[i] = day
        else:
            # parse_parts already guarantees at least one of day/window is
            # valid, so window is not None here.
            lo, hi = window_day_range(window)
            days[i] = rng.randint(lo, hi)

    created_at_dt = [make_created_at(days[i], rng) for i in range(n)]
    created_at_str = [dt.strftime("%Y-%m-%dT%H:%M:%SZ") for dt in created_at_dt]

    order = sorted(range(n), key=lambda i: created_at_dt[i])
    rank_of = [0] * n
    for rank, i in enumerate(order):
        rank_of[i] = rank
    age_norm = [1.0 - (rank_of[i] / max(1, n - 1)) for i in range(n)]  # 1.0 = oldest

    importance = []
    for i, d in enumerate(rows):
        v = coerce_importance(d.get("importance"))
        importance.append(v if v is not None else sample_importance(rng))
    imp_max = max(max(importance), 1.0)
    importance_norm = [v / imp_max for v in importance]

    # --- state / fading / content_snapshot ---
    # A row is only eligible to fade if make_decayed_content() can actually
    # produce a COHERENT, strictly-shorter result for it -- can_decay() (see
    # module scope above) gates on a real clause/sentence boundary existing,
    # not merely on length. Restricting the CANDIDATE POOL itself (rather
    # than just downweighting) guarantees every fading row gets readable
    # decayed content by construction; ~94% of real writer content has such
    # a boundary, so this leaves the fading fraction close to unaffected.
    #
    # A second, independent way a row can fail to fade cleanly: some writer
    # content shares a templated leading clause across many rows (e.g. many
    # rows opening "tenderness thick enough to feel almost physical, ..."
    # before diverging), so two different rows' ONLY clean cut can collide
    # on the exact same decayed text. Uniqueness is therefore resolved at
    # SELECTION time, not after the fact: walk the eligible pool in its
    # weighted-random priority order, and only actually commit a row to
    # fading if make_decayed_content() can find a cut not already taken;
    # a row that can't is simply left active/inactive (per the fix spec:
    # "not eligible for fading... choose other eligible rows"), and the next
    # candidate in priority order is tried so the fading count stays
    # proportional.
    fading_ratio = 6669.0 / 19452.0
    eligible_fading_idx = [i for i in range(n) if can_decay(rows[i]["content"])]
    fading_target = min(round(n * fading_ratio), len(eligible_fading_idx))
    fading_weight = [
        (0.35 + 0.65 * age_norm[i]) * (0.35 + 0.65 * (1.0 - importance_norm[i]))
        * rng.uniform(0.5, 1.5)
        for i in eligible_fading_idx
    ]
    fading_priority = weighted_priority_order(eligible_fading_idx, fading_weight, rng)

    inactive_ratio = 7.0 / 19452.0
    inactive_count = min(n, round(n * inactive_ratio))
    inactive_idx = set(rng.sample(range(n), inactive_count)) if inactive_count else set()

    # Seed with every row's ORIGINAL content: originals are already
    # pairwise-unique (the dedup pass above), and active/inactive rows keep
    # their original untouched, so this is exactly the set a decayed
    # candidate must avoid colliding with, before any decays are chosen.
    seen_final_content = set(d["content"] for d in rows)

    fading_idx = set()
    decayed_content_for = {}
    for i in fading_priority:
        if len(fading_idx) >= fading_target:
            break
        if i in inactive_idx:
            continue  # inactive takes precedence over fading, as before
        decayed = make_decayed_content(rows[i]["content"], rng, avoid=seen_final_content)
        if decayed is None:
            continue  # every clean cut of this row collides; skip it, stays active
        fading_idx.add(i)
        decayed_content_for[i] = decayed
        seen_final_content.add(decayed)

    states = []
    content_snapshots = []
    contents = []
    active_flags = []
    for i, d in enumerate(rows):
        if i in inactive_idx:
            states.append("inactive")
            active_flags.append(0)
            content_snapshots.append(None)
            contents.append(d["content"])
        elif i in fading_idx:
            states.append("fading")
            active_flags.append(1)
            # content_snapshot = the writer's full original text (matches
            # MemoryStore.fade()'s content_snapshot = content assignment);
            # content = a shorter, deterministic decayed version.
            content_snapshots.append(d["content"])
            contents.append(decayed_content_for[i])
        else:
            states.append("active")
            active_flags.append(1)
            content_snapshots.append(None)
            contents.append(d["content"])

    # --- recall_count ---
    MU_RC, SIGMA_RC = 4.3, 1.75
    recall_count = []
    for i in range(n):
        base = rng.lognormvariate(MU_RC, SIGMA_RC)
        bias = 0.6 + 0.9 * age_norm[i] + 0.5 * importance_norm[i]
        val = min(max(base * bias, 0.0), 2045.0)
        recall_count.append(val)

    # --- score ---
    MU_F, SIGMA_F = 0.93, 0.6
    score = []
    for i in range(n):
        factor = rng.lognormvariate(MU_F, SIGMA_F)
        if rng.random() < 0.006:
            factor *= rng.uniform(5, 20)
        score.append(importance[i] * factor)

    # --- emotions_json / tags_json ---
    # SKELETON.md section 4: ~40% of rows tagged at all; of tagged rows, most
    # carry exactly 3 tags (memory_type + a source marker + 1 topical tag),
    # small tails at 2 (memory_type + source marker only) and at 4-8
    # (more topical tags), rare up to ~13. Store-wide mean ~1.2 tags/row.
    emotions_final = []
    tags_final = []
    for i, d in enumerate(rows):
        emotions = coerce_emotions(d.get("emotions"))
        writer_topical = coerce_tags(d.get("tags"))
        mtype = d["memory_type"]
        tags = build_tags(writer_topical, mtype, rng)
        emotions_final.append(emotions)
        tags_final.append(tags)

    # --- peak_emotion_intensity ---
    # Zero-inflated Gamma (defect 3 retune). The zero/nonzero coin flip is
    # decoupled from whether the row has emotions_json populated and uses a
    # single constant probability -- an earlier tuning that conditioned
    # nonzero-probability on emotions_final[i] made the median flip sharply
    # around the ~29% emotions-population point (see the sampling sweep in
    # the PR/report), which is too fragile. Here only the MAGNITUDE (via the
    # scale parameter) varies with emotions_final[i], while the zero rate
    # and shape stay fixed, so mean ~2.5-4.0, median ~1.0-2.0 and
    # nonzero-fraction ~55% hold robustly across a wide range of
    # emotions-population fractions (checked 0.15-0.5 in the sweep).
    PEAK_EMOTION_SHAPE = 2.5
    PEAK_EMOTION_ZERO_PROB = 0.45
    PEAK_EMOTION_SCALE_HAS_EMOTIONS = 2.8
    PEAK_EMOTION_SCALE_NO_EMOTIONS = 1.8
    PEAK_EMOTION_CAP = 41.0
    peak_emotion = []
    for i in range(n):
        if rng.random() < PEAK_EMOTION_ZERO_PROB:
            v = 0.0
        else:
            scale = (
                PEAK_EMOTION_SCALE_HAS_EMOTIONS if emotions_final[i]
                else PEAK_EMOTION_SCALE_NO_EMOTIONS
            )
            v = rng.gammavariate(PEAK_EMOTION_SHAPE, scale)
        peak_emotion.append(min(max(v, 0.0), PEAK_EMOTION_CAP))

    # --- metadata_json ---
    META_FRACTION = 0.47
    meta_count = round(n * META_FRACTION)
    meta_idx = set(rng.sample(range(n), meta_count)) if meta_count else set()
    metadata_final = []
    for i in range(n):
        if i in meta_idx:
            metadata_final.append(json.dumps(make_metadata(importance[i], tags_final[i], rng)))
        else:
            metadata_final.append("{}")

    # --- last_accessed_at ---
    LAST_ACCESSED_NULL_FRACTION = 0.03
    null_count = round(n * LAST_ACCESSED_NULL_FRACTION)
    null_la_idx = set(rng.sample(range(n), null_count)) if null_count else set()
    last_accessed_str = []
    for i in range(n):
        if i in null_la_idx:
            last_accessed_str.append(None)
        else:
            delta_hours = min(rng.expovariate(1.0 / 20.0), 500.0)
            dt = created_at_dt[i] + timedelta(hours=delta_hours)
            last_accessed_str.append(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))

    # --- protected (small importance-biased minority; not in NOT-NULL list
    #     of columns the task requires deriving, but the column defaults to 0
    #     which is a valid NOT NULL value on its own) ---
    PROTECTED_FRACTION = 0.03
    protected_count = round(n * PROTECTED_FRACTION)
    prot_weight = [(0.2 + 0.8 * importance_norm[i]) * rng.uniform(0.6, 1.4) for i in range(n)]
    protected_idx = set(weighted_sample_without_replacement(list(range(n)), prot_weight, protected_count, rng))
    protected_flags = [1 if i in protected_idx else 0 for i in range(n)]

    # --- ids: derived from the seeded rng, never uuid.uuid4() (which draws
    #     from os.urandom and would break run-to-run reproducibility) ---
    ids = ["%032x" % rng.getrandbits(128) for _ in range(n)]

    # --- write DB ---
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, "memories.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)

    insert_rows = []
    for i, d in enumerate(rows):
        insert_rows.append((
            ids[i],
            contents[i],
            d["memory_type"],
            domains[i],
            json.dumps(emotions_final[i]),
            json.dumps(tags_final[i]),
            float(importance[i]),
            float(score[i]),
            created_at_str[i],
            last_accessed_str[i],
            active_flags[i],
            protected_flags[i],
            metadata_final[i],
            states[i],
            content_snapshots[i],
            float(recall_count[i]),
            float(peak_emotion[i]),
        ))

    conn.executemany(
        """
        INSERT INTO memories (
            id, content, memory_type, domain, emotions_json, tags_json,
            importance, score, created_at, last_accessed_at, active,
            protected, metadata_json, state, content_snapshot, recall_count,
            peak_emotion_intensity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        insert_rows,
    )
    conn.commit()

    # --- manifest ---
    type_counts = {}
    window_counts = {}
    day_counts = {}
    for i, d in enumerate(rows):
        mtype = d["memory_type"]
        type_counts[mtype] = type_counts.get(mtype, 0) + 1
        window = normalize_window(d.get("window"))
        if window is not None:
            window_counts[window] = window_counts.get(window, 0) + 1
        day_counts[days[i]] = day_counts.get(days[i], 0) + 1

    day_values = list(day_counts.values())
    manifest = {
        "seed": seed,
        "parts_dir": parts_dir,
        "out_dir": out_dir,
        "total_rows": n,
        "rows_before_dedup": n_before_dedup,
        "dedup_dropped": dedup_dropped,
        "skipped_lines": skip_counts,
        "per_memory_type_counts": type_counts,
        "per_window_counts": window_counts,
        "per_day_stats": {
            "min": min(day_values) if day_values else 0,
            "mean": (sum(day_values) / len(day_values)) if day_values else 0.0,
            "max": max(day_values) if day_values else 0,
        },
        "active_count": sum(active_flags),
        "fading_count": states.count("fading"),
        "inactive_count": states.count("inactive"),
        "day_span": {
            "min": min(days) if days else None,
            "max": max(days) if days else None,
        },
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    conn.close()
    return db_path, manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(
        description="Build a synthetic companion-memory memories.db from parts/*.jsonl."
    )
    parser.add_argument("--parts-dir", default=DEFAULT_PARTS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Fixed seed for reproducible generation (default: %(default)s).",
    )
    args = parser.parse_args()

    db_path, manifest_path, manifest = build(args.parts_dir, args.out_dir, args.seed)

    print("Built %s" % db_path)
    print("Manifest %s" % manifest_path)
    print("Total rows: %d (before dedup: %d, dropped: %d)" % (
        manifest["total_rows"], manifest["rows_before_dedup"], manifest["dedup_dropped"]
    ))
    print("Skipped lines: %s" % manifest["skipped_lines"])
    print("Active/fading/inactive: %d/%d/%d" % (
        manifest["active_count"], manifest["fading_count"], manifest["inactive_count"]
    ))
    print("Day span: %s" % manifest["day_span"])


if __name__ == "__main__":
    main()
