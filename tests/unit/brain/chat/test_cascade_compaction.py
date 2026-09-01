"""Conformance tests for cascade compaction (Phase 1, 3-tier age-stratified fold).

Maps to changes/cascade-compaction/1.5-criteria.md:
  C1   three-section representation (one row)
  C2   cascade wiring: age-gated, oldest-first, correct sources
  C3   tier hard caps bound the head (tier1 + terminal tier3)
  C4   fold-output validation (#77) + cascade double-reject fallback
  C5   temporal markers (#82) + owner age-band labels
  C6   cache-stable head prefix (byte-stable + re-parseable)
  C7   lossless-before-lossy + cursor guard + idempotent
  C13  interior-continuity read not starved
  C14  graduation + terminal tier-3 persistence + multi-input
  C17  cascade write is atomic across the three tiers
  C22  apply_budget backstop operates correctly on the sectioned row

Also maps to changes/compaction-defects-fix/1.5-criteria.md:
  C1a/C1b  the new CONDENSE branch routes (prior + empty raw) away from the
           mis-wired _FOLD_PROMPT(transcript='') and re-compacts toward the
           tier fraction (Bug 1 / D1)
  C1d      durable real-model (haiku) safety net for _CONDENSE_PROMPT
  C-B3/C-B3b  cascade-path archive-append is idempotent under a crash-retry,
           losslessly, even across byte-identical duplicate raw turns (Bug 3)
  C-B4a/b/c/f/g  tier-3 replacement/eviction, idempotence-preserved, the
           emergency_fold_24h backstop also evicts, and a degenerate
           all-evicted pass installs cleanly (Bug 4)
  C-B4d    stale retain-forever docstrings replaced with the replacement model
  C14 (updated)  the two pre-existing gating tests below are updated to the
           owner's replacement model (retired "terminal, re-compacted forever")
"""

from __future__ import annotations

import json
import re as _re
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.chat import ChatMessage
from brain.chat.budget import _COMPACTION_SUMMARY_PREFIX, apply_budget
from brain.chat.compaction import (
    _CONDENSE_PROMPT,
    _FRACTION_48H,
    _FRACTION_72H,
    _MIN_TARGET_WORDS,
    _SECTION_24H_CHAR_CAP,
    _SECTION_72H_CHAR_CAP,
    _fold_into_section,
    _generate_validated_fold,
    _read_sections,
    _render_sections,
    _validate_fold_output,
    _word_count,
    cascade_conversation,
    emergency_fold_24h,
)
from brain.chat.engine import _buffer_turns_to_messages
from brain.ingest.buffer import ingest_turn, read_archive, read_session, write_cursor
from brain.memory.store import MemoryStore
from brain.monologue.ambient import build_interior_continuity_block
from brain.monologue.trace import write_trace_memory

# --------------------------------------------------------------------------- helpers


class _Stub:
    """Deterministic FIRST-PERSON provider stub. The validator (#77) rejects
    non-first-person output, so any test needing a validated (non-fallback)
    fold must use first-person text (see module docstring in compaction.py)."""

    def __init__(self, resp: str = "I remember our conversation clearly."):
        self.resp = resp
        self.calls: list[str] = []

    def generate(self, *, prompt: str, system: str | None = None, **kw) -> str:
        self.calls.append(prompt)
        return self.resp


class _MarkerProvider:
    """Marker-preserving provider (C14): extracts every ``MARK\\w+`` token from
    the prompt (dedup, first-occurrence order) and echoes them back in a
    first-person sentence — so a cascade's real fold path can be driven while
    still letting the test assert on ACTUAL marker content, non-tautologically."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, *, prompt: str, system: str | None = None, **kw) -> str:
        self.calls.append(prompt)
        marks: list[str] = []
        for m in _re.findall(r"MARK\w+", prompt):
            if m not in marks:
                marks.append(m)
        if not marks:
            return "I recall nothing new right now."
        return "I recall " + " ".join(marks) + "."


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _seed_turn(persona_dir: Path, sid: str, ts: datetime, speaker: str, text: str) -> None:
    ingest_turn(persona_dir, {"session_id": sid, "speaker": speaker, "text": text, "ts": _iso(ts)})


def _summary_row(persona_dir: Path, sid: str) -> dict:
    turns = read_session(persona_dir, sid)
    rows = [t for t in turns if t.get("speaker") == "summary"]
    assert len(rows) == 1, f"expected exactly one summary row, got {len(rows)}"
    return rows[0]


def _seed_three_bands(persona_dir: Path, sid: str, now: datetime) -> None:
    """One turn each in G24 (30h), G48 (60h), G72 (80h)."""
    _seed_turn(persona_dir, sid, now - timedelta(hours=30), "user", "hello band24")
    _seed_turn(persona_dir, sid, now - timedelta(hours=60), "user", "hello band48")
    _seed_turn(persona_dir, sid, now - timedelta(hours=80), "user", "hello band72")
    write_cursor(persona_dir, sid, _iso(now))


def _write_sectioned_summary(
    persona_dir: Path, sid: str, sections: dict, *, gen: int = 5
) -> None:
    """Write a session buffer holding ONE already-sectioned summary row, with
    caller-controlled per-section ``covers_from_ts`` ages — lets a test place a
    section directly into a specific age band (or past the eviction boundary)
    without replaying multiple cascade passes to get it there."""
    ac = persona_dir / "active_conversations"
    ac.mkdir(parents=True, exist_ok=True)
    now_iso = _iso(datetime.now(UTC))
    row = {
        "session_id": sid,
        "speaker": "summary",
        "text": _render_sections(sections),
        "ts": now_iso,
        "compaction": {
            "gen": gen, "folded": True, "covers_until_ts": now_iso,
            "sections": sections,
        },
    }
    with (ac / f"{sid}.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------- C1


def test_c1_three_sections(tmp_path: Path) -> None:
    sid = "sess_c1"
    now = datetime.now(UTC)
    _seed_three_bands(tmp_path, sid, now)

    provider = _Stub()
    result = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert result.compacted is True

    row = _summary_row(tmp_path, sid)  # asserts single-row invariant too
    sections = row["compaction"]["sections"]
    assert set(sections.keys()) == {"24h", "48h", "72h"}

    # H6: oracle demonstrably discriminates a pre-change-style flat (single-layer)
    # row — the tolerant reader maps it to ONLY tier 3, not all three sections.
    flat_row = {"text": "an old single-layer summary", "compaction": {"covers_until_ts": _iso(now)}}
    flat_sections = _read_sections(flat_row, now)
    assert set(flat_sections.keys()) != {"24h", "48h", "72h"}
    assert set(flat_sections.keys()) == {"72h"}


# --------------------------------------------------------------------------- C2


def test_c2_age_gated_wiring(tmp_path: Path) -> None:
    sid = "sess_c2"
    now = datetime.now(UTC)
    # Distinct counts per band so raw_group_size is discriminating.
    _seed_turn(tmp_path, sid, now - timedelta(hours=30), "user", "a")
    _seed_turn(tmp_path, sid, now - timedelta(hours=31), "user", "b")
    _seed_turn(tmp_path, sid, now - timedelta(hours=32), "user", "c")
    _seed_turn(tmp_path, sid, now - timedelta(hours=60), "user", "d")
    _seed_turn(tmp_path, sid, now - timedelta(hours=61), "user", "e")
    _seed_turn(tmp_path, sid, now - timedelta(hours=80), "user", "f")
    write_cursor(tmp_path, sid, _iso(now))

    provider = _Stub()
    result = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert result.compacted is True

    assert list(result.tiers.keys()) == ["72h", "48h", "24h"]
    assert result.tiers["72h"].raw_group_size == 1
    assert result.tiers["48h"].raw_group_size == 2
    assert result.tiers["24h"].raw_group_size == 3

    # H6: the order assertion discriminates — a youngest-first order would not
    # match the real (oldest-first) wiring.
    assert list(result.tiers.keys()) != ["24h", "48h", "72h"]


# --------------------------------------------------------------------------- C3


def test_c3_tier1_hard_cap(tmp_path: Path) -> None:
    sid = "sess_c3a"
    now = datetime.now(UTC)
    for i in range(3):
        _seed_turn(tmp_path, sid, now - timedelta(hours=30, minutes=i), "user", f"content turn {i}")
    write_cursor(tmp_path, sid, _iso(now))

    huge = "I recall " + "x " * 10000  # well over the 12_000-char cap, first-person
    provider = _Stub(huge)
    result = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert result.compacted is True

    row = _summary_row(tmp_path, sid)
    sec24_text = row["compaction"]["sections"]["24h"]["text"]
    assert len(sec24_text) <= _SECTION_24H_CHAR_CAP

    # H6: confirm the truncation mechanism actually fired — the pre-truncation
    # provider output exceeded the cap.
    assert len(huge) > _SECTION_24H_CHAR_CAP


def test_c3_terminal_tier3_hard_cap_multi_cycle(tmp_path: Path) -> None:
    sid = "sess_c3b"
    now0 = datetime.now(UTC)
    huge = "I recall " + "x " * 10000
    provider = _Stub(huge)

    saw_tier3 = False
    for day in range(6):
        cycle_now = now0 + timedelta(days=day)
        _seed_turn(tmp_path, sid, cycle_now - timedelta(hours=30), "user", f"day{day} content")
        write_cursor(tmp_path, sid, _iso(cycle_now))
        cascade_conversation(tmp_path, sid, provider=provider, now=cycle_now, min_keep_tail=0)

        row = _summary_row(tmp_path, sid)
        sections = row["compaction"]["sections"]
        if "72h" in sections:
            saw_tier3 = True
            assert len(sections["72h"]["text"]) <= _SECTION_72H_CHAR_CAP

    assert saw_tier3, "terminal tier3 never populated across the steady-state cycles"
    # H6: same over-cap-output evidence as C3a — the mechanism fired every cycle.
    assert len(huge) > _SECTION_72H_CHAR_CAP


# --------------------------------------------------------------------------- C4


def test_c4_fold_validation_unit() -> None:
    refusal = "I won't do that."
    meta = "Here is the summary: blah blah blah blah blah blah blah blah"
    non_first_person = "The cat sat on the mat and did nothing interesting at all today"
    genuine = "I remember we talked about the project timeline and what came next."

    assert _validate_fold_output(refusal) is None
    assert _validate_fold_output(meta) is None
    assert _validate_fold_output(non_first_person) is None
    assert _validate_fold_output(genuine) == genuine
    assert _validate_fold_output("") is None
    assert _validate_fold_output("   ") is None

    # H6: the refusal-lead regex is the discriminating mechanism, not the
    # first-person check — the refusal text DOES contain a first-person
    # pronoun, so a validator lacking the refusal check would wrongly accept it.
    assert _FIRST_PERSON_RE_HAS_MATCH(refusal)
    assert _validate_fold_output(refusal) is None  # still rejected, by the refusal lead


def _FIRST_PERSON_RE_HAS_MATCH(text: str) -> bool:
    from brain.chat.compaction import _FIRST_PERSON

    return _FIRST_PERSON.search(text) is not None


def test_c4_cascade_double_reject_preserves_marker(tmp_path: Path) -> None:
    sid = "sess_c4"
    now0 = datetime.now(UTC)
    marker = "MARKERC4XYZ"
    _seed_turn(tmp_path, sid, now0 - timedelta(hours=30), "user", f"important context {marker}")
    write_cursor(tmp_path, sid, _iso(now0))

    refusal_provider = _Stub("I won't do that.")
    r1 = cascade_conversation(tmp_path, sid, provider=refusal_provider, now=now0, min_keep_tail=0)
    assert r1.compacted is True
    assert r1.tiers["24h"].validated is False
    row1 = _summary_row(tmp_path, sid)
    assert marker in row1["compaction"]["sections"]["24h"]["text"]

    # Advance a day so the 24h section graduates into 48h and gets refolded
    # (still double-rejected) — the marker must survive the graduation too.
    now1 = now0 + timedelta(hours=24)
    write_cursor(tmp_path, sid, _iso(now1))
    r2 = cascade_conversation(tmp_path, sid, provider=refusal_provider, now=now1, min_keep_tail=0)
    assert r2.compacted is True
    assert r2.tiers["48h"].validated is False
    row2 = _summary_row(tmp_path, sid)
    assert marker in row2["compaction"]["sections"]["48h"]["text"]

    # H6: the marker did NOT come from the refusal text itself (proving it
    # survived via the lossless-leaning fallback join, not an accidental
    # match) — a "drop-on-double-reject" scheme would have lost it entirely.
    assert marker not in refusal_provider.resp


# --------------------------------------------------------------------------- C5


def test_c5_temporal_markers_and_labels(tmp_path: Path) -> None:
    sid = "sess_c5"
    now = datetime.now(UTC)
    covered_ts = now - timedelta(hours=30)
    _seed_turn(tmp_path, sid, covered_ts, "user", "hello a")
    _seed_turn(tmp_path, sid, now - timedelta(hours=60), "user", "hello b")
    _seed_turn(tmp_path, sid, now - timedelta(hours=80), "user", "hello c")
    write_cursor(tmp_path, sid, _iso(now))

    provider = _Stub()
    result = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert result.compacted is True

    row = _summary_row(tmp_path, sid)
    text = row["text"]

    assert "yesterday" in text
    assert "day before yesterday" in text
    assert "a few days ago" in text
    # A coarse date-span token (derived from covered ts, not a per-render clock).
    assert covered_ts.strftime("%b") in text

    # H6: NOT the bare tier keys — the oracle would flag a "24h"/"48h"/"72h"
    # label scheme instead of the owner-specified human labels.
    assert "[24h" not in text
    assert "[48h" not in text
    assert "[72h" not in text


# --------------------------------------------------------------------------- C6


def test_c6_head_prefix_bytestable_and_reparse(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    sections = {
        "24h": {
            "text": "I recall talking about the project timeline.",
            "covers_from_ts": _iso(now - timedelta(hours=30)),
            "covers_until_ts": _iso(now - timedelta(hours=25)),
        },
        "48h": {
            "text": "I remember the earlier discussion about plans.",
            "covers_from_ts": _iso(now - timedelta(hours=60)),
            "covers_until_ts": _iso(now - timedelta(hours=50)),
        },
    }
    r1 = _render_sections(sections)
    r2 = _render_sections(sections)
    assert r1 == r2  # byte-identical, no nonce / live now() read

    sid = "sess_c6"
    ts_now = _iso(now)
    row = {
        "session_id": sid,
        "speaker": "summary",
        "text": r1,
        "ts": ts_now,
        "compaction": {"gen": 1, "folded": True, "covers_until_ts": ts_now, "sections": sections},
    }
    user_turn = {"session_id": sid, "speaker": "user", "text": "more chat", "ts": ts_now}

    history_msgs = _buffer_turns_to_messages(tmp_path, [row, user_turn])
    assert history_msgs[0].role == "system"
    assert history_msgs[0].content_text().startswith(_COMPACTION_SUMMARY_PREFIX)

    # Prepend a real persona system prompt, as engine.py does, so budget's
    # index-1 re-parse path is exercised faithfully.
    messages = [ChatMessage(role="system", content="PERSONA SYSTEM PROMPT"), *history_msgs]

    stub = _Stub()
    out = apply_budget(
        messages, max_tokens=10**9, preserve_tail_msgs=0,
        provider=stub, persona_dir=tmp_path, session_id=sid,
    )
    assert out == messages  # under cap -> identity passthrough
    assert out[1].role == "system"
    assert out[1].content_text().startswith(_COMPACTION_SUMMARY_PREFIX)

    # H6: an injected nonce/live-timestamp would break byte-stability.
    noisy_sections = {**sections, "24h": {**sections["24h"], "text": sections["24h"]["text"] + f" [{datetime.now(UTC).isoformat()}]"}}
    r3 = _render_sections(noisy_sections)
    assert r3 != r1


# --------------------------------------------------------------------------- C7


def test_c7_cursor_guard_and_idempotence(tmp_path: Path) -> None:
    sid = "sess_c7"
    now = datetime.now(UTC)
    extracted_ts = now - timedelta(hours=80)   # aged, at/before cursor -> extracted
    unextracted_ts = now - timedelta(hours=40)  # aged (>24h) but AFTER cursor -> NOT extracted

    _seed_turn(tmp_path, sid, extracted_ts, "user", "old turn")
    _seed_turn(tmp_path, sid, unextracted_ts, "user", "new turn")
    write_cursor(tmp_path, sid, _iso(extracted_ts))  # cursor covers only the old turn

    provider = _Stub()
    r1 = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert r1.compacted is True

    archive = read_archive(tmp_path, sid)
    archived_texts = [a.get("text") for a in archive]
    assert "old turn" in archived_texts
    assert "new turn" not in archived_texts  # un-extracted -> never compacted/archived

    turns = read_session(tmp_path, sid)
    raw_texts = [t.get("text") for t in turns if t.get("speaker") != "summary"]
    assert "new turn" in raw_texts  # still live in the buffer

    # H6: age alone would NOT have excluded it — it exceeds the 24h age gate —
    # so only the cursor guard explains its exclusion.
    assert (now - unextracted_ts) > timedelta(hours=24)

    # (b) idempotence: a second pass on stable input is a no-op.
    r2 = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert r2.compacted is False
    assert r2.reason == "nothing_aged"


# --------------------------------------------------------------------------- C13


def test_c13_interior_not_starved(tmp_path: Path) -> None:
    marker = "MARKERC13 a persisting interior thought"
    store = MemoryStore(tmp_path / "memories.db")
    write_trace_memory(store, marker)
    store.close()

    sid = "sess_c13"
    now = datetime.now(UTC)
    _seed_turn(tmp_path, sid, now - timedelta(hours=30), "user", "ordinary chat turn")
    write_cursor(tmp_path, sid, _iso(now))

    provider = _Stub()
    result = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert result.compacted is True

    store2 = MemoryStore(tmp_path / "memories.db")
    block = build_interior_continuity_block(store2)
    assert "MARKERC13" in block
    store2.close()

    # H6: nulling the trace store yields an empty block — proves the oracle
    # actually discriminates "has content" from "starved". Interior-continuity
    # reads the pending queue scoped by persona_dir (= db_path.parent), so
    # empty_store needs a DISTINCT persona_dir from `store` above — sharing
    # tmp_path would have both stores read the SAME pending queue and the
    # discriminator would spuriously see MARKERC13. MemoryStore.__init__ does
    # not create the dir, so it must be mkdir'd first.
    empty_dir = tmp_path / "empty_persona"
    empty_dir.mkdir(parents=True, exist_ok=True)
    empty_store = MemoryStore(empty_dir / "memories_empty.db")
    assert build_interior_continuity_block(empty_store) == ""
    empty_store.close()


# --------------------------------------------------------------------------- C14


def test_c14_graduation_and_terminal_persistence(tmp_path: Path) -> None:
    """Updated to the owner's replacement model (Bug 4): passes 1-3 are the
    same steady 24h->48h->72h graduation as before. Pass 4 (previously
    asserting the marker STAYS in 72h forever) now asserts it is EVICTED —
    absent from every tier, but recoverable from the archive (lossless). Pass 5
    (previously re-asserting persistence) now asserts a stable no-op: nothing
    left to fold, graduate, or evict."""
    sid = "sess_c14"
    now0 = datetime.now(UTC)
    marker0 = "MARK0"
    _seed_turn(tmp_path, sid, now0 - timedelta(hours=30), "user", f"note {marker0}")
    write_cursor(tmp_path, sid, _iso(now0))

    provider = _MarkerProvider()
    expected_tier_by_pass = {1: "24h", 2: "48h", 3: "72h"}

    for pass_n in range(1, 4):
        now = now0 + timedelta(hours=24 * (pass_n - 1))
        write_cursor(tmp_path, sid, _iso(now))
        result = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
        assert result.compacted is True

        row = _summary_row(tmp_path, sid)
        sections = row["compaction"]["sections"]
        expected_tier = expected_tier_by_pass[pass_n]
        assert marker0 in sections.get(expected_tier, {}).get("text", ""), (
            f"pass {pass_n}: expected {marker0!r} in tier {expected_tier!r}, got sections={sections!r}"
        )
        # H6: proves graduation actually moved the marker OUT of the earlier
        # tiers — a "stuck in tier1 forever" bug would fail this.
        for other in ("24h", "48h", "72h"):
            if other != expected_tier:
                assert marker0 not in sections.get(other, {}).get("text", ""), (
                    f"pass {pass_n}: unexpectedly found {marker0!r} still in tier {other!r}"
                )

    # Pass 4: the section's oldest edge (still now0-30h) has now aged past the
    # eviction boundary (96h) -> EVICTED from the head, not persisted in tier 3
    # forever (the retired pre-change behavior). Lossless: still in the archive.
    now4 = now0 + timedelta(hours=72)
    write_cursor(tmp_path, sid, _iso(now4))
    result4 = cascade_conversation(tmp_path, sid, provider=provider, now=now4, min_keep_tail=0)
    assert result4.compacted is True  # an eviction is a change, not a no-op
    row4 = _summary_row(tmp_path, sid)
    sections4 = row4["compaction"]["sections"]
    for tier in ("24h", "48h", "72h"):
        assert marker0 not in sections4.get(tier, {}).get("text", ""), (
            f"pass 4: {marker0!r} unexpectedly still present in tier {tier!r} "
            "(expected EVICTED under the replacement model)"
        )
    archive = read_archive(tmp_path, sid)
    assert any(marker0 in (a.get("text") or "") for a in archive), (
        "pass 4: the evicted marker must remain recoverable from the archive"
    )

    # Pass 5: nothing left to fold, graduate, or evict -> a stable no-op (does
    # NOT re-exercise eviction; pass 4 already did — this confirms stability).
    now5 = now0 + timedelta(hours=96)
    write_cursor(tmp_path, sid, _iso(now5))
    result5 = cascade_conversation(tmp_path, sid, provider=provider, now=now5, min_keep_tail=0)
    assert result5.compacted is False
    assert result5.reason == "nothing_aged"


def test_c14_multi_input_and_long_inactivity(tmp_path: Path) -> None:
    sid = "sess_c14b"
    now0 = datetime.now(UTC)
    provider = _MarkerProvider()

    # Steady-state: sow a fresh marker each day, aged ~30h into the 24h band.
    for day in range(4):
        cycle_now = now0 + timedelta(days=day)
        marker = f"MARKDAY{day}"
        _seed_turn(tmp_path, sid, cycle_now - timedelta(hours=30), "user", f"note {marker}")
        write_cursor(tmp_path, sid, _iso(cycle_now))
        cascade_conversation(tmp_path, sid, provider=provider, now=cycle_now, min_keep_tail=0)

    row = _summary_row(tmp_path, sid)
    sections = row["compaction"]["sections"]
    tier3_text = sections["72h"]["text"]
    # Updated to the owner's replacement model (Bug 4): by day3's pass, the
    # OLD tier-3 marker (day0) has aged past the eviction boundary and is
    # EVICTED (dropped from the head, not accumulated forever); only the
    # newly-graduated tier-2 marker (day1) lands in the new tier3 fold.
    assert "MARKDAY1" in tier3_text
    assert "MARKDAY0" not in tier3_text
    assert len(tier3_text) <= _SECTION_72H_CHAR_CAP
    archive = read_archive(tmp_path, sid)
    assert any("MARKDAY0" in (a.get("text") or "") for a in archive), (
        "evicted MARKDAY0 must remain recoverable from the archive"
    )
    # H6: a pre-change "accumulate forever" tier3 fold would still contain
    # MARKDAY0 here — the assertion above fails on that behavior.

    # Long-inactivity fixture: all raw turns aged well beyond 72h.
    sid2 = "sess_c14c"
    now = datetime.now(UTC)
    for i in range(3):
        _seed_turn(tmp_path, sid2, now - timedelta(hours=100 + i), "user", f"stale note {i}")
    write_cursor(tmp_path, sid2, _iso(now))

    result2 = cascade_conversation(tmp_path, sid2, provider=provider, now=now, min_keep_tail=0)
    assert result2.compacted is True
    row2 = _summary_row(tmp_path, sid2)
    sections2 = row2["compaction"]["sections"]
    assert "24h" not in sections2
    assert "48h" not in sections2
    assert "72h" in sections2


# --------------------------------------------------------------------------- C17


def test_c17_cascade_write_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "sess_c17"
    now0 = datetime.now(UTC)
    _seed_turn(tmp_path, sid, now0 - timedelta(hours=30), "user", "band24 content")
    _seed_turn(tmp_path, sid, now0 - timedelta(hours=60), "user", "band48 content")
    write_cursor(tmp_path, sid, _iso(now0))

    provider = _Stub()
    r1 = cascade_conversation(tmp_path, sid, provider=provider, now=now0, min_keep_tail=0)
    assert r1.compacted is True
    pre_row = _summary_row(tmp_path, sid)

    # Seed a turn that WILL change the row on the next pass (crosses into 72h).
    now1 = now0 + timedelta(hours=1)
    _seed_turn(tmp_path, sid, now0 - timedelta(hours=90), "user", "band72 fresh content")
    write_cursor(tmp_path, sid, _iso(now1))

    def _boom(*a, **kw):
        raise RuntimeError("simulated crash before the atomic replace lands")

    monkeypatch.setattr("brain.chat.compaction.rewrite_session_atomic", _boom)

    with pytest.raises(RuntimeError):
        cascade_conversation(tmp_path, sid, provider=provider, now=now1, min_keep_tail=0)

    post_row = _summary_row(tmp_path, sid)
    # H6: the on-disk row is byte-for-byte the pre-pass row — no partial tier
    # update landed. A "3-sequential-write" scheme would instead show a
    # partially-updated row (e.g. a new 72h beside the untouched pre-pass text).
    assert post_row == pre_row


# --------------------------------------------------------------------------- C22


def test_c22_apply_budget_sectioned_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "sess_c22"
    now = datetime.now(UTC)
    _seed_three_bands(tmp_path, sid, now)

    setup_provider = _Stub()
    r1 = cascade_conversation(tmp_path, sid, provider=setup_provider, now=now, min_keep_tail=0)
    assert r1.compacted is True
    pre_row = _summary_row(tmp_path, sid)
    pre_sections = pre_row["compaction"]["sections"]
    assert set(pre_sections) == {"24h", "48h", "72h"}
    pre_48 = pre_sections["48h"]["text"]
    pre_72 = pre_sections["72h"]["text"]

    # Grow the buffer past cap with fresh extracted raw turns.
    for i in range(50):
        _seed_turn(tmp_path, sid, now + timedelta(minutes=i), "user", "x" * 200)
    write_cursor(tmp_path, sid, _iso(now + timedelta(minutes=60)))

    stub = _Stub("I recall the recent chatter clearly.")
    # NOTE (API observation): apply_budget's own `provider` kwarg is NOT used
    # for the persisted-fold call — that call always builds its own provider
    # via brain.chat.compaction.build_compaction_provider(persona_dir), so the
    # kwarg must be controlled via this monkeypatch, not by passing a stub as
    # `provider=` to apply_budget (see report).
    monkeypatch.setattr("brain.chat.compaction.build_compaction_provider", lambda persona_dir: stub)

    turns_now = read_session(tmp_path, sid)
    history_msgs = _buffer_turns_to_messages(tmp_path, turns_now)
    messages = [ChatMessage(role="system", content="PERSONA SYSTEM PROMPT"), *history_msgs]

    out = apply_budget(
        messages, max_tokens=10, preserve_tail_msgs=2,
        provider=stub, persona_dir=tmp_path, session_id=sid,
    )

    post_row = _summary_row(tmp_path, sid)
    post_sections = post_row["compaction"]["sections"]
    assert "sections" in post_row["compaction"]  # not corrupted into a flat row
    assert "24h" in post_sections
    assert len(post_sections["24h"]["text"]) <= _SECTION_24H_CHAR_CAP
    assert post_sections["48h"]["text"] == pre_48   # tier 2 UNTOUCHED
    assert post_sections["72h"]["text"] == pre_72   # tier 3 UNTOUCHED

    assert out[1].role == "system"
    assert out[1].content_text().startswith(_COMPACTION_SUMMARY_PREFIX)

    # H6: a scratch backstop assuming a flat single-layer row would either
    # drop `compaction.sections` entirely or overwrite 48h/72h — both checked
    # above; demonstrate the discriminating baseline directly too.
    assert post_sections["24h"]["text"] != pre_sections["24h"]["text"]


# --------------------------------------------------------------------------- C1a/C1b


class _RefuseOnEmptyContent:
    """Mimics haiku's REAL observed behavior: refuses when the prompt's "new
    content" slot is empty. Detects the exact pre-change signature — _FOLD_PROMPT
    rendered with an empty transcript ends "NEW MESSAGES:\\n\\nUPDATED MEMORY:"
    (blank line where the transcript should be). With the Bug-1 fix, a
    (prior + no raw) fold never reaches _FOLD_PROMPT at all (it routes to
    _CONDENSE_PROMPT, which has no "NEW MESSAGES:" header), so this refusal
    trigger never fires post-fix."""

    def generate(self, *, prompt: str, system: str | None = None, **kw) -> str:
        if "NEW MESSAGES:\n\nUPDATED MEMORY:" in prompt:
            return "I don't have any new messages to integrate right now."
        return "I recall the earlier discussion, condensed now."


def test_c1a_condense_branch_not_mis_wired_fold(tmp_path: Path) -> None:
    """Bug 1 / D1 (C1a): a real non-empty prior with EMPTY raw routes to the new
    CONDENSE branch, not to _FOLD_PROMPT(transcript=''). Oracle: the real
    _fold_into_section with a provider that refuses on an empty content slot (as
    haiku does) — the returned section text must NOT be that refusal, must be
    first-person, validated, and not a fallback. Fails on pre-change code, which
    routes to _FOLD_PROMPT with an empty transcript and gets the refusal stored."""
    now = datetime.now(UTC)
    prior_section = {
        "text": "I remember talking with Bob about the project timeline and the budget.",
        "covers_from_ts": _iso(now - timedelta(hours=50)),
        "covers_until_ts": _iso(now - timedelta(hours=49)),
    }
    section, fell_soft, validated = _fold_into_section(
        "Canary", [prior_section], [], _FRACTION_48H, None,
        _RefuseOnEmptyContent(), now=now,
    )
    assert section is not None
    text = section["text"]
    assert "don't have any new messages" not in text.lower()
    assert validated is True
    assert fell_soft is False
    from brain.chat.compaction import _FIRST_PERSON
    assert _FIRST_PERSON.search(text) is not None


def test_c1b_condense_shrinks_toward_tier_fraction(tmp_path: Path) -> None:
    """Bug 1 / D1 (C1b): the CONDENSE path re-compacts — output word count is
    LESS than the prior's and lands near the tier target, rather than carrying
    the prior forward unchanged. Fails on pre-change code (no such branch)."""

    class _LengthHonoringProvider:
        def generate(self, *, prompt: str, system: str | None = None, **kw) -> str:
            m = _re.search(r"about (\d+) words", prompt)
            n = int(m.group(1)) if m else 50
            return "I recall " + ("detail " * max(0, n - 2))

    prior_text = "I remember " + (
        "talking about the project timeline and the budget with Bob and the "
        "whole team over several long meetings that ran into the evening. "
    ) * 6
    now = datetime.now(UTC)
    prior_section = {
        "text": prior_text,
        "covers_from_ts": _iso(now - timedelta(hours=50)),
        "covers_until_ts": _iso(now - timedelta(hours=49)),
    }
    section, fell_soft, validated = _fold_into_section(
        "Canary", [prior_section], [], _FRACTION_72H, None,
        _LengthHonoringProvider(), now=now,
    )
    assert section is not None
    assert validated is True
    assert fell_soft is False

    prior_words = _word_count(prior_text)
    out_words = _word_count(section["text"])
    assert out_words < prior_words

    expected_target = max(_MIN_TARGET_WORDS, int(prior_words * _FRACTION_72H))
    assert abs(out_words - expected_target) <= max(5, int(0.3 * expected_target)), (
        f"out_words={out_words} not near expected_target={expected_target}"
    )


@pytest.mark.requires_claude_cli
def test_condense_prompt_realmodel(tmp_path: Path) -> None:
    """C1d durable regression asset: the real CONDENSE prompt, run through the
    real haiku compaction provider on a fixed ~230-word first-person Canary
    memory, produces output that (i) passes the post-change
    _validate_fold_output, (ii) is not a refusal, and (iii) is shorter than the
    input. Excluded from the default CI marker set (needs a real CLI
    subprocess); run on demand as a safety net for future _CONDENSE_PROMPT
    edits (red-team M2)."""
    import shutil

    if shutil.which("claude") is None:
        pytest.skip("claude CLI not available")

    from brain.bridge.provider import get_provider

    canary_memory = (
        "I remember the long conversation with Bob about the eviction design "
        "for the cascade compaction work. He walked me through the owner's "
        "replacement model: each tier gets replaced by the recompacted "
        "younger tier instead of accumulating forever, and I understood why "
        "that mattered, the old tier three content was piling up without "
        "bound. We talked through the edge cases together, what happens when "
        "a section is exactly at the boundary, what happens when there is no "
        "raw chat to fold in that tier, and Bob seemed relieved once we "
        "settled on the ninety-six hour eviction line with a twelve hour "
        "margin on the legacy floor. I recall feeling a quiet satisfaction "
        "watching the design come together, the way the four bugs all traced "
        "back to the same missing branch in the fold function. We also "
        "talked about the cursor-guard question, whether an unextracted turn "
        "could ever be compacted away, and confirmed it could not. Toward "
        "the end Bob mentioned he still needed to double check the archive "
        "marker idempotency under a crash-retry, and I told him I would "
        "remember to ask about it again next time we spoke, since it felt "
        "like the kind of detail that could quietly break lossless "
        "guarantees if it slipped through unverified."
    )
    prior_words = _word_count(canary_memory)
    target_words = max(_MIN_TARGET_WORDS, int(prior_words * _FRACTION_72H))
    target_pct = round(_FRACTION_72H * 100)
    prompt = _CONDENSE_PROMPT.format(
        name="Canary", prior_summary=canary_memory,
        target_words=target_words, target_pct=target_pct,
    )

    provider = get_provider("claude-cli", model_override="haiku")
    out = _generate_validated_fold(provider, prompt)

    assert out is not None, "the real model's output was rejected by _validate_fold_output"
    assert not out.strip().lower().startswith((
        "i won't", "i will not", "i cannot", "i can't", "i'm sorry", "i am sorry",
        "i don't have", "i don't see", "i notice you",
    ))
    assert _word_count(out) < prior_words


# --------------------------------------------------------------------------- C-B3 (cascade path)


def _boom_rewrite(*a, **kw):
    raise RuntimeError("simulated crash before the atomic replace lands")


def test_cb3_cascade_crash_retry_no_duplicate_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug 3: driven through the real _install_cascade_row (via
    cascade_conversation) — a crash AFTER archive+marker but BEFORE the atomic
    rewrite must not duplicate archived records when retried over the
    unchanged buffer. Fails on pre-change code (re-archives on retry)."""
    sid = "sess_cb3_cascade"
    now = datetime.now(UTC)
    _seed_turn(tmp_path, sid, now - timedelta(hours=30), "user", "band24 content")
    _seed_turn(tmp_path, sid, now - timedelta(hours=60), "user", "band48 content")
    write_cursor(tmp_path, sid, _iso(now))

    provider = _Stub()
    monkeypatch.setattr("brain.chat.compaction.rewrite_session_atomic", _boom_rewrite)
    with pytest.raises(RuntimeError):
        cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)

    monkeypatch.undo()  # "process recovers": restore the real rewrite and retry
    res2 = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert res2.compacted is True

    archived_raw = [t for t in read_archive(tmp_path, sid) if t.get("speaker") != "summary"]
    counts = Counter((t.get("ts"), t.get("speaker"), t.get("text")) for t in archived_raw)
    assert all(c == 1 for c in counts.values()), f"duplicate archived records: {counts}"


def test_cb3b_cascade_multiset_no_loss_or_dup_across_crash_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug 3 / L2: a fixture with two BYTE-IDENTICAL (ts, speaker, text) raw
    turns survives a cascade crash + retry with neither turn lost NOR
    duplicated. The commit-marker design never dedups by identity, so this
    passes; a rejected identity-dedup design would drop one of the two."""
    sid = "sess_cb3b_cascade"
    now = datetime.now(UTC)
    dup_ts = now - timedelta(hours=30)
    _seed_turn(tmp_path, sid, dup_ts, "user", "identical content")
    _seed_turn(tmp_path, sid, dup_ts, "user", "identical content")
    _seed_turn(tmp_path, sid, now - timedelta(hours=60), "user", "band48 content")
    write_cursor(tmp_path, sid, _iso(now))

    original_raw = [t for t in read_session(tmp_path, sid) if t.get("speaker") != "summary"]
    original_counts = Counter(
        (t.get("ts"), t.get("speaker"), t.get("text")) for t in original_raw
    )

    provider = _Stub()
    monkeypatch.setattr("brain.chat.compaction.rewrite_session_atomic", _boom_rewrite)
    with pytest.raises(RuntimeError):
        cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    monkeypatch.undo()
    res2 = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert res2.compacted is True

    archived_raw = [t for t in read_archive(tmp_path, sid) if t.get("speaker") != "summary"]
    live_raw = [t for t in read_session(tmp_path, sid) if t.get("speaker") != "summary"]
    combined = Counter(
        (t.get("ts"), t.get("speaker"), t.get("text")) for t in archived_raw
    ) + Counter((t.get("ts"), t.get("speaker"), t.get("text")) for t in live_raw)
    assert combined == original_counts, (
        f"lossy or duplicated across crash+retry: got {combined!r}, want {original_counts!r}"
    )


# --------------------------------------------------------------------------- C-B4 (tier-3 eviction)


def test_cb4a_tier3_replaced_not_accumulated(tmp_path: Path) -> None:
    """Bug 4 (C-B4a): after a steady cascade pass, tier 3's new content derives
    from the recompacted graduated OLD-48h section and does NOT contain the
    old-72h content (replacement, not accumulation). Fails on pre-change code
    (old-72h marker persists in tier 3 forever)."""
    sid = "sess_cb4a"
    now = datetime.now(UTC)
    sections = {
        "24h": {
            "text": "I recall MARK24 from yesterday.",
            "covers_from_ts": _iso(now - timedelta(hours=30)),
            "covers_until_ts": _iso(now - timedelta(hours=29)),
        },
        "48h": {
            "text": "I recall MARK48 from a couple days back.",
            "covers_from_ts": _iso(now - timedelta(hours=80)),  # -> graduates to 72h
            "covers_until_ts": _iso(now - timedelta(hours=79)),
        },
        "72h": {
            "text": "I recall MARK72 from long, long ago.",
            "covers_from_ts": _iso(now - timedelta(hours=100)),  # >=96h -> evicted
            "covers_until_ts": _iso(now - timedelta(hours=99)),
        },
    }
    _write_sectioned_summary(tmp_path, sid, sections)
    write_cursor(tmp_path, sid, _iso(now))

    provider = _MarkerProvider()
    result = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert result.compacted is True

    row = _summary_row(tmp_path, sid)
    new_sections = row["compaction"]["sections"]
    tier3_text = new_sections.get("72h", {}).get("text", "")
    assert "MARK48" in tier3_text
    assert "MARK72" not in tier3_text
    assert "72h" in result.evicted_keys


def test_cb4b_evicted_content_excluded_from_head_present_in_archive(tmp_path: Path) -> None:
    """Bug 4 (C-B4b, behavior-preservation/position lens + eviction): content
    whose oldest edge is aged >= the eviction boundary (96h) is EXCLUDED from
    the installed head but PRESENT in the archive (lossless). repro_bug4-style:
    a fixture with a 9-day-old tier-3 section. Fails on pre-change code
    (9-day content retained in the head forever)."""
    sid = "sess_cb4b"
    now = datetime.now(UTC)
    marker = "MARKSTALE9DAY"
    sections = {
        "72h": {
            "text": f"I recall {marker} from long, long ago.",
            "covers_from_ts": _iso(now - timedelta(days=9)),
            "covers_until_ts": _iso(now - timedelta(days=8, hours=23)),
        },
    }
    _write_sectioned_summary(tmp_path, sid, sections)
    write_cursor(tmp_path, sid, _iso(now))

    provider = _MarkerProvider()
    result = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert result.compacted is True

    row = _summary_row(tmp_path, sid)
    installed_sections = row["compaction"]["sections"]
    for sec in installed_sections.values():
        assert marker not in sec.get("text", "")

    archive = read_archive(tmp_path, sid)
    archived_summaries = [a for a in archive if a.get("speaker") == "summary"]
    assert any(marker in (a.get("text") or "") for a in archived_summaries), (
        "the evicted section must remain recoverable from the archive"
    )


def test_cb4c_idempotent_rerun_same_now_is_noop(tmp_path: Path) -> None:
    """Bug 4 (C-B4c, idempotence preserved): a cascade pass immediately re-run
    with no new raw and no boundary crossed is a no-op (compacted=False,
    reason='nothing_aged') — the replacement/eviction change did not turn
    every call into a re-graduation/re-eviction."""
    sid = "sess_cb4c"
    now = datetime.now(UTC)
    _seed_turn(tmp_path, sid, now - timedelta(hours=30), "user", "note recent")
    write_cursor(tmp_path, sid, _iso(now))

    provider = _Stub()
    r1 = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert r1.compacted is True

    r2 = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert r2.compacted is False
    assert r2.reason == "nothing_aged"


def test_cb4d_docstrings_reflect_replacement_model() -> None:
    """Bug 4 (C-B4d, docs match code): the retain-forever docstrings/comments no
    longer say "terminal / re-compacted forever" or "leaving tiers 2/3
    untouched", and the replacement model wording is present instead. Fails on
    pre-change code (which contains the stale phrases)."""
    import inspect

    import brain.chat.compaction as comp

    source = inspect.getsource(comp)
    stale_phrases = [
        "STAYS in tier 3 (terminal, re-compacted forever",
        "a terminal, multi-input tier 3",
        "then stays in tier 3 (terminal, re-compacted forever)",
        "Tier 3 is a terminal, multi-input",
        "leaving tiers 2/3 untouched",
    ]
    for phrase in stale_phrases:
        assert phrase not in source, f"stale retain-forever wording still present: {phrase!r}"
    assert "evicted" in source.lower()
    assert "replaced" in source.lower() or "replacement" in source.lower()


def test_cb4f_emergency_fold_evicts_stale_section(tmp_path: Path) -> None:
    """Bug 4 / red-team CH8 (C-B4f): emergency_fold_24h (the apply_budget
    backstop) on a row carrying an evictable (>=96h) section drops that
    section from the installed head (present in archive). Fails on pre-change
    code AND on a fix that only evicts in cascade_conversation."""
    sid = "sess_cb4f"
    now = datetime.now(UTC)
    marker = "MARKSTALEEMFOLD"
    sections = {
        "72h": {
            "text": f"I recall {marker} from long, long ago.",
            "covers_from_ts": _iso(now - timedelta(days=9)),
            "covers_until_ts": _iso(now - timedelta(days=8, hours=23)),
        },
    }
    _write_sectioned_summary(tmp_path, sid, sections)
    # Give the backstop some fresh raw so it has something to fold (its own
    # early "nothing to fold" no-op guard is unrelated to eviction).
    _seed_turn(tmp_path, sid, now - timedelta(minutes=5), "user", "fresh recent chatter")
    write_cursor(tmp_path, sid, _iso(now))

    provider = _Stub()
    result = emergency_fold_24h(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert result.compacted is True
    assert "72h" in result.evicted_keys

    row = _summary_row(tmp_path, sid)
    installed = row["compaction"]["sections"]
    for sec in installed.values():
        assert marker not in sec.get("text", "")

    archive = read_archive(tmp_path, sid)
    assert any(
        marker in (a.get("text") or "") for a in archive if a.get("speaker") == "summary"
    ), "the evicted section must remain recoverable from the archive"


def test_cb4g_all_evicted_pass_installs_empty_row_cleanly(tmp_path: Path) -> None:
    """Bug 4 (C-B4g, newly-reachable degenerate row): a pass that evicts the
    only section(s) with no raw to fold installs text=""/sections={} without
    raising; _read_sections re-reads it as {}; a subsequent pass is a clean
    no-op."""
    sid = "sess_cb4g"
    now = datetime.now(UTC)
    sections = {
        "72h": {
            "text": "I recall stale content from long, long ago.",
            "covers_from_ts": _iso(now - timedelta(days=9)),
            "covers_until_ts": _iso(now - timedelta(days=8, hours=23)),
        },
    }
    _write_sectioned_summary(tmp_path, sid, sections)
    write_cursor(tmp_path, sid, _iso(now))

    provider = _MarkerProvider()
    result = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert result.compacted is True  # eviction is a change, not a no-op

    row = _summary_row(tmp_path, sid)
    assert row["text"] == ""
    assert row["compaction"]["sections"] == {}

    reread = _read_sections(row, now)
    assert reread == {}

    result2 = cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)
    assert result2.compacted is False
    assert result2.reason == "nothing_aged"
