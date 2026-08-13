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
"""

from __future__ import annotations

import re as _re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.chat import ChatMessage
from brain.chat.budget import _COMPACTION_SUMMARY_PREFIX, apply_budget
from brain.chat.compaction import (
    _SECTION_24H_CHAR_CAP,
    _SECTION_72H_CHAR_CAP,
    _read_sections,
    _render_sections,
    _validate_fold_output,
    cascade_conversation,
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
    # actually discriminates "has content" from "starved".
    empty_store = MemoryStore(tmp_path / "memories_empty.db")
    assert build_interior_continuity_block(empty_store) == ""
    empty_store.close()


# --------------------------------------------------------------------------- C14


def test_c14_graduation_and_terminal_persistence(tmp_path: Path) -> None:
    sid = "sess_c14"
    now0 = datetime.now(UTC)
    marker0 = "MARK0"
    _seed_turn(tmp_path, sid, now0 - timedelta(hours=30), "user", f"note {marker0}")
    write_cursor(tmp_path, sid, _iso(now0))

    provider = _MarkerProvider()
    expected_tier_by_pass = {1: "24h", 2: "48h", 3: "72h", 4: "72h", 5: "72h"}

    for pass_n in range(1, 6):
        now = now0 + timedelta(hours=24 * (pass_n - 1))
        write_cursor(tmp_path, sid, _iso(now))
        cascade_conversation(tmp_path, sid, provider=provider, now=now, min_keep_tail=0)

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
    tier3_text = row["compaction"]["sections"]["72h"]["text"]
    # By day3's pass, the persisting-prior-tier3 marker (day0) AND the
    # newly-graduated-tier2 marker (day1) BOTH land in the new tier3 fold —
    # neither input dropped (multi-input, F1) — and it stays within cap.
    assert "MARKDAY0" in tier3_text
    assert "MARKDAY1" in tier3_text
    assert len(tier3_text) <= _SECTION_72H_CHAR_CAP
    # H6: a single-input tier3 fold that kept only ONE input would fail one
    # of the two assertions above.

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
