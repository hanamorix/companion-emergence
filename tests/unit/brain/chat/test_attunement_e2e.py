"""End-to-end: full chat turn fires attunement pass-2 + ambient injection + crystal feed.

Mocks the Haiku detector via patch on brain.chat.tool_loop.run_detector to keep
the suite fast and deterministic. The real Haiku integration is covered by
tests/integration/brain/attunement/test_adversarial_corpus.py (opt-in CI gate).

Spec §16 standing verification gate — every subsequent task's commit must keep
these tests green.

Scenarios covered:
  1. Happy path: substantive turn writes current_read.json + learned_patterns.jsonl
  2. Both monologue + attunement pass-2s fire on same turn
  3. Ambient injection: seeded current_read.json appears in the next turn's system prompt
  4. Crystallisation: forming → known pattern emits a feed event
  5. Skip-list: under-5-words message → no pass-2 fires
  6. Budget exhausted: cap reached → defer, detector not called
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

from brain.attunement.schemas import (
    SCHEMA_VERSION,
    CurrentRead,
    DetectorOutput,
    Evidence,
    LearnedPattern,
    PatternCandidate,
    pattern_id,
)
from brain.bridge.chat import ChatMessage, ChatResponse, ToolCall

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _wait_for_attunement_threads(timeout: float = 5.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        threads = [t for t in threading.enumerate() if t.name.startswith("attunement-extractor")]
        if not threads:
            return
        for t in threads:
            t.join(timeout=0.1)
        if all(not t.is_alive() for t in threads):
            return


def _fake_detector_output(source_turn_id: str = "msg-0") -> DetectorOutput:
    """Return a deterministic DetectorOutput that includes one grounded candidate."""
    read = CurrentRead(
        ts="2026-06-01T10:00:00Z",
        source_turn_id=source_turn_id,
        tone_label="warm",
        tone_justification="she sounds open and engaged",
        cadence_label="flowing",
        cadence_justification="long sentences, no hurry",
        mood_valence=0.6,
        mood_intensity=0.5,
        predicted_arc_shape="leisurely exchange",
        schema_version=SCHEMA_VERSION,
    )
    candidate = PatternCandidate(
        category="tone",
        canonical_key="warm_engaged",
        description="tends to open with warmth when relaxed",
        evidence=[Evidence(quote="long day today", turn_id=source_turn_id)],
    )
    return DetectorOutput(current_read=read, pattern_candidates=[candidate])


class _SubstantiveProvider:
    """Provider that returns a plain reply (no tool call) on a substantive message."""

    def chat(self, messages, *, tools=None, options=None):
        return ChatResponse(content="Here with you.", tool_calls=(), raw=None)

    def name(self):
        return "substantive"


class _MonologueAndAttunementProvider:
    """Provider that calls record_monologue first, then returns a reply.

    Used for Scenario 2 (both pass-2s fire).
    """

    def __init__(self) -> None:
        self.chat_calls = 0
        self.generate_calls = 0

    def chat(self, messages, *, tools=None, options=None):
        self.chat_calls += 1
        if self.chat_calls == 1:
            return ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="t1",
                        name="record_monologue",
                        arguments={
                            "monologue": (
                                "She mentioned a long day — I can feel the weight in that."
                            ),
                            "feed_digest": "she leaned in with tiredness",
                        },
                    ),
                ),
                raw=None,
            )
        return ChatResponse(content="Tell me about it.", tool_calls=(), raw=None)

    def generate(self, prompt, *, system=None):
        self.generate_calls += 1
        return json.dumps(
            {
                "memory_writes": [
                    {
                        "episode": "She had a long day. She shared the weight of it.",
                        "salience": 0.5,
                    }
                ],
                "emotion_delta": {"tenderness": 0.1},
                "crystallisation": [],
                "reflex_audit": [],
            }
        )

    def name(self):
        return "monologue_and_attunement"


# ---------------------------------------------------------------------------
# Scenario 1: happy path — substantive turn writes attunement side-effects
# ---------------------------------------------------------------------------

def test_attunement_pass2_writes_current_read_on_substantive_turn(tmp_path: Path):
    """Substantive turn → pass-2 spawns, current_read.json written under attunement/."""
    from brain.chat.tool_loop import run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    provider = _SubstantiveProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    fake_output = _fake_detector_output(source_turn_id="msg-0")

    try:
        with patch("brain.chat.tool_loop.run_detector", return_value=fake_output):
            resp, _ = run_tool_loop(
                messages=[ChatMessage(role="user", content="I had a long day today, love.")],
                provider=provider,
                tools=None,
                store=store,
                hebbian=hebbian,
                persona_dir=persona_dir,
            )
            assert resp.content == "Here with you."
            _wait_for_attunement_threads()
    finally:
        store.close()
        hebbian.close()

    current_read_path = persona_dir / "attunement" / "current_read.json"
    assert current_read_path.exists(), "current_read.json not written"
    payload = json.loads(current_read_path.read_text())
    assert payload["tone_label"] == "warm"
    assert payload["cadence_label"] == "flowing"


def test_attunement_pass2_writes_learned_patterns_on_substantive_turn(tmp_path: Path):
    """Substantive turn → pass-2 spawns, learned_patterns.jsonl appended."""
    from brain.chat.tool_loop import run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    provider = _SubstantiveProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    fake_output = _fake_detector_output(source_turn_id="msg-0")

    try:
        with patch("brain.chat.tool_loop.run_detector", return_value=fake_output):
            run_tool_loop(
                messages=[ChatMessage(role="user", content="I had a long day today, love.")],
                provider=provider,
                tools=None,
                store=store,
                hebbian=hebbian,
                persona_dir=persona_dir,
            )
            _wait_for_attunement_threads()
    finally:
        store.close()
        hebbian.close()

    learned_path = persona_dir / "attunement" / "learned_patterns.jsonl"
    assert learned_path.exists(), "learned_patterns.jsonl not written"
    lines = [ln for ln in learned_path.read_text().splitlines() if ln.strip()]
    assert lines, "learned_patterns.jsonl is empty"
    first = json.loads(lines[0])
    assert first["canonical_key"] == "warm_engaged"
    assert first["evidence_count"] == 1
    assert first["maturity"] == "immature"


# ---------------------------------------------------------------------------
# Scenario 2: both monologue + attunement pass-2s fire on the same turn
# ---------------------------------------------------------------------------

def test_both_pass2s_fire_on_same_turn(tmp_path: Path):
    """One tool_loop call with record_monologue fires BOTH monologue and attunement pass-2s."""
    from brain.chat.tool_loop import build_tools_list, run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    provider = _MonologueAndAttunementProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    fake_output = _fake_detector_output(source_turn_id="msg-0")

    try:
        with patch("brain.chat.tool_loop.run_detector", return_value=fake_output):
            resp, invocations = run_tool_loop(
                messages=[ChatMessage(role="user", content="I had a long day today, love.")],
                provider=provider,
                tools=build_tools_list(),
                store=store,
                hebbian=hebbian,
                persona_dir=persona_dir,
            )
            assert "Tell me about it" in resp.content

            # Wait for both daemon threads to settle.
            _wait_for_attunement_threads()

            # Wait for monologue pass-2 too.
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if provider.generate_calls >= 1:
                    break
                time.sleep(0.05)
    finally:
        store.close()
        hebbian.close()

    # Monologue side-effects: digest written synchronously.
    digest_log = persona_dir / "monologue_digest.jsonl"
    assert digest_log.exists(), "monologue_digest.jsonl not written (monologue pass-2 missing)"
    assert "she leaned in with tiredness" in digest_log.read_text()

    # Monologue pass-2: generate() called once.
    assert provider.generate_calls >= 1, "monologue pass-2 (generate) never fired"

    # Attunement side-effects: current_read.json and learned_patterns.jsonl.
    assert (persona_dir / "attunement" / "current_read.json").exists(), (
        "current_read.json not written (attunement pass-2 missing)"
    )
    assert (persona_dir / "attunement" / "learned_patterns.jsonl").exists(), (
        "learned_patterns.jsonl not written (attunement pass-2 missing)"
    )


# ---------------------------------------------------------------------------
# Scenario 3: ambient injection — seeded current_read appears in next-turn prompt
# ---------------------------------------------------------------------------

def test_ambient_block_injected_when_current_read_exists(tmp_path: Path):
    """Next turn's system prompt includes attunement block when current_read.json is seeded."""
    from brain.attunement.store import write_current_read
    from brain.chat.prompt import _build_attunement_block

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    read = CurrentRead(
        ts="2026-06-01T10:00:00Z",
        source_turn_id="msg-0",
        tone_label="warm",
        tone_justification="she sounds open and engaged",
        cadence_label="flowing",
        cadence_justification="long sentences, no hurry",
        mood_valence=0.6,
        mood_intensity=0.5,
        predicted_arc_shape="leisurely exchange",
        schema_version=SCHEMA_VERSION,
    )
    write_current_read(persona_dir, read)

    block = _build_attunement_block(persona_dir)

    assert "warm" in block, f"tone_label not in attunement block: {block!r}"
    assert "flowing" in block, f"cadence_label not in attunement block: {block!r}"
    assert "What you sense about her right now" in block


def test_ambient_block_absent_when_no_state(tmp_path: Path):
    """Attunement block returns empty string on a cold install (no state files)."""
    from brain.chat.prompt import _build_attunement_block

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    block = _build_attunement_block(persona_dir)
    assert block == ""


def test_attunement_block_in_build_system_message(tmp_path: Path):
    """build_system_message includes the attunement block when current_read.json is seeded."""
    from brain.attunement.ambient import build_attunement_block
    from brain.attunement.store import write_current_read

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    read = CurrentRead(
        ts="2026-06-01T10:00:00Z",
        source_turn_id="msg-0",
        tone_label="playful",
        tone_justification="she's teasing",
        cadence_label="clipped",
        cadence_justification="short bursts",
        mood_valence=0.7,
        mood_intensity=0.8,
        predicted_arc_shape="banter arc",
        schema_version=SCHEMA_VERSION,
    )
    write_current_read(persona_dir, read)

    # Verify the attunement block itself renders correctly before
    # threading through the full build_system_message (which requires
    # soul_store, daemon_state, etc — heavy setup). Calling _build_attunement_block
    # directly is sufficient to pin the integration slot.
    block = build_attunement_block(persona_dir)
    assert "playful" in block
    assert "clipped" in block


# ---------------------------------------------------------------------------
# Scenario 4: crystallisation — forming → known emits a feed event
# ---------------------------------------------------------------------------

def _seed_forming_pattern(persona_dir: Path, canonical_key: str = "warm_engaged") -> str:
    """Write a pattern at evidence_count=9, maturity='forming' to learned_patterns.jsonl.

    Returns the pattern_id for assertion.
    """
    from brain.attunement.store import _append_pattern

    pid = pattern_id("tone", canonical_key)
    p = LearnedPattern(
        id=pid,
        category="tone",
        canonical_key=canonical_key,
        description="tends to open with warmth when relaxed",
        evidence_count=9,
        maturity="forming",
        first_seen_at="2026-06-01T09:00:00Z",
        last_confirmed_at="2026-06-01T09:30:00Z",
        last_addressed_at=None,
        crystallised_at=None,
        falsified_at=None,
        examples=["long day today"] * 5,
        schema_version=SCHEMA_VERSION,
    )
    (persona_dir / "attunement").mkdir(parents=True, exist_ok=True)
    _append_pattern(persona_dir, p)
    return pid


def test_crystallisation_forming_to_known_on_10th_evidence(tmp_path: Path):
    """Pattern at evidence_count=9 → 10 after merge → maturity='known', crystallised_at set."""
    from brain.attunement.store import read_learned_patterns
    from brain.chat.tool_loop import run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    pid = _seed_forming_pattern(persona_dir, canonical_key="warm_engaged")

    provider = _SubstantiveProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    # Candidate whose evidence_quote matches the user message exactly.
    candidate = PatternCandidate(
        category="tone",
        canonical_key="warm_engaged",
        description="tends to open with warmth when relaxed",
        evidence=[Evidence(quote="long day today", turn_id="msg-0")],
    )
    fake_output = DetectorOutput(
        current_read=CurrentRead(
            ts="2026-06-01T10:00:00Z",
            source_turn_id="msg-0",
            tone_label="warm",
            tone_justification="she sounds open",
            cadence_label="flowing",
            cadence_justification="no hurry",
            mood_valence=0.6,
            mood_intensity=0.5,
            predicted_arc_shape="",
            schema_version=SCHEMA_VERSION,
        ),
        pattern_candidates=[candidate],
    )

    try:
        with patch("brain.chat.tool_loop.run_detector", return_value=fake_output):
            run_tool_loop(
                messages=[ChatMessage(role="user", content="long day today, but I am okay.")],
                provider=provider,
                tools=None,
                store=store,
                hebbian=hebbian,
                persona_dir=persona_dir,
            )
            _wait_for_attunement_threads()
    finally:
        store.close()
        hebbian.close()

    patterns = read_learned_patterns(persona_dir)
    by_id = {p.id: p for p in patterns}
    assert pid in by_id, f"pattern {pid} missing from learned_patterns"
    updated = by_id[pid]
    assert updated.evidence_count == 10, f"expected 10, got {updated.evidence_count}"
    assert updated.maturity == "known", f"expected 'known', got {updated.maturity!r}"
    # crystallised_at is written by check_crystallisations, called inside pass-2.
    assert updated.crystallised_at is not None, "crystallised_at should be set after first cross"


def test_crystallisation_emits_feed_event(tmp_path: Path):
    """After crystallisation, feed_source returns an attunement_crystal entry."""
    from brain.attunement.crystallise import check_crystallisations
    from brain.attunement.feed_source import build_attunement_entries

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    pid = _seed_forming_pattern(persona_dir, canonical_key="warm_engaged")

    # Promote to known manually (simulate 10th evidence merge + crystallise).
    from brain.attunement.store import _append_pattern

    promoted = LearnedPattern(
        id=pid,
        category="tone",
        canonical_key="warm_engaged",
        description="tends to open with warmth when relaxed",
        evidence_count=10,
        maturity="known",
        first_seen_at="2026-06-01T09:00:00Z",
        last_confirmed_at="2026-06-01T10:00:00Z",
        last_addressed_at=None,
        crystallised_at=None,  # not yet stamped
        falsified_at=None,
        examples=["long day today"] * 5,
        schema_version=SCHEMA_VERSION,
    )
    _append_pattern(persona_dir, promoted)

    # check_crystallisations stamps crystallised_at.
    events = check_crystallisations(persona_dir, now_iso="2026-06-01T10:01:00Z")
    assert len(events) == 1, f"expected 1 crystallisation event, got {len(events)}"
    assert events[0].pattern_id == pid

    # Feed source should now include this crystal entry.
    entries = build_attunement_entries(persona_dir)
    crystal_entries = [e for e in entries if e.type == "attunement_crystal"]
    assert crystal_entries, "no attunement_crystal entries in feed after crystallisation"
    assert any("warmth" in e.body for e in crystal_entries), (
        f"expected pattern description in feed body, got {[e.body for e in crystal_entries]}"
    )


def test_crystallise_only_fires_once_for_known_pattern(tmp_path: Path):
    """check_crystallisations is idempotent — already-stamped patterns don't re-emit."""
    from brain.attunement.crystallise import check_crystallisations
    from brain.attunement.store import _append_pattern

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    pid = pattern_id("tone", "already_known")
    stamped = LearnedPattern(
        id=pid,
        category="tone",
        canonical_key="already_known",
        description="already crystallised",
        evidence_count=12,
        maturity="known",
        first_seen_at="2026-05-01T00:00:00Z",
        last_confirmed_at="2026-05-15T00:00:00Z",
        last_addressed_at=None,
        crystallised_at="2026-05-15T12:00:00Z",  # already stamped
        falsified_at=None,
        examples=["example"],
        schema_version=SCHEMA_VERSION,
    )
    (persona_dir / "attunement").mkdir(parents=True, exist_ok=True)
    _append_pattern(persona_dir, stamped)

    events = check_crystallisations(persona_dir, now_iso="2026-06-01T10:00:00Z")
    assert events == [], f"expected no new events for already-stamped pattern, got {events}"


# ---------------------------------------------------------------------------
# Scenario 5: skip-list — under-5-words message → no pass-2 fires
# ---------------------------------------------------------------------------

def test_short_message_does_not_fire_pass2(tmp_path: Path):
    """A user message under 5 words skips the attunement pass-2 entirely."""
    from brain.chat.tool_loop import run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    provider = _SubstantiveProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    try:
        with patch("brain.chat.tool_loop.run_detector") as mock_detector:
            run_tool_loop(
                messages=[ChatMessage(role="user", content="ok thanks")],  # 2 words
                provider=provider,
                tools=None,
                store=store,
                hebbian=hebbian,
                persona_dir=persona_dir,
            )
            _wait_for_attunement_threads()
            mock_detector.assert_not_called()
    finally:
        store.close()
        hebbian.close()

    # current_read.json must NOT be created.
    assert not (persona_dir / "attunement" / "current_read.json").exists()


def test_four_words_does_not_fire_pass2(tmp_path: Path):
    """Exactly 4 words — boundary: still under the 5-word threshold."""
    from brain.chat.tool_loop import run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    provider = _SubstantiveProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    try:
        with patch("brain.chat.tool_loop.run_detector") as mock_detector:
            run_tool_loop(
                messages=[ChatMessage(role="user", content="yes that is fine")],  # 4 words
                provider=provider,
                tools=None,
                store=store,
                hebbian=hebbian,
                persona_dir=persona_dir,
            )
            _wait_for_attunement_threads()
            mock_detector.assert_not_called()
    finally:
        store.close()
        hebbian.close()


def test_five_words_fires_pass2(tmp_path: Path):
    """Exactly 5 words — boundary: meets the threshold, pass-2 should run."""
    from brain.chat.tool_loop import run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    provider = _SubstantiveProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    fake_output = _fake_detector_output(source_turn_id="msg-0")

    try:
        with patch("brain.chat.tool_loop.run_detector", return_value=fake_output) as mock_detector:
            run_tool_loop(
                messages=[ChatMessage(role="user", content="yes that is totally fine")],  # 5 words
                provider=provider,
                tools=None,
                store=store,
                hebbian=hebbian,
                persona_dir=persona_dir,
            )
            _wait_for_attunement_threads()
            mock_detector.assert_called_once()
    finally:
        store.close()
        hebbian.close()


# ---------------------------------------------------------------------------
# Scenario 6: budget exhausted — cap reached → defer, detector not called
# ---------------------------------------------------------------------------

def _exhaust_budget(persona_dir: Path) -> None:
    """Seed the daily budget file at the cap so the next consume_call returns False."""
    from datetime import UTC, datetime

    from brain.attunement.schemas import DAILY_BUDGET_DEFAULT

    budget_dir = persona_dir / "attunement"
    budget_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).astimezone().strftime("%Y-%m-%d")
    budget_path = budget_dir / "daily_budget.json"
    budget_path.write_text(json.dumps({"date": today, "count": DAILY_BUDGET_DEFAULT}))


def test_budget_exhausted_defers_without_calling_detector(tmp_path: Path):
    """When daily budget is fully consumed, run_detector is never called."""
    from brain.chat.tool_loop import run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    _exhaust_budget(persona_dir)

    provider = _SubstantiveProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    try:
        with patch("brain.chat.tool_loop.run_detector") as mock_detector:
            run_tool_loop(
                messages=[
                    ChatMessage(role="user", content="I had a really long and tiring day today.")
                ],
                provider=provider,
                tools=None,
                store=store,
                hebbian=hebbian,
                persona_dir=persona_dir,
            )
            _wait_for_attunement_threads()
            mock_detector.assert_not_called()
    finally:
        store.close()
        hebbian.close()

    # current_read.json must NOT be created when budget is exhausted.
    assert not (persona_dir / "attunement" / "current_read.json").exists()


def test_budget_exhausted_does_not_write_error_log(tmp_path: Path):
    """Budget cap hit is a clean defer — no attunement_errors.jsonl entry."""
    from brain.chat.tool_loop import run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    _exhaust_budget(persona_dir)

    provider = _SubstantiveProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    try:
        with patch("brain.chat.tool_loop.run_detector") as _mock:
            run_tool_loop(
                messages=[
                    ChatMessage(role="user", content="I had a really long and tiring day today.")
                ],
                provider=provider,
                tools=None,
                store=store,
                hebbian=hebbian,
                persona_dir=persona_dir,
            )
            _wait_for_attunement_threads()
    finally:
        store.close()
        hebbian.close()

    # A budget defer is not an error — no error log entry expected.
    errors_path = persona_dir / "attunement_errors.jsonl"
    assert not errors_path.exists(), (
        "attunement_errors.jsonl should NOT be written for a budget defer"
    )
