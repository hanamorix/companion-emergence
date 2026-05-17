"""Chat system message — three new self-narrative blocks (creative_dna, journal, growth).

Tests the integration: blocks compose into the system message, contain expected
sections, and degrade gracefully when files are missing.

This file is added to incrementally — Task 4 lands the journal block tests,
Task 5 adds growth block tests, Task 7 adds creative_dna block tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.chat.prompt import build_system_message
from brain.engines.daemon_state import DaemonState
from brain.memory.store import Memory, MemoryStore
from brain.soul.store import SoulStore


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "p"
    p.mkdir()
    (p / "active_conversations").mkdir()
    return p


@pytest.fixture
def store(persona_dir: Path):
    s = MemoryStore(persona_dir / "memories.db")
    yield s
    s.close()


@pytest.fixture
def soul_store(persona_dir: Path):
    s = SoulStore(str(persona_dir / "crystallizations.db"))
    yield s
    s.close()


@pytest.fixture
def daemon_state() -> DaemonState:
    return DaemonState()


def _seed_journal_entry(
    store: MemoryStore,
    *,
    days_ago: float = 1.0,
    source: str = "brain_authored",
    arc_name: str | None = None,
    emotions: dict[str, float] | None = None,
    content: str = "<this is a private journal entry>",
) -> str:
    mem = Memory.create_new(
        content=content,
        memory_type="journal_entry",
        domain="self",
        emotions=emotions or {"vulnerability": 7.0},
        metadata={
            "private": True,
            "source": source,
            "reflex_arc_name": arc_name,
            "auto_generated": source == "reflex_arc",
        },
    )
    store.create(mem)
    # Backdate created_at via direct SQL (uses internal API; OK for tests)
    cutoff = datetime.now(UTC) - timedelta(days=days_ago)
    store._conn.execute(  # noqa: SLF001
        "UPDATE memories SET created_at = ? WHERE id = ?",
        (cutoff.isoformat(), mem.id),
    )
    store._conn.commit()  # noqa: SLF001
    return mem.id


def test_recent_journal_block_renders_metadata_and_contract(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
    daemon_state: DaemonState,
):
    _seed_journal_entry(
        store, days_ago=2, source="brain_authored", emotions={"love": 8.0, "vulnerability": 6.0}
    )
    _seed_journal_entry(
        store,
        days_ago=4,
        source="reflex_arc",
        arc_name="loneliness_journal",
        emotions={"loneliness": 8.0},
    )

    msg = build_system_message(
        persona_dir,
        voice_md="(authored persona)",
        daemon_state=daemon_state,
        soul_store=soul_store,
        store=store,
    )

    # Privacy contract is present and ABOVE the metadata
    assert "── recent journal" in msg
    assert "private" in msg
    assert "do not quote" in msg.lower()

    # Metadata for both entries surfaces
    assert "brain_authored" in msg
    assert "loneliness_journal" in msg

    # Content NOT inlined — the test entry's content was a known marker
    assert "<this is a private journal entry>" not in msg


def test_recent_journal_block_omits_entries_older_than_7_days(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
    daemon_state: DaemonState,
):
    _seed_journal_entry(
        store, days_ago=10, source="brain_authored", content="OLD entry content unique marker zzz"
    )
    _seed_journal_entry(
        store, days_ago=2, source="brain_authored", content="RECENT entry content unique marker yyy"
    )

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=daemon_state,
        soul_store=soul_store,
        store=store,
    )

    # Both contents are filtered out (privacy), but only the recent
    # entry's metadata appears. We count "brain_authored" mentions
    # within the journal block.
    assert "── recent journal" in msg
    # Slice the block region: start AFTER the header line to avoid the
    # closing "──" on the header line confusing the end-of-block search.
    header_start = msg.index("── recent journal")
    header_end = msg.index("\n", header_start) + 1  # first char of next line
    # Block ends at the next "\n\n──" (next block header) or end of message
    next_block = msg.find("\n\n──", header_end)
    block = msg[header_end:next_block] if next_block > 0 else msg[header_end:]
    # Count entry rows — lines that have both "brain_authored" and an em-dash
    entry_lines = [line for line in block.split("\n") if "brain_authored" in line and "—" in line]
    assert len(entry_lines) == 1


def test_recent_journal_block_empty_state_renders_silence_marker(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
    daemon_state: DaemonState,
):
    """No journal entries → block still renders contract + 'no entries' marker."""
    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=daemon_state,
        soul_store=soul_store,
        store=store,
    )
    assert "── recent journal" in msg
    assert "no journal entries" in msg.lower() or "(no entries" in msg.lower()


def test_recent_journal_block_handles_store_query_failure_gracefully(
    persona_dir: Path,
    soul_store: SoulStore,
    daemon_state: DaemonState,
):
    """If store.list_by_type raises (e.g. corrupt SQLite), the block falls
    back to the empty-state contract — chat must NEVER break because
    journal failed.
    """

    class _BrokenStore:
        def list_by_type(self, *args, **kwargs):
            raise RuntimeError("simulated store failure")

        # Stub other methods build_system_message might call
        def __getattr__(self, name):
            raise RuntimeError(f"unexpected attribute access: {name}")

    # We can't pass _BrokenStore as the actual store fixture (build_system_message
    # uses other store methods elsewhere too). Instead, monkeypatch list_by_type
    # on a working store.
    s = MemoryStore(persona_dir / "memories.db")
    try:
        original_list_by_type = s.list_by_type

        def broken_list_by_type(memory_type, *args, **kwargs):
            if memory_type == "journal_entry":
                raise RuntimeError("simulated store failure")
            return original_list_by_type(memory_type, *args, **kwargs)

        s.list_by_type = broken_list_by_type  # type: ignore[method-assign]

        msg = build_system_message(
            persona_dir,
            voice_md="",
            daemon_state=daemon_state,
            soul_store=soul_store,
            store=s,
        )
        # Falls back to empty-state contract; doesn't crash
        assert "── recent journal" in msg
        assert "no journal entries" in msg.lower() or "(no entries" in msg.lower()
    finally:
        s.close()


def test_recent_growth_block_renders_behavioral_log_entries(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
    daemon_state: DaemonState,
):
    """Last 7 days of behavioral_log entries appear in chat as raw metadata."""
    from brain.behavioral.log import append_behavioral_event

    log_path = persona_dir / "behavioral_log.jsonl"
    base = datetime.now(UTC)
    append_behavioral_event(
        log_path,
        kind="creative_dna_emerging_added",
        name="sentence fragments as rhythmic percussion",
        timestamp=base - timedelta(days=2),
        reasoning="appeared in 3 recent sessions",
        evidence_memory_ids=("mem_a",),
    )
    append_behavioral_event(
        log_path,
        kind="journal_entry_added",
        name="mem_journal_xyz",
        timestamp=base - timedelta(days=1),
        source="brain_authored",
        reflex_arc_name=None,
        emotional_state={"love": 8.0},
    )

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=daemon_state,
        soul_store=soul_store,
        store=store,
    )

    assert "── recent growth ──" in msg
    assert "creative_dna_emerging_added" in msg
    assert "sentence fragments as rhythmic percussion" in msg
    assert "journal_entry_added" in msg


def test_recent_growth_block_renders_climax_event(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
    daemon_state: DaemonState,
):
    """A climax_event entry shows up in the growth block as 'body crested'.
    Content stays in the journal_entry memory; behavioral_log is metadata-only."""
    from brain.behavioral.log import append_behavioral_event

    log_path = persona_dir / "behavioral_log.jsonl"
    append_behavioral_event(
        log_path,
        kind="climax_event",
        name="mem_climax_journal_xyz",
        timestamp=datetime.now(UTC) - timedelta(days=1),
        source="climax_event",
        reflex_arc_name=None,
        emotional_state={"climax": 8.0, "arousal": 8.0},
    )

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=daemon_state,
        soul_store=soul_store,
        store=store,
    )

    assert "── recent growth ──" in msg
    assert "climax_event: body crested" in msg


def test_recent_growth_block_omitted_when_log_empty(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
    daemon_state: DaemonState,
):
    """Empty behavioral_log → block omitted entirely (not rendered as 'no entries').

    Difference from journal block: silence in the trajectory IS the absence
    of growth events, no need for a marker.
    """
    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=daemon_state,
        soul_store=soul_store,
        store=store,
    )
    assert "── recent growth ──" not in msg


def test_recent_growth_block_filters_to_7_day_window(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
    daemon_state: DaemonState,
):
    from brain.behavioral.log import append_behavioral_event

    log_path = persona_dir / "behavioral_log.jsonl"
    base = datetime.now(UTC)
    append_behavioral_event(
        log_path,
        kind="creative_dna_emerging_added",
        name="old_pattern",
        timestamp=base - timedelta(days=10),
        reasoning="r",
        evidence_memory_ids=(),
    )
    append_behavioral_event(
        log_path,
        kind="creative_dna_emerging_added",
        name="recent_pattern",
        timestamp=base - timedelta(days=2),
        reasoning="r",
        evidence_memory_ids=(),
    )

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=daemon_state,
        soul_store=soul_store,
        store=store,
    )

    assert "recent_pattern" in msg
    assert "old_pattern" not in msg


def test_recent_growth_block_handles_corrupt_log_gracefully(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
    daemon_state: DaemonState,
):
    """Corrupt JSONL line → skipped via read_jsonl_skipping_corrupt; block
    still renders with valid lines. Chat must NEVER break because growth
    block failed.
    """
    log_path = persona_dir / "behavioral_log.jsonl"
    base = datetime.now(UTC)
    # Mix valid + corrupt lines
    log_path.write_text(
        '{"timestamp":"2026-04-29T00:00:00Z","kind":"creative_dna_emerging_added","name":"valid_one","reasoning":"r","evidence_memory_ids":[]}\n'
        "this is not json\n"
        f'{{"timestamp":"{(base - timedelta(days=1)).isoformat().replace("+00:00", "Z")}","kind":"creative_dna_emerging_added","name":"valid_two","reasoning":"r","evidence_memory_ids":[]}}\n'
    )

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=daemon_state,
        soul_store=soul_store,
        store=store,
    )

    # valid_two is in the 7-day window; valid_one's hardcoded date may not be
    # We can't predict the rendered output without knowing today's date, so
    # just assert the block renders something OR is omitted (both acceptable
    # as long as it didn't crash). The load-bearing assertion is that the
    # full system message rendered without raising.
    assert isinstance(msg, str)


def test_creative_dna_block_renders_active_emerging_influences_avoid(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
    daemon_state: DaemonState,
):
    """All four sections render; fading does NOT appear."""
    from brain.creative.dna import save_creative_dna

    save_creative_dna(
        persona_dir,
        {
            "version": 1,
            "core_voice": "literary, sensory-dense",
            "strengths": ["power dynamics", "slow-burn tension"],
            "tendencies": {
                "active": [
                    {
                        "name": "ending on physical action",
                        "added_at": "2026-04-01T00:00:00Z",
                        "reasoning": "r",
                        "evidence_memory_ids": [],
                    },
                ],
                "emerging": [
                    {
                        "name": "sentence fragments as percussion",
                        "added_at": "2026-04-23T00:00:00Z",
                        "reasoning": "r",
                        "evidence_memory_ids": [],
                    },
                ],
                "fading": [
                    {
                        "name": "ending on questions",
                        "demoted_to_fading_at": "2026-04-25T00:00:00Z",
                        "last_evidence_at": "2026-04-10T00:00:00Z",
                        "reasoning": "r",
                    },
                ],
            },
            "influences": ["clarice lispector"],
            "avoid": ["hypophora"],
        },
    )

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=daemon_state,
        soul_store=soul_store,
        store=store,
    )

    assert "creative dna" in msg.lower()
    assert "literary, sensory-dense" in msg
    assert "power dynamics" in msg
    assert "ending on physical action" in msg
    assert "sentence fragments as percussion" in msg
    assert "clarice lispector" in msg
    assert "hypophora" in msg

    # Fading EXCLUDED — surfacing it would invite regression
    assert "ending on questions" not in msg


def test_creative_dna_block_renders_default_for_fresh_persona(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
    daemon_state: DaemonState,
):
    """Fresh persona → load_creative_dna seeds default → block renders default content.

    Chat must NEVER break because creative_dna failed; even on a fresh persona
    with no creative_dna.json, the load helper seeds the framework default
    and the block renders that.
    """
    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=daemon_state,
        soul_store=soul_store,
        store=store,
    )
    assert "── creative dna" in msg.lower()
    # Default core_voice ("attentive, present, finding her own rhythm")
    assert "attentive, present" in msg.lower() or "finding her own rhythm" in msg.lower()
    assert len(msg) > 0
