"""MCP tool: list_open_arcs + recall_arc."""

from __future__ import annotations

from pathlib import Path

from brain.narrative_memory.arc import Arc, ArcMember
from brain.narrative_memory.state import ArcsState, append_event, save_state
from brain.narrative_memory.tool import list_open_arcs, recall_arc


def _arc(
    arc_id: str,
    title: str,
    state: str = "open",
    last_extended_iso: str = "2026-05-19T10:00:00+00:00",
    member_count: int = 1,
) -> Arc:
    members = tuple(
        ArcMember(
            memory_id=f"mem_{arc_id}_{i}",
            joined_at_iso=last_extended_iso,
            lived_age_at_join=412.0,
            salience_at_join=0.7,
        )
        for i in range(member_count)
    )
    return Arc(
        id=arc_id,
        state=state,
        seed_anchor_type="dream",
        seed_anchor_ref=f"dream_{arc_id}",
        seed_memory_ids=(f"mem_{arc_id}_0",),
        title=title,
        opened_at_iso="2026-05-15T10:00:00+00:00",
        lived_age_at_open=412.0,
        last_extended_at_iso=last_extended_iso,
        closed_at_iso="2026-05-22T10:00:00+00:00" if state == "closed" else None,
        lived_age_at_close=484.7 if state == "closed" else None,
        members=members,
    )


def test_list_open_arcs_returns_open_and_recently_closed(tmp_path: Path):
    save_state(
        tmp_path,
        ArcsState(
            open={"arc_1": _arc("arc_1", "the boat one", member_count=8)},
            recently_closed=[_arc("arc_old", "the prior one", state="closed", member_count=14)],
            last_pass_ts_iso="2026-05-19T11:00:00+00:00",
        ),
    )
    result = list_open_arcs(persona_dir=tmp_path)
    assert len(result["open"]) == 1
    assert result["open"][0]["title"] == "the boat one"
    assert result["open"][0]["member_count"] == 8
    assert len(result["recently_closed"]) == 1
    assert result["recently_closed"][0]["title"] == "the prior one"
    assert result["recently_closed"][0]["final_member_count"] == 14


def test_list_open_arcs_cold_start_returns_empty(tmp_path: Path):
    result = list_open_arcs(persona_dir=tmp_path)
    assert result == {"open": [], "recently_closed": []}


def test_recall_arc_exact_id_match(tmp_path: Path):
    save_state(
        tmp_path,
        ArcsState(
            open={
                "arc_20260519_a1b2c3d4": _arc(
                    "arc_20260519_a1b2c3d4", "the boat one", member_count=2
                )
            },
            last_pass_ts_iso="2026-05-19T11:00:00+00:00",
        ),
    )
    result = recall_arc(query="arc_20260519_a1b2c3d4", persona_dir=tmp_path)
    assert result["match_type"] == "exact"
    assert result["arc"]["id"] == "arc_20260519_a1b2c3d4"
    assert result["arc"]["title"] == "the boat one"
    assert len(result["arc"]["members"]) == 2


def test_recall_arc_substring_match_single(tmp_path: Path):
    save_state(
        tmp_path,
        ArcsState(
            open={"arc_1": _arc("arc_1", "the boat one")},
            last_pass_ts_iso="2026-05-19T11:00:00+00:00",
        ),
    )
    result = recall_arc(query="boat", persona_dir=tmp_path)
    assert result["match_type"] == "exact"
    assert result["arc"]["title"] == "the boat one"


def test_recall_arc_substring_match_multiple_returns_top_3(tmp_path: Path):
    save_state(
        tmp_path,
        ArcsState(
            open={
                "arc_1": _arc(
                    "arc_1", "the boat one", last_extended_iso="2026-05-19T10:00:00+00:00"
                ),
                "arc_2": _arc(
                    "arc_2", "the boat dream", last_extended_iso="2026-05-19T11:00:00+00:00"
                ),
                "arc_3": _arc(
                    "arc_3", "boat journey", last_extended_iso="2026-05-19T09:00:00+00:00"
                ),
                "arc_4": _arc(
                    "arc_4", "old boat thoughts", last_extended_iso="2026-05-19T08:00:00+00:00"
                ),
            },
            last_pass_ts_iso="2026-05-19T11:00:00+00:00",
        ),
    )
    result = recall_arc(query="boat", persona_dir=tmp_path)
    assert result["match_type"] == "multiple"
    assert len(result["arcs"]) == 3
    # Sorted by last_extended desc — arc_2 first
    assert result["arcs"][0]["title"] == "the boat dream"


def test_recall_arc_no_match_falls_back_to_log_scan(tmp_path: Path):
    """If state has no match, scan arcs.log.jsonl for older closed arcs."""
    # Old arc only in log, not in state.recently_closed
    append_event(
        tmp_path,
        {
            "event": "arc_opened",
            "arc_id": "arc_ancient",
            "seed_anchor_type": "dream",
            "seed_anchor_ref": "dream_old",
            "seed_memory_ids": ["mem_old"],
            "title": "ancient dream-arc",
            "ts_iso": "2026-01-01T00:00:00+00:00",
            "lived_age_hours": 100.0,
        },
    )
    append_event(
        tmp_path,
        {
            "event": "member_added",
            "arc_id": "arc_ancient",
            "memory_id": "mem_old",
            "ts_iso": "2026-01-01T00:00:00+00:00",
            "lived_age_hours": 100.0,
            "salience_at_join": 0.5,
            "via": "seed",
        },
    )
    append_event(
        tmp_path,
        {
            "event": "arc_closed",
            "arc_id": "arc_ancient",
            "ts_iso": "2026-01-04T00:00:00+00:00",
            "lived_age_hours": 172.0,
            "reason": "stale_72h",
            "final_member_count": 1,
        },
    )
    # State has no recently_closed (cap evicted it)
    save_state(tmp_path, ArcsState(last_pass_ts_iso="2026-05-19T11:00:00+00:00"))

    result = recall_arc(query="ancient", persona_dir=tmp_path)
    assert result["match_type"] in {"exact", "multiple", "log"}
    if result["match_type"] == "log":
        assert any(a["title"] == "ancient dream-arc" for a in result["arcs"])
    else:
        # Even single log-fallback hits can report as exact via title substring
        assert result.get("arc", {}).get("title") == "ancient dream-arc" or any(
            a["title"] == "ancient dream-arc" for a in result.get("arcs", [])
        )
