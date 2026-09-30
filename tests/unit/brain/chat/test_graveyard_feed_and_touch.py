"""Passive recall's forgotten-memory ("lost") feed and grief touch (name-recall
fix S102, S119, S123, S125, S128): the names reach the graveyard, the word feed is
today's 10 words minus contraction fragments, hits lead with names, and a grief
touch fires only for a forgotten memory SHOWN in the "lost" section.

Drives the REAL `_build_recall_block`; semantic recall patched out (no model).
Synthetic entries only (user "Bob")."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

from brain.chat.prompt import _build_recall_block, _extract_recall_tokens, _legacy_capped_tokens
from brain.felt_time.state import FeltTimeState
from brain.felt_time.state import persist as persist_felt_time
from brain.forgetting import graveyard as gv
from brain.grief import policy
from brain.memory import known_names as kn
from brain.memory.store import Memory, MemoryStore


def _entry(mid: str, summary: str, i: int, *, joy: float = 9.0, domain: str = "memory") -> dict:
    return {
        "memory_id": mid,
        "summary": summary,
        "domain": domain,
        "memory_type": "episodic",
        "forgotten_at_iso": f"2026-01-01T01:{i // 60:02d}:{i % 60:02d}+00:00",
        "lived_age_hours_at_forgetting": 40.0,
        "emotion_at_ingest": {"joy": joy},
        "graveyard_reason": "test-seed",
    }


def _world(tmp_path: Path, entries: list[dict], *, store_texts=("a quiet filler line",)) -> MemoryStore:
    persist_felt_time(FeltTimeState(lived_age_hours=48.0), tmp_path)
    (tmp_path / gv.GRAVEYARD_FILENAME).write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    store = MemoryStore(":memory:")
    for t in store_texts:
        store.create(Memory.create_new(content=t, memory_type="conversation", domain="us"))
    return store


def _render(store, tmp_path, message, *, spy_touch=True):
    with patch("brain.chat.prompt.run_semantic_recall", return_value=None):
        if not spy_touch:
            return _build_recall_block(store, message, persona_dir=tmp_path), None
        with patch("brain.grief.handle_recall_touch") as touch:
            return _build_recall_block(store, message, persona_dir=tmp_path), touch


def _lost_lines(block: str) -> list[str]:
    out, on = [], False
    for ln in block.splitlines():
        if ln.strip().startswith("lost ("):
            on = True
            continue
        if on and ln.startswith("    - "):
            out.append(ln)
        elif on:
            break
    return out


def _spy_search(monkeypatch):
    calls: list[tuple[str, list[str]]] = []
    real = gv.search

    def spy(persona_dir, query, **kw):
        calls.append((query, list(kw.get("names", ()))))
        return real(persona_dir, query, **kw)

    monkeypatch.setattr("brain.forgetting.recall.graveyard.search", spy)
    return calls


# G8 passive: fragments are not fed
def test_dont_know_feeds_no_fragment_and_matches_no_dont_entry(tmp_path, monkeypatch):
    entries = [
        _entry("m1", "I don't recall", 1),
        _entry("m2", "don't wait up", 2),
        _entry("m3", "Don's boat", 3),
        _entry("m4", "we know home", 4),
    ]
    store = _world(tmp_path, entries, store_texts=("t is a letter", "a quiet know filler"))
    tokens = _extract_recall_tokens("I don't know", store)
    assert "don" in tokens, "precondition: the real selector hands the fragment to the graveyard feed"
    calls = _spy_search(monkeypatch)
    block, touch = _render(store, tmp_path, "I don't know")
    (query, names), = calls
    assert "don" not in query.split() and names == []
    lost = _lost_lines(block)
    assert len(lost) == 1 and "we know home" in lost[0]
    assert touch.call_args.kwargs["touched_ids"] == ["m4"]


def test_a_listed_name_equal_to_a_stem_does_not_flood_on_dont(tmp_path):
    kn.admit_names(tmp_path, ["Don"], "tool")
    entries = [_entry("m1", "I don't recall", 1), _entry("m2", "don't wait up", 2)]
    store = _world(tmp_path, entries)
    block, touch = _render(store, tmp_path, "I don't know")
    assert _lost_lines(block) == [] and touch.call_count == 0


def test_four_letter_stems_are_not_fed_on_the_passive_path_either(tmp_path):
    # T1: the stems that matter are 4+ characters (the floor drops `don` anyway).
    for message, stem, noisy in [
        ("I couldn't find it", "couldn", "she couldn't stop crying"),
        ("I didn't find it", "didn", "he didn't stop"),
        ("it doesn't find", "doesn", "it doesn't stop"),
    ]:
        sub = tmp_path / stem
        sub.mkdir()
        entries = [_entry("m1", noisy, 1), _entry("m2", "find the old boat", 2)]
        store = _world(sub, entries, store_texts=("a quiet filler line",))
        tokens = _extract_recall_tokens(message, store)
        assert stem in tokens, "precondition: the real selector hands the stem to the graveyard feed"
        block, touch = _render(store, sub, message)
        lost = _lost_lines(block)
        assert len(lost) == 1 and "find the" in lost[0], (message, lost)
        assert touch.call_args.kwargs["touched_ids"] == ["m2"]


# G10 names reach the graveyard; G11 name-only; G12 one call
def test_a_listed_name_reaches_the_graveyard_and_an_unlisted_one_does_not(tmp_path, monkeypatch):
    kn.admit_names(tmp_path, ["Al"], "tool")
    entries = [_entry("m1", "Al waved", 1), _entry("m2", "Bo waved from the pier", 2)]
    store = _world(tmp_path, entries)
    calls = _spy_search(monkeypatch)
    block, touch = _render(store, tmp_path, "did Al and Bo come by the harbour today")
    assert len(calls) == 1 and calls[0][1] == ["al"]
    lost = _lost_lines(block)
    assert len(lost) == 1 and "Al waved" in lost[0]
    assert touch.call_args.kwargs["touched_ids"] == ["m1"]


def test_a_name_only_message_searches_the_graveyard_once_with_the_name(tmp_path, monkeypatch):
    kn.admit_names(tmp_path, ["Al"], "tool")
    entries = [_entry("m1", "Al waved", 1), _entry("m2", "some long forgotten harbour afternoon", 2)]
    store = _world(tmp_path, entries)
    calls = _spy_search(monkeypatch)
    block, touch = _render(store, tmp_path, "Al")
    assert len(calls) == 1 and calls[0][1] == ["al"] and gv.query_words(calls[0][0]) == []
    lost = _lost_lines(block)
    assert len(lost) == 1 and "Al waved" in lost[0]
    assert touch.call_args.kwargs["touched_ids"] == ["m1"]


# G9: today's 10 words, no backfill
def test_word_feed_is_the_old_capped_selection_without_backfill(tmp_path, monkeypatch):
    rare = ["zephyr", "quokka", "marmot", "nutmeg", "obelisk", "pumice", "quartz", "rowan", "saffron"]
    common = ["harbour", "lantern"]
    texts = ["decoy don " + " ".join(rare)] + [f"{common[i % 2]} filler line {i} about nothing" for i in range(6)]
    message = "don't " + " ".join(rare + common)
    store = _world(tmp_path, [_entry("m1", "a summary mentioning the harbour once", 1)], store_texts=texts)
    tokens = _extract_recall_tokens(message, store)
    legacy = _legacy_capped_tokens(tokens)
    assert "don" in legacy and len(tokens) > len(legacy), "precondition: don inside the old top 10, the 11th beyond it"
    calls = _spy_search(monkeypatch)
    block, touch = _render(store, tmp_path, message)
    (query, _), = calls
    assert query.split() == [t for t in legacy if t != "don"], "minus the fragment, no 11th word pulled in"
    assert _lost_lines(block) == [] and touch.call_count == 0


# G13 order
def test_name_hits_lead_the_lost_section(tmp_path):
    kn.admit_names(tmp_path, ["Pretzel"], "tool")
    entries = [_entry("old-name", "Pretzel the cat", 0)] + [
        _entry(f"w{i}", "the harbour again", i + 1) for i in range(20)
    ]
    store = _world(tmp_path, entries)
    block, touch = _render(store, tmp_path, "Pretzel harbour")
    lost = _lost_lines(block)
    assert len(lost) == 8 and "Pretzel the cat" in lost[0]
    assert touch.call_args.kwargs["touched_ids"] == sorted(["old-name"] + [f"w{i}" for i in range(19, 13, -1)] + ["w13"])


# G14 shown only
def test_a_grief_touch_covers_only_the_entries_shown(tmp_path):
    entries = [_entry(f"m{i:02d}", f"harbour m{i:02d}", i) for i in range(12)]
    store = _world(tmp_path, entries)
    block, touch = _render(store, tmp_path, "harbour")
    lost = _lost_lines(block)
    assert len(lost) == 8
    assert touch.call_count == 1
    kw = touch.call_args.kwargs
    shown = sorted(f"m{i:02d}" for i in range(4, 12))
    assert kw["touched_ids"] == shown
    rendered = {re.search(r"m(\d\d)", ln).group(1) for ln in lost}
    assert rendered == {f"{i:02d}" for i in range(4, 12)}
    assert [e["memory_id"] for e in kw["graveyard_entries"]] == [f"m{i:02d}" for i in range(11, 3, -1)]


def test_a_duplicate_memory_id_is_scored_with_the_shown_row(tmp_path):
    entries = [_entry("dup", "harbour sad", 1, joy=1.0), _entry("dup", "harbour joy", 2, joy=9.0)]
    store = _world(tmp_path, entries)
    block, touch = _render(store, tmp_path, "harbour")
    kw = touch.call_args.kwargs
    assert kw["touched_ids"] == ["dup"] and [e["summary"] for e in kw["graveyard_entries"]] == ["harbour joy"]


# G15 real breadcrumb, non-vacuous precondition
def test_only_shown_forgotten_memories_write_grief(tmp_path):
    entries = [_entry(f"m{i:02d}", f"harbour m{i:02d}", i) for i in range(12)]
    store = _world(tmp_path, entries)
    _render(store, tmp_path, "harbour", spy_touch=False)
    for i in range(4, 12):
        assert store.exists_recent_grief_touch(f"m{i:02d}", hours=policy.DEBOUNCE_HOURS), i
    for i in range(0, 4):
        assert not store.exists_recent_grief_touch(f"m{i:02d}", hours=policy.DEBOUNCE_HOURS), i
    # precondition (the old behaviour): touching a below-cut entry DOES write a breadcrumb here
    from brain.felt_time.state import load_or_recover
    from brain.grief import handle_recall_touch

    felt, _ = load_or_recover(tmp_path)
    handle_recall_touch(
        touched_ids=["m00"], graveyard_entries=gv.read_all(tmp_path), persona_dir=tmp_path, store=store,
        lived_age_hours_now=felt.lived_age_hours,
    )
    assert store.exists_recent_grief_touch("m00", hours=policy.DEBOUNCE_HOURS)


# G16 once, never when nothing shown
def test_no_touch_when_nothing_is_shown(tmp_path):
    store = _world(tmp_path, [_entry("m1", "nothing relevant here", 1)])
    block, touch = _render(store, tmp_path, "harbour")
    assert touch.call_count == 0 and _lost_lines(block) == []


def test_the_touch_fires_once_on_a_conclusive_semantic_result_too(tmp_path):
    from brain.memory.semantic_recall import SemanticRecallResult

    store = _world(tmp_path, [_entry("m1", "harbour lights", 1)])
    sem_mem = Memory.create_new(content="the sunlit dock at low tide", memory_type="conversation", domain="us")
    store.create(sem_mem)
    result = SemanticRecallResult(full=[sem_mem], snippet=[], scores={sem_mem.id: 5.0})
    with patch("brain.chat.prompt.run_semantic_recall", return_value=result), patch(
        "brain.grief.handle_recall_touch"
    ) as touch:
        block = _build_recall_block(store, "harbour", persona_dir=tmp_path)
    assert touch.call_count == 1 and touch.call_args.kwargs["touched_ids"] == ["m1"]
    assert len(_lost_lines(block)) == 1


# G18 fault isolation + block unchanged
def test_a_failing_touch_does_not_break_the_block(tmp_path):
    store = _world(tmp_path, [_entry("m1", "harbour lights", 1)])
    with patch("brain.chat.prompt.run_semantic_recall", return_value=None), patch(
        "brain.grief.handle_recall_touch", side_effect=RuntimeError("boom")
    ):
        block = _build_recall_block(store, "harbour", persona_dir=tmp_path)
    assert len(_lost_lines(block)) == 1


def test_the_lost_section_render_is_unchanged(tmp_path):
    entries = [_entry(f"m{i}", f"harbour item {i}", i) for i in range(3)]
    store = _world(tmp_path, entries)
    block, _ = _render(store, tmp_path, "harbour")
    lines = block.splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.strip().startswith("lost ("))
    assert lines[at] == "  lost (no longer in active memory):"
    assert [ln.split('"')[1] for ln in lines[at + 1 : at + 4]] == [f"harbour item {i}" for i in (2, 1, 0)]
    assert all(ln.endswith("[forgotten — test-seed]") for ln in lines[at + 1 : at + 4])
