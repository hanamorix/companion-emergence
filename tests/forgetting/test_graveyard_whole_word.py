"""Whole-word forgotten-memory search (name-recall fix S102, S119, S121, S123,
S125, S128): matching, the four-letter minimum, the word filter, listed names,
order, and the shared rules. Synthetic entries only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.forgetting import graveyard
from brain.forgetting.graveyard import GRAVEYARD_FILENAME, query_words, search
from brain.forgetting.recall import search_with_loss
from brain.forgetting.tool import recall_forgotten
from brain.memory import known_names as kn
from brain.memory.store import MemoryStore, _to_fts_match


def _write(persona: Path, rows: list[tuple[str, str, str]]) -> None:
    """Rows are (memory_id, summary, domain); forgotten_at rises with the index."""
    lines = []
    for i, (mid, summary, domain) in enumerate(rows):
        lines.append(
            json.dumps(
                {
                    "memory_id": mid,
                    "summary": summary,
                    "domain": domain,
                    "forgotten_at_iso": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00",
                }
            )
        )
    (persona / GRAVEYARD_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ids(hits: list[dict]) -> list[str]:
    return [h["memory_id"] for h in hits]


# G1 whole word
def test_whole_word_only_rain_never_matches_training(tmp_path):
    _write(
        tmp_path,
        [("a", "the rain came", "chat"), ("b", "a training day", "chat"), ("c", "Rain, again.", "chat"),
         ("d", "no match here", "training"), ("e", "nothing at all", "rain"),
         ("f", "rain-soaked boots", "chat"), ("g", "rainy days and rain2", "chat")],
    )
    assert set(_ids(search(tmp_path, "rain"))) == {"a", "c", "e", "f"}


def test_words_under_four_characters_are_not_query_words(tmp_path):
    _write(tmp_path, [("a", "the ox pulled the cart", "chat"), ("b", "a dog barked", "chat")])
    assert search(tmp_path, "ox") == []
    assert search(tmp_path, "dog") == []
    assert query_words("ox dog not") == []


def test_four_letter_words_match(tmp_path):
    _write(tmp_path, [("a", "the boat left", "chat")])
    assert _ids(search(tmp_path, "boat")) == ["a"]


def test_the_split_is_the_keyword_searchs_split():
    # Every term the graveyard derives (4+ characters, before its stopword and
    # fragment filters) is a term the keyword search derives from the same string.
    for q in ["harbour-side, snake_case! boat2 (Jonas's) O'Brien", "north/south east:west", "ünïcode café"]:
        fts_terms = {t.strip('"').lower() for t in _to_fts_match(q).split(" OR ") if t}
        assert set(query_words(q)) <= {t for t in fts_terms if len(t) >= 4}
        assert {t for t in fts_terms if len(t) >= 4 and t not in {"one", "not"}} >= set(query_words(q))
    assert query_words("harbour-side snake_case boat2") == ["harbour", "side", "snake", "case", "boat2"]


def test_underscore_splits_like_the_keyword_search(tmp_path):
    _write(tmp_path, [("a", "the snake_case name", "chat")])
    assert _ids(search(tmp_path, "snake")) == ["a"]
    assert _ids(search(tmp_path, "case")) == ["a"]


def test_agrees_with_a_regular_memory_for_ascii_words_of_four_plus():
    from brain.memory.store import Memory

    texts = ["the rain came", "a training day", "rain-soaked boots", "snake_case name", "Harbour lights, again."]
    words = ["rain", "training", "boots", "snake", "case", "harbour", "lights", "again"]
    for text in texts:
        for word in words:
            store = MemoryStore(":memory:")
            store.create(Memory.create_new(content=text, memory_type="conversation", domain="us"))
            fts = bool(store.search_fts_scored(word, limit=5)) if hasattr(store, "search_fts_scored") else None
            if fts is None:
                pytest.skip("no scored FTS accessor")
            import tempfile

            with tempfile.TemporaryDirectory() as d:
                p = Path(d)
                _write(p, [("x", text, "chat")])
                assert bool(search(p, word)) == fts, (text, word)


# G3 any-word OR
def test_multiword_query_matches_any_kept_word(tmp_path):
    _write(tmp_path, [("a", "Henryk enjoys morning coffee", "chat"), ("b", "unrelated weather", "chat")])
    assert _ids(search(tmp_path, "Henryk preferences personality identity")) == ["a"]


# G4 phrase names, per field
def test_multiword_name_matches_as_a_phrase(tmp_path):
    _write(
        tmp_path,
        [("a", "moved to New York last year", "chat"), ("b", "a new dress on york street", "chat"),
         ("c", "the new-york skyline", "chat"), ("d", "york new", "chat"), ("e", "was new", "york")],
    )
    assert set(_ids(search(tmp_path, "", names=["new york"]))) == {"a", "c"}


def test_short_names_match_at_any_length_and_whole_word(tmp_path):
    _write(tmp_path, [("a", "Al waved from the pier", "chat"), ("b", "alone at the pier", "chat")])
    assert _ids(search(tmp_path, "", names=["al"])) == ["a"]
    assert _ids(search(tmp_path, "", names=["Al"])) == ["a"]


# G5 union and empties
def test_words_and_names_are_ored_and_empty_matches_nothing(tmp_path):
    _write(tmp_path, [("a", "the boat left", "chat"), ("b", "Al waved", "chat"), ("c", "nothing here", "chat")])
    assert set(_ids(search(tmp_path, "boat", names=["al"]))) == {"a", "b"}
    assert search(tmp_path, "", names=[]) == []
    assert search(tmp_path, "the and", names=[]) == []


# G7 stopwords, G8 fragments
def test_stopwords_are_not_query_words(tmp_path):
    rows = [("h", "the old harbour lights", "chat")] + [(f"t{i}", "about what did the thing", "chat") for i in range(10)]
    _write(tmp_path, rows)
    assert query_words("what did I know about the harbour") == ["know", "harbour"]
    assert _ids(search(tmp_path, "what did I know about the harbour")) == ["h"]
    assert search(tmp_path, "what did about the") == []


@pytest.mark.parametrize("apos", ["'", "’"])
def test_contraction_stems_are_not_query_words(apos):
    assert query_words(f"didn{apos}t doesn{apos}t couldn{apos}t know") == ["know"]
    assert query_words(f"DIDN{apos}T PANIC") == ["panic"]
    assert query_words(f"Bob{apos}ll know, they{apos}re gone, {apos}we{apos}ve") == ["know", "gone"]
    # kept: the possessive stem, o'clock
    assert "clock" in query_words(f"five o{apos}clock")
    assert query_words(f"Jonas{apos}s boat, Marcus{apos}s boat") == ["jonas", "boat", "marcus"]


def test_no_standalone_wins_exception(tmp_path):
    # S126: the stem is dropped even when the same word also appears standalone.
    assert query_words("Couldn came by. I couldn't know.") == ["came", "know"]


def test_dont_know_does_not_match_dont_entries(tmp_path):
    _write(tmp_path, [("a", "I don't remember the boat", "chat"), ("b", "don't wait up", "chat"), ("c", "Don's boat", "chat"),
                      ("d", "we know the way", "chat")])
    assert _ids(search(tmp_path, "I don't know")) == ["d"]


def test_a_listed_name_is_not_filtered_by_the_word_rules(tmp_path):
    _write(tmp_path, [("a", "Don's boat", "chat"), ("b", "don't wait up", "chat")])
    assert _ids(search(tmp_path, "I don't know", names=["don"])) == ["a"]  # S125 entry side: not "don't"


# G13 order
def test_name_hits_first_then_newest(tmp_path):
    rows = [("name-old", "Pretzel the cat", "chat")]
    rows += [(f"w{i}", "the harbour again", "chat") for i in range(20)]
    rows += [("both", "Pretzel at the harbour", "chat"), ("name-new", "another Pretzel day", "chat")]
    _write(tmp_path, rows)
    hits = _ids(search(tmp_path, "harbour", names=["pretzel"], limit=16))
    assert hits[:3] == ["name-new", "both", "name-old"]
    assert hits[3:] == [f"w{i}" for i in range(19, 19 - 13, -1)]


def test_the_limit_applies_after_the_name_partition(tmp_path):
    rows = [(f"n{i}", "Pretzel visits", "chat") for i in range(6)] + [(f"w{i}", "harbour again", "chat") for i in range(4)]
    _write(tmp_path, rows)
    assert _ids(search(tmp_path, "harbour", names=["pretzel"], limit=4)) == ["n5", "n4", "n3", "n2"]


def test_without_names_newest_first(tmp_path):
    _write(tmp_path, [("a", "harbour one", "chat"), ("b", "harbour two", "chat")])
    assert _ids(search(tmp_path, "harbour")) == ["b", "a"]


# G19 stopword-name entry side (S121)
@pytest.mark.parametrize(
    ("summary", "domain", "expected"),
    [
        ("saw Will today", "chat", True),
        ("Will you come", "chat", False),
        ("he left. Will you come", "chat", False),
        ("really! Will you", "chat", False),
        ("really? Will you", "chat", False),
        ("first line\nWill you", "chat", False),
        ("I will go", "chat", False),
        ("we said Will", "chat", True),
        ("Bob: Will you come", "chat", True),
        ('so he asked "Will you?"', "chat", True),
        ("nothing", "team Will", True),
        ("nothing", "Will", False),
        ("Hi, Will", "chat", True),  # a comma is not a sentence end (S104)
        ("\u0130\u0130\u0130 saw Will", "chat", True),  # F12: spans on the original text
        ("\u0130 was there. Will you", "chat", False),
    ],
)
def test_a_stopword_name_matches_only_capitalized_mid_sentence(tmp_path, summary, domain, expected):
    _write(tmp_path, [("a", summary, domain)])
    assert bool(search(tmp_path, "", names=["will"])) is expected


def test_a_non_stopword_name_matches_case_insensitively_at_a_sentence_start(tmp_path):
    _write(tmp_path, [("a", "pretzel is here", "chat"), ("b", "PRETZEL. Again", "chat")])
    assert set(_ids(search(tmp_path, "", names=["pretzel"]))) == {"a", "b"}


# G28 entry-side n't skip
def test_a_single_word_name_does_not_match_an_nt_stem(tmp_path):
    _write(tmp_path, [("a", "don't wait up", "chat"), ("b", "I don’t remember", "chat"), ("c", "Don waved", "chat"),
                      ("d", "Don's boat", "chat"), ("e", "DON'T", "chat"), ("f", "the don'ts", "chat")])
    assert set(_ids(search(tmp_path, "", names=["don"]))) == {"c", "d", "f"}


def test_a_multi_word_name_ignores_the_nt_skip(tmp_path):
    _write(tmp_path, [("a", "Don Juan's ship", "chat")])
    assert _ids(search(tmp_path, "", names=["don juan"])) == ["a"]


# G6 prefilter pin
def test_an_entry_with_no_substring_hit_is_never_split(tmp_path, monkeypatch):
    _write(tmp_path, [("a", "nothing relevant at all", "chat"), ("b", "the harbour at dusk", "chat")])
    calls: list[str] = []

    class Counting:
        def __init__(self, real):
            self._real = real

        def findall(self, text):
            calls.append(text)
            return self._real.findall(text)

        def finditer(self, text):
            calls.append(text)
            return self._real.finditer(text)

    monkeypatch.setattr(graveyard, "_WORD_RE", Counting(graveyard._WORD_RE))
    calls.clear()
    assert _ids(search(tmp_path, "harbour")) == ["b"]
    assert "the harbour at dusk" in calls
    assert "nothing relevant at all" not in calls


# G30 shared definition
def test_the_graveyard_follows_name_rules(monkeypatch, tmp_path):
    _write(tmp_path, [("a", "don't wait", "chat")])
    assert search(tmp_path, "", names=["don"]) == []
    monkeypatch.setattr("brain.forgetting.graveyard.followed_by_apostrophe_t", lambda *_a: False)
    assert _ids(search(tmp_path, "", names=["don"])) == ["a"]
    monkeypatch.setattr("brain.forgetting.graveyard.contraction_stems", lambda _t: frozenset())
    assert query_words("didn't know") == ["didn", "know"]
    # the stopword-name predicate is the shared one too (T2)
    _write(tmp_path, [("b", "Will you come", "chat")])
    assert search(tmp_path, "", names=["will"]) == []
    monkeypatch.setattr("brain.forgetting.graveyard.is_stopword_name", lambda _n: False)
    assert _ids(search(tmp_path, "", names=["will"])) == ["b"]


def test_one_word_split_everywhere():
    from brain.memory import name_rules

    assert graveyard._WORD_RE is name_rules.WORD_RE
    assert kn._WORD_RE is name_rules.WORD_RE


# G23 isolation, search_with_loss
def test_a_failing_graveyard_search_keeps_the_ranked_results(tmp_path, monkeypatch):
    from brain.memory.store import Memory

    store = MemoryStore(":memory:")
    store.create(Memory.create_new(content="harbour lights at dusk", memory_type="conversation", domain="us"))
    monkeypatch.setattr(graveyard, "search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    result = search_with_loss(tmp_path, store, "harbour")
    assert [m.content for m in result.active] == ["harbour lights at dusk"]
    assert result.lost == []


def test_search_with_loss_passes_names_and_searches_a_name_only_turn(tmp_path):
    store = MemoryStore(":memory:")
    _write(tmp_path, [("a", "Al waved", "chat")])
    result = search_with_loss(tmp_path, store, "", lost_names=["al"])
    assert _ids(result.lost) == ["a"] and result.active == [] and result.fading == []
    assert search_with_loss(tmp_path, store, "", lost_names=[]).lost == []


# G21 tool
def test_the_tool_is_whole_word_with_names_first(tmp_path):
    kn.admit_names(tmp_path, ["Al"], "tool")
    _write(tmp_path, [("a", "the rain came", "chat"), ("b", "a training day", "chat"), ("c", "Al waved", "chat"),
                      ("d", "alone at the pier", "chat")])
    assert _ids(recall_forgotten(arguments={"query": "rain"}, persona_dir=tmp_path)["hits"]) == ["a"]
    assert _ids(recall_forgotten(arguments={"query": "what about Al and rain"}, persona_dir=tmp_path)["hits"]) == ["c", "a"]
    assert recall_forgotten(arguments={"query": "AI"}, persona_dir=tmp_path)["hits"] == []
    assert recall_forgotten(arguments={"query": "won't"}, persona_dir=tmp_path)["hits"] == []


# G22 wording
def test_recall_forgotten_wording_is_byte_exact():
    from brain.tools.schemas import SCHEMAS

    d = SCHEMAS["recall_forgotten"]
    assert d["description"] == (
        "Reach back into the graveyard for memories you used to have but no longer do. "
        "Returns up to 5 most-recent hits whose summary or domain contains the query's words "
        "(whole words, not parts of words). Use this when you want to honestly acknowledge "
        "knowing-something-once: 'I had something about that, but it's gone now.' Each hit "
        "carries the summary the memory had when it faded, the lived-age at the moment of loss, "
        "and the reason the salience dropped enough to trigger the drop."
    )
    assert "substring" not in d["description"] and "—" not in d["description"]
    q = d["parameters"]["properties"]["query"]
    assert q["description"] == (
        "Words to look for in your forgotten memories. Whole words are matched, not parts of words."
    )
    assert d["parameters"]["required"] == ["query"] and set(d["parameters"]["properties"]) == {"query"}
    assert q["type"] == "string"


def test_a_name_only_turn_does_not_run_the_ranked_search(tmp_path, monkeypatch):
    store = MemoryStore(":memory:")
    _write(tmp_path, [("a", "Al waved", "chat")])
    called = []
    monkeypatch.setattr("brain.forgetting.recall.rank_memories", lambda *a, **k: called.append(1) or [])
    assert _ids(search_with_loss(tmp_path, store, "", lost_names=["al"]).lost) == ["a"]
    assert called == []


def test_a_failing_graveyard_search_is_logged(tmp_path, monkeypatch, caplog):
    store = MemoryStore(":memory:")
    monkeypatch.setattr(graveyard, "search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with caplog.at_level("ERROR"):
        assert search_with_loss(tmp_path, store, "harbour").lost == []
    assert "graveyard search failed" in caplog.text
