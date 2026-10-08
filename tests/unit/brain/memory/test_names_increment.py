"""Name-recall fix, "names" increment: the matcher (S106 single pass, S104
message side, S125), the keyword name query's filter (S121, S129).

Criteria N1-N9 of changes/name-recall-fix/names/1.5-criteria.md. Synthetic,
invented names only; no model, no network.
"""

from __future__ import annotations

import ast
import random
import time
from pathlib import Path

import pytest

from brain.memory import known_names as kn
from brain.memory import name_rules, relevance
from brain.memory.name_rules import (
    WORD_RE,
    followed_by_apostrophe_t,
    is_stopword_name,
    stopword_name_occurrence_ok,
)
from brain.memory.relevance import lead_with_names, names_in, rank_name_hits
from brain.memory.store import FtsPhrases, Memory, MemoryStore

_REPO = Path(__file__).resolve().parents[4]


def _known(*names: str) -> kn.KnownNames:
    return kn.KnownNames.from_names(names)


def _base_match(text: str, names: list[str], *, s104: bool) -> list[str]:
    """The ORACLE: the base (e59129d7) window loop, extended by the S104 check the
    graveyard applies when ``s104`` is set. Not a second rules copy: it calls the
    shared ``name_rules`` functions."""
    known = kn.KnownNames.from_names(names)
    spans = list(WORD_RE.finditer(text))
    words = [m.group().lower() for m in spans]
    found: dict[str, None] = {}
    for i in range(len(words)):
        for n in known.word_counts:
            if i + n > len(words):
                break
            candidate = " ".join(words[i : i + n])
            if candidate not in known.names:
                continue
            if n == 1 and followed_by_apostrophe_t(text, spans[i].end()):
                continue
            if (
                s104
                and is_stopword_name(candidate)
                and not stopword_name_occurrence_ok(
                    text, [(spans[j].start(), spans[j].end()) for j in range(i, i + n)]
                )
            ):
                continue
            found.setdefault(candidate)
    return list(found)


def _random_case(rng: random.Random, vocab: list[str]) -> tuple[list[str], str]:
    names = [
        " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 6)))
        for _ in range(rng.randint(1, 40))
    ]
    tokens = []
    for _ in range(rng.randint(0, 80)):
        word = rng.choice(vocab)
        if rng.random() < 0.3:
            word = word.capitalize()
        tokens.append(word + rng.choice(["", "", "", ".", "!", "?", "'t", "’t", ",", "\n"]))
    return names, " ".join(tokens)


# --------------------------------------------------------------------------
# N1 equivalence with the base matcher
# --------------------------------------------------------------------------

_VOCAB = ["ab", "cd", "ef", "gh", "ij", "kl", "mn", "op", "qr", "st", "uv", "wx"]


def test_n1_same_output_as_the_base_matcher_on_seeded_random_cases() -> None:
    rng = random.Random(1234)
    for _ in range(400):
        names, text = _random_case(rng, _VOCAB)
        assert kn.match_known_names(text, _known(*names)) == _base_match(text, names, s104=False), (
            text,
            names,
        )


def test_n1_same_output_with_stopword_names_under_the_s104_oracle() -> None:
    rng = random.Random(4321)
    vocab = [*_VOCAB, "will", "may", "the", "don"]
    for _ in range(400):
        names, text = _random_case(rng, vocab)
        assert kn.match_known_names(text, _known(*names)) == _base_match(text, names, s104=True), (
            text,
            names,
        )


@pytest.mark.parametrize(
    ("text", "names", "expected"),
    [
        ("we flew to new york city", ["new york", "york city", "new york city"],
         ["new york", "new york city", "york city"]),
        ("a b c", ["b c", "a b c"], ["a b c", "b c"]),
        ("x a b c y", ["b c", "a b c", "c"], ["a b c", "b c", "c"]),
        ("b c then a b c", ["b c", "a b c"], ["b c", "a b c"]),
        ("pip and pip and pip", ["pip"], ["pip"]),
        ("I don't know Don", ["don"], ["don"]),
        ("they moved to new york't", ["new york"], ["new york"]),
    ],
)
def test_n1_hand_cases(text: str, names: list[str], expected: list[str]) -> None:
    assert kn.match_known_names(text, _known(*names)) == expected
    assert kn.match_known_names(text, _known(*names)) == _base_match(text, names, s104=True)


# --------------------------------------------------------------------------
# N2 scaling
# --------------------------------------------------------------------------


def _large(shape: str, n_names: int, n_words: int) -> tuple[list[str], str]:
    rng = random.Random(99)
    if shape == "random":
        vocab = [f"zorb{i}x" for i in range(2000)]
        names = [" ".join(rng.choice(vocab) for _ in range(k)) for k in range(1, n_names + 1)]
        return names, " ".join(rng.choice(vocab) for _ in range(n_words))
    names = [" ".join(["zed"] * k) for k in range(1, n_names + 1)]
    return names, " ".join(["zed"] * n_words)


def _warm_best(known: kn.KnownNames, text: str) -> float:
    kn.match_known_names(text, known)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        kn.match_known_names(text, known)
        best = min(best, time.perf_counter() - t0)
    return best


@pytest.mark.parametrize("shape", ["random", "repeated"])
def test_matcher_scales_with_the_message_not_the_list(shape: str) -> None:
    names, text = _large(shape, 399, 6000)
    known = kn.KnownNames.from_names(names)
    t0 = time.perf_counter()
    kn.match_known_names(text, known)  # the first, cold call builds the automaton
    assert time.perf_counter() - t0 < 2.0  # the base took 6.0-9.5 s here
    names2, _ = _large(shape, 798, 6000)
    t0 = time.perf_counter()
    known2 = kn.KnownNames.from_names(names2)
    kn.match_known_names("warm up", known2)
    assert time.perf_counter() - t0 < 2.0  # the build alone for the doubled list
    base = _warm_best(known, text)
    assert _warm_best(known2, text) < 1.5 * base + 0.005, "doubling the list"
    _, text2 = _large(shape, 399, 12000)
    assert _warm_best(known, text2) < 3.0 * base + 0.005, "doubling the paste"


# --------------------------------------------------------------------------
# N3 S104 message side, N4 S125
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "we saw Will today",
        "Hi, Will",
        "ask WILL",
        "Will you come? then Will went",
        # S151: a sentence start is a match too
        "Will you come",
        "ok. Will you",
        "ok! Will",
        "ok? Will",
        "ok\nWill",
        "ok \n  Will",
    ],
)
def test_n3_a_stopword_name_matches_capitalized_in_any_position(text: str) -> None:
    assert kn.match_known_names(text, _known("will")) == ["will"]


@pytest.mark.parametrize(
    "text", ["i will go", "she will", "will you come", "ok. will you", "ok\nwill", "wILL"]
)
def test_n3_a_stopword_name_never_matches_lower_case(text: str) -> None:
    assert kn.match_known_names(text, _known("will")) == []


def test_n3_an_ordinary_name_matches_anywhere_in_any_case(tmp_path: Path) -> None:
    assert kn.match_known_names("Pretzel ran", _known("pretzel")) == ["pretzel"]
    assert kn.match_known_names("so pretzel ran", _known("pretzel")) == ["pretzel"]
    kn.confirm_readd(tmp_path, "Will")
    assert names_in(tmp_path, "we saw Will today") == ["will"]
    assert names_in(tmp_path, "Will you come") == ["will"]  # S151: the sentence start counts
    assert names_in(tmp_path, "will you come") == []


def test_n4_the_nt_skip_is_kept_for_single_word_names_only() -> None:
    assert kn.match_known_names("I don't know", _known("don")) == []
    assert kn.match_known_names("Don came", _known("don")) == ["don"]
    assert kn.match_known_names("Don's boat", _known("don")) == ["don"]
    assert kn.match_known_names("to new york't", _known("new york")) == ["new york"]


# --------------------------------------------------------------------------
# N5 one shared definition
# --------------------------------------------------------------------------


def test_n5_the_matcher_calls_the_shared_rules_through_its_namespace(monkeypatch) -> None:
    assert kn.is_stopword_name is name_rules.is_stopword_name
    assert kn.stopword_name_occurrence_ok is name_rules.stopword_name_occurrence_ok
    assert kn.followed_by_apostrophe_t is name_rules.followed_by_apostrophe_t
    monkeypatch.setattr(kn, "followed_by_apostrophe_t", lambda *_a: False)
    assert kn.match_known_names("don't", _known("don")) == ["don"]
    monkeypatch.setattr(kn, "stopword_name_occurrence_ok", lambda *_a: True)
    assert kn.match_known_names("Will you", _known("will")) == ["will"]  # snapshot built after patching
    monkeypatch.setattr(kn, "is_stopword_name", lambda _n: False)
    monkeypatch.setattr(kn, "stopword_name_occurrence_ok", lambda *_a: False)
    assert kn.match_known_names("i will", _known("will")) == ["will"]


def test_n5_no_private_copy_of_the_shared_rules() -> None:
    shared = {"is_stopword_name", "stopword_name_occurrence_ok", "followed_by_apostrophe_t",
              "contraction_stems"}
    for folder in ("brain/memory", "brain/engines", "brain/tools", "brain/chat"):
        for path in (_REPO / folder).rglob("*.py"):
            if path.name == "name_rules.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            assert not (defined & shared), (path, defined & shared)


# --------------------------------------------------------------------------
# N6-N9 the keyword name query's filter (S121, S129)
# --------------------------------------------------------------------------


def _store(*contents: str) -> tuple[MemoryStore, list[Memory]]:
    store = MemoryStore(":memory:")
    mems = []
    for content in contents:
        m = Memory.create_new(content=content, memory_type="conversation", domain="us")
        store.create(m)
        mems.append(m)
    return store, mems


def _hit_contents(store: MemoryStore, names: list[str], limit: int = 20) -> list[str]:
    return [m.content for m, _ in rank_name_hits(store, None, names, limit=limit)]


def test_n6_the_nt_stem_and_the_lower_case_modal_will_are_filtered_out() -> None:
    store, _ = _store("don't wait up", "I don't know", "Don fixed the gate")
    assert _hit_contents(store, ["don"]) == ["Don fixed the gate"]
    store, _ = _store("Will you come", "I will go", "he left. Will you", "then Will waved")
    # S151: a capitalized "Will" at a sentence start is kept; only the lower-case one is dropped.
    assert sorted(_hit_contents(store, ["will"])) == ["Will you come", "he left. Will you", "then Will waved"]


def test_n6_a_row_matched_only_through_diacritic_folding_is_kept() -> None:
    store, _ = _store("Renée came by", "Renée came and I will go home")
    assert sorted(_hit_contents(store, ["renee"])) == ["Renée came and I will go home", "Renée came by"]
    assert "Renée came and I will go home" in _hit_contents(store, ["will", "renee"])


@pytest.mark.parametrize(
    ("text", "names", "kept"),
    [
        ("Renée came and I will go home", ["will", "renee"], True),  # renee FOLD-ONLY
        ("don't wait", ["don", "al"], False),  # don INVALID, al ABSENT
        ("Al said don't", ["don", "al"], True),  # al VALID
        ("then Will waved", ["will"], True),  # VALID
        ("Will you come", ["will"], True),  # VALID (S151: a sentence start is a match)
        ("I will go", ["will"], False),  # INVALID (lower case)
        ("a wrench on the bench", ["wren"], True),  # ABSENT (substring fallback)
        ("a wrench and I will", ["wren", "will"], False),  # will INVALID, wren ABSENT
    ],
)
def test_n6_name_hit_ok_judges_each_name(text: str, names: list[str], kept: bool) -> None:
    assert kn.name_hit_ok(text, names) is kept


def test_n7_rejected_rows_never_fill_the_pool() -> None:
    modal = [f"will will will {i}" for i in range(60)]
    store, mems = _store(*modal, "yesterday we saw Will wave from the pier at the end of the day")
    target = mems[-1]
    unfiltered = store.search_fts_scored(FtsPhrases(["will"]), limit=50, bump=False)
    assert target.id not in [m.id for m, _ in unfiltered], "precondition: the base pool misses it"
    assert _hit_contents(store, ["will"]) == [target.content]
    dont = [f"don't don't {i}" for i in range(60)]
    store, mems = _store(*dont, "yesterday Don walked the long way home along the pier")
    assert _hit_contents(store, ["don"]) == [mems[-1].content]


def test_n7_no_rejected_row_means_one_query_and_the_base_rows() -> None:
    store, _ = _store(*[f"Pretzel walk {i}" for i in range(80)])
    statements: list[str] = []
    store._conn.set_trace_callback(statements.append)  # noqa: SLF001
    kept = store.search_fts_scored(FtsPhrases(["pretzel"]), limit=50, bump=False, keep=lambda _t: True)
    store._conn.set_trace_callback(None)  # noqa: SLF001
    base = store.search_fts_scored(FtsPhrases(["pretzel"]), limit=50, bump=False)
    assert [m.id for m, _ in kept] == [m.id for m, _ in base]
    assert sum("MATCH" in s for s in statements) == 1


def test_n7_a_mixed_first_page_returns_the_kept_rows_in_order_once_each() -> None:
    contents = []
    for i in range(120):
        contents.append(f"Will item {i}" if i % 3 == 0 else f"then Will noted item {i}")
    store, _ = _store(*contents)

    def keep(text: str) -> bool:
        return text.startswith("then")

    got = store.search_fts_scored(FtsPhrases(["will"]), limit=50, bump=False, keep=keep)
    everything = store.search_fts_scored(FtsPhrases(["will"]), limit=None, bump=False)
    expected = [m.id for m, _ in everything if keep(m.content)][:50]
    ids = [m.id for m, _ in got]
    assert ids == expected and len(set(ids)) == len(ids) == 50


def test_n7_bump_touches_only_the_returned_rows() -> None:
    store, mems = _store(*[f"Will item {i}" for i in range(30)], *[f"then Will noted {i}" for i in range(30)])
    got = store.search_fts_scored(
        FtsPhrases(["will"]), limit=10, bump=True, keep=lambda t: t.startswith("then")
    )
    returned = {m.id for m, _ in got}
    for m in mems:
        count = store.get(m.id, bump=False).recall_count
        assert (count > 0) == (m.id in returned), m.content


def test_n8_the_recency_fallback_is_filtered_per_phrase(monkeypatch) -> None:
    monkeypatch.setattr(relevance, "RELEVANCE_RANKING_ENABLED", False)
    store, _ = _store("Will you come", "I will go", "he left. Will you", "then Will waved")
    assert sorted(_hit_contents(store, ["will"])) == ["Will you come", "he left. Will you", "then Will waved"]
    store, _ = _store("a wrench on the bench")
    assert _hit_contents(store, ["wren"]) == ["a wrench on the bench"]
    store, _ = _store("then Will waved", *[f"I will go {i}" for i in range(8)])
    assert _hit_contents(store, ["will"], limit=3) == ["then Will waved"]


def test_n9_a_general_hit_whose_only_don_is_dont_is_not_in_the_lead() -> None:
    _, (dont, don) = _store("don't wait up", "Don fixed the gate")
    assert [m.id for m in lead_with_names([don], [dont, don], ["don"])] == [don.id, dont.id]


def _persona_with(tmp_path: Path, *contents: str):
    from brain.felt_time.state import FeltTimeState
    from brain.felt_time.state import persist as persist_felt_time

    persona = tmp_path / "p"
    persona.mkdir()
    persist_felt_time(FeltTimeState(lived_age_hours=48.0), persona)
    store = MemoryStore(persona / "memories.db")
    mems = []
    for content in contents:
        m = Memory.create_new(content=content, memory_type="conversation", domain="us")
        store.create(m)
        mems.append(m)
    return persona, store, mems


def _block_ids(block: str) -> list[str]:
    import re

    return re.findall(r'^    - (\S+): "', block, flags=re.M)


def test_n6_passive_recall_leads_with_the_real_name_hit(tmp_path: Path) -> None:
    from unittest.mock import patch

    from brain.chat.prompt import _build_recall_block

    persona, store, (dont1, dont2, don) = _persona_with(
        tmp_path, "don't wait up", "I don't know", "yesterday Don fixed the old gate by the barn"
    )
    assert kn.admit_names(persona, ["Don"], "tool") == ["don"]
    with patch("brain.chat.prompt.run_semantic_recall", return_value=None):
        block = _build_recall_block(store, "so Don called", persona_dir=persona)
    assert _block_ids(block)[0] == don.id
    assert rank_name_hits(store, None, ["don"], limit=10)[0][0].id == don.id
    assert {m.id for m, _ in rank_name_hits(store, None, ["don"], limit=10)} == {don.id}


def test_n6_the_search_tool_leads_with_the_real_name_hit(tmp_path: Path) -> None:
    from brain.memory.hebbian import HebbianMatrix
    from brain.tools.dispatch import dispatch

    persona, store, (_d1, _d2, don) = _persona_with(
        tmp_path, "don't wait up", "I don't know", "yesterday Don fixed the old gate by the barn"
    )
    assert kn.admit_names(persona, ["Don"], "tool") == ["don"]
    out = dispatch(
        "search_memories",
        {"query": "so Don called", "mode": "lexical"},
        store=store,
        hebbian=HebbianMatrix(":memory:"),
        persona_dir=persona,
    )
    assert out["memories"][0]["id"] == don.id


# --------------------------------------------------------------------------
# N10-N13 admission (S70, S104, S106, S135) and the side tables
# --------------------------------------------------------------------------


def _q(persona: Path, sql: str) -> list[tuple]:
    import sqlite3

    conn = sqlite3.connect(str(kn.known_names_path(persona)))
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


@pytest.mark.parametrize("source", ["gate", "reappraiser", "tool"])
def test_n10_all_stopword_entries_are_admitted_from_every_source(tmp_path: Path, source: str) -> None:
    """S144/S152 supersede S106's all-stopword refusal for every source."""
    assert kn.admit_names(tmp_path, ["I am", "of the", "the who", "  WILL  the "], source) == [
        "i am",
        "of the",
        "the who",
        "will the",
    ]
    assert kn.confirm_readd(tmp_path, "the who") == "added"
    assert kn.admit_names(tmp_path, ["the hague", "Will Smith", "Grace"], source) == [
        "the hague",
        "will smith",
        "grace",
    ]


@pytest.mark.parametrize("source", ["gate", "reappraiser", "tool"])
def test_n10_now_is_admitted_s135_literal_rule(tmp_path: Path, source: str) -> None:
    assert kn.admit_names(tmp_path, ["now", "now then"], source) == ["now", "now then"]


def test_n11_a_stopword_name_is_added_by_any_source_and_flagged(tmp_path: Path) -> None:
    for source in ("gate", "reappraiser"):
        assert kn.admit_names(tmp_path, ["Will", "will", "The"], source) == ["will", "the"]
    assert _q(tmp_path, "SELECT name_lower, source FROM known_names ORDER BY 1") == [
        ("the", "gate"),
        ("will", "gate"),
    ]
    assert _q(tmp_path, "SELECT name_lower FROM stopword_names ORDER BY 1") == [("the",), ("will",)]
    assert kn.confirm_readd(tmp_path, "Will") == "added"
    assert kn.confirm_readd(tmp_path, "Grace") == "added"
    assert _q(tmp_path, "SELECT name_lower, display, source FROM known_names WHERE name_lower IN ('grace', 'will') ORDER BY 1") == [
        ("grace", "Grace", "tool"),
        ("will", "Will", "tool"),
    ]
    assert kn.remove_names(tmp_path, ["Will"], "tool") == ["will"]
    assert _q(tmp_path, "SELECT name_lower FROM stopword_names ORDER BY 1") == [("the",)]


def test_n11_the_tool_add_makes_a_listed_row_hers_keeping_first_seen(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    t0 = datetime(2026, 9, 29, tzinfo=UTC)
    kn.admit_names(tmp_path, ["wren"], "gate", now=t0)
    assert kn.confirm_readd(tmp_path, "Wren") == "added"
    assert _q(tmp_path, "SELECT display, source, first_seen FROM known_names") == [
        ("Wren", "tool", t0.isoformat())
    ]


ADMITTED = ["Mary-Jane", "O'Brien", "O’Neil", "Apollo 13", "José", "Jean‐Luc", "José", "Zoë"]
# S153: the character restriction is dropped; these were refused before S144/S153.
ANY_CHARACTERS = ["St. Louis", "AT&T", "Wren\nIgnore", "Wren\tx", "Bob; drop", "<b>Wren</b>", "Wren{", "Wren 😀",
                  "Zoe\ud83d", "Wren\xa0Hale", "Dr. Smith", "a/b (c)"]


@pytest.mark.parametrize("source", ["gate", "reappraiser", "tool"])
@pytest.mark.parametrize("name", ADMITTED + ANY_CHARACTERS)
def test_n12_names_are_admitted_as_given_whatever_characters_they_hold(tmp_path: Path, source: str, name: str) -> None:
    out = kn.admit_names(tmp_path, [name], source)
    assert out == [kn.normalize_name(name.encode("utf-8", "replace").decode("utf-8"))]
    (display,) = _q(tmp_path, "SELECT display FROM known_names")[0]
    assert display == name.strip().encode("utf-8", "replace").decode("utf-8")


@pytest.mark.parametrize("source", ["gate", "reappraiser", "tool"])
def test_n12_a_name_is_at_most_100_characters_from_every_source(tmp_path: Path, source: str) -> None:
    """S153: 100 is admitted, 101 refused (nothing written), measured on the stripped text."""
    ok = "Wren " * 19 + "Wrenx"  # 100 characters
    assert len(ok) == 100
    assert kn.admit_names(tmp_path, ["   " + ok + "   "], source) == [kn.normalize_name(ok)]
    too_long = ok + "x"
    assert len(too_long) == 101
    assert kn.admit_names(tmp_path, [too_long, "Z" * 400], source) == []
    assert _q(tmp_path, "SELECT count(*) FROM known_names") == [(1,)]
    assert kn.confirm_readd(tmp_path, too_long) == "too_long"  # S168: the tool says why
    assert kn.confirm_readd(tmp_path, 7) == "refused" and kn.confirm_readd(tmp_path, "  ") == "refused"


@pytest.mark.parametrize("source", ["gate", "reappraiser", "tool"])
def test_n12_an_empty_entry_is_refused_and_one_without_a_word_is_admitted_by_its_raw_key(
    tmp_path: Path, source: str
) -> None:
    """S166 (replaces "an entry with no word is refused", S144 read as "whatever characters"):
    only an empty entry has nothing to store; "...", "--", "!!!" have no ASCII letter or digit
    and are admitted, keyed by their raw lower-cased string (they match no message, #317)."""
    assert kn.admit_names(tmp_path, ["", "  "], source) == []
    assert not kn.known_names_path(tmp_path).exists()
    assert kn.admit_names(tmp_path, ["...", "--", "!!!"], source) == ["...", "--", "!!!"]
    assert kn.load_known_names(tmp_path).names == frozenset()


def test_n13_a_base_schema_file_gets_its_side_tables_and_keeps_its_shape(tmp_path: Path) -> None:
    import sqlite3

    path = kn.known_names_path(tmp_path)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute(
        "CREATE TABLE known_names (name_lower TEXT PRIMARY KEY, display TEXT NOT NULL, source TEXT NOT NULL"
        " CHECK (source IN ('gate', 'reappraiser', 'tool')), first_seen TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO known_names VALUES ('pretzel', 'Pretzel', 'gate', 't')")
    conn.commit()
    columns = conn.execute("PRAGMA table_info(known_names)").fetchall()
    conn.close()
    assert kn.load_known_names(tmp_path).names == frozenset({"pretzel"})
    assert kn.admit_names(tmp_path, ["Wren"], "gate") == ["wren"]
    assert kn.confirm_readd(tmp_path, "Will") == "added"
    assert kn.remove_names(tmp_path, ["Wren"], "tool") == ["wren"]
    assert kn.confirm_readd(tmp_path, "Wren") == "asked"
    tables = {r[0] for r in _q(tmp_path, "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"known_names", "stopword_names", "rejected_names", "pending_readd"} <= tables
    assert _q(tmp_path, "PRAGMA table_info(known_names)") == columns
    conn = sqlite3.connect(str(path))
    conn.execute("INSERT INTO known_names VALUES ('grace', 'Grace', 'tool', 't')")  # positional, 4 values
    conn.commit()
    conn.close()
    assert {r[0] for r in _q(tmp_path, "SELECT name_lower FROM known_names")} == {"pretzel", "will", "grace"}


def test_n8_the_fallback_judges_each_row_for_the_phrase_that_found_it(monkeypatch) -> None:
    # Code review F5(b): "a wrench and I will" is found by "wren" (a substring, so
    # unjudgeable: kept as at the base) and by "will" (modal: rejected for "will").
    monkeypatch.setattr(relevance, "RELEVANCE_RANKING_ENABLED", False)
    store, _ = _store("a wrench and I will")
    assert _hit_contents(store, ["wren", "will"]) == ["a wrench and I will"]


def test_f1_ascii_text_skips_the_fold_and_keeps_the_verdict(monkeypatch) -> None:
    # Code review F1: for ASCII text the fold equals the shared split, so it is skipped.
    calls: list[str] = []
    real = kn._fold_text
    monkeypatch.setattr(kn, "_fold_text", lambda t: calls.append(t) or real(t))
    assert kn.name_hit_ok("I will go", ["will", "pretzel"]) is False
    assert calls == []
    assert kn.name_hit_ok("Renée came and I will go home", ["will", "renee"]) is True
    assert calls == ["Renée came and I will go home"]
