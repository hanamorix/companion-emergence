"""Tests for research_session parser."""
import random
import re
import time

import pytest

from brain.engines.research_session import parse_session_output


def _reply(notes="facts here", memory="I dug in today.", verdict="continue"):
    return f"NOTES:\n{notes}\n\nMEMORY:\n{memory}\n\nVERDICT:\n{verdict}\n"


def test_parses_three_sections():
    out = parse_session_output(_reply())
    assert out.notes == "facts here" and out.memory == "I dug in today."
    assert out.verdict == "continue" and not out.degraded and out.spawn_topics == ()


def test_close_verdict():
    assert parse_session_output(_reply(verdict="close")).verdict == "close"


def test_spawn_verdict_parses_and_caps_two():
    out = parse_session_output(_reply(verdict="spawn: topic a; topic b; topic c"))
    assert out.verdict == "continue"
    assert out.spawn_topics == ("topic a", "topic b")


def test_missing_markers_degrades_to_notes_only():
    out = parse_session_output("just prose, no markers at all")
    assert out.degraded and out.memory is None and out.verdict == "continue"
    assert out.notes == "just prose, no markers at all"


def test_case_insensitive_markers():
    out = parse_session_output("notes:\nn\n\nMemory:\nm\n\nverdict:\nclose")
    assert out.notes == "n" and out.memory == "m" and out.verdict == "close"


def test_unknown_verdict_treated_as_continue():
    assert parse_session_output(_reply(verdict="maybe??")).verdict == "continue"


def test_empty_memory_section_skips_memory():
    assert parse_session_output(_reply(memory="")).memory is None


# ---- #591: a marker and its text on one line ----------------------------------------------------------------------


def test_inline_memory_and_verdict_close():
    # The layout the model produced from 2026-09-11 (shape only; invented text).
    out = parse_session_output("NOTES:\nread two papers\nsecond line\nMEMORY: I felt it click today.\nVERDICT: close\n")
    assert out.memory == "I felt it click today."
    assert out.verdict == "close"
    assert out.notes == "read two papers\nsecond line"
    assert not out.degraded


def test_inline_verdict_spawn_keeps_topics_and_cap():
    out = parse_session_output("NOTES:\nn\nMEMORY: m\nVERDICT: spawn: topic a; topic b; topic c\n")
    assert out.verdict == "continue"
    assert out.spawn_topics == ("topic a", "topic b")
    assert out.memory == "m"


def test_all_three_markers_inline():
    out = parse_session_output("NOTES: first notes line\nsecond line\nMEMORY: the memory\nVERDICT: close")
    assert out.notes == "first notes line\nsecond line"
    assert out.memory == "the memory"
    assert out.verdict == "close"
    assert not out.degraded


def test_inline_memory_text_may_continue_on_following_lines():
    out = parse_session_output("NOTES:\nn\nMEMORY: line one\nline two\n\nline three\nVERDICT: continue\n")
    assert out.memory == "line one\nline two\n\nline three"


_MEMORY_LINES = [
    "**MEMORY:** t",
    "**MEMORY**: t",
    "__MEMORY:__ t",
    "__MEMORY__: t",
    "**MEMORY:**\nt",
    "**MEMORY**\nt",
    "__MEMORY__\nt",
]


@pytest.mark.parametrize("crlf", [False, True], ids=["lf", "crlf"])
@pytest.mark.parametrize("line", _MEMORY_LINES)
def test_bold_and_underscore_memory_markers(line, crlf):
    raw = f"NOTES:\nn\n{line}\nVERDICT: close\n"
    if crlf:
        raw = raw.replace("\n", "\r\n")
    out = parse_session_output(raw)
    assert out.memory == "t"
    assert out.verdict == "close"
    assert out.notes == "n"


@pytest.mark.parametrize("crlf", [False, True], ids=["lf", "crlf"])
def test_inline_markers_in_crlf_reply(crlf):
    raw = "NOTES:\r\nn\r\nMEMORY: m\r\nVERDICT: close\r\n" if crlf else "NOTES:\nn\nMEMORY: m\nVERDICT: close\n"
    out = parse_session_output(raw)
    assert (out.notes, out.memory, out.verdict) == ("n", "m", "close")


# ---- the case rule: colon forms any case, colon-less forms upper case only -----------------------------------------


@pytest.mark.parametrize(
    "line",
    ["MEMORY: t", "Memory: t", "memory: t", "**Memory:** t", "__memory:__ t", "**memory**: t"],
)
def test_colon_forms_are_recognised_in_any_case(line):
    out = parse_session_output(f"NOTES:\nn\n{line}\nVERDICT: close\n")
    assert out.memory == "t"
    assert out.verdict == "close"
    assert out.notes == "n"


def test_colon_form_verdict_in_any_case():
    out = parse_session_output("NOTES:\nn\nMEMORY: m\n**verdict**: close\n")
    assert out.memory == "m" and out.verdict == "close"
    out = parse_session_output("NOTES:\nn\nMEMORY: m\nVerdict: close\n")
    assert out.verdict == "close"


@pytest.mark.parametrize("line", ["**Memory**", "**memory**", "__Memory__", "MEMORY text", "## MEMORY", "VERDICT - x"])
def test_colon_less_forms_other_than_upper_case_bold_are_not_markers(line):
    # A real inline VERDICT marker is in the reply so the case also tells the old parser from the new one.
    out = parse_session_output(f"NOTES:\nn\n{line}\nt\nVERDICT: close\n")
    assert out.memory is None
    assert out.verdict == "close"
    assert line in out.notes and "\nt" in out.notes


def test_upper_case_colon_less_bold_marker_is_recognised():
    out = parse_session_output("NOTES:\nn\n**MEMORY**\nt\n**VERDICT**\nclose\n")
    assert (out.notes, out.memory, out.verdict) == ("n", "t", "close")


# ---- mentions are not markers ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mention",
    [
        "MEMORY's role in sleep is debated",
        "VERDICT- not enough evidence",
        "the MEMORY: field was empty",
        "a line that ends with VERDICT:",  # mid-line, ends with the marker word
        "  MEMORY: indented text",
        "- MEMORY: a bullet",
        "## MEMORY: heading",
    ],
)
def test_mentions_are_not_markers(mention):
    out = parse_session_output(f"NOTES:\nbefore\n{mention}\nafter\nMEMORY: real\nVERDICT: close\n")
    assert out.memory == "real"
    assert out.verdict == "close"
    assert out.notes == f"before\n{mention}\nafter"


def test_inline_line_of_a_name_that_has_a_bare_marker_is_not_a_marker():
    # A real inline VERDICT after the bare MEMORY makes the case bite; the MEMORY: line in the notes stays in notes.
    raw = "NOTES:\nbefore\nMEMORY: a mention in the notes\nafter\nMEMORY:\nthe real memory\nVERDICT: close\n"
    out = parse_session_output(raw)
    assert out.memory == "the real memory"
    assert out.verdict == "close"
    assert out.notes == "before\nMEMORY: a mention in the notes\nafter"


def test_inline_verdict_line_ahead_of_a_bare_memory_is_not_a_marker():
    raw = "NOTES:\nreview of trial\nVERDICT: guilty on all counts\nmore notes\n\nMEMORY:\nI felt it.\nVERDICT: close\n"
    out = parse_session_output(raw)
    assert out.notes == "review of trial\nVERDICT: guilty on all counts\nmore notes"
    assert out.memory == "I felt it."
    assert out.verdict == "close"


def test_two_inline_lines_of_one_name_only_the_last_splits():
    raw = "NOTES:\nn1\nMEMORY: first, a mention\nn2\nMEMORY: the real one\nVERDICT: close\n"
    out = parse_session_output(raw)
    assert out.notes == "n1\nMEMORY: first, a mention\nn2"
    assert out.memory == "the real one"


def test_prose_verdict_line_ahead_of_the_inline_memory_is_not_a_marker():
    # Order is chained through accepted inline markers too: a VERDICT line before the accepted MEMORY is prose.
    raw = "NOTES:\nmy notes line 1\nVerdict: mixed on the claims\nmy notes line 2\nmy notes line 3\nMEMORY: I felt it.\n"
    out = parse_session_output(raw)
    assert out.notes == "my notes line 1\nVerdict: mixed on the claims\nmy notes line 2\nmy notes line 3"
    assert out.memory == "I felt it."
    assert out.verdict == "continue"
    assert out.missing == ("VERDICT",)


def test_prose_notes_line_after_a_preamble_is_not_a_marker_and_loses_nothing():
    raw = "Today I read about X.\nIt was long.\nNotes: it was hard going\nmore\nMEMORY: m\nVERDICT: close\n"
    out = parse_session_output(raw)
    # NOTES is only a marker as the first non-blank line, so the reply is kept whole as degraded notes (as before).
    assert out.degraded and out.missing == ("MEMORY", "VERDICT")
    assert "Today I read about X." in out.notes and "Notes: it was hard going" in out.notes
    assert out.memory is None


def test_notes_marker_on_the_first_non_blank_line_splits():
    out = parse_session_output("\n\nNotes: it was hard going\nmore\nMEMORY: m\nVERDICT: close\n")
    assert not out.degraded
    assert out.notes == "it was hard going\nmore"
    assert (out.memory, out.verdict) == ("m", "close")


# ---- the accepted consequence of the case ruling (owner ruling 2026-10-28), pinned ------------------------------------


def test_title_case_prose_line_splits_the_notes_when_no_other_memory_marker_exists():
    # Accepted by the owner's ruling "case insensitive when followed by a colon": a notes line beginning "Memory:" is read as
    # the MEMORY marker in a reply with no other MEMORY marker, when it is the last such line and in prompt order.
    raw = "NOTES:\nthe notes\nMemory: consolidation happens in sleep\nand more prose\nVERDICT:\nclose\n"
    out = parse_session_output(raw)
    assert out.notes == "the notes"
    assert out.memory == "consolidation happens in sleep\nand more prose"
    assert out.verdict == "close"


def test_title_case_prose_line_does_not_split_when_a_bare_memory_marker_exists():
    raw = "NOTES:\nthe notes\nMemory: consolidation happens in sleep\nand more prose\nMEMORY:\nreal memory\nVERDICT: close\n"
    out = parse_session_output(raw)
    assert out.notes == "the notes\nMemory: consolidation happens in sleep\nand more prose"
    assert out.memory == "real memory"


def test_title_case_verdict_line_ahead_of_a_bare_memory_does_not_split():
    raw = "NOTES:\nthe notes\n**Verdict:** mixed\nmore\nMEMORY:\nreal memory\nVERDICT: close\n"
    out = parse_session_output(raw)
    assert out.notes == "the notes\n**Verdict:** mixed\nmore"
    assert out.memory == "real memory"
    assert out.verdict == "close"


# ---- a prose "Verdict:" line is never taken for the marker (a VERDICT section keeps only its first line) --------------------


def test_a_prose_verdict_line_does_not_end_the_notes_and_loses_nothing():
    raw = "NOTES:\nfinding one\nVerdict: this approach is promising\nfinding two\nfinding three\n"
    out = parse_session_output(raw)
    assert out.notes == "finding one\nVerdict: this approach is promising\nfinding two\nfinding three"
    assert out.missing == ("MEMORY", "VERDICT")  # recorded by the caller as an unread layout


def test_a_prose_verdict_line_does_not_end_a_bare_memory():
    raw = "NOTES:\nn\nMEMORY:\nA trial changed how I think.\nVerdict: the evidence convinced me.\nIt stayed with me.\n"
    out = parse_session_output(raw)
    assert out.memory == "A trial changed how I think.\nVerdict: the evidence convinced me.\nIt stayed with me."
    assert out.missing == ("VERDICT",)


@pytest.mark.parametrize("value", ["close", "continue", "spawn: a topic", "Close", "CONTINUE"])
def test_an_inline_verdict_must_be_one_the_prompt_offers(value):
    out = parse_session_output(f"NOTES:\nn\nMEMORY: m\nVerdict: {value}\n")
    assert out.missing == () and out.memory == "m"


@pytest.mark.parametrize("value", ["close.", "**close**", "maybe", "mixed", "this approach is promising"])
def test_an_inline_verdict_that_is_not_one_the_prompt_offers_is_text(value):
    out = parse_session_output(f"NOTES:\nn\nMEMORY: m\nVERDICT: {value}\n")
    assert out.missing == ("VERDICT",) and out.memory == f"m\nVERDICT: {value}"


def test_the_last_inline_verdict_line_that_is_a_verdict_is_the_marker():
    out = parse_session_output("NOTES:\nn\nMEMORY: m\nVERDICT: close\nVerdict: because it was done\n")
    assert out.verdict == "close" and out.memory == "m"


def test_an_inline_marker_after_a_later_bare_marker_is_not_a_marker():
    out = parse_session_output("NOTES:\nn\nVERDICT:\nclose\nMemory: afterthought")
    assert out.memory is None and out.verdict == "close" and out.missing == ("MEMORY",)


# ---- the missing-section record ---------------------------------------------------------------------------------------


def test_missing_lists_sections_without_text():
    assert parse_session_output(_reply()).missing == ()
    assert parse_session_output("NOTES:\nn\nMEMORY: m\nVERDICT: close").missing == ()
    assert parse_session_output("NOTES:\nn\nVERDICT: close").missing == ("MEMORY",)
    assert parse_session_output(_reply(memory="")).missing == ("MEMORY",)
    assert parse_session_output("NOTES:\nn\nMEMORY: m").missing == ("VERDICT",)
    assert parse_session_output("NOTES:\nn").missing == ("MEMORY", "VERDICT")
    assert parse_session_output("plain prose").missing == ("MEMORY", "VERDICT")
    assert parse_session_output(_reply(verdict="maybe??")).missing == ()  # an unknown value is not a missing section


# ---- preservation: nothing that parsed before parses differently --------------------------------------------------------

_OLD_MARKER = re.compile(r"^(NOTES|MEMORY|VERDICT):\s*$", re.IGNORECASE | re.MULTILINE)


def _old_parse(raw):
    """Test oracle: a frozen copy of the parser as it was before #591 (commit 3225c1b3), returning plain fields."""
    sections: dict[str, str] = {}
    matches = list(_OLD_MARKER.finditer(raw))
    for n, m in enumerate(matches):
        end = matches[n + 1].start() if n + 1 < len(matches) else len(raw)
        sections[m.group(1).upper()] = raw[m.end():end].strip()
    if "NOTES" not in sections:
        return (raw.strip(), None, "continue", (), True)
    memory = sections.get("MEMORY", "").strip() or None
    verdict_line = sections.get("VERDICT", "").strip().splitlines()
    verdict_raw = verdict_line[0].strip().lower() if verdict_line else ""
    spawn: tuple[str, ...] = ()
    verdict = "continue"
    if verdict_raw == "close":
        verdict = "close"
    elif verdict_raw.startswith("spawn:"):
        topics = [t.strip() for t in verdict_raw[len("spawn:"):].split(";")]
        spawn = tuple(t for t in topics if t)[:2]
    return (sections["NOTES"], memory, verdict, spawn, False)


def _fields(raw):
    out = parse_session_output(raw)
    return (out.notes, out.memory, out.verdict, out.spawn_topics, out.degraded)


_MARKER_WORDS = ("NOTES", "MEMORY", "VERDICT")
# Body lines that an inline/bold-aware parser might be tempted to read: every one must stay put when the reply carries a bare
# marker of that name (the first pool) and, in the second pool, lines that are not markers at all.
_TEMPTING_LINES = (
    "MEMORY: an upper-case inline line", "Memory: a title-case line", "memory: a lower-case line", "VERDICT: close",
    "Verdict: mixed", "VERDICT: spawn: a; b", "NOTES: an inline notes line", "**MEMORY:** bold text", "**VERDICT:**",
    "__NOTES:__ underscore text", "**MEMORY**", "__VERDICT__: x",
)
_PLAIN_LINES = (
    "plain prose line", "", "  ", "MEMORY's role is debated", "VERDICT- hmm", "the MEMORY: field", "x" * 80,
    "## MEMORY", "- VERDICT: bullet", "  indented NOTES: x", "close", "spawn: not a verdict here",
)


def _random_marker(rng, word):
    w = "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in word)
    return w + ":" + rng.choice(["", " ", "  ", "\t"])


def _gen_reply(rng, bare_words, pool):
    parts = []
    if rng.random() < 0.3:
        parts.append("preamble line")
    words = list(bare_words)
    rng.shuffle(words)
    if rng.random() < 0.3 and words:
        words.append(rng.choice(words))  # a duplicate bare marker
    for w in words:
        parts.append(_random_marker(rng, w))
        for _ in range(rng.randint(0, 4)):
            parts.append(rng.choice(pool))
    eol = rng.choice(["\n", "\r\n"])
    return eol.join(parts) + rng.choice(["", "\n", "\r\n", "\n\n"])


@pytest.mark.no_base_failure(
    reason="preservation guard: it pins that every reply carrying all three bare markers parses as it did before "
    "#591, so the old parser passes it by design"
)
def test_replies_with_all_three_bare_markers_parse_exactly_as_before():
    rng = random.Random(591)
    table = [
        _reply(),
        _reply(notes="a\nMEMORY: x\nb", memory="m\nVERDICT: close\nn", verdict="close"),
        "NOTES:\n\nMEMORY:\n\nVERDICT:\n",
        "memory:\nm\nNOTES:\nn\nverdict: \nclose",
    ]
    for raw in table:
        assert _fields(raw) == _old_parse(raw), raw
    pool = _TEMPTING_LINES + _PLAIN_LINES
    for _ in range(300):
        raw = _gen_reply(rng, _MARKER_WORDS, pool)
        assert _fields(raw) == _old_parse(raw), repr(raw)


@pytest.mark.no_base_failure(
    reason="preservation guard: replies missing a bare marker and holding no inline or bold line parse as before "
    "#591, so the old parser passes it by design"
)
def test_replies_missing_a_bare_marker_without_inline_lines_parse_exactly_as_before():
    rng = random.Random(5910)
    for _ in range(300):
        words = rng.sample(_MARKER_WORDS, rng.choice([0, 1, 2]))
        raw = _gen_reply(rng, words, _PLAIN_LINES)
        assert _fields(raw) == _old_parse(raw), repr(raw)


def test_a_missing_section_with_an_inline_line_of_it_splits_there():
    # The documented boundary of the preservation claim: this is the shape the fix exists to read.
    raw = "NOTES:\nn\nMEMORY: real\n\nVERDICT:\nclose"
    assert _old_parse(raw)[1] is None
    out = parse_session_output(raw)
    assert (out.notes, out.memory, out.verdict) == ("n", "real", "close")


# ---- totality ---------------------------------------------------------------------------------------------------------


@pytest.mark.no_base_failure(
    reason="totality guard: the old parser never raised on any string either, so it passes by design; the inputs include "
    "the fold-case names that made a draft of the new parser raise KeyError"
)
def test_parser_never_raises_and_stays_linear():
    # The bound is wall-clock because the failure it guards is super-linear time (a draft of the ordering rule took 39 s
    # on 40k lines); a linear parser needs well under half a second for any input below.
    inputs = [
        "",
        "NOTES:",
        "MEMORY: ",
        "**VERDICT:**",
        "NOTES:\nMEMORY: x\nVERDICT:",
        "\r",
        "a\rb\rMEMORY: x\r",
        "café ☃ \U0001f600\nNOTES:\né\nMEMORY: ☃\n",
        "NOTES:\nx\nMEMORY: a\nVERDİCT: close",
        "NOTES:\nx\nMEMORY: a\n**VERDİCT:** close",
        "VERDİCT:\nx\nMEMORY: a\nNOTES:\nz\nVERDICT: close",
        "NOTES:\nx\nMEMORY: a\nſ: x\nı: y",
    ]
    for raw in inputs:
        parse_session_output(raw)
    big = {
        "one 200k line": "MEMORY: " + "x" * 200_000,
        "200k white space": "NOTES:\n" + " " * 200_000 + "\nMEMORY: x",
        "20k marker-like lines": "\n".join(["MEMORY: x", "**VERDICT:** y", "NOTES:"] * 7000),
        "20k NOTES then 20k MEMORY": "NOTES:\n" * 20_000 + "MEMORY: x\n" * 20_000,
        "interleaved x5000": "NOTES:\nMEMORY: a\n**VERDICT:** b\nMEMORY:\n" * 5000,
        "100k bold lines": "NOTES:\n" + "**MEMORY**\n" * 100_000,
    }
    for name, raw in big.items():
        start = time.perf_counter()
        parse_session_output(raw)
        assert time.perf_counter() - start < 2.0, name


# ---- data safety: nothing the model wrote in the notes or the memory is lost ---------------------------------------------


@pytest.mark.parametrize("memory_marker", ["MEMORY: {first}", "**MEMORY:** {first}", "Memory: {first}", "__MEMORY:__ {first}"])
@pytest.mark.parametrize("crlf", [False, True], ids=["lf", "crlf"])
def test_every_notes_and_memory_line_survives_an_inline_split(memory_marker, crlf):
    rng = random.Random(7)
    for _ in range(50):
        notes_lines = [f"notes-{i}-{rng.randint(0, 999)}" for i in range(rng.randint(1, 5))]
        mem_first = f"first-{rng.randint(0, 999)}"
        mem_more = [f"mem-{i}-{rng.randint(0, 999)}" for i in range(rng.randint(0, 4))]
        lines = ["NOTES:", *notes_lines, memory_marker.format(first=mem_first), *mem_more, "VERDICT: close"]
        raw = ("\r\n" if crlf else "\n").join(lines) + "\n"
        out = parse_session_output(raw)
        norm = lambda s: s.replace("\r\n", "\n")  # noqa: E731
        assert norm(out.notes) == "\n".join(notes_lines)
        assert norm(out.memory) == "\n".join([mem_first, *mem_more])
        assert out.verdict == "close"


def test_documented_drops_are_exactly_these():
    # Text after the first line of a VERDICT section is dropped (the rule for a bare VERDICT, now equally for an inline one).
    out = parse_session_output("NOTES:\nn\nMEMORY: m\nVERDICT: close\na postscript the model added\n")
    assert out.verdict == "close"
    assert "postscript" not in out.notes and "postscript" not in (out.memory or "")
    # A verdict line inside the memory text, with no bare VERDICT, ends the memory there (second residual): the model
    # wrote a line that is exactly a verdict the prompt offers.
    out = parse_session_output("NOTES:\nn\nMEMORY: m1\nVERDICT: continue\nm2 is dropped\n")
    assert out.memory == "m1" and out.verdict == "continue"
    # A prose line that merely starts with the word is not a verdict, so nothing after it is lost.
    out = parse_session_output("NOTES:\nn\nMEMORY: m1\nVERDICT: x\nm2 is kept\n")
    assert out.memory == "m1\nVERDICT: x\nm2 is kept" and out.missing == ("VERDICT",)
    # Text before a BARE NOTES marker is dropped (today's rule, unchanged).
    out = parse_session_output("preamble\nNOTES:\nn\nMEMORY: m\nVERDICT: close")
    assert out.notes == "n" and "preamble" not in out.notes
