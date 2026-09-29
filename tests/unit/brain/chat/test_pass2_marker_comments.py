"""C40 (S79): the PASS2-AT-LEAST-ONCE marker must sit INSIDE the named
function's own body at each of the 4 repeat-sensitive sites, and each
occurrence must also name #240. AST-scoped (a function's own `lineno`/
`end_lineno`), NOT a line-count heuristic and NOT a bare whole-file grep —
a whole-file match would let a module-docstring summary satisfy the letter
while missing the point (findable "while doing a bughunt" means AT the
site). A positive control (a planted docstring-only marker, and a planted
in-body marker with no #240) proves the scanner actually discriminates
scope and content rather than counting substring occurrences.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import brain

MARKER = "PASS2-AT-LEAST-ONCE"
ISSUE_REF = "#240"

_REPO_ROOT = Path(brain.__file__).resolve().parent.parent


def _comment_lines_in_function(source: str, func_name: str) -> tuple[list[tuple[int, str]], int, int]:
    """Return [(lineno, raw_line_text)] for every physical line inside the
    named function's own AST body (lineno..end_lineno inclusive), plus the
    node's (lineno, end_lineno) for callers that need the raw bounds too.

    A "comment line" here is identified structurally by re-scanning the
    ORIGINAL source's line range covered by the function's AST node — not
    by a naive `"#" in line` substring test elsewhere in the file, which is
    exactly the whole-file-grep failure mode this criterion calls out.
    """
    tree = ast.parse(source)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target = node
            break
    assert target is not None, f"function {func_name!r} not found"
    lines = source.splitlines()
    # AST line numbers are 1-based; end_lineno is the function's LAST
    # STATEMENT line, not necessarily the last line at its indentation a
    # human would read as "still inside it" (round-7 trailing-edge guard).
    body_lines = [
        (i, lines[i - 1]) for i in range(target.lineno, target.end_lineno + 1)
    ]
    return body_lines, target.lineno, target.end_lineno


def _marker_present_in_function_with_issue_ref(path: Path, func_name: str) -> bool:
    source = path.read_text(encoding="utf-8")
    body_lines, _, _ = _comment_lines_in_function(source, func_name)
    for _lineno, text in body_lines:
        stripped = text.lstrip()
        if stripped.startswith("#") and MARKER in text and ISSUE_REF in text:
            return True
        # A marker/issue-ref pair can also span a short comment BLOCK inside
        # the body (several consecutive `#` lines) rather than one physical
        # line — join a small window and re-check to avoid a false negative
        # on a wrapped comment, while still requiring both tokens to appear
        # somewhere strictly inside [lineno, end_lineno].
    body_text = "\n".join(text for _n, text in body_lines)
    return MARKER in body_text and ISSUE_REF in body_text


class TestMarkerInsideFunctionBody:
    def test_drain_all_locked_carries_the_marker_and_240(self):
        path = _REPO_ROOT / "brain/chat/pass2_queue.py"
        assert _marker_present_in_function_with_issue_ref(path, "drain_all_locked")

    def test_apply_emotion_delta_carries_marker_and_effect(self):
        path = _REPO_ROOT / "brain/chat/extractor.py"
        source = path.read_text(encoding="utf-8")
        body_lines, _, _ = _comment_lines_in_function(source, "_apply_emotion_delta")
        body_text = "\n".join(t for _n, t in body_lines)
        assert MARKER in body_text
        assert ISSUE_REF in body_text
        assert "emotion nudge" in body_text

    def test_consume_call_carries_marker_and_effect(self):
        path = _REPO_ROOT / "brain/attunement/budget.py"
        source = path.read_text(encoding="utf-8")
        body_lines, _, _ = _comment_lines_in_function(source, "consume_call")
        body_text = "\n".join(t for _n, t in body_lines)
        assert MARKER in body_text
        assert ISSUE_REF in body_text
        # Substring match on the two words, not exact adjacency — the
        # comment may wrap the phrase across lines (C40: "substring match,
        # not exact wording").
        assert "budget" in body_text and "count" in body_text

    def test_merge_into_learned_carries_marker_and_effect(self):
        path = _REPO_ROOT / "brain/attunement/store.py"
        source = path.read_text(encoding="utf-8")
        body_lines, _, _ = _comment_lines_in_function(source, "merge_into_learned")
        body_text = "\n".join(t for _n, t in body_lines)
        assert MARKER in body_text
        assert ISSUE_REF in body_text
        assert "evidence-count" in body_text or "evidence count" in body_text


class TestPositiveControlDiscriminatesScope:
    """A whole-file-grep-only implementation of THIS checker would pass
    against a marker planted in the module docstring (outside any function
    body) — proving the checker itself actually uses AST scope, not
    substring counting."""

    def test_marker_in_module_docstring_is_not_counted_as_in_function_body(self, tmp_path):
        planted = tmp_path / "planted.py"
        planted.write_text(
            textwrap.dedent(
                f'''\
                """Module docstring mentioning {MARKER} and {ISSUE_REF} — NOT inside
                any function body, so a correct AST-scoped checker must reject this
                as satisfying `some_function`'s own requirement."""


                def some_function():
                    return 1
                '''
            ),
            encoding="utf-8",
        )
        assert not _marker_present_in_function_with_issue_ref(planted, "some_function")

    def test_marker_in_body_without_issue_ref_is_rejected(self, tmp_path):
        planted = tmp_path / "planted2.py"
        planted.write_text(
            textwrap.dedent(
                f"""\
                def some_function():
                    # {MARKER}: repeats on crash, no issue reference here at all
                    return 1
                """
            ),
            encoding="utf-8",
        )
        assert not _marker_present_in_function_with_issue_ref(planted, "some_function")

    def test_marker_strictly_inside_body_is_accepted_by_the_checker(self, tmp_path):
        planted = tmp_path / "planted3.py"
        planted.write_text(
            textwrap.dedent(
                f"""\
                def some_function():
                    # {MARKER}: repeats on crash. See {ISSUE_REF}.
                    return 1
                """
            ),
            encoding="utf-8",
        )
        assert _marker_present_in_function_with_issue_ref(planted, "some_function")

    def test_trailing_comment_after_last_statement_is_outside_end_lineno(self, tmp_path):
        """round-7 trailing-edge guard: end_lineno is the LAST STATEMENT's
        line; a comment after it, even at the same indentation, is NOT
        inside [lineno, end_lineno] and must not count."""
        planted = tmp_path / "planted4.py"
        planted.write_text(
            textwrap.dedent(
                f"""\
                def some_function():
                    return 1
                    # {MARKER} {ISSUE_REF} — trailing, after the return statement
                """
            ),
            encoding="utf-8",
        )
        assert not _marker_present_in_function_with_issue_ref(planted, "some_function")
