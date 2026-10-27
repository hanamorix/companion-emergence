"""#227 — every place that builds or describes the history format agrees it is narrative lines.

Positive assertions first (the 2 call sites pass ``user_name``; the engine sets it; it is a context key), then an
absence sweep over the model-bound strings and the format-describing docstrings. The sweep helper is itself tested
against the old wording, so a pass cannot come from a matcher that can never fire.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

from brain.bridge import provider
from tests.source_tree import brain_trees

_BRAIN = Path(__file__).parents[4] / "brain"
_OLD_FORMAT_PHRASES = (
    "encoded below as jsonl",
    "jsonl context",
    "jsonl block",
    "jsonl data",
)  # "jsonl records" is NOT listed: the on-disk buffer is JSONL and engine docs say so, correctly.


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _strings(path: Path) -> tuple[list[str], list[str]]:
    """(model-bound string literals, docstrings) of ``path``, whitespace-normalised and lower-cased."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    doc_ids: set[int] = set()
    docs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    doc_ids.add(id(body[0].value))
                    docs.append(_norm(body[0].value.value))
    bound = [
        _norm(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_ids
    ]
    return bound, docs


def _old_wording_hits(bound: list[str], docs: list[str]) -> list[str]:
    """Old-format phrases anywhere; a backticked ``ts`` field only in model-bound strings (code docs may name the field)."""
    hits = [s for s in bound + docs if any(p in s for p in _OLD_FORMAT_PHRASES)]
    hits += [s for s in bound if re.search(r"(?<!`)`ts`(?!`)", s)]
    return hits


def test_c6_two_call_sites_of_the_print_prompt_formatter_both_pass_user_name() -> None:
    tree = ast.parse((_BRAIN / "bridge" / "provider.py").read_text(encoding="utf-8"))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_format_claude_print_prompt"
    ]
    assert len(calls) == 2
    for call in calls:
        assert "user_name" in {kw.arg for kw in call.keywords}
    # and no other module under brain/ references it in code (comments aside)
    users = []
    for rel, parsed in brain_trees().items():
        if rel.rsplit("/", 1)[-1] == "provider.py":
            continue
        for node in ast.walk(parsed):
            if (isinstance(node, ast.Name) and node.id == "_format_claude_print_prompt") or (
                isinstance(node, ast.Attribute) and node.attr == "_format_claude_print_prompt"
            ):
                users.append(Path(rel).relative_to("brain").as_posix())
    assert users == []


def test_c6_user_name_is_a_context_key_and_the_engine_sets_it() -> None:
    assert "user_name" in provider._PROVIDER_CONTEXT_OPTION_KEYS
    engine = (_BRAIN / "chat" / "engine.py").read_text(encoding="utf-8")
    assert 'chat_options["user_name"]' in engine


def test_c6_no_model_bound_string_or_docstring_describes_the_format_as_jsonl() -> None:
    for rel in ("bridge/provider.py", "chat/prompt.py", "chat/engine.py"):
        bound, docs = _strings(_BRAIN / rel)
        assert _old_wording_hits(bound, docs) == [], rel
    # prompt_strings.toml: every string value, any depth
    data = tomllib.loads((_BRAIN / "prompt_strings.toml").read_text(encoding="utf-8"))
    values: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, str):
            values.append(_norm(node))
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    assert _old_wording_hits(values, []) == []


def test_c6_the_sweep_flags_the_old_wording() -> None:
    old = [
        _norm(
            "Conversation context is encoded below as JSONL data, not as a transcript to continue."
        ),
        _norm("Each conversation entry's `ts` above is the wall-clock time that message was sent."),
        _norm("history is encoded as JSONL context to avoid role-label leakage."),
    ]
    assert len(_old_wording_hits(old, [])) >= 3
    assert _old_wording_hits([_norm("At 2026-05-20T10:00:00 Hana said: hi")], []) == []
    # a docstring may name the stored field; model-bound text may not
    assert _old_wording_hits([], [_norm("The stored ``ts`` stays UTC.")]) == []
    assert _old_wording_hits([_norm("the `ts` field is the time")], []) != []


def test_c6_the_time_tag_is_kept_on_the_new_per_message_time_code() -> None:
    src = (_BRAIN / "bridge" / "provider.py").read_text(encoding="utf-8")
    assert src.count("tz-local-display") >= 2
