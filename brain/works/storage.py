"""brain.works.storage — markdown file I/O for one work.

Stores each work as `persona/<name>/data/works/<id>.md` with YAML
frontmatter mirroring the SQLite row, so the file is self-describing
if extracted from the index.

Atomic writes via brain.health.attempt_heal.save_with_backup_text. ID
is validated at read time to defend against path traversal even though
make_work_id guarantees hex-only by construction.
"""

from __future__ import annotations

import re
from pathlib import Path

from brain.health.attempt_heal import save_with_backup_text
from brain.works import Work

_ID_REGEX = re.compile(r"^[0-9a-f]{12}$")
_BACKUP_COUNT = 3


def _works_dir(persona_dir: Path) -> Path:
    return persona_dir / "data" / "works"


def _work_path(persona_dir: Path, work_id: str) -> Path:
    if not _ID_REGEX.fullmatch(work_id):
        raise ValueError(f"invalid work id {work_id!r} — must be 12 lowercase hex chars")
    return _works_dir(persona_dir) / f"{work_id}.md"


def _serialize_yaml_value(value: object) -> str:
    """Conservatively-quoting YAML serializer for the frontmatter we use.

    Audit 2026-05-07 P3-6: titles or summaries with newlines, ``---``,
    colons, leading dashes, brackets, or YAML-reserved scalars
    (``true``/``false``/``null``/``yes``/``no``/numeric-looking strings)
    used to corrupt the frontmatter or change its meaning. Now any
    string with such risky content gets emitted as a double-quoted
    scalar with proper escaping; multi-line strings use the literal
    block scalar form ``|`` so newlines round-trip cleanly.

    None becomes empty string so the key has no value (no literal "None").
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)

    # Multi-line → literal block scalar with two-space indent. The
    # parser side reads YAML naively (line-by-line key:value), but
    # any consumer using a real YAML parser (downstream tooling)
    # gets correct multi-line round-trip.
    if "\n" in value:
        indented = "\n".join(f"  {line}" for line in value.split("\n"))
        return f"|-\n{indented}"

    # Quote when the value would otherwise mis-parse as something
    # other than a plain string.
    risky = (
        not value
        or value[0] in "[](){}#&*!|>%@`,?-"
        or ":" in value
        or "---" in value
        or value.lower() in {"true", "false", "yes", "no", "null", "~"}
        or value.lstrip("-+").replace(".", "", 1).isdigit()
    )
    if risky:
        # Double-quoted scalar with backslash escaping.
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def write_markdown(persona_dir: Path, work: Work, *, content: str) -> Path:
    """Write a work as <persona_dir>/data/works/<id>.md with frontmatter.

    Returns the absolute path written. Creates the works/ subdir if missing.
    Atomic via the same pattern as brain.health.save_with_backup —
    concurrent writes won't tear and the prior version is preserved as .bak1.
    """
    dest = _work_path(persona_dir, work.id)
    dest.parent.mkdir(parents=True, exist_ok=True)

    fm_lines = [
        "---",
        f"id: {_serialize_yaml_value(work.id)}",
        f"title: {_serialize_yaml_value(work.title)}",
        f"type: {_serialize_yaml_value(work.type)}",
        f"created_at: {work.created_at.isoformat()}",
        f"session_id: {_serialize_yaml_value(work.session_id)}",
        f"word_count: {work.word_count}",
        f"summary: {_serialize_yaml_value(work.summary)}",
        "---",
        "",
        content,
    ]
    save_with_backup_text(dest, "\n".join(fm_lines), backup_count=_BACKUP_COUNT)
    return dest


def read_markdown(persona_dir: Path, work_id: str) -> tuple[Work, str]:
    """Read a work's frontmatter + content. Raises FileNotFoundError if missing.

    Returns (Work, content). Frontmatter parsing is permissive: keys with
    empty values become None; word_count is coerced to int.
    """
    path = _work_path(persona_dir, work_id)
    if not path.exists():
        raise FileNotFoundError(f"work not found: {path}")
    text = path.read_text(encoding="utf-8")
    return _parse_markdown(text)


def _parse_markdown(text: str) -> tuple[Work, str]:
    """Split a markdown-with-frontmatter string into (Work, content)."""
    if not text.startswith("---\n"):
        raise ValueError("missing frontmatter opening delimiter")
    rest = text[4:]
    end = rest.find("\n---\n")
    if end < 0:
        raise ValueError("missing frontmatter closing delimiter")
    fm_block = rest[:end]
    content = rest[end + 5 :]
    # Strip a single leading blank line between frontmatter and content
    if content.startswith("\n"):
        content = content[1:]

    fields: dict[str, str | None] = {}
    fm_lines = fm_block.split("\n")
    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]
        if ":" not in line:
            i += 1
            continue
        key, _, raw_value = line.partition(":")
        key = key.strip()
        value = raw_value.strip()

        # Audit 2026-05-07 P3-6: handle the new YAML serializer's
        # block scalar (|-) and double-quoted strings.
        if value == "|-":
            # Literal block scalar — collect indented continuation
            # lines until we hit a non-indented line.
            block_lines: list[str] = []
            i += 1
            while i < len(fm_lines):
                nxt = fm_lines[i]
                if nxt.startswith("  "):
                    block_lines.append(nxt[2:])
                    i += 1
                else:
                    break
            fields[key] = "\n".join(block_lines) if block_lines else None
            continue
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            # Double-quoted scalar — unescape \" and \\.
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        fields[key] = value if value else None
        i += 1

    required = {"id", "title", "type", "created_at", "word_count"}
    missing = required - {k for k, v in fields.items() if v is not None}
    if missing:
        raise ValueError(f"frontmatter missing required keys: {sorted(missing)}")

    word_count_raw = fields["word_count"]
    assert word_count_raw is not None  # required check above
    work = Work.from_dict(
        {
            "id": fields["id"],
            "title": fields["title"],
            "type": fields["type"],
            "created_at": fields["created_at"],
            "session_id": fields.get("session_id"),
            "word_count": int(word_count_raw),
            "summary": fields.get("summary"),
        }
    )
    return work, content
