#!/usr/bin/env python3
"""Leak guard — multi-arm scanner for the public repo boundary.

The one lasting piece of the public-sync re-root design
(docs/guarded-change/public-sync-reroot/2-plan.md step 7). Engine behind
hooks/pre-push (local, plaintext rules) and CI marker-scan (hashed
fingerprints). Stdlib only: runs as a bare git hook, no venv.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import subprocess
import unicodedata
from pathlib import Path

_PUNCT_RE = re.compile(r"[^\w\s/]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFC → strip zero-width/format chars → casefold → fold \\ to / →
    strip punctuation (path separators kept) → collapse whitespace.

    Zero-width strip (red-team m2): U+200B-class chars would otherwise split
    tokens and evade matching. Backslash fold (red-team m1): Windows-style
    paths must hit path-class fingerprints."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = text.casefold()
    text = text.replace("\\", "/")
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _hmac_hex(salt_hex: str, message: str) -> str:
    return hmac.new(bytes.fromhex(salt_hex), message.encode("utf-8"), hashlib.sha256).hexdigest()


def build_fingerprint(marker: str, *, salt_hex: str, kind: str = "words") -> dict:
    """Fingerprint one marker. kind="words" → word n-gram; kind="path" → segment prefix."""
    norm = normalize(marker)
    if kind == "path":
        token = norm.replace(" ", "")
        return {"h": _hmac_hex(salt_hex, token), "kind": "path", "segs": token.count("/")}
    n = len(norm.split())
    return {"h": _hmac_hex(salt_hex, norm), "kind": "words", "n": n}


def _line_word_ngrams(line: str, n: int):
    words = normalize(line).split()
    for i in range(len(words) - n + 1):
        yield " ".join(words[i : i + n])


def _line_path_prefixes(line: str, segs: int):
    """Yield leading path prefixes with `segs` separators from path-like tokens."""
    for token in normalize(line).split():
        if "/" not in token:
            continue
        parts = token.split("/")
        if len(parts) - 1 >= segs:
            yield "/".join(parts[: segs + 1])


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def check_path(path: str, denied_prefixes: list[str]) -> str | None:
    folded = path.casefold()
    for prefix in denied_prefixes:
        if folded.startswith(prefix.casefold()):
            return f"path under retired private tree: {path} (rule {prefix})"
    return None


def check_binary(path: str, allowed_prefixes: list[str]) -> str | None:
    if Path(path).suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    for prefix in allowed_prefixes:
        if path.startswith(prefix):
            return None
    return f"image outside allowed paths: {path}"


def check_identity(*, author: str, committer: str, allowlist: list[str]) -> tuple[bool, str]:
    """Both identities must be allowlisted. `committer-only:` entries never
    validate an author (N5 — GitHub's web-merge committer). `glob:` entries
    match the email part."""
    import fnmatch

    def _allowed(identity: str, as_author: bool) -> bool:
        email_match = re.search(r"<([^>]*)>", identity)
        email = email_match.group(1) if email_match else ""
        for entry in allowlist:
            if entry.startswith("committer-only:"):
                if as_author:
                    continue
                entry = entry[len("committer-only:"):]
            if entry.startswith("glob:"):
                if fnmatch.fnmatch(email, entry[len("glob:"):]):
                    return True
            elif entry == identity:
                return True
        return False

    if not _allowed(author, as_author=True):
        return False, f"author not allowlisted: {author}"
    if not _allowed(committer, as_author=False):
        return False, f"committer not allowlisted: {committer}"
    return True, ""


def load_plaintext_rules(path) -> list[tuple[str, object]]:
    """Parse leak-rules.local: `substr:<literal>` / `re:<regex>` lines, # comments.
    substr rules match against the NORMALIZED line (punctuation-proof); re rules
    see the raw line."""
    rules: list[tuple[str, object]] = []
    path = Path(path)
    if not path.exists():
        return rules
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("substr:"):
            rules.append(("substr", normalize(line[len("substr:"):])))
        elif line.startswith("re:"):
            rules.append(("re", re.compile(line[len("re:"):], re.IGNORECASE)))
    return rules


def scan_line_plaintext(line: str, rules: list[tuple[str, object]]) -> bool:
    norm = normalize(line)
    for kind, rule in rules:
        if kind == "substr" and rule in norm:
            return True
        if kind == "re" and rule.search(line):
            return True
    return False


def scan_line_fingerprints(line: str, entries: list[dict], *, salt_hex: str) -> bool:
    """True if any fingerprint entry matches the line (N4: path kind matches
    a marker that is a PREFIX of a longer token, e.g. /Users/x inside
    /Users/x/deeper)."""
    for entry in entries:
        if entry.get("kind") == "path":
            candidates = _line_path_prefixes(line, entry["segs"])
        else:
            candidates = _line_word_ngrams(line, entry["n"])
        for candidate in candidates:
            if hmac.compare_digest(_hmac_hex(salt_hex, candidate), entry["h"]):
                return True
    return False


# ------------------------------------------------------------ range scanning

def _load_fingerprints(fingerprint_file):
    if fingerprint_file is None or not Path(fingerprint_file).exists():
        return None, []
    data = json.loads(Path(fingerprint_file).read_text(encoding="utf-8"))
    return data.get("salt"), data.get("entries", [])


def _git(args: list[str], cwd: str | None = None) -> str:
    # errors="replace": diffs can contain raw binary (e.g. image blobs) — the
    # guard must scan around them, not crash on them.
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        check=True,
    ).stdout


def _list_file(path) -> list[str]:
    if path is None or not Path(path).exists():
        return []
    return [
        ln.strip()
        for ln in Path(path).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def scan_diff_text(diff_text: str, *, plain_rules, salt, entries) -> list[dict]:
    """Content arm over unified-diff text. Findings never echo the matched
    content back — the match may BE the secret."""
    findings = []
    current_file = "?"
    previous = ""
    for line in diff_text.splitlines():
        # A real `+++` file header only follows a real OLD-FILE header
        # (`--- a/…` or `--- /dev/null`) or a diff/index line. Red-team B1: an
        # ADDED line starting `++` must still be scanned; re-review NEW-1: a
        # REMOVED content line starting `--` renders as `--- …` and must NOT
        # shield the next `+++` line.
        is_header = line.startswith("+++ ") and (
            previous.startswith("--- a/")
            or previous.startswith('--- "a/')
            or previous == "--- /dev/null"
            or previous.startswith("diff ")
            or previous.startswith("index ")
        )
        previous = line
        if is_header:
            if line.startswith("+++ b/"):
                current_file = line[len("+++ b/"):]
            continue
        if not line.startswith("+"):
            continue
        hit = bool(plain_rules) and scan_line_plaintext(line, plain_rules)
        if not hit and salt and entries:
            hit = scan_line_fingerprints(line, entries, salt_hex=salt)
        if hit:
            findings.append(
                {"arm": "content", "file": current_file, "detail": "added line matches a leak rule"}
            )
    return findings


def scan_range(
    range_spec: str,
    *,
    repo: str | None = None,
    fingerprint_file=None,
    plaintext_rules_file=None,
    allowlist_file=None,
    denied_paths_file=None,
    allowed_binary_file=None,
) -> dict:
    """Run all five arms over a commit range. Returns a JSON-able report."""
    salt, entries = _load_fingerprints(fingerprint_file)
    plain = load_plaintext_rules(plaintext_rules_file) if plaintext_rules_file else []
    allowlist = _list_file(allowlist_file)
    denied = _list_file(denied_paths_file)
    allowed_binary = _list_file(allowed_binary_file)

    findings: list[dict] = []

    # ALL arms run per-commit (red-team B2: an endpoint diff misses a marker
    # committed then reverted before push — the blob still lands in public
    # history; history is what leaks, not the tip tree).
    for sha in _git(["rev-list", range_spec], cwd=repo).split():
        # content arm — this commit's own diff (-m: merges diffed vs first parent)
        diff_text = _git(["show", "-m", "--first-parent", "--format=", sha], cwd=repo)
        for finding in scan_diff_text(diff_text, plain_rules=plain, salt=salt, entries=entries):
            findings.append({**finding, "commit": sha})
        message = _git(["log", "-1", "--format=%B", sha], cwd=repo)
        for line in message.splitlines():
            if (plain and scan_line_plaintext(line, plain)) or (
                salt and entries and scan_line_fingerprints(line, entries, salt_hex=salt)
            ):
                findings.append(
                    {"arm": "message", "commit": sha, "detail": "commit message matches a leak rule"}
                )
                break
        if allowlist:
            author, committer = _git(
                ["log", "-1", "--format=%an <%ae>%n%cn <%ce>", sha], cwd=repo
            ).splitlines()[:2]
            ok, why = check_identity(author=author, committer=committer, allowlist=allowlist)
            if not ok:
                findings.append({"arm": "metadata", "commit": sha, "detail": why})
        added = _git(
            ["diff-tree", "--no-commit-id", "--name-only", "--diff-filter=A", "-r", sha],
            cwd=repo,
        ).split("\n")
        for path in filter(None, added):
            if denied and (finding := check_path(path, denied)):
                findings.append({"arm": "path", "commit": sha, "detail": finding})
            if allowed_binary and (finding := check_binary(path, allowed_binary)):
                findings.append({"arm": "binary", "commit": sha, "detail": finding})

    return {"clean": not findings, "range": range_spec, "findings": findings}


# ------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Leak guard — multi-arm public-boundary scanner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check", help="scan a commit range (exit 1 on findings)")
    check.add_argument("--range", required=True, dest="range_spec")
    check.add_argument("--repo", default=None)
    check.add_argument("--fingerprints", type=Path, default=None)
    check.add_argument("--rules", type=Path, default=None)
    check.add_argument("--allowlist", type=Path, default=None)
    check.add_argument("--denied-paths", type=Path, default=None)
    check.add_argument("--allowed-binary", type=Path, default=None)
    check.add_argument("--report", action="store_true", help="print JSON report")

    fingerprint = sub.add_parser(
        "fingerprint", help="build fingerprint JSON from a plaintext markers file (run locally)"
    )
    fingerprint.add_argument("markers_file", type=Path)
    fingerprint.add_argument("--salt-hex", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "fingerprint":
        entries = []
        for raw in args.markers_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            kind = "path" if line.startswith("/") else "words"
            entries.append(build_fingerprint(line, salt_hex=args.salt_hex, kind=kind))
        print(json.dumps({"salt": args.salt_hex, "entries": entries}, indent=2))
        return 0

    report = scan_range(
        args.range_spec,
        repo=args.repo,
        fingerprint_file=args.fingerprints,
        plaintext_rules_file=args.rules,
        allowlist_file=args.allowlist,
        denied_paths_file=args.denied_paths,
        allowed_binary_file=args.allowed_binary,
    )
    if args.report:
        print(json.dumps(report, indent=2))
    elif not report["clean"]:
        for finding in report["findings"]:
            print(f"LEAK-GUARD [{finding['arm']}] {finding['detail']}", file=sys.stderr)
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
