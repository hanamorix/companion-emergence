"""Guard: brain/memory/store.py is the only module that writes memories.db.

#507 put every store's connection open behind one helper, and every memories.db write behind
MemoryStore. Nothing kept it that way: a new module could `sqlite3.connect` the file, or run
`store._conn.execute("UPDATE memories ...")`, and the tangle of overlapping writers would be back.
This test scans brain/ with ast and fails on that.

How "the memories db" is recognised (a path variable defeats any single test, so three):

1. Opens. Every `sqlite3.connect`, `sqlite3.Connection(...)` and `open_store_connection` in brain/ must
   sit in a file listed in OPENERS (with the database it opens) or in a (file, function) listed in
   READER_EXCEPTIONS, where a method is keyed as `Class.method`. An unlisted opener fails whatever path it passes. Adding a new SQLite file
   costs one line in OPENERS.
2. Write statements. Any `execute` / `executemany` / `executescript` whose SQL text starts with
   INSERT, UPDATE, DELETE, REPLACE, CREATE, DROP or ALTER (a CTE counts when one of those verbs
   appears) and names a table of the memories schema fails, in every module but store.py. The table
   names are read from `brain.memory.store._SCHEMA`, not listed here, so a new memories table is
   covered at once (test_derived_memories_tables_cover_the_schema pins that the derivation works).
   A bare VACUUM or REINDEX fails outside OPENERS files whatever it names.
3. `executescript` fails in every module that is not in OPENERS, whatever its text.

READER_EXCEPTIONS: the owner ruled that code which only READS a memories.db may stay outside the
store for now ("make a comment naming them if a problem ever comes up so they're easy to find").
Each entry is named below with why, and the scan also requires the comment
`memdb-single-writer reader exception` within 3 lines above the call, so the comment cannot drift
away from the call site. If this test ever fails on one of them, start there.

Limits (the scanner is a name-and-text check, not a type check):
- SQL with no resolvable literal prefix (built by a function call, read from a file, or whose first
  piece is a variable) is not analysed. So are: a table name held in a variable (`UPDATE {t}`), SQL in
  a class attribute, dict value, ternary or parameter default, and a literal handed to a wrapper
  function that runs it. A bare name is judged against every value assigned to it in scope, except
  through a chained assignment (`a = b = "DELETE ..."`), a loop over a named tuple of SQL strings, or a
  constant re-exported through a second module.
- PRAGMA statements that write (journal_mode, user_version) are not flagged.
- Registered other-db openers (OPENERS) are trusted not to open memories.db; only humans reviewing a
  registry edit catch that. Likewise a second open in the same function of a reader exception.
  Other drivers (aiosqlite, the sqlite3 command line, SQLAlchemy) and `getattr(sqlite3, "connect")`
  are not seen; `open_store_connection` is followed through `from ... import ... as` aliases.
- Scope is brain/ only. Test fixtures, wizard-validation/ and file-level copies (shutil) are not seen.
- Calls to the MemoryStore `fts_*` and `connect_for_fts_health` static methods are the store itself,
  so passing them any connection is not flagged.
- `MemoryStore(...)` instances are deliberately many (one per thread or request, see
  brain/bridge/server.py "HA hardening"); this test is about who writes, not how many store
  objects exist.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

STORE = "brain/memory/store.py"
MARKER = "memdb-single-writer reader exception"
MARKER_WINDOW = 3  # the marker comment sits on the call line or up to this many lines above it

# The scanner lives in this test file, not in product code, so the tests that feed it synthetic trees (or that
# pin facts the base commit already had) pass against the base commit by design. The tests that fail there are
# the real-tree scan and the new MemoryStore method tests.
_NO_BASE_FAILURE_REASON = (
    "exercises the scanner that this file itself adds, on synthetic trees or on facts the base commit already "
    "had, so the old product code cannot make it fail; the real-tree scan test is the one that bites on base"
)

# Files that may open a SQLite connection, and which database each one opens.
OPENERS: dict[str, str] = {
    STORE: "memories.db. THE writer: MemoryStore, plus the FTS health connection used before any store exists.",
    "brain/utils/sqlite_open.py": "the shared open helper (#507); it opens whatever path a store hands it.",
    "brain/memory/hebbian.py": "hebbian.db (HebbianMatrix, through open_store_connection).",
    "brain/soul/store.py": "crystallizations.db (SoulStore, through open_store_connection).",
    "brain/kindled_link/store.py": "the kindled-link db (KindledLinkStore, through open_store_connection).",
    "brain/works/store.py": "works.db (WorksStore, through open_store_connection).",
    "brain/memory/known_names.py": "known_names.db, a separate small file (read-only URIs plus its own writer).",
    "brain/initiate/resonance.py": "memory_activation_baseline.db, the resonance tick's own file.",
    "brain/health/walker.py": "works.db integrity check only; the other databases go through their stores.",
}

# (file, innermost function; a method is Class.method) -> why this read of a memories.db is allowed outside the store.
READER_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("brain/memory/embedding_matrix.py", "EmbeddingMatrix._load_from_db"): (
        "SELECT id, embedding only, into the in-process embedding matrix (plus PRAGMA busy_timeout)."
    ),
    ("brain/recovery/source_reader.py", "_ro_connect"): (
        "read-only (mode=ro URI) read of a SOURCE persona's memories.db that must not be altered."
    ),
    ("brain/migrator/companion_emergence.py", "_count_rows"): (
        "SELECT COUNT(*) of a SOURCE persona's tables during migrate preflight."
    ),
}

_WRITE_VERBS = ("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER")
_WRITE_VERB_RE = re.compile(r"\b(?:" + "|".join(_WRITE_VERBS) + r")\b")
_BARE_MAINTENANCE = ("VACUUM", "REINDEX")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str  # "open", "marker", "write", "executescript"
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.detail}"


@dataclass
class ScanResult:
    findings: list[Finding] = field(default_factory=list)
    used_openers: set[str] = field(default_factory=set)
    used_readers: set[tuple[str, str]] = field(default_factory=set)


def memories_tables() -> frozenset[str]:
    """Every table `MemoryStore` creates in memories.db, read from its schema text."""
    from brain.memory.store import _SCHEMA

    text = re.sub(r"--[^\n]*", "", _SCHEMA)
    names = re.findall(r"CREATE\s+(?:VIRTUAL\s+)?TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)", text, re.IGNORECASE)
    return frozenset(n.lower() for n in names)


def _pieces(node: ast.AST, consts: list[dict[str, list[ast.AST]]], depth: int = 0) -> list[str | None]:
    """The literal pieces of a SQL expression, in order; None marks a part that is not a literal."""
    if depth > 6:
        return [None]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        out: list[str | None] = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(part.value)
            elif isinstance(part, ast.FormattedValue):
                out.extend(_pieces(part.value, consts, depth + 1))
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _pieces(node.left, consts, depth + 1) + _pieces(node.right, consts, depth + 1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _pieces(node.left, consts, depth + 1)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return _pieces(node.func.value, consts, depth + 1)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("strip", "lstrip", "rstrip"):
        return _pieces(node.func.value, consts, depth + 1)  # the judge already skips leading whitespace
    if (
        isinstance(node, ast.Call)
        and node.args
        and (getattr(node.func, "attr", None) == "dedent" or getattr(node.func, "id", None) == "dedent")
    ):
        return _pieces(node.args[0], consts, depth + 1)
    if isinstance(node, (ast.Name, ast.Attribute)):
        key = node.id if isinstance(node, ast.Name) else ast.unparse(node)  # "q.SQL" for a module's constant
        for scope in reversed(consts):
            if key in scope:
                return _pieces(scope[key][-1], consts, depth + 1)
    return [None]


def _sql_texts(node: ast.AST, consts: list[dict[str, list[ast.AST]]]) -> list[str]:
    """Every SQL text the argument can hold. A bare name gives one text per value assigned to it
    anywhere in scope (so `sql = 'UPDATE ...'` in an `if` and `sql = 'SELECT 1'` in the `else` are
    both judged); anything else gives its one text. Texts that do not start with a literal are dropped."""
    if isinstance(node, ast.Name):
        values = [v for scope in consts for v in scope.get(node.id, [])]
        candidates = values or [node]
    else:
        candidates = [node]
    texts = [_sql_text(c, consts) for c in candidates]
    return [t for t in texts if t is not None]


def _sql_text(node: ast.AST, consts: list[dict[str, list[ast.AST]]]) -> str | None:
    """Upper-cased SQL with comments stripped, or None when it does not start with a literal."""
    pieces = _pieces(node, consts)
    if not pieces or pieces[0] is None:
        return None
    text = "".join(p if p is not None else " " for p in pieces)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"--[^\n]*", " ", text)
    return text.upper()


def _assignments(scope: ast.AST) -> dict[str, list[ast.AST]]:
    """name -> every value assigned to it, for a function (anywhere inside it) or a module (top level only)."""
    out: dict[str, list[ast.AST]] = {}
    nodes = scope.body if isinstance(scope, ast.Module) else list(ast.walk(scope))
    for node in nodes:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out.setdefault(node.targets[0].id, []).append(node.value)
        elif (  # sql, params = "...", ()
            isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Tuple) and len(node.targets[0].elts) == len(node.value.elts)
        ):
            for target, value in zip(node.targets[0].elts, node.value.elts, strict=True):
                if isinstance(target, ast.Name):
                    out.setdefault(target.id, []).append(value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            out.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.NamedExpr):  # (sql := "...")
            out.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add) and isinstance(node.target, ast.Name):
            out.setdefault(node.target.id, []).append(node.value)  # each appended piece is judged on its own
        elif (  # for s in ("...", "..."):
            isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.target, ast.Name)
            and isinstance(node.iter, (ast.Tuple, ast.List, ast.Set))
        ):
            out.setdefault(node.target.id, []).extend(node.iter.elts)
    return out


class _FileScan(ast.NodeVisitor):
    def __init__(
        self,
        rel: str,
        source: str,
        tree: ast.Module,
        tables: frozenset[str],
        openers: dict[str, str],
        readers: dict[tuple[str, str], str],
        result: ScanResult,
        module_consts: dict[str, dict[str, list[ast.AST]]] | None = None,
    ) -> None:
        module_consts = module_consts or {}
        self.rel = rel
        # lines holding the marker in a comment token, not in a string (#603)
        self.marker_lines = {
            tok.start[0]
            for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type == tokenize.COMMENT and MARKER in tok.string
        }
        self.tables = tables
        self.table_re = (
            re.compile(r"\b(?:" + "|".join(re.escape(t.upper()) for t in sorted(tables)) + r")\w*")
            if tables
            else None
        )
        self.openers = openers
        self.readers = readers
        self.result = result
        self.funcs: list[str] = []
        self.classes: list[tuple[str, int]] = []  # (class name, depth of self.funcs where it sits)
        self.consts: list[dict[str, list[ast.AST]]] = [_assignments(tree)]
        # Names that mean sqlite3.connect / sqlite3.Connection in this file.
        self.module_names = {"sqlite3"}
        self.connect_names: set[str] = set()
        self.connection_names: set[str] = set()
        self.store_open_names = {"open_store_connection"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "sqlite3":
                        self.module_names.add(alias.asname or "sqlite3")
                    elif alias.name in module_consts:  # import brain.sqlconsts [as q]
                        self._bind_module(alias.asname or alias.name, module_consts[alias.name])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if node.module == "sqlite3" and alias.name == "connect":
                        self.connect_names.add(alias.asname or "connect")
                    elif node.module == "sqlite3" and alias.name == "Connection":
                        self.connection_names.add(alias.asname or "Connection")
                    elif alias.name == "open_store_connection":
                        self.store_open_names.add(alias.asname or "open_store_connection")
                    elif alias.name in module_consts.get(node.module or "", {}):
                        imported = module_consts[node.module][alias.name]
                        self.consts[0].setdefault(alias.asname or alias.name, []).extend(imported)
                    elif f"{node.module}.{alias.name}" in module_consts:  # from brain import sqlconsts as q
                        self._bind_module(alias.asname or alias.name, module_consts[f"{node.module}.{alias.name}"])

    def _bind_module(self, local: str, consts: dict[str, list[ast.AST]]) -> None:
        """Make `<local>.NAME` resolve to that module's NAME (keys are the dotted source text, see _pieces)."""
        for name, values in consts.items():
            self.consts[0].setdefault(f"{local}.{name}", []).extend(values)

    # -- scope tracking -----------------------------------------------------
    def _enter(self, node: ast.AST, name: str) -> None:
        # a method is Class.method, so a READER_EXCEPTIONS entry covers that class's method only (#603)
        if self.classes and self.classes[-1][1] == len(self.funcs):
            name = f"{self.classes[-1][0]}.{name}"
        self.funcs.append(name)
        self.consts.append(_assignments(node))
        self.generic_visit(node)
        self.consts.pop()
        self.funcs.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append((node.name, len(self.funcs)))
        self.generic_visit(node)
        self.classes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter(node, node.name)

    # -- opens --------------------------------------------------------------
    def _open_ref(self, node: ast.AST, *, called: bool) -> str | None:
        """Name the open this node refers to, or None."""
        if isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in self.module_names:
                if node.attr == "connect" or (node.attr == "Connection" and called):
                    return f"sqlite3.{node.attr}"
            if node.attr == "open_store_connection":
                return "open_store_connection"
        elif isinstance(node, ast.Name):
            if node.id in self.connect_names or (node.id in self.connection_names and called):
                return f"sqlite3 ({node.id})"
            if node.id in self.store_open_names:
                return "open_store_connection"
        return None

    def _check_open(self, node: ast.AST, what: str) -> None:
        func = self.funcs[-1] if self.funcs else "<module>"
        line = node.lineno
        if self.rel in self.openers:
            self.result.used_openers.add(self.rel)
            return
        key = (self.rel, func)
        if key in self.readers:
            self.result.used_readers.add(key)
            if not any(line - MARKER_WINDOW <= n <= line for n in self.marker_lines):
                self.result.findings.append(
                    Finding(
                        self.rel,
                        line,
                        "marker",
                        f"{what} in a named reader exception has no `{MARKER}` comment within "
                        f"{MARKER_WINDOW} lines above it",
                    )
                )
            return
        self.result.findings.append(
            Finding(
                self.rel,
                line,
                "open",
                f"{what} opens a SQLite connection outside the registered openers. If it touches "
                "memories.db, go through MemoryStore. If it opens another database, add this file to "
                "OPENERS with the database name; if it only reads a memories.db, add a READER_EXCEPTIONS entry.",
            )
        )

    def visit_Call(self, node: ast.Call) -> None:
        what = self._open_ref(node.func, called=True)
        if what is not None:
            self._check_open(node.func, what)
            # do not also visit node.func as a bare reference
            for child in [*node.args, *[k.value for k in node.keywords]]:
                self.visit(child)
            self._check_sql(node)
            return
        self._check_sql(node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        what = self._open_ref(node, called=False)
        if what is not None:
            self._check_open(node, what)
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            what = self._open_ref(node, called=False)
            if what is not None:
                self._check_open(node, what)

    # -- write statements ---------------------------------------------------
    def _check_sql(self, node: ast.Call) -> None:
        if self.rel == STORE or not isinstance(node.func, ast.Attribute):
            return
        method = node.func.attr
        if method not in ("execute", "executemany", "executescript"):
            return
        if method == "executescript" and self.rel not in self.openers:
            self.result.findings.append(
                Finding(self.rel, node.lineno, "executescript", "executescript outside the registered openers")
            )
        if not node.args:
            return
        for text in _sql_texts(node.args[0], self.consts):
            detail = self._judge(text, method)
            if detail is not None:
                self.result.findings.append(Finding(self.rel, node.lineno, "write", detail))
                return

    def _judge(self, text: str, method: str) -> str | None:
        """Why this SQL text is a forbidden write, or None."""
        first = text.split(None, 1)[0] if text.strip() else ""
        if first in _BARE_MAINTENANCE and self.rel not in self.openers:
            return f"{first} outside the registered openers"
        if method == "executescript":
            is_write = bool(_WRITE_VERB_RE.search(text))
        else:
            is_write = first in _WRITE_VERBS or (first == "WITH" and bool(_WRITE_VERB_RE.search(text)))
        if not is_write or self.table_re is None:
            return None
        hit = self.table_re.search(text)
        if hit is None:
            return None
        return (
            f"{method} writes the memories table {hit.group(0).lower()!r}; only "
            f"{STORE} may write memories.db. Add a MemoryStore method and call it."
        )


def scan_tree(
    root: Path,
    openers: dict[str, str] | None = None,
    readers: dict[tuple[str, str], str] | None = None,
    tables: frozenset[str] | None = None,
) -> ScanResult:
    openers = OPENERS if openers is None else openers
    readers = READER_EXCEPTIONS if readers is None else readers
    tables = memories_tables() if tables is None else tables
    result = ScanResult()
    parsed: list[tuple[str, str, ast.Module]] = []
    for path in sorted((root / "brain").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        parsed.append((rel, source, ast.parse(source)))
    # module-level assignments of every brain module, so SQL imported from another module is judged by its text (#603)
    module_consts = {
        rel.removesuffix(".py").removesuffix("/__init__").replace("/", "."): _assignments(tree) for rel, _s, tree in parsed
    }
    for rel, source, tree in parsed:
        _FileScan(rel, source, tree, tables, openers, readers, result, module_consts).visit(tree)
    result.findings.sort(key=lambda f: (f.path, f.line, f.rule))
    return result


def registry_problems(result: ScanResult) -> list[str]:
    """Registry entries that no longer match a real call (so the lists cannot rot)."""
    problems = [f"OPENERS file {p} has no SQLite open in it any more" for p in OPENERS if p not in result.used_openers]
    problems += [
        f"READER_EXCEPTIONS {key} has no SQLite open in that function any more"
        for key in READER_EXCEPTIONS
        if key not in result.used_readers
    ]
    return problems


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def _plant(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, source in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    return tmp_path


@pytest.fixture(scope="module")
def real_scan() -> ScanResult:
    """The scan of the real tree, done once in setup: it costs seconds of CPU on CI, which belongs to no one test."""
    return scan_tree(REPO)


def test_real_tree_has_no_memories_db_writer_outside_the_store(real_scan):
    result = real_scan
    assert not result.findings, (
        "Only brain/memory/store.py may write memories.db (see this file's docstring).\n"
        + "\n".join(str(f) for f in result.findings)
    )


@pytest.mark.no_base_failure(reason=_NO_BASE_FAILURE_REASON)
def test_registry_entries_all_match_a_real_call(tmp_path, real_scan):
    assert not registry_problems(real_scan)
    # the oracle can fail: on a tree with no brain/ files every entry is reported stale
    assert len(registry_problems(scan_tree(tmp_path))) == len(OPENERS) + len(READER_EXCEPTIONS)


@pytest.mark.no_base_failure(reason=_NO_BASE_FAILURE_REASON)
def test_derived_memories_tables_cover_the_schema():
    tables = memories_tables()
    assert {"memories", "memories_fts", "calibration_log", "cost_series_samples", "cluster_centroids"} <= tables


_RAW_OPEN = (
    "import sqlite3\n"
    "def go(persona_dir):\n"
    "    return sqlite3.connect(str(persona_dir / 'memories.db'))\n"
)
_READER_OK = (
    "import sqlite3\n"
    "class EmbeddingMatrix:\n"
    "    def _load_from_db(self, path):\n"
    f"        # {MARKER}: reads embeddings only\n"
    "        conn = sqlite3.connect(path)\n"
    "        return conn.execute('SELECT id FROM memories').fetchall()\n"
)

PLANTED = {
    "raw_connect_on_memories_db": ("brain/rogue.py", _RAW_OPEN, "open"),
    "open_store_connection_in_unregistered_module": (
        "brain/rogue.py",
        "from brain.utils.sqlite_open import open_store_connection\n"
        "def go(p):\n    return open_store_connection(p, caller='x', integrity_check=False)\n",
        "open",
    ),
    "raw_update_through_store_conn": (
        "brain/rogue.py",
        "def go(store, i):\n    store._conn.execute(\"UPDATE memories SET state='active' WHERE id=?\", (i,))\n",
        "write",
    ),
    "write_sql_passed_by_module_constant": (
        "brain/rogue.py",
        "_SQL = \"DELETE FROM calibration_log WHERE ts < ?\"\ndef go(store):\n    store._conn.execute(_SQL, (1,))\n",
        "write",
    ),
    "executescript_in_unregistered_module": (
        "brain/rogue.py",
        "def go(conn):\n    conn.executescript('SELECT 1;')\n",
        "executescript",
    ),
    "fts_rebuild_outside_the_store": (
        "brain/rogue.py",
        "def go(conn):\n    conn.execute(\"INSERT INTO memories_fts(memories_fts) VALUES('rebuild')\")\n",
        "write",
    ),
    "from_import_alias_connect": (
        "brain/rogue.py",
        "from sqlite3 import connect as c\ndef go(p):\n    return c(p)\n",
        "open",
    ),
    "reader_exception_without_marker": (
        "brain/memory/embedding_matrix.py",
        "import sqlite3\nclass EmbeddingMatrix:\n    def _load_from_db(self, path):\n        return sqlite3.connect(path)\n",
        "marker",
    ),
    "reader_exception_marker_too_far_above": (
        "brain/memory/embedding_matrix.py",
        "import sqlite3\nclass EmbeddingMatrix:\n    def _load_from_db(self, path):\n"
        f"        # {MARKER}\n        a = 1\n        b = 2\n        c = 3\n        d = 4\n        return sqlite3.connect(path)\n",
        "marker",
    ),
    "reader_exception_marker_only_in_a_string": (  # #603: the marker must be a comment
        "brain/memory/embedding_matrix.py",
        "import sqlite3\nclass EmbeddingMatrix:\n    def _load_from_db(self, path):\n"
        f"        note = '{MARKER}'\n        return sqlite3.connect(path)\n",
        "marker",
    ),
    "second_open_in_another_function_of_a_reader_file": (
        "brain/memory/embedding_matrix.py",
        _READER_OK + "def other(path):\n    return sqlite3.connect(path)\n",
        "open",
    ),
    "concatenated_sql_with_unresolved_tail": (
        "brain/rogue.py",
        "def go(store, cols):\n    store._conn.execute('UPDATE memories SET ' + cols)\n",
        "write",
    ),
    "fstring_sql": (
        "brain/rogue.py",
        "def go(store, cols):\n    store._conn.execute(f'UPDATE memories SET {cols}')\n",
        "write",
    ),
    "lowercase_sql": (
        "brain/rogue.py",
        "def go(store):\n    store._conn.execute('update memories set state = 1')\n",
        "write",
    ),
    "block_comment_prefix": (
        "brain/rogue.py",
        "def go(store):\n    store._conn.execute('/* x */ UPDATE memories SET state = 1')\n",
        "write",
    ),
    "bare_vacuum": (
        "brain/rogue.py",
        "def go(conn):\n    conn.execute('VACUUM')\n",
        "write",
    ),
    "memories_write_inside_a_registered_opener_script": (
        "brain/memory/hebbian.py",
        "import sqlite3\ndef go(p):\n    c = sqlite3.connect(p)\n    c.executescript('CREATE TABLE IF NOT EXISTS memories (id TEXT);')\n",
        "write",
    ),
    "open_store_connection_imported_under_an_alias": (
        "brain/rogue.py",
        "from brain.utils.sqlite_open import open_store_connection as osc\n"
        "def go(p):\n    return osc(p, caller='x', integrity_check=False)\n",
        "open",
    ),
    "write_sql_assigned_in_one_branch_and_a_select_in_the_other": (
        "brain/rogue.py",
        "def go(store, flag):\n"
        "    if flag:\n        sql = 'UPDATE memories SET state = 1'\n"
        "    else:\n        sql = 'SELECT 1'\n"
        "    store._conn.execute(sql)\n",
        "write",
    ),
    "write_sql_wrapped_in_strip": (  # #603
        "brain/rogue.py",
        "def go(store):\n    store._conn.execute(\"\"\"\n        DELETE FROM memories WHERE id = ?\n    \"\"\".strip(), (1,))\n",
        "write",
    ),
    "write_sql_wrapped_in_textwrap_dedent": (  # #603
        "brain/rogue.py",
        "import textwrap\ndef go(store):\n"
        "    store._conn.execute(textwrap.dedent(\"\"\"\n        UPDATE memories SET state = 1\n    \"\"\"))\n",
        "write",
    ),
    "write_sql_wrapped_in_bare_dedent": (  # #603
        "brain/rogue.py",
        "from textwrap import dedent\ndef go(store):\n    store._conn.execute(dedent('DELETE FROM memories'))\n",
        "write",
    ),
    "write_sql_from_a_tuple_unpack": (  # #603
        "brain/rogue.py",
        "def go(store):\n    sql, params = 'DELETE FROM memories WHERE id = ?', (1,)\n    store._conn.execute(sql, params)\n",
        "write",
    ),
    "write_sql_from_a_walrus": (  # #603
        "brain/rogue.py",
        "def go(store):\n    if (sql := 'UPDATE memories SET state = 1'):\n        store._conn.execute(sql)\n",
        "write",
    ),
    "write_sql_built_with_plus_equals": (  # #603
        "brain/rogue.py",
        "def go(store):\n    sql = ''\n    sql += 'DELETE FROM memories WHERE id = ?'\n    store._conn.execute(sql, (1,))\n",
        "write",
    ),
    "write_sql_from_a_for_loop": (  # #603
        "brain/rogue.py",
        "def go(store):\n    for s in ('SELECT 1', 'DELETE FROM memories'):\n        store._conn.execute(s)\n",
        "write",
    ),
    "reader_exception_name_reused_by_another_class": (  # #603: exceptions are keyed by Class.method
        "brain/memory/embedding_matrix.py",
        "import sqlite3\nclass Impostor:\n    def _load_from_db(self, path):\n"
        f"        # {MARKER}\n        return sqlite3.connect(path)\n",
        "open",
    ),
    "connect_taken_as_a_reference": (
        "brain/rogue.py",
        "import sqlite3\ndef go(p):\n    c = sqlite3.connect\n    return c(p)\n",
        "open",
    ),
}


@pytest.mark.no_base_failure(reason=_NO_BASE_FAILURE_REASON)
@pytest.mark.parametrize("name", sorted(PLANTED))
def test_scanner_flags_each_planted_violation(tmp_path, name):
    rel, source, rule = PLANTED[name]
    result = scan_tree(_plant(tmp_path, {rel: source}))
    assert any(f.path == rel and f.rule == rule for f in result.findings), (
        f"expected a {rule!r} finding in {rel}, got: {[str(f) for f in result.findings]}"
    )


@pytest.mark.no_base_failure(reason=_NO_BASE_FAILURE_REASON)
def test_scanner_follows_a_sql_constant_imported_from_another_module(tmp_path):
    """#603: `from brain.x import SQL; conn.executemany(SQL, rows)` is judged by the constant's text in brain/x.py."""
    files = {
        "brain/sqlconsts.py": "SQL = 'DELETE FROM memories WHERE id = ?'\n",
        "brain/rogue.py": "from brain.sqlconsts import SQL\ndef go(store, rows):\n    store._conn.executemany(SQL, rows)\n",
    }
    result = scan_tree(_plant(tmp_path, files))
    assert any(f.path == "brain/rogue.py" and f.rule == "write" for f in result.findings), [str(f) for f in result.findings]


@pytest.mark.no_base_failure(reason=_NO_BASE_FAILURE_REASON)
def test_scanner_follows_a_sql_constant_read_as_a_module_attribute(tmp_path):
    """#603: `from brain import sqlconsts as q; conn.execute(q.SQL)` is judged by the constant's text."""
    files = {
        "brain/sqlconsts.py": "SQL = 'UPDATE memories SET state = 1'\n",
        "brain/rogue.py": "from brain import sqlconsts as q\ndef go(store):\n    store._conn.execute(q.SQL)\n",
    }
    result = scan_tree(_plant(tmp_path, files))
    assert any(f.path == "brain/rogue.py" and f.rule == "write" for f in result.findings), [str(f) for f in result.findings]


INNOCENT = {
    "select_on_memories_through_store_conn": (
        "brain/ok.py",
        "def go(store):\n    return store._conn.execute('SELECT id FROM memories').fetchall()\n",
    ),
    "update_on_another_databases_table": (
        "brain/ok.py",
        "def go(conn):\n    conn.execute('UPDATE works SET shared_at = ? WHERE id = ?', (1, 2))\n",
    ),
    "memory_store_construction": (
        "brain/ok.py",
        "from brain.memory.store import MemoryStore\ndef go(p):\n    s = MemoryStore(p / 'memories.db')\n    s.close()\n",
    ),
    "anything_inside_the_store_module": (
        STORE,
        "import sqlite3\ndef go(p):\n    c = sqlite3.connect(p)\n    c.executescript('CREATE TABLE IF NOT EXISTS memories (id TEXT);')\n"
        "    c.execute(\"UPDATE memories SET state='x'\")\n",
    ),
    "select_only_in_a_named_reader_exception": (
        "brain/memory/embedding_matrix.py",
        _READER_OK,
    ),
    "annotation_only_use_of_sqlite3_connection": (
        "brain/ok.py",
        "import sqlite3\ndef go(conn: sqlite3.Connection) -> None:\n    conn.execute('SELECT 1')\n",
    ),
}


@pytest.mark.no_base_failure(reason=_NO_BASE_FAILURE_REASON)
@pytest.mark.parametrize("name", sorted(INNOCENT))
def test_scanner_leaves_innocent_code_alone(tmp_path, name):
    rel, source = INNOCENT[name]
    result = scan_tree(_plant(tmp_path, {rel: source}))
    assert not result.findings, [str(f) for f in result.findings]

