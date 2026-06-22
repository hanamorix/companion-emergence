"""Guarded-change Phase 3 conformance oracle (spec §7.1 / §7.2). Static checks
that no test can drift past: peer-session modules IMPORT/REFERENCE no tool path
(AST-based, so prose mentions are allowed), and the chat path imports no
kindled_link."""
import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PEER_MODULES = [
    _ROOT / "brain" / "kindled_link" / "peer_prompt.py",
    _ROOT / "brain" / "kindled_link" / "session_engine.py",
    _ROOT / "brain" / "kindled_link" / "privacy_gate.py",
    _ROOT / "brain" / "kindled_link" / "relationship.py",
    _ROOT / "brain" / "kindled_link" / "feed_source.py",
    # Phase 7a orchestration that touches peer text → provider (stage-6 review:
    # these must not be able to reach the agentic tool surface either).
    _ROOT / "brain" / "kindled_link" / "tick.py",
    _ROOT / "brain" / "kindled_link" / "transport.py",
    _ROOT / "brain" / "kindled_link" / "session.py",
]
# the genuine injection-hole names (verified in the red-team against the real code)
_FORBIDDEN = {"dispatch", "tool_recruit", "reach_for_capability",
              "read_file", "NELL_TOOL_NAMES"}


def _referenced_names(tree: ast.AST) -> set[str]:
    """Collect names actually USED as code: imports, attribute accesses, and
    bare name references. Strings, comments, and docstrings are excluded by
    construction (ast never yields them as Name/Attribute/import nodes)."""
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                used.add(alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                used.add(node.module.split(".")[-1])
            for alias in node.names:
                used.add(alias.name)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
    return used


def test_peer_modules_reference_no_tool_path():
    # CRITERION 1 (static, AST): the deepest injection holes are not importable
    # or referenced as code. A docstring naming them is allowed.
    for mod in _PEER_MODULES:
        tree = ast.parse(mod.read_text(encoding="utf-8"), filename=str(mod))
        hits = _FORBIDDEN & _referenced_names(tree)
        assert not hits, f"{mod.name} references forbidden tool path(s) {hits!r}"


# Allowlist of brain.* MODULE paths that peer modules may import from.
# Expressed as the dotted module name (not the symbol): `from brain.X import Y`
# is allowed iff "brain.X" is in this set; `import brain.X` is allowed iff
# "brain.X" is in this set.
# Any new brain.* module import in a peer module must be added here + justified.
# Allowlist of effective brain.* import paths that peer modules may use.
# For ``import brain.X.Y``: the full dotted name "brain.X.Y".
# For ``from brain.X import Y``: the effective path "brain.X.Y".
# Keeping entries at symbol/submodule level means adding ``brain.bridge`` would
# permit ANY export from brain.bridge — too broad. Each entry must name the
# specific thing being imported so future additions get explicit review.
_BRAIN_IMPORT_ALLOWLIST = {
    "brain.kindled_link.gate",       # gate protocol types only — no tool surface
    "brain.kindled_link.peer_prompt",  # peer prompt builder — no tool surface
    "brain.kindled_link.limits",     # shared caps and budget constants — no tool surface
    "brain.kindled_link.privacy_gate",  # privacy reflection gate — no tool surface
    "brain.kindled_link.store",      # persistence layer — no tool surface (session_engine imports)
    "brain.kindled_link.relationship",  # relationship state + maturation — no tool surface (session_engine imports)
    "brain.bridge.cli_throttle",     # throttle module — no tool surface
    "brain.chat.extractor",          # extractor symbols only — no tool surface (feed_source.py candidate reader)
    "brain.memory.store",            # MemoryStore symbols only — no tool surface (future kindled_peer memory writes)
    # Phase 7a internal orchestration modules — all kindled-internal, no tool surface:
    "brain.kindled_link.cadence",       # persisted tick cadence — no tool surface
    "brain.kindled_link.session",       # X25519 session handshake — no tool surface
    "brain.kindled_link.session_engine",  # pacing/gate orchestration — no tool surface
    "brain.kindled_link.transport",     # relay send/poll adapter — no tool surface
    "brain.kindled_link.protocol",      # envelope codec/crypto — no tool surface
    "brain.kindled_link.relay_client",  # outbound-only relay client — no tool surface
    "brain.kindled_link.audit",         # transport audit log — no tool surface
    "brain.kindled_link.identity",      # Ed25519 identity — no tool surface
    "brain.kindled_link.codec",         # canonical JSON — no tool surface
}


def _collect_brain_import_paths(tree: ast.AST) -> list[str]:
    """Return the effective dotted paths of every brain.* import in the AST.

    For ``import brain.foo.bar``               → ["brain.foo.bar"]
    For ``from brain.foo import bar``          → ["brain.foo.bar"]
    For ``from brain.foo.bar import X, Y``    → ["brain.foo.bar.X", "brain.foo.bar.Y"]

    Using ``module + "." + symbol`` for ImportFrom means the allowlist expresses
    exactly which symbol is permitted from each brain package (e.g.
    ``brain.bridge.cli_throttle`` allows only the throttle submodule, not
    arbitrary exports from ``brain.bridge``).

    Relative imports are skipped — they cannot name an absolute brain.* path.
    """
    paths: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("brain."):
                    paths.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative — skip
            if node.module and node.module.startswith("brain."):
                for alias in node.names:
                    paths.append(f"{node.module}.{alias.name}")
    return paths


def _is_allowlisted(import_path: str) -> bool:
    """True if import_path is covered by _BRAIN_IMPORT_ALLOWLIST.

    An import path is covered if it exactly matches an allowlist entry (the
    import IS the allowlisted module/submodule) OR if it starts with
    ``entry + "."`` (the import is a symbol FROM an allowlisted module).

    Examples:
      "brain.kindled_link.gate"           → exact match → allowed
      "brain.kindled_link.gate.DenyAllGate" → prefixed by "brain.kindled_link.gate." → allowed
      "brain.bridge.cli_throttle"         → exact match → allowed
      "brain.bridge.cli_throttle.foo"     → prefixed → allowed
      "brain.chat.tool_recruit"           → no match → FORBIDDEN
    """
    for entry in _BRAIN_IMPORT_ALLOWLIST:
        if import_path == entry or import_path.startswith(entry + "."):
            return True
    return False


def test_peer_modules_import_only_allowlisted_modules():
    """FIX 1: future-proof import allowlist oracle.

    Any ``brain.*`` module imported by a peer module that is NOT covered by
    ``_BRAIN_IMPORT_ALLOWLIST`` is a potential tool-injection vector — a
    future developer could add ``from brain.chat.tool_recruit import …``
    and the 5-name denylist (test_peer_modules_reference_no_tool_path)
    would miss it if the symbol name doesn't happen to be on the list.
    This test catches ANY new brain.* import regardless of symbol name.

    Stdlib, third-party, and relative imports are unconditionally allowed.
    """
    for mod_path in _PEER_MODULES:
        src = mod_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(mod_path))
        for import_path in _collect_brain_import_paths(tree):
            assert _is_allowlisted(import_path), (
                f"{mod_path.name} imports {import_path!r} which is NOT covered by "
                f"_BRAIN_IMPORT_ALLOWLIST {_BRAIN_IMPORT_ALLOWLIST!r}. "
                "If this import is safe, add it to _BRAIN_IMPORT_ALLOWLIST and "
                "document why it introduces no tool-path reachability."
            )


def test_chat_path_does_not_import_kindled_link():
    # §7.2: Phase 3 cannot regress the chat token path — it isn't reachable from it.
    targets = list((_ROOT / "brain" / "chat").rglob("*.py"))
    targets.append(_ROOT / "brain" / "bridge" / "provider.py")
    for py in targets:
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "kindled_link" not in node.module, f"{py} imports kindled_link"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "kindled_link" not in alias.name, f"{py} imports kindled_link"
