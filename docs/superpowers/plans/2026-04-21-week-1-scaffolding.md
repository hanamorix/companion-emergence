# Week 1 — Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a cloneable, installable, CI-green Python package skeleton that the 7 subsequent weeks build on. By end of Week 1: `uv sync` installs cleanly, `nell --version` runs, pytest passes, and GitHub Actions reports green on macOS + Windows + Linux.

**Architecture:** Python 3.12+ package `brain` (framework code, persona-agnostic) plus `persona/` directory (per-persona configs + data) plus `examples/` starter personas plus standard pyproject.toml layout. CLI entry point via `[project.scripts]`. `platformdirs` for OS-aware paths. Three-source config precedence (persona.toml → .env → env var). pytest + ruff for quality. GitHub Actions matrix for cross-platform CI.

**Tech Stack:** Python 3.12, uv (package manager), hatchling (build backend), platformdirs, pytest, ruff, GitHub Actions.

---

## Context: what already exists

The companion-emergence repo was bootstrapped 2026-04-21 with:

- `.gitignore` — protects persona data/memories from ever entering git
- `LICENSE` — MIT
- `README.md` — Week 0 status
- `docs/source-spec/2026-04-21-framework-rebuild-design.md` — the 14,275-word spec this plan executes against
- Remote: `https://github.com/hanamorix/companion-emergence` (private)
- First commit: `307084f`

This plan adds everything Week 1 needs on top of that foundation. Nothing in Week 1 implements *product* functionality — it establishes the skeleton every subsequent week depends on.

---

## File structure (what this plan creates)

```
companion-emergence/
├── pyproject.toml                          (Task 1)
├── .env.example                            (Task 6)
├── .gitattributes                          (Task 6)
├── .github/
│   └── workflows/
│       └── test.yml                        (Task 7)
├── brain/
│   ├── __init__.py                         (Task 1)
│   ├── paths.py                            (Task 2)
│   ├── config.py                           (Task 3)
│   └── cli.py                              (Task 1 + Task 4)
├── examples/
│   └── starter-thoughtful/
│       ├── persona.toml                    (Task 5)
│       ├── personality.json                (Task 5)
│       ├── soul.json                       (Task 5)
│       ├── self_model.json                 (Task 5)
│       ├── voice.json                      (Task 5)
│       └── emotions/
│           └── extensions.json             (Task 5)
└── tests/
    ├── conftest.py                         (Task 1)
    └── unit/
        └── brain/
            ├── test_cli.py                 (Task 1 + Task 4)
            ├── test_paths.py               (Task 2)
            ├── test_config.py              (Task 3)
            └── test_starter_persona.py     (Task 5)
```

Already committed and deliberately NOT changed in Week 1: `.gitignore`, `LICENSE`, `README.md`, `docs/source-spec/*`.

---

## Task 1: Package skeleton + uv sync + basic CLI

**Goal of this task:** `uv sync` installs the project; `uv run nell --version` prints the version; `uv run pytest -v` runs a green test suite (with one real test).

**Files:**
- Create: `/Users/hanamori/companion-emergence/pyproject.toml`
- Create: `/Users/hanamori/companion-emergence/brain/__init__.py`
- Create: `/Users/hanamori/companion-emergence/brain/cli.py`
- Create: `/Users/hanamori/companion-emergence/tests/__init__.py`
- Create: `/Users/hanamori/companion-emergence/tests/conftest.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/__init__.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/__init__.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/test_cli.py`

- [ ] **Step 1: Write `pyproject.toml`**

Create `/Users/hanamori/companion-emergence/pyproject.toml`:

```toml
[project]
name = "companion-emergence"
version = "0.0.1"
description = "Framework for building persistent, emotionally aware AI companions"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }
authors = [
    { name = "Hanamori" },
]
dependencies = [
    "platformdirs>=4.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.5",
]

[project.scripts]
nell = "brain.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["brain"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
pythonpath = ["."]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "B", "C4", "UP"]
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["N802"]
```

- [ ] **Step 2: Write `brain/__init__.py`**

Create `/Users/hanamori/companion-emergence/brain/__init__.py`:

```python
"""companion-emergence — framework for building emotionally aware AI companions.

Reference implementation: Nell. See docs/source-spec/ for the full design.
"""

__version__ = "0.0.1"
```

- [ ] **Step 3: Write `brain/cli.py` (minimal — just --version + help)**

Create `/Users/hanamori/companion-emergence/brain/cli.py`:

```python
"""Entry point CLI for companion-emergence.

Invoked as `nell <subcommand> [options]`. Week 1 wires up `--version` and
the subcommand skeleton; subcommands themselves are stubs until their
respective weeks land.
"""

from __future__ import annotations

import argparse
import sys

from brain import __version__


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser."""
    parser = argparse.ArgumentParser(
        prog="nell",
        description=(
            "companion-emergence — CLI for building emotionally aware "
            "AI companions"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"companion-emergence {__version__}",
    )
    parser.add_subparsers(dest="command", title="subcommands")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns shell exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `tests/conftest.py`**

Create `/Users/hanamori/companion-emergence/tests/conftest.py`:

```python
"""Shared pytest fixtures and configuration for companion-emergence tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove companion-emergence-relevant env vars for isolation."""
    for key in [
        "NELLBRAIN_HOME",
        "NELL_IPC_JID",
        "BRIDGE_BIND",
        "PROVIDER",
        "MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    yield
```

- [ ] **Step 5: Write `tests/__init__.py`, `tests/unit/__init__.py`, `tests/unit/brain/__init__.py`**

Create all three as empty files:

```bash
touch /Users/hanamori/companion-emergence/tests/__init__.py
touch /Users/hanamori/companion-emergence/tests/unit/__init__.py
touch /Users/hanamori/companion-emergence/tests/unit/brain/__init__.py
```

- [ ] **Step 6: Write the failing CLI test**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/test_cli.py`:

```python
"""Tests for brain.cli entry point."""

from __future__ import annotations

import pytest

from brain import cli


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    """`nell --version` prints version and exits with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "companion-emergence" in captured.out


def test_no_args_prints_help_and_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`nell` with no args shows help and returns 1."""
    result = cli.main([])
    assert result == 1
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()
```

- [ ] **Step 7: Run `uv sync`**

```bash
cd /Users/hanamori/companion-emergence
uv sync --all-extras
```

Expected: `Installed N packages in ...s`. A `.venv` directory appears and `uv.lock` is created.

If `uv` is not installed, install it first via `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`.

- [ ] **Step 8: Run pytest — should pass**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
```

Expected: `2 passed` green for `test_version_flag_prints_version` and `test_no_args_prints_help_and_exits_nonzero`.

- [ ] **Step 9: Run `nell --version` via uv**

```bash
cd /Users/hanamori/companion-emergence
uv run nell --version
```

Expected output: `companion-emergence 0.0.1`

- [ ] **Step 10: Run ruff to verify no lint errors**

```bash
cd /Users/hanamori/companion-emergence
uv run ruff check .
```

Expected: `All checks passed!` (or no output with exit code 0).

- [ ] **Step 11: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add pyproject.toml brain/ tests/
git commit -m "feat(scaffold): pyproject.toml + brain package skeleton + basic CLI

- pyproject.toml with hatchling build backend, uv-managed deps, nell entry point
- brain/__init__.py exposing __version__
- brain/cli.py with --version and help-on-no-args
- tests/conftest.py with clean_env fixture
- tests/unit/brain/test_cli.py with 2 passing tests
- ruff + pytest configured

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `brain/paths.py` — platformdirs-based path resolution (TDD)

**Goal:** All user-facing paths route through one module. `NELLBRAIN_HOME` env var overrides; otherwise platformdirs picks OS-appropriate default.

**Files:**
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/test_paths.py`
- Create: `/Users/hanamori/companion-emergence/brain/paths.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/test_paths.py`:

```python
"""Tests for brain.paths — platformdirs-aware path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain import paths


def test_get_home_returns_a_path(clean_env: None) -> None:
    """get_home() returns a pathlib.Path."""
    result = paths.get_home()
    assert isinstance(result, Path)


def test_get_home_respects_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NELLBRAIN_HOME env var fully overrides the platformdirs default."""
    override = tmp_path / "custom_home"
    monkeypatch.setenv("NELLBRAIN_HOME", str(override))
    result = paths.get_home()
    assert result == override.resolve()


def test_get_home_falls_back_to_platformdirs(clean_env: None) -> None:
    """Without NELLBRAIN_HOME, get_home() returns a concrete Path."""
    result = paths.get_home()
    # Just verify it's a Path and is absolute; platform-specific
    # content varies
    assert isinstance(result, Path)
    assert result.is_absolute()


def test_get_persona_dir_nests_under_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """get_persona_dir('nell') returns <home>/personas/nell."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    result = paths.get_persona_dir("nell")
    assert result == tmp_path.resolve() / "personas" / "nell"


def test_get_persona_dir_handles_multiple_personas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Different persona names resolve to different dirs under /personas/."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    nell = paths.get_persona_dir("nell")
    sage = paths.get_persona_dir("sage")
    assert nell != sage
    assert nell.parent == sage.parent


def test_get_cache_dir_is_path(clean_env: None) -> None:
    """get_cache_dir() returns a Path."""
    result = paths.get_cache_dir()
    assert isinstance(result, Path)


def test_get_log_dir_is_path(clean_env: None) -> None:
    """get_log_dir() returns a Path."""
    result = paths.get_log_dir()
    assert isinstance(result, Path)
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/test_paths.py -v
```

Expected: 7 errors like `ModuleNotFoundError: No module named 'brain.paths'`.

- [ ] **Step 3: Write `brain/paths.py`**

Create `/Users/hanamori/companion-emergence/brain/paths.py`:

```python
"""Platform-aware path resolution for companion-emergence.

All user-facing paths route through this module so we never hard-code
OS-specific locations. Uses platformdirs for the OS-appropriate default
with NELLBRAIN_HOME env var for full override.
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import PlatformDirs

_APP_NAME = "companion-emergence"
_APP_AUTHOR = "hanamorix"

_dirs = PlatformDirs(appname=_APP_NAME, appauthor=_APP_AUTHOR)


def get_home() -> Path:
    """Root directory for all companion-emergence state.

    Resolution order:
    1. NELLBRAIN_HOME env var if set (supports ~ expansion)
    2. platformdirs user_data_dir for the current OS
    """
    override = os.environ.get("NELLBRAIN_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path(_dirs.user_data_dir)


def get_persona_dir(name: str) -> Path:
    """Return the directory for a specific persona's private data."""
    return get_home() / "personas" / name


def get_cache_dir() -> Path:
    """Return the cache directory (embeddings, computed matrices, etc)."""
    return Path(_dirs.user_cache_dir)


def get_log_dir() -> Path:
    """Return the log file directory."""
    return Path(_dirs.user_log_dir)
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/test_paths.py -v
```

Expected: `7 passed`.

- [ ] **Step 5: Run ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/paths.py tests/unit/brain/test_paths.py
git commit -m "feat(brain/paths): platformdirs-aware path resolution

NELLBRAIN_HOME env var overrides; otherwise platformdirs picks the
OS-appropriate default. Covers home, persona, cache, and log dirs.
7 tests green on macOS; will validate Windows + Linux via CI in Task 7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `brain/config.py` — three-source precedence (TDD)

**Goal:** Config merges from `persona.toml` → `.env` → env vars, with each value's source traced. Spec Section 4.5.

**Files:**
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/test_config.py`
- Create: `/Users/hanamori/companion-emergence/brain/config.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/test_config.py`:

```python
"""Tests for brain.config — three-source config merge."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain import config


def _write_persona(persona_dir: Path, toml_body: str) -> None:
    """Helper: write persona.toml with the given body to a persona dir."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    (persona_dir / "persona.toml").write_text(toml_body)


def test_persona_toml_provides_baseline(
    tmp_path: Path, clean_env: None
) -> None:
    """Values from persona.toml become the baseline config."""
    persona_dir = tmp_path / "nell"
    _write_persona(
        persona_dir,
        """
[bridge]
bind = "127.0.0.1:9000"

[model]
provider = "ollama"
tag = "my-model"
""",
    )

    result = config.load_config(persona_dir)
    assert result.bridge_bind == "127.0.0.1:9000"
    assert result.provider == "ollama"
    assert result.model == "my-model"
    assert result.source_trace["BRIDGE_BIND"] == "persona.toml"
    assert result.source_trace["PROVIDER"] == "persona.toml"
    assert result.source_trace["MODEL"] == "persona.toml"


def test_env_var_overrides_persona_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment variables override persona.toml values."""
    persona_dir = tmp_path / "nell"
    _write_persona(
        persona_dir,
        """
[bridge]
bind = "127.0.0.1:9000"
""",
    )

    monkeypatch.setenv("BRIDGE_BIND", "127.0.0.1:8000")

    result = config.load_config(persona_dir)
    assert result.bridge_bind == "127.0.0.1:8000"
    assert result.source_trace["BRIDGE_BIND"] == "env"


def test_env_file_overrides_persona_toml(
    tmp_path: Path, clean_env: None
) -> None:
    """A .env file overrides persona.toml when no env var is set."""
    persona_dir = tmp_path / "nell"
    _write_persona(
        persona_dir,
        '[model]\nprovider = "from-toml"\n',
    )

    env_file = tmp_path / ".env"
    env_file.write_text("PROVIDER=from-env-file\n")

    result = config.load_config(persona_dir, env_file=env_file)
    assert result.provider == "from-env-file"
    assert result.source_trace["PROVIDER"] == ".env"


def test_env_var_beats_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env var takes precedence over .env file."""
    persona_dir = tmp_path / "nell"
    _write_persona(persona_dir, '[model]\nprovider = "from-toml"\n')

    env_file = tmp_path / ".env"
    env_file.write_text("PROVIDER=from-env-file\n")

    monkeypatch.setenv("PROVIDER", "from-env-var")

    result = config.load_config(persona_dir, env_file=env_file)
    assert result.provider == "from-env-var"
    assert result.source_trace["PROVIDER"] == "env"


def test_sensible_defaults_when_nothing_configured(
    tmp_path: Path, clean_env: None
) -> None:
    """When no config present, sensible defaults apply."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()

    result = config.load_config(persona_dir)
    assert result.bridge_bind == "127.0.0.1:8765"
    assert result.provider == "ollama"
    assert result.model == ""


def test_env_file_ignores_comments_and_blank_lines(
    tmp_path: Path, clean_env: None
) -> None:
    """.env parser skips # comments and blank lines."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()

    env_file = tmp_path / ".env"
    env_file.write_text(
        "# this is a comment\n"
        "\n"
        "PROVIDER=openai\n"
        "# another comment\n"
        'MODEL="claude-sonnet-4"\n'
    )

    result = config.load_config(persona_dir, env_file=env_file)
    assert result.provider == "openai"
    assert result.model == "claude-sonnet-4"


def test_persona_name_derived_from_dir(
    tmp_path: Path, clean_env: None
) -> None:
    """persona_name on the Config matches the persona_dir basename."""
    persona_dir = tmp_path / "sage"
    persona_dir.mkdir()

    result = config.load_config(persona_dir)
    assert result.persona_name == "sage"
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/test_config.py -v
```

Expected: 7 errors like `ModuleNotFoundError: No module named 'brain.config'`.

- [ ] **Step 3: Write `brain/config.py`**

Create `/Users/hanamori/companion-emergence/brain/config.py`:

```python
"""Configuration loader for companion-emergence.

Merges settings from three sources in increasing priority:

1. persona/<name>/persona.toml  - persona defaults (baseline)
2. .env in repo root            - local overrides
3. Environment variables        - runtime overrides (highest priority)

Each resolved value is traced so startup can report the effective source.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Config keys supported by the framework. Extended in later weeks.
_SUPPORTED_KEYS: tuple[str, ...] = (
    "BRIDGE_BIND",
    "PROVIDER",
    "MODEL",
    "NELL_IPC_JID",
)


@dataclass
class Config:
    """Resolved configuration for a persona session.

    Attributes:
        persona_name: Name of the active persona (derived from dir basename).
        bridge_bind: Host:port the bridge listens on.
        provider: LLM provider key (ollama, claude, openai, kimi).
        model: Model tag for the active provider.
        source_trace: Maps each resolved key to the source that provided it.
    """

    persona_name: str = ""
    bridge_bind: str = "127.0.0.1:8765"
    provider: str = "ollama"
    model: str = ""
    source_trace: dict[str, str] = field(default_factory=dict)


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file. No shell expansion, no exports."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _load_persona_toml(persona_dir: Path) -> dict[str, str]:
    """Flatten persona.toml into the framework's config key namespace."""
    toml_path = persona_dir / "persona.toml"
    if not toml_path.exists():
        return {}
    with toml_path.open("rb") as fh:
        data = tomllib.load(fh)

    flat: dict[str, str] = {}
    bridge = data.get("bridge", {})
    if "bind" in bridge:
        flat["BRIDGE_BIND"] = str(bridge["bind"])
    model = data.get("model", {})
    if "provider" in model:
        flat["PROVIDER"] = str(model["provider"])
    if "tag" in model:
        flat["MODEL"] = str(model["tag"])
    return flat


def _read_env_vars() -> dict[str, str]:
    """Pull supported keys from os.environ."""
    return {k: v for k, v in os.environ.items() if k in _SUPPORTED_KEYS}


def load_config(
    persona_dir: Path, env_file: Path | None = None
) -> Config:
    """Load and merge config from persona.toml + .env + env vars.

    Args:
        persona_dir: Directory containing persona.toml and persona data.
        env_file: Optional path to a .env file to load.

    Returns:
        Resolved Config with source_trace recording where each value came from.
    """
    sources: list[tuple[str, dict[str, str]]] = [
        ("persona.toml", _load_persona_toml(persona_dir)),
    ]
    if env_file is not None:
        sources.append((".env", _load_env_file(env_file)))
    sources.append(("env", _read_env_vars()))

    trace: dict[str, str] = {}
    merged: dict[str, str] = {}
    for source_name, source_values in sources:
        for key, value in source_values.items():
            merged[key] = value
            trace[key] = source_name

    return Config(
        persona_name=persona_dir.name,
        bridge_bind=merged.get("BRIDGE_BIND", "127.0.0.1:8765"),
        provider=merged.get("PROVIDER", "ollama"),
        model=merged.get("MODEL", ""),
        source_trace=trace,
    )
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/test_config.py -v
```

Expected: `7 passed`.

- [ ] **Step 5: Run ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/config.py tests/unit/brain/test_config.py
git commit -m "feat(brain/config): 3-source precedence (toml > .env > env var)

Implements spec Section 4.5 config precedence. Every resolved value
carries a source_trace so startup can log effective config origin.
7 tests green covering each source independently and precedence order.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `brain/cli.py` — subcommand stubs (TDD extension)

**Goal:** Extend the CLI with stub subcommands for every future entry point. Running `nell <cmd>` prints a "not implemented yet" message and returns 0. This gives the framework a stable CLI surface Week 1 forward.

**Files:**
- Modify: `/Users/hanamori/companion-emergence/brain/cli.py`
- Modify: `/Users/hanamori/companion-emergence/tests/unit/brain/test_cli.py`

- [ ] **Step 1: Add failing tests for stub subcommands**

Append to `/Users/hanamori/companion-emergence/tests/unit/brain/test_cli.py`:

```python


STUB_COMMANDS = [
    "supervisor",
    "dream",
    "heartbeat",
    "reflex",
    "status",
    "rest",
    "soul",
    "memory",
    "works",
    "migrate",
]


@pytest.mark.parametrize("name", STUB_COMMANDS)
def test_stub_subcommand_runs_and_reports_not_implemented(
    capsys: pytest.CaptureFixture[str], name: str
) -> None:
    """Every stub subcommand exits 0 and prints 'not implemented yet'."""
    result = cli.main([name])
    assert result == 0
    captured = capsys.readouterr()
    assert "not implemented" in captured.out.lower()
    assert name in captured.out


def test_stub_subcommand_help_works(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each stub subcommand supports --help without crashing."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["supervisor", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "supervisor" in captured.out.lower()
```

- [ ] **Step 2: Run the new tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/test_cli.py -v
```

Expected: 10 parametrized failures + 1 help test failure. Old tests still pass.

- [ ] **Step 3: Replace `brain/cli.py` with the extended version**

Overwrite `/Users/hanamori/companion-emergence/brain/cli.py`:

```python
"""Entry point CLI for companion-emergence.

Invoked as `nell <subcommand> [options]`. Week 1 ships `--version`, help,
and a set of stub subcommands that print "not implemented yet" so the CLI
surface is stable while subsequent weeks fill in functionality.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from brain import __version__

# Subcommands the framework plans to ship. Each is a stub in Week 1;
# filled in across Weeks 2-8 as respective modules come online.
_STUB_COMMANDS: tuple[str, ...] = (
    "supervisor",
    "dream",
    "heartbeat",
    "reflex",
    "status",
    "rest",
    "soul",
    "memory",
    "works",
    "migrate",
)


def _make_stub(name: str) -> Callable[[argparse.Namespace], int]:
    """Factory: build a stub command handler that prints + returns 0."""

    def _handler(args: argparse.Namespace) -> int:
        print(
            f"nell {name} — not implemented yet. "
            "This subcommand is wired in a future week per the implementation plan."
        )
        return 0

    return _handler


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser with all stub subcommands."""
    parser = argparse.ArgumentParser(
        prog="nell",
        description=(
            "companion-emergence — CLI for building emotionally aware "
            "AI companions"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"companion-emergence {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", title="subcommands")

    for name in _STUB_COMMANDS:
        sub = subparsers.add_parser(
            name,
            help=f"(stub) {name} — wired in a later week",
        )
        sub.set_defaults(func=_make_stub(name))

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns shell exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run all CLI tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/test_cli.py -v
```

Expected: `13 passed` (original 2 + 10 stubs + 1 stub-help).

- [ ] **Step 5: Manual smoke test — `nell status`**

```bash
cd /Users/hanamori/companion-emergence
uv run nell status
```

Expected output starting with: `nell status — not implemented yet.`

- [ ] **Step 6: Run ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/cli.py tests/unit/brain/test_cli.py
git commit -m "feat(brain/cli): stub subcommands for supervisor/dream/heartbeat/...

10 stub subcommands registered. Each prints 'not implemented yet' and
exits 0. Gives the framework a stable CLI surface — subsequent weeks
fill in each one without changing the entry-point API.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Starter persona template — `examples/starter-thoughtful/`

**Goal:** One complete starter persona a forker can copy and edit. Validates that the schema we've been specifying actually works as empty-but-valid JSON/TOML files. Subsequent weeks add `starter-creative/` and `starter-steady/` following this template.

**Files:**
- Create: `/Users/hanamori/companion-emergence/examples/starter-thoughtful/persona.toml`
- Create: `/Users/hanamori/companion-emergence/examples/starter-thoughtful/personality.json`
- Create: `/Users/hanamori/companion-emergence/examples/starter-thoughtful/soul.json`
- Create: `/Users/hanamori/companion-emergence/examples/starter-thoughtful/self_model.json`
- Create: `/Users/hanamori/companion-emergence/examples/starter-thoughtful/voice.json`
- Create: `/Users/hanamori/companion-emergence/examples/starter-thoughtful/emotions/extensions.json`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/test_starter_persona.py`

- [ ] **Step 1: Write the failing validation test**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/test_starter_persona.py`:

```python
"""Integration test: the shipped starter personas load cleanly."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from brain import config

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_STARTER_THOUGHTFUL = _REPO_ROOT / "examples" / "starter-thoughtful"


def test_starter_thoughtful_dir_exists() -> None:
    """The starter-thoughtful example directory is present."""
    assert _STARTER_THOUGHTFUL.is_dir()


def test_starter_thoughtful_persona_toml_parses() -> None:
    """persona.toml is valid TOML."""
    toml_path = _STARTER_THOUGHTFUL / "persona.toml"
    assert toml_path.exists()
    with toml_path.open("rb") as fh:
        data = tomllib.load(fh)
    assert data["persona"]["type"] == "starter"


@pytest.mark.parametrize(
    "filename",
    [
        "personality.json",
        "soul.json",
        "self_model.json",
        "voice.json",
        "emotions/extensions.json",
    ],
)
def test_starter_thoughtful_json_files_parse(filename: str) -> None:
    """All shipped JSON files in the starter persona are valid JSON."""
    json_path = _STARTER_THOUGHTFUL / filename
    assert json_path.exists()
    with json_path.open() as fh:
        json.load(fh)


def test_starter_thoughtful_loads_via_brain_config(
    clean_env: None,
) -> None:
    """brain.config.load_config accepts the starter persona dir."""
    result = config.load_config(_STARTER_THOUGHTFUL)
    assert result.persona_name == "starter-thoughtful"
    assert result.bridge_bind == "127.0.0.1:8765"
    assert result.provider == "ollama"
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/test_starter_persona.py -v
```

Expected: 7 failures because no files exist yet.

- [ ] **Step 3: Create the `examples/starter-thoughtful/` directory and all files**

```bash
mkdir -p /Users/hanamori/companion-emergence/examples/starter-thoughtful/emotions
```

Create `/Users/hanamori/companion-emergence/examples/starter-thoughtful/persona.toml`:

```toml
# A starter persona: measured pace, literary voice, low arousal ceiling.
# Forkers copy this directory to persona/<their-name>/ and edit fields marked FORKER.

[persona]
name = "starter-thoughtful"
pronouns = "they/them"
type = "starter"
created = ""

[model]
# FORKER: set this to your Ollama model tag, or switch provider entirely.
provider = "ollama"
tag = ""
fallback_provider = ""
fallback_model = ""

[rhythms]
heartbeat_seconds = 90
dream_hour = 3
reflex_enabled = true
growth_weekly_hour = "sunday 02:00"

[privacy]
# All integrations are opt-in. Enable deliberately; defaults protect by default.
obsidian_integration = false
ipc_integration = false
filesystem_scan = false

[bridge]
bind = "127.0.0.1:8765"
token_file = "data/bridge.token"

[developmental]
# Forced phase override. Leave empty to let the framework compute from data depth.
phase_override = ""
```

Create `/Users/hanamori/companion-emergence/examples/starter-thoughtful/personality.json`:

```json
{
  "version": "1.0",
  "created": null,
  "daily_rhythms": {},
  "idiosyncrasies": {},
  "deeper_traits": {},
  "voice_modifiers": {},
  "preferences": {},
  "voice_state": {}
}
```

Create `/Users/hanamori/companion-emergence/examples/starter-thoughtful/soul.json`:

```json
{
  "created": null,
  "crystallizations": []
}
```

Create `/Users/hanamori/companion-emergence/examples/starter-thoughtful/self_model.json`:

```json
{
  "generated_at": null,
  "observation_window_days": 30,
  "self_description": "",
  "self_claims": [],
  "behavioral_summary": {
    "type_counts": {},
    "top_emotions": [],
    "top_topics": []
  },
  "soul_themes": [],
  "creative_tendencies": [],
  "network_summary": {
    "formed_last_tick": 0,
    "strengthened_last_tick": 0
  },
  "prior_model": null
}
```

Create `/Users/hanamori/companion-emergence/examples/starter-thoughtful/voice.json`:

```json
{
  "schema_version": "1.0",
  "baseline": null,
  "per_state": {}
}
```

Create `/Users/hanamori/companion-emergence/examples/starter-thoughtful/emotions/extensions.json`:

```json
{
  "proposed": [],
  "accepted": []
}
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/test_starter_persona.py -v
```

Expected: `7 passed`.

- [ ] **Step 5: Run full test suite to catch regressions**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
```

Expected: All tests pass (Task 1 + 2 + 3 + 4 + 5 counts combined).

- [ ] **Step 6: Run ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add examples/ tests/unit/brain/test_starter_persona.py
git commit -m "feat(examples): starter-thoughtful persona template

One complete starter persona covering persona.toml + personality.json +
soul.json + self_model.json + voice.json + emotions/extensions.json.
Validates the schema loads cleanly through brain.config. Forkers copy
this directory as the starting point for their own persona.

starter-creative and starter-steady follow this template in later weeks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `.env.example` + `.gitattributes`

**Goal:** `.env.example` documents env vars a user might set; `.gitattributes` normalises line endings to prevent CRLF/LF edit wars between macOS and Windows contributors.

**Files:**
- Create: `/Users/hanamori/companion-emergence/.env.example`
- Create: `/Users/hanamori/companion-emergence/.gitattributes`

- [ ] **Step 1: Write `.env.example`**

Create `/Users/hanamori/companion-emergence/.env.example`:

```
# ═══════════════════════════════════════════════════════════
# companion-emergence environment configuration
# Copy this to `.env` and fill in your values. `.env` is gitignored.
# ═══════════════════════════════════════════════════════════

# ── Core framework ────────────────────────────────────────
# Override the default platformdirs location for all state.
# Useful for testing, for running multiple independent instances,
# or for pinning state to a specific disk.
# NELLBRAIN_HOME=

# ── Bridge ────────────────────────────────────────────────
# Host:port the bridge daemon binds to (default: 127.0.0.1:8765)
# BRIDGE_BIND=

# ── LLM provider selection ────────────────────────────────
# Overrides persona.toml's [model] section at runtime.
# PROVIDER=ollama
# MODEL=nell-stage13-voice

# ── IPC integration (optional, opt-in) ────────────────────
# For NanoClaw / WhatsApp outbox integration. Leave unset to
# disable; set to your JID to enable outbox→WhatsApp delivery.
# Format for WhatsApp: <country-code><number>@s.whatsapp.net
# NELL_IPC_JID=

# ── Provider API keys (if using commercial LLM) ───────────
# ANTHROPIC_API_KEY=
# OPENAI_API_KEY=
# KIMI_API_KEY=
```

- [ ] **Step 2: Write `.gitattributes`**

Create `/Users/hanamori/companion-emergence/.gitattributes`:

```
# Normalise line endings. Prevents CRLF vs LF edit-war churn between
# Hana's macOS machine and her Windows test machine.
* text=auto

# Binary files stay binary.
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.ico binary
*.pdf binary
*.gguf binary
*.bin binary
*.safetensors binary
*.pt binary
*.pth binary
*.ckpt binary
*.npy binary
```

- [ ] **Step 3: Verify files are valid (no test needed, just read-back)**

```bash
cd /Users/hanamori/companion-emergence
test -s .env.example && echo ".env.example OK"
test -s .gitattributes && echo ".gitattributes OK"
```

Expected: both lines print OK.

- [ ] **Step 4: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add .env.example .gitattributes
git commit -m "chore: .env.example + .gitattributes

.env.example documents every env var the framework reads (NELLBRAIN_HOME,
BRIDGE_BIND, PROVIDER, MODEL, NELL_IPC_JID, provider API keys).

.gitattributes normalises line endings via * text=auto; prevents CRLF/LF
edit-war churn between macOS and Windows. Binary extensions listed
explicitly to avoid corruption.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: CI matrix — GitHub Actions on macOS + Windows + Linux

**Goal:** Every PR runs pytest + ruff on macos-latest, windows-latest, and ubuntu-latest. Green required before merge. This is the enforcement mechanism for spec Section 14.8 Cross-platform discipline — if Windows breaks, CI catches it per-PR instead of at the end of the rebuild.

**Files:**
- Create: `/Users/hanamori/companion-emergence/.github/workflows/test.yml`

- [ ] **Step 1: Create the workflows directory**

```bash
mkdir -p /Users/hanamori/companion-emergence/.github/workflows
```

- [ ] **Step 2: Write `test.yml`**

Create `/Users/hanamori/companion-emergence/.github/workflows/test.yml`:

```yaml
name: test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    name: ${{ matrix.os }} / py ${{ matrix.python-version }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.12"]
    runs-on: ${{ matrix.os }}

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run pytest
        run: uv run pytest -v --tb=short

      - name: Lint with ruff
        run: uv run ruff check .
```

- [ ] **Step 3: Commit locally**

```bash
cd /Users/hanamori/companion-emergence
git add .github/workflows/test.yml
git commit -m "ci: pytest + ruff matrix on macOS/Windows/Linux

Implements spec Section 14.8 — cross-platform discipline enforced at
the PR boundary. Every push and PR runs pytest and ruff on all three
OSes. Green required before merge; no 'fix Windows later' path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Push to GitHub and watch the CI run**

```bash
cd /Users/hanamori/companion-emergence
git push origin main
```

- [ ] **Step 5: Verify CI green on all three OSes**

```bash
cd /Users/hanamori/companion-emergence
# Wait ~2 minutes for the run to start, then check
gh run list --limit 1
gh run watch  # interactive; or use `gh run view --log` for the latest
```

Expected: run completes with `✓ success` across ubuntu-latest, macos-latest, windows-latest. If any OS fails, diagnose with `gh run view --log-failed`, fix, commit, and push again until all three are green.

---

## Task 8: Week 1 green-light verification

**Goal:** Prove every piece of the Week 1 deliverable works end-to-end. This is the gate that closes Week 1 — everything in Section 16 Week 1's bullet list is demonstrated, and CI is green.

- [ ] **Step 1: Clean install from scratch**

```bash
cd /Users/hanamori/companion-emergence
rm -rf .venv uv.lock
uv sync --all-extras
```

Expected: fresh `.venv` created, `uv.lock` regenerated, all deps install without errors. Proves the repo is cloneable and installable on a fresh machine.

- [ ] **Step 2: Verify CLI entry point**

```bash
cd /Users/hanamori/companion-emergence
uv run nell --version
uv run nell supervisor
uv run nell --help
```

Expected:
- `nell --version` prints `companion-emergence 0.0.1`
- `nell supervisor` prints `nell supervisor — not implemented yet. ...`
- `nell --help` prints usage with all 10 stub subcommands listed

- [ ] **Step 3: Full pytest suite green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
```

Expected: All tests pass. Count is approximately 35 across the matrix: 2 from Task 1 CLI (version + help) + 7 from paths + 7 from config + 11 from CLI stubs (10 parametrized + 1 help) + 8 from starter persona (1 dir + 1 toml + 5 parametrized json + 1 load-via-config).

- [ ] **Step 4: Lint clean**

```bash
cd /Users/hanamori/companion-emergence
uv run ruff check .
uv run ruff format --check .
```

Expected: both report clean.

- [ ] **Step 5: CI green on the latest commit**

```bash
cd /Users/hanamori/companion-emergence
gh run list --limit 1 --json status,conclusion,name
```

Expected: `"status": "completed"` and `"conclusion": "success"`.

- [ ] **Step 6: Tag Week 1 complete**

```bash
cd /Users/hanamori/companion-emergence
git tag -a week-1-complete -m "Week 1 scaffolding complete

Deliverables per spec Section 16 Week 1:
- Repo cloneable + installable via uv sync
- CLI entry point runs (nell --version + stubs)
- pyproject.toml configured
- brain/paths.py platformdirs-aware (tested)
- brain/config.py 3-source precedence (tested)
- brain/cli.py with 10 stub subcommands (tested)
- examples/starter-thoughtful/ full template (tested)
- .env.example documented
- .gitattributes normalising line endings
- CI matrix green on macOS + Windows + Linux

Total tests: ~34 passing across the matrix.
Week 2 opens with the emotion package."
git push origin week-1-complete
```

- [ ] **Step 7: Update memory index**

Append a new entry to `/Users/hanamori/.claude/projects/-Users-hanamori-nanoclaw/memory/MEMORY.md`:

```
- [companion-emergence Week 1 complete](project_companion_emergence_week_1_complete.md) — 2026-04-XX: scaffolding landed, CI green on 3 OSes, tagged week-1-complete. Next: Week 2 emotion package.
```

And create the supporting memory file at `/Users/hanamori/.claude/projects/-Users-hanamori-nanoclaw/memory/project_companion_emergence_week_1_complete.md` with relevant details.

---

## Week 1 green-light criterion

Week 1 is green when ALL of the following are true:

1. `uv sync --all-extras` succeeds on a fresh clone on macOS
2. `uv run nell --version` prints the version
3. `uv run pytest -v` all pass
4. `uv run ruff check .` reports clean
5. GitHub Actions CI shows `✓ success` on macos-latest AND windows-latest AND ubuntu-latest for the latest commit on main
6. Hana has cloned the repo on her Windows machine (or verified via CI) and confirmed it installs + runs there too
7. Tag `week-1-complete` is pushed

When all seven are true, Week 2 begins with `superpowers:writing-plans` on Section 15.x (the emotion package).

---

## Notes for the engineer executing this plan

- **Do not invent work.** If a step says "run X and expect Y," run X and check for Y. If Y doesn't happen, diagnose — don't push forward.
- **Commit at the end of every task.** The plan's commits are the unit of rollback if something goes wrong.
- **Do not skip ruff.** If ruff reports issues, fix them before committing. Clean ruff output is a precondition for CI green; catching lint issues locally saves CI cycles.
- **Watch Windows CI especially closely** on the first push. If it fails and local-macOS passed, it's usually a path-handling issue (backslash vs forward slash) or a line-ending issue the `.gitattributes` should have caught.
- **If a test starts failing unexpectedly after a commit,** check you haven't broken a dependency from a prior task. The test count should only grow; it should never shrink.

---

*End of Week 1 plan.*
