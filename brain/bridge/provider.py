"""LLM provider abstraction — ABC + concrete providers + factory."""

from __future__ import annotations

import hashlib
import json
import subprocess
from abc import ABC, abstractmethod

_DEFAULT_TIMEOUT_SECONDS = 300


class LLMProvider(ABC):
    """Abstract LLM provider. Subclasses implement `generate` and `name`."""

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Return the LLM's completion for the given prompt."""

    @abstractmethod
    def name(self) -> str:
        """Return a short provider name (e.g. 'fake', 'claude-cli:sonnet')."""


class FakeProvider(LLMProvider):
    """Deterministic hash-based echo provider for tests — zero network calls."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        seed_input = (system or "").encode("utf-8") + b"||" + prompt.encode("utf-8")
        h = hashlib.sha256(seed_input).hexdigest()[:16]
        return f"DREAM: test dream {h} — an associative thread"

    def name(self) -> str:
        return "fake"


class ClaudeCliProvider(LLMProvider):
    """Shells out to `claude -p <prompt> --output-format json`.

    Uses Hana's Claude Code subscription — no per-token API billing.
    Per the feedback memory: this is the default Claude path for
    companion-emergence and Hana's other projects.
    """

    def __init__(
        self,
        model: str = "sonnet",
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model
        self._timeout = timeout_seconds

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", self._model]
        if system is not None:
            cmd.extend(["--system-prompt", system])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"ClaudeCliProvider: subprocess timed out after {self._timeout}s"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"ClaudeCliProvider failed (exit {result.returncode}): {result.stderr.strip()}"
            )

        try:
            payload = json.loads(result.stdout)
            return str(payload["result"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"ClaudeCliProvider: unexpected output format: {result.stdout[:200]!r}"
            ) from exc

    def name(self) -> str:
        return f"claude-cli:{self._model}"


class OllamaProvider(LLMProvider):
    """Placeholder for local Ollama integration.

    Stub until Hana's local Ollama stack is back up. Fill in by:
    1. Replacing raise with an httpx POST to {host}/api/generate
    2. Parsing the streamed/non-streamed response
    3. Adding the httpx dep to pyproject
    """

    def __init__(self, model: str = "nell-dpo", host: str = "http://localhost:11434") -> None:
        self._model = model
        self._host = host

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise NotImplementedError(
            "OllamaProvider is a stub; fill in when the local Ollama stack is available."
        )

    def name(self) -> str:
        return f"ollama:{self._model}"


def get_provider(name: str) -> LLMProvider:
    """Resolve a provider identifier to an instance. Raises ValueError on unknown."""
    if name == "fake":
        return FakeProvider()
    if name == "claude-cli":
        return ClaudeCliProvider()
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unknown provider: {name!r}")
