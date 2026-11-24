"""LLM provider abstraction for companion-emergence engines.

Exports LLMProvider ABC + three concrete providers + factory.
Its design (the 2026-04-23 week-4 dream-engine design) was never committed and is lost (#797).
"""

from brain.bridge.provider import (
    ClaudeCliProvider,
    FakeProvider,
    LLMProvider,
    OllamaProvider,
    get_provider,
)

__all__ = [
    "ClaudeCliProvider",
    "FakeProvider",
    "LLMProvider",
    "OllamaProvider",
    "get_provider",
]
