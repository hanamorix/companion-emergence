"""Privacy reflection gate (parent design §12) — the safety spine. Reviews every
outbound peer payload: a deterministic pre-filter catches hard PII/credential
leaks without an LLM call; anything subtler goes to a tool-less provider.complete
reflection. No peer-derived content may flip a hold/revise into a send (parent
§5). Fails closed to hold on any uncertainty.

Tool-path isolation (carried from Phase 3): this module imports ONLY stdlib +
{gate, limits, cli_throttle}. The conformance oracle enforces it by AST."""
from __future__ import annotations

import json
import logging
import re

from brain.bridge import cli_throttle as _default_throttle
from brain.kindled_link.gate import GateDecision, OutboundPayload

log = logging.getLogger(__name__)

# Hard-leak patterns (structural PII / credential shapes). A hit short-circuits
# to hold/revise with NO provider call. Verbatim-user-quote detection is NOT
# here — the gate never receives user-chat turns — it is a reflection concern.
_PATTERNS = [
    re.compile(r"(?:^|[\s\"'(])/(?:Users|home|etc|var|tmp|private)/\S+"),  # POSIX path
    re.compile(r"[A-Za-z]:\\[\\\w.\- ]+"),                                  # Windows path
    re.compile(r"~/[\w./\- ]+"),                                            # home path
    re.compile(r"file://\S+"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),                                # email
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),                                 # api key
    # credential assignment: require a CREDENTIAL-SHAPED value (>=16 non-space
    # chars) so ordinary introspective prose ("a token: small gesture") does NOT
    # trip — only "api_key = AKIA1234567890ABCD"-shaped strings do (red-team M4).
    re.compile(r"\b(?:bearer|api[_-]?key|token|secret)\b\s*[:=]\s*[A-Za-z0-9_\-]{16,}", re.I),
    re.compile(r"-----BEGIN [A-Z ]*KEY-----"),
]


def _prefilter(text: str) -> GateDecision | None:
    """Return a hold/revise decision on a hard structural leak, else None.
    NEVER returns send."""
    hits = [p.pattern for p in _PATTERNS if p.search(text)]
    if not hits:
        return None
    action = "revise" if len(hits) == 1 else "hold"
    return GateDecision(
        action=action,
        reason="prefilter: structural PII/credential leak",
        revision_constraints=(
            "Remove the file path, address, email, or credential; speak only in "
            "broad non-identifying texture."
        ),
        texture_score=1.0,
    )


class PrivacyGate:
    def __init__(self, *, provider, store, throttle=_default_throttle):
        self._provider = provider
        self._store = store
        self._throttle = throttle

    @staticmethod
    def _payload_text(payload: OutboundPayload) -> str:
        parts = [payload.body or ""]
        if payload.relationship_hint:
            parts.append(json.dumps(payload.relationship_hint, sort_keys=True))
        return "\n".join(parts)
