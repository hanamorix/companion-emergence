"""brain.chat — SP-6 chat engine.

Public surface: respond(), ChatResult, SessionState, create_session,
AS_NELL_PREAMBLE.
"""

from brain.chat.engine import ChatResult, respond
from brain.chat.prompt import AS_NELL_PREAMBLE
from brain.chat.session import SessionState, create_session

__all__ = [
    "AS_NELL_PREAMBLE",
    "ChatResult",
    "SessionState",
    "create_session",
    "respond",
]
