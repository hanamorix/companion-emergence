"""Crystallizers — pure functions that examine brain state and return proposals.

Each crystallizer takes the brain's current state (MemoryStore + relevant
engine context) and returns a list of `*Proposal` objects. The scheduler
applies them atomically.

Phase 2a ships a no-op vocabulary crystallizer (returns []). Phase 2b
populates the body with pattern-matching against memories + relational
dynamics + LLM-mediated naming.
"""
