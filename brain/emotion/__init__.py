"""The emotional core — organising principle of companion-emergence.

Sub-modules, each with a single responsibility:
- vocabulary: typed emotion taxonomy + persona extension registry
- state: current emotional state (dict + dominant)
- arousal: 7-tier body-coupled arousal spectrum
- influence: state → biasing hints for provider abstraction
- expression: state → face/voice parameters for NellFace

See spec Section 5 for design rationale.
"""

from brain.emotion.vocabulary import Emotion, by_category, get, list_all, register

__all__ = ["Emotion", "get", "list_all", "by_category", "register"]
