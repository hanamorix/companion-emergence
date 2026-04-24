"""Web search layer — pluggable searcher backends.

Default is DdgsWebSearcher (DuckDuckGo via the `ddgs` library, no API
key). NoopWebSearcher used in tests and CI for zero-network behavior.
"""
