"""Initiate physiology — autonomous outbound channel.

Mirrors the _run_soul_review_tick architecture from v0.0.4. Events emit
candidates into initiate_candidates.jsonl; a supervisor tick reviews
queued candidates with cost-cap + cooldown gates; decisions land in
initiate_audit.jsonl + MemoryStore.

Spec: docs/superpowers/specs/2026-05-11-initiate-physiology-design.md
"""
