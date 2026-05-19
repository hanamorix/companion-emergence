"""brain.grief — affective layer over forgetting + narrative-memory loss.

Public surface (filled in by Phase 5):

    handle_drop(*, memory, salience_at_drop, persona_dir, store)
    handle_arc_close(*, arc, persona_dir, store)
    handle_recall_touch(*, touched_ids, graveyard_entries, persona_dir, store,
                        triggering_arc_id=None)

Spec: docs/superpowers/specs/2026-05-19-grief-design.md
"""
