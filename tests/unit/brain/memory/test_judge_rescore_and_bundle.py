"""F2c inc5b-2 — the re-score store extraction + the peft/datasets bundle.

Covers: the accept-path re-score set (`judge_knob_refit_rescore_items`) and
the 2B doc-having subset (`rows_with_doc_snapshot`); plus C19 (peft/datasets
are base deps) + C20 (no residual optional-extra plumbing in brain/).
"""

from __future__ import annotations

from pathlib import Path

from brain.memory.store import MemoryStore


def _seed(store, *, query, docs, local_labels, haiku_labels, raw_scores):
    n = len(local_labels)
    store.log_calibration_sample(
        query=query,
        candidate_ids=[f"{query}-m{i}" for i in range(n)],
        reranker_scores=[1.0] * n,
        reranker_model_id="jina",
        candidate_docs=docs,
    )
    row_id = store._conn.execute(
        "SELECT id FROM calibration_log ORDER BY id DESC LIMIT 1"
    ).fetchone()["id"]
    store.write_calibration_labels(row_id, local_labels, haiku_labels, raw_scores)
    return row_id


def test_rescore_items_broad_effective_label_doc_required() -> None:
    store = MemoryStore(db_path=":memory:")
    # Haiku-over-local precedence; local-only positions ARE included (broad);
    # unknown/error skipped; doc-absent position skipped.
    r = _seed(
        store, query="q", docs=["da", "db", "dc", ""],
        local_labels=["relevant", "relevant", "unknown", "relevant"],
        haiku_labels=[None, "irrelevant", None, None],
        raw_scores=[0.5, -0.5, None, 0.2],
    )
    items = store.judge_knob_refit_rescore_items([r])
    # idx0: local "relevant" (broad, no haiku) + doc "da" -> included
    # idx1: haiku "irrelevant" overrides local -> ("q","db","irrelevant")
    # idx2: "unknown" -> skipped
    # idx3: doc is "" (empty) -> skipped (cannot forward-pass)
    assert items == [("q", "da", "relevant"), ("q", "db", "irrelevant")]


def test_rescore_items_skips_doc_absent_legacy_row() -> None:
    store = MemoryStore(db_path=":memory:")
    r = _seed(
        store, query="legacy", docs=None,
        local_labels=["relevant"], haiku_labels=["relevant"], raw_scores=[0.9],
    )
    assert store.judge_knob_refit_rescore_items([r]) == []


def test_rescore_items_empty_and_ordered() -> None:
    store = MemoryStore(db_path=":memory:")
    assert store.judge_knob_refit_rescore_items([]) == []
    r1 = _seed(store, query="a", docs=["d1"], local_labels=["relevant"],
               haiku_labels=[None], raw_scores=[0.1])
    r2 = _seed(store, query="b", docs=["d2"], local_labels=["irrelevant"],
               haiku_labels=[None], raw_scores=[0.2])
    items = store.judge_knob_refit_rescore_items([r2, r1])  # order by row id, not arg order
    assert items == [("a", "d1", "relevant"), ("b", "d2", "irrelevant")]


def test_rows_with_doc_snapshot_returns_only_doc_having() -> None:
    store = MemoryStore(db_path=":memory:")
    r_doc = _seed(store, query="d", docs=["x"], local_labels=["relevant"],
                  haiku_labels=["relevant"], raw_scores=[0.5])
    r_absent = _seed(store, query="l", docs=None, local_labels=["relevant"],
                     haiku_labels=["relevant"], raw_scores=[0.5])
    got = store.rows_with_doc_snapshot([r_doc, r_absent])
    assert got == [r_doc]
    assert store.rows_with_doc_snapshot([]) == []


# --- C19 / C20: the peft/datasets bundle reversal (now base deps) -----------


def test_peft_and_datasets_are_base_deps_importable() -> None:
    # C19: base deps (not an optional f2c-training extra) -> importable in the
    # plain env (no --extra needed). A missing one is now a broken install.
    import datasets  # noqa: F401
    import peft  # noqa: F401


def test_no_residual_optional_extra_plumbing_in_brain() -> None:
    # C20: the inc5a optional-extra plumbing is fully removed from brain/.
    # Positive absence sweep (would fire on the pre-change tree, which had
    # `lora_available` / `_downgrade_for_missing_lora_extra` / the extra hint).
    brain_root = Path(__file__).resolve().parents[4] / "brain"
    banned = ["lora_available", "_downgrade_for_missing_lora_extra", "_F2C_TRAINING_EXTRA_HINT"]
    offenders: list[str] = []
    for py in brain_root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                offenders.append(f"{py}: {token}")
    assert not offenders, f"residual optional-extra plumbing: {offenders}"
