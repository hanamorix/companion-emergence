"""Full fine-tune weight-retrain MECHANISM for the judge self-tune's
BEEFY-RAM-tier grade (F2c inc6; spec `f2c-judge-selftune-spec.md` §1 [tier
table: beefy = full fine-tune + knob], §5 [tiered weight-retrain build
detail, "Full FT (beefy)"], §4 [eval + guardrail, via
`judge_eval.run_champion_challenger`'s `retrain_fn`/`RollbackHandle`
contract], AC7-analogue [full-model reload round-trip]).

This is the beefy-tier SIBLING of `judge_lora.py`: same `(query, doc,
label)` triples, same `BinaryCrossEntropyLoss` via sbert 6.1.0's
`CrossEncoderTrainer`, same `retrain_fn` shape that plugs straight into
`judge_eval.run_champion_challenger`. The difference is the update itself: a
FULL fine-tune (ALL parameters trainable, no LoRA, no `peft`).

  - **One evolving model (F2c inc7, spec §1/§5):** training starts from the
    persona's CURRENT model — its own plain checkpoint, or the base judge if
    it has never been weight-tuned — never from the base by default. Because
    every LoRA week is merged in (`judge_lora.build_lora_retrain_fn`), the
    current model is always a plain checkpoint, so a full week simply
    continues from it.
  - **Persistence + reload is the ORDINARY sbert path.** A full fine-tune
    writes a COMPLETE model via `CrossEncoder.save_pretrained` (no
    `adapter_config.json`, no adapter state dict), so it reloads with a plain
    `CrossEncoder(saved_dir)`. `load_full_scorer` below is the single loader
    for every tuned checkpoint (LoRA-merged or fully fine-tuned): champion
    eval, accept-path re-score, and serving.

TORCH-SCOPED, LAZY (I6): `torch`/`sentence_transformers`/`datasets` are
imported LAZILY, function-scoped inside `build_full_ft_retrain_fn`'s and
`load_full_scorer`'s bodies — never at this module's top level — mirroring
`judge_lora.py`'s and `relevance_judge.TorchCrossEncoderJudge`'s exact
convention. Importing this module never pulls torch into `sys.modules`; only
CALLING one of those two functions does. `datasets` is a BASE dependency
(F2c inc5b-2, Roy 2026-09-24 BUNDLE decision), imported lazily here only for
torch-scoping, not because it is optional.

Scope: this module builds the full-FT train/save/reload MECHANISM only. The
per-persona store helpers (`champion_dir`/`staged_adapter_path`/
`swap_champion_pointer`/`resolve_current_checkpoint`/`cleanup_stale_adapters`)
live in `judge_lora.py` and are torch-free; `judge_selftune._run_weight_retrain`
uses the ONE per-persona store for both tiers. It proves everything against a
FROM-SCRATCH TINY model in its tests, never the real bge-reranker-v2-m3
(deferred to real HW).

DURABLE HAIKU-ORACLE NOTE (spec §6, mirrors the notes in `judge_eval.py`,
`judge_selftune.py`, and F2a's `relevance_judge.py` judge/label site): the
`(query, doc, label)` triples this module trains on carry the EFFECTIVE
(Haiku-over-local) label, and the champion/challenger step that gates this
retrain scores both judges against the accumulated HAIKU tie-break decisions
as the oracle. Haiku is the effective relevance
ORACLE this full fine-tune converges the local judge toward, NOT an
independently verified ground truth. A systematic Haiku bias would propagate
into the fine-tuned weights; if a relevance-quality problem shows up
downstream, this trainer (alongside the knob-refit's training data) is one
of the places to look first.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from brain import tunables
from brain.memory.judge_lora import LORA_MAX_LENGTH_DEFAULT
from brain.memory.relevance_judge import label_for_score

logger = logging.getLogger(__name__)

# `(query, doc, label)` triple — label is the EFFECTIVE (Haiku-over-local)
# "relevant"/"irrelevant" label, exactly as `judge_lora.LabeledTriple`. The
# caller (the tick's Haiku-only data-assembly step) filters "unknown"/"error"
# before building `train_items`; this module is content-agnostic.
LabeledTriple = tuple[str, str, str]

# Tunables (I3/I7) — shares the `judge_selftune.*` namespace (one F2c
# feature). PROVISIONAL placeholder default: the real value is F2c inc5's
# deferred timed dry-run's job (spec Open Reconfirmations), not this
# increment's — this exists only so the knob is live/overridable from day
# one. The bounded sequence length reuses the shared
# `judge_selftune.lora_max_length` tunable (see `build_full_ft_retrain_fn`).
FULL_FT_EPOCHS_DEFAULT: int = tunables.register("judge_selftune.full_ft_epochs", 1)


def _label_to_float(label: str) -> float:
    """"relevant" -> 1.0, "irrelevant" -> 0.0 — `BinaryCrossEntropyLoss`'s
    expected target encoding (matches `judge_lora._label_to_float` and
    `relevance_judge.label_for_score`'s two-label vocabulary; any other
    string is a caller bug, not silently coerced)."""
    if label == "relevant":
        return 1.0
    if label == "irrelevant":
        return 0.0
    raise ValueError(f"LabeledTriple label must be 'relevant' or 'irrelevant', got {label!r}")


def _dataset_from_triples(triples: Sequence[LabeledTriple]) -> Any:
    # Deliberately duplicated from judge_lora._dataset_from_triples (3 lines)
    # rather than cross-importing a private helper — keeps the beefy-tier
    # module decoupled from the LoRA module (they share nothing at runtime).
    from datasets import Dataset  # base dep (F2c inc5b-2); lazy for torch-scoping only

    return Dataset.from_dict(
        {
            "sentence1": [q for q, _d, _l in triples],
            "sentence2": [d for _q, d, _l in triples],
            "label": [_label_to_float(label) for _q, _d, label in triples],
        }
    )


# ---------------------------------------------------------------------------
# Training entrypoint (spec §5 "Full FT (beefy)"): a plain CrossEncoder full
# fine-tune + BinaryCrossEntropyLoss via the sbert 6.1.0 CrossEncoderTrainer
# — the SAME loss/data/trainer as the LoRA tier, minus the adapter.
# ---------------------------------------------------------------------------


def build_full_ft_retrain_fn(
    start_model_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    epochs: int | None = None,
    max_length: int | None = None,
    activation_fn: Callable[[Any], Any] | None = None,
    save_full_dir: str | Path | None = None,
) -> Callable[[Sequence[LabeledTriple]], Callable[[tuple[str, str]], str]]:
    """Build a full-FT `retrain_fn` matching `judge_eval.run_champion_
    challenger`'s `Callable[[Sequence[Any]], Callable[[Any], str]]` contract
    — the beefy-tier analogue of `judge_lora.build_lora_retrain_fn`, with the
    SAME shape but a FULL fine-tune (no LoRA adapter).

    `save_full_dir` (mirrors LoRA's `save_dir`): when set,
    `retrain_fn` ALSO saves the trained FULL model to this STAGED directory
    (via `CrossEncoder.save_pretrained` — a complete model, no
    `adapter_config.json`) before returning the label-fn, so the tick can,
    ONLY on an AC6 ACCEPT, atomically swap it into the persona's
    pointer and `load_full_scorer` it (both to re-score the knob on
    the tuned model and to serve it). The label-fn still binds the trained
    IN-MEMORY model for the forward-only eval regardless. `None` (default) =
    no save (in-memory-only).

    `start_model_path` (F2c inc7): the model this week's fine-tune continues
    from (spec §1) — the persona's own plain checkpoint dir, or the base
    judge's model_tier id for a never-weight-tuned persona (the tick's
    `_training_start` decides). TESTS pass a from-scratch tiny local model dir,
    never the real bge.

    Returns `retrain_fn(train_items) -> label_fn`: calling `retrain_fn`
    loads a FRESH `CrossEncoder` from `start_model_path` (ALL params
    trainable — no `add_adapter`), trains it on `train_items` via
    `BinaryCrossEntropyLoss` (`epochs` resolved from the shared tunable when
    not passed), and returns a plain callable `(query, document) ->
    "relevant" | "irrelevant"` bound to the trained IN-MEMORY model — no
    save/reload round-trip in this path (AC6's champion/challenger eval is
    forward-only against the just-trained model; save/reload is the tick's
    concern, for cross-process persistence).

    Never called with `test_items` (AC5 no-leakage contract, enforced by the
    caller — `judge_eval.run_champion_challenger` — not by this function).
    """

    def retrain_fn(train_items: Sequence[LabeledTriple]) -> Callable[[tuple[str, str]], str]:
        # Lazy imports (module docstring: never at this module's top level,
        # to keep torch scoped to actually RUNNING training — I6). No peft /
        # LoraConfig import: this is a FULL fine-tune, not an adapter.
        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
        from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
        from sentence_transformers.cross_encoder.training_args import (
            CrossEncoderTrainingArguments,
        )

        resolved_epochs = (
            epochs
            if epochs is not None
            else tunables.get_tunable("judge_selftune.full_ft_epochs", FULL_FT_EPOCHS_DEFAULT)
        )
        # Bounded sequence length — reuse the SAME shared judge tunable the
        # LoRA tier reads (`judge_selftune.lora_max_length`; the "lora_"
        # prefix is historical — it is the judge's practical bge sequence
        # length, tier-independent). A full model saved via
        # `save_pretrained` persists this `max_length` in its own config, so
        # `load_full_scorer`'s plain `CrossEncoder(dir)` reload restores it
        # automatically — no separate serve-side truncation dance is needed
        # (unlike LoRA, where the serve tokenizer is rebuilt independently).
        resolved_max_length = (
            max_length
            if max_length is not None
            else tunables.get_tunable("judge_selftune.lora_max_length", LORA_MAX_LENGTH_DEFAULT)
        )

        model = CrossEncoder(
            str(start_model_path),
            cache_folder=str(cache_dir) if cache_dir is not None else None,
            config_kwargs={"num_labels": 1},
            max_length=resolved_max_length,
            activation_fn=activation_fn if activation_fn is not None else (lambda x: x),
        )
        # NOTE: no model.add_adapter(...) — every parameter is trainable.

        dataset = _dataset_from_triples(train_items)
        loss_fn = BinaryCrossEntropyLoss(model)

        with tempfile.TemporaryDirectory(prefix="judge_full_ft_train_") as scratch_dir:
            args = CrossEncoderTrainingArguments(
                output_dir=scratch_dir,
                num_train_epochs=resolved_epochs,
                per_device_train_batch_size=min(8, max(1, len(train_items))),
                report_to=[],
                logging_steps=1_000_000,  # effectively silent
                save_strategy="no",
                disable_tqdm=True,
            )
            trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=dataset, loss=loss_fn)
            trainer.train()

        # Persist the trained FULL model to the caller's STAGED dir (outside
        # the deleted scratch dir) so the tick can atomically swap + reload
        # it on an AC6 ACCEPT. A full model save writes NO adapter_config.json
        # (there is no adapter), so the reload is the plain CrossEncoder path.
        if save_full_dir is not None:
            model.save_pretrained(str(save_full_dir))

        def label_fn(item: tuple[str, str]) -> str:
            query, document = item[0], item[1]
            raw_score = float(model.predict([(query, document)], activation_fn=lambda x: x)[0])
            # Route the RAW logit through relevance_judge.label_for_score —
            # the single source of truth for turning a judge logit into a
            # label (identical to judge_lora's label_fn; discard the
            # ambiguous-band flag, which is the live-tick's concern, not the
            # champion/challenger eval's).
            provisional_label, _is_ambiguous = label_for_score(raw_score)
            return provisional_label

        return label_fn

    return retrain_fn


# ---------------------------------------------------------------------------
# Reload — the ORDINARY sbert path. Every stored tuned judge is a complete
# plain checkpoint (full fine-tune, or a LoRA week merged in memory), so it
# reloads with a plain CrossEncoder; there is no adapter to load.
# ---------------------------------------------------------------------------


def load_full_scorer(
    full_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    max_length: int | None = None,
) -> Callable[[tuple[str, str]], float]:
    """Reload a saved tuned judge checkpoint (a full fine-tune or a merged
    LoRA week — both are plain checkpoints since F2c inc7) and return a
    raw-score callable `(query, document) -> float`, via the ORDINARY
    `sentence_transformers.CrossEncoder(full_dir)` path (there is no adapter).

    Returns RAW scores (no sigmoid), matching `RelevanceJudgeProvider.score`'s
    convention in `relevance_judge.py` — a
    load-side caller applies `relevance_judge.label_for_score` on top, exactly
    as the live daily-tick judge does for the base judge. `activation_fn` is
    forced to identity in `predict` so a `num_labels=1` model's default
    Sigmoid is NOT applied here (the same double-sigmoid trap
    `TorchCrossEncoderJudge.score` documents).

    `max_length`: the saved full model already carries its trained
    `max_length` in its own config, so a value is only needed to further
    bound a reload; `None` leaves the saved config's value in force.
    """
    from sentence_transformers import CrossEncoder  # lazy for torch-scoping (I6)

    model = CrossEncoder(
        str(full_dir),
        cache_folder=str(cache_dir) if cache_dir is not None else None,
        **({"max_length": max_length} if max_length is not None else {}),
    )

    def score(item: tuple[str, str]) -> float:
        query, document = item[0], item[1]
        return float(model.predict([(query, document)], activation_fn=lambda x: x)[0])

    return score
