"""LoRA weight-retrain MECHANISM for the judge self-tune's mid-RAM-tier
grade (F2c inc5a, reworked in inc7; spec `f2c-judge-selftune-spec.md` §5
["LoRA (mid)", merge-each-week], §4 [eval + guardrail, via
`judge_eval.run_champion_challenger`'s `retrain_fn` contract], AC6 [champion/
challenger + rollback]), plus the per-persona tuned-judge STORE helpers.

ONE EVOLVING MODEL, MERGED EACH WEEK (F2c inc7, OWNER 2026-09-25 "go with
merging each week"): a LoRA week starts from the persona's CURRENT model (its
own plain checkpoint, or the base judge if it has never been weight-tuned),
wraps it with `peft.get_peft_model`, trains, and merges the adapter back into
the weights IN MEMORY (`merge_and_unload`). Only the merged PLAIN checkpoint
is ever saved; no adapter is kept between weeks, and nothing in this codebase
saves or reloads an adapter. Serving and re-scoring load plain checkpoints
through `judge_full_ft.load_full_scorer`.

TORCH-SCOPED, LAZY (I6): `torch`/`sentence_transformers`/`peft`/`datasets`
are imported LAZILY, function-scoped inside `build_lora_retrain_fn`'s body,
never at this module's top level, mirroring
`relevance_judge.TorchCrossEncoderJudge`'s convention. Importing this module
never pulls torch into `sys.modules`. The store helpers at the bottom are
plain filesystem code and never load a model. `peft` and `datasets` are base
dependencies (Roy 2026-09-24); they are imported lazily only for
torch-scoping, so a missing one is a broken install, surfaced plainly.

HISTORY, sbert bug #3980 (inc5a verdict, superseded in inc7): a LoRA
`CrossEncoder` trained with `modules_to_save` could not be reloaded through
transformers' native `load_adapter` on this stack (sentence-transformers
6.1.0, transformers 5.17.0, peft 0.21.0): the adapter-name prefix of the
trained head is dropped and the head is silently re-initialized, while
`peft.PeftModel.from_pretrained` restored it. inc5-6 therefore reloaded
adapters only through peft and kept a strict-xfail tripwire on the native
path. Since inc7 merges every LoRA week in memory and saves only the merged
plain checkpoint, no adapter is ever saved or reloaded, the reload bug has no
path left to bite, and the adapter reload code and its tripwire were removed
(spec §5 "SUPERSEDED 2026-09-25 by the merge ruling"). The merged head is
the trained head (verified by the inc7 merge round-trip tests).
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from brain import tunables
from brain.memory.judge_eval import RollbackHandle
from brain.memory.relevance_judge import label_for_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables (I3/I7) — shares the `judge_selftune.*` namespace with
# judge_selftune.py/judge_eval.py (one F2c feature, per those modules' own
# convention). PROVISIONAL placeholder defaults (per BUILD instructions):
# the real values are F2c inc5's timed LoRA dry-run's job, not this
# increment's — these exist only so the knobs are live/overridable from
# day one rather than being retrofitted later.
# ---------------------------------------------------------------------------

LORA_RANK_DEFAULT: int = tunables.register("judge_selftune.lora_rank", 8)
LORA_EPOCHS_DEFAULT: int = tunables.register("judge_selftune.lora_epochs", 1)

# Max sequence length for training (`build_lora_retrain_fn`'s `CrossEncoder`)
# — and, because the merged checkpoint is saved through
# `CrossEncoder.save_pretrained`, which persists `max_length` in the saved
# config, also for serving/re-scoring through `judge_full_ft.load_full_scorer`.
# One source for both sides, so the train-time and serve-time tokenizations
# cannot skew (F2c inc5b, task item 5; inc7 moved the serve side from the
# removed adapter scorer to the saved checkpoint's own config). The full-FT
# tier reads the same tunable. PROVISIONAL 512 (the practical bge-reranker
# sequence length); the real value is the deferred timed dry-run's job,
# overridable meanwhile.
LORA_MAX_LENGTH_DEFAULT: int = tunables.register("judge_selftune.lora_max_length", 512)

# `(query, doc, label)` triple — label is "relevant" or "irrelevant", the
# EFFECTIVE (Haiku-over-local) label per judge_eval.py's docstring / spec
# AC4. This module never sees "unknown"/"error" rows — the caller (inc5b's
# data-assembly step) is responsible for filtering those out before
# building `train_items`, mirroring `judge_eval.split_train_test`'s own
# content-agnostic, caller-filters-first posture.
LabeledTriple = tuple[str, str, str]

# Real bge-reranker-v2-m3 LoRA module names (F2c inc5b, task item 6 — the
# recon inc5a's own docstring flagged as owed). CONFIG-ONLY recon of the
# local cached checkpoint (the real model is NEVER loaded — deferred to real
# HW): `BAAI/bge-reranker-v2-m3`'s config.json declares
# `architectures=["XLMRobertaForSequenceClassification"]`,
# `model_type="xlm-roberta"`. For that architecture:
#   - LoRA attention targets = peft 0.21.0's own
#     `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["xlm-roberta"]`
#     == `["query", "value"]` (matched by name suffix, so they resolve
#     under sbert's CrossEncoder wrapper prefix regardless of the wrapper);
#   - the sequence-classification head kept fully trainable
#     (`modules_to_save`) = transformers 5.17.0's
#     `XLMRobertaForSequenceClassification.classifier`
#     (an `XLMRobertaClassificationHead`) == `"classifier"` (NOT `"score"`;
#     the tiny GPT2 test model's `.score` head is a DIFFERENT architecture
#     used only to exercise the mechanism; `get_peft_model` +
#     `merge_and_unload` carry any `modules_to_save` head into the merged
#     weights regardless of its name).
# These are architecture facts tied to `model_tier.MODEL_RELEVANCE_JUDGE`:
# swapping the judge to a NON-xlm-roberta model requires updating them here
# (a deliberate code change, not a silent tunable override that could
# misconfigure the LoRA target modules invisibly). The inc5b tick-lifecycle
# wiring passes these as `build_lora_retrain_fn`'s required
# `target_modules`/`modules_to_save` args.
BGE_RERANKER_LORA_TARGET_MODULES: tuple[str, ...] = ("query", "value")
BGE_RERANKER_LORA_MODULES_TO_SAVE: tuple[str, ...] = ("classifier",)


def _label_to_float(label: str) -> float:
    """"relevant" -> 1.0, "irrelevant" -> 0.0 — `BinaryCrossEntropyLoss`'s
    expected target encoding (matches `relevance_judge.label_for_score`'s
    own two-label vocabulary; any other string is a caller bug, not
    silently coerced)."""
    if label == "relevant":
        return 1.0
    if label == "irrelevant":
        return 0.0
    raise ValueError(f"LabeledTriple label must be 'relevant' or 'irrelevant', got {label!r}")


def _dataset_from_triples(triples: Sequence[LabeledTriple]) -> Any:
    from datasets import Dataset  # base dep (F2c inc5b-2); lazy for torch-scoping only

    return Dataset.from_dict(
        {
            "sentence1": [q for q, _d, _l in triples],
            "sentence2": [d for _q, d, _l in triples],
            "label": [_label_to_float(label) for _q, _d, label in triples],
        }
    )


# ---------------------------------------------------------------------------
# Training entrypoint (spec §5 "LoRA (mid)"): CrossEncoder + peft
# `get_peft_model(LoraConfig)` + BinaryCrossEntropyLoss via the sbert 6.1.0
# CrossEncoderTrainer (the CURRENT training entrypoint for this installed
# version — CrossEncoder's older `.fit()` is a deprecated thin wrapper OVER
# this same trainer), then an in-memory `merge_and_unload` (F2c inc7).
# ---------------------------------------------------------------------------


def build_lora_retrain_fn(
    start_model_path: str | Path,
    *,
    target_modules: Sequence[str],
    modules_to_save: Sequence[str],
    cache_dir: str | Path | None = None,
    lora_rank: int | None = None,
    lora_alpha: int | None = None,
    epochs: int | None = None,
    max_length: int | None = None,
    activation_fn: Callable[[Any], Any] | None = None,
    save_dir: str | Path | None = None,
) -> Callable[[Sequence[LabeledTriple]], Callable[[tuple[str, str]], str]]:
    """Build a `retrain_fn` matching `judge_eval.run_champion_challenger`'s
    `Callable[[Sequence[Any]], Callable[[Any], str]]` contract.

    `start_model_path` (F2c inc7): the model this week's update is applied ON
    TOP OF (spec §1) — the persona's own plain checkpoint dir, or the base
    judge's model id when the persona has never been weight-tuned. Never the
    base by default: the caller (`judge_selftune._training_start`) decides.

    Calling `retrain_fn(train_items)` loads a FRESH `CrossEncoder` from
    `start_model_path`, wraps its underlying transformers model with
    `peft.get_peft_model(LoraConfig(...))` (`lora_rank`/`lora_alpha`/`epochs`
    resolved from tunables when not passed), trains it on `train_items` via
    `BinaryCrossEntropyLoss`, then MERGES the adapter into the weights in
    memory (`merge_and_unload`) — spec §5, OWNER 2026-09-25 "go with merging
    each week". The returned label-fn scores the MERGED model, i.e. exactly
    the model that is saved and would serve.

    `save_dir`: when set, the merged model is saved there as a PLAIN
    checkpoint via `CrossEncoder.save_pretrained` (no adapter files; it
    reloads with a plain `CrossEncoder(save_dir)` / `load_full_scorer`), so the
    tick can, only on an AC6 ACCEPT, swap it into the persona's pointer.
    `None` = no save. No adapter is written anywhere at any point.

    `target_modules`/`modules_to_save`: REQUIRED, no defaults — the LoRA
    target attention modules and the classification head module name(s) kept
    fully trainable, passed straight through to `peft.LoraConfig`.

    Never called with `test_items` (AC5 no-leakage contract, enforced by the
    caller — `judge_eval.run_champion_challenger`).
    """

    def retrain_fn(train_items: Sequence[LabeledTriple]) -> Callable[[tuple[str, str]], str]:
        # Lazy imports (module docstring: never at this module's top level,
        # to keep torch scoped to actually RUNNING training — I6).
        from peft import LoraConfig, get_peft_model
        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
        from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
        from sentence_transformers.cross_encoder.training_args import (
            CrossEncoderTrainingArguments,
        )

        resolved_rank = (
            lora_rank
            if lora_rank is not None
            else tunables.get_tunable("judge_selftune.lora_rank", LORA_RANK_DEFAULT)
        )
        resolved_alpha = lora_alpha if lora_alpha is not None else resolved_rank * 2
        resolved_epochs = (
            epochs
            if epochs is not None
            else tunables.get_tunable("judge_selftune.lora_epochs", LORA_EPOCHS_DEFAULT)
        )
        # Train/serve tokenization parity: the CrossEncoder truncates training
        # inputs at this length, and `save_pretrained` persists it in the
        # merged checkpoint's config, which `load_full_scorer` reloads.
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
        lora_config = LoraConfig(
            r=resolved_rank,
            lora_alpha=resolved_alpha,
            target_modules=list(target_modules),
            modules_to_save=list(modules_to_save),
            task_type="SEQ_CLS",
        )
        # Wrap the CrossEncoder's own loaded transformers model in place
        # (`model[0]` is sbert's Transformer module; `.model` is the HF
        # model it runs) — one load of the start weights, no second copy.
        model[0].model = get_peft_model(model[0].model, lora_config)

        dataset = _dataset_from_triples(train_items)
        loss_fn = BinaryCrossEntropyLoss(model)

        import tempfile

        with tempfile.TemporaryDirectory(prefix="judge_lora_train_") as scratch_dir:
            args = CrossEncoderTrainingArguments(
                output_dir=scratch_dir,
                num_train_epochs=resolved_epochs,
                per_device_train_batch_size=min(8, max(1, len(train_items))),
                report_to=[],
                logging_steps=1_000_000,  # effectively silent; caller doesn't want HF's own logging noise
                save_strategy="no",
                disable_tqdm=True,
            )
            trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=dataset, loss=loss_fn)
            trainer.train()

        # Merge the trained adapter (incl. the `modules_to_save` head) into the
        # weights IN MEMORY; from here on the model is a plain transformers
        # model and nothing adapter-shaped exists (spec §5 merge ruling).
        model[0].model = model[0].model.merge_and_unload()

        if save_dir is not None:
            model.save_pretrained(str(save_dir))

        def label_fn(item: tuple[str, str]) -> str:
            query, document = item[0], item[1]
            raw_score = float(model.predict([(query, document)], activation_fn=lambda x: x)[0])
            # Route through relevance_judge.label_for_score — the single
            # source of truth for turning a raw judge logit into a label
            # (default slope=1.0/intercept=0.0: relevant iff sigmoid(raw)
            # >= 0.5, i.e. raw >= 0.0). `is_ambiguous` is discarded: the
            # champion/challenger eval (AC6) wants a single hard label.
            provisional_label, _is_ambiguous = label_for_score(raw_score)
            return provisional_label

        return label_fn

    return retrain_fn


# ---------------------------------------------------------------------------
# Rollback handle (judge_eval.RollbackHandle contract) — a DURABLE on-disk
# copy, restorable even in a fresh process (BUILD instructions), never an
# in-memory-only snapshot: `judge_eval.run_champion_challenger` calls
# `record()` UNCONDITIONALLY before `retrain_fn` runs and `restore()` only
# on an AC6 revert, so this handle must be able to put the pre-retrain
# adapter directory back regardless of what happened to the in-memory model
# in between.
# ---------------------------------------------------------------------------


class LoraRollbackHandle(RollbackHandle):
    """`judge_eval.RollbackHandle` for the LoRA tier: `record()` copies the
    CURRENT `adapter_dir` (the deployed champion's saved adapter, if any)
    to a fresh, uniquely-named snapshot directory under `snapshot_root`;
    `restore(snapshot)` replaces `adapter_dir`'s contents with that
    snapshot's.

    `adapter_dir` not existing at `record()` time (no prior tuned adapter —
    e.g. the very first weekly tune for this persona) is a valid, ABSENT-
    SAFE state: the snapshot token is `None`, and `restore(None)` means
    "there was no prior adapter" — it removes `adapter_dir` if a
    (rejected) challenger wrote one, rather than restoring nonexistent
    content. This mirrors `relevance_judge`'s absent-is-safe convention for
    a persona that has never had a knob-refit complete.
    """

    def __init__(self, adapter_dir: str | Path, snapshot_root: str | Path | None = None) -> None:
        self._adapter_dir = Path(adapter_dir)
        self._snapshot_root = (
            Path(snapshot_root)
            if snapshot_root is not None
            else self._adapter_dir.parent / f"{self._adapter_dir.name}.snapshots"
        )

    def record(self) -> Path | None:
        if not self._adapter_dir.exists():
            return None
        self._snapshot_root.mkdir(parents=True, exist_ok=True)
        snapshot_dir = self._snapshot_root / f"snapshot-{uuid.uuid4().hex}"
        shutil.copytree(self._adapter_dir, snapshot_dir)
        return snapshot_dir

    def restore(self, snapshot: Path | None) -> None:
        if self._adapter_dir.exists():
            shutil.rmtree(self._adapter_dir)
        if snapshot is not None:
            shutil.copytree(snapshot, self._adapter_dir)


# ---------------------------------------------------------------------------
# Per-persona tuned-judge store (F2c inc5b-2; ONE lineage since inc7, spec
# §5 "One stored tuned model per persona"). TORCH-FREE, filesystem-only —
# these never load a model. The persona's single tuned judge lives under
# `champion_dir(persona_dir)` as a PLAIN merged/fine-tuned checkpoint in a
# versioned subdir (the `adapter-` name prefix is historical: since inc7 no
# adapter is ever stored) plus a `current` POINTER FILE naming the active
# subdir. Persist = write a fresh staged subdir, then atomically repoint
# `current` (write-temp + os.replace on the same filesystem). A mid-tick crash
# resolves `current` to the last-known-good checkpoint; an incomplete staged
# subdir is orphaned scratch, never served, and is reaped by the next tick
# (`reap_unreferenced`). No per-tier stores, no sibling pointers.
# ---------------------------------------------------------------------------

_CHAMPION_POINTER = "current"
_ADAPTER_PREFIX = "adapter-"
_ADAPTER_CONFIG_FILE = "adapter_config.json"


def champion_dir(persona_dir: str | Path) -> Path:
    """The per-persona tuned-judge root: `persona_dir/models/relevance_judge/`
    (spec §5). Does not create it."""
    return Path(persona_dir) / "models" / "relevance_judge"


def resolve_current_checkpoint(persona_dir: str | Path) -> Path | None:
    """The persona's CURRENT tuned judge (F2c inc7): the plain checkpoint dir
    its `current` pointer names, or `None` = the shared base judge (never
    weight-tuned, or the pointer is absent/unreadable/dangling — absent-safe,
    I9). The single source of truth for BOTH the weekly tick (what this week
    trains from and what the champion is) and the daily calibration tick's
    serve path.

    A dir holding `adapter_config.json` (the never-shipped inc5/inc6 LoRA
    adapter layout) is NOT a plain checkpoint and is treated as unresolvable
    (→ base), so it is never handed to the plain-checkpoint loader. Fail-soft:
    never raises."""
    target = resolve_champion_adapter(champion_dir(persona_dir))
    if target is None:
        return None
    try:
        if (target / _ADAPTER_CONFIG_FILE).exists():
            logger.warning(
                "resolve_current_checkpoint: %s holds an adapter (pre-inc7 layout); serving the base judge",
                target,
            )
            return None
    except OSError:
        logger.warning("resolve_current_checkpoint: failed to inspect %s", target, exc_info=True)
        return None
    return target


def read_pointer_name(champion_root: str | Path) -> str | None:
    """The raw name the `current` pointer holds (no existence or form check),
    or `None` if there is no pointer / it is empty / it cannot be read. Used
    to restore the pointer byte-for-byte on a pre-commit rollback and to keep
    the pointer-named dir when discarding a staged one. Never raises."""
    pointer = Path(champion_root) / _CHAMPION_POINTER
    try:
        if not pointer.is_file():
            return None
        name = pointer.read_text(encoding="utf-8").strip()
        return name or None
    except OSError:
        logger.warning("read_pointer_name: failed to read %s", pointer, exc_info=True)
        return None


def discard_stored_dir(path: str | Path) -> None:
    """Best-effort delete of ONE stored dir (a rejected or faulted staged
    checkpoint). Never raises; a refused delete is logged and left for
    `reap_unreferenced` on a later tick."""
    p = Path(path)
    if not p.exists():
        return
    try:
        shutil.rmtree(p)
    except OSError:
        logger.warning("discard_stored_dir: failed to remove %s", p, exc_info=True)


def reap_unreferenced(champion_root: str | Path) -> None:
    """Delete every stored `adapter-*` checkpoint dir except the one the
    `current` pointer names (F2c inc7, ruling Q1 retry path). Never raises.

    Runs at the start of every weekly tick, so a previous checkpoint whose
    post-swap delete the OS refused (e.g. files still open or memory-mapped on
    Windows, I13) is retried on the next tick, and a staged dir orphaned by a
    crash is cleaned up. SAFETY: the pointer file is read raw; if it cannot be
    read (`OSError`) nothing is reaped this tick. The name the pointer holds is
    always kept, even if that dir is missing or holds an adapter; an empty
    pointer also skips reaping. An absent pointer means no tuned model, so
    every `adapter-*` dir is a leftover."""
    root = Path(champion_root)
    pointer = root / _CHAMPION_POINTER
    try:
        if not root.is_dir():
            return
        keep: list[str] = []
        if pointer.exists():
            name = pointer.read_text(encoding="utf-8").strip()
            if not name:
                # An empty pointer names nothing we could safely keep; do not
                # guess, leave everything for a later tick.
                logger.warning("reap_unreferenced: %s is empty; skipping this tick", pointer)
                return
            keep.append(name)
    except OSError:
        logger.warning("reap_unreferenced: cannot read %s; skipping this tick", pointer, exc_info=True)
        return
    cleanup_stale_adapters(root, keep_names=keep)


def staged_adapter_path(champion_root: str | Path) -> Path:
    """A fresh, unique `adapter-<uuid>` subdir under `champion_root` for a
    staged write of a plain checkpoint (NOT yet the champion; the prefix is
    historical). Same filesystem as the `current`
    pointer (both under `champion_root`), so the later `os.replace` swap is
    atomic (C22). Creates `champion_root` (parents) but not the subdir
    itself — `CrossEncoder.save_pretrained` creates it."""
    root = Path(champion_root)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_ADAPTER_PREFIX}{uuid.uuid4().hex}"


def swap_champion_pointer(champion_root: str | Path, staged_subdir: str | Path) -> None:
    """Atomically repoint `current` at `staged_subdir` (its basename). Writes
    a temp pointer file under `champion_root` (same filesystem) and
    `os.replace`s it onto `current` — a single atomic rename, so a concurrent
    reader of `current` sees either the old or the new complete pointer,
    never a torn one (C12)."""
    root = Path(champion_root)
    root.mkdir(parents=True, exist_ok=True)
    name = Path(staged_subdir).name
    tmp = root / f".{_CHAMPION_POINTER}.tmp-{uuid.uuid4().hex}"
    tmp.write_text(name, encoding="utf-8")
    os.replace(tmp, root / _CHAMPION_POINTER)


def clear_champion_pointer(champion_root: str | Path) -> None:
    """Remove the `current` pointer (best-effort, never raises) so the persona
    resolves to the BASE judge again — used by the tick's post-swap-fault
    rollback when there was no prior champion to roll back to (a first-ever
    tune that faulted after its swap)."""
    pointer = Path(champion_root) / _CHAMPION_POINTER
    try:
        pointer.unlink(missing_ok=True)
    except OSError:
        logger.warning("clear_champion_pointer: failed to remove %s", pointer, exc_info=True)


def resolve_champion_adapter(champion_root: str | Path) -> Path | None:
    """Read `current` and return the stored subdir it names, or `None`
    (absent pointer, dangling target, or any read error). Fail-soft — a
    resolve failure must degrade to the base judge (I9), never crash the
    serve/tick path."""
    root = Path(champion_root)
    pointer = root / _CHAMPION_POINTER
    try:
        if not pointer.is_file():
            return None
        name = pointer.read_text(encoding="utf-8").strip()
        if not name:
            return None
        target = root / name
        return target if target.is_dir() else None
    except OSError:
        logger.warning("resolve_champion_adapter: failed to read %s", pointer, exc_info=True)
        return None


def cleanup_stale_adapters(champion_root: str | Path, keep_names: Sequence[str]) -> None:
    """Best-effort removal of `adapter-*` subdirs whose basename is NOT in
    `keep_names` (never raises; a refused delete, e.g. an open file on
    Windows, is logged and left for `reap_unreferenced` on a later tick). The
    tick passes `judge_selftune._keep_after_accept(...)` after an accepted
    swap (F2c inc7 ruling Q1: only the new checkpoint; the previous one is
    deleted in the same tick), or {current} on a revert/fault to reap a
    discarded staged subdir (C21). A `None`/empty name in `keep_names` is
    ignored (first-ever tune has no prior)."""
    root = Path(champion_root)
    keep = {n for n in keep_names if n}
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.name.startswith(_ADAPTER_PREFIX):
            continue
        if entry.name in keep:
            continue
        try:
            shutil.rmtree(entry)
        except OSError:
            logger.warning("cleanup_stale_adapters: failed to remove %s", entry, exc_info=True)
