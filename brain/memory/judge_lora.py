"""LoRA weight-retrain MECHANISM for the judge self-tune's mid-RAM-tier
grade (F2c inc5a; spec `f2c-judge-selftune-spec.md` §5 [tiered
weight-retrain build detail, "LoRA (mid)"], §4 [eval + guardrail, via
`judge_eval.run_champion_challenger`'s `retrain_fn`/`RollbackHandle`
contract], AC6 [champion/challenger + rollback], AC7 [LoRA reload / bug
#3980 guard]).

TORCH-SCOPED, LAZY (I6): `torch`/`sentence_transformers`/`peft`/`datasets`
are imported LAZILY, function-scoped inside `build_lora_retrain_fn`'s and
`load_lora_scorer`'s bodies — never at this module's top level — mirroring
`relevance_judge.TorchCrossEncoderJudge`'s exact convention (see that
module's docstring for the full rationale). Importing this module never
pulls torch into `sys.modules`; only actually CALLING one of those two
functions does. `LoraRollbackHandle.record`/`restore` are plain filesystem
copies (`shutil.copytree`/`rmtree`) — genuinely torch-free, not merely
scoped: they never load a model at all, by construction, which is also why
a rollback is cheap enough to call on every champion/challenger cycle.

OPTIONAL DEPENDENCY (Opus cold-review round 2, Planning ruling): `peft` and
`datasets` are NOT part of this project's always-installed baseline — they
live behind the optional `f2c-training` extra in `pyproject.toml`
(`uv sync --extra f2c-training`), since only the LoRA/full-FT weight-retrain
tiers need them (a weak-tier box never does). `lora_available()` below is a
cheap try-import check (no heavy/model load) for callers — in particular
`judge_selftune.py`'s tier-detect, which downgrades to the always-safe
knob-refit floor when this extra isn't installed, mirroring its existing
cgroup-aware OOM-safety downgrade. Every lazy-import site in this module
that needs `peft`/`datasets`/`sentence_transformers` catches `ImportError`
and re-raises with a clear pointer at the extra, rather than surfacing a
bare `ModuleNotFoundError` from deep inside a training/reload call.

Scope (inc5a only — mirrors `judge_eval.py`'s own scoping note): this
module builds the LoRA train/save/reload MECHANISM and proves it against a
FROM-SCRATCH TINY model in its tests, never the real bge-reranker-v2-m3.
It is NOT wired into `judge_selftune._run_judge_selftune_tick`'s tier
dispatch (that wiring is inc5b, alongside the Haiku-only extraction
`judge_eval.py`'s docstring already flags as owed). `target_modules` and
`modules_to_save` are REQUIRED, caller-supplied arguments here, not
hardcoded defaults: bge-reranker-v2-m3's actual attention/head module
names are inc5b's job to confirm against the real checkpoint, and guessing
at them here would be exactly the kind of unverified magic-number/model-
specific assumption this codebase avoids (`roy-dislikes-magic-numbers`).

=== AC7 / bug #3980 — VERDICT (inc5a's primary deliverable) ===

sbert #3980 (a LoRA `CrossEncoder` trained with `modules_to_save=["score"]`
can fail to reload the trained classification head) REPRODUCES cleanly in
this stack (sentence-transformers==6.1.0, transformers==5.17.0,
peft==0.21.0), confirmed against a from-scratch tiny GPT2-for-sequence-
classification model (GPT2 was chosen because its `ForSequenceClassification`
head is a `nn.Linear` literally named `.score`, matching the bug report's
own `modules_to_save=["score"]` naming exactly):

  - ROOT CAUSE (traced, not guessed): `peft.utils.save_and_load.
    get_peft_model_state_dict` (what `CrossEncoder.save_pretrained` /
    `PreTrainedModel.save_pretrained` call under the hood) saves a
    `modules_to_save` head's weights under the adapter-name-STRIPPED key
    (`base_model.model.score.weight` on disk) — this is documented,
    intentional peft behavior (the adapter name is meant to be
    re-insertable under a DIFFERENT name at load time). Reloading via
    `sentence_transformers.CrossEncoder(saved_dir)` (which auto-detects
    `adapter_config.json` and defers to `transformers.integrations.peft.
    PreTrainedModel.load_adapter`, transformers' OWN native adapter-load
    path — NOT peft's) does NOT restore the `modules_to_save.default.`
    prefix before matching checkpoint keys against the freshly-injected
    model structure: it expects `score.modules_to_save.default.weight` but
    finds only `score.weight` in the checkpoint, so the trained head is
    silently newly-initialized (RANDOM) instead of loaded — reproduced
    scores differ from trained scores. Calling `model.load_adapter(...)`
    directly (bypassing `CrossEncoder`'s own wrapper) reproduces the exact
    same failure — the gap is in transformers' native PEFT integration
    path, not sbert's plumbing around it.
  - MITIGATION CONFIRMED (inc5a, empirically verified — see
    `tests/unit/brain/memory/test_judge_lora.py::
    test_bug_3980_reload_via_peft_native_reproduces_exact_scores`):
    reloading via peft's OWN native path —
    `peft.PeftModel.from_pretrained(base_model, saved_dir)` — instead of
    transformers' built-in adapter auto-load, DOES correctly restore the
    trained head: `peft.utils.save_and_load.set_peft_model_state_dict`
    (which `PeftModel.from_pretrained`/`PeftModel.load_adapter` call, and
    which `transformers.integrations.peft.PreTrainedModel.load_adapter`
    does NOT) explicitly re-inserts the `modules_to_save.{adapter_name}.`
    prefix via each `ModulesToSaveWrapper`'s own `adapter_state_dict_load_
    map`. Reloaded scores then match trained scores bit-for-bit.
  - CONSEQUENCE FOR THIS MODULE: every reload path in this module
    (`load_lora_scorer` below, and `LoraRollbackHandle.restore`'s
    verification) goes through `peft.PeftModel.from_pretrained`, never
    through `sentence_transformers.CrossEncoder(adapter_dir, ...)`'s own
    auto-detection or a bare `model.load_adapter(...)` call — so the LoRA
    path IS shippable, but only via this specific reload mechanism. A
    regression test
    (`test_bug_3980_reload_via_crossencoder_autoload_reproduces_the_bug`)
    is kept in place, XFAIL(strict) with this reasoning, as a tripwire: if
    a future sbert/transformers/peft upgrade fixes the native path, that
    test starts unexpectedly passing (XPASS) and the strict xfail fails
    the suite, forcing a conscious decision about whether to simplify this
    module's reload path at that point.
"""

from __future__ import annotations

import logging
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
# Optional-dependency guard (Opus cold-review round 2, Planning ruling):
# `peft`/`datasets` live behind pyproject.toml's `f2c-training` extra, not
# this project's always-installed baseline (see module docstring). Every
# lazy-import site below that needs them funnels through
# `_require_training_deps`'s `ImportError` wrapping so a missing extra
# fails with ONE clear, actionable message instead of a bare
# `ModuleNotFoundError: No module named 'peft'` surfacing from deep inside
# a training/reload call.
# ---------------------------------------------------------------------------

_F2C_TRAINING_EXTRA_HINT = (
    "requires the optional 'f2c-training' extra (peft + datasets), which "
    "is not installed -- install it with `uv sync --extra f2c-training` "
    "(or `uv pip install '.[f2c-training]'` outside a uv-managed venv)."
)


def lora_available() -> bool:
    """Cheap availability check: True iff `peft`, `datasets`, and
    `sentence_transformers` are all INSTALLABLE-AND-FINDABLE. NO heavy/
    model load, and — deliberately — no actual import either:
    `importlib.util.find_spec` locates each package on the import path
    without executing its `__init__.py`, so this check never triggers
    `sentence_transformers`' own (transitive) `import torch` as a side
    effect. A plain `try: import sentence_transformers` here would DO that
    unconditionally, on every box whose RAM tier resolves to LoRA/full-FT —
    defeating the very invariant this module's docstring and
    `judge_selftune.py`'s own torch-free tick test guard for (torch stays
    scoped to actually RUNNING the training/reload mechanism, never to
    merely checking whether it's available). Safe to call on every weekly
    tick regardless of which RAM tier is in play.

    Used by `judge_selftune.py`'s tier-detect to decide whether a RAM tier
    that WOULD select LoRA/full-FT is actually usable on this install —
    downgrading to the always-safe knob-refit floor when the optional
    `f2c-training` extra was never installed, the same posture as the
    existing cgroup-aware OOM-safety downgrade (a tier this process cannot
    actually execute is not a tier to select, RAM or extras).
    """
    import importlib.util

    try:
        return all(
            importlib.util.find_spec(name) is not None
            for name in ("peft", "datasets", "sentence_transformers")
        )
    except (ImportError, ValueError):
        # find_spec can raise (rather than return None) for a malformed or
        # partially-broken install; fail toward "not available" -- the
        # same safe-floor posture as everywhere else in this guard.
        return False


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

# Max sequence length for BOTH training (`build_lora_retrain_fn`'s
# `CrossEncoder`) and serving (`load_lora_scorer`'s tokenizer) — the SINGLE
# source both sides read, so the train-time and serve-time tokenizations
# cannot skew (F2c inc5b, task item 5). A LoRA adapter trained on inputs
# truncated at length L must be SERVED on inputs truncated the same way, or
# the trained head sees a differently-tokenized input than it learned on —
# a silent scores-differ failure, the same CLASS the #3980 guard prevents on
# a different axis. Before this, `load_lora_scorer` applied no truncation at
# all while training used the CrossEncoder's own (unset -> tokenizer
# `model_max_length`, ~8194 for bge) default: a real divergence on any doc
# long enough to truncate at one and not the other. PROVISIONAL 512 (the
# practical bge-reranker sequence length); the real value is the deferred
# inc5 timed dry-run's job, overridable meanwhile.
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
#     the tiny GPT2 test model's `.score` head, which reproduces bug #3980,
#     is a DIFFERENT architecture used only to exercise the mechanism — the
#     #3980 peft-reload mitigation is head-name-agnostic, so it holds for
#     `classifier` too).
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
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(f"judge_lora training {_F2C_TRAINING_EXTRA_HINT}") from exc

    return Dataset.from_dict(
        {
            "sentence1": [q for q, _d, _l in triples],
            "sentence2": [d for _q, d, _l in triples],
            "label": [_label_to_float(label) for _q, _d, label in triples],
        }
    )


# ---------------------------------------------------------------------------
# Training entrypoint (spec §5 "LoRA (mid)"): CrossEncoder + LoraConfig +
# BinaryCrossEntropyLoss via the sbert 6.1.0 CrossEncoderTrainer (the
# CURRENT training entrypoint for this installed version — CrossEncoder's
# older `.fit()` is a deprecated thin wrapper OVER this same trainer, per
# sbert 6.1.0's own `fit_mixin.py` docstring, so there is no lighter
# training API to use instead).
# ---------------------------------------------------------------------------


def build_lora_retrain_fn(
    base_model_path: str | Path,
    *,
    target_modules: Sequence[str],
    modules_to_save: Sequence[str],
    cache_dir: str | Path | None = None,
    lora_rank: int | None = None,
    lora_alpha: int | None = None,
    epochs: int | None = None,
    max_length: int | None = None,
    activation_fn: Callable[[Any], Any] | None = None,
) -> Callable[[Sequence[LabeledTriple]], Callable[[tuple[str, str]], str]]:
    """Build a `retrain_fn` matching `judge_eval.run_champion_challenger`'s
    `Callable[[Sequence[Any]], Callable[[Any], str]]` contract.

    `base_model_path`: a local path or hub id `sentence_transformers.
    CrossEncoder` can load (inc5a's TESTS always pass a from-scratch tiny
    local model dir — see this module's docstring; inc5b wires the real
    bge-reranker-v2-m3 model_tier id here).

    `target_modules`/`modules_to_save`: REQUIRED, no defaults (see module
    docstring) — the LoRA adapter's target attention modules and the
    classification head module name(s) to keep fully trainable
    (`modules_to_save`), passed straight through to `peft.LoraConfig`.

    Returns `retrain_fn(train_items) -> label_fn`: calling `retrain_fn`
    loads a FRESH `CrossEncoder` from `base_model_path`, adds a LoRA
    adapter (`lora_rank`/`lora_alpha`/`epochs` resolved from tunables when
    not passed explicitly), trains it on `train_items` via
    `BinaryCrossEntropyLoss`, and returns a plain callable
    `(query, document) -> "relevant" | "irrelevant"` bound to the trained
    IN-MEMORY model — no save/reload round-trip in this path (AC6's
    champion/challenger eval is forward-only against the just-trained
    model; save/reload is `LoraRollbackHandle`'s concern, for
    cross-process persistence, not this function's).

    Never called with `test_items` (AC5 no-leakage contract, enforced by
    the caller — `judge_eval.run_champion_challenger` — not by this
    function, which has no way to distinguish train from test items on its
    own).
    """

    def retrain_fn(train_items: Sequence[LabeledTriple]) -> Callable[[tuple[str, str]], str]:
        # Lazy imports (module docstring: never at this module's top
        # level), import-guarded (Opus cold-review round 2): peft/
        # sentence_transformers live behind the optional `f2c-training`
        # extra (peft does; sentence-transformers is always installed, but
        # is guarded alongside it here for ONE unified, clear error path
        # regardless of which piece is actually missing).
        try:
            from peft import LoraConfig
            from sentence_transformers import CrossEncoder
            from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
            from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
            from sentence_transformers.cross_encoder.training_args import (
                CrossEncoderTrainingArguments,
            )
        except ImportError as exc:
            raise ImportError(f"build_lora_retrain_fn's retrain_fn {_F2C_TRAINING_EXTRA_HINT}") from exc

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
        # Train/serve tokenization parity (task item 5): the CrossEncoder
        # truncates training inputs at this length; `load_lora_scorer`
        # reads the SAME tunable to truncate serve inputs identically.
        resolved_max_length = (
            max_length
            if max_length is not None
            else tunables.get_tunable("judge_selftune.lora_max_length", LORA_MAX_LENGTH_DEFAULT)
        )

        model = CrossEncoder(
            str(base_model_path),
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
        model.add_adapter(lora_config)

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

        def label_fn(item: tuple[str, str]) -> str:
            query, document = item[0], item[1]
            raw_score = float(model.predict([(query, document)], activation_fn=lambda x: x)[0])
            # Route through relevance_judge.label_for_score — the single
            # source of truth for turning a raw judge logit into a label
            # (default slope=1.0/intercept=0.0: relevant iff sigmoid(raw)
            # >= 0.5, i.e. raw >= 0.0 — NOT raw >= 0.5). A local re-
            # implementation here previously thresholded the RAW logit at
            # 0.5 directly, which is wrong (biases against "relevant" for
            # raw in [0.0, 0.5)) and Platt-inconsistent with inc3's fitted
            # knob. `is_ambiguous` is discarded: the champion/challenger
            # eval (AC6) wants a single hard label per item, not an
            # ambiguous-band routing decision (that's relevance_judge's
            # own live-tick concern, not training/eval here).
            provisional_label, _is_ambiguous = label_for_score(raw_score)
            return provisional_label

        return label_fn

    return retrain_fn


# ---------------------------------------------------------------------------
# AC7-SAFE reload — the fix this module's docstring verdict identifies:
# peft's OWN native reload path, never sbert's/transformers' auto-detect.
# ---------------------------------------------------------------------------


def load_lora_scorer(
    base_model_path: str | Path,
    adapter_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    max_length: int | None = None,
) -> Callable[[tuple[str, str]], float]:
    """Reload a saved LoRA adapter and return a raw-score callable
    `(query, document) -> float`, via the bug-#3980-SAFE path (this
    module's docstring verdict): `peft.PeftModel.from_pretrained` against a
    freshly-loaded plain base model, NEVER `sentence_transformers.
    CrossEncoder(adapter_dir)`'s own auto-detection (which reproduces
    #3980 — see `test_bug_3980_reload_via_crossencoder_autoload_
    reproduces_the_bug`).

    Returns RAW scores (no sigmoid), matching `RelevanceJudgeProvider.
    score`'s convention in `relevance_judge.py` — a future load-side caller
    applies `relevance_judge.label_for_score` on top, exactly as the live
    daily-tick judge already does for the base (non-tuned) judge.

    TRAIN/SERVE TOKENIZATION PARITY (task item 5): `max_length` (None ->
    the `judge_selftune.lora_max_length` tunable) truncates serve inputs at
    the SAME length `build_lora_retrain_fn` truncated training inputs at, so
    the adapter is scored on inputs tokenized the way it was trained. Before
    this, the serve tokenizer applied no truncation while training did (via
    the CrossEncoder's own max_length) — a silent skew on long inputs.
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(f"load_lora_scorer {_F2C_TRAINING_EXTRA_HINT}") from exc

    resolved_max_length = (
        max_length
        if max_length is not None
        else tunables.get_tunable("judge_selftune.lora_max_length", LORA_MAX_LENGTH_DEFAULT)
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        str(base_model_path),
        local_files_only=True,
        num_labels=1,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    peft_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), local_files_only=True)

    def score(item: tuple[str, str]) -> float:
        import torch

        query, document = item[0], item[1]
        encoded = tokenizer(
            [query],
            [document],
            padding=True,
            truncation=True,
            max_length=resolved_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = peft_model(**encoded)
        return float(output.logits.squeeze(-1)[0])

    return score


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
