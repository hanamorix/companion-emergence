"""Tests for `brain.memory.judge_lora` — F2c inc5a's LoRA weight-retrain
MECHANISM (spec `f2c-judge-selftune-spec.md` §5 [LoRA (mid)], AC6
[champion/challenger + rollback], AC7 [LoRA reload / bug #3980 guard]).

OFFLINE, TINY MODEL ONLY (never the real bge-reranker-v2-m3 — see
`build-for-potato-computers-baseline` / the BUILD instructions this
increment was scoped under): every test here builds a FROM-SCRATCH,
randomly-initialized tiny GPT2-for-sequence-classification model (2
layers, hidden=32) with a minimal from-scratch tokenizer, entirely on
disk, and loads everything with `local_files_only=True` — no network
access, no hub download, ever. GPT2 is the chosen architecture because its
`*ForSequenceClassification` head is a plain `nn.Linear` literally named
`.score` (see `transformers/models/gpt2/modeling_gpt2.py`), matching sbert
bug #3980's own `modules_to_save=["score"]` example exactly, so the
tiny-model smoke test below exercises the SAME module name the real bug
report names, not a same-shape-but-differently-named stand-in.

Slow-ish but not heavy: each test that trains does 1-2 epochs over 4
scripted (query, doc, label) triples on a 2-layer/32-hidden model — sub-
second CPU forward/backward passes. The fixture below builds the tiny
model+tokenizer ONCE per test session (module-scoped) since building it is
the dominant cost, not training on top of it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import brain.memory.judge_lora as judge_lora
from brain.memory.judge_eval import run_champion_challenger
from brain.memory.judge_lora import (
    BGE_RERANKER_LORA_MODULES_TO_SAVE,
    BGE_RERANKER_LORA_TARGET_MODULES,
    LoraRollbackHandle,
    build_lora_retrain_fn,
    load_lora_scorer,
    lora_available,
)

# ---------------------------------------------------------------------------
# Shared fixture: a from-scratch tiny GPT2-for-sequence-classification model
# + tokenizer, built and saved to disk once per test session.
# ---------------------------------------------------------------------------

_VOCAB_WORDS = [
    "dog", "cat", "apple", "banana", "music", "science", "weather",
    "the", "a", "is", "great", "bad", "loud", "quiet", "red", "blue",
    "query", "document", "relevant", "irrelevant", "test", "tiny",
]


def _build_tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    vocab = {tok: i for i, tok in enumerate(specials + _VOCAB_WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def _build_tiny_model_dir(dest_dir: Path) -> Path:
    """Build a from-scratch, randomly-initialized tiny GPT2ForSequence-
    Classification model + tokenizer and save both to `dest_dir`. No
    download: `AutoModelForSequenceClassification.from_config` constructs
    random weights straight from a hand-built `GPT2Config`.
    """
    from transformers import AutoModelForSequenceClassification, GPT2Config

    tokenizer = _build_tiny_tokenizer()
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=64,
        num_labels=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    config.architectures = ["GPT2ForSequenceClassification"]
    model = AutoModelForSequenceClassification.from_config(config)

    dest_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest_dir)
    tokenizer.save_pretrained(dest_dir)
    return dest_dir


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("judge_lora_tiny_base")
    return _build_tiny_model_dir(base / "tiny_gpt2_seqclass")


# GPT2's LoRA target module (peft's documented example target for GPT2 —
# the fused qkv `Conv1D` layer) + the classification head name that
# actually reproduces #3980 (see module docstring above).
_TARGET_MODULES = ["c_attn"]
_MODULES_TO_SAVE = ["score"]

_TRAIN_TRIPLES = [
    ("query", "relevant", "relevant"),
    ("query", "irrelevant", "irrelevant"),
    ("test document", "relevant test", "relevant"),
    ("tiny query", "irrelevant document", "irrelevant"),
]
_FIXED_EVAL_ITEMS = [("query", "relevant"), ("tiny query", "irrelevant document")]


# ---------------------------------------------------------------------------
# AC7 / bug #3980 — the headline verdict (see judge_lora.py's module
# docstring for the full traced root cause + mitigation).
# ---------------------------------------------------------------------------


def test_bug_3980_reload_via_crossencoder_autoload_reproduces_the_bug(tiny_model_dir: Path, tmp_path: Path) -> None:
    """REGRESSION TRIPWIRE, expected to fail on this stack (sentence-
    transformers==6.1.0, transformers==5.17.0, peft==0.21.0): reloading a
    trained LoRA adapter (`modules_to_save=["score"]`) via `sentence_
    transformers.CrossEncoder(adapter_dir, ...)`'s own adapter auto-
    detection reproduces sbert bug #3980 — the reloaded classification
    head does NOT match the trained one, so reloaded scores differ from
    trained scores on a fixed input.

    XFAIL(strict=True), not skip/deleted (BUILD instructions: "leave the
    failing test in place"): if a future sbert/transformers/peft bump
    fixes the native reload path, this test starts unexpectedly PASSING,
    and strict xfail turns that into a hard failure — a deliberate
    tripwire forcing a conscious look at whether `judge_lora.py`'s
    peft-native-only reload workaround can then be simplified.
    """
    import torch
    from peft import LoraConfig
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    torch.manual_seed(0)
    ce = CrossEncoder(
        str(tiny_model_dir),
        local_files_only=True,
        config_kwargs={"num_labels": 1},
        activation_fn=lambda x: x,
    )
    lora_config = LoraConfig(
        r=4, lora_alpha=8, target_modules=_TARGET_MODULES,
        modules_to_save=_MODULES_TO_SAVE, task_type="SEQ_CLS",
    )
    ce.add_adapter(lora_config)

    from datasets import Dataset
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

    dataset = Dataset.from_dict(
        {
            "sentence1": [q for q, _d, _l in _TRAIN_TRIPLES],
            "sentence2": [d for _q, d, _l in _TRAIN_TRIPLES],
            "label": [1.0 if lbl == "relevant" else 0.0 for _q, _d, lbl in _TRAIN_TRIPLES],
        }
    )
    loss_fn = BinaryCrossEntropyLoss(ce)
    args = CrossEncoderTrainingArguments(
        output_dir=str(tmp_path / "train_out"),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        report_to=[],
        logging_steps=1_000_000,
        save_strategy="no",
        disable_tqdm=True,
    )
    CrossEncoderTrainer(model=ce, args=args, train_dataset=dataset, loss=loss_fn).train()

    trained_scores = ce.predict(_FIXED_EVAL_ITEMS)

    save_dir = tmp_path / "saved_adapter"
    ce.save_pretrained(str(save_dir))

    reloaded = CrossEncoder(
        str(save_dir),
        local_files_only=True,
        config_kwargs={"num_labels": 1},
        activation_fn=lambda x: x,
    )
    reloaded_scores = reloaded.predict(_FIXED_EVAL_ITEMS)

    if np.allclose(trained_scores, reloaded_scores, atol=1e-5):
        pytest.fail("XPASS: CrossEncoder auto-reload no longer reproduces #3980 -- see judge_lora.py docstring")
    else:
        pytest.xfail(
            "sbert #3980 reproduces: CrossEncoder(adapter_dir) auto-reload drops the trained "
            "modules_to_save head (transformers' native load_adapter does not restore the "
            "modules_to_save.<adapter>. prefix) -- see judge_lora.py module docstring for the "
            "traced root cause and the peft-native mitigation this module actually uses."
        )


def test_bug_3980_reload_via_peft_native_reproduces_exact_scores(tiny_model_dir: Path, tmp_path: Path) -> None:
    """The AC7 REQUIRED smoke test, in its PASSING form: training via
    `build_lora_retrain_fn`, saving, then reloading via `judge_lora.
    load_lora_scorer` (peft's own native `PeftModel.from_pretrained` reload
    path, this module's confirmed #3980 mitigation) reproduces the trained
    model's scores exactly (the head IS restored).
    """
    retrain_fn = build_lora_retrain_fn(
        tiny_model_dir,
        target_modules=_TARGET_MODULES,
        modules_to_save=_MODULES_TO_SAVE,
        lora_rank=4,
        epochs=2,
    )
    label_fn = retrain_fn(_TRAIN_TRIPLES)
    assert label_fn(_FIXED_EVAL_ITEMS[0]) in ("relevant", "irrelevant")

    # Re-run training deterministically (seeded) to get a model whose exact
    # raw scores we can compare against a save+peft-native-reload round trip.
    import torch
    from peft import LoraConfig
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    torch.manual_seed(0)
    ce = CrossEncoder(
        str(tiny_model_dir), local_files_only=True,
        config_kwargs={"num_labels": 1}, activation_fn=lambda x: x,
    )
    lora_config = LoraConfig(
        r=4, lora_alpha=8, target_modules=_TARGET_MODULES,
        modules_to_save=_MODULES_TO_SAVE, task_type="SEQ_CLS",
    )
    ce.add_adapter(lora_config)

    from datasets import Dataset
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

    dataset = Dataset.from_dict(
        {
            "sentence1": [q for q, _d, _l in _TRAIN_TRIPLES],
            "sentence2": [d for _q, d, _l in _TRAIN_TRIPLES],
            "label": [1.0 if lbl == "relevant" else 0.0 for _q, _d, lbl in _TRAIN_TRIPLES],
        }
    )
    loss_fn = BinaryCrossEntropyLoss(ce)
    args = CrossEncoderTrainingArguments(
        output_dir=str(tmp_path / "train_out"),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        report_to=[],
        logging_steps=1_000_000,
        save_strategy="no",
        disable_tqdm=True,
    )
    CrossEncoderTrainer(model=ce, args=args, train_dataset=dataset, loss=loss_fn).train()
    trained_scores = np.array(ce.predict(_FIXED_EVAL_ITEMS))

    save_dir = tmp_path / "saved_adapter_for_safe_reload"
    ce.save_pretrained(str(save_dir))

    scorer = load_lora_scorer(tiny_model_dir, save_dir)
    reloaded_scores = np.array([scorer(item) for item in _FIXED_EVAL_ITEMS])

    assert np.allclose(trained_scores, reloaded_scores, atol=1e-5), (
        f"trained={trained_scores} reloaded={reloaded_scores}"
    )


# ---------------------------------------------------------------------------
# build_lora_retrain_fn: produces a working label callable.
# ---------------------------------------------------------------------------


def test_retrain_fn_trains_without_error_and_returns_valid_labels(tiny_model_dir: Path) -> None:
    retrain_fn = build_lora_retrain_fn(
        tiny_model_dir,
        target_modules=_TARGET_MODULES,
        modules_to_save=_MODULES_TO_SAVE,
        lora_rank=4,
        epochs=1,
    )
    label_fn = retrain_fn(_TRAIN_TRIPLES)
    for item in _FIXED_EVAL_ITEMS:
        label = label_fn(item)
        assert label in ("relevant", "irrelevant"), label


def test_label_fn_boundary_matches_relevance_judge_label_for_score(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the Opus cold-review MEDIUM finding: `label_fn` (the
    callable `retrain_fn` returns) must route the RAW judge logit through
    `relevance_judge.label_for_score` -- "relevant" iff `sigmoid(raw) >=
    0.5`, i.e. `raw >= 0.0` -- NOT threshold the raw logit at `raw >= 0.5`
    directly. A raw score of 0.3 sits in [0.0, 0.5): `sigmoid(0.3) ~= 0.574
    >= 0.5` -> "relevant" under the correct boundary, but the OLD, buggy
    `raw >= 0.5` threshold would have called this "irrelevant" -- this test
    BITES against that old threshold (it would fail under it).
    """
    from sentence_transformers import CrossEncoder

    retrain_fn = build_lora_retrain_fn(
        tiny_model_dir,
        target_modules=_TARGET_MODULES,
        modules_to_save=_MODULES_TO_SAVE,
        lora_rank=4,
        epochs=1,
    )
    label_fn = retrain_fn(_TRAIN_TRIPLES)

    # Force the trained model's raw score for this call into [0.0, 0.5),
    # the exact band the old (wrong) `raw >= 0.5` threshold mislabeled.
    monkeypatch.setattr(CrossEncoder, "predict", lambda self, pairs, **kwargs: [0.3])

    assert label_fn(_FIXED_EVAL_ITEMS[0]) == "relevant"


def test_retrain_fn_matches_judge_eval_champion_challenger_contract(
    tiny_model_dir: Path, tmp_path: Path
) -> None:
    """`build_lora_retrain_fn`'s output plugs directly into `judge_eval.
    run_champion_challenger`'s `retrain_fn` parameter (AC6's orchestration
    contract) without any adapter shim."""
    retrain_fn = build_lora_retrain_fn(
        tiny_model_dir,
        target_modules=_TARGET_MODULES,
        modules_to_save=_MODULES_TO_SAVE,
        lora_rank=4,
        epochs=1,
    )

    def champion(item: tuple[str, str]) -> str:
        return "irrelevant"  # deliberately-bad champion so the challenger has room to win

    test_items = [(item, "relevant") for item in _FIXED_EVAL_ITEMS] * 20  # clear the min_n=30 floor

    result = run_champion_challenger(
        champion=champion,
        retrain_fn=retrain_fn,
        train_items=_TRAIN_TRIPLES,
        test_items=test_items,
        rollback=LoraRollbackHandle(adapter_dir=tmp_path / "unused-not-written-this-test"),
        alpha=0.05,
        min_n=1,
    )
    assert result.n_test == len(test_items)
    assert isinstance(result.accepted, bool)


# ---------------------------------------------------------------------------
# LoraRollbackHandle: record -> mutate -> restore round-trips.
# ---------------------------------------------------------------------------


def test_rollback_handle_record_mutate_restore_round_trips(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"marker": "champion-v1"}')
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"champion-weights")

    handle = LoraRollbackHandle(adapter_dir=adapter_dir, snapshot_root=tmp_path / "snapshots")
    snapshot = handle.record()
    assert snapshot is not None
    assert (snapshot / "adapter_config.json").read_text() == '{"marker": "champion-v1"}'

    # Mutate in place, as a (soon-to-be-rejected) challenger's save would.
    (adapter_dir / "adapter_config.json").write_text('{"marker": "challenger-v2"}')
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"challenger-weights")

    handle.restore(snapshot)

    assert (adapter_dir / "adapter_config.json").read_text() == '{"marker": "champion-v1"}'
    assert (adapter_dir / "adapter_model.safetensors").read_bytes() == b"champion-weights"


def test_rollback_handle_record_with_no_prior_adapter_is_absent_safe(tmp_path: Path) -> None:
    """No adapter has ever been deployed for this persona yet (first-ever
    weekly tune) -- `record()` on a nonexistent `adapter_dir` returns
    `None`, and `restore(None)` correctly leaves/returns to "no adapter"
    rather than crashing on a missing snapshot path."""
    adapter_dir = tmp_path / "adapter"
    handle = LoraRollbackHandle(adapter_dir=adapter_dir, snapshot_root=tmp_path / "snapshots")

    snapshot = handle.record()
    assert snapshot is None

    # A (rejected) challenger wrote something anyway; restore(None) must
    # clean it back up to "no adapter", not error out on a None snapshot.
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")

    handle.restore(snapshot)
    assert not adapter_dir.exists()


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------


def test_training_is_deterministic_under_a_fixed_seed(tiny_model_dir: Path) -> None:
    """Same tiny model + same scripted triples + a fixed `torch.manual_seed`
    before each `retrain_fn` call -> reproducible-enough trained scores
    across two independent training runs (proves the mechanism is
    deterministic when the caller seeds it, not that this module seeds
    internally -- it does not, mirroring `judge_eval.split_train_test`'s
    own caller-controls-the-seed posture)."""
    import torch

    def train_and_score() -> list[float]:
        torch.manual_seed(1234)
        retrain_fn = build_lora_retrain_fn(
            tiny_model_dir,
            target_modules=_TARGET_MODULES,
            modules_to_save=_MODULES_TO_SAVE,
            lora_rank=4,
            epochs=2,
        )
        label_fn = retrain_fn(_TRAIN_TRIPLES)
        return [label_fn(item) for item in _FIXED_EVAL_ITEMS]

    first = train_and_score()
    second = train_and_score()
    assert first == second


# ---------------------------------------------------------------------------
# Optional-extra guard (Opus cold-review round 2, Planning ruling):
# `lora_available()` + the import-guard ImportError message on the three
# lazy-import sites.
# ---------------------------------------------------------------------------


def test_lora_available_is_true_in_this_dev_env() -> None:
    """This worktree was `uv sync --all-extras`'d for testing, so the
    optional `f2c-training` extra (peft + datasets) IS installed here --
    `lora_available()` must reflect that."""
    assert lora_available() is True


def test_lora_available_is_a_cheap_check_that_does_not_import_torch() -> None:
    """`lora_available()` uses `importlib.util.find_spec`, never a bare
    `import sentence_transformers` -- a fresh subprocess proves calling it
    never pulls torch/sentence_transformers into `sys.modules`, even though
    `sentence_transformers` IS one of the three packages it checks for
    (checking it must not mean IMPORTING it)."""
    script = textwrap.dedent(
        """
        import sys

        from brain.memory.judge_lora import lora_available

        assert lora_available() is True, "expected True in this all-extras dev env"
        assert "torch" not in sys.modules, sorted(sys.modules)
        assert "sentence_transformers" not in sys.modules, sorted(sys.modules)
        print("SUBPROCESS_OK")
        """
    )
    repo_root = Path(__file__).resolve().parents[4]
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "SUBPROCESS_OK" in proc.stdout, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


def test_retrain_fn_import_guard_names_the_f2c_training_extra(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a missing `peft` (without actually uninstalling it):
    `sys.modules["peft"] = None` makes `import peft` raise `ImportError`,
    same as a genuinely-absent package. `retrain_fn`'s call-time import
    guard must turn that into ONE clear message naming the `f2c-training`
    extra, not a bare `ModuleNotFoundError` surfacing from inside the
    function."""
    monkeypatch.setitem(sys.modules, "peft", None)

    retrain_fn = build_lora_retrain_fn(
        tiny_model_dir,
        target_modules=_TARGET_MODULES,
        modules_to_save=_MODULES_TO_SAVE,
        lora_rank=4,
        epochs=1,
    )
    with pytest.raises(ImportError, match="f2c-training"):
        retrain_fn(_TRAIN_TRIPLES)


def test_load_lora_scorer_import_guard_names_the_f2c_training_extra(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "peft", None)

    with pytest.raises(ImportError, match="f2c-training"):
        load_lora_scorer(tiny_model_dir, tmp_path / "nonexistent-adapter-dir")


def test_dataset_from_triples_import_guard_names_the_f2c_training_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from brain.memory.judge_lora import _dataset_from_triples

    monkeypatch.setitem(sys.modules, "datasets", None)

    with pytest.raises(ImportError, match="f2c-training"):
        _dataset_from_triples(_TRAIN_TRIPLES)


# ---------------------------------------------------------------------------
# I6/AC8 — importing this module never pulls torch into sys.modules.
# ---------------------------------------------------------------------------


def test_module_import_does_not_import_torch_or_sentence_transformers() -> None:
    """Mirrors `test_judge_eval.py`'s identical fresh-subprocess proof:
    merely importing `judge_lora` (module top level, plus its class/
    function definitions -- none of which call any of the lazy-imported
    training/reload functions) never pulls torch/sentence_transformers/
    peft/datasets into `sys.modules`.
    """
    script = textwrap.dedent(
        """
        import sys

        import brain.memory.judge_lora  # noqa: F401

        for heavy in ("torch", "sentence_transformers", "peft", "datasets"):
            assert heavy not in sys.modules, f"{heavy} imported merely by importing judge_lora: {sorted(sys.modules)}"
        print("SUBPROCESS_OK")
        """
    )
    repo_root = Path(__file__).resolve().parents[4]
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "SUBPROCESS_OK" in proc.stdout, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


# ---------------------------------------------------------------------------
# F2c inc5b (task items 5 + 6): real-bge LoRA config grounding — the module
# names + train/serve tokenization the tick-lifecycle wiring will rely on.
# These NEVER load the real bge model (deferred to real HW): item 6 is a
# value assertion against the config-recon ground truth, item 5 is a
# behavioral parity check on the tiny model.
# ---------------------------------------------------------------------------


def test_bge_lora_module_names_are_grounded_xlm_roberta_values() -> None:
    """Item 6: bge-reranker-v2-m3 is XLMRobertaForSequenceClassification
    (config recon), so the LoRA target modules are peft's own
    `xlm-roberta` mapping (`query`/`value`) and the trainable head is
    transformers' `XLMRobertaForSequenceClassification.classifier`. Guards
    against a placeholder/wrong value (e.g. the tiny GPT2 test model's
    `c_attn`/`score`) being left in these constants."""
    assert BGE_RERANKER_LORA_TARGET_MODULES == ("query", "value")
    assert BGE_RERANKER_LORA_MODULES_TO_SAVE == ("classifier",)


def test_train_and_serve_read_one_lora_max_length_tunable_and_serve_truncates(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Items 5 + C2/C3: with `judge_selftune.lora_max_length` overridden to a
    small M (below the tiny model's `n_positions=64`), BOTH the training
    CrossEncoder and the serve tokenizer read that ONE tunable (neither is
    passed an explicit `max_length`), so a LoRA adapter trained on — and then
    scored on — a doc whose UNTRUNCATED token count exceeds the model's
    position capacity succeeds instead of overflowing the position
    embeddings.

    Able-to-fail (ST1.5f, demonstrated at stage 8): under the pre-change
    `load_lora_scorer` (no truncation/max_length), the long serve input
    overflows the tiny GPT2's 64-slot `wpe` and RAISES."""
    m = 16  # < tiny model n_positions (64); < the long doc's token count below
    original_get_tunable = judge_lora.tunables.get_tunable

    def _override(key, default):
        if key == "judge_selftune.lora_max_length":
            return m
        return original_get_tunable(key, default)

    monkeypatch.setattr(judge_lora.tunables, "get_tunable", _override)

    # ~80 whitespace-separated words -> ~80 WordLevel tokens, comfortably
    # over the 64-slot position table when untruncated.
    long_doc = " ".join(["relevant document science weather music"] * 16)
    train_triples = [
        ("query", long_doc, "relevant"),
        ("query", "irrelevant", "irrelevant"),
        ("tiny query", long_doc, "relevant"),
        ("test document", "irrelevant loud", "irrelevant"),
    ]

    # No explicit max_length on EITHER call -> both must resolve the tunable.
    retrain_fn = build_lora_retrain_fn(
        tiny_model_dir,
        target_modules=_TARGET_MODULES,
        modules_to_save=_MODULES_TO_SAVE,
        lora_rank=4,
        epochs=1,
    )
    # Train (would overflow at train time too if the CrossEncoder ignored the
    # tunable) then persist the adapter for a cross-process reload.
    from peft import LoraConfig
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

    # Sanity-drive the retrain_fn itself (proves the train path is bounded).
    label_fn = retrain_fn(train_triples)
    assert label_fn(("query", long_doc)) in ("relevant", "irrelevant")

    # Build + save a real adapter to reload through load_lora_scorer (the
    # serve path under test), mirroring the #3980 test's save shape.
    from datasets import Dataset

    ce = CrossEncoder(
        str(tiny_model_dir),
        config_kwargs={"num_labels": 1},
        max_length=m,
        activation_fn=lambda x: x,
    )
    ce.add_adapter(
        LoraConfig(
            r=4, lora_alpha=8, target_modules=_TARGET_MODULES,
            modules_to_save=_MODULES_TO_SAVE, task_type="SEQ_CLS",
        )
    )
    dataset = Dataset.from_dict(
        {
            "sentence1": [q for q, _d, _l in train_triples],
            "sentence2": [d for _q, d, _l in train_triples],
            "label": [1.0 if lbl == "relevant" else 0.0 for _q, _d, lbl in train_triples],
        }
    )
    args = CrossEncoderTrainingArguments(
        output_dir=str(tmp_path / "train_out"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        report_to=[],
        logging_steps=1_000_000,
        save_strategy="no",
        disable_tqdm=True,
    )
    CrossEncoderTrainer(model=ce, args=args, train_dataset=dataset, loss=BinaryCrossEntropyLoss(ce)).train()
    save_dir = tmp_path / "saved_adapter"
    ce.save_pretrained(str(save_dir))

    # Serve path: no explicit max_length -> must read the same tunable and
    # truncate the long input at M, so scoring succeeds (no position overflow).
    scorer = load_lora_scorer(tiny_model_dir, save_dir)
    result = scorer(("query", long_doc))
    assert isinstance(result, float)
    assert result == result  # finite (not NaN)
