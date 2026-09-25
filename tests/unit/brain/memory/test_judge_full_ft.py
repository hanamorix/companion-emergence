"""Tests for `brain.memory.judge_full_ft` — F2c inc6's FULL fine-tune
weight-retrain MECHANISM (spec `f2c-judge-selftune-spec.md` §1 [beefy tier =
full fine-tune], §5 [Full FT (beefy)], AC7-analogue [full-model reload
round-trip], AC9 [Haiku-oracle note]).

OFFLINE, TINY MODEL ONLY (never the real bge-reranker-v2-m3 — the BUILD
deferral + `build-for-potato-computers-baseline`): every test builds a
FROM-SCRATCH, randomly-initialized tiny GPT2-for-sequence-classification
model + a minimal from-scratch tokenizer entirely on disk, loaded with
`local_files_only=True` — no network, no hub download, ever. Unlike the LoRA
tests (`test_judge_lora.py`), there is NO adapter: `build_full_ft_retrain_fn`
trains ALL params, saves a COMPLETE model, and reloads it via a PLAIN
`CrossEncoder` — proving the full-FT path does NOT go through peft (bug #3980
is LoRA-adapter-specific and cannot apply here).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from brain.memory.judge_eval import RollbackHandle, run_champion_challenger
from brain.memory.judge_full_ft import build_full_ft_retrain_fn, load_full_scorer

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
    """A from-scratch, randomly-initialized tiny GPT2ForSequenceClassification
    model + tokenizer saved to `dest_dir` — no download."""
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
    base = tmp_path_factory.mktemp("judge_full_ft_tiny_base")
    return _build_tiny_model_dir(base / "tiny_gpt2_seqclass")


_TRAIN_TRIPLES = [
    ("query", "relevant", "relevant"),
    ("query", "irrelevant", "irrelevant"),
    ("test document", "relevant test", "relevant"),
    ("tiny query", "irrelevant document", "irrelevant"),
]
_FIXED_EVAL_ITEMS = [("query", "relevant"), ("tiny query", "irrelevant document")]


# ---------------------------------------------------------------------------
# C1 — build_full_ft_retrain_fn trains (all params) + returns a valid label fn
# and persists a full model to save_full_dir.
# ---------------------------------------------------------------------------


def test_full_ft_retrain_fn_trains_and_persists_a_full_model(
    tiny_model_dir: Path, tmp_path: Path
) -> None:
    staged = tmp_path / "staged_full"
    retrain_fn = build_full_ft_retrain_fn(tiny_model_dir, epochs=1, save_full_dir=staged)
    label_fn = retrain_fn(_TRAIN_TRIPLES)
    for item in _FIXED_EVAL_ITEMS:
        assert label_fn(item) in ("relevant", "irrelevant"), item
    # A full model was saved to the staged dir (weights present).
    assert staged.is_dir() and any(staged.iterdir()), "full model not persisted to save_full_dir"
    assert any(p.name.startswith("model") or p.suffix in (".safetensors", ".bin") for p in staged.iterdir()), (
        "no model weight file in the saved full-model dir"
    )


def test_full_ft_save_full_dir_none_leaves_no_save(tiny_model_dir: Path, tmp_path: Path) -> None:
    retrain_fn = build_full_ft_retrain_fn(tiny_model_dir, epochs=1)
    retrain_fn(_TRAIN_TRIPLES)
    assert not (tmp_path / "staged_full").exists()


# ---------------------------------------------------------------------------
# C2 — the full-FT path does NOT go through peft: no adapter_config.json, and
# load_full_scorer works even when PeftModel.from_pretrained is made to fail.
# ---------------------------------------------------------------------------


def test_full_ft_save_writes_no_adapter_config(tiny_model_dir: Path, tmp_path: Path) -> None:
    staged = tmp_path / "staged_full"
    build_full_ft_retrain_fn(tiny_model_dir, epochs=1, save_full_dir=staged)(_TRAIN_TRIPLES)
    assert not (staged / "adapter_config.json").exists(), (
        "a full fine-tune must NOT write an adapter_config.json (that would trigger the #3980 "
        "peft auto-load path)"
    )


def test_load_full_scorer_does_not_use_peft_reload(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C2: reloading a saved full model must use the PLAIN CrossEncoder path,
    NEVER `peft.PeftModel.from_pretrained`. Make the peft reload fail loudly;
    `load_full_scorer` must still succeed (proving it never called it)."""
    staged = tmp_path / "staged_full"
    build_full_ft_retrain_fn(tiny_model_dir, epochs=1, save_full_dir=staged)(_TRAIN_TRIPLES)

    import peft

    def _boom(*a, **k):
        raise AssertionError("load_full_scorer must NOT use PeftModel.from_pretrained (#3980 is LoRA-only)")

    monkeypatch.setattr(peft.PeftModel, "from_pretrained", staticmethod(_boom))

    scorer = load_full_scorer(staged)
    result = scorer(_FIXED_EVAL_ITEMS[0])
    assert isinstance(result, float) and result == result  # finite, no peft path taken


# ---------------------------------------------------------------------------
# C3 — AC7 analogue: full-model train -> save -> reload reproduces scores.
# ---------------------------------------------------------------------------


def test_full_model_train_save_reload_reproduces_scores(tiny_model_dir: Path, tmp_path: Path) -> None:
    """The AC7 ANALOGUE for the full-FT tier: training a FULL CrossEncoder,
    saving it, and reloading via `load_full_scorer` reproduces the trained
    model's RAW scores on a fixed input to atol=1e-5. Compared on raw logits
    (identity activation) on BOTH sides so a forgotten identity (double-
    sigmoid) would fail (stage-3 F4). Unlike LoRA #3980, the plain reload
    restores the WHOLE model, so this passes."""
    import torch
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

    torch.manual_seed(0)
    ce = CrossEncoder(
        str(tiny_model_dir), local_files_only=True,
        config_kwargs={"num_labels": 1}, activation_fn=lambda x: x,
    )
    # NO add_adapter — full fine-tune.
    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "sentence1": [q for q, _d, _l in _TRAIN_TRIPLES],
            "sentence2": [d for _q, d, _l in _TRAIN_TRIPLES],
            "label": [1.0 if lbl == "relevant" else 0.0 for _q, _d, lbl in _TRAIN_TRIPLES],
        }
    )
    args = CrossEncoderTrainingArguments(
        output_dir=str(tmp_path / "train_out"),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        report_to=[],
        logging_steps=1_000_000,
        save_strategy="no",
        disable_tqdm=True,
    )
    CrossEncoderTrainer(model=ce, args=args, train_dataset=dataset, loss=BinaryCrossEntropyLoss(ce)).train()
    trained_scores = np.array(ce.predict(_FIXED_EVAL_ITEMS, activation_fn=lambda x: x))

    save_dir = tmp_path / "saved_full_model"
    ce.save_pretrained(str(save_dir))
    # No adapter written for a full model.
    assert not (save_dir / "adapter_config.json").exists()

    scorer = load_full_scorer(save_dir)
    reloaded_scores = np.array([scorer(item) for item in _FIXED_EVAL_ITEMS])

    assert np.allclose(trained_scores, reloaded_scores, atol=1e-5), (
        f"trained={trained_scores} reloaded={reloaded_scores}"
    )


# ---------------------------------------------------------------------------
# C4 — plugs into run_champion_challenger with no shim.
# ---------------------------------------------------------------------------


def test_full_ft_retrain_fn_matches_champion_challenger_contract(tiny_model_dir: Path) -> None:
    retrain_fn = build_full_ft_retrain_fn(tiny_model_dir, epochs=1)

    def champion(item: tuple[str, str]) -> str:
        return "irrelevant"  # deliberately-bad champion so the challenger has room

    test_items = [(item, "relevant") for item in _FIXED_EVAL_ITEMS] * 20

    result = run_champion_challenger(
        champion=champion,
        retrain_fn=retrain_fn,
        train_items=_TRAIN_TRIPLES,
        test_items=test_items,
        rollback=RollbackHandle(),
        alpha=0.05,
        min_n=1,
    )
    assert result.n_test == len(test_items)
    assert isinstance(result.accepted, bool)


# ---------------------------------------------------------------------------
# C5 — deterministic under a fixed seed.
# ---------------------------------------------------------------------------


def test_full_ft_training_is_deterministic_under_a_fixed_seed(tiny_model_dir: Path) -> None:
    import torch

    def train_and_score() -> list[str]:
        torch.manual_seed(1234)
        retrain_fn = build_full_ft_retrain_fn(tiny_model_dir, epochs=2)
        label_fn = retrain_fn(_TRAIN_TRIPLES)
        return [label_fn(item) for item in _FIXED_EVAL_ITEMS]

    assert train_and_score() == train_and_score()


# ---------------------------------------------------------------------------
# C6 — importing the module never pulls torch/sentence_transformers/peft/
# datasets (I6/AC8).
# ---------------------------------------------------------------------------


def test_module_import_does_not_import_torch_or_sentence_transformers() -> None:
    script = textwrap.dedent(
        """
        import sys

        import brain.memory.judge_full_ft  # noqa: F401

        for heavy in ("torch", "sentence_transformers", "peft", "datasets"):
            assert heavy not in sys.modules, f"{heavy} imported merely by importing judge_full_ft: {sorted(sys.modules)}"
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
# C16 — AC9 Haiku-oracle durable note present in the training code.
# ---------------------------------------------------------------------------


def test_haiku_oracle_note_present_in_full_ft_module() -> None:
    src = (Path(__file__).resolve().parents[4] / "brain" / "memory" / "judge_full_ft.py").read_text()
    assert "Haiku" in src and "ORACLE" in src, "missing the durable Haiku-oracle note (spec §6/AC9)"


# ---------------------------------------------------------------------------
# F2c inc7 — M3: a full fine-tune continues from the persona's own checkpoint
# (one evolving model), never silently from the base.
# ---------------------------------------------------------------------------

_M3_EVAL = [("query", "relevant"), ("tiny query", "irrelevant document"), ("test document", "relevant test")]


def _m3_logits(d: Path) -> np.ndarray:
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(str(d))
    return np.array([float(ce.predict([p], activation_fn=lambda x: x)[0]) for p in _M3_EVAL])


def test_full_ft_continues_from_a_checkpoint(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M3. Bite: at d25e1002 the tick always passed the base as the start."""
    import torch
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer

    torch.manual_seed(31)
    c = tmp_path / "C"
    build_full_ft_retrain_fn(tiny_model_dir, epochs=2, save_full_dir=c)(_TRAIN_TRIPLES)
    c_logits = _m3_logits(c)
    assert np.max(np.abs(c_logits - _m3_logits(tiny_model_dir))) > 1e-4

    before: list[np.ndarray] = []
    real_train = CrossEncoderTrainer.train

    def train(self, *a, **k):
        before.append(
            np.array([float(self.model.predict([p], activation_fn=lambda x: x)[0]) for p in _M3_EVAL])
        )
        return real_train(self, *a, **k)

    monkeypatch.setattr(CrossEncoderTrainer, "train", train)
    torch.manual_seed(32)
    d = tmp_path / "D"
    build_full_ft_retrain_fn(c, epochs=2, save_full_dir=d)(_TRAIN_TRIPLES)
    np.testing.assert_allclose(before[0], c_logits, atol=1e-5)  # started FROM C
    d_logits = _m3_logits(d)
    assert np.max(np.abs(d_logits - c_logits)) > 1e-5, "weights moved"
    scorer = load_full_scorer(d)
    np.testing.assert_allclose(np.array([scorer(p) for p in _M3_EVAL]), d_logits, atol=1e-5)
    assert not any(d.rglob("adapter_config.json"))
