"""Tests for `brain.memory.judge_lora` — the LoRA weight-retrain MECHANISM
(F2c inc5a; LoRA-then-merge since inc7) (spec `f2c-judge-selftune-spec.md` §5
[LoRA (mid), merge each week], AC6 [champion/challenger + rollback], AC7 as
the merge→save→reload round trip).

OFFLINE, TINY MODEL ONLY (never the real bge-reranker-v2-m3 — see
`build-for-potato-computers-baseline` / the BUILD instructions this
increment was scoped under): every test here builds a FROM-SCRATCH,
randomly-initialized tiny GPT2-for-sequence-classification model (2
layers, hidden=32) with a minimal from-scratch tokenizer, entirely on
disk, and loads everything with `local_files_only=True` — no network
access, no hub download, ever. GPT2 is the chosen architecture because its
`*ForSequenceClassification` head is a plain `nn.Linear` literally named
`.score` (see `transformers/models/gpt2/modeling_gpt2.py`), so the tests
below exercise a `modules_to_save` head that must be carried into the merged
weights.

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
from brain.memory.judge_full_ft import build_full_ft_retrain_fn, load_full_scorer
from brain.memory.judge_lora import (
    BGE_RERANKER_LORA_MODULES_TO_SAVE,
    BGE_RERANKER_LORA_TARGET_MODULES,
    LoraRollbackHandle,
    build_lora_retrain_fn,
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
# the fused qkv `Conv1D` layer) + its classification head name.
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
# F2c inc7 — LoRA-then-merge (route ii): M1 / M2 / M4 / M6 (1.5-criteria).
# The LoRA is built with peft `get_peft_model`, trained, merged in memory, and
# ONLY the merged plain checkpoint is saved; nothing saves or reloads an
# adapter (spec §5 merge ruling; the adapter-reload tripwire was removed with
# the reload path it guarded).
# ---------------------------------------------------------------------------

_EVAL_PAIRS = [
    ("query", "relevant"), ("tiny query", "irrelevant document"), ("dog", "the dog is great"),
    ("cat", "red apple banana"), ("music", "loud music"), ("science", "the science test"),
]
_TRAIN2 = [
    ("music", "loud music", "relevant"),
    ("music", "quiet weather", "irrelevant"),
    ("science", "science is great", "relevant"),
    ("weather", "blue banana", "irrelevant"),
]


def _logits(model_or_dir) -> np.ndarray:
    from sentence_transformers import CrossEncoder

    ce = model_or_dir if not isinstance(model_or_dir, (str, Path)) else CrossEncoder(str(model_or_dir))
    return np.array([float(ce.predict([p], activation_fn=lambda x: x)[0]) for p in _EVAL_PAIRS])


def _head(hf_model) -> np.ndarray:
    """The GPT2 `.score` head as the model will use it (for a PeftModel: the
    active modules_to_save copy)."""
    m = hf_model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    head = m.score
    if hasattr(head, "modules_to_save"):
        head = head.modules_to_save["default"]
    return head.weight.detach().cpu().numpy().copy()


def _file_hashes(d: Path) -> dict[str, str]:
    import hashlib

    return {
        str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(d.rglob("*")) if p.is_file()
    }


class _TrainSpy:
    """Wraps CrossEncoderTrainer.train to record the model's logits (and the
    trained head) immediately BEFORE and AFTER training, i.e. the start-from
    model and the trained pre-merge LoRA model."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer

        self.before: list[np.ndarray] = []
        self.after: list[np.ndarray] = []
        self.heads: list[np.ndarray] = []
        real_train = CrossEncoderTrainer.train
        spy = self

        def train(trainer_self, *a, **k):
            spy.before.append(_logits(trainer_self.model))
            out = real_train(trainer_self, *a, **k)
            spy.after.append(_logits(trainer_self.model))
            spy.heads.append(_head(trainer_self.model[0].model))
            return out

        monkeypatch.setattr(CrossEncoderTrainer, "train", train)


def _forbid_adapter_io(monkeypatch: pytest.MonkeyPatch) -> None:
    from peft import PeftModel
    from transformers import PreTrainedModel

    def boom(*a, **k):
        raise AssertionError("no adapter may be saved or reloaded (spec §5 merge ruling)")

    monkeypatch.setattr(PeftModel, "from_pretrained", classmethod(lambda cls, *a, **k: boom()))
    monkeypatch.setattr(PeftModel, "save_pretrained", boom)
    monkeypatch.setattr(PreTrainedModel, "load_adapter", boom, raising=False)


def _no_adapter_files(d: Path) -> bool:
    return not any(d.rglob("adapter_config.json")) and not any(d.rglob("adapter_model*"))


def test_lora_then_merge_on_base_saves_a_plain_checkpoint_that_reproduces(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1. Bite: at d25e1002 the builder saved an adapter (adapter_config.json)."""
    from transformers import AutoModelForSequenceClassification

    spy = _TrainSpy(monkeypatch)
    _forbid_adapter_io(monkeypatch)
    torch = pytest.importorskip("torch")
    torch.manual_seed(11)
    save = tmp_path / "merged"
    label_fn = build_lora_retrain_fn(
        tiny_model_dir, target_modules=_TARGET_MODULES, modules_to_save=_MODULES_TO_SAVE,
        lora_rank=4, epochs=3, save_dir=save,
    )(_TRAIN_TRIPLES)

    assert _no_adapter_files(save), sorted(p.name for p in save.rglob("*"))
    trained = spy.after[0]
    assert np.max(np.abs(trained - spy.before[0])) > 1e-4, "training moved the model"
    np.testing.assert_allclose(_logits(save), trained, atol=1e-5)
    scorer = load_full_scorer(save)
    np.testing.assert_allclose(np.array([scorer(p) for p in _EVAL_PAIRS]), trained, atol=1e-5)
    merged_head = _head(AutoModelForSequenceClassification.from_pretrained(str(save), local_files_only=True))
    np.testing.assert_allclose(merged_head, spy.heads[0], atol=1e-6)
    # The eval label-fn scores the MERGED model: its labels agree with the
    # reloaded checkpoint's scores through the same label mapping.
    from brain.memory.relevance_judge import label_for_score

    assert [label_fn(p) for p in _EVAL_PAIRS] == [label_for_score(float(x))[0] for x in _logits(save)]


def test_save_dir_none_writes_nothing(tiny_model_dir: Path, tmp_path: Path) -> None:
    label_fn = build_lora_retrain_fn(
        tiny_model_dir, target_modules=_TARGET_MODULES, modules_to_save=_MODULES_TO_SAVE,
        lora_rank=4, epochs=1,
    )(_TRAIN_TRIPLES)
    assert label_fn(_FIXED_EVAL_ITEMS[0]) in ("relevant", "irrelevant")
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("start_kind", ["merged", "full_ft"])
def test_lora_then_merge_on_a_plain_checkpoint_starts_from_it(
    start_kind: str, tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M2 (week 2+). Bite: at d25e1002 the tick always passed the base."""
    import torch

    torch.manual_seed(21)
    c = tmp_path / "C"
    if start_kind == "merged":
        build_lora_retrain_fn(
            tiny_model_dir, target_modules=_TARGET_MODULES, modules_to_save=_MODULES_TO_SAVE,
            lora_rank=4, epochs=3, save_dir=c,
        )(_TRAIN_TRIPLES)
    else:
        build_full_ft_retrain_fn(tiny_model_dir, epochs=2, save_full_dir=c)(_TRAIN_TRIPLES)
    c_logits = _logits(c)
    base_logits = _logits(tiny_model_dir)
    assert np.max(np.abs(c_logits - base_logits)) > 1e-4, "C differs from base (so start-from can bite)"
    c_hashes = _file_hashes(c)

    spy = _TrainSpy(monkeypatch)
    _forbid_adapter_io(monkeypatch)
    torch.manual_seed(22)
    d = tmp_path / "D"
    build_lora_retrain_fn(
        c, target_modules=_TARGET_MODULES, modules_to_save=_MODULES_TO_SAVE,
        lora_rank=4, epochs=3, save_dir=d,
    )(_TRAIN2)

    np.testing.assert_allclose(spy.before[0], c_logits, atol=1e-5)  # started FROM C, not the base
    assert np.max(np.abs(spy.after[0] - c_logits)) > 1e-4
    np.testing.assert_allclose(_logits(d), spy.after[0], atol=1e-5)
    assert _no_adapter_files(d)
    assert _file_hashes(c) == c_hashes, "the start checkpoint is not rewritten"


def test_training_wraps_the_loaded_model_once(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M6 (ADVISORY, memory peak): one CrossEncoder load of the start weights
    per retrain; the LoRA wraps and merges that same model in place."""
    import sentence_transformers

    real = sentence_transformers.CrossEncoder
    loads: list[str] = []

    class Counting(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            loads.append(str(a[0]) if a else "")
            super().__init__(*a, **k)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", Counting)
    build_lora_retrain_fn(
        tiny_model_dir, target_modules=_TARGET_MODULES, modules_to_save=_MODULES_TO_SAVE,
        lora_rank=4, epochs=1,
    )(_TRAIN_TRIPLES)
    assert loads == [str(tiny_model_dir)]


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


def test_merged_checkpoint_carries_the_lora_max_length_to_the_serve_loader(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M5: with `judge_selftune.lora_max_length` overridden to a small M (below
    the tiny model's 64 positions) and no explicit max_length, a LoRA week's
    merged checkpoint reloads with max length M (the truncation length travels
    with the saved checkpoint), and `load_full_scorer` scores a doc longer than
    64 tokens. Oracle self-test: the same reload with the saved truncation
    stripped comes back with a different max length (sbert then clamps to the
    model's 64 positions rather than overflowing on this stack), so the
    equality check can fail."""
    import json
    import shutil

    from sentence_transformers import CrossEncoder

    m = 16
    original_get_tunable = judge_lora.tunables.get_tunable

    def _override(key, default):
        if key == "judge_selftune.lora_max_length":
            return m
        return original_get_tunable(key, default)

    monkeypatch.setattr(judge_lora.tunables, "get_tunable", _override)
    long_doc = " ".join(["relevant document science weather music"] * 16)  # ~80 tokens
    train = [
        ("query", long_doc, "relevant"),
        ("query", "irrelevant", "irrelevant"),
        ("tiny query", long_doc, "relevant"),
        ("test document", "irrelevant loud", "irrelevant"),
    ]
    save = tmp_path / "merged"
    build_lora_retrain_fn(
        tiny_model_dir, target_modules=_TARGET_MODULES, modules_to_save=_MODULES_TO_SAVE,
        lora_rank=4, epochs=1, save_dir=save,
    )(train)
    assert CrossEncoder(str(save)).max_length == m
    score = load_full_scorer(save)(("query", long_doc))
    assert isinstance(score, float) and score == score

    # Self-test: strip the saved truncation -> the reloaded max length is no
    # longer M.
    broken = tmp_path / "broken"
    shutil.copytree(save, broken)
    cfg = broken / "tokenizer_config.json"
    data = json.loads(cfg.read_text())
    data["model_max_length"] = 1_000_000
    cfg.write_text(json.dumps(data))
    tok = broken / "tokenizer.json"
    tdata = json.loads(tok.read_text())
    tdata["truncation"] = None
    tok.write_text(json.dumps(tdata))
    assert CrossEncoder(str(broken)).max_length != m
