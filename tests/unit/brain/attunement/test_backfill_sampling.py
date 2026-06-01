"""Tests for backfill.select_sample — stratified sample with topic diversity."""
from brain.attunement.backfill import Window, select_sample
from brain.attunement.store import BufferTurn


def test_select_sample_empty_returns_empty():
    assert select_sample([]) == []


def test_select_sample_returns_rate_proportion():
    turns = [BufferTurn(id=f"t{i}", content=f"topic {i % 5}") for i in range(100)]
    windows = [Window(id=f"w{i}", turns=tuple(turns[i:i+10])) for i in range(50)]
    sample = select_sample(windows, rate=0.2)
    # 50 windows, rate=0.2 → target 10. With 5 distinct topics, clusters reduce
    # available distinct-fingerprint picks, so first-pass yields ~5; second pass
    # tops up to 10.
    assert 5 <= len(sample) <= 12


def test_select_sample_prefers_topic_diversity():
    # 5 identical-topic windows + 5 unique-topic windows; rate=0.2 → target 2
    same = [
        Window(
            id=f"sw{i}",
            turns=(BufferTurn(id=f"st{i}", content="same topic same words"),),
        )
        for i in range(5)
    ]
    diverse = [
        Window(
            id=f"dw{i}",
            turns=(BufferTurn(id=f"dt{i}", content=f"unique words {i}"),),
        )
        for i in range(5)
    ]
    sample = select_sample(same + diverse, rate=0.2)
    # The sample should include at least one from the diverse cluster set,
    # not just two from the same cluster
    diverse_ids = {f"dw{i}" for i in range(5)}
    assert any(w.id in diverse_ids for w in sample)


def test_select_sample_floors_at_one_window():
    # Tiny input: 3 windows, rate=0.2 → 0.6 → at-least-1
    windows = [
        Window(id=f"w{i}", turns=(BufferTurn(id=f"t{i}", content=f"x{i}"),))
        for i in range(3)
    ]
    sample = select_sample(windows, rate=0.2)
    assert len(sample) >= 1


def test_select_sample_deterministic_same_input_same_output():
    turns = [BufferTurn(id=f"t{i}", content=f"word {i % 3}") for i in range(30)]
    windows = [Window(id=f"w{i}", turns=tuple(turns[i:i+10])) for i in range(20)]
    sample1 = select_sample(windows, rate=0.3)
    sample2 = select_sample(windows, rate=0.3)
    assert [w.id for w in sample1] == [w.id for w in sample2]
