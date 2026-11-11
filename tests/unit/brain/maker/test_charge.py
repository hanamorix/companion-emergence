from datetime import UTC, datetime

from brain.maker.charge import MakerCharge, accumulate, load_charge, save_charge


def test_accumulate_adds_weighted_signals_and_decays(tmp_path):
    save_charge(tmp_path, MakerCharge(charge=0.0, last_tick_ts="2026-06-14T00:00:00+00:00",
                                      last_fire_ts=None, prior_soul_count=0))
    now = datetime(2026, 6, 14, 1, 0, 0, tzinfo=UTC)  # 1h later
    c = accumulate(tmp_path, emotional_intensity=4.0, soul_delta=1, dream_count=2, now=now,
                   w_emotion=0.1, w_soul=2.0, w_dream=0.5, decay_per_hour=0.9)
    # decay applies to prior 0.0 (stays 0), then += 0.1*4 + 2.0*1 + 0.5*2 = 0.4+2.0+1.0 = 3.4
    assert abs(c.charge - 3.4) < 1e-9
    assert c.last_tick_ts == now.isoformat()


def test_decay_reduces_idle_charge(tmp_path):
    save_charge(tmp_path, MakerCharge(charge=10.0, last_tick_ts="2026-06-14T00:00:00+00:00",
                                      last_fire_ts=None, prior_soul_count=0))
    now = datetime(2026, 6, 14, 2, 0, 0, tzinfo=UTC)  # 2h later
    c = accumulate(tmp_path, emotional_intensity=0.0, soul_delta=0, dream_count=0, now=now,
                   w_emotion=0.1, w_soul=2.0, w_dream=0.5, decay_per_hour=0.5)
    # 10 * 0.5**2 = 2.5, plus zero additions
    assert abs(c.charge - 2.5) < 1e-9


def test_load_missing_returns_cold_default(tmp_path):
    c = load_charge(tmp_path)
    assert c.charge == 0.0
    assert c.last_fire_ts is None


def test_load_corrupt_returns_cold_default(tmp_path):
    (tmp_path / "maker_charge.json").write_text("{ broken", encoding="utf-8")
    c = load_charge(tmp_path)
    assert c.charge == 0.0  # fail-safe


# ---- #721: the charge remembers what built it ----


def test_accumulate_tracks_each_source_part_and_decays_them(tmp_path):
    import pytest

    save_charge(tmp_path, MakerCharge(charge=0.0, last_tick_ts="2026-06-14T00:00:00+00:00",
                                      last_fire_ts=None, prior_soul_count=0))
    first = datetime(2026, 6, 14, 1, 0, 0, tzinfo=UTC)
    accumulate(tmp_path, emotional_intensity=4.0, soul_delta=1, dream_count=2, now=first,
               w_emotion=0.1, w_soul=2.0, w_dream=0.5, decay_per_hour=0.5)
    c = accumulate(tmp_path, emotional_intensity=0.0, soul_delta=0, dream_count=0, now=datetime(2026, 6, 14, 2, 0, 0, tzinfo=UTC),
                   w_emotion=0.1, w_soul=2.0, w_dream=0.5, decay_per_hour=0.5)
    # Each part decays with the total: 0.4, 2.0 and 1.0 halve after an idle hour.
    assert (c.emotion_part, c.soul_part, c.dream_part) == pytest.approx((0.2, 1.0, 0.5))
    assert c.emotion_part + c.soul_part + c.dream_part == pytest.approx(c.charge)
    assert load_charge(tmp_path).soul_part == pytest.approx(1.0)


def test_an_older_charge_file_loads_with_zero_parts(tmp_path):
    """#721: the parts are additive and optional; a file an older brain wrote has none, and bad values read as 0."""
    (tmp_path / "maker_charge.json").write_text(
        '{"charge": 3.0, "last_tick_ts": null, "last_fire_ts": null, "prior_soul_count": 2, "soul_part": "lots"}',
        encoding="utf-8",
    )
    c = load_charge(tmp_path)
    assert (c.charge, c.prior_soul_count) == (3.0, 2)
    assert (c.emotion_part, c.soul_part, c.dream_part) == (0.0, 0.0, 0.0)
