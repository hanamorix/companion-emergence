"""S220 H10: the per-token RAM figure is replaced by the RAM fit, and the replacements exist and are used.
(That none of the retired RAM names comes back is listed in tests/unit/test_removed_names_stay_removed.py.)"""

from __future__ import annotations


def test_the_replacements_exist_and_are_used() -> None:
    from brain.memory import cost_series, peak_rss, ram_fit
    from brain.memory import reranker as reranker_mod

    assert cost_series.RAM_SHAPE_SERIES in cost_series.SHAPE_SERIES
    assert reranker_mod._ram_shapes.name == cost_series.RAM_SHAPE_SERIES  # noqa: SLF001
    assert callable(ram_fit.fit_ram) and callable(peak_rss.begin)
    fields = reranker_mod.RerankCostEstimate.__dataclass_fields__
    assert "ram_fit" in fields and "ram_fit_kept" in fields
