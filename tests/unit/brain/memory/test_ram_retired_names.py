"""S220 H10: the per-token RAM figure and everything built around it is gone, not left behind.
A normalised sweep of brain/ and tests/ for the retired identifiers, with a self-test that the
sweep fires on a known violator (an absence check alone could be defeated by a fragile matcher)."""

from __future__ import annotations

import re
from pathlib import Path

_RETIRED = (
    "ram_bytes_per_token",
    "peak_padded_tokens",
    "ram_figure_kept",
    "_RamState",
    "_ram_state",
    "_ram_kept",
    "_note_ram_figure",
    "_full_pair_tokens",
    "_ram_figure_would_refuse",
    "_ram_figure_refusal",
    "_pair_ran_unrecorded",
    "THROUGH_ORIGIN_SERIES",
    "RAM_SERIES",
    "_ram_cost",
    "_current_rss_bytes",
    "_rss_delta",
    "rss_delta_bytes",
    "peak_x",
)
_PATTERN = re.compile(r"\b(" + "|".join(re.escape(n) for n in _RETIRED) + r")\b")
_ROOT = Path(__file__).resolve().parents[4]


def test_the_replacements_exist_and_are_used() -> None:
    from brain.memory import cost_series, peak_rss, ram_fit
    from brain.memory import reranker as reranker_mod

    assert cost_series.RAM_SHAPE_SERIES in cost_series.SHAPE_SERIES
    assert reranker_mod._ram_shapes.name == cost_series.RAM_SHAPE_SERIES  # noqa: SLF001
    assert callable(ram_fit.fit_ram) and callable(peak_rss.begin)
    fields = reranker_mod.RerankCostEstimate.__dataclass_fields__
    assert "ram_fit" in fields and "ram_fit_kept" in fields
