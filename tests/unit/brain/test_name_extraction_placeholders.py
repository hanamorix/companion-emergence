"""Names increment (name-recall fix): the persona-facing strings carry no em-dash or other LLM tell, and the app's
job labels are the approved ones (S107; criteria N18, N23-N25).

The approved wording itself (the texts of ``~/.claude/plans/name-recall-fix-wording.md`` sections 1, 4, 7 and 8, the two
consolidation prompts, the emitted schema entries) is held by ``tests/unit/test_persona_wording_snapshot.py``, which fails
with a diff when it changes. The owner's wording replaced every placeholder (the last one, the `rerank_width_remeasure`
label, is final: wording file section 8); that no placeholder comes back is listed in
tests/unit/test_removed_names_stay_removed.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain import prompt_strings
from brain.tools import schemas

_REPO = Path(__file__).resolve().parents[3]

_TELLS = ("—", "–", "delve", "tapestry", "it's not just", "not just a", "testament")

# The persona-facing texts, read from the live code (the approved wording itself is held by
# tests/unit/test_persona_wording_snapshot.py; these checks keep the claudism rule on whatever text is there).
_LIVE_TEXTS = {
    "add_name description": schemas.ADD_NAME_TOOL_DESCRIPTION,
    "add_name param": schemas.ADD_NAME_PARAM_DESCRIPTION,
    "remove_name description": schemas.REMOVE_NAME_TOOL_DESCRIPTION,
    "remove_name param": schemas.REMOVE_NAME_PARAM_DESCRIPTION,
    "rejected reply": prompt_strings.get("tools.add_name.rejected_reply"),
    "too long reply": prompt_strings.get("tools.add_name.too_long_reply"),
    "reappraiser prompt": prompt_strings.get("engines.consolidation.reappraiser_prompt"),
    "classifier prompt": prompt_strings.get("engines.consolidation.classifier_prompt"),
}


@pytest.mark.no_base_failure(reason='#540 audit guard, snapshot or registry of what already exists (it replaces removed tests); passes on the old code by design')
@pytest.mark.parametrize("text", list(_LIVE_TEXTS.values()), ids=list(_LIVE_TEXTS))
def test_no_em_dash_or_llm_tell(text):
    for tell in _TELLS:
        assert tell not in text.lower(), tell


@pytest.mark.no_base_failure(reason='#540 audit guard, snapshot or registry of what already exists (it replaces removed tests); passes on the old code by design')
def test_app_job_descriptors_carry_the_three_final_labels():
    """Wording file section 4 (two labels) and section 8 (`rerank_width_remeasure`, final):
    each label is present at its site."""
    lines = (_REPO / "app/src/App.tsx").read_text(encoding="utf-8").splitlines()
    stripped = [line.strip() for line in lines]
    assert 'cosine_floor_bootstrap: "similarity pass mark setup",' in stripped
    assert 'rerank_floor_bootstrap: "re-scoring pass mark setup",' in stripped
    assert 'rerank_width_remeasure: "re-scoring cost check",' in stripped
