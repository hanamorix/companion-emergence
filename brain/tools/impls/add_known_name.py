"""add_name tool implementation: the kindled adds a name to her known-names list.

The third way a name reaches the list (spec §5, "Names are added three ways",
item 3; the other two are the temp gate's judge and the on-recall re-appraiser).
Dispatched in the MCP subprocess like ``add_memory`` (not bridge-routed): it only
writes the small known-names file under the OS file lock.

Its one writer is ``brain.memory.known_names.confirm_readd`` (names increment,
S104, S135, S137, S138):

* an entry refused by the admission rules (longer than 100 characters, S153; empty; not a
  string) is not added; the over-length one gets the approved reply (wording file section
  9, S168). Everything else is added as given (S144, S166): periods, several words, common
  words and names with no ASCII letter or digit pass. A name whose whole form is ONE recall stopword
  ("Will") is added, flagged: it then matches only written capitalized, in any position
  (S104, S151);
* a name on the rejects list (removed earlier, by her or by the re-appraiser) is
  not added on the first call: the reply carries the approved question (wording
  file section 7). If her NEXT tool call is ``add_name`` with the same name, it is
  added and taken off the rejects list; any other next call lapses the question
  (the dispatch hook, ``known_names.note_tool_call``);
* any other name is added, or made hers if already listed (her spelling, source
  ``tool``).

The tool's identifier and persona-facing text are the owner's approved wording in
``brain/tools/schemas.py``. This module's own name is not persona-facing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from brain import prompt_strings
from brain.memory.known_names import confirm_readd

logger = logging.getLogger(__name__)

# The approved reply for a name on the rejects list (S137, wording file section 7).
_REJECTED_REPLY = prompt_strings.register("tools.add_name.rejected_reply")
# The approved reply for a name over 100 characters (S168, wording file section 9).
_TOO_LONG_REPLY = prompt_strings.register("tools.add_name.too_long_reply")


def add_known_name(name: Any, *, persona_dir: Path, **_unused: Any) -> dict:
    """Add ``name`` to the persona's known-names list (source ``tool``).

    Returns:
      - ``added``: True when the name is on the list after the call (a name
        already listed counts); False when the admission rules refused it, the
        write could not be made (logged, never raised), the name is over 100
        characters, or the name is on the rejects list and this call asked the S137
        question;
      - ``name``: the name as given, whitespace-trimmed (display form intact); an
        empty string when the argument was not a string;
      - ``message`` (only when this call asked the S137 question, or refused a name over
        100 characters, S168): the approved text, with ``{name}`` = the name as given.

    Never raises for a bad argument or an I/O failure.
    """
    given = name.strip() if isinstance(name, str) else ""
    if not given:
        return {"added": False, "name": given}
    outcome = confirm_readd(persona_dir, given)
    if outcome == "asked":
        return {"added": False, "name": given, "message": _REJECTED_REPLY.format(name=given)}
    if outcome == "too_long":
        return {"added": False, "name": given, "message": _TOO_LONG_REPLY.format(name=given)}
    return {"added": outcome == "added", "name": given}
