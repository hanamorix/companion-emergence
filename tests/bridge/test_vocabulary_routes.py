"""POST /vocabulary/lookup and POST /vocabulary/add, and the bridge as the one owner of her vocabulary (#801 section 5).

In-process app with background threads off (the bridge test conftest); the two-process case with a real bridge is in
tests/bridge/test_vocabulary_two_process.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.emotion import vocabulary
from brain.health import json_io

FILE = "emotion_vocabulary.json"
AUTH = {"Authorization": "Bearer secret"}


def _entry(name: str, desc: str = "a real description") -> dict:
    return {"name": name, "description": desc, "category": "persona_extension", "decay_half_life_days": 6.0, "intensity_clamp": 10.0}


def _client(persona_dir: Path) -> TestClient:
    return TestClient(build_app(persona_dir=persona_dir, client_origin="tests", auth_token="secret"))


def _entries(persona_dir: Path) -> list[dict]:
    return json.loads((persona_dir / FILE).read_text(encoding="utf-8"))["emotions"]


@pytest.mark.parametrize("route", ["/vocabulary/lookup", "/vocabulary/add"])
def test_both_routes_need_the_bearer_token(persona_dir: Path, route: str) -> None:
    body = {"names": ["joy"]} if route.endswith("lookup") else {"items": [{"name": "x"}]}
    with _client(persona_dir) as c:
        assert c.post(route, json=body).status_code in (401, 403)
        assert c.post(route, json=body, headers={"Authorization": "Bearer wrong"}).status_code in (401, 403)
    assert not (persona_dir / FILE).exists()


def test_lookup_answers_each_name_by_its_entry_or_unknown(persona_dir: Path) -> None:
    (persona_dir / FILE).write_text(json.dumps({"version": 1, "emotions": [_entry("Quiet Dread")]}), encoding="utf-8")
    with _client(persona_dir) as c:
        r = c.post("/vocabulary/lookup", json={"names": ["quiet_dread", "JOY", "no_such_word"]}, headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {
        "entries": [
            {"name": "quiet_dread", "spelling": "Quiet Dread", "category": "persona_extension", "description": "a real description", "half_life_days": 6.0},
            {"name": "JOY", "spelling": "joy", "category": "core", "description": "happiness, delight, warmth", "half_life_days": 3.0},
            {"name": "no_such_word", "unknown": True},
        ]
    }


def test_lookup_all_lists_every_entry_by_its_tidied_name(persona_dir: Path) -> None:
    (persona_dir / FILE).write_text(json.dumps({"version": 1, "emotions": [_entry("Quiet Dread")]}), encoding="utf-8")
    with _client(persona_dir) as c:
        r = c.post("/vocabulary/lookup", json={"names": "all"}, headers=AUTH)
    by_name = {e["name"]: e for e in r.json()["entries"]}
    assert by_name["quiet_dread"]["spelling"] == "Quiet Dread"
    assert by_name["love"]["category"] == "core"
    assert len(by_name) == len(vocabulary.list_all())


def test_lookup_rejects_a_body_that_is_neither_a_list_nor_all(persona_dir: Path) -> None:
    with _client(persona_dir) as c:
        assert c.post("/vocabulary/lookup", json={"names": "joy"}, headers=AUTH).status_code == 422


def test_add_writes_through_the_shared_writer_and_the_next_lookup_sees_it(persona_dir: Path) -> None:
    with _client(persona_dir) as c:
        r = c.post(
            "/vocabulary/add",
            json={"items": [{"name": "Fresh Word", "fields": {"description": "new", "decay_half_life_days": 3.0}}, {"name": "joy"}, {"name": "---"}]},
            headers=AUTH,
        )
        assert r.status_code == 200
        assert r.json() == {
            "results": [
                {"name": "Fresh Word", "spelling": "fresh_word", "outcome": "used", "changed": True},
                {"name": "joy", "spelling": None, "outcome": "refused", "changed": False},  # a core feeling is never written
                {"name": "---", "spelling": None, "outcome": "refused", "changed": False},
            ]
        }
        seen = c.post("/vocabulary/lookup", json={"names": ["Fresh Word"]}, headers=AUTH).json()["entries"][0]
    assert seen["spelling"] == "fresh_word" and seen["description"] == "new" and seen["half_life_days"] == 3.0
    assert [e["name"] for e in _entries(persona_dir)] == ["fresh_word"]


def test_add_ignores_any_field_that_is_not_an_entry_field(persona_dir: Path) -> None:
    with _client(persona_dir) as c:
        c.post("/vocabulary/add", json={"items": [{"name": "word_a", "fields": {"name": "other", "description": "d", "extra": 1}}]}, headers=AUTH)
    entry = _entries(persona_dir)[0]
    assert entry["name"] == "word_a" and entry["description"] == "d" and "extra" not in entry


def test_add_on_a_file_damaged_under_the_running_bridge_heals_it_and_then_writes_the_name(persona_dir: Path) -> None:
    # The owner's write first follows the file, and the owner's reload repairs a damaged file (the quarantine copy keeps the bad bytes).
    with _client(persona_dir) as c:
        (persona_dir / FILE).write_text(json.dumps({"version": 1, "emotions": "not a list"}), encoding="utf-8")
        r = c.post("/vocabulary/add", json={"items": [{"name": "word_a"}]}, headers=AUTH)
    assert r.json()["results"][0]["outcome"] == "used"
    assert [e["name"] for e in _entries(persona_dir)] == ["word_a"]
    assert list(persona_dir.glob(f"{FILE}.corrupt-*"))


@pytest.mark.no_base_failure(reason="the pin of the old cannot_write answer on a damaged file, kept for the case where the repair itself fails: on base nothing repaired on an add, so the add was cannot_write as well")
def test_add_where_the_repair_of_a_damaged_file_is_refused_is_cannot_write_and_leaves_the_file(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_a: object, **_k: object) -> None:
        raise OSError("no room for the quarantine copy")

    with _client(persona_dir) as c:
        (persona_dir / FILE).write_text(json.dumps({"version": 1, "emotions": "not a list"}), encoding="utf-8")
        monkeypatch.setattr(json_io, "_quarantine_copy", refuse)  # the repair cannot start, so the file stays as found
        r = c.post("/vocabulary/add", json={"items": [{"name": "word_a"}]}, headers=AUTH)
    assert r.json()["results"][0]["outcome"] == "cannot_write"
    assert json.loads((persona_dir / FILE).read_text(encoding="utf-8"))["emotions"] == "not a list"
    assert not list(persona_dir.glob(f"{FILE}.corrupt-*"))


def test_the_bridge_is_the_owner_while_it_runs_and_not_after(persona_dir: Path) -> None:
    assert vocabulary.is_bridge_owner() is False
    with _client(persona_dir):
        assert vocabulary.is_bridge_owner() is True
    assert vocabulary.is_bridge_owner() is False


def test_start_up_merges_two_entries_of_one_name_under_the_first_spelling(persona_dir: Path) -> None:
    """Criterion 7 (in-process app): `tender longing` and `tender_longing` end as one entry under `tender longing`.

    (The spec's own example, `comfort seeking`, is a core feeling, so since #801 section 8 its entries are removed, not merged:
    see `test_core_entries_removed_startup.py`.) The registry holds the first spelling, checked on `quiet dread`.
    """
    (persona_dir / FILE).write_text(
        json.dumps(
            {
                "version": 1,
                "emotions": [
                    _entry("tender longing", "first"),
                    _entry("other_word"),
                    _entry("tender_longing", "second"),
                    _entry("quiet dread", "first"),
                    _entry("Quiet_Dread", "second"),
                ],
            }
        ),
        encoding="utf-8",
    )
    with _client(persona_dir) as c:
        entries = _entries(persona_dir)
        assert [e["name"] for e in entries] == ["tender longing", "other_word", "quiet dread"]
        assert [e["description"] for e in entries] == ["first", "a real description", "first"]
        seen = c.post("/vocabulary/lookup", json={"names": ["quiet_dread"]}, headers=AUTH).json()["entries"][0]
    assert seen["spelling"] == "quiet dread"


@pytest.mark.no_base_failure(reason="the other half of the twin merge: with no twin nothing is written, which is also what the code before the start-up merge did")
def test_start_up_with_no_twin_rewrites_nothing(persona_dir: Path) -> None:
    path = persona_dir / FILE
    path.write_text(json.dumps({"version": 1, "emotions": [_entry("one_word"), _entry("Two Words")]}), encoding="utf-8")
    before = path.read_bytes()
    with _client(persona_dir):
        pass
    assert path.read_bytes() == before
