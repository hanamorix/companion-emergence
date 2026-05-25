import sqlite3
from datetime import UTC, datetime

from brain.recovery.source_reader import read_source_edges, read_source_memories


def _mk_old_schema_source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    conn = sqlite3.connect(src / "memories.db")
    # v0.0.11-shaped: NO state / content_snapshot / recall_count columns
    conn.execute(
        "CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, memory_type TEXT,"
        " domain TEXT, emotions_json TEXT, tags_json TEXT, importance REAL, score REAL,"
        " created_at TEXT, last_accessed_at TEXT, active INTEGER, protected INTEGER,"
        " metadata_json TEXT)"
    )
    conn.execute(
        "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("m1", "original full text", "conversation", "us", "{}", "[]", 5.0, 5.0,
         datetime(2026, 4, 1, tzinfo=UTC).isoformat(), None, 1, 0, "{}"),
    )
    conn.commit()
    conn.close()
    h = sqlite3.connect(src / "hebbian.db")
    h.execute("CREATE TABLE hebbian_edges (memory_a TEXT, memory_b TEXT, weight REAL,"
              " last_strengthened_at TEXT, PRIMARY KEY (memory_a, memory_b))")
    h.execute("INSERT INTO hebbian_edges (memory_a, memory_b, weight) VALUES ('m1','m2',0.7)")
    h.commit()
    h.close()
    return src


def test_reads_old_schema_memories_and_edges(tmp_path):
    src = _mk_old_schema_source(tmp_path)
    mems = read_source_memories(src)
    assert set(mems) == {"m1"}
    assert mems["m1"].content == "original full text"
    assert mems["m1"].state == "active"        # defaulted (column absent)
    edges = read_source_edges(src)
    assert ("m1", "m2", 0.7) in edges


def test_source_not_mutated(tmp_path):
    import hashlib
    src = _mk_old_schema_source(tmp_path)
    before = hashlib.sha256((src / "memories.db").read_bytes()).hexdigest()
    read_source_memories(src)
    after = hashlib.sha256((src / "memories.db").read_bytes()).hexdigest()
    assert before == after


def test_missing_dbs_return_empty(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert read_source_memories(empty) == {}
    assert read_source_edges(empty) == []
