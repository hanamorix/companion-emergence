# tests/unit/brain/files/test_audit.py
from brain.files.audit import audit


def test_audit_reports_whether_the_row_was_written(tmp_path):
    """#346: the append is fail-soft, but a caller that already changed state needs to know a
    row was lost so it can say so loudly instead of leaving a terminal record with no trace."""
    assert audit(tmp_path, event="commit", id="r1", op="create", path="/x") is True
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the persona dir should be")
    assert audit(blocked, event="commit", id="r2", op="create", path="/x") is False
