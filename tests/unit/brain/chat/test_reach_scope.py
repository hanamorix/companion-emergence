from brain.chat.tool_recruit import _FILE_TOOLS, REFLEXIVE_CORE, tools_for_capability


def test_files_capability_scopes_to_file_tools():
    got = set(tools_for_capability("files"))
    assert set(_FILE_TOOLS) <= got
    assert set(REFLEXIVE_CORE) <= got
    # NOT the full suite — works/heavy memory tools absent
    assert "crystallize_soul" not in got


def test_unknown_capability_falls_back_to_full_suite():
    from brain.tools import NELL_TOOL_NAMES
    assert set(tools_for_capability("???")) == set(NELL_TOOL_NAMES)
