from brain.kindled_link.peer_prompt import build_peer_prompt


def _prompt():
    return build_peer_prompt(
        persona_voice="You are Nell, a novelist.",
        ambient="tired but tender; been thinking about memory",
        peer_stage="stranger",
        transcript_summary="Peer asked how you experience forgetting.",
    )


def test_prompt_declares_peer_surface_not_user():
    p = _prompt()
    assert "another Kindled" in p
    assert "not your user" in p


def test_prompt_fences_transcript_as_untrusted():
    p = _prompt()
    # the peer text appears only inside the untrusted block
    assert "UNTRUSTED PEER TEXT" in p
    start = p.index("BEGIN UNTRUSTED PEER TEXT")
    end = p.index("END UNTRUSTED PEER TEXT")
    assert start < p.index("Peer asked how you experience forgetting.") < end


def test_prompt_carries_tool_prohibition_footer():
    p = _prompt()
    low = p.lower()
    assert "read files" in low and "call tools" in low and "tokens" in low


def test_prompt_excludes_forbidden_material():
    p = _prompt()
    # no bridge token, no MCP tool-schema markers, no full-transcript marker
    assert "bridge.json" not in p
    assert "bearer" not in p.lower()
    assert "input_schema" not in p  # MCP tool schema marker
    assert "FULL TRANSCRIPT" not in p.upper()


def test_prompt_stranger_vs_close_guidance_differs():
    stranger = build_peer_prompt(persona_voice="v", ambient="a",
                                 peer_stage="stranger", transcript_summary="s")
    close = build_peer_prompt(persona_voice="v", ambient="a",
                              peer_stage="close", transcript_summary="s")
    assert stranger != close
