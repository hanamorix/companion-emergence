from brain.kindled_link.codec import canonical_json


def test_sorts_keys_and_strips_whitespace() -> None:
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_utf8_not_ascii_escaped() -> None:
    # ensure_ascii=False → real UTF-8 bytes, not \uXXXX
    assert canonical_json({"x": "café"}) == '{"x":"café"}'.encode()


def test_nested_and_null() -> None:
    out = canonical_json({"z": None, "a": {"d": 1, "c": 2}})
    assert out == b'{"a":{"c":2,"d":1},"z":null}'
