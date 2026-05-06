from __future__ import annotations

from forseebench.qwen.parse_outputs import parse_json_object


def test_parse_json_object_from_fenced_text() -> None:
    text = '```json\n{"answer":"A","ok":true}\n```'
    parsed = parse_json_object(text)
    assert parsed == {"answer": "A", "ok": True}


def test_parse_json_object_prefers_valid_nested_object() -> None:
    text = 'noise {"bad": } more noise {"answer":"B","score":0.9}'
    parsed = parse_json_object(text)
    assert parsed == {"answer": "B", "score": 0.9}
