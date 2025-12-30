"""Tests for delta.utils module."""

import pytest

from delta.utils import extract_json, parse_json


class TestExtractJson:
    """Tests for extract_json function."""

    def test_extracts_from_markdown_code_block(self):
        """Should extract JSON from markdown code block."""
        response = """Here is the result:
```json
{"key": "value", "number": 42}
```
That's the output."""
        result = extract_json(response)
        assert result == '{"key": "value", "number": 42}'

    def test_extracts_raw_json_object(self):
        """Should extract raw JSON object when no code block present."""
        response = 'Some text {"key": "value"} more text'
        result = extract_json(response)
        assert result == '{"key": "value"}'

    def test_extracts_raw_json_array(self):
        """Should extract raw JSON array when no code block or object present."""
        response = 'The tasks are: ["task 1", "task 2", "task 3"]'
        result = extract_json(response)
        assert result == '["task 1", "task 2", "task 3"]'

    def test_raises_on_no_json(self):
        """Should raise ValueError when no JSON found."""
        response = "This is plain text with no JSON"
        with pytest.raises(ValueError, match="No JSON found"):
            extract_json(response)

    def test_prefers_code_block_over_raw(self):
        """Should prefer markdown code block over raw JSON."""
        response = '{"outer": true}\n```json\n{"inner": true}\n```'
        result = extract_json(response)
        assert result == '{"inner": true}'


class TestParseJson:
    """Tests for parse_json function."""

    def test_parses_json_from_code_block(self):
        """Should parse JSON from markdown code block."""
        response = '```json\n{"status": "ok"}\n```'
        result = parse_json(response)
        assert result == {"status": "ok"}

    def test_parses_array(self):
        """Should parse JSON array."""
        response = '```json\n["a", "b", "c"]\n```'
        result = parse_json(response)
        assert result == ["a", "b", "c"]

    def test_raises_on_invalid_json(self):
        """Should raise ValueError on invalid JSON."""
        response = '```json\n{invalid json}\n```'
        with pytest.raises(ValueError):
            parse_json(response)
