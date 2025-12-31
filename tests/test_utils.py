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

    def test_repairs_invalid_json(self):
        """Should repair invalid JSON using json_repair fallback."""
        response = '```json\n{invalid json}\n```'
        result = parse_json(response)
        # json_repair repairs this to some valid JSON structure
        assert isinstance(result, (dict, list))

    def test_repairs_truncated_json_with_unterminated_string(self):
        """Should repair truncated JSON with unterminated string using json_repair."""
        # Simulates LLM output truncated mid-string
        response = '```json\n{"key": "value", "desc": "truncated string\n```'
        result = parse_json(response)
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert "truncated" in result["desc"]

    def test_repairs_truncated_json_missing_closing_brace(self):
        """Should repair truncated JSON missing closing brace."""
        response = '```json\n{"sections": [{"number": 1, "name": "Test"}\n```'
        result = parse_json(response)
        assert isinstance(result, dict)
        assert "sections" in result

    def test_repairs_truncated_nested_json(self):
        """Should repair truncated nested JSON structure."""
        # Include closing brace so extract_json can find it, but with truncated content
        response = '{"outer": {"inner": "value", "list": [1, 2, 3}'
        result = parse_json(response)
        assert isinstance(result, dict)
        assert "outer" in result
