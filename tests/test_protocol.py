"""Tests for delta.protocol module."""

import pytest
from pathlib import Path
from unittest.mock import patch

from delta.protocol import (
    compute_edit_result,
    format_tool_action,
    read_file_content,
    read_resource_link,
)


class TestReadFileContent:
    """Tests for read_file_content function."""

    def test_returns_none_for_nonexistent_file(self):
        """Should return None if file does not exist."""
        result = read_file_content("/nonexistent/path/file.txt")
        assert result is None

    def test_reads_existing_file(self, tmp_path):
        """Should read content from existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = read_file_content(str(test_file))
        assert result == "Hello, World!"


class TestComputeEditResult:
    """Tests for compute_edit_result function."""

    def test_returns_none_when_old_content_is_none(self):
        """Should return None if old_content is None."""
        result = compute_edit_result(None, "old", "new")
        assert result is None

    def test_returns_none_when_old_string_not_found(self):
        """Should return None if old_string not in content."""
        result = compute_edit_result("some content", "not found", "new")
        assert result is None

    def test_replaces_single_occurrence_by_default(self):
        """Should replace only first occurrence by default."""
        result = compute_edit_result("a b a b", "a", "X")
        assert result == "X b a b"

    def test_replaces_all_occurrences_when_specified(self):
        """Should replace all occurrences when replace_all=True."""
        result = compute_edit_result("a b a b", "a", "X", replace_all=True)
        assert result == "X b X b"


class TestFormatToolAction:
    """Tests for format_tool_action function."""

    def test_formats_bash_command(self):
        """Should format Bash tool call."""
        result = format_tool_action("Bash", {"command": "git status"})
        assert result == "Execute shell command: git status"

    def test_formats_write_file(self):
        """Should format Write tool call."""
        result = format_tool_action("Write", {"file_path": "/path/to/file.py"})
        assert result == "Write file: /path/to/file.py"

    def test_formats_edit_file(self):
        """Should format Edit tool call."""
        result = format_tool_action("Edit", {"file_path": "/path/to/file.py"})
        assert result == "Edit file: /path/to/file.py"

    def test_formats_read_file(self):
        """Should format Read tool call."""
        result = format_tool_action("Read", {"file_path": "/path/to/file.py"})
        assert result == "Read file: /path/to/file.py"

    def test_formats_glob_search(self):
        """Should format Glob tool call."""
        result = format_tool_action("Glob", {"pattern": "**/*.py"})
        assert result == "Search files matching: **/*.py"

    def test_formats_grep_search(self):
        """Should format Grep tool call."""
        result = format_tool_action("Grep", {"pattern": "TODO"})
        assert result == "Search content matching: TODO"

    def test_formats_web_fetch(self):
        """Should format WebFetch tool call."""
        result = format_tool_action("WebFetch", {"url": "https://example.com"})
        assert result == "Fetch URL: https://example.com"

    def test_formats_unknown_tool(self):
        """Should format unknown tool with generic format."""
        result = format_tool_action("CustomTool", {"arg1": "val1"})
        assert "CustomTool" in result
        assert "arg1" in result

    def test_handles_mcp_prefixed_tools(self):
        """Should handle MCP-prefixed tool names."""
        result = format_tool_action("mcp__acp__Bash", {"command": "ls"})
        assert result == "Execute shell command: ls"
