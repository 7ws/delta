"""Tests for delta.protocol module."""

from unittest.mock import MagicMock, patch

from delta.protocol import (
    compute_edit_result,
    format_tool_action,
    is_working_tree_clean,
    read_file_content,
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

    def test_formats_bash_git_status(self):
        """Should format git status command contextually."""
        result = format_tool_action("Bash", {"command": "git status"})
        assert result == "Checking git status"

    def test_formats_bash_git_commit(self):
        """Should format git commit command contextually."""
        result = format_tool_action("Bash", {"command": "git commit -m 'test'"})
        assert result == "Committing changes"

    def test_formats_bash_git_add(self):
        """Should format git add command contextually."""
        result = format_tool_action("Bash", {"command": "git add file.py"})
        assert result == "Staging changes"

    def test_formats_bash_git_push(self):
        """Should format git push command contextually."""
        result = format_tool_action("Bash", {"command": "git push origin main"})
        assert result == "Pushing to remote"

    def test_formats_bash_pytest(self):
        """Should format pytest command contextually."""
        result = format_tool_action("Bash", {"command": "pytest tests/"})
        assert result == "Running tests"

    def test_formats_bash_make(self):
        """Should format make command contextually."""
        result = format_tool_action("Bash", {"command": "make lint"})
        assert result == "Running make: lint"

    def test_formats_bash_generic(self):
        """Should format generic bash command with truncation."""
        result = format_tool_action("Bash", {"command": "ls -la"})
        assert result == "Executing: ls -la"

    def test_formats_bash_long_command(self):
        """Should truncate long commands."""
        long_cmd = "find / -name '*.py' -exec grep -l 'pattern' {} \\; -print -verbose"
        result = format_tool_action("Bash", {"command": long_cmd})
        assert len(result) <= 65  # "Executing: " prefix (11) + 47 truncated + "..." (3)
        assert "..." in result

    def test_formats_write_file(self):
        """Should format Write tool with filename only."""
        result = format_tool_action("Write", {"file_path": "/path/to/file.py"})
        assert result == "Writing file.py"

    def test_formats_edit_file(self):
        """Should format Edit tool with filename only."""
        result = format_tool_action("Edit", {"file_path": "/path/to/file.py"})
        assert result == "Editing file.py"

    def test_formats_read_file(self):
        """Should format Read tool with filename only."""
        result = format_tool_action("Read", {"file_path": "/path/to/file.py"})
        assert result == "Reading file.py"

    def test_formats_glob_search(self):
        """Should format Glob tool call."""
        result = format_tool_action("Glob", {"pattern": "**/*.py"})
        assert result == "Finding files: **/*.py"

    def test_formats_grep_search(self):
        """Should format Grep tool call."""
        result = format_tool_action("Grep", {"pattern": "TODO"})
        assert result == "Searching for: TODO"

    def test_formats_web_fetch(self):
        """Should format WebFetch tool call with truncation."""
        result = format_tool_action("WebFetch", {"url": "https://example.com/path"})
        assert "Fetching:" in result
        assert "example.com" in result

    def test_formats_task_tool(self):
        """Should format Task tool with description."""
        result = format_tool_action("Task", {"description": "Search for patterns"})
        assert result == "Delegating: Search for patterns"

    def test_formats_unknown_tool(self):
        """Should format unknown tool with generic format."""
        result = format_tool_action("CustomTool", {"arg1": "val1"})
        assert "CustomTool" in result
        assert "arg1" in result

    def test_handles_mcp_prefixed_tools(self):
        """Should handle MCP-prefixed tool names."""
        result = format_tool_action("mcp__acp__Bash", {"command": "git diff"})
        assert result == "Viewing changes"

    def test_formats_bash_empty_command(self):
        """Should handle empty command gracefully."""
        result = format_tool_action("Bash", {"command": ""})
        assert result == "Execute command"

    def test_formats_npm_command(self):
        """Should format npm command contextually."""
        result = format_tool_action("Bash", {"command": "npm install"})
        assert result == "Running npm: install"

    def test_formats_uv_command(self):
        """Should format uv command contextually."""
        result = format_tool_action("Bash", {"command": "uv sync"})
        assert result == "Installing packages"


class TestIsWorkingTreeClean:
    """Tests for is_working_tree_clean function."""

    def test_given_clean_tree_when_check_then_returns_true(self):
        """Given empty stdout, When is_working_tree_clean called, Then returns True."""
        # Given
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        # When
        with patch("subprocess.run", return_value=mock_result):
            result = is_working_tree_clean()

        # Then
        assert result is True

    def test_given_dirty_tree_when_check_then_returns_false(self):
        """Given file changes in stdout, When is_working_tree_clean called, Then returns False."""
        # Given
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "M  src/delta/workflow.py\n"

        # When
        with patch("subprocess.run", return_value=mock_result):
            result = is_working_tree_clean()

        # Then
        assert result is False

    def test_given_git_unavailable_when_check_then_returns_false(self):
        """Given FileNotFoundError, When is_working_tree_clean called, Then returns False."""
        # Given/When
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = is_working_tree_clean()

        # Then
        assert result is False

    def test_given_git_command_fails_when_check_then_returns_false(self):
        """Given non-zero exit code, When is_working_tree_clean called, Then returns False."""
        # Given
        mock_result = MagicMock()
        mock_result.returncode = 128

        # When
        with patch("subprocess.run", return_value=mock_result):
            result = is_working_tree_clean()

        # Then
        assert result is False

    def test_given_git_timeout_when_check_then_returns_false(self):
        """Given git command times out, When is_working_tree_clean called, Then returns False."""
        # Given
        import subprocess

        # When
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 5)):
            result = is_working_tree_clean()

        # Then
        assert result is False
