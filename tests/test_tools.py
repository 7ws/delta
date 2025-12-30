"""Tests for delta.tools module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestToolPermissionHandler:
    """Tests for ToolPermissionHandler class."""

    @pytest.fixture
    def mock_conn(self):
        """Create a mock ACP connection."""
        conn = MagicMock()
        conn.session_update = AsyncMock()
        conn.request_permission = AsyncMock()
        return conn

    @pytest.fixture
    def mock_state(self):
        """Create a mock ComplianceState."""
        state = MagicMock()
        state.record_tool_call = MagicMock()
        state.has_write_operations = False
        return state

    @pytest.mark.asyncio
    async def test_builds_tool_display_for_write(self, mock_conn, mock_state):
        """Should build correct display for Write tool."""
        from delta.tools import ToolPermissionHandler

        handler = ToolPermissionHandler(
            conn=mock_conn,
            state=mock_state,
            session_id="test-session",
        )

        content, locations, kind, title = handler._build_tool_display(
            "Write",
            {"file_path": "/path/to/file.py", "content": "new content"},
            "Write file: /path/to/file.py",
        )

        assert kind == "edit"
        assert "Write" in title
        assert locations is not None

    @pytest.mark.asyncio
    async def test_builds_tool_display_for_bash(self, mock_conn, mock_state):
        """Should build correct display for Bash tool."""
        from delta.tools import ToolPermissionHandler

        handler = ToolPermissionHandler(
            conn=mock_conn,
            state=mock_state,
            session_id="test-session",
        )

        content, locations, kind, title = handler._build_tool_display(
            "Bash",
            {"command": "git status"},
            "Execute shell command: git status",
        )

        assert kind == "execute"
        assert "git status" in title

    @pytest.mark.asyncio
    async def test_builds_tool_display_for_read(self, mock_conn, mock_state):
        """Should build correct display for Read tool."""
        from delta.tools import ToolPermissionHandler

        handler = ToolPermissionHandler(
            conn=mock_conn,
            state=mock_state,
            session_id="test-session",
        )

        content, locations, kind, title = handler._build_tool_display(
            "Read",
            {"file_path": "/path/to/file.py"},
            "Read file: /path/to/file.py",
        )

        assert kind == "read"
        assert locations is not None

    @pytest.mark.asyncio
    async def test_builds_tool_display_for_search(self, mock_conn, mock_state):
        """Should build correct display for Grep tool."""
        from delta.tools import ToolPermissionHandler

        handler = ToolPermissionHandler(
            conn=mock_conn,
            state=mock_state,
            session_id="test-session",
        )

        content, locations, kind, title = handler._build_tool_display(
            "Grep",
            {"pattern": "TODO"},
            "Search content matching: TODO",
        )

        assert kind == "search"
        assert "TODO" in title
