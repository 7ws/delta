"""Tool permission handling for Delta."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from acp import start_tool_call, update_tool_call
from acp.schema import PermissionOption, ToolCallLocation, ToolCallUpdate
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny, ToolPermissionContext

from delta.llm import InvalidWriteClassificationResponse, classify_write_operation, get_classify_client
from delta.protocol import compute_edit_result, format_tool_action, read_file_content
from acp import tool_diff_content

if TYPE_CHECKING:
    from acp.interfaces import Client

    from delta.acp_server import ComplianceState

logger = logging.getLogger(__name__)


ToolKind = Literal[
    "read",
    "edit",
    "delete",
    "move",
    "search",
    "execute",
    "think",
    "fetch",
    "switch_mode",
    "other",
]


class ToolPermissionHandler:
    """Handles tool permission requests for the inner agent.

    Extracts the complex tool permission logic from the DeltaAgent class
    into a focused, testable component.
    """

    def __init__(
        self,
        conn: Client,
        state: ComplianceState,
        session_id: str,
        classify_model: str = "haiku",
    ) -> None:
        """Initialize the tool permission handler.

        Args:
            conn: ACP client connection.
            state: Compliance state for the session.
            session_id: ACP session identifier.
            classify_model: Model for write operation classification.
        """
        self._conn = conn
        self._state = state
        self._session_id = session_id
        self._classify_model = classify_model

    async def handle(
        self,
        tool_name: str,
        input_params: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Request user permission for tool execution.

        Tools proceed with user permission only. Compliance review happens
        at batch level (after work is ready), not per-action.

        Args:
            tool_name: Name of the tool being called.
            input_params: Parameters for the tool.
            context: Tool permission context from SDK.

        Returns:
            PermissionResultAllow or PermissionResultDeny.
        """
        tool_description = format_tool_action(tool_name, input_params)
        logger.info(f"Tool call: {tool_name} - {tool_description}")

        tool_call_id = f"tool_{uuid4().hex[:8]}"

        # Build rich content for file operations
        tool_content, tool_locations, tool_kind, tool_title = self._build_tool_display(
            tool_name, input_params, tool_description
        )

        # Send ToolCallStart to display the tool call in UI with diff
        tool_start = start_tool_call(
            tool_call_id=tool_call_id,
            title=tool_title,
            kind=tool_kind,
            status="pending",
            content=tool_content,
            locations=tool_locations,
            raw_input=input_params,
        )
        await self._conn.session_update(session_id=self._session_id, update=tool_start)

        # Build ToolCallUpdate for permission request
        tool_call = ToolCallUpdate(
            tool_call_id=tool_call_id,
            title=tool_title,
            kind=tool_kind,
            content=tool_content,
            locations=tool_locations,
            raw_input=input_params,
        )

        options = [
            PermissionOption(option_id="allow_once", name="Allow", kind="allow_once"),
            PermissionOption(option_id="allow_always", name="Always Allow", kind="allow_always"),
            PermissionOption(option_id="reject_once", name="Reject", kind="reject_once"),
            PermissionOption(option_id="reject_always", name="Never Allow", kind="reject_always"),
        ]

        logger.debug(f"Requesting user permission: {tool_title}")
        response = await self._conn.request_permission(
            options=options,
            session_id=self._session_id,
            tool_call=tool_call,
        )

        return await self._handle_permission_response(
            response,
            tool_call_id,
            tool_name,
            tool_title,
            tool_description,
            input_params,
        )

    def _build_tool_display(
        self,
        tool_name: str,
        input_params: dict[str, Any],
        tool_description: str,
    ) -> tuple[list[Any] | None, list[ToolCallLocation] | None, ToolKind | None, str]:
        """Build display content for a tool call.

        Returns:
            Tuple of (content, locations, kind, title).
        """
        tool_content: list[Any] | None = None
        tool_locations: list[ToolCallLocation] | None = None
        tool_kind: ToolKind | None = None
        tool_title = tool_description

        if tool_name in ("Write", "mcp__acp__Write"):
            file_path = input_params.get("file_path", "")
            new_text = input_params.get("content", "")
            old_text = read_file_content(file_path)
            tool_content = [tool_diff_content(file_path, new_text, old_text)]
            tool_locations = [ToolCallLocation(path=file_path)]
            tool_kind = "edit"
            tool_title = f"Write {file_path}"

        elif tool_name in ("Edit", "mcp__acp__Edit"):
            file_path = input_params.get("file_path", "")
            old_string = input_params.get("old_string", "")
            new_string = input_params.get("new_string", "")
            replace_all = input_params.get("replace_all", False)
            old_text = read_file_content(file_path)
            new_text = compute_edit_result(old_text, old_string, new_string, replace_all)
            if new_text is not None:
                tool_content = [tool_diff_content(file_path, new_text, old_text)]
            tool_locations = [ToolCallLocation(path=file_path)]
            tool_kind = "edit"
            tool_title = f"Edit {file_path}"

        elif tool_name in ("Read", "mcp__acp__Read"):
            file_path = input_params.get("file_path", "")
            tool_locations = [ToolCallLocation(path=file_path)]
            tool_kind = "read"
            tool_title = f"Read {file_path}"

        elif tool_name in ("Bash", "mcp__acp__Bash"):
            command = input_params.get("command", "")
            tool_kind = "execute"
            tool_title = f"Run {command}"

        elif tool_name in ("Grep", "Glob"):
            pattern = input_params.get("pattern", "")
            tool_kind = "search"
            tool_title = f"Search {pattern}"

        elif tool_name == "WebFetch":
            url = input_params.get("url", "")
            tool_kind = "fetch"
            tool_title = f"Fetch {url}"

        return tool_content, tool_locations, tool_kind, tool_title

    async def _handle_permission_response(
        self,
        response: Any,
        tool_call_id: str,
        tool_name: str,
        tool_title: str,
        tool_description: str,
        input_params: dict[str, Any],
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Handle the user's permission response.

        Args:
            response: Permission response from ACP.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            tool_title: Display title for the tool.
            tool_description: Human-readable description.
            input_params: Original input parameters.

        Returns:
            PermissionResultAllow or PermissionResultDeny.
        """
        outcome_type = response.outcome.outcome
        logger.info(f"User permission response: {outcome_type}")

        if outcome_type == "selected":
            selected_id = response.outcome.option_id
            logger.debug(f"User selected option: {selected_id}")

            if selected_id in ("allow_once", "allow_always"):
                return await self._handle_allow(
                    tool_call_id, tool_name, tool_description, input_params
                )

            # User rejected
            return await self._handle_reject(
                tool_call_id, tool_title, tool_description, selected_id
            )

        # Cancelled
        return await self._handle_cancel(tool_call_id, tool_title, tool_description)

    async def _handle_allow(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_description: str,
        input_params: dict[str, Any],
    ) -> PermissionResultAllow:
        """Handle allowed tool permission."""
        tool_progress = update_tool_call(tool_call_id=tool_call_id, status="in_progress")
        await self._conn.session_update(session_id=self._session_id, update=tool_progress)

        self._state.record_tool_call(tool_description, allowed=True)

        # Track write operations using AI classification
        await self._track_write_operation(tool_name, tool_description)

        return PermissionResultAllow(updated_input=input_params)

    async def _handle_reject(
        self,
        tool_call_id: str,
        tool_title: str,
        tool_description: str,
        selected_id: str,
    ) -> PermissionResultDeny:
        """Handle rejected tool permission."""
        logger.warning(f"User rejected: {tool_title} ({selected_id})")

        tool_progress = update_tool_call(tool_call_id=tool_call_id, status="failed")
        await self._conn.session_update(session_id=self._session_id, update=tool_progress)

        self._state.record_tool_call(tool_description, allowed=False)

        return PermissionResultDeny(
            message="User rejected this action. Stop and ask for clarification before proceeding.",
            interrupt=True,
        )

    async def _handle_cancel(
        self,
        tool_call_id: str,
        tool_title: str,
        tool_description: str,
    ) -> PermissionResultDeny:
        """Handle cancelled permission prompt."""
        logger.warning(f"Permission prompt cancelled: {tool_title}")

        tool_progress = update_tool_call(tool_call_id=tool_call_id, status="failed")
        await self._conn.session_update(session_id=self._session_id, update=tool_progress)

        self._state.record_tool_call(tool_description, allowed=False)

        return PermissionResultDeny(
            message="Permission prompt was cancelled. Stop and ask for clarification before proceeding.",
            interrupt=True,
        )

    async def _track_write_operation(self, tool_name: str, tool_description: str) -> None:
        """Track whether a tool operation is a write operation.

        Uses AI classification per architectural decision.
        """
        try:
            classify_client = get_classify_client(self._classify_model)
            is_write = await asyncio.to_thread(
                classify_write_operation,
                classify_client,
                tool_name,
                tool_description,
            )
            if is_write:
                self._state.has_write_operations = True
                logger.info(f"Write operation tracked: {tool_description}")
        except InvalidWriteClassificationResponse:
            # Default to write operation if classification fails (safe side)
            self._state.has_write_operations = True
            logger.warning(f"Write classification failed, treating as write: {tool_description}")
