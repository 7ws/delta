"""ACP server implementation for Delta compliance wrapper.

Delta acts as a proxy between the editor and an inner AI agent, intercepting
all actions and performing compliance reviews before allowing execution.

Architecture: Editor <-> Delta (ACP) <-> Inner Agent (Claude Agent SDK)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from acp import (
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    plan_entry,
    run_agent,
    start_tool_call,
    text_block,
    tool_diff_content,
    update_agent_message,
    update_plan,
    update_tool_call,
)
from acp.interfaces import Agent, Client
from acp.schema import (
    AgentCapabilities,
    AudioContentBlock,
    AuthenticateResponse,
    BlobResourceContents,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    PermissionOption,
    PromptCapabilities,
    ResourceContentBlock,
    SetSessionModelResponse,
    SetSessionModeResponse,
    SseMcpServer,
    TextContentBlock,
    TextResourceContents,
    ToolCallLocation,
    ToolCallUpdate,
)
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookContext,
    HookJSONOutput,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    PostToolUseHookInput,
    PreCompactHookInput,
    PreToolUseHookInput,
    StopHookInput,
    SubagentStopHookInput,
    TextBlock,
    ToolPermissionContext,
    ToolUseBlock,
    UserPromptSubmitHookInput,
)

from delta.compliance import (
    ComplianceReport,
    build_batch_work_review_prompt,
    build_plan_review_prompt,
    parse_compliance_response,
)
from delta.guidelines import (
    AgentsDocument,
    find_agents_md,
    get_bundled_agents_md,
    parse_agents_md,
)
from delta.llm import (
    ClaudeCodeClient,
    InvalidPlanParseResponse,
    InvalidReviewReadinessResponse,
    InvalidTriageResponse,
    detect_task_progress,
    generate_clarifying_questions,
    get_classify_client,
    get_llm_client,
    is_ready_for_review,
    parse_plan_tasks,
    triage_user_message,
)

logger = logging.getLogger(__name__)


def _read_file_content(file_path: str) -> str | None:
    """Read current file content, or None if file doesn't exist."""
    try:
        path = Path(file_path)
        if path.exists():
            return path.read_text()
        return None
    except (OSError, UnicodeDecodeError) as e:
        logger.warning(f"Failed to read file {file_path}: {e}")
        return None


def _compute_edit_result(
    old_content: str | None,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str | None:
    """Compute the result of an Edit operation.

    Returns the new file content after applying the replacement,
    or None if the old_string was not found.
    """
    if old_content is None:
        return None

    if replace_all:
        if old_string not in old_content:
            return None
        return old_content.replace(old_string, new_string)
    else:
        if old_string not in old_content:
            return None
        return old_content.replace(old_string, new_string, 1)


PromptBlock = (
    TextContentBlock
    | ImageContentBlock
    | AudioContentBlock
    | ResourceContentBlock
    | EmbeddedResourceContentBlock
)


def _extract_prompt_content(
    blocks: list[PromptBlock], cwd: Path | None = None
) -> list[dict[str, Any]]:
    """Extract content blocks from prompt for Claude API.

    Converts ACP prompt blocks to Claude API content format, supporting
    text, images, and embedded resources.

    Returns:
        List of Claude API content blocks (text and image types).
    """
    content: list[dict[str, Any]] = []
    text_parts: list[str] = []

    def flush_text() -> None:
        """Add accumulated text as a content block."""
        if text_parts:
            content.append({"type": "text", "text": "\n\n".join(text_parts)})
            text_parts.clear()

    for block in blocks:
        if isinstance(block, dict):
            # Handle dict-style blocks
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "image":
                flush_text()
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": block.get(
                                "mimeType", block.get("mime_type", "image/png")
                            ),
                            "data": block.get("data", ""),
                        },
                    }
                )
            elif block_type == "resource":
                resource = block.get("resource", {})
                uri = resource.get("uri", "")
                text = resource.get("text")
                if text:
                    text_parts.append(f'<file uri="{uri}">\n{text}\n</file>')
            elif block_type == "resource_link":
                uri = block.get("uri", "")
                name = block.get("name", uri)
                file_content = _read_resource_link(uri, cwd)
                if file_content:
                    text_parts.append(f'<file uri="{uri}" name="{name}">\n{file_content}\n</file>')
                else:
                    text_parts.append(f"[Referenced file: {name} ({uri})]")
        elif isinstance(block, ImageContentBlock):
            flush_text()
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type,
                        "data": block.data,
                    },
                }
            )
        elif isinstance(block, TextContentBlock):
            text_parts.append(block.text)
        elif isinstance(block, EmbeddedResourceContentBlock):
            resource = block.resource
            if isinstance(resource, TextResourceContents):
                text_parts.append(f'<file uri="{resource.uri}">\n{resource.text}\n</file>')
            elif isinstance(resource, BlobResourceContents):
                text_parts.append(f"[Binary file: {resource.uri}]")
        elif isinstance(block, ResourceContentBlock):
            file_content = _read_resource_link(block.uri, cwd)
            if file_content:
                text_parts.append(
                    f'<file uri="{block.uri}" name="{block.name}">\n{file_content}\n</file>'
                )
            else:
                text_parts.append(f"[Referenced file: {block.name} ({block.uri})]")
        elif hasattr(block, "text"):
            text_parts.append(str(getattr(block, "text", "")))

    flush_text()
    return content


def _extract_prompt_text(blocks: list[PromptBlock], cwd: Path | None = None) -> str:
    """Extract text content from prompt blocks.

    Handles text blocks, embedded resources, and resource links.
    Returns only the text portion, ignoring images.
    """
    content = _extract_prompt_content(blocks, cwd)
    text_parts = [block["text"] for block in content if block.get("type") == "text"]
    return "\n\n".join(text_parts)


def _read_resource_link(uri: str, cwd: Path | None = None) -> str | None:
    """Read content from a resource link URI.

    Supports file:// URIs and relative paths.
    """
    if uri.startswith("file://"):
        file_path = Path(uri[7:])
    elif not uri.startswith(("http://", "https://")):
        file_path = Path(uri)
        if cwd and not file_path.is_absolute():
            file_path = cwd / file_path
    else:
        return None

    try:
        return file_path.read_text()
    except (OSError, UnicodeDecodeError):
        logger.warning(f"Failed to read resource: {uri}")
        return None


class WorkflowPhase(Enum):
    """Current phase of the workflow."""

    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETE = "complete"


@dataclass
class PlanTask:
    """A single task from the approved plan."""

    content: str
    status: Literal["pending", "in_progress", "completed"] = "pending"


@dataclass
class ComplianceState:
    """Track compliance state for a session.

    Workflow phases:
    1. PLANNING: Inner agent creates a plan, reviewed up to 3 times
    2. EXECUTING: Inner agent implements the plan, tools proceed with user permission
    3. REVIEWING: Haiku detected readiness, Sonnet reviews the batch
    4. COMPLETE: All sections scored 5/5

    Plan review has a hard limit of 3 attempts before escalation.
    Execution review has unlimited attempts (the agent keeps revising).
    """

    # Workflow state
    phase: WorkflowPhase = WorkflowPhase.PLANNING
    approved_plan: str = ""
    work_summary: str = ""
    plan_review_attempts: int = 0
    max_plan_attempts: int = 5
    plan_tasks: list[PlanTask] = field(default_factory=list)

    # Session state
    agents_doc: AgentsDocument | None = None
    inner_client: ClaudeSDKClient | None = None
    cwd: Path | None = None
    current_user_prompt: str = ""
    user_request_history: list[str] = field(default_factory=list)
    tool_call_history: list[str] = field(default_factory=list)

    def record_tool_call(self, tool_description: str, allowed: bool) -> None:
        """Record a tool call in the session history."""
        status = "ALLOWED" if allowed else "DENIED"
        self.tool_call_history.append(f"[{status}] {tool_description}")

    def reset_for_new_prompt(self) -> None:
        """Reset state for a new user prompt.

        Resets workflow state while preserving tool call history
        for compliance reviewer context.
        """
        self.phase = WorkflowPhase.PLANNING
        self.approved_plan = ""
        self.work_summary = ""
        self.plan_review_attempts = 0
        self.plan_tasks = []


class DeltaAgent(Agent):
    """ACP agent that wraps another agent with compliance enforcement."""

    _conn: Client

    def __init__(
        self,
        agents_md_path: Path | None = None,
        review_model: str | None = None,
        classify_model: str = "haiku",
    ) -> None:
        """Initialize Delta agent."""
        self.sessions: dict[str, ComplianceState] = {}
        self.llm_client: ClaudeCodeClient = get_llm_client(model=review_model)
        self.classify_model = classify_model

        self._explicit_agents_md_path = agents_md_path
        self.agents_md_path: Path | None = None
        self.agents_doc: AgentsDocument | None = None

    def on_connect(self, conn: Client) -> None:
        """Handle connection from editor."""
        self._conn = conn

    def _get_state(self, session_id: str) -> ComplianceState:
        """Get or create compliance state for session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ComplianceState()
        return self.sessions[session_id]

    def _format_tool_action(self, tool_name: str, input_params: dict[str, Any]) -> str:
        """Format a tool call as a human-readable action description.

        Args:
            tool_name: Name of the tool being called.
            input_params: Parameters passed to the tool.

        Returns:
            Action description in imperative form for compliance review.
        """
        if tool_name in ("Bash", "mcp__acp__Bash"):
            command = input_params.get("command", "")
            return f"Execute shell command: {command}"
        elif tool_name in ("Write", "mcp__acp__Write"):
            file_path = input_params.get("file_path", "")
            return f"Write file: {file_path}"
        elif tool_name in ("Edit", "mcp__acp__Edit"):
            file_path = input_params.get("file_path", "")
            return f"Edit file: {file_path}"
        elif tool_name in ("Read", "mcp__acp__Read"):
            file_path = input_params.get("file_path", "")
            return f"Read file: {file_path}"
        elif tool_name == "Glob":
            pattern = input_params.get("pattern", "")
            return f"Search files matching: {pattern}"
        elif tool_name == "Grep":
            pattern = input_params.get("pattern", "")
            return f"Search content matching: {pattern}"
        elif tool_name == "WebFetch":
            url = input_params.get("url", "")
            return f"Fetch URL: {url}"
        else:
            # Generic format for unknown tools
            params_str = ", ".join(f"{k}={v!r}" for k, v in input_params.items())
            return f"Call {tool_name}({params_str})"

    def _load_agents_md(self, cwd: Path | None = None) -> None:
        """Load or reload AGENTS.md.

        Searches for AGENTS.md in the following order:
        1. Explicit path provided via constructor
        2. Target codebase (searching upward from cwd)
        3. Bundled AGENTS.md shipped with Delta (fallback)
        """
        if self.agents_md_path is None:
            if self._explicit_agents_md_path is not None:
                self.agents_md_path = self._explicit_agents_md_path
            else:
                self.agents_md_path = find_agents_md(cwd)
                if self.agents_md_path is None:
                    # Fall back to bundled AGENTS.md
                    self.agents_md_path = get_bundled_agents_md()
                    logger.info(
                        f"No AGENTS.md in target codebase, using bundled: {self.agents_md_path}"
                    )
        self.agents_doc = parse_agents_md(self.agents_md_path)

    async def _review_plan(
        self,
        state: ComplianceState,
        plan: str,
        session_id: str,
    ) -> ComplianceReport:
        """Review a plan for compliance before execution.

        Args:
            state: Compliance state for the session.
            plan: The proposed implementation plan.
            session_id: ACP session identifier.

        Returns:
            ComplianceReport with scores and revision guidance.
        """
        self._load_agents_md()

        if self.agents_doc is None:
            raise RuntimeError("Failed to load AGENTS.md")

        prompt = build_plan_review_prompt(
            self.agents_doc,
            state.current_user_prompt,
            plan,
        )

        # Retry up to 3 times on JSON parse failures
        max_parse_retries = 3
        last_error: Exception | None = None

        for parse_attempt in range(max_parse_retries):
            response = await asyncio.to_thread(
                self.llm_client.complete,
                prompt=prompt,
                system="You are a strict compliance reviewer. Output only valid JSON.",
            )

            try:
                report = parse_compliance_response(response, self.agents_doc)
                state.plan_review_attempts += 1

                logger.info(
                    f"Plan review attempt {state.plan_review_attempts}: "
                    f"compliant={report.is_compliant}"
                )

                return report
            except (ValueError, KeyError) as e:
                last_error = e
                logger.warning(
                    f"JSON parse error on attempt {parse_attempt + 1}/{max_parse_retries}: {e}"
                )
                # Modify prompt to emphasize JSON format
                prompt = (
                    f"Your previous response had invalid JSON. "
                    f"You MUST output valid JSON only.\n\n{prompt}"
                )

        # All retries failed - raise the last error
        raise RuntimeError(
            f"Failed to parse compliance response after {max_parse_retries} attempts: "
            f"{last_error}"
        )

    async def _review_work(
        self,
        state: ComplianceState,
        work_summary: str,
        session_id: str,
    ) -> ComplianceReport:
        """Review accumulated work for compliance.

        Args:
            state: Compliance state for the session.
            work_summary: Summary of work completed by the inner agent.
            session_id: ACP session identifier.

        Returns:
            ComplianceReport with scores and revision guidance.
        """
        self._load_agents_md()

        if self.agents_doc is None:
            raise RuntimeError("Failed to load AGENTS.md")

        prompt = build_batch_work_review_prompt(
            self.agents_doc,
            state.current_user_prompt,
            state.approved_plan,
            work_summary,
            state.tool_call_history,
        )

        # Retry up to 3 times on JSON parse failures
        max_parse_retries = 3
        last_error: Exception | None = None

        for parse_attempt in range(max_parse_retries):
            response = await asyncio.to_thread(
                self.llm_client.complete,
                prompt=prompt,
                system="You are a strict compliance reviewer. Output only valid JSON.",
            )

            try:
                report = parse_compliance_response(response, self.agents_doc)
                logger.info(f"Work review: compliant={report.is_compliant}")
                return report
            except (ValueError, KeyError) as e:
                last_error = e
                logger.warning(
                    f"JSON parse error on attempt {parse_attempt + 1}/{max_parse_retries}: {e}"
                )
                prompt = (
                    f"Your previous response had invalid JSON. "
                    f"You MUST output valid JSON only.\n\n{prompt}"
                )

        raise RuntimeError(
            f"Failed to parse compliance response after {max_parse_retries} attempts: "
            f"{last_error}"
        )

    async def _check_ready_for_review(
        self,
        state: ComplianceState,
        recent_output: str,
    ) -> bool:
        """Check if work is ready for compliance review using Haiku.

        Args:
            state: Compliance state for the session.
            recent_output: Recent inner agent output to evaluate.

        Returns:
            True if work is ready for review.
        """
        classify_client = get_classify_client(self.classify_model)

        try:
            ready = await asyncio.to_thread(
                is_ready_for_review,
                classify_client,
                recent_output,
            )
            logger.info(f"Review readiness check: ready={ready}")
            return ready
        except InvalidReviewReadinessResponse as e:
            logger.warning(f"Review readiness check failed: {e}")
            # Default to not ready if check fails
            return False

    async def _parse_and_send_plan(
        self,
        state: ComplianceState,
        plan_text: str,
        session_id: str,
    ) -> None:
        """Parse an approved plan into tasks and send the plan widget.

        Args:
            state: Compliance state for the session.
            plan_text: The approved plan text.
            session_id: ACP session identifier.
        """
        classify_client = get_classify_client(self.classify_model)

        try:
            task_descriptions = await asyncio.to_thread(
                parse_plan_tasks,
                classify_client,
                plan_text,
            )

            # Store tasks in state
            state.plan_tasks = [PlanTask(content=desc) for desc in task_descriptions]

            # Send plan widget to UI
            entries = [
                plan_entry(task.content, status=task.status)
                for task in state.plan_tasks
            ]
            plan_update = update_plan(entries)
            await self._conn.session_update(session_id=session_id, update=plan_update)

            logger.info(f"Sent plan widget with {len(state.plan_tasks)} tasks")

        except InvalidPlanParseResponse as e:
            logger.warning(f"Failed to parse plan into tasks: {e}")
            # Continue without plan widget if parsing fails

    async def _send_plan_update(
        self,
        state: ComplianceState,
        session_id: str,
    ) -> None:
        """Send current plan task status to UI.

        Args:
            state: Compliance state for the session.
            session_id: ACP session identifier.
        """
        if not state.plan_tasks:
            return

        entries = [
            plan_entry(task.content, status=task.status)
            for task in state.plan_tasks
        ]
        plan_update = update_plan(entries)
        await self._conn.session_update(session_id=session_id, update=plan_update)

    async def _update_task_progress(
        self,
        state: ComplianceState,
        recent_output: str,
        session_id: str,
    ) -> None:
        """Detect and update task progress based on recent output.

        Args:
            state: Compliance state for the session.
            recent_output: Recent inner agent output to analyze.
            session_id: ACP session identifier.
        """
        if not state.plan_tasks:
            return

        classify_client = get_classify_client(self.classify_model)

        try:
            task_descriptions = [task.content for task in state.plan_tasks]
            progress = await asyncio.to_thread(
                detect_task_progress,
                classify_client,
                task_descriptions,
                recent_output,
            )

            if not progress:
                return

            # Update task statuses
            for idx, new_status in progress.items():
                if 0 <= idx < len(state.plan_tasks):
                    if new_status == "pending":
                        state.plan_tasks[idx].status = "pending"
                    elif new_status == "in_progress":
                        state.plan_tasks[idx].status = "in_progress"
                    elif new_status == "completed":
                        state.plan_tasks[idx].status = "completed"

            # Send updated plan to UI
            await self._send_plan_update(state, session_id)
            logger.info(f"Updated task progress: {progress}")

        except Exception as e:
            logger.warning(f"Failed to detect task progress: {e}")

    def _build_system_prompt(self) -> str:
        """Build system prompt that includes AGENTS.md."""
        self._load_agents_md()

        if self.agents_doc is None:
            return ""

        return (
            "You MUST read and follow the guidelines in AGENTS.md for every response.\n\n"
            "CRITICAL - PROACTIVE THOROUGHNESS (guideline 2.2.8-2.2.9):\n"
            "When a change affects multiple files, update ALL affected files without "
            "asking for permission. Do NOT stop after one file and ask 'Do you want me "
            "to update the others?' - finish the job. Incomplete work is a violation.\n\n"
            f"AGENTS.md content:\n\n{self.agents_doc.raw_content}"
        )

    async def _get_inner_client(self, state: ComplianceState, session_id: str) -> ClaudeSDKClient:
        """Get or create the inner Claude SDK client for this session."""
        if state.inner_client is None:

            async def handle_tool_permission(
                tool_name: str,
                input_params: dict[str, Any],
                context: ToolPermissionContext,
            ) -> PermissionResultAllow | PermissionResultDeny:
                """Request user permission for tool execution.

                Tools proceed with user permission only. Compliance review happens
                at batch level (after work is ready), not per-action.

                Returns:
                    PermissionResultAllow or PermissionResultDeny from claude_agent_sdk.
                """
                tool_description = self._format_tool_action(tool_name, input_params)
                logger.info(f"Tool call: {tool_name} - {tool_description}")

                tool_call_id = f"tool_{uuid4().hex[:8]}"

                # Build rich content for file operations
                tool_content: list[Any] | None = None
                tool_locations: list[ToolCallLocation] | None = None
                tool_kind: (
                    Literal[
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
                    | None
                ) = None
                tool_title = tool_description  # Use human-readable description

                if tool_name in ("Write", "mcp__acp__Write"):
                    file_path = input_params.get("file_path", "")
                    new_text = input_params.get("content", "")
                    old_text = _read_file_content(file_path)
                    tool_content = [tool_diff_content(file_path, new_text, old_text)]
                    tool_locations = [ToolCallLocation(path=file_path)]
                    tool_kind = "edit"
                    tool_title = f"Write {file_path}"

                elif tool_name in ("Edit", "mcp__acp__Edit"):
                    file_path = input_params.get("file_path", "")
                    old_string = input_params.get("old_string", "")
                    new_string = input_params.get("new_string", "")
                    replace_all = input_params.get("replace_all", False)
                    old_text = _read_file_content(file_path)
                    new_text = _compute_edit_result(old_text, old_string, new_string, replace_all)
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
                await self._conn.session_update(session_id=session_id, update=tool_start)

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
                    PermissionOption(
                        option_id="allow_once",
                        name="Allow",
                        kind="allow_once",
                    ),
                    PermissionOption(
                        option_id="allow_always",
                        name="Always Allow",
                        kind="allow_always",
                    ),
                    PermissionOption(
                        option_id="reject_once",
                        name="Reject",
                        kind="reject_once",
                    ),
                    PermissionOption(
                        option_id="reject_always",
                        name="Never Allow",
                        kind="reject_always",
                    ),
                ]

                logger.debug(f"Requesting user permission: {tool_title}")
                response = await self._conn.request_permission(
                    options=options,
                    session_id=session_id,
                    tool_call=tool_call,
                )

                # Per ACP spec: outcome.outcome is "selected" (user chose an option)
                # or "cancelled" (prompt was cancelled). When "selected", check
                # outcome.option_id to see which option was chosen.
                outcome_type = response.outcome.outcome
                logger.info(f"User permission response: {outcome_type}")

                if outcome_type == "selected":
                    selected_id = response.outcome.option_id
                    logger.debug(f"User selected option: {selected_id}")

                    if selected_id in ("allow_once", "allow_always"):
                        # Update tool call status to in_progress
                        tool_progress = update_tool_call(
                            tool_call_id=tool_call_id,
                            status="in_progress",
                        )
                        await self._conn.session_update(session_id=session_id, update=tool_progress)
                        # Record allowed tool call in history for future compliance reviews
                        state.record_tool_call(tool_description, allowed=True)
                        return PermissionResultAllow(updated_input=input_params)

                    # User rejected (reject_once or reject_always) - interrupt
                    # and ask for clarification
                    logger.warning(f"User rejected: {tool_title} ({selected_id})")
                    # Update tool call status to failed
                    tool_progress = update_tool_call(
                        tool_call_id=tool_call_id,
                        status="failed",
                    )
                    await self._conn.session_update(session_id=session_id, update=tool_progress)
                    state.record_tool_call(tool_description, allowed=False)
                    return PermissionResultDeny(
                        message=(
                            "User rejected this action. "
                            "Stop and ask for clarification before proceeding."
                        ),
                        interrupt=True,
                    )

                # Cancelled - prompt was dismissed without selection
                logger.warning(f"Permission prompt cancelled: {tool_title}")
                # Update tool call status to failed
                tool_progress = update_tool_call(
                    tool_call_id=tool_call_id,
                    status="failed",
                )
                await self._conn.session_update(session_id=session_id, update=tool_progress)
                state.record_tool_call(tool_description, allowed=False)
                return PermissionResultDeny(
                    message=(
                        "Permission prompt was cancelled. "
                        "Stop and ask for clarification before proceeding."
                    ),
                    interrupt=True,
                )

            async def track_tool_call(
                hook_input: (
                    PreToolUseHookInput
                    | PostToolUseHookInput
                    | UserPromptSubmitHookInput
                    | StopHookInput
                    | SubagentStopHookInput
                    | PreCompactHookInput
                ),
                tool_use_id: str | None,
                context: HookContext,
            ) -> HookJSONOutput:
                """Track all tool calls for compliance review context.

                This hook fires for ALL tools (Read, Glob, Grep, etc.) before
                they execute. We use it to record tool usage so the compliance
                reviewer knows what actions have been taken.

                Returns empty dict to allow the tool call to proceed.
                """
                # Only process PreToolUse hooks - others lack tool_name/tool_input
                if "tool_name" not in hook_input:
                    return {}

                # Cast to PreToolUseHookInput after runtime check above
                pre_tool_input: PreToolUseHookInput = hook_input  # type: ignore[assignment]
                tool_name = pre_tool_input["tool_name"]
                tool_input = pre_tool_input["tool_input"]
                tool_description = self._format_tool_action(tool_name, tool_input)

                # Record in history (allowed=True since this hook doesn't block)
                # For tools that go through can_use_tool, the history will be
                # updated again with the actual allow/deny decision
                if tool_name in ("Read", "mcp__acp__Read", "Glob", "Grep", "Task"):
                    # Auto-approved tools - record them directly
                    state.record_tool_call(tool_description, allowed=True)
                    logger.debug(f"Auto-approved tool tracked: {tool_description}")

                # Return empty dict to allow the tool call to proceed
                return {}

            options = ClaudeAgentOptions(
                system_prompt=self._build_system_prompt(),
                cwd=str(Path.cwd()),
                can_use_tool=handle_tool_permission,
                hooks={
                    "PreToolUse": [
                        HookMatcher(
                            matcher=None,  # Match all tools
                            hooks=[track_tool_call],
                        ),
                    ],
                },
            )
            state.inner_client = ClaudeSDKClient(options=options)
            await state.inner_client.connect()
            logger.info("Inner Claude SDK client connected")
        return state.inner_client

    async def _call_inner_agent(
        self,
        state: ComplianceState,
        prompt: str | list[dict[str, Any]],
        session_id: str,
    ) -> str:
        """Call the inner agent and return its response.

        Args:
            state: Compliance state for the session.
            prompt: Either a string or list of Claude API content blocks.
            session_id: ACP session identifier.
        """
        client = await self._get_inner_client(state, session_id)

        if isinstance(prompt, str):
            await client.query(prompt)
        else:
            # For structured content (with images), create async message stream
            async def message_stream() -> AsyncIterator[dict[str, Any]]:
                yield {
                    "type": "user",
                    "message": {"role": "user", "content": prompt},
                    "parent_tool_use_id": None,
                    "session_id": "default",
                }

            await client.query(message_stream())

        response_text = ""
        tool_calls: dict[str, str] = {}  # tool_use_id -> tool_call_id for UI

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text = block.text
                        # Add blank line between text blocks if needed
                        needs_separator = (
                            response_text
                            and not response_text.endswith("\n\n")
                            and text
                            and not text.startswith("\n")
                        )
                        if needs_separator:
                            text = "\n\n" + text
                        response_text += text
                        chunk = update_agent_message(text_block(text))
                        await self._conn.session_update(session_id=session_id, update=chunk)

                    elif isinstance(block, ToolUseBlock):
                        # Tool calls are displayed by handle_tool_permission
                        # Just track the ID for result handling
                        tool_calls[block.id] = block.id

        return response_text

    async def _call_inner_agent_silent(
        self,
        state: ComplianceState,
        prompt: str | list[dict[str, Any]],
        session_id: str,
    ) -> str:
        """Call the inner agent without displaying output to the UI.

        Used during planning phase where only the final approved plan is shown.

        Args:
            state: Compliance state for the session.
            prompt: Either a string or list of Claude API content blocks.
            session_id: ACP session identifier.

        Returns:
            The inner agent's text response.
        """
        client = await self._get_inner_client(state, session_id)

        if isinstance(prompt, str):
            await client.query(prompt)
        else:
            async def message_stream() -> AsyncIterator[dict[str, Any]]:
                yield {
                    "type": "user",
                    "message": {"role": "user", "content": prompt},
                    "parent_tool_use_id": None,
                    "session_id": "default",
                }

            await client.query(message_stream())

        response_text = ""

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text

        return response_text

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        """Handle initialization request."""
        logger.info(f"Initializing with protocol version {protocol_version}")
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(
                prompt_capabilities=PromptCapabilities(
                    image=True,
                    embedded_context=True,
                ),
            ),
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
        **kwargs: Any,
    ) -> NewSessionResponse:
        """Create a new session."""
        logger.info(f"Creating new session, cwd={cwd}")
        session_id = uuid4().hex
        state = self._get_state(session_id)
        state.cwd = Path(cwd)

        self._load_agents_md(state.cwd)
        logger.info(f"Session {session_id} created, AGENTS.md loaded from {self.agents_md_path}")

        return NewSessionResponse(session_id=session_id)

    async def load_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
        session_id: str,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        """Load an existing session."""
        return None

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        """List available sessions."""
        return ListSessionsResponse(sessions=[])

    async def set_session_mode(
        self,
        mode_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModeResponse | None:
        """Set session mode."""
        return None

    async def set_session_model(
        self,
        model_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModelResponse | None:
        """Set session model."""
        return None

    async def authenticate(
        self,
        method_id: str,
        **kwargs: Any,
    ) -> AuthenticateResponse | None:
        """Handle authentication."""
        return None

    async def prompt(
        self,
        prompt: list[PromptBlock],
        session_id: str,
        **kwargs: Any,
    ) -> PromptResponse:
        """Handle a prompt through triage, planning, execution, and review.

        Workflow:
        1. Triage: Haiku determines if message needs planning or direct answer
        2. Planning: Request YAML plan, review until 5/5 (target: 3 attempts, max: 5)
        3. Execution: Inner agent implements plan
        4. Review: Check work compliance, revise if needed
        """
        logger.info(f"Received prompt for session {session_id}")
        state = self._get_state(session_id)

        # Extract structured content (supports images)
        prompt_content = _extract_prompt_content(prompt, cwd=state.cwd)
        prompt_text = _extract_prompt_text(prompt, cwd=state.cwd)

        if not prompt_text.strip():
            chunk = update_agent_message(text_block("Error: Empty prompt"))
            await self._conn.session_update(session_id=session_id, update=chunk)
            return PromptResponse(stop_reason="end_turn")

        # Reset state for new user prompt
        state.reset_for_new_prompt()
        state.current_user_prompt = prompt_text
        state.user_request_history.append(prompt_text)

        # Check if prompt contains images
        has_images = any(block.get("type") == "image" for block in prompt_content)

        try:
            # === TRIAGE: Determine if planning is needed ===
            classify_client = get_classify_client(self.classify_model)
            try:
                triage_result = await asyncio.to_thread(
                    triage_user_message,
                    classify_client,
                    prompt_text,
                )
            except InvalidTriageResponse:
                # Default to planning if triage fails
                triage_result = "PLAN"

            if triage_result == "ANSWER":
                # Direct answer - no planning needed
                logger.info("Triage: Direct answer (no planning)")
                if has_images:
                    await self._call_inner_agent(state, prompt_content, session_id)
                else:
                    await self._call_inner_agent(state, prompt_text, session_id)
                return PromptResponse(stop_reason="end_turn")

            # === PHASE 1: PLANNING ===
            logger.info("Phase 1: Planning")
            state.phase = WorkflowPhase.PLANNING

            # Request YAML plan from inner agent (silently - no text output)
            plan_prompt = f"""\
Create an implementation plan for this request. Output ONLY a YAML plan.

REQUEST:
{prompt_text}

PLAN FORMAT (YAML only, no code, no markdown):
```yaml
goal: <one-line description of what will be achieved>
tasks:
  - description: <what this task accomplishes>
    files: [<files to read or modify>]
  - description: <next task>
    files: [<files>]
documentation:
  - <doc file to update, if any exist for affected area>
tests:
  - <test file to create or update>
```

RULES:
- Output ONLY the YAML plan, no other text
- Do NOT include code snippets in the plan
- Each task is a discrete step the user can track
- Include documentation updates if docs exist for the affected area
- Include tests for new features or bug fixes
"""
            if has_images:
                plan_prompt_content = [*prompt_content, {"type": "text", "text": plan_prompt}]
                plan_response = await self._call_inner_agent_silent(
                    state, plan_prompt_content, session_id
                )
            else:
                plan_response = await self._call_inner_agent_silent(
                    state, plan_prompt, session_id
                )

            # Review plan (target: pass in 3, max: 5 attempts)
            while state.plan_review_attempts < state.max_plan_attempts:
                attempt = state.plan_review_attempts + 1
                max_attempts = state.max_plan_attempts
                remaining = max_attempts - attempt

                # Create a review block for this attempt
                review_block_id = f"plan_review_{attempt}_{uuid4().hex[:8]}"
                review_block = start_tool_call(
                    tool_call_id=review_block_id,
                    title=f"Reviewing plan (attempt {attempt}/{max_attempts})",
                    kind="think",
                    status="in_progress",
                )
                await self._conn.session_update(session_id=session_id, update=review_block)

                report = await self._review_plan(state, plan_response, session_id)

                if report.is_compliant:
                    state.approved_plan = plan_response

                    # Update review block to completed
                    review_update = update_tool_call(
                        tool_call_id=review_block_id,
                        title=f"Plan review {attempt}/{max_attempts}: Approved",
                        status="completed",
                    )
                    await self._conn.session_update(session_id=session_id, update=review_update)

                    # Parse plan into tasks and send plan widget
                    await self._parse_and_send_plan(state, plan_response, session_id)

                    # Show the approved plan
                    plan_msg = f"\n\n**Approved Plan:**\n\n{plan_response}\n"
                    chunk = update_agent_message(text_block(plan_msg))
                    await self._conn.session_update(session_id=session_id, update=chunk)
                    break

                # Build violation details for display
                violation_lines = []
                for section in report.failing_sections:
                    for g in section.guideline_scores:
                        if not g.score.is_passing:
                            violation_lines.append(
                                f"  - {g.guideline_id} ({g.score}): {g.justification}"
                            )

                violations_text = "\n".join(violation_lines) if violation_lines else "  (none)"

                if state.plan_review_attempts >= state.max_plan_attempts:
                    # Final attempt failed - update block with violations
                    review_update = update_tool_call(
                        tool_call_id=review_block_id,
                        title=f"Plan review {attempt}/{max_attempts}: Failed",
                        status="failed",
                    )
                    await self._conn.session_update(session_id=session_id, update=review_update)

                    # Show violations in a message
                    violations_msg = (
                        f"\n\n**Violations (attempt {attempt}):**\n{violations_text}\n"
                    )
                    chunk = update_agent_message(text_block(violations_msg))
                    await self._conn.session_update(session_id=session_id, update=chunk)

                    # Build list of failing guidelines for escalation
                    failing_guidelines = []
                    for section in report.failing_sections:
                        for g in section.guideline_scores:
                            if not g.score.is_passing:
                                failing_guidelines.append(
                                    f"- **{g.guideline_id}**: {g.justification}"
                                )

                    # Generate context-specific questions using Haiku
                    violations_for_questions = [
                        f"{g.guideline_id}: {g.justification}"
                        for section in report.failing_sections
                        for g in section.guideline_scores
                        if not g.score.is_passing
                    ]
                    questions = await asyncio.to_thread(
                        generate_clarifying_questions,
                        classify_client,
                        prompt_text,
                        violations_for_questions,
                    )

                    # Format numbered questions
                    numbered_questions = "\n".join(
                        f"{i + 1}. {q}" for i, q in enumerate(questions)
                    )

                    # Escalate to user with specific questions
                    escalate_msg = (
                        f"\n\n**I need clarification to proceed.**\n\n"
                        f"After {max_attempts} attempts, I could not create a compliant plan.\n\n"
                        f"**Unresolved issues:**\n"
                        + "\n".join(failing_guidelines)
                        + f"\n\n**Questions:**\n{numbered_questions}\n"
                    )
                    chunk = update_agent_message(text_block(escalate_msg))
                    await self._conn.session_update(session_id=session_id, update=chunk)
                    return PromptResponse(stop_reason="end_turn")

                # Build revision strategy text
                revision_strategy = report.revision_guidance or "Address all violations listed."

                # Update review block with violations and revision strategy
                review_update = update_tool_call(
                    tool_call_id=review_block_id,
                    title=f"Plan review {attempt}/{max_attempts}: Revising",
                    status="failed",
                )
                await self._conn.session_update(session_id=session_id, update=review_update)

                # Show violations and revision strategy in a message
                attempt_summary = (
                    f"\n\n**Plan Review (attempt {attempt}/{max_attempts})**\n\n"
                    f"**Violations:**\n{violations_text}\n\n"
                    f"**Revision Strategy:**\n{revision_strategy}\n"
                )
                chunk = update_agent_message(text_block(attempt_summary))
                await self._conn.session_update(session_id=session_id, update=chunk)

                # Build clear violation feedback for the agent
                violations = []
                for section in report.failing_sections:
                    for g in section.guideline_scores:
                        if not g.score.is_passing:
                            violations.append(
                                f"- {g.guideline_id} ({g.score}): {g.justification}"
                            )

                urgency = ""
                if remaining <= 2:
                    urgency = (
                        f"\n\n⚠️ URGENT: You have {remaining} attempt(s) remaining. "
                        f"Address ALL violations below or provide numbered questions "
                        f"if you need clarification from the user."
                    )

                # Request revised plan with clear feedback
                revise_prompt = f"""\
Your plan FAILED compliance review. You MUST fix these violations:{urgency}

VIOLATIONS:
{chr(10).join(violations)}

{f"REVISION GUIDANCE: {report.revision_guidance}" if report.revision_guidance else ""}

REQUIREMENTS:
1. Output ONLY a YAML plan (no code, no markdown explanation)
2. Address EVERY violation listed above
3. If you cannot address a violation, explain why in a numbered question format

ORIGINAL REQUEST:
{prompt_text}

Provide your revised YAML plan now:
"""
                plan_response = await self._call_inner_agent_silent(
                    state, revise_prompt, session_id
                )

            # === PHASE 2: EXECUTION ===
            logger.info("Phase 2: Execution")
            state.phase = WorkflowPhase.EXECUTING

            execute_prompt = (
                f"Implement the approved plan. Proceed with the implementation now.\n\n"
                f"Plan:\n{state.approved_plan}"
            )
            work_response = await self._call_inner_agent(state, execute_prompt, session_id)
            state.work_summary = work_response

            # Update task progress after execution
            await self._update_task_progress(state, work_response, session_id)

            # === PHASE 3: REVIEW CYCLE ===
            while True:
                # Check if work is ready for review
                ready = await self._check_ready_for_review(state, work_response)

                if ready:
                    logger.info("Phase 3: Reviewing work")
                    state.phase = WorkflowPhase.REVIEWING

                    # Show review block with spinner
                    review_block_id = f"review_{uuid4().hex[:8]}"
                    review_block = start_tool_call(
                        tool_call_id=review_block_id,
                        title="Reviewing completed work...",
                        kind="think",
                        status="in_progress",
                    )
                    await self._conn.session_update(
                        session_id=session_id, update=review_block
                    )

                    report = await self._review_work(state, state.work_summary, session_id)

                    if report.is_compliant:
                        state.phase = WorkflowPhase.COMPLETE

                        # Mark all tasks as completed
                        for task in state.plan_tasks:
                            task.status = "completed"
                        await self._send_plan_update(state, session_id)

                        # Update review block to completed
                        review_update = update_tool_call(
                            tool_call_id=review_block_id,
                            title="Work approved",
                            status="completed",
                        )
                        await self._conn.session_update(
                            session_id=session_id, update=review_update
                        )
                        break

                    # Update review block to failed
                    review_update = update_tool_call(
                        tool_call_id=review_block_id,
                        title="Work review failed - revising",
                        status="failed",
                    )
                    await self._conn.session_update(
                        session_id=session_id, update=review_update
                    )

                    # Request revision from inner agent
                    revise_prompt = (
                        f"Your work failed compliance review. Please revise:\n\n"
                        f"{report.format()}\n\n"
                        f"Revision guidance: {report.revision_guidance}\n\n"
                        f"Make the necessary changes to achieve 5/5 on all guidelines."
                    )
                    work_response = await self._call_inner_agent(
                        state, revise_prompt, session_id
                    )
                    state.work_summary += f"\n\n[Revision]\n{work_response}"

                    # Update task progress after revision
                    await self._update_task_progress(state, work_response, session_id)
                else:
                    # Not ready - work is still in progress, wait for next iteration
                    # The inner agent should continue working
                    logger.info("Work not ready for review, continuing execution")
                    break

        except Exception as e:
            error_msg = f"Error: {e}"
            chunk = update_agent_message(text_block(error_msg))
            await self._conn.session_update(session_id=session_id, update=chunk)
            logger.exception(f"Workflow error: {e}")

        return PromptResponse(stop_reason="end_turn")

    async def cancel(
        self,
        session_id: str,
        **kwargs: Any,
    ) -> None:
        """Cancel current operation."""
        state = self._get_state(session_id)
        if state.inner_client:
            await state.inner_client.interrupt()

    async def ext_method(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle extension method."""
        return {}

    async def ext_notification(
        self,
        method: str,
        params: dict[str, Any],
    ) -> None:
        """Handle extension notification."""
        pass


async def run_server(
    agents_md_path: Path | None = None,
    review_model: str | None = None,
    classify_model: str = "haiku",
) -> None:
    """Run the Delta ACP server."""
    from acp import stdio_streams

    agent = DeltaAgent(
        agents_md_path=agents_md_path,
        review_model=review_model,
        classify_model=classify_model,
    )

    # Use larger buffer limit (16MB) to handle base64-encoded images
    # Default 64KB limit causes crashes with image content
    output_stream, input_stream = await stdio_streams(limit=16 * 1024 * 1024)
    await run_agent(agent, input_stream=input_stream, output_stream=output_stream)


def main() -> None:
    """Entry point for Delta ACP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
