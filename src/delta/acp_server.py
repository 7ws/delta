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
    UserPromptSubmitHookInput,
)

from delta.compliance import ComplianceReport
from delta.guidelines import (
    AgentsDocument,
    find_agents_md,
    get_bundled_agents_md,
    parse_agents_md,
)
from delta.llm import (
    ClaudeCodeClient,
    InvalidComplexityResponse,
    InvalidPlanParseResponse,
    InvalidReviewReadinessResponse,
    InvalidTaskDuplicateResponse,
    InvalidTriageResponse,
    InvalidWriteClassificationResponse,
    classify_task_complexity,
    classify_task_duplicate,
    classify_write_operation,
    detect_task_progress,
    generate_clarifying_questions,
    get_classify_client,
    get_llm_client,
    is_ready_for_review,
    parse_plan_tasks,
    triage_user_message,
)
from delta.plan_widget import PlanTask, PlanWidgetManager
from delta.protocol import (
    compute_edit_result,
    extract_prompt_content,
    extract_prompt_text,
    format_tool_action,
    read_file_content,
)
from delta.review import ParseError, ReviewPhaseHandler
from delta.tools import ToolPermissionHandler
from delta.workflow import WorkflowContext, WorkflowOrchestrator

logger = logging.getLogger(__name__)


# Type alias for prompt blocks
PromptBlock = (
    TextContentBlock
    | ImageContentBlock
    | AudioContentBlock
    | ResourceContentBlock
    | EmbeddedResourceContentBlock
)


class WorkflowPhase(Enum):
    """Current phase of the workflow."""

    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETE = "complete"


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

    # Context tracking for logging
    current_action: str = ""  # What the agent is currently doing

    # Write operation tracking - ensures compliance review for ANY writes
    has_write_operations: bool = False

    def record_tool_call(self, tool_description: str, allowed: bool) -> None:
        """Record a tool call in the session history."""
        status = "ALLOWED" if allowed else "DENIED"
        self.tool_call_history.append(f"[{status}] {tool_description}")

    def set_current_action(self, action: str) -> None:
        """Set the current action being performed (for logging)."""
        self.current_action = action

    def reset_for_new_prompt(self) -> None:
        """Reset state for a new user prompt.

        Resets workflow state while preserving tool call history
        for compliance reviewer context. Preserves incomplete tasks
        to ensure follow-up prompts don't lose pending work.
        """
        self.phase = WorkflowPhase.PLANNING
        self.approved_plan = ""
        self.work_summary = ""
        self.plan_review_attempts = 0
        # Preserve incomplete tasks - only remove completed ones
        self.plan_tasks = [t for t in self.plan_tasks if t.status != "completed"]
        self.current_action = ""
        self.has_write_operations = False


class DeltaAgent(Agent):
    """ACP agent that wraps another agent with compliance enforcement."""

    _conn: Client

    def __init__(
        self,
        agents_md_path: Path | None = None,
        review_model: str | None = None,
        classify_model: str = "haiku",
    ) -> None:
        """Initialize Delta agent.

        Args:
            agents_md_path: Path to AGENTS.md file (auto-detected if not specified).
            review_model: Model for compliance reviews.
            classify_model: Model for action classification (default: haiku).
        """
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

    def _get_review_handler(self) -> ReviewPhaseHandler:
        """Get or create the review handler with current agents_doc."""
        self._load_agents_md()
        if self.agents_doc is None:
            raise RuntimeError("Failed to load AGENTS.md")
        return ReviewPhaseHandler(self.llm_client, self.agents_doc)

    async def _review_simple_plan(
        self,
        state: ComplianceState,
        plan: str,
        session_id: str,
    ) -> ComplianceReport:
        """Review a simple plan with lightweight validation."""
        handler = self._get_review_handler()
        report = await handler.review_simple_plan(state.current_user_prompt, plan)
        state.plan_review_attempts += 1
        return report

    async def _review_plan(
        self,
        state: ComplianceState,
        plan: str,
        session_id: str,
    ) -> ComplianceReport:
        """Review a plan for compliance before execution."""
        handler = self._get_review_handler()
        report = await handler.review_plan(state.current_user_prompt, plan)
        state.plan_review_attempts += 1
        return report

    async def _review_work(
        self,
        state: ComplianceState,
        work_summary: str,
        session_id: str,
    ) -> ComplianceReport:
        """Review accumulated work for compliance."""
        handler = self._get_review_handler()
        return await handler.review_work(
            state.current_user_prompt,
            state.approved_plan,
            work_summary,
            state.tool_call_history,
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
        """Parse an approved plan into tasks and merge with existing tasks.

        Preserves existing pending/in_progress tasks and appends new unique
        tasks from the approved plan. Uses AI to detect duplicates.

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

            # Get existing task descriptions for deduplication
            existing_descriptions = [t.content for t in state.plan_tasks]

            # Filter out duplicate tasks using AI classification
            new_tasks: list[PlanTask] = []
            for desc in task_descriptions:
                try:
                    is_duplicate = await asyncio.to_thread(
                        classify_task_duplicate,
                        classify_client,
                        existing_descriptions + [t.content for t in new_tasks],
                        desc,
                    )
                    if not is_duplicate:
                        new_tasks.append(PlanTask(content=desc))
                        logger.debug(f"Adding new task: {desc}")
                    else:
                        logger.debug(f"Skipping duplicate task: {desc}")
                except InvalidTaskDuplicateResponse:
                    # When uncertain, add the task (safer than losing work)
                    new_tasks.append(PlanTask(content=desc))
                    logger.warning(f"Duplicate check failed, adding task: {desc}")

            # Merge: existing tasks + new unique tasks
            state.plan_tasks.extend(new_tasks)

            # Send plan widget to UI
            entries = [
                plan_entry(task.content, status=task.status)
                for task in state.plan_tasks
            ]
            plan_update = update_plan(entries)
            await self._conn.session_update(session_id=session_id, update=plan_update)

            logger.info(
                f"Plan widget: {len(state.plan_tasks)} total tasks "
                f"({len(new_tasks)} new, {len(existing_descriptions)} existing)"
            )

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
        """Build system prompt that includes AGENTS.md and user communication guidelines."""
        self._load_agents_md()

        if self.agents_doc is None:
            return ""

        return (
            "You MUST read and follow the guidelines in AGENTS.md for every response.\n\n"
            "CRITICAL - PROACTIVE THOROUGHNESS (guideline 2.2.8-2.2.9):\n"
            "When a change affects multiple files, update ALL affected files without "
            "asking for permission. Do not stop after one file and ask whether to "
            "update the others. Complete the job. Incomplete work is a violation.\n\n"
            "USER COMMUNICATION GUIDELINES:\n"
            "You are talking directly to the user. Write as if you are the only agent.\n"
            "- DO: Share progress naturally (\"Reading the file\", \"Creating the component\")\n"
            "- DO: Explain what you are doing and why in user-friendly terms\n"
            "- DO: Show code, file paths, and technical results\n"
            "- DO NOT: Mention compliance, reviews, guidelines, or AGENTS.md\n"
            "- DO NOT: Reference internal scores, attempts, or workflow phases\n"
            "- DO NOT: Say \"the user chose\" or \"the user acknowledged\"\n"
            "- DO NOT: Mention section references like §X.X.X\n"
            "Keep your output clean and direct.\n\n"
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
                tool_description = format_tool_action(tool_name, input_params)
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

                        # Track write operations - they MUST trigger compliance review
                        # Uses AI classification (no hardcoded patterns - architectural decision)
                        try:
                            classify_client = get_classify_client(self.classify_model)
                            is_write = await asyncio.to_thread(
                                classify_write_operation,
                                classify_client,
                                tool_name,
                                tool_description,
                            )
                            if is_write:
                                state.has_write_operations = True
                                logger.info(f"Write operation tracked: {tool_description}")
                        except InvalidWriteClassificationResponse:
                            # Default to write operation if classification fails (safe side)
                            state.has_write_operations = True
                            logger.warning(
                                f"Write classification failed, treating as write: {tool_description}"
                            )

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
                tool_description = format_tool_action(tool_name, tool_input)

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
        """Call the inner agent and stream its response to the user.

        The inner agent is instructed via system prompt to only emit user-relevant
        content, so we stream directly without filtering.

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

                        # Stream directly to user (inner agent is instructed to emit clean output)
                        if text.strip():
                            chunk = update_agent_message(text_block(text))
                            await self._conn.session_update(session_id=session_id, update=chunk)

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

    def _get_workflow_orchestrator(self, session_id: str) -> WorkflowOrchestrator:
        """Create a workflow orchestrator with callbacks to this agent."""
        return WorkflowOrchestrator(
            conn=self._conn,
            classify_model=self.classify_model,
            call_inner_agent=self._call_inner_agent,
            call_inner_agent_silent=self._call_inner_agent_silent,
            review_simple_plan=self._review_simple_plan,
            review_plan=self._review_plan,
            review_work=self._review_work,
            check_ready_for_review=self._check_ready_for_review,
            parse_and_send_plan=self._parse_and_send_plan,
            send_plan_update=self._send_plan_update,
            update_task_progress=self._update_task_progress,
        )

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
        prompt_content = extract_prompt_content(prompt, cwd=state.cwd)
        prompt_text = extract_prompt_text(prompt, cwd=state.cwd)

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

        # Create workflow context
        ctx = WorkflowContext(
            prompt_text=prompt_text,
            prompt_content=prompt_content,
            has_images=has_images,
            session_id=session_id,
            state=state,
        )

        # Create orchestrator with callbacks
        orchestrator = self._get_workflow_orchestrator(session_id)

        try:
            # === TRIAGE: Determine if planning is needed ===
            triage_result = await orchestrator.triage(ctx)

            if not triage_result.needs_planning:
                # Direct answer - no planning needed
                await orchestrator.handle_direct_answer(ctx)
                return PromptResponse(stop_reason="end_turn")

            # === PLANNING, EXECUTION, AND REVIEW ===
            plan_approved = await orchestrator.handle_planning(
                ctx, triage_result.complexity or "MODERATE", WorkflowPhase
            )

            if not plan_approved:
                # Planning escalated or failed
                return PromptResponse(stop_reason="end_turn")

            # Execute the approved plan
            await orchestrator.handle_execution(ctx, WorkflowPhase)

            # Review cycle
            await orchestrator.handle_review_cycle(ctx, WorkflowPhase)

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
    """Run the Delta ACP server.

    Args:
        agents_md_path: Path to AGENTS.md file (auto-detected if not specified).
        review_model: Model for compliance reviews.
        classify_model: Model for action classification (default: haiku).
    """
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
