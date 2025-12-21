"""ACP server implementation for Delta compliance wrapper.

Delta acts as a proxy between the editor and an inner AI agent, intercepting
all actions and performing compliance reviews before allowing execution.

Architecture: Editor <-> Delta (ACP) <-> Inner Agent (Claude Agent SDK)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from acp import (
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    start_tool_call,
    text_block,
    tool_diff_content,
    update_agent_message,
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
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    PreToolUseHookInput,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from delta.compliance import (
    ComplianceReport,
    build_compliance_prompt,
    build_lightweight_prompt,
    parse_compliance_response,
    parse_lightweight_response,
)
from delta.guidelines import (
    AgentsDocument,
    find_agents_md,
    get_bundled_agents_md,
    parse_agents_md,
)
from delta.llm import LLMClient, classify_action_type, get_fast_client, get_llm_client

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
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.get("mimeType", block.get("mime_type", "image/png")),
                        "data": block.get("data", ""),
                    },
                })
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
                    text_parts.append(
                        f'<file uri="{uri}" name="{name}">\n{file_content}\n</file>'
                    )
                else:
                    text_parts.append(f"[Referenced file: {name} ({uri})]")
        elif isinstance(block, ImageContentBlock):
            flush_text()
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block.mime_type,
                    "data": block.data,
                },
            })
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


@dataclass
class ComplianceState:
    """Track compliance state for a session.

    Architecture: The agent gets `max_attempts` tries to propose a compliant action.
    After exhausting attempts, Delta resets the counter and provides actionable
    guidance. The agent should either fix the issue or ask the user for clarification
    if the feedback is unclear.

    This design prevents infinite loops while allowing the agent to self-correct
    when given clear feedback (e.g., "run pre-commit first").
    """

    max_attempts: int = 3
    current_attempt: int = 0
    blocked: bool = False
    previous_reports: list[ComplianceReport] = field(default_factory=list)
    agents_doc: AgentsDocument | None = None
    inner_client: ClaudeSDKClient | None = None
    cwd: Path | None = None
    current_user_prompt: str = ""
    tool_call_history: list[str] = field(default_factory=list)
    last_failure_was_runtime: bool = False

    def record_attempt(self, report: ComplianceReport) -> None:
        """Record a compliance attempt.

        Only compliance violations count toward the block threshold.
        If the last failure was a runtime error and this attempt is compliant,
        the agent successfully recovered - reset the attempt counter.
        """
        if report.is_compliant:
            # Successful action - check if this is recovery from runtime error
            if self.last_failure_was_runtime:
                logger.info("Agent recovered from runtime error with compliant action")
                self.current_attempt = 0
                self.last_failure_was_runtime = False
            return

        # Non-compliant action
        self.current_attempt += 1
        self.previous_reports.append(report)
        self.last_failure_was_runtime = False

        if self.current_attempt >= self.max_attempts:
            self.blocked = True

    def record_runtime_error(self, tool_description: str, error: str) -> None:
        """Record a runtime error (not a compliance violation).

        Runtime errors (interactive mode unavailable, file not found, etc.)
        do not count toward the block threshold. They signal that the agent
        should try a different approach.
        """
        self.last_failure_was_runtime = True
        self.tool_call_history.append(f"[RUNTIME_ERROR] {tool_description}: {error}")
        logger.info(f"Runtime error recorded (not counting toward block): {error}")

    def record_tool_call(self, tool_description: str, allowed: bool) -> None:
        """Record a tool call in the session history."""
        status = "ALLOWED" if allowed else "DENIED"
        self.tool_call_history.append(f"[{status}] {tool_description}")

    def reset(self) -> None:
        """Reset compliance attempts for a new user prompt (but keep history)."""
        self.current_attempt = 0
        self.blocked = False
        self.previous_reports.clear()
        self.last_failure_was_runtime = False
        # NOTE: We don't clear tool_call_history - it persists across prompts
        # so the compliance reviewer knows what actions were taken earlier


class DeltaAgent(Agent):
    """ACP agent that wraps another agent with compliance enforcement."""

    _conn: Client

    def __init__(
        self,
        agents_md_path: Path | None = None,
        max_attempts: int = 2,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        """Initialize Delta agent."""
        self.max_attempts = max_attempts
        self.sessions: dict[str, ComplianceState] = {}
        self.llm_client: LLMClient = get_llm_client(
            provider=llm_provider,
            model=llm_model,
        )

        self._explicit_agents_md_path = agents_md_path
        self.agents_md_path: Path | None = None
        self.agents_doc: AgentsDocument | None = None

    def on_connect(self, conn: Client) -> None:
        """Handle connection from editor."""
        self._conn = conn

    def _get_state(self, session_id: str) -> ComplianceState:
        """Get or create compliance state for session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ComplianceState(max_attempts=self.max_attempts)
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

    async def _perform_compliance_review(
        self,
        state: ComplianceState,
        action: str,
        user_prompt: str = "",
        lightweight: bool = False,
        progress_callback: Any = None,
    ) -> ComplianceReport:
        """Perform compliance review on a proposed action.

        Args:
            state: Compliance state for the session.
            action: The proposed action to review.
            user_prompt: The user's original request.
            lightweight: Use lightweight review for read operations.
            progress_callback: Async callback(elapsed_seconds) to update progress UI.
        """
        start_time = time.monotonic()

        self._load_agents_md()

        if self.agents_doc is None:
            raise RuntimeError("Failed to load AGENTS.md")

        review_type = "quick" if lightweight else "full"

        if lightweight:
            prompt = build_lightweight_prompt(self.agents_doc, action)
            logger.debug(f"Lightweight compliance prompt length: {len(prompt)} chars")
        else:
            prompt = build_compliance_prompt(
                self.agents_doc, action, user_prompt, state.tool_call_history
            )
            logger.debug(f"Full compliance prompt length: {len(prompt)} chars")

        llm_start = time.monotonic()

        # Run LLM call with progress updates
        llm_task = asyncio.create_task(
            asyncio.to_thread(
                self.llm_client.complete,
                prompt=prompt,
                system="You are a strict compliance reviewer. Output only valid JSON.",
            )
        )

        # Update progress every 0.5 seconds while waiting for LLM
        while not llm_task.done():
            elapsed = time.monotonic() - llm_start
            if progress_callback:
                await progress_callback(elapsed, review_type)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(llm_task), timeout=0.5)

        review_response = await llm_task
        llm_duration = time.monotonic() - llm_start
        logger.debug(f"LLM compliance review took {llm_duration:.2f}s")

        if lightweight:
            report = parse_lightweight_response(review_response)
        else:
            report = parse_compliance_response(review_response, self.agents_doc)
        report.proposed_action = action[:100] + "..." if len(action) > 100 else action
        report.attempt_number = state.current_attempt + 1

        state.record_attempt(report)

        total_duration = time.monotonic() - start_time
        review_type = "lightweight" if lightweight else "full"
        logger.info(
            f"Compliance review ({review_type}) completed in {total_duration:.2f}s, "
            f"compliant={report.is_compliant}"
        )

        return report

    def _build_system_prompt(self) -> str:
        """Build system prompt that includes AGENTS.md."""
        self._load_agents_md()

        if self.agents_doc is None:
            return ""

        return (
            "You MUST read and follow the guidelines in AGENTS.md for every response.\n\n"
            f"AGENTS.md content:\n\n{self.agents_doc.raw_content}"
        )

    async def _get_inner_client(self, state: ComplianceState, session_id: str) -> ClaudeSDKClient:
        """Get or create the inner Claude SDK client for this session."""
        if state.inner_client is None:

            async def handle_tool_permission(
                tool_name: str,
                input_params: dict[str, Any],
                context: dict[str, Any],
            ) -> PermissionResultAllow | PermissionResultDeny:
                """Review tool for compliance and request user permission.

                Compliance review occurs before tool execution. If compliance
                fails, the tool is denied without executing.

                Returns:
                    PermissionResultAllow or PermissionResultDeny from claude_agent_sdk.
                    Must return SDK types, not raw dicts, or the inner agent receives
                    confusing error messages about "invalid response type".

                Note on interrupt flag:
                    When denying, interrupt=False allows the inner agent to continue
                    and try a different approach. interrupt=True abruptly stops the
                    agent's turn, which can cause confusion as the agent does not
                    receive proper context about why it was stopped.
                """
                import sys

                tool_description = self._format_tool_action(tool_name, input_params)
                logger.info(f"Tool call: {tool_name} - {tool_description}")

                # Determine if this is a read-only operation (lightweight review)
                # Known read-only tools get instant classification
                known_read_only_tools = {
                    "Read", "mcp__acp__Read",
                    "Glob", "Grep",
                    "Task",  # Task spawns subagents for exploration
                }
                if tool_name in known_read_only_tools:
                    is_read_only = True
                else:
                    # Use fast LLM (Haiku) to classify unknown actions
                    fast_client = get_fast_client()
                    is_read_only = await asyncio.to_thread(
                        classify_action_type, fast_client, tool_description
                    )

                try:
                    # Show progress as a native tool call block
                    review_id = f"review_{uuid4().hex[:8]}"
                    review_type_label = "quick" if is_read_only else "full"
                    logger.debug(f"Creating progress block: {review_id}")
                    sys.stderr.flush()
                    review_start = start_tool_call(
                        tool_call_id=review_id,
                        title=f"Reviewing compliance ({review_type_label})...",
                        kind="think",
                        status="in_progress",
                    )
                    logger.debug("start_tool_call returned")
                    sys.stderr.flush()
                    await self._conn.session_update(session_id=session_id, update=review_start)
                    logger.debug("session_update completed")

                    # Progress callback to update UI with elapsed time
                    async def update_progress(elapsed: float, review_type: str) -> None:
                        title = f"Reviewing compliance ({review_type}, {elapsed:.1f}s)..."
                        progress_update = update_tool_call(
                            tool_call_id=review_id,
                            title=title,
                        )
                        await self._conn.session_update(
                            session_id=session_id, update=progress_update
                        )

                    report = await self._perform_compliance_review(
                        state,
                        tool_description,
                        state.current_user_prompt,
                        lightweight=is_read_only,
                        progress_callback=update_progress,
                    )
                    logger.info(f"Compliance review done: {report.is_compliant}")

                    if not report.is_compliant:
                        # Build detailed failure message with justifications
                        failure_details = []
                        for section in report.failing_sections:
                            failure_details.append(
                                f"- §{section.section_number} {section.section_name}: "
                                f"{section.format_score()}"
                            )
                            for g in section.guideline_scores:
                                if not g.score.is_passing and g.justification:
                                    failure_details.append(
                                        f"  - {g.guideline_id}: {g.justification}"
                                    )

                        failing_text = "\n".join(failure_details)
                        logger.warning(f"Tool action failed compliance:\n{failing_text}")

                        # Update tool call to show failure
                        review_update = update_tool_call(
                            tool_call_id=review_id,
                            status="failed",
                        )
                        await self._conn.session_update(session_id=session_id, update=review_update)

                        # Show failure details as text
                        compliance_msg = f"\n\n**Compliance Review Failed**\n\n{failing_text}\n"
                        chunk = update_agent_message(text_block(compliance_msg))
                        await self._conn.session_update(session_id=session_id, update=chunk)

                        # After max_attempts failures, guide the agent to fix or escalate
                        # Architecture: 2 attempts to comply, then either guide (actionable)
                        # or ask for clarification (ambiguous)
                        if state.blocked:
                            # Reset for another round - the feedback is actionable
                            state.blocked = False
                            state.current_attempt = 0

                            return PermissionResultDeny(
                                message=(
                                    f"Compliance review failed after {self.max_attempts} "
                                    f"attempts:\n{failing_text}\n\n"
                                    "The compliance feedback above is actionable. "
                                    "Address these issues before retrying this action. "
                                    "If you cannot determine how to proceed, ask the user "
                                    "for clarification."
                                ),
                                interrupt=False,
                            )

                        return PermissionResultDeny(
                            message=(
                                f"Action failed compliance review:\n{failing_text}\n\n"
                                "Review the guidelines and try a compliant approach."
                            ),
                            interrupt=False,
                        )

                    # Update tool call to show success
                    review_update = update_tool_call(
                        tool_call_id=review_id,
                        status="completed",
                    )
                    await self._conn.session_update(session_id=session_id, update=review_update)
                except Exception as e:
                    # Fail-safe: if compliance review errors, deny the action
                    logger.exception(f"Compliance review error: {e}")
                    sys.stderr.flush()
                    return PermissionResultDeny(
                        message=(
                            f"Compliance review failed with error: {e}. Action denied for safety."
                        ),
                        interrupt=True,
                    )

                # Compliance passed; request user permission
                tool_call_id = f"tool_{uuid4().hex[:8]}"

                # Build rich content for file operations
                tool_content = None
                tool_locations = None
                tool_kind = None
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
                        await self._conn.session_update(
                            session_id=session_id, update=tool_progress
                        )
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
                    await self._conn.session_update(
                        session_id=session_id, update=tool_progress
                    )
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
                await self._conn.session_update(
                    session_id=session_id, update=tool_progress
                )
                state.record_tool_call(tool_description, allowed=False)
                return PermissionResultDeny(
                    message=(
                        "Permission prompt was cancelled. "
                        "Stop and ask for clarification before proceeding."
                    ),
                    interrupt=True,
                )

            async def track_tool_call(
                hook_input: PreToolUseHookInput,
                tool_use_id: str | None,
                context: HookContext,
            ) -> dict[str, Any]:
                """Track all tool calls for compliance review context.

                This hook fires for ALL tools (Read, Glob, Grep, etc.) before
                they execute. We use it to record tool usage so the compliance
                reviewer knows what actions have been taken.

                Returns empty dict to allow the tool call to proceed.
                """
                tool_name = hook_input["tool_name"]
                tool_input = hook_input["tool_input"]
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
                        # Show tool use in UI
                        tool_call_id = f"inner_{block.id[:8]}"
                        tool_calls[block.id] = tool_call_id
                        tool_title = self._format_tool_action(block.name, block.input)
                        tool_start = start_tool_call(
                            tool_call_id=tool_call_id,
                            title=tool_title,
                            status="in_progress",
                        )
                        await self._conn.session_update(session_id=session_id, update=tool_start)

                    elif isinstance(block, ToolResultBlock):
                        # Update tool call status based on result
                        tool_call_id = tool_calls.get(block.tool_use_id)
                        if tool_call_id:
                            status = "failed" if block.is_error else "completed"
                            tool_update = update_tool_call(
                                tool_call_id=tool_call_id,
                                status=status,
                            )
                            await self._conn.session_update(
                                session_id=session_id, update=tool_update
                            )

                        # Record runtime errors so recovery attempts are not penalized
                        if block.is_error:
                            tool_use_id = block.tool_use_id
                            # Extract error message from content
                            error_msg = ""
                            if isinstance(block.content, str):
                                error_msg = block.content
                            elif isinstance(block.content, list):
                                for item in block.content:
                                    if hasattr(item, "text"):
                                        error_msg = item.text
                                        break
                            # Get the tool description from history if available
                            tool_desc = f"tool_{tool_use_id[:8]}"
                            state.record_runtime_error(tool_desc, error_msg)

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
        """Handle a prompt by forwarding to the inner agent.

        Compliance is enforced at the tool level: each tool call is reviewed
        before execution via handle_tool_permission. This method forwards
        prompts to the inner agent and streams responses back.
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

        # Reset compliance attempt counter for new user prompts
        state.reset()
        state.current_user_prompt = prompt_text

        # Check if prompt contains images
        has_images = any(block.get("type") == "image" for block in prompt_content)
        inner_prompt: str | list[dict[str, Any]] = prompt_content if has_images else prompt_text

        logger.info(f"Calling inner agent (has_images={has_images})")

        try:
            await self._call_inner_agent(state, inner_prompt, session_id)
        except Exception as e:
            error_msg = f"Error: {e}"
            chunk = update_agent_message(text_block(error_msg))
            await self._conn.session_update(session_id=session_id, update=chunk)
            logger.exception(f"Inner agent error: {e}")

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
    max_attempts: int = 2,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> None:
    """Run the Delta ACP server."""
    agent = DeltaAgent(
        agents_md_path=agents_md_path,
        max_attempts=max_attempts,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    await run_agent(agent)


def main() -> None:
    """Entry point for Delta ACP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
