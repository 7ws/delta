"""Workflow orchestration for Delta compliance wrapper.

This module breaks down the complex prompt() method into focused phase handlers,
making the workflow easier to understand, test, and maintain.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from acp import text_block, update_agent_message

from delta.compliance import ComplianceReport
from delta.llm import (
    InvalidComplexityResponse,
    InvalidTriageResponse,
    classify_task_complexity,
    generate_clarifying_questions,
    get_classify_client,
    triage_user_message,
)
from delta.thinking_status import ThinkingStatusManager, WorkflowStep

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from acp.interfaces import Client

    from delta.acp_server import ComplianceState, WorkflowPhase

logger = logging.getLogger(__name__)


@dataclass
class WorkflowContext:
    """Context passed through workflow phases."""

    prompt_text: str
    prompt_content: list[dict[str, Any]]
    has_images: bool
    session_id: str
    state: ComplianceState


@dataclass
class TriageResult:
    """Result of the triage phase."""

    needs_planning: bool
    complexity: str | None = None  # SIMPLE, MODERATE, COMPLEX


class WorkflowOrchestrator:
    """Orchestrates the workflow phases for prompt handling.

    Breaks down the complex prompt() method into focused handlers:
    - Triage: Determine if planning is needed
    - Planning: Create and review implementation plan
    - Execution: Implement the approved plan
    - Review: Check work compliance and request revisions

    The orchestrator coordinates these phases and manages state transitions.
    """

    def __init__(
        self,
        conn: Client,
        thinking_status: ThinkingStatusManager,
        classify_model: str = "haiku",
        *,
        call_inner_agent: Callable[..., Coroutine[Any, Any, str]],
        call_inner_agent_silent: Callable[..., Coroutine[Any, Any, str]],
        review_simple_plan: Callable[..., Coroutine[Any, Any, ComplianceReport]],
        review_plan: Callable[..., Coroutine[Any, Any, ComplianceReport]],
        review_work: Callable[..., Coroutine[Any, Any, ComplianceReport]],
        check_ready_for_review: Callable[..., Coroutine[Any, Any, bool]],
        parse_and_send_plan: Callable[..., Coroutine[Any, Any, None]],
        send_plan_update: Callable[..., Coroutine[Any, Any, None]],
        update_task_progress: Callable[..., Coroutine[Any, Any, None]],
    ) -> None:
        """Initialize the workflow orchestrator.

        Args:
            conn: ACP client connection.
            thinking_status: Manager for real-time thinking status updates.
            classify_model: Model for triage and complexity classification.
            call_inner_agent: Callback to call inner agent with streaming output.
            call_inner_agent_silent: Callback to call inner agent silently.
            review_simple_plan: Callback to review a simple plan.
            review_plan: Callback to review a plan with full compliance check.
            review_work: Callback to review completed work.
            check_ready_for_review: Callback to check if work is ready for review.
            parse_and_send_plan: Callback to parse plan and send widget.
            send_plan_update: Callback to send plan widget update.
            update_task_progress: Callback to update task progress.
        """
        self._conn = conn
        self._thinking_status = thinking_status
        self._classify_model = classify_model

        # Callbacks to DeltaAgent methods
        self._call_inner_agent = call_inner_agent
        self._call_inner_agent_silent = call_inner_agent_silent
        self._review_simple_plan = review_simple_plan
        self._review_plan = review_plan
        self._review_work = review_work
        self._check_ready_for_review = check_ready_for_review
        self._parse_and_send_plan = parse_and_send_plan
        self._send_plan_update = send_plan_update
        self._update_task_progress = update_task_progress

    async def triage(self, ctx: WorkflowContext) -> TriageResult:
        """Triage phase: Determine if planning is needed.

        Uses AI classification to determine:
        - ANSWER: Direct response, no planning needed
        - PLAN: Planning and review workflow required

        Args:
            ctx: Workflow context with prompt and state.

        Returns:
            TriageResult indicating if planning is needed.
        """
        classify_client = get_classify_client(self._classify_model)

        try:
            result = await asyncio.to_thread(
                triage_user_message,
                classify_client,
                ctx.prompt_text,
            )
        except InvalidTriageResponse:
            # Default to planning if triage fails (safer)
            result = "PLAN"

        if result == "ANSWER":
            logger.info("Triage: Direct answer (no planning)")
            return TriageResult(needs_planning=False)

        # Classify complexity for planning
        try:
            complexity = await asyncio.to_thread(
                classify_task_complexity,
                classify_client,
                ctx.prompt_text,
            )
        except InvalidComplexityResponse:
            complexity = "MODERATE"

        logger.info(f"Triage: Planning required, complexity={complexity}")
        return TriageResult(needs_planning=True, complexity=complexity)

    async def handle_direct_answer(self, ctx: WorkflowContext) -> bool:
        """Handle direct answer flow (ANSWER triage result).

        Calls the inner agent directly without planning. If write operations
        are blocked due to missing plan, signals that planning is required.
        If write operations succeed, triggers mandatory compliance review.

        Args:
            ctx: Workflow context.

        Returns:
            True if direct answer completed, False if planning is required.
        """
        if ctx.has_images:
            response_text = await self._call_inner_agent(
                ctx.state, ctx.prompt_content, ctx.session_id
            )
        else:
            response_text = await self._call_inner_agent(
                ctx.state, ctx.prompt_text, ctx.session_id
            )

        # Check if a write was blocked due to missing plan
        if ctx.state.write_blocked_for_plan:
            logger.info("ANSWER flow blocked write - transitioning to planning")
            ctx.state.write_blocked_for_plan = False
            return False

        # If writes occurred, compliance review is mandatory
        if ctx.state.has_write_operations:
            logger.info("ANSWER flow had write operations - triggering mandatory review")
            await self._review_answer_writes(ctx, response_text)

        return True

    async def _review_answer_writes(self, ctx: WorkflowContext, response_text: str) -> None:
        """Review writes that occurred during direct answer flow.

        Args:
            ctx: Workflow context.
            response_text: Response from inner agent.
        """
        ctx.state.work_summary = response_text
        ctx.state.approved_plan = f"Respond to user query:\n{ctx.prompt_text}"

        await self._thinking_status.set_step(WorkflowStep.REVIEWING)

        report = await self._review_work(ctx.state, ctx.state.work_summary, ctx.session_id)

        if report.is_compliant:
            await self._thinking_status.stop("Changes verified")
        else:
            await self._thinking_status.set_step(WorkflowStep.REVIEWING_CORRECTIONS)

            logger.warning(f"ANSWER flow work failed compliance: {report.format()}")

            revise_prompt = (
                f"Your changes failed compliance review. Please revise:\n\n"
                f"{report.format()}\n\n"
                f"Revision guidance: {report.revision_guidance}\n\n"
                f"Make the necessary corrections."
            )
            await self._call_inner_agent(ctx.state, revise_prompt, ctx.session_id)

            await self._thinking_status.stop("Corrections applied")

    async def handle_planning(
        self,
        ctx: WorkflowContext,
        complexity: str,
        workflow_phase_enum: type[WorkflowPhase],
    ) -> bool:
        """Handle planning phase: Create and review implementation plan.

        Args:
            ctx: Workflow context.
            complexity: Task complexity (SIMPLE, MODERATE, COMPLEX).
            workflow_phase_enum: WorkflowPhase enum class.

        Returns:
            True if plan was approved and execution should proceed.
        """
        ctx.state.phase = workflow_phase_enum.PLANNING
        logger.info("Phase 1: Planning")

        # Request YAML plan from inner agent
        plan_prompt = self._build_plan_prompt(ctx.prompt_text)

        if ctx.has_images:
            plan_prompt_content = [*ctx.prompt_content, {"type": "text", "text": plan_prompt}]
            plan_response = await self._call_inner_agent_silent(
                ctx.state, plan_prompt_content, ctx.session_id
            )
        else:
            plan_response = await self._call_inner_agent_silent(
                ctx.state, plan_prompt, ctx.session_id
            )

        # Review based on complexity
        if complexity == "SIMPLE":
            approved = await self._review_simple_plan_flow(ctx, plan_response)
            if approved:
                return True
            # Upgrade to full review if simple plan rejected
            complexity = "MODERATE"

        if not ctx.state.approved_plan:
            return await self._review_full_plan_flow(ctx, plan_response)

        return bool(ctx.state.approved_plan)

    def _build_plan_prompt(self, prompt_text: str) -> str:
        """Build the plan request prompt."""
        return f"""\
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

    async def _review_simple_plan_flow(
        self, ctx: WorkflowContext, plan_response: str
    ) -> bool:
        """Handle simple plan review flow.

        Returns True if plan was approved, False to upgrade to full review.
        """
        ctx.state.set_current_action("Preparing approach")

        await self._thinking_status.set_step(WorkflowStep.PLANNING_SIMPLE)

        report = await self._review_simple_plan(ctx.state, plan_response, ctx.session_id)

        if report.is_compliant:
            ctx.state.approved_plan = plan_response

            await self._thinking_status.stop("Ready to proceed")

            await self._parse_and_send_plan(ctx.state, plan_response, ctx.session_id)
            await self._show_plan(ctx, plan_response)
            return True

        # Simple task rejected - signal upgrade needed
        logger.info("Simple task rejected, upgrading to full review")
        await self._thinking_status.set_step(WorkflowStep.PLANNING_REFINING)
        return False

    async def _review_full_plan_flow(
        self, ctx: WorkflowContext, plan_response: str
    ) -> bool:
        """Handle full plan review flow with multiple attempts.

        Returns True if plan was approved.
        """
        classify_client = get_classify_client(self._classify_model)

        await self._thinking_status.set_step(WorkflowStep.PLANNING_FULL)

        while ctx.state.plan_review_attempts < ctx.state.max_plan_attempts:
            attempt = ctx.state.plan_review_attempts + 1
            remaining = ctx.state.max_plan_attempts - attempt

            # Update progress step based on attempt
            step = self._get_planning_step(attempt)
            await self._thinking_status.set_step(step)

            ctx.state.set_current_action(f"Planning (iteration {attempt})")
            report = await self._review_plan(ctx.state, plan_response, ctx.session_id)

            if report.is_compliant:
                ctx.state.approved_plan = plan_response

                await self._thinking_status.stop("Ready to implement")

                await self._parse_and_send_plan(ctx.state, plan_response, ctx.session_id)
                await self._show_plan(ctx, plan_response)
                return True

            # Log violations
            self._log_violations(report)

            # Check if max attempts reached
            if ctx.state.plan_review_attempts >= ctx.state.max_plan_attempts:
                await self._handle_planning_escalation(ctx, report, classify_client)
                return False

            # Request revision
            plan_response = await self._request_plan_revision(
                ctx, report, remaining
            )

        return False

    def _get_planning_step(self, attempt: int) -> WorkflowStep:
        """Get workflow step based on attempt number."""
        steps = {
            1: WorkflowStep.PLANNING_FULL,
            2: WorkflowStep.PLANNING_REFINING,
            3: WorkflowStep.PLANNING_REVISING,
        }
        return steps.get(attempt, WorkflowStep.PLANNING_FINALIZING)

    def _log_violations(self, report: ComplianceReport) -> None:
        """Log plan violations for debugging."""
        violation_lines = []
        for section in report.failing_sections:
            for g in section.guideline_scores:
                if not g.score.is_passing:
                    violation_lines.append(f"  - {g.guideline_id} ({g.score}): {g.justification}")
        logger.debug(f"Plan violations: {violation_lines}")

    async def _handle_planning_escalation(
        self,
        ctx: WorkflowContext,
        report: ComplianceReport,
        classify_client: Any,
    ) -> None:
        """Handle planning escalation when max attempts reached."""
        await self._thinking_status.stop("Need more information")

        # Build conversation context from session state
        conversation_context = self._build_conversation_context(ctx)

        # Generate clarifying questions (may return empty if context is sufficient)
        violations_for_questions = [
            f"{g.guideline_id}: {g.justification}"
            for section in report.failing_sections
            for g in section.guideline_scores
            if not g.score.is_passing
        ]
        questions = await asyncio.to_thread(
            generate_clarifying_questions,
            classify_client,
            ctx.prompt_text,
            violations_for_questions,
            conversation_context=conversation_context,
        )

        # If no questions needed, context provides sufficient information
        # The inner agent should infer intent and provide a status update
        if not questions:
            logger.info("Context sufficient - inferring intent instead of asking questions")
            infer_prompt = self._build_context_inference_prompt(ctx)
            await self._call_inner_agent(ctx.state, infer_prompt, ctx.session_id)
            return

        numbered_questions = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        escalate_msg = f"\n\n**Additional information required:**\n\n{numbered_questions}\n"
        chunk = update_agent_message(text_block(escalate_msg))
        await self._conn.session_update(session_id=ctx.session_id, update=chunk)

    def _build_conversation_context(self, ctx: WorkflowContext) -> str:
        """Build conversation context string from session state.

        Args:
            ctx: Workflow context with state containing history.

        Returns:
            Formatted string with recent tool calls and prior user requests.
        """
        parts = []

        # Include recent tool call history (last 20 actions)
        if ctx.state.tool_call_history:
            recent_tools = ctx.state.tool_call_history[-20:]
            parts.append("Recent actions:")
            parts.extend(f"  - {action}" for action in recent_tools)

        # Include prior user requests in this session
        if ctx.state.user_request_history and len(ctx.state.user_request_history) > 1:
            prior_requests = ctx.state.user_request_history[:-1][-5:]  # Last 5 prior requests
            parts.append("\nPrior requests in this session:")
            for i, req in enumerate(prior_requests, 1):
                # Truncate long requests
                truncated = req[:200] + "..." if len(req) > 200 else req
                parts.append(f"  {i}. {truncated}")

        return "\n".join(parts) if parts else ""

    def _build_context_inference_prompt(self, ctx: WorkflowContext) -> str:
        """Build prompt for inferring intent from conversation context.

        Used when the conversation context provides sufficient information
        to understand the user's intent without asking clarifying questions.

        Args:
            ctx: Workflow context with state containing history.

        Returns:
            Prompt instructing the inner agent to infer intent and respond.
        """
        context = self._build_conversation_context(ctx)

        return f"""\
The user's request was ambiguous, but the conversation context provides sufficient
information to understand their intent. Infer what the user wants based on the
recent actions and provide an appropriate response.

USER REQUEST:
{ctx.prompt_text}

CONVERSATION CONTEXT:
{context}

INSTRUCTIONS:
1. Review the recent actions to understand what work was just completed
2. Infer what the user likely wants (status update, next steps, summary, etc.)
3. Provide a helpful response that addresses their implicit question
4. If recent work was completed (commits, tests, file changes), summarize what was done
5. If there are logical next steps, suggest them

Respond directly to the user based on the context. Do not ask clarifying questions.
"""

    async def _request_plan_revision(
        self,
        ctx: WorkflowContext,
        report: ComplianceReport,
        remaining: int,
    ) -> str:
        """Request a revised plan from the inner agent."""
        violations = []
        for section in report.failing_sections:
            for g in section.guideline_scores:
                if not g.score.is_passing:
                    violations.append(f"- {g.guideline_id} ({g.score}): {g.justification}")

        urgency = ""
        if remaining <= 2:
            urgency = (
                f"\n\n⚠️ URGENT: You have {remaining} attempt(s) remaining. "
                f"Address ALL violations below or provide numbered questions "
                f"if you need clarification from the user."
            )

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
{ctx.prompt_text}

Provide your revised YAML plan now:
"""
        return await self._call_inner_agent_silent(ctx.state, revise_prompt, ctx.session_id)

    async def _show_plan(self, ctx: WorkflowContext, plan_response: str) -> None:
        """Show the approved plan to the user."""
        plan_msg = f"\n\n**Plan:**\n\n{plan_response}\n"
        chunk = update_agent_message(text_block(plan_msg))
        await self._conn.session_update(session_id=ctx.session_id, update=chunk)

    async def handle_execution(
        self,
        ctx: WorkflowContext,
        workflow_phase_enum: type[WorkflowPhase],
    ) -> None:
        """Handle execution phase: Implement the approved plan.

        Args:
            ctx: Workflow context.
            workflow_phase_enum: WorkflowPhase enum class.
        """
        logger.info("Phase 2: Execution")
        ctx.state.phase = workflow_phase_enum.EXECUTING
        ctx.state.set_current_action("Implementing the plan")

        await self._thinking_status.start(WorkflowStep.EXECUTING)

        execute_prompt = (
            f"Implement the approved plan. Proceed with the implementation now.\n\n"
            f"Plan:\n{ctx.state.approved_plan}"
        )
        work_response = await self._call_inner_agent(ctx.state, execute_prompt, ctx.session_id)
        ctx.state.work_summary = work_response

        # Update task progress
        await self._update_task_progress(ctx.state, work_response, ctx.session_id)

    async def handle_review_cycle(
        self,
        ctx: WorkflowContext,
        workflow_phase_enum: type[WorkflowPhase],
    ) -> None:
        """Handle review cycle: Check compliance and request revisions.

        Args:
            ctx: Workflow context.
            workflow_phase_enum: WorkflowPhase enum class.
        """
        work_response = ctx.state.work_summary

        while True:
            ready = await self._check_ready_for_review(ctx.state, work_response)

            if ready:
                logger.info("Phase 3: Reviewing work")
                ctx.state.phase = workflow_phase_enum.REVIEWING
                ctx.state.set_current_action("Finalizing changes")

                await self._thinking_status.set_step(WorkflowStep.REVIEWING)

                report = await self._review_work(ctx.state, ctx.state.work_summary, ctx.session_id)

                if report.is_compliant:
                    ctx.state.phase = workflow_phase_enum.COMPLETE

                    # Mark all tasks completed
                    for task in ctx.state.plan_tasks:
                        task.status = "completed"
                    await self._send_plan_update(ctx.state, ctx.session_id)

                    await self._thinking_status.stop("Done")
                    break

                # Request revision
                await self._thinking_status.set_step(WorkflowStep.REVIEWING_CORRECTIONS)
                logger.debug(f"Work review failed: {report.format()}")

                revise_prompt = (
                    f"Your work failed compliance review. Please revise:\n\n"
                    f"{report.format()}\n\n"
                    f"Revision guidance: {report.revision_guidance}\n\n"
                    f"Make the necessary changes to achieve 5/5 on all guidelines."
                )
                work_response = await self._call_inner_agent(
                    ctx.state, revise_prompt, ctx.session_id
                )
                ctx.state.work_summary += f"\n\n[Revision]\n{work_response}"

                await self._update_task_progress(ctx.state, work_response, ctx.session_id)
            else:
                # Not ready - work is still in progress
                logger.info("Work not ready for review, continuing execution")
                await self._thinking_status.stop()
                break
