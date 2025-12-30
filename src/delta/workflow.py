"""Workflow orchestration for Delta compliance wrapper.

This module breaks down the complex prompt() method into focused phase handlers,
making the workflow easier to understand, test, and maintain.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from acp import start_tool_call, text_block, update_agent_message, update_tool_call

from delta.compliance import ComplianceReport
from delta.llm import (
    InvalidComplexityResponse,
    InvalidTriageResponse,
    classify_task_complexity,
    generate_clarifying_questions,
    get_classify_client,
    triage_user_message,
)

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

    async def handle_direct_answer(self, ctx: WorkflowContext) -> None:
        """Handle direct answer flow (ANSWER triage result).

        Calls the inner agent directly without planning. If write operations
        occur, triggers mandatory compliance review.

        Args:
            ctx: Workflow context.
        """
        if ctx.has_images:
            response_text = await self._call_inner_agent(
                ctx.state, ctx.prompt_content, ctx.session_id
            )
        else:
            response_text = await self._call_inner_agent(
                ctx.state, ctx.prompt_text, ctx.session_id
            )

        # If writes occurred, compliance review is mandatory
        if ctx.state.has_write_operations:
            logger.info("ANSWER flow had write operations - triggering mandatory review")
            await self._review_answer_writes(ctx, response_text)

    async def _review_answer_writes(self, ctx: WorkflowContext, response_text: str) -> None:
        """Review writes that occurred during direct answer flow.

        Args:
            ctx: Workflow context.
            response_text: Response from inner agent.
        """
        ctx.state.work_summary = response_text
        ctx.state.approved_plan = f"Respond to user query:\n{ctx.prompt_text}"

        review_block_id = f"verify_{uuid4().hex[:8]}"
        review_block = start_tool_call(
            tool_call_id=review_block_id,
            title="Verifying changes",
            kind="think",
            status="in_progress",
        )
        await self._conn.session_update(session_id=ctx.session_id, update=review_block)

        report = await self._review_work(ctx.state, ctx.state.work_summary, ctx.session_id)

        if report.is_compliant:
            review_update = update_tool_call(
                tool_call_id=review_block_id,
                title="Changes verified",
                status="completed",
            )
            await self._conn.session_update(session_id=ctx.session_id, update=review_update)
        else:
            # Request revision
            review_update = update_tool_call(
                tool_call_id=review_block_id,
                title="Applying corrections",
                status="in_progress",
            )
            await self._conn.session_update(session_id=ctx.session_id, update=review_update)

            logger.warning(f"ANSWER flow work failed compliance: {report.format()}")

            revise_prompt = (
                f"Your changes failed compliance review. Please revise:\n\n"
                f"{report.format()}\n\n"
                f"Revision guidance: {report.revision_guidance}\n\n"
                f"Make the necessary corrections."
            )
            await self._call_inner_agent(ctx.state, revise_prompt, ctx.session_id)

            review_update = update_tool_call(
                tool_call_id=review_block_id,
                title="Corrections applied",
                status="completed",
            )
            await self._conn.session_update(session_id=ctx.session_id, update=review_update)

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

        review_block_id = f"plan_review_simple_{uuid4().hex[:8]}"
        review_block = start_tool_call(
            tool_call_id=review_block_id,
            title="Preparing",
            kind="think",
            status="in_progress",
        )
        await self._conn.session_update(session_id=ctx.session_id, update=review_block)

        report = await self._review_simple_plan(ctx.state, plan_response, ctx.session_id)

        if report.is_compliant:
            ctx.state.approved_plan = plan_response

            review_update = update_tool_call(
                tool_call_id=review_block_id,
                title="Ready to proceed",
                status="completed",
            )
            await self._conn.session_update(session_id=ctx.session_id, update=review_update)

            await self._parse_and_send_plan(ctx.state, plan_response, ctx.session_id)
            await self._show_plan(ctx, plan_response)
            return True

        # Simple task rejected - signal upgrade needed
        logger.info("Simple task rejected, upgrading to full review")
        review_update = update_tool_call(
            tool_call_id=review_block_id,
            title="Refining approach",
            status="in_progress",
        )
        await self._conn.session_update(session_id=ctx.session_id, update=review_update)
        return False

    async def _review_full_plan_flow(
        self, ctx: WorkflowContext, plan_response: str
    ) -> bool:
        """Handle full plan review flow with multiple attempts.

        Returns True if plan was approved.
        """
        classify_client = get_classify_client(self._classify_model)

        planning_block_id = f"planning_{uuid4().hex[:8]}"
        planning_block = start_tool_call(
            tool_call_id=planning_block_id,
            title="Analyzing request",
            kind="think",
            status="in_progress",
        )
        await self._conn.session_update(session_id=ctx.session_id, update=planning_block)

        while ctx.state.plan_review_attempts < ctx.state.max_plan_attempts:
            attempt = ctx.state.plan_review_attempts + 1
            remaining = ctx.state.max_plan_attempts - attempt

            # Update progress title
            title = self._get_planning_title(attempt)
            planning_update = update_tool_call(
                tool_call_id=planning_block_id,
                title=title,
                status="in_progress",
            )
            await self._conn.session_update(session_id=ctx.session_id, update=planning_update)

            ctx.state.set_current_action(f"Planning (iteration {attempt})")
            report = await self._review_plan(ctx.state, plan_response, ctx.session_id)

            if report.is_compliant:
                ctx.state.approved_plan = plan_response

                planning_update = update_tool_call(
                    tool_call_id=planning_block_id,
                    title="Ready to implement",
                    status="completed",
                )
                await self._conn.session_update(session_id=ctx.session_id, update=planning_update)

                await self._parse_and_send_plan(ctx.state, plan_response, ctx.session_id)
                await self._show_plan(ctx, plan_response)
                return True

            # Log violations
            self._log_violations(report)

            # Check if max attempts reached
            if ctx.state.plan_review_attempts >= ctx.state.max_plan_attempts:
                await self._handle_planning_escalation(
                    ctx, planning_block_id, report, classify_client
                )
                return False

            # Request revision
            plan_response = await self._request_plan_revision(
                ctx, report, remaining
            )

        return False

    def _get_planning_title(self, attempt: int) -> str:
        """Get user-friendly planning title based on attempt number."""
        titles = {
            1: "Analyzing request",
            2: "Refining approach",
            3: "Revising solution",
        }
        return titles.get(attempt, "Finalizing plan")

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
        planning_block_id: str,
        report: ComplianceReport,
        classify_client: Any,
    ) -> None:
        """Handle planning escalation when max attempts reached."""
        planning_update = update_tool_call(
            tool_call_id=planning_block_id,
            title="Need more information",
            status="failed",
        )
        await self._conn.session_update(session_id=ctx.session_id, update=planning_update)

        # Generate clarifying questions
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
        )

        numbered_questions = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        escalate_msg = f"\n\n**Additional information required:**\n\n{numbered_questions}\n"
        chunk = update_agent_message(text_block(escalate_msg))
        await self._conn.session_update(session_id=ctx.session_id, update=chunk)

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

                review_block_id = f"finalize_{uuid4().hex[:8]}"
                review_block = start_tool_call(
                    tool_call_id=review_block_id,
                    title="Verifying changes",
                    kind="think",
                    status="in_progress",
                )
                await self._conn.session_update(session_id=ctx.session_id, update=review_block)

                report = await self._review_work(ctx.state, ctx.state.work_summary, ctx.session_id)

                if report.is_compliant:
                    ctx.state.phase = workflow_phase_enum.COMPLETE

                    # Mark all tasks completed
                    for task in ctx.state.plan_tasks:
                        task.status = "completed"
                    await self._send_plan_update(ctx.state, ctx.session_id)

                    review_update = update_tool_call(
                        tool_call_id=review_block_id,
                        title="Done",
                        status="completed",
                    )
                    await self._conn.session_update(session_id=ctx.session_id, update=review_update)
                    break

                # Request revision
                review_update = update_tool_call(
                    tool_call_id=review_block_id,
                    title="Applying adjustments",
                    status="in_progress",
                )
                await self._conn.session_update(session_id=ctx.session_id, update=review_update)
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
                break
