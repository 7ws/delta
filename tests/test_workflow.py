"""Tests for delta.workflow module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from delta.thinking_status import ThinkingStatusManager, WorkflowStep
from delta.workflow import TriageResult, WorkflowContext, WorkflowOrchestrator


class TestWorkflowContext:
    """Tests for WorkflowContext dataclass."""

    def test_creates_context(self):
        """Should create a workflow context with all required fields."""
        state = MagicMock()
        ctx = WorkflowContext(
            prompt_text="Add a button",
            prompt_content=[{"type": "text", "text": "Add a button"}],
            has_images=False,
            session_id="test-session",
            state=state,
        )

        assert ctx.prompt_text == "Add a button"
        assert ctx.session_id == "test-session"
        assert ctx.has_images is False
        assert ctx.state is state


class TestTriageResult:
    """Tests for TriageResult dataclass."""

    def test_needs_planning_false(self):
        """Should indicate no planning needed."""
        result = TriageResult(needs_planning=False)
        assert not result.needs_planning
        assert result.complexity is None

    def test_needs_planning_with_complexity(self):
        """Should include complexity when planning is needed."""
        result = TriageResult(needs_planning=True, complexity="MODERATE")
        assert result.needs_planning
        assert result.complexity == "MODERATE"


class TestWorkflowOrchestrator:
    """Tests for WorkflowOrchestrator class."""

    @pytest.fixture
    def mock_conn(self):
        """Create a mock ACP connection."""
        conn = MagicMock()
        conn.session_update = AsyncMock()
        return conn

    @pytest.fixture
    def mock_thinking_status(self):
        """Create a mock ThinkingStatusManager."""
        thinking_status = MagicMock(spec=ThinkingStatusManager)
        thinking_status.start = AsyncMock()
        thinking_status.stop = AsyncMock()
        thinking_status.set_step = AsyncMock()
        return thinking_status

    @pytest.fixture
    def mock_callbacks(self):
        """Create mock callbacks."""
        return {
            "call_inner_agent": AsyncMock(return_value="Response text"),
            "call_inner_agent_silent": AsyncMock(return_value="Plan text"),
            "review_simple_plan": AsyncMock(),
            "review_plan": AsyncMock(),
            "review_work": AsyncMock(),
            "check_ready_for_review": AsyncMock(return_value=True),
            "parse_and_send_plan": AsyncMock(),
            "send_plan_update": AsyncMock(),
            "update_task_progress": AsyncMock(),
        }

    @pytest.fixture
    def orchestrator(self, mock_conn, mock_thinking_status, mock_callbacks):
        """Create a WorkflowOrchestrator instance."""
        return WorkflowOrchestrator(
            conn=mock_conn,
            thinking_status=mock_thinking_status,
            classify_model="haiku",
            **mock_callbacks,
        )

    @pytest.fixture
    def ctx(self):
        """Create a workflow context."""
        state = MagicMock()
        state.has_write_operations = False
        state.write_blocked_for_plan = False
        state.approved_plan = ""
        state.plan_review_attempts = 0
        state.max_plan_attempts = 5
        state.work_summary = ""
        state.plan_tasks = []
        return WorkflowContext(
            prompt_text="Add a button",
            prompt_content=[{"type": "text", "text": "Add a button"}],
            has_images=False,
            session_id="test-session",
            state=state,
        )

    @pytest.mark.asyncio
    async def test_triage_answer(self, orchestrator, ctx):
        """Should return needs_planning=False for ANSWER triage."""
        with patch("delta.workflow.triage_user_message", return_value="ANSWER"):
            result = await orchestrator.triage(ctx)

        assert not result.needs_planning

    @pytest.mark.asyncio
    async def test_triage_plan(self, orchestrator, ctx):
        """Should return needs_planning=True for PLAN triage."""
        with (
            patch("delta.workflow.triage_user_message", return_value="PLAN"),
            patch("delta.workflow.classify_task_complexity", return_value="MODERATE"),
        ):
            result = await orchestrator.triage(ctx)

        assert result.needs_planning
        assert result.complexity == "MODERATE"

    @pytest.mark.asyncio
    async def test_handle_direct_answer_no_writes(self, orchestrator, ctx, mock_callbacks):
        """Should call inner agent for direct answer without writes."""
        result = await orchestrator.handle_direct_answer(ctx)

        mock_callbacks["call_inner_agent"].assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_handle_direct_answer_with_writes(
        self, orchestrator, ctx, mock_callbacks, mock_conn
    ):
        """Should trigger review when writes occurred in direct answer."""
        ctx.state.has_write_operations = True
        mock_report = MagicMock()
        mock_report.is_compliant = True
        mock_callbacks["review_work"].return_value = mock_report

        result = await orchestrator.handle_direct_answer(ctx)

        mock_callbacks["review_work"].assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_handle_direct_answer_write_blocked_for_plan(
        self, orchestrator, ctx, mock_callbacks
    ):
        """Should return False when write was blocked due to missing plan."""
        # Given
        ctx.state.write_blocked_for_plan = True

        # When
        result = await orchestrator.handle_direct_answer(ctx)

        # Then
        assert result is False
        assert ctx.state.write_blocked_for_plan is False

    @pytest.mark.asyncio
    async def test_handle_direct_answer_write_blocked_skips_review(
        self, orchestrator, ctx, mock_callbacks
    ):
        """Should not trigger review when write was blocked for plan."""
        # Given
        ctx.state.write_blocked_for_plan = True
        ctx.state.has_write_operations = False

        # When
        await orchestrator.handle_direct_answer(ctx)

        # Then
        mock_callbacks["review_work"].assert_not_called()

    def test_build_plan_prompt(self, orchestrator):
        """Should build a valid plan prompt."""
        prompt = orchestrator._build_plan_prompt("Add a button")

        assert "Add a button" in prompt
        assert "YAML" in prompt
        assert "goal:" in prompt
        assert "tasks:" in prompt

    def test_get_planning_step(self, orchestrator):
        """Given an attempt number, when getting planning step, then returns correct step."""
        assert orchestrator._get_planning_step(1) == WorkflowStep.PLANNING_FULL
        assert orchestrator._get_planning_step(2) == WorkflowStep.PLANNING_REFINING
        assert orchestrator._get_planning_step(3) == WorkflowStep.PLANNING_REVISING
        assert orchestrator._get_planning_step(4) == WorkflowStep.PLANNING_FINALIZING
        assert orchestrator._get_planning_step(5) == WorkflowStep.PLANNING_FINALIZING
