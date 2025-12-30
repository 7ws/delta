"""Tests for delta.workflow module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from delta.workflow import WorkflowContext, WorkflowOrchestrator, TriageResult


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
    def orchestrator(self, mock_conn, mock_callbacks):
        """Create a WorkflowOrchestrator instance."""
        return WorkflowOrchestrator(
            conn=mock_conn,
            classify_model="haiku",
            **mock_callbacks,
        )

    @pytest.fixture
    def ctx(self):
        """Create a workflow context."""
        state = MagicMock()
        state.has_write_operations = False
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
        with patch("delta.workflow.triage_user_message", return_value="PLAN"):
            with patch("delta.workflow.classify_task_complexity", return_value="MODERATE"):
                result = await orchestrator.triage(ctx)

        assert result.needs_planning
        assert result.complexity == "MODERATE"

    @pytest.mark.asyncio
    async def test_handle_direct_answer_no_writes(self, orchestrator, ctx, mock_callbacks):
        """Should call inner agent for direct answer without writes."""
        await orchestrator.handle_direct_answer(ctx)

        mock_callbacks["call_inner_agent"].assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_direct_answer_with_writes(self, orchestrator, ctx, mock_callbacks, mock_conn):
        """Should trigger review when writes occurred in direct answer."""
        ctx.state.has_write_operations = True
        mock_report = MagicMock()
        mock_report.is_compliant = True
        mock_callbacks["review_work"].return_value = mock_report

        await orchestrator.handle_direct_answer(ctx)

        mock_callbacks["review_work"].assert_called_once()

    def test_build_plan_prompt(self, orchestrator):
        """Should build a valid plan prompt."""
        prompt = orchestrator._build_plan_prompt("Add a button")

        assert "Add a button" in prompt
        assert "YAML" in prompt
        assert "goal:" in prompt
        assert "tasks:" in prompt

    def test_get_planning_title(self, orchestrator):
        """Should return appropriate titles for each attempt."""
        assert orchestrator._get_planning_title(1) == "Analyzing request"
        assert orchestrator._get_planning_title(2) == "Refining approach"
        assert orchestrator._get_planning_title(3) == "Revising solution"
        assert orchestrator._get_planning_title(4) == "Finalizing plan"
        assert orchestrator._get_planning_title(5) == "Finalizing plan"
