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
        with patch("delta.workflow.triage_user_message", return_value="PLAN"):
            result = await orchestrator.triage(ctx)

        assert result.needs_planning

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


class TestHandlePlanningEscalation:
    """Tests for _handle_planning_escalation status transitions."""

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
            "call_inner_agent": AsyncMock(return_value="Inferred response"),
            "call_inner_agent_silent": AsyncMock(return_value="Plan text"),
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
        """Create a workflow context for escalation tests."""
        state = MagicMock()
        state.tool_call_history = []
        state.user_request_history = []
        return WorkflowContext(
            prompt_text="Add a button",
            prompt_content=[{"type": "text", "text": "Add a button"}],
            has_images=False,
            session_id="test-session",
            state=state,
        )

    @pytest.fixture
    def mock_report(self):
        """Create a mock compliance report with violations."""
        report = MagicMock()
        report.failing_sections = []
        return report

    @pytest.mark.asyncio
    async def test_escalation_sets_planning_escalating_step(
        self, orchestrator, ctx, mock_report, mock_thinking_status
    ):
        """Given escalation, when handling, then sets PLANNING_ESCALATING step."""
        # Given
        classify_client = MagicMock()

        # When
        with patch("delta.workflow.generate_clarifying_questions", return_value=[]):
            await orchestrator._handle_planning_escalation(
                ctx, mock_report, classify_client
            )

        # Then
        mock_thinking_status.set_step.assert_called_with(WorkflowStep.PLANNING_ESCALATING)

    @pytest.mark.asyncio
    async def test_escalation_stops_status_after_questions(
        self, orchestrator, ctx, mock_report, mock_thinking_status
    ):
        """Given questions generated, when escalation completes, then stops status."""
        # Given
        classify_client = MagicMock()
        questions = ["What feature do you want?", "Where should it go?"]

        # When
        with patch("delta.workflow.generate_clarifying_questions", return_value=questions):
            await orchestrator._handle_planning_escalation(
                ctx, mock_report, classify_client
            )

        # Then
        mock_thinking_status.stop.assert_called_with("Need more information")

    @pytest.mark.asyncio
    async def test_escalation_stops_status_after_inference(
        self, orchestrator, ctx, mock_report, mock_thinking_status
    ):
        """Given no questions needed, when escalation completes, then stops status."""
        # Given
        classify_client = MagicMock()

        # When
        with patch("delta.workflow.generate_clarifying_questions", return_value=[]):
            await orchestrator._handle_planning_escalation(
                ctx, mock_report, classify_client
            )

        # Then
        mock_thinking_status.stop.assert_called_with("Information gathered")


class TestHandleReviewCycleCommitCheck:
    """Tests for commit check in handle_review_cycle."""

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
            "call_inner_agent": AsyncMock(return_value="Committed changes"),
            "call_inner_agent_silent": AsyncMock(return_value="Plan text"),
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
    def ctx_with_compliant_work(self):
        """Create a workflow context with compliant work."""
        state = MagicMock()
        state.has_write_operations = True
        state.write_blocked_for_plan = False
        state.approved_plan = "Test plan"
        state.plan_review_attempts = 0
        state.max_plan_attempts = 5
        state.work_summary = "Test work"
        state.plan_tasks = [MagicMock(status="in_progress")]
        state.skip_commit_check = False
        state.cwd = None
        return WorkflowContext(
            prompt_text="Add a button",
            prompt_content=[{"type": "text", "text": "Add a button"}],
            has_images=False,
            session_id="test-session",
            state=state,
        )

    @pytest.mark.asyncio
    async def test_given_dirty_tree_after_execution_when_review_cycle_then_requests_commit(
        self, orchestrator, ctx_with_compliant_work, mock_callbacks
    ):
        """Given dirty tree and compliant report, When review_cycle, Then requests commit."""
        # Given
        mock_report = MagicMock()
        mock_report.is_compliant = True
        mock_callbacks["review_work"].return_value = mock_report

        # Simulate: first call dirty, second call clean (after commit)
        clean_call_count = [0]

        def mock_is_clean(cwd=None):
            clean_call_count[0] += 1
            return clean_call_count[0] > 1

        # When
        with patch("delta.workflow.is_working_tree_clean", side_effect=mock_is_clean):
            from delta.acp_server import WorkflowPhase
            await orchestrator.handle_review_cycle(ctx_with_compliant_work, WorkflowPhase)

        # Then
        assert mock_callbacks["call_inner_agent"].call_count >= 1
        call_args = mock_callbacks["call_inner_agent"].call_args_list[0]
        assert "uncommitted changes remain" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_given_clean_tree_after_execution_when_review_cycle_then_marks_tasks_completed(
        self, orchestrator, ctx_with_compliant_work, mock_callbacks
    ):
        """Given clean tree and compliant report, When review_cycle, Then marks tasks completed."""
        # Given
        mock_report = MagicMock()
        mock_report.is_compliant = True
        mock_callbacks["review_work"].return_value = mock_report

        # When
        with patch("delta.workflow.is_working_tree_clean", return_value=True):
            from delta.acp_server import WorkflowPhase
            await orchestrator.handle_review_cycle(ctx_with_compliant_work, WorkflowPhase)

        # Then
        assert ctx_with_compliant_work.state.plan_tasks[0].status == "completed"

    @pytest.mark.asyncio
    async def test_given_skip_commit_flag_when_dirty_tree_then_allows_completion(
        self, orchestrator, ctx_with_compliant_work, mock_callbacks
    ):
        """Given skip_commit_check True and dirty tree, When review_cycle, Then completes."""
        # Given
        ctx_with_compliant_work.state.skip_commit_check = True
        mock_report = MagicMock()
        mock_report.is_compliant = True
        mock_callbacks["review_work"].return_value = mock_report

        # When
        with patch("delta.workflow.is_working_tree_clean", return_value=False):
            from delta.acp_server import WorkflowPhase
            await orchestrator.handle_review_cycle(ctx_with_compliant_work, WorkflowPhase)

        # Then
        assert ctx_with_compliant_work.state.plan_tasks[0].status == "completed"
