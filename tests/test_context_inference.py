"""Tests for context inference in clarifying questions.

These tests verify that Delta infers user intent from conversation context
instead of asking redundant clarifying questions when recent actions provide
sufficient information.
"""

from unittest.mock import MagicMock

import pytest

from delta.llm import ClaudeCodeClient, generate_clarifying_questions
from delta.workflow import WorkflowContext, WorkflowOrchestrator


class TestGenerateClarifyingQuestionsWithContext:
    """Tests for context-aware clarifying question generation."""

    def test_given_empty_context_when_questions_generated_then_returns_questions(
        self,
    ) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = '["What feature do you want?", "Where should it go?"]'

        # When
        questions = generate_clarifying_questions(
            client,
            user_request="Add something",
            violations=["2.1.1: Scope unclear"],
            conversation_context=None,
        )

        # Then
        assert len(questions) == 2
        assert "feature" in questions[0].lower()

    def test_given_sufficient_context_when_questions_generated_then_returns_empty_list(
        self,
    ) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        # AI recognizes context is sufficient and returns empty array
        client.complete.return_value = "[]"

        context = """Recent actions:
  - [ALLOWED] Execute shell command: git status
  - [ALLOWED] Execute shell command: git add core/views.py
  - [ALLOWED] Execute shell command: git commit -m "Add dashboard feature"
  - [ALLOWED] Execute shell command: git log --oneline -6"""

        # When
        questions = generate_clarifying_questions(
            client,
            user_request="What is missing? Give me an update",
            violations=["2.1.1: Request is ambiguous"],
            conversation_context=context,
        )

        # Then
        assert questions == []

    def test_given_partial_context_when_questions_generated_then_returns_targeted_questions(
        self,
    ) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = '["What specific test failures need to be addressed?"]'

        context = """Recent actions:
  - [ALLOWED] Execute shell command: pytest tests/ -v
  - [ALLOWED] Read file: tests/test_views.py"""

        # When
        questions = generate_clarifying_questions(
            client,
            user_request="Fix it",
            violations=["2.2.1: Scope unclear"],
            conversation_context=context,
        )

        # Then
        assert len(questions) == 1
        assert "test" in questions[0].lower()

    def test_given_context_with_recent_commit_when_status_requested_then_returns_empty(
        self,
    ) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = "[]"

        context = """Recent actions:
  - [ALLOWED] Execute shell command: uv run pytest tests/ -q
  - [ALLOWED] Execute shell command: git status
  - [ALLOWED] Execute shell command: git add .
  - [ALLOWED] Execute shell command: git commit -m "Add feature"
  - [ALLOWED] Execute shell command: git log --oneline -5

Prior requests in this session:
  1. Add a dark mode toggle to the settings page"""

        # When
        questions = generate_clarifying_questions(
            client,
            user_request="What's next?",
            violations=["2.1.1: Request is ambiguous"],
            conversation_context=context,
        )

        # Then
        assert questions == []


class TestBuildConversationContext:
    """Tests for building conversation context string."""

    @pytest.fixture
    def mock_callbacks(self):
        """Create mock callbacks for orchestrator."""
        from unittest.mock import AsyncMock

        return {
            "call_inner_agent": AsyncMock(return_value="Response"),
            "call_inner_agent_silent": AsyncMock(return_value="Plan"),
            "review_simple_plan": AsyncMock(),
            "review_plan": AsyncMock(),
            "review_work": AsyncMock(),
            "check_ready_for_review": AsyncMock(return_value=True),
            "parse_and_send_plan": AsyncMock(),
            "send_plan_update": AsyncMock(),
            "update_task_progress": AsyncMock(),
        }

    @pytest.fixture
    def orchestrator(self, mock_callbacks):
        """Create orchestrator with mock connection."""
        from unittest.mock import MagicMock

        conn = MagicMock()
        return WorkflowOrchestrator(conn=conn, **mock_callbacks)

    def test_given_tool_history_when_context_built_then_includes_recent_actions(
        self, orchestrator
    ) -> None:
        # Given
        state = MagicMock()
        state.tool_call_history = [
            "[ALLOWED] Read file: src/main.py",
            "[ALLOWED] Execute shell command: git status",
            "[ALLOWED] Execute shell command: git commit -m 'Fix bug'",
        ]
        state.user_request_history = ["Fix the bug in main.py"]

        ctx = WorkflowContext(
            prompt_text="What now?",
            prompt_content=[],
            has_images=False,
            session_id="test",
            state=state,
        )

        # When
        context = orchestrator._build_conversation_context(ctx)

        # Then
        assert "Recent actions:" in context
        assert "git commit" in context
        assert "git status" in context

    def test_given_user_history_when_context_built_then_includes_prior_requests(
        self, orchestrator
    ) -> None:
        # Given
        state = MagicMock()
        state.tool_call_history = []
        state.user_request_history = [
            "Add a new endpoint for user profiles",
            "Run the tests",
            "What's the status?",
        ]

        ctx = WorkflowContext(
            prompt_text="What's the status?",
            prompt_content=[],
            has_images=False,
            session_id="test",
            state=state,
        )

        # When
        context = orchestrator._build_conversation_context(ctx)

        # Then
        assert "Prior requests in this session:" in context
        assert "endpoint" in context
        assert "tests" in context

    def test_given_empty_history_when_context_built_then_returns_empty_string(
        self, orchestrator
    ) -> None:
        # Given
        state = MagicMock()
        state.tool_call_history = []
        state.user_request_history = ["Current request only"]

        ctx = WorkflowContext(
            prompt_text="Current request only",
            prompt_content=[],
            has_images=False,
            session_id="test",
            state=state,
        )

        # When
        context = orchestrator._build_conversation_context(ctx)

        # Then
        assert context == ""

    def test_given_long_tool_history_when_context_built_then_truncates_to_20(
        self, orchestrator
    ) -> None:
        # Given
        state = MagicMock()
        state.tool_call_history = [f"[ALLOWED] Action {i}" for i in range(30)]
        state.user_request_history = []

        ctx = WorkflowContext(
            prompt_text="Test",
            prompt_content=[],
            has_images=False,
            session_id="test",
            state=state,
        )

        # When
        context = orchestrator._build_conversation_context(ctx)

        # Then
        # Should include actions 10-29 (last 20)
        assert "Action 10" in context
        assert "Action 29" in context
        assert "Action 9" not in context


class TestBuildContextInferencePrompt:
    """Tests for building context inference prompt."""

    @pytest.fixture
    def mock_callbacks(self):
        """Create mock callbacks for orchestrator."""
        from unittest.mock import AsyncMock

        return {
            "call_inner_agent": AsyncMock(return_value="Response"),
            "call_inner_agent_silent": AsyncMock(return_value="Plan"),
            "review_simple_plan": AsyncMock(),
            "review_plan": AsyncMock(),
            "review_work": AsyncMock(),
            "check_ready_for_review": AsyncMock(return_value=True),
            "parse_and_send_plan": AsyncMock(),
            "send_plan_update": AsyncMock(),
            "update_task_progress": AsyncMock(),
        }

    @pytest.fixture
    def orchestrator(self, mock_callbacks):
        """Create orchestrator with mock connection."""
        from unittest.mock import MagicMock

        conn = MagicMock()
        return WorkflowOrchestrator(conn=conn, **mock_callbacks)

    def test_given_context_when_inference_prompt_built_then_includes_user_request(
        self, orchestrator
    ) -> None:
        # Given
        state = MagicMock()
        state.tool_call_history = ["[ALLOWED] git commit"]
        state.user_request_history = ["Previous request"]

        ctx = WorkflowContext(
            prompt_text="What's missing?",
            prompt_content=[],
            has_images=False,
            session_id="test",
            state=state,
        )

        # When
        prompt = orchestrator._build_context_inference_prompt(ctx)

        # Then
        assert "What's missing?" in prompt
        assert "USER REQUEST:" in prompt

    def test_given_context_when_inference_prompt_built_then_includes_instructions(
        self, orchestrator
    ) -> None:
        # Given
        state = MagicMock()
        state.tool_call_history = ["[ALLOWED] git commit"]
        state.user_request_history = []

        ctx = WorkflowContext(
            prompt_text="Update?",
            prompt_content=[],
            has_images=False,
            session_id="test",
            state=state,
        )

        # When
        prompt = orchestrator._build_context_inference_prompt(ctx)

        # Then
        assert "INSTRUCTIONS:" in prompt
        assert "Infer what the user" in prompt
        assert "Do not ask clarifying questions" in prompt
