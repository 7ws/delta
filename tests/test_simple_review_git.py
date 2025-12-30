"""Tests for Git state validation in simple plan reviews.

These tests verify that simple plans are rejected when they modify files
on a dirty Git state without addressing it, and approved when the Git state
is clean or properly handled.
"""

from unittest.mock import MagicMock

import pytest

from delta.prompts import build_simple_plan_review_prompt
from delta.protocol import capture_git_state
from delta.review import ReviewPhaseHandler


class TestBuildSimplePlanReviewPromptWithGitState:
    """Tests for simple plan review prompt with Git state."""

    def test_given_clean_git_state_when_prompt_built_then_includes_clean_status(
        self,
    ) -> None:
        # Given
        git_state = "Branch: main\nWorking tree: clean"

        # When
        prompt = build_simple_plan_review_prompt(
            user_prompt="Add a feature",
            plan="1. Add file",
            git_state=git_state,
        )

        # Then
        assert "Working tree: clean" in prompt
        assert "Branch: main" in prompt

    def test_given_dirty_git_state_when_prompt_built_then_includes_dirty_status(
        self,
    ) -> None:
        # Given
        git_state = (
            "Branch: feature/test\n"
            "Working tree: DIRTY (uncommitted changes)\n"
            "Unstaged changes: src/main.py"
        )

        # When
        prompt = build_simple_plan_review_prompt(
            user_prompt="Add a feature",
            plan="1. Edit src/main.py",
            git_state=git_state,
        )

        # Then
        assert "DIRTY" in prompt
        assert "Unstaged changes" in prompt
        assert "feature/test" in prompt

    def test_given_no_git_state_when_prompt_built_then_shows_not_available(
        self,
    ) -> None:
        # Given/When
        prompt = build_simple_plan_review_prompt(
            user_prompt="Add a feature",
            plan="1. Add file",
            git_state=None,
        )

        # Then
        assert "(Git state not available)" in prompt

    def test_prompt_includes_git_compliance_rules(self) -> None:
        # Given/When
        prompt = build_simple_plan_review_prompt(
            user_prompt="Test",
            plan="Test plan",
            git_state="Branch: main\nWorking tree: clean",
        )

        # Then
        assert "Git state compliance" in prompt
        assert "dirty working tree" in prompt.lower()
        assert "branch management" in prompt.lower()


class TestReviewSimplePlanWithGitState:
    """Tests for ReviewPhaseHandler.review_simple_plan with Git state."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        return MagicMock()

    @pytest.fixture
    def mock_agents_doc(self):
        """Create a mock AGENTS.md document."""
        doc = MagicMock()
        doc.raw_content = "# AGENTS.md\nGuidelines here"
        doc.major_sections = []
        return doc

    @pytest.fixture
    def handler(self, mock_llm_client, mock_agents_doc):
        """Create a ReviewPhaseHandler instance."""
        return ReviewPhaseHandler(mock_llm_client, mock_agents_doc)

    @pytest.mark.asyncio
    async def test_given_clean_git_state_when_plan_approved_then_returns_compliant(
        self, handler, mock_llm_client
    ) -> None:
        # Given
        mock_llm_client.complete.return_value = (
            '```json\n{"approved": true, "reason": "Plan is safe"}\n```'
        )
        git_state = "Branch: main\nWorking tree: clean"

        # When
        report = await handler.review_simple_plan(
            user_prompt="Add a feature",
            plan="1. Create new file",
            git_state=git_state,
        )

        # Then
        assert report.is_compliant

    @pytest.mark.asyncio
    async def test_given_dirty_state_ignored_when_reviewed_then_returns_non_compliant(
        self, handler, mock_llm_client
    ) -> None:
        # Given
        mock_llm_client.complete.return_value = (
            '```json\n{"approved": false, "reason": "Plan ignores dirty Git state"}\n```'
        )
        git_state = (
            "Branch: feature/test\n"
            "Working tree: DIRTY (uncommitted changes)\n"
            "Unstaged changes: src/main.py"
        )

        # When
        report = await handler.review_simple_plan(
            user_prompt="Edit src/main.py",
            plan="1. Edit src/main.py\n2. Run tests",
            git_state=git_state,
        )

        # Then
        assert not report.is_compliant

    @pytest.mark.asyncio
    async def test_given_dirty_state_addressed_when_reviewed_then_returns_compliant(
        self, handler, mock_llm_client
    ) -> None:
        # Given
        mock_llm_client.complete.return_value = (
            '```json\n{"approved": true, "reason": "Plan addresses Git state"}\n```'
        )
        git_state = (
            "Branch: main\n"
            "Working tree: DIRTY (uncommitted changes)\n"
            "Unstaged changes: src/config.py"
        )

        # When
        report = await handler.review_simple_plan(
            user_prompt="Edit config",
            plan="1. Stash current changes\n2. Edit config\n3. Pop stash",
            git_state=git_state,
        )

        # Then
        assert report.is_compliant

    @pytest.mark.asyncio
    async def test_given_feature_branch_with_commit_when_acknowledged_then_compliant(
        self, handler, mock_llm_client
    ) -> None:
        # Given
        mock_llm_client.complete.return_value = (
            '```json\n{"approved": true, "reason": "Branch acknowledged"}\n```'
        )
        git_state = "Branch: feature/new-feature\nWorking tree: clean"

        # When
        report = await handler.review_simple_plan(
            user_prompt="Commit changes",
            plan="1. Stage files\n2. Commit to current branch (feature/new-feature)",
            git_state=git_state,
        )

        # Then
        assert report.is_compliant


class TestCaptureGitState:
    """Tests for capture_git_state function.

    Note: These tests verify output format, not actual Git operations.
    """

    def test_given_git_repo_when_captured_then_includes_branch(self) -> None:
        # Given/When
        result = capture_git_state()

        # Then
        assert "Branch:" in result

    def test_given_git_repo_when_captured_then_includes_working_tree_status(
        self,
    ) -> None:
        # Given/When
        result = capture_git_state()

        # Then
        assert "Working tree:" in result or "DIRTY" in result or "clean" in result

    def test_returns_string(self) -> None:
        # Given/When
        result = capture_git_state()

        # Then
        assert isinstance(result, str)
        assert len(result) > 0
