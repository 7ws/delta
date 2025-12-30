"""Tests for delta.prompts module."""

import pytest
from unittest.mock import MagicMock

from delta.prompts import (
    PromptTemplate,
    build_simple_plan_review_prompt,
    build_plan_review_prompt,
    build_batch_work_review_prompt,
)


class TestPromptTemplate:
    """Tests for PromptTemplate helper methods."""

    def test_format_tool_history_with_items(self):
        """Should format tool history as bullet list."""
        history = ["Read file: test.py", "Edit file: main.py"]
        result = PromptTemplate.format_tool_history(history)
        assert "- Read file: test.py" in result
        assert "- Edit file: main.py" in result

    def test_format_tool_history_empty(self):
        """Should return placeholder for empty history."""
        result = PromptTemplate.format_tool_history(None)
        assert "No previous actions" in result

        result = PromptTemplate.format_tool_history([])
        assert "No previous actions" in result

    def test_format_user_history_with_multiple(self):
        """Should format prior user requests."""
        history = ["First request", "Second request", "Current request"]
        result = PromptTemplate.format_user_history(history)
        assert "[Message 1]" in result
        assert "First request" in result
        assert "[Message 2]" in result
        assert "Second request" in result
        # Current request should not be in prior requests
        assert "Current request" not in result

    def test_format_user_history_single(self):
        """Should return placeholder for single request."""
        result = PromptTemplate.format_user_history(["Only request"])
        assert "No prior requests" in result


class TestBuildSimplePlanReviewPrompt:
    """Tests for build_simple_plan_review_prompt function."""

    def test_includes_user_prompt(self):
        """Should include user prompt in output."""
        result = build_simple_plan_review_prompt("Add a button", "1. Add button")
        assert "Add a button" in result

    def test_includes_plan(self):
        """Should include plan in output."""
        result = build_simple_plan_review_prompt("Request", "My plan here")
        assert "My plan here" in result

    def test_requests_json_output(self):
        """Should request JSON format."""
        result = build_simple_plan_review_prompt("Request", "Plan")
        assert "json" in result.lower()
        assert "approved" in result.lower()


class TestBuildPlanReviewPrompt:
    """Tests for build_plan_review_prompt function."""

    def test_includes_agents_doc_content(self):
        """Should include AGENTS.md content."""
        agents_doc = MagicMock()
        agents_doc.raw_content = "# AGENTS.md Guidelines"
        agents_doc.major_sections = []

        result = build_plan_review_prompt(agents_doc, "Request", "Plan")
        assert "# AGENTS.md Guidelines" in result

    def test_includes_user_prompt(self):
        """Should include user prompt."""
        agents_doc = MagicMock()
        agents_doc.raw_content = "Content"
        agents_doc.major_sections = []

        result = build_plan_review_prompt(agents_doc, "Add feature X", "Plan")
        assert "Add feature X" in result

    def test_includes_section_checklist(self):
        """Should include section checklist."""
        section = MagicMock()
        section.number = 2
        section.name = "Code Standards"
        agents_doc = MagicMock()
        agents_doc.raw_content = "Content"
        agents_doc.major_sections = [section]

        result = build_plan_review_prompt(agents_doc, "Request", "Plan")
        assert "§2 Code Standards" in result


class TestBuildBatchWorkReviewPrompt:
    """Tests for build_batch_work_review_prompt function."""

    def test_includes_all_context(self):
        """Should include all review context."""
        agents_doc = MagicMock()
        agents_doc.raw_content = "Guidelines"
        agents_doc.major_sections = []

        result = build_batch_work_review_prompt(
            agents_doc,
            user_prompt="Add tests",
            plan="1. Write tests",
            work_summary="Added test file",
            tool_history=["Write file: test.py"],
        )

        assert "Add tests" in result
        assert "1. Write tests" in result
        assert "Added test file" in result
        assert "Write file: test.py" in result
