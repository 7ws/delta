"""Tests for delta.prompts module."""

from unittest.mock import MagicMock

from delta.prompts import (
    PromptTemplate,
    build_batch_work_review_prompt,
    build_plan_review_prompt,
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


class TestTemplateCompliance:
    """Tests for template compliance with AGENTS.md guidelines."""

    def test_plan_review_no_section_highlighting(self):
        """PLAN_REVIEW_TEMPLATE must not contain section-specific highlighting."""
        # Given
        from delta.prompts import PLAN_REVIEW_TEMPLATE

        # Then
        assert "Key planning guidelines" not in PLAN_REVIEW_TEMPLATE
        assert "Key guidelines" not in PLAN_REVIEW_TEMPLATE

    def test_batch_review_no_section_highlighting(self):
        """BATCH_WORK_REVIEW_TEMPLATE must not contain section-specific highlighting."""
        # Given
        from delta.prompts import BATCH_WORK_REVIEW_TEMPLATE

        # Then
        assert "Key execution guidelines" not in BATCH_WORK_REVIEW_TEMPLATE
        assert "Key guidelines" not in BATCH_WORK_REVIEW_TEMPLATE

    def test_no_inclusive_language_violations(self):
        """Templates must not contain inclusive language violations per section 1.5.2."""
        # Given
        from delta.prompts import (
            BATCH_WORK_REVIEW_TEMPLATE,
            PLAN_REVIEW_TEMPLATE,
        )

        forbidden_terms = [
            "sanity check",
            "sanity test",
            "whitelist",
            "blacklist",
            "master/slave",
            "man hours",
        ]

        templates = [
            ("PLAN_REVIEW_TEMPLATE", PLAN_REVIEW_TEMPLATE),
            ("BATCH_WORK_REVIEW_TEMPLATE", BATCH_WORK_REVIEW_TEMPLATE),
        ]

        # Then
        for template_name, template_content in templates:
            template_lower = template_content.lower()
            for term in forbidden_terms:
                assert term not in template_lower, (
                    f"{template_name} contains forbidden term: {term}"
                )
