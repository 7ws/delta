"""Tests for delta.review module."""

from unittest.mock import MagicMock

import pytest

from delta.compliance import ComplianceReport
from delta.review import ParseError, ReviewPhaseHandler


class TestReviewPhaseHandler:
    """Tests for ReviewPhaseHandler class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        return client

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
    async def test_review_simple_plan_compliant(self, handler, mock_llm_client):
        """Should return compliant report for approved simple plan."""
        mock_llm_client.complete.return_value = (
            '```json\n{"approved": true, "reason": "Looks good"}\n```'
        )

        report = await handler.review_simple_plan("Add a button", "1. Add button to UI")

        assert report.is_compliant

    @pytest.mark.asyncio
    async def test_review_simple_plan_not_compliant(self, handler, mock_llm_client):
        """Should return non-compliant report for rejected simple plan."""
        mock_llm_client.complete.return_value = (
            '```json\n{"approved": false, "reason": "Too risky"}\n```'
        )

        report = await handler.review_simple_plan("Delete everything", "1. rm -rf /")

        assert not report.is_compliant

    @pytest.mark.asyncio
    async def test_review_plan_success(self, handler, mock_llm_client, mock_agents_doc):
        """Should parse and return compliance report."""
        mock_llm_client.complete.return_value = '''```json
{
    "sections": [
        {
            "number": 1,
            "name": "Test Section",
            "average_score": 5.0,
            "guidelines": [
                {"id": "1.1", "score": 5, "justification": "Good"}
            ]
        }
    ],
    "overall_compliant": true,
    "revision_guidance": ""
}
```'''

        report = await handler.review_plan("Add feature", "1. Implement feature")

        assert isinstance(report, ComplianceReport)

    @pytest.mark.asyncio
    async def test_review_work_success(self, handler, mock_llm_client):
        """Should review completed work."""
        mock_llm_client.complete.return_value = '''```json
{
    "sections": [
        {
            "number": 1,
            "name": "Test",
            "average_score": 5.0,
            "guidelines": [{"id": "1.1", "score": 5, "justification": "Done"}]
        }
    ],
    "overall_compliant": true,
    "revision_guidance": ""
}
```'''

        report = await handler.review_work(
            user_prompt="Add tests",
            approved_plan="1. Write tests",
            work_summary="Added test file",
            tool_history=["Write file: test.py"],
        )

        assert isinstance(report, ComplianceReport)


class TestParseError:
    """Tests for ParseError exception."""

    def test_is_exception(self):
        """Should be an exception."""
        error = ParseError("Failed to parse")
        assert isinstance(error, Exception)
        assert "Failed to parse" in str(error)
