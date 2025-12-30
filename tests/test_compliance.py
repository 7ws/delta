"""Tests for compliance checking and scoring."""

from pathlib import Path

import pytest

from delta.compliance import (
    ComplianceReport,
    Score,
    SectionScore,
    build_simple_plan_review_prompt,
    parse_compliance_response,
    parse_simple_plan_response,
)
from delta.guidelines import parse_agents_md


@pytest.fixture
def sample_agents_md(tmp_path: Path) -> Path:
    """Sample AGENTS.md file for testing."""
    content = """# 1. Writing Style

## 1.1 Voice and Tense

- 1.1.1: Use active voice.
- 1.1.2: Use simple present tense.

# 2. Technical Conduct

- 2.0.1: Read code before proposing changes.

# 3. Git Operations

- 3.0.1: Name branches as type/component/short-title.
"""
    agents_path = tmp_path / "AGENTS.md"
    agents_path.write_text(content)
    return agents_path


class TestParseComplianceResponse:
    """Tests for compliance response parsing."""

    def test_given_complete_response_when_parsed_then_returns_all_sections(
        self, sample_agents_md: Path
    ) -> None:
        # Given
        doc = parse_agents_md(sample_agents_md)
        response = """{
            "sections": [
                {"number": 1, "name": "Writing Style", "average_score": 5.0,
                 "justification": "OK", "guidelines": [
                    {"id": "1.1.1", "score": 5, "justification": "OK"}
                ]},
                {"number": 2, "name": "Technical Conduct", "average_score": 5.0,
                 "justification": "OK", "guidelines": [
                    {"id": "2.0.1", "score": 5, "justification": "OK"}
                ]},
                {"number": 3, "name": "Git Operations", "average_score": null,
                 "justification": "N/A", "guidelines": [
                    {"id": "3.0.1", "score": "N/A", "justification": "Not applicable"}
                ]}
            ],
            "overall_compliant": true,
            "summary": "Action complies"
        }"""

        # When
        report = parse_compliance_response(response, doc)

        # Then
        assert len(report.section_scores) == 3
        assert report.section_scores[0].section_number == 1
        assert report.section_scores[1].section_number == 2
        assert report.section_scores[2].section_number == 3

    def test_given_response_missing_section_when_parsed_then_adds_failing_section(
        self, sample_agents_md: Path
    ) -> None:
        # Given
        doc = parse_agents_md(sample_agents_md)
        # Response is missing section 2 (Technical Conduct)
        response = """{
            "sections": [
                {"number": 1, "name": "Writing Style", "average_score": 5.0,
                 "justification": "OK", "guidelines": [
                    {"id": "1.1.1", "score": 5, "justification": "OK"}
                ]},
                {"number": 3, "name": "Git Operations", "average_score": null,
                 "justification": "N/A", "guidelines": [
                    {"id": "3.0.1", "score": "N/A", "justification": "Not applicable"}
                ]}
            ],
            "overall_compliant": true,
            "summary": "Action complies"
        }"""

        # When
        report = parse_compliance_response(response, doc)

        # Then
        assert len(report.section_scores) == 3
        # Sections should be in order from AGENTS.md
        assert report.section_scores[0].section_number == 1
        assert report.section_scores[1].section_number == 2
        assert report.section_scores[2].section_number == 3
        # Missing section should fail
        missing_section = report.section_scores[1]
        assert missing_section.section_name == "Technical Conduct"
        assert not missing_section.is_passing
        assert missing_section.guideline_scores[0].score == Score.NONE
        assert "skipped" in missing_section.guideline_scores[0].justification.lower()

    def test_given_response_missing_multiple_sections_when_parsed_then_all_missing_fail(
        self, sample_agents_md: Path
    ) -> None:
        # Given
        doc = parse_agents_md(sample_agents_md)
        # Response is missing sections 1 and 3
        response = """{
            "sections": [
                {"number": 2, "name": "Technical Conduct", "average_score": 5.0,
                 "justification": "OK", "guidelines": [
                    {"id": "2.0.1", "score": 5, "justification": "OK"}
                ]}
            ],
            "overall_compliant": true,
            "summary": "Action complies"
        }"""

        # When
        report = parse_compliance_response(response, doc)

        # Then
        assert len(report.section_scores) == 3
        # Section 1 should fail (missing)
        assert not report.section_scores[0].is_passing
        # Section 2 should pass (present)
        assert report.section_scores[1].is_passing
        # Section 3 should fail (missing)
        assert not report.section_scores[2].is_passing
        # Overall report should not be compliant
        assert not report.is_compliant

    def test_given_empty_sections_when_parsed_then_all_sections_fail(
        self, sample_agents_md: Path
    ) -> None:
        # Given
        doc = parse_agents_md(sample_agents_md)
        response = """{
            "sections": [],
            "overall_compliant": true,
            "summary": "Empty"
        }"""

        # When
        report = parse_compliance_response(response, doc)

        # Then
        assert len(report.section_scores) == 3
        assert not report.is_compliant
        for section in report.section_scores:
            assert not section.is_passing


class TestComplianceReportFormat:
    """Tests for compliance report formatting."""

    def test_given_report_with_sections_when_formatted_then_includes_all_sections(
        self,
    ) -> None:
        # Given
        report = ComplianceReport(proposed_action="Test action")
        report.section_scores = [
            SectionScore(section_number=1, section_name="Writing Style"),
            SectionScore(section_number=2, section_name="Technical Conduct"),
        ]

        # When
        formatted = report.format()

        # Then
        assert "§1 Writing Style" in formatted
        assert "§2 Technical Conduct" in formatted


class TestBuildSimplePlanReviewPrompt:
    """Tests for simple plan review prompt generation."""

    def test_given_user_prompt_and_plan_when_built_then_includes_both(self) -> None:
        # Given
        user_prompt = "Rewrite git history with new author"
        plan = "goal: Rewrite commits\ntasks:\n  - Run git filter-repo"

        # When
        prompt = build_simple_plan_review_prompt(user_prompt, plan)

        # Then
        assert user_prompt in prompt
        assert plan in prompt
        assert "SIMPLE task" in prompt
        assert "sanity check" in prompt


class TestParseSimplePlanResponse:
    """Tests for simple plan review response parsing."""

    def test_given_approved_response_when_parsed_then_report_is_compliant(self) -> None:
        # Given
        response = '{"approved": true, "reason": "Plan addresses the request"}'

        # When
        report = parse_simple_plan_response(response)

        # Then
        assert report.is_compliant
        assert len(report.section_scores) == 1
        assert report.section_scores[0].section_name == "Simple Review"

    def test_given_rejected_response_when_parsed_then_report_is_not_compliant(
        self,
    ) -> None:
        # Given
        response = '{"approved": false, "reason": "Plan has safety concerns"}'

        # When
        report = parse_simple_plan_response(response)

        # Then
        assert not report.is_compliant
        assert report.section_scores[0].guideline_scores[0].score == Score.NONE

    def test_given_json_in_code_block_when_parsed_then_extracts_correctly(self) -> None:
        # Given
        response = """```json
{"approved": true, "reason": "Looks good"}
```"""

        # When
        report = parse_simple_plan_response(response)

        # Then
        assert report.is_compliant

    def test_given_invalid_json_when_parsed_then_raises_error(self) -> None:
        # Given
        response = "This is not JSON"

        # When/Then
        with pytest.raises(ValueError, match="No JSON found"):
            parse_simple_plan_response(response)


