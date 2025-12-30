"""Compliance checking and scoring system."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from delta.utils import extract_json

if TYPE_CHECKING:
    from delta.guidelines import AgentsDocument


class Score(Enum):
    """Compliance score for a guideline."""

    FULL = 5
    MOSTLY = 4
    PARTIAL = 3
    BARELY = 2
    NONE = 1
    NOT_APPLICABLE = 0

    def __str__(self) -> str:
        if self == Score.NOT_APPLICABLE:
            return "N/A"
        return f"{self.value}/5"

    @property
    def is_passing(self) -> bool:
        """Return True if this score passes compliance."""
        return self == Score.FULL or self == Score.NOT_APPLICABLE


@dataclass
class GuidelineScore:
    """Score for a single guideline."""

    guideline_id: str
    guideline_text: str
    score: Score
    justification: str


@dataclass
class SectionScore:
    """Aggregated score for a major section."""

    section_number: int
    section_name: str
    guideline_scores: list[GuidelineScore] = field(default_factory=list)

    @property
    def average_score(self) -> float | None:
        """Calculate average score, excluding N/A."""
        applicable: list[int] = [
            g.score.value for g in self.guideline_scores if g.score != Score.NOT_APPLICABLE
        ]
        if not applicable:
            return None
        return float(sum(applicable)) / len(applicable)

    @property
    def is_passing(self) -> bool:
        """Return True if all applicable guidelines pass."""
        return all(g.score.is_passing for g in self.guideline_scores)

    def format_score(self) -> str:
        """Format score for display."""
        avg = self.average_score
        if avg is None:
            return "N/A"
        return f"{avg:.1f}/5"


@dataclass
class ComplianceReport:
    """Full compliance report for a proposed action."""

    proposed_action: str
    section_scores: list[SectionScore] = field(default_factory=list)
    attempt_number: int = 1
    revision_guidance: str = ""

    @property
    def is_compliant(self) -> bool:
        """Return True if all sections pass compliance."""
        return all(s.is_passing for s in self.section_scores)

    @property
    def failing_sections(self) -> list[SectionScore]:
        """Return list of sections that failed compliance."""
        return [s for s in self.section_scores if not s.is_passing]

    def format(self) -> str:
        """Format report for display."""
        lines = [f"Proposed action: {self.proposed_action}"]
        for section in self.section_scores:
            avg = section.average_score
            if avg is None:
                score_str = "N/A"
                justifications = [
                    g.justification for g in section.guideline_scores if g.justification
                ]
                justification = justifications[0] if justifications else "No applicable guidelines"
            else:
                score_str = f"{avg:.1f}/5"
                failing = [g for g in section.guideline_scores if not g.score.is_passing]
                if failing:
                    justification = "; ".join(g.justification for g in failing[:2])
                else:
                    justification = "All guidelines satisfied"
            lines.append(
                f"- §{section.section_number} {section.section_name}: {score_str} ({justification})"
            )
        return "\n".join(lines)


def parse_simple_plan_response(response: str) -> ComplianceReport:
    """Parse simple plan review response.

    Args:
        response: Raw response from the simple plan reviewer.

    Returns:
        ComplianceReport with approved/rejected status.
    """
    json_str = extract_json(response)
    data = json.loads(json_str)

    report = ComplianceReport(proposed_action=data.get("reason", "Simple task"))

    is_approved = data.get("approved", False)

    section = SectionScore(section_number=0, section_name="Simple Review")
    section.guideline_scores.append(
        GuidelineScore(
            guideline_id="simple",
            guideline_text="Simple task sanity check",
            score=Score.FULL if is_approved else Score.NONE,
            justification=data.get("reason", ""),
        )
    )
    report.section_scores.append(section)

    return report


def parse_compliance_response(response: str, agents_doc: AgentsDocument) -> ComplianceReport:
    """Parse the compliance reviewer's response into a report.

    Args:
        response: Raw response from the compliance reviewer.
        agents_doc: Parsed AGENTS.md for reference.

    Returns:
        Structured compliance report.

    Raises:
        ValueError: If JSON cannot be parsed from response.
    """
    json_str = extract_json(response)
    data = json.loads(json_str)

    report = ComplianceReport(
        proposed_action=data.get("summary", "Unknown action"),
        revision_guidance=data.get("revision_guidance", ""),
    )

    # Parse sections from LLM response
    parsed_sections: dict[int, SectionScore] = {}
    for section_data in data.get("sections", []):
        section_score = SectionScore(
            section_number=section_data["number"],
            section_name=section_data["name"],
        )

        for guideline_data in section_data.get("guidelines", []):
            score_value = guideline_data.get("score")
            if score_value is None or score_value == "N/A":
                score = Score.NOT_APPLICABLE
            else:
                score = Score(int(score_value))

            section_score.guideline_scores.append(
                GuidelineScore(
                    guideline_id=guideline_data["id"],
                    guideline_text="",
                    score=score,
                    justification=guideline_data.get("justification", ""),
                )
            )

        parsed_sections[section_score.section_number] = section_score

    # Ensure ALL major sections from AGENTS.md are included
    for major_section in agents_doc.major_sections:
        if major_section.number in parsed_sections:
            report.section_scores.append(parsed_sections[major_section.number])
        else:
            # Section was skipped by reviewer - add as failed
            missing_section = SectionScore(
                section_number=major_section.number,
                section_name=major_section.name,
            )
            missing_section.guideline_scores.append(
                GuidelineScore(
                    guideline_id=f"{major_section.number}.0",
                    guideline_text="Section review",
                    score=Score.NONE,
                    justification="Section was skipped by compliance reviewer",
                )
            )
            report.section_scores.append(missing_section)

    return report


# Re-export prompt builders from prompts module for backward compatibility
from delta.prompts import (  # noqa: E402, F401
    build_batch_work_review_prompt,
    build_plan_review_prompt,
    build_simple_plan_review_prompt,
)
