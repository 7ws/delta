"""Compliance checking and scoring system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from delta.guidelines import AgentsDocument, MajorSection


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


SIMPLE_PLAN_REVIEW_PROMPT = """\
You are a compliance reviewer performing a quick sanity check on a SIMPLE task plan.

Simple tasks are well-understood operations: git rewrites, file renames, config changes,
standard git operations, running commands. They do not require full guideline evaluation.

## User Request

{user_prompt}

## Proposed Plan

{plan}

## Quick Check

Verify these essentials only:
1. Does the plan address what the user asked for?
2. Are there obvious safety concerns (data loss, security issues)?
3. Is the approach reasonable for this type of task?

Do NOT evaluate:
- Every guideline in AGENTS.md
- Minor stylistic concerns
- Hypothetical edge cases
- Documentation requirements for simple operations

## Required Output Format

```json
{{
  "approved": true,
  "reason": "Brief explanation (1-2 sentences)"
}}
```

Set approved=false only if the plan:
- Does not address the user's request
- Has obvious safety concerns
- Uses a clearly wrong approach
"""


LIGHTWEIGHT_REVIEW_PROMPT = """\
You are a compliance reviewer performing a quick check on a READ-ONLY operation.

Consider these guidelines:

{agents_md_content}

Now consider this action:

> {proposed_action}

This is a read-only operation (file read, search, or information gathering).
Read operations are low-risk but must still follow guidelines about:
- Reading files before proposing changes (2.1.1)
- Researching existing patterns (2.1.2)
- Not using interactive tools (2.5.1)

Score this action: Does it fully comply with applicable guidelines?

Output JSON:
```json
{{
  "score": 5,
  "compliant": true,
  "reason": "Brief explanation"
}}
```

Rules:
- Score 5 and compliant=true if the action follows guidelines
- Score below 5 and compliant=false if there is a violation
- Be concise - this is a quick check, not a full audit
"""


COMPLIANCE_REVIEW_PROMPT = """\
You are a compliance reviewer. Your ONLY task: check whether a proposed action violates
an explicit guideline in AGENTS.md.

## Your Scope (What You Evaluate)

1. Does the action contradict a specific, numbered guideline in AGENTS.md?
2. If yes, cite the guideline number and explain the violation.
3. If no explicit guideline is violated, the action is COMPLIANT.

## Outside Your Scope (Do Not Evaluate)

- Runtime success or failure (handled separately)
- Implementation strategy or efficiency
- Whether a different approach would be better
- Hypothetical problems that might occur

## Authorization Context

Prior user requests authorize related actions. If the user requested something in an
earlier message, the agent does not need to re-ask permission.

## AGENTS.md Content

{agents_md_content}

## Prior User Requests (earlier messages in this session)

{prior_user_requests}

## Current User Request (most recent message from the human)

{user_prompt}

## Previous Actions in This Session (tool calls already made)

{tool_history}

## Proposed Action (what the agent wants to do next)

{proposed_action}

## Instructions

Evaluate the proposed action against EVERY major section in AGENTS.md. For each major section:

1. Review all guidelines in that section (including minor sections)
2. Score each applicable guideline using this scale:
   - 5/5: Fully complies with the guideline
   - 4/5: Mostly complies, with minor issues
   - 3/5: Partially complies, with significant issues
   - 2/5: Barely complies, with major issues
   - 1/5: Does not comply at all
   - N/A: Not applicable to the proposed action

3. Calculate the average score for the major section (excluding N/A scores)
4. Provide a brief justification

Evaluation principles:
- Evaluate the proposed action independently of previous failures
- When guidelines conflict, the more specific guideline applies
- [RUNTIME_ERROR] entries are environmental failures, not compliance violations

## Required Output Format

You MUST output a JSON object with this exact structure:

```json
{{
  "sections": [
    {{
      "number": 1,
      "name": "Writing Style",
      "average_score": 5.0,
      "justification": "Brief explanation",
      "guidelines": [
        {{
          "id": "1.1.1",
          "score": 5,
          "justification": "Specific reason"
        }}
      ]
    }}
  ],
  "overall_compliant": true,
  "summary": "Brief overall assessment"
}}
```

Output requirements:
- Include ALL major sections from AGENTS.md (use N/A for inapplicable guidelines)
- Set overall_compliant to false if ANY section scores below 5.0 average
- Only cite explicit guideline violations, not hypothetical issues

Section checklist:
{section_checklist}
"""


PLAN_REVIEW_PROMPT = """\
You are a compliance reviewer evaluating an implementation plan.

## Your Task

Review the proposed plan against AGENTS.md guidelines. The plan must score 5/5 on all
applicable guidelines before implementation begins.

## AGENTS.md Content

{agents_md_content}

## User Request

{user_prompt}

## Proposed Plan

{plan}

## Instructions

Evaluate the plan against EVERY major section in AGENTS.md. For each section:

1. Review all guidelines that apply to planning and approach
2. Score each applicable guideline:
   - 5/5: Plan fully addresses the guideline
   - 4/5: Plan mostly addresses, minor gaps
   - 3/5: Plan partially addresses, significant gaps
   - 2/5: Plan barely addresses, major gaps
   - 1/5: Plan ignores or violates the guideline
   - N/A: Guideline does not apply to planning

3. Calculate the average score for each section
4. Provide specific feedback for any score below 5/5

Key planning guidelines to evaluate:
- §2.1: Research and understanding (read code first, search existing patterns)
- §2.2: Scope and focus (minimal changes, no unsolicited additions)
- §2.8: Documentation and test maintenance (tests AND documentation in same commit)
  - CRITICAL: If the change affects user-visible behaviour and documentation exists for the
    affected area, the plan MUST include documentation updates. "Documentation does not
    mention this feature" is NOT a valid reason to skip documentation - it means the
    documentation is incomplete and MUST be updated.
- §3: Git operations (branch naming, staging, pre-commit)
- §4: Commit messages (atomicity, format)
- §11: Testing (tests required for new features and fixes)

## Required Output Format

```json
{{
  "sections": [
    {{
      "number": 1,
      "name": "Section Name",
      "average_score": 5.0,
      "justification": "Brief explanation",
      "guidelines": [
        {{
          "id": "1.1.1",
          "score": 5,
          "justification": "Specific reason"
        }}
      ]
    }}
  ],
  "overall_compliant": true,
  "revision_guidance": "If not compliant, specific changes needed to achieve 5/5"
}}
```

Section checklist:
{section_checklist}
"""


BATCH_WORK_REVIEW_PROMPT = """\
You are a compliance reviewer evaluating completed work.

## Your Task

Review the accumulated work against AGENTS.md guidelines. All applicable guidelines
must score 5/5 for the work to pass.

## AGENTS.md Content

{agents_md_content}

## User Request

{user_prompt}

## Approved Plan

{plan}

## Work Completed

{work_summary}

## Actions Taken

{tool_history}

## Instructions

Evaluate the completed work against EVERY major section in AGENTS.md. For each section:

1. Review all guidelines that apply to the work performed
2. Score each applicable guideline:
   - 5/5: Work fully complies
   - 4/5: Work mostly complies, minor issues
   - 3/5: Work partially complies, significant issues
   - 2/5: Work barely complies, major issues
   - 1/5: Work violates the guideline
   - N/A: Guideline does not apply

3. Calculate the average score for each section
4. Provide specific feedback for any score below 5/5

Key execution guidelines to evaluate:
- §2.1: Did the agent read files before editing?
- §2.2: Are changes minimal and focused?
- §2.3: Are warnings and errors addressed?
- §2.8: Are documentation and tests updated together?
  - CRITICAL: If the change affects user-visible behaviour (UI, API, user-facing features)
    and documentation exists for the affected area, documentation MUST be updated.
    "The existing documentation does not mention this feature" is a FAILURE, not a pass.
    Missing documentation for a feature means the documentation is incomplete and must be
    fixed as part of this work. Score 1/5 if documentation updates are skipped when they
    should have been included.
- §3.2: Are files staged explicitly?
- §3.3: Did pre-commit verification pass?
- §4.4: Is the commit atomic (code + tests + documentation together)?
- §11: Are tests included for new features and fixes?

## Required Output Format

```json
{{
  "sections": [
    {{
      "number": 1,
      "name": "Section Name",
      "average_score": 5.0,
      "justification": "Brief explanation",
      "guidelines": [
        {{
          "id": "1.1.1",
          "score": 5,
          "justification": "Specific reason"
        }}
      ]
    }}
  ],
  "overall_compliant": true,
  "revision_guidance": "If not compliant, specific changes needed to achieve 5/5"
}}
```

Section checklist:
{section_checklist}
"""


SECTION_COMPLIANCE_PROMPT = """\
You are a compliance reviewer evaluating ONLY §{section_number} {section_name}.

## Your Scope

Evaluate ONLY the guidelines in §{section_number} {section_name}. Ignore all other sections.

1. Does the action contradict a specific, numbered guideline in this section?
2. If yes, cite the guideline number and explain the violation.
3. If no explicit guideline is violated, the action is COMPLIANT for this section.

## Outside Your Scope

- Guidelines from other sections (handled by parallel reviewers)
- Runtime success or failure (handled separately)
- Implementation strategy or efficiency
- Hypothetical problems that might occur

## Authorization Context

Prior user requests authorize related actions. If the user requested something in an
earlier message, the agent does not need to re-ask permission.

## Section Content (§{section_number} {section_name})

{section_content}

## Prior User Requests (earlier messages in this session)

{prior_user_requests}

## Current User Request (most recent message from the human)

{user_prompt}

## Previous Actions in This Session (tool calls already made)

{tool_history}

## Proposed Action (what the agent wants to do next)

{proposed_action}

## Instructions

Evaluate the proposed action against §{section_number} {section_name} ONLY.

1. Review all guidelines in this section (including minor sections)
2. Score each applicable guideline using this scale:
   - 5/5: Fully complies with the guideline
   - 4/5: Mostly complies, with minor issues
   - 3/5: Partially complies, with significant issues
   - 2/5: Barely complies, with major issues
   - 1/5: Does not comply at all
   - N/A: Not applicable to the proposed action

3. Calculate the average score for this section (excluding N/A scores)
4. Provide a brief justification

## Required Output Format

Output a JSON object with this exact structure:

```json
{{
  "number": {section_number},
  "name": "{section_name}",
  "average_score": 5.0,
  "justification": "Brief explanation",
  "guidelines": [
    {{
      "id": "{section_number}.X.X",
      "score": 5,
      "justification": "Specific reason"
    }}
  ]
}}
```

Output requirements:
- Evaluate ONLY guidelines in §{section_number} {section_name}
- Use N/A for inapplicable guidelines
- Be concise in justifications
"""


def build_section_compliance_prompt(
    section: MajorSection,
    section_content: str,
    proposed_action: str,
    user_prompt: str = "",
    tool_history: list[str] | None = None,
    user_request_history: list[str] | None = None,
) -> str:
    """Build the prompt for reviewing a single section.

    Args:
        section: The major section to review.
        section_content: Raw content of this section from AGENTS.md.
        proposed_action: The action to evaluate.
        user_prompt: The user's current request.
        tool_history: List of previous tool calls in the session.
        user_request_history: All user requests in this session.

    Returns:
        Formatted prompt for single-section compliance review.
    """
    # Format tool history
    if tool_history:
        history_text = "\n".join(f"- {action}" for action in tool_history)
    else:
        history_text = "(No previous actions in this session)"

    # Format user request history
    if user_request_history and len(user_request_history) > 1:
        prior_requests = user_request_history[:-1]
        prior_requests_text = "\n\n".join(
            f"[Message {i + 1}]\n{req}" for i, req in enumerate(prior_requests)
        )
    else:
        prior_requests_text = "(No prior requests in this session)"

    return SECTION_COMPLIANCE_PROMPT.format(
        section_number=section.number,
        section_name=section.name,
        section_content=section_content,
        user_prompt=user_prompt or "(No user prompt provided)",
        prior_user_requests=prior_requests_text,
        tool_history=history_text,
        proposed_action=proposed_action,
    )


def build_simple_plan_review_prompt(
    user_prompt: str,
    plan: str,
) -> str:
    """Build prompt for simple plan review (lightweight validation).

    Args:
        user_prompt: The user's request.
        plan: The proposed implementation plan.

    Returns:
        Formatted prompt for quick sanity check.
    """
    return SIMPLE_PLAN_REVIEW_PROMPT.format(
        user_prompt=user_prompt,
        plan=plan,
    )


def parse_simple_plan_response(response: str) -> ComplianceReport:
    """Parse simple plan review response.

    Args:
        response: Raw response from the simple plan reviewer.

    Returns:
        ComplianceReport with approved/rejected status.
    """
    import json
    import re

    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
        else:
            raise ValueError("No JSON found in simple plan review response")

    data = json.loads(json_str)

    report = ComplianceReport(proposed_action=data.get("reason", "Simple task"))

    # Create a single section for the simple review
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


def build_lightweight_prompt(
    agents_doc: AgentsDocument,
    proposed_action: str,
) -> str:
    """Build prompt for lightweight compliance review of read operations.

    Args:
        agents_doc: Parsed AGENTS.md document.
        proposed_action: The read operation to evaluate.

    Returns:
        Formatted prompt for quick compliance check.
    """
    return LIGHTWEIGHT_REVIEW_PROMPT.format(
        agents_md_content=agents_doc.raw_content,
        proposed_action=proposed_action,
    )


def build_compliance_prompt(
    agents_doc: AgentsDocument,
    proposed_action: str,
    user_prompt: str = "",
    tool_history: list[str] | None = None,
    user_request_history: list[str] | None = None,
) -> str:
    """Build the prompt for full compliance review.

    Args:
        agents_doc: Parsed AGENTS.md document.
        proposed_action: The action to evaluate.
        user_prompt: The user's current request that triggered this action.
        tool_history: List of previous tool calls in the session.
        user_request_history: All user requests in this session (for authorization context).

    Returns:
        Formatted prompt for the compliance reviewer.
    """
    # Build section checklist so LLM knows exactly which sections to include
    section_checklist = "\n".join(f"- §{s.number} {s.name}" for s in agents_doc.major_sections)

    # Format tool history
    if tool_history:
        history_text = "\n".join(f"- {action}" for action in tool_history)
    else:
        history_text = "(No previous actions in this session)"

    # Format user request history (prior requests that may authorize current action)
    if user_request_history and len(user_request_history) > 1:
        # Exclude current prompt (last item) since it is shown separately
        prior_requests = user_request_history[:-1]
        prior_requests_text = "\n\n".join(
            f"[Message {i + 1}]\n{req}" for i, req in enumerate(prior_requests)
        )
    else:
        prior_requests_text = "(No prior requests in this session)"

    return COMPLIANCE_REVIEW_PROMPT.format(
        agents_md_content=agents_doc.raw_content,
        user_prompt=user_prompt or "(No user prompt provided)",
        prior_user_requests=prior_requests_text,
        tool_history=history_text,
        proposed_action=proposed_action,
        section_checklist=section_checklist,
    )


def build_plan_review_prompt(
    agents_doc: AgentsDocument,
    user_prompt: str,
    plan: str,
) -> str:
    """Build the prompt for plan compliance review.

    Args:
        agents_doc: Parsed AGENTS.md document.
        user_prompt: The user's request.
        plan: The proposed implementation plan.

    Returns:
        Formatted prompt for plan review.
    """
    section_checklist = "\n".join(f"- §{s.number} {s.name}" for s in agents_doc.major_sections)

    return PLAN_REVIEW_PROMPT.format(
        agents_md_content=agents_doc.raw_content,
        user_prompt=user_prompt,
        plan=plan,
        section_checklist=section_checklist,
    )


def build_batch_work_review_prompt(
    agents_doc: AgentsDocument,
    user_prompt: str,
    plan: str,
    work_summary: str,
    tool_history: list[str] | None = None,
) -> str:
    """Build the prompt for batch work compliance review.

    Args:
        agents_doc: Parsed AGENTS.md document.
        user_prompt: The user's request.
        plan: The approved implementation plan.
        work_summary: Summary of work completed by the inner agent.
        tool_history: List of tool calls made during execution.

    Returns:
        Formatted prompt for batch work review.
    """
    section_checklist = "\n".join(f"- §{s.number} {s.name}" for s in agents_doc.major_sections)

    if tool_history:
        history_text = "\n".join(f"- {action}" for action in tool_history)
    else:
        history_text = "(No actions recorded)"

    return BATCH_WORK_REVIEW_PROMPT.format(
        agents_md_content=agents_doc.raw_content,
        user_prompt=user_prompt,
        plan=plan,
        work_summary=work_summary,
        tool_history=history_text,
        section_checklist=section_checklist,
    )


def parse_lightweight_response(response: str) -> ComplianceReport:
    """Parse lightweight compliance review response.

    Args:
        response: Raw response from the lightweight reviewer.

    Returns:
        Simplified compliance report.
    """
    import json
    import re

    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
        else:
            raise ValueError("No JSON found in lightweight compliance response")

    data = json.loads(json_str)

    report = ComplianceReport(proposed_action=data.get("reason", "Read operation"))

    # Create a single section for the lightweight review
    score_value = data.get("score", 5)
    is_compliant = data.get("compliant", True)

    section = SectionScore(section_number=0, section_name="Quick Check")
    section.guideline_scores.append(
        GuidelineScore(
            guideline_id="quick",
            guideline_text="Lightweight review",
            score=Score.FULL if is_compliant else Score(min(score_value, 4)),
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
    import json
    import re

    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
        else:
            raise ValueError("No JSON found in compliance response")

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
    # Missing sections are treated as failures (the reviewer skipped them)
    for major_section in agents_doc.major_sections:
        if major_section.number in parsed_sections:
            report.section_scores.append(parsed_sections[major_section.number])
        else:
            # Section was skipped by reviewer - add as failed
            missing_section = SectionScore(
                section_number=major_section.number,
                section_name=major_section.name,
            )
            # Add a failing score to indicate the section was not reviewed
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


def parse_section_response(response: str, section: MajorSection) -> SectionScore:
    """Parse the response from a single-section compliance review.

    Args:
        response: Raw response from the section reviewer.
        section: The major section that was reviewed.

    Returns:
        SectionScore for the reviewed section.

    Raises:
        ValueError: If JSON cannot be parsed from response.
    """
    import json
    import re

    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
        else:
            raise ValueError(f"No JSON found in section {section.number} response")

    data = json.loads(json_str)

    section_score = SectionScore(
        section_number=data.get("number", section.number),
        section_name=data.get("name", section.name),
    )

    for guideline_data in data.get("guidelines", []):
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

    # If no guidelines were parsed, add a default passing score
    if not section_score.guideline_scores:
        section_score.guideline_scores.append(
            GuidelineScore(
                guideline_id=f"{section.number}.0",
                guideline_text="Section review",
                score=Score.FULL,
                justification=data.get("justification", "No specific violations found"),
            )
        )

    return section_score
