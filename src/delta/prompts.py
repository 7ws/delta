"""Prompt templates and builders for compliance reviews."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from delta.guidelines import AgentsDocument


class PromptTemplate:
    """Base class for prompt templates with common formatting."""

    @staticmethod
    def format_tool_history(tool_history: list[str] | None) -> str:
        """Format tool history for prompt inclusion."""
        if tool_history:
            return "\n".join(f"- {action}" for action in tool_history)
        return "(No previous actions in this session)"

    @staticmethod
    def format_user_history(user_request_history: list[str] | None) -> str:
        """Format user request history for prompt inclusion."""
        if user_request_history and len(user_request_history) > 1:
            prior_requests = user_request_history[:-1]
            return "\n\n".join(
                f"[Message {i + 1}]\n{req}" for i, req in enumerate(prior_requests)
            )
        return "(No prior requests in this session)"

    @staticmethod
    def format_section_checklist(agents_doc: AgentsDocument) -> str:
        """Format section checklist for prompt inclusion."""
        return "\n".join(f"- §{s.number} {s.name}" for s in agents_doc.major_sections)


PLAN_REVIEW_TEMPLATE = """\
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


BATCH_WORK_REVIEW_TEMPLATE = """\
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
    section_checklist = PromptTemplate.format_section_checklist(agents_doc)

    return PLAN_REVIEW_TEMPLATE.format(
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
    section_checklist = PromptTemplate.format_section_checklist(agents_doc)
    history_text = PromptTemplate.format_tool_history(tool_history)

    return BATCH_WORK_REVIEW_TEMPLATE.format(
        agents_md_content=agents_doc.raw_content,
        user_prompt=user_prompt,
        plan=plan,
        work_summary=work_summary,
        tool_history=history_text,
        section_checklist=section_checklist,
    )
