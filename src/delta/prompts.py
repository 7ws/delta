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


SIMPLE_PLAN_REVIEW_TEMPLATE = """\
You are a compliance reviewer performing a quick sanity check on a SIMPLE task plan.

Simple tasks are well-understood operations: git rewrites, file renames, config changes,
standard git operations, running commands. They do not require full guideline evaluation.

## User Request

{user_prompt}

## Proposed Plan

{plan}

## Git State

{git_state}

## Quick Check

Verify these essentials only:
1. Does the plan address what the user asked for?
2. Are there obvious safety concerns (data loss, security issues)?
3. Is the approach reasonable for this type of task?
4. **Git state compliance** (CRITICAL):
   - If the working tree is dirty (uncommitted changes) and the plan modifies files,
     the plan MUST address the Git state (stash changes, commit first, or ask user)
   - If the user is NOT on main/master branch and the plan creates commits,
     the plan should acknowledge the current branch or ask the user about branch management
   - Plans that ignore dirty Git state when modifying files MUST be rejected

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

Set approved=false if the plan:
- Does not address the user's request
- Has obvious safety concerns
- Uses a clearly wrong approach
- Modifies files on a dirty working tree without addressing it
- Creates commits without acknowledging non-main branch
"""


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

Key planning guidelines to evaluate:
- §2.1: Research and understanding (read code first, search existing patterns)
- §2.2: Scope and focus (minimal changes, no unsolicited additions)
- §2.8: Documentation and test maintenance (tests AND documentation in same commit)
  - CRITICAL: If the change affects user-visible behaviour and documentation exists for the
    affected area, the plan MUST include documentation updates. "Documentation does not
    mention this feature" is NOT a valid reason to skip documentation - it means the
    documentation is incomplete and MUST be updated.
- §3: Git operations (branch naming, staging)
  - Pre-commit (§3.3) applies ONLY if configured in the project. If no pre-commit config
    exists (no .pre-commit-config.yaml, not in pyproject.toml), score N/A and skip.
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
- §3.3: Pre-commit verification (ONLY if pre-commit is configured in the project)
  - If pre-commit is NOT available (not in pyproject.toml, no .pre-commit-config.yaml),
    score N/A and skip this check entirely. Do not penalize for missing pre-commit.
  - If pre-commit IS configured and was not run, score 1/5.
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


def build_simple_plan_review_prompt(
    user_prompt: str,
    plan: str,
    git_state: str | None = None,
) -> str:
    """Build prompt for simple plan review (lightweight validation).

    Args:
        user_prompt: The user's request.
        plan: The proposed implementation plan.
        git_state: Current Git state (branch, working tree status).

    Returns:
        Formatted prompt for quick sanity check.
    """
    git_state_text = git_state or "(Git state not available)"
    return SIMPLE_PLAN_REVIEW_TEMPLATE.format(
        user_prompt=user_prompt,
        plan=plan,
        git_state=git_state_text,
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
