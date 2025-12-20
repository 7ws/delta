"""Agent wrapper that enforces AGENTS.md compliance."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from delta.compliance import (
    ComplianceReport,
    build_compliance_prompt,
    parse_compliance_response,
)
from delta.guidelines import AgentsDocument, find_agents_md, parse_agents_md


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt to the LLM and return the response."""
        ...


@dataclass
class ComplianceState:
    """Track compliance state across attempts."""

    max_attempts: int = 2
    current_attempt: int = 0
    blocked: bool = False
    previous_reports: list[ComplianceReport] = field(default_factory=list)

    def record_attempt(self, report: ComplianceReport) -> None:
        """Record a compliance attempt."""
        self.current_attempt += 1
        self.previous_reports.append(report)
        if self.current_attempt >= self.max_attempts and not report.is_compliant:
            self.blocked = True

    def reset(self) -> None:
        """Reset state for a new action."""
        self.current_attempt = 0
        self.blocked = False
        self.previous_reports.clear()


@dataclass
class ActionProposal:
    """A proposed action from the agent."""

    action_type: str
    description: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_imperative(self) -> str:
        """Format action in imperative form for compliance review."""
        return f"{self.action_type}: {self.description}"


class DeltaWrapper:
    """Wrapper that enforces AGENTS.md compliance on agent actions."""

    def __init__(
        self,
        llm_client: LLMClient,
        agents_md_path: Path | None = None,
        max_attempts: int = 2,
    ) -> None:
        """Initialize the wrapper.

        Args:
            llm_client: LLM client for compliance reviews.
            agents_md_path: Path to AGENTS.md. Auto-detected if None.
            max_attempts: Maximum compliance attempts before blocking.
        """
        self.llm_client = llm_client
        self.max_attempts = max_attempts

        if agents_md_path is None:
            agents_md_path = find_agents_md()
        if agents_md_path is None:
            raise FileNotFoundError("AGENTS.md not found")

        self.agents_md_path = agents_md_path
        self.agents_doc: AgentsDocument | None = None
        self.state = ComplianceState(max_attempts=max_attempts)
        self._initialized = False

    def initialize(self) -> str:
        """Initialize the wrapper and return the welcome message.

        Returns:
            Welcome message acknowledging AGENTS.md.
        """
        self.agents_doc = parse_agents_md(self.agents_md_path)
        self._initialized = True

        section_names = self.agents_doc.get_section_names()
        sections_list = "\n".join(f"  - {name}" for name in section_names)

        return (
            "I have read and understood AGENTS.md. "
            "I will ensure all my actions comply with its guidelines.\n\n"
            f"Loaded {len(self.agents_doc.major_sections)} major sections:\n{sections_list}"
        )

    def _reload_agents_md(self) -> None:
        """Reload AGENTS.md from disk.

        This ensures guidelines are always fresh and not from context.
        """
        self.agents_doc = parse_agents_md(self.agents_md_path)

    def review_action(self, proposal: ActionProposal) -> ComplianceReport:
        """Review a proposed action for compliance.

        Args:
            proposal: The action to review.

        Returns:
            Compliance report for the action.
        """
        if not self._initialized or self.agents_doc is None:
            raise RuntimeError("Wrapper not initialized. Call initialize() first.")

        self._reload_agents_md()

        if self.agents_doc is None:
            raise RuntimeError("Failed to load AGENTS.md")

        prompt = build_compliance_prompt(self.agents_doc, proposal.to_imperative())

        response = self.llm_client.complete(
            prompt=prompt,
            system="You are a strict compliance reviewer. Output only valid JSON.",
        )

        report = parse_compliance_response(response, self.agents_doc)
        report.proposed_action = proposal.to_imperative()
        report.attempt_number = self.state.current_attempt + 1

        self.state.record_attempt(report)

        return report

    def can_proceed(self) -> bool:
        """Check if the agent can propose more actions.

        Returns:
            True if not blocked, False otherwise.
        """
        return not self.state.blocked

    def get_rejection_context(self) -> str:
        """Get context for why an action was rejected.

        Returns:
            Formatted context including previous reports.
        """
        if not self.state.previous_reports:
            return ""

        lines = ["Previous compliance failures:"]
        for i, report in enumerate(self.state.previous_reports, 1):
            lines.append(f"\nAttempt {i}:")
            lines.append(report.format())
            lines.append("\nFailing sections:")
            for section in report.failing_sections:
                lines.append(f"  - §{section.section_number} {section.section_name}")
                for g in section.guideline_scores:
                    if not g.score.is_passing:
                        lines.append(f"    - {g.guideline_id}: {g.justification}")

        return "\n".join(lines)

    def reset_for_new_action(self) -> None:
        """Reset state for a new action sequence."""
        self.state.reset()

    def format_blocked_message(self) -> str:
        """Format message when agent is blocked.

        Returns:
            Message explaining the block and requesting user clarification.
        """
        return (
            "BLOCKED: Unable to propose a compliant action after "
            f"{self.max_attempts} attempts.\n\n"
            f"{self.get_rejection_context()}\n\n"
            "Please provide clarification on how to proceed while "
            "maintaining compliance with AGENTS.md guidelines."
        )


SYSTEM_PROMPT_TEMPLATE = """\
You are an AI assistant wrapped by Delta, a compliance enforcement system.

CRITICAL PROTOCOL:
1. Before EVERY action, you MUST read AGENTS.md completely
2. Every action you propose will be reviewed for compliance
3. If an action fails compliance, you must propose an alternative
4. After {max_attempts} failed attempts, you will be blocked until the user clarifies

AGENTS.md Location: {agents_md_path}

Your first message must be:
"I have read and understood AGENTS.md. I will ensure all my actions comply with its guidelines."

For every action you propose, expect a compliance review in this format:
```
Proposed action: <your action>
- §1 <Section>: <score>/5 (<justification>)
- §2 <Section>: <score>/5 (<justification>)
...
```

If ANY section scores below 5/5, propose a different action addressing the compliance issues.

Remember: A slow correct result is better than a fast incorrect one.
"""


def create_system_prompt(agents_md_path: Path, max_attempts: int = 2) -> str:
    """Create the system prompt for the wrapped agent.

    Args:
        agents_md_path: Path to AGENTS.md.
        max_attempts: Maximum compliance attempts.

    Returns:
        Formatted system prompt.
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        agents_md_path=agents_md_path,
        max_attempts=max_attempts,
    )
