"""Review phase handling for Delta compliance workflow."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from delta.compliance import ComplianceReport, parse_compliance_response, parse_simple_plan_response
from delta.prompts import (
    build_batch_work_review_prompt,
    build_plan_review_prompt,
    build_simple_plan_review_prompt,
)
from delta.utils import parse_with_retry

if TYPE_CHECKING:
    from delta.guidelines import AgentsDocument
    from delta.llm import ClaudeCodeClient

logger = logging.getLogger(__name__)


class ReviewError(Exception):
    """Base exception for review errors."""

    pass


class ParseError(ReviewError):
    """Raised when response parsing fails after retries."""

    pass


class ReviewPhaseHandler:
    """Handles plan and work review phases.

    Consolidates review logic from DeltaAgent into a focused component
    with standardized error handling and retry logic.
    """

    def __init__(
        self,
        llm_client: ClaudeCodeClient,
        agents_doc: AgentsDocument,
        max_parse_retries: int = 3,
    ) -> None:
        """Initialize the review handler.

        Args:
            llm_client: LLM client for reviews.
            agents_doc: Parsed AGENTS.md document.
            max_parse_retries: Maximum retries for JSON parsing.
        """
        self._llm_client = llm_client
        self._agents_doc = agents_doc
        self._max_parse_retries = max_parse_retries

    async def _llm_call(self, prompt: str, system: str) -> str:
        """Make an async LLM call."""
        return await asyncio.to_thread(
            self._llm_client.complete,
            prompt=prompt,
            system=system,
        )

    async def review_simple_plan(
        self,
        user_prompt: str,
        plan: str,
        git_state: str | None = None,
    ) -> ComplianceReport:
        """Review a simple plan with lightweight validation.

        Simple tasks bypass full guideline evaluation and receive only a sanity
        check for obvious issues. Git state is checked to ensure plans do not
        modify files on dirty working trees without user consent.

        Args:
            user_prompt: The user's request.
            plan: The proposed implementation plan.
            git_state: Current Git state (branch, working tree status).

        Returns:
            ComplianceReport with approved/rejected status.

        Raises:
            ParseError: If parsing fails after retries.
        """
        prompt = build_simple_plan_review_prompt(user_prompt, plan, git_state)

        async def llm_call(p: str) -> str:
            return await self._llm_call(
                p, "You are a quick sanity checker. Output only valid JSON."
            )

        try:
            report = await parse_with_retry(
                llm_call=llm_call,
                initial_prompt=prompt,
                parser=parse_simple_plan_response,
                max_retries=self._max_parse_retries,
            )
            logger.info(f"Simple plan review: compliant={report.is_compliant}")
            return report
        except RuntimeError as e:
            raise ParseError(str(e)) from e

    async def review_plan(
        self,
        user_prompt: str,
        plan: str,
    ) -> ComplianceReport:
        """Review a plan for compliance before execution.

        Args:
            user_prompt: The user's request.
            plan: The proposed implementation plan.

        Returns:
            ComplianceReport with scores and revision guidance.

        Raises:
            ParseError: If parsing fails after retries.
        """
        prompt = build_plan_review_prompt(self._agents_doc, user_prompt, plan)

        async def llm_call(p: str) -> str:
            return await self._llm_call(
                p, "You are a strict compliance reviewer. Output only valid JSON."
            )

        def parser(response: str) -> ComplianceReport:
            return parse_compliance_response(response, self._agents_doc)

        try:
            report = await parse_with_retry(
                llm_call=llm_call,
                initial_prompt=prompt,
                parser=parser,
                max_retries=self._max_parse_retries,
            )
            logger.info(f"Plan review: compliant={report.is_compliant}")
            return report
        except RuntimeError as e:
            raise ParseError(str(e)) from e

    async def review_work(
        self,
        user_prompt: str,
        approved_plan: str,
        work_summary: str,
        tool_history: list[str],
    ) -> ComplianceReport:
        """Review accumulated work for compliance.

        Args:
            user_prompt: The user's request.
            approved_plan: The approved implementation plan.
            work_summary: Summary of work completed by the inner agent.
            tool_history: List of tool calls made during execution.

        Returns:
            ComplianceReport with scores and revision guidance.

        Raises:
            ParseError: If parsing fails after retries.
        """
        prompt = build_batch_work_review_prompt(
            self._agents_doc,
            user_prompt,
            approved_plan,
            work_summary,
            tool_history,
        )

        async def llm_call(p: str) -> str:
            return await self._llm_call(
                p, "You are a strict compliance reviewer. Output only valid JSON."
            )

        def parser(response: str) -> ComplianceReport:
            return parse_compliance_response(response, self._agents_doc)

        try:
            report = await parse_with_retry(
                llm_call=llm_call,
                initial_prompt=prompt,
                parser=parser,
                max_retries=self._max_parse_retries,
            )
            logger.info(f"Work review: compliant={report.is_compliant}")
            return report
        except RuntimeError as e:
            raise ParseError(str(e)) from e
