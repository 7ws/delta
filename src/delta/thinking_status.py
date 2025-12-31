"""Thinking status indicator for Delta.

Provides real-time feedback during long-running operations by displaying
elapsed time, current step, and estimated completion time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from uuid import uuid4

from acp import start_tool_call, update_tool_call

if TYPE_CHECKING:
    from acp.interfaces import Client

logger = logging.getLogger(__name__)


class WorkflowStep(Enum):
    """Workflow steps with descriptions and time estimates."""

    TRIAGE = ("Analyzing request", 2)
    PLANNING_SIMPLE = ("Preparing", 3)
    PLANNING_FULL = ("Analyzing request", 10)
    PLANNING_REFINING = ("Refining approach", 8)
    PLANNING_REVISING = ("Revising solution", 8)
    PLANNING_FINALIZING = ("Finalizing plan", 5)
    EXECUTING = ("Implementing", 30)
    REVIEWING = ("Verifying changes", 5)
    REVIEWING_CORRECTIONS = ("Applying corrections", 10)

    def __init__(self, description: str, estimated_seconds: int) -> None:
        self._description = description
        self._estimated_seconds = estimated_seconds

    @property
    def description(self) -> str:
        return self._description

    @property
    def estimated_seconds(self) -> int:
        return self._estimated_seconds


@dataclass
class ThinkingStatus:
    """Current thinking status state."""

    step: WorkflowStep
    elapsed_seconds: float
    step_elapsed_seconds: float

    def format_title(self) -> str:
        """Format the status title with elapsed time and estimate."""
        elapsed = int(self.elapsed_seconds)
        step_elapsed = int(self.step_elapsed_seconds)
        estimate = self.step.estimated_seconds

        if step_elapsed < estimate:
            remaining = estimate - step_elapsed
            return f"{self.step.description} ({elapsed}s, ~{remaining}s remaining)"
        return f"{self.step.description} ({elapsed}s)"


class ThinkingStatusManager:
    """Manages real-time thinking status updates.

    Displays elapsed time, current step, and time estimates via periodic
    UI updates every 0.5 seconds.
    """

    UPDATE_INTERVAL = 0.5

    def __init__(
        self,
        conn: Client,
        session_id: str,
    ) -> None:
        """Initialize the thinking status manager.

        Args:
            conn: ACP client connection.
            session_id: ACP session identifier.
        """
        self._conn = conn
        self._session_id = session_id
        self._tool_call_id: str = ""
        self._start_time: float = 0.0
        self._step_start_time: float = 0.0
        self._current_step: WorkflowStep = WorkflowStep.TRIAGE
        self._update_task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time since start."""
        if self._start_time == 0.0:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def step_elapsed_seconds(self) -> float:
        """Elapsed time since current step started."""
        if self._step_start_time == 0.0:
            return 0.0
        return time.monotonic() - self._step_start_time

    def _get_status(self) -> ThinkingStatus:
        """Get current thinking status."""
        return ThinkingStatus(
            step=self._current_step,
            elapsed_seconds=self.elapsed_seconds,
            step_elapsed_seconds=self.step_elapsed_seconds,
        )

    async def start(self, step: WorkflowStep = WorkflowStep.TRIAGE) -> None:
        """Start the thinking status indicator.

        Args:
            step: Initial workflow step.
        """
        self._tool_call_id = f"thinking_{uuid4().hex[:8]}"
        self._start_time = time.monotonic()
        self._step_start_time = self._start_time
        self._current_step = step
        self._running = True

        status = self._get_status()
        tool_call = start_tool_call(
            tool_call_id=self._tool_call_id,
            title=status.format_title(),
            kind="think",
            status="in_progress",
        )
        await self._conn.session_update(session_id=self._session_id, update=tool_call)

        self._update_task = asyncio.create_task(self._update_loop())
        logger.debug(f"Started thinking status: {step.description}")

    async def set_step(self, step: WorkflowStep) -> None:
        """Update the current workflow step.

        Args:
            step: New workflow step.
        """
        if not self._running:
            return

        self._current_step = step
        self._step_start_time = time.monotonic()

        status = self._get_status()
        tool_update = update_tool_call(
            tool_call_id=self._tool_call_id,
            title=status.format_title(),
            status="in_progress",
        )
        await self._conn.session_update(session_id=self._session_id, update=tool_update)
        logger.debug(f"Updated thinking step: {step.description}")

    async def stop(self, final_title: str | None = None) -> None:
        """Stop the thinking status indicator.

        Args:
            final_title: Optional final title to display.
        """
        if not self._running:
            return

        self._running = False

        if self._update_task is not None:
            self._update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._update_task
            self._update_task = None

        title = final_title if final_title else self._current_step.description
        tool_update = update_tool_call(
            tool_call_id=self._tool_call_id,
            title=title,
            status="completed",
        )
        await self._conn.session_update(session_id=self._session_id, update=tool_update)
        logger.debug(f"Stopped thinking status: {title}")

    async def _update_loop(self) -> None:
        """Background loop that updates the status every 0.5 seconds."""
        while self._running:
            await asyncio.sleep(self.UPDATE_INTERVAL)

            if not self._running:
                break

            status = self._get_status()
            tool_update = update_tool_call(
                tool_call_id=self._tool_call_id,
                title=status.format_title(),
                status="in_progress",
            )
            try:
                await self._conn.session_update(
                    session_id=self._session_id, update=tool_update
                )
            except Exception as e:
                logger.warning(f"Failed to update thinking status: {e}")
                break
