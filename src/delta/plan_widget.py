"""Plan widget management for Delta."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from acp import plan_entry, update_plan

from delta.llm import (
    InvalidPlanParseResponse,
    InvalidTaskDuplicateResponse,
    classify_task_duplicate,
    detect_task_progress,
    get_classify_client,
    parse_plan_tasks,
)

if TYPE_CHECKING:
    from acp.interfaces import Client

logger = logging.getLogger(__name__)


@dataclass
class PlanTask:
    """A single task from the approved plan."""

    content: str
    status: Literal["pending", "in_progress", "completed"] = "pending"


class PlanWidgetManager:
    """Manages plan task state and UI updates.

    Consolidates plan widget logic from the DeltaAgent into a focused component.
    """

    def __init__(
        self,
        conn: Client,
        session_id: str,
        classify_model: str = "haiku",
    ) -> None:
        """Initialize the plan widget manager.

        Args:
            conn: ACP client connection.
            session_id: ACP session identifier.
            classify_model: Model for task classification.
        """
        self._conn = conn
        self._session_id = session_id
        self._classify_model = classify_model
        self.tasks: list[PlanTask] = []

    async def parse_and_send_plan(self, plan_text: str) -> None:
        """Parse an approved plan into tasks and merge with existing tasks.

        Preserves existing pending/in_progress tasks and appends new unique
        tasks from the approved plan. Uses AI to detect duplicates.

        Args:
            plan_text: The approved plan text.
        """
        classify_client = get_classify_client(self._classify_model)

        try:
            task_descriptions = await asyncio.to_thread(
                parse_plan_tasks,
                classify_client,
                plan_text,
            )

            # Get existing task descriptions for deduplication
            existing_descriptions = [t.content for t in self.tasks]

            # Filter out duplicate tasks using AI classification
            new_tasks = await self._filter_duplicates(
                classify_client, existing_descriptions, task_descriptions
            )

            # Merge: existing tasks + new unique tasks
            self.tasks.extend(new_tasks)

            # Send plan widget to UI
            await self.send_update()

            logger.info(
                f"Plan widget: {len(self.tasks)} total tasks "
                f"({len(new_tasks)} new, {len(existing_descriptions)} existing)"
            )

        except InvalidPlanParseResponse as e:
            logger.warning(f"Failed to parse plan into tasks: {e}")

    async def _filter_duplicates(
        self,
        classify_client: any,
        existing_descriptions: list[str],
        task_descriptions: list[str],
    ) -> list[PlanTask]:
        """Filter out duplicate tasks using AI classification.

        Args:
            classify_client: Client for AI classification.
            existing_descriptions: Existing task descriptions.
            task_descriptions: New task descriptions to filter.

        Returns:
            List of non-duplicate PlanTask objects.
        """
        new_tasks: list[PlanTask] = []

        for desc in task_descriptions:
            try:
                is_duplicate = await asyncio.to_thread(
                    classify_task_duplicate,
                    classify_client,
                    existing_descriptions + [t.content for t in new_tasks],
                    desc,
                )
                if not is_duplicate:
                    new_tasks.append(PlanTask(content=desc))
                    logger.debug(f"Adding new task: {desc}")
                else:
                    logger.debug(f"Skipping duplicate task: {desc}")
            except InvalidTaskDuplicateResponse:
                # When uncertain, add the task (safer than losing work)
                new_tasks.append(PlanTask(content=desc))
                logger.warning(f"Duplicate check failed, adding task: {desc}")

        return new_tasks

    async def send_update(self) -> None:
        """Send current plan task status to UI."""
        if not self.tasks:
            return

        entries = [plan_entry(task.content, status=task.status) for task in self.tasks]
        plan_update = update_plan(entries)
        await self._conn.session_update(session_id=self._session_id, update=plan_update)

    async def update_progress(self, recent_output: str) -> None:
        """Detect and update task progress based on recent output.

        Args:
            recent_output: Recent inner agent output to analyze.
        """
        if not self.tasks:
            return

        classify_client = get_classify_client(self._classify_model)

        try:
            task_descriptions = [task.content for task in self.tasks]
            progress = await asyncio.to_thread(
                detect_task_progress,
                classify_client,
                task_descriptions,
                recent_output,
            )

            if not progress:
                return

            # Update task statuses
            for idx, new_status in progress.items():
                if 0 <= idx < len(self.tasks):
                    if new_status in ("pending", "in_progress", "completed"):
                        self.tasks[idx].status = new_status

            # Send updated plan to UI
            await self.send_update()
            logger.info(f"Updated task progress: {progress}")

        except Exception as e:
            logger.warning(f"Failed to detect task progress: {e}")

    def mark_all_completed(self) -> None:
        """Mark all tasks as completed."""
        for task in self.tasks:
            task.status = "completed"

    def reset_for_new_prompt(self) -> None:
        """Reset tasks for a new user prompt.

        Preserves incomplete tasks to ensure follow-up prompts
        do not lose pending work.
        """
        self.tasks = [t for t in self.tasks if t.status != "completed"]
