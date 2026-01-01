"""Tests for delta.thinking_status module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from delta.thinking_status import ThinkingStatus, ThinkingStatusManager, WorkflowStep


class TestWorkflowStep:
    """Tests for WorkflowStep enum."""

    def test_step_has_description(self):
        """Given a workflow step, when accessing description, then returns string."""
        # When
        description = WorkflowStep.TRIAGE.description

        # Then
        assert description == "Analyzing request"

    def test_step_has_estimated_seconds(self):
        """Given a workflow step, when accessing estimated_seconds, then returns int."""
        # When
        estimate = WorkflowStep.TRIAGE.estimated_seconds

        # Then
        assert estimate == 2

    def test_all_steps_have_descriptions(self):
        """Given all workflow steps, when iterating, then all have descriptions."""
        for step in WorkflowStep:
            assert isinstance(step.description, str)
            assert len(step.description) > 0

    def test_all_steps_have_estimates(self):
        """Given all workflow steps, when iterating, then all have positive estimates."""
        for step in WorkflowStep:
            assert isinstance(step.estimated_seconds, int)
            assert step.estimated_seconds > 0


class TestThinkingStatus:
    """Tests for ThinkingStatus dataclass."""

    def test_format_title_with_remaining_time(self):
        """Given status with time remaining, when formatting title, then includes remaining."""
        # Given
        status = ThinkingStatus(
            step=WorkflowStep.TRIAGE,
            elapsed_seconds=5.0,
            step_elapsed_seconds=1.0,
        )

        # When
        title = status.format_title()

        # Then
        assert "Analyzing request" in title
        assert "5s" in title
        assert "1s remaining" in title

    def test_format_title_without_remaining_time(self):
        """Given status past estimate, when formatting title, then omits remaining."""
        # Given
        status = ThinkingStatus(
            step=WorkflowStep.TRIAGE,
            elapsed_seconds=10.0,
            step_elapsed_seconds=5.0,  # Past the 2s estimate
        )

        # When
        title = status.format_title()

        # Then
        assert "Analyzing request" in title
        assert "10s" in title
        assert "remaining" not in title


class TestThinkingStatusManager:
    """Tests for ThinkingStatusManager class."""

    @pytest.fixture
    def mock_conn(self):
        """Create a mock ACP connection."""
        conn = MagicMock()
        conn.session_update = AsyncMock()
        return conn

    @pytest.fixture
    def manager(self, mock_conn):
        """Create a ThinkingStatusManager instance."""
        return ThinkingStatusManager(mock_conn, "test-session")

    @pytest.mark.asyncio
    async def test_start_creates_status_block(self, manager, mock_conn):
        """Given a manager, when start is called, then session_update is called with in_progress."""
        # When
        await manager.start(WorkflowStep.TRIAGE)

        # Then
        mock_conn.session_update.assert_called_once()
        call_args = mock_conn.session_update.call_args
        assert call_args.kwargs["session_id"] == "test-session"

        # Clean up
        await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_completes_status_block(self, manager, mock_conn):
        """Given a running manager, when stop is called, then status is completed."""
        # Given
        await manager.start(WorkflowStep.TRIAGE)
        mock_conn.session_update.reset_mock()

        # When
        await manager.stop("Done")

        # Then
        mock_conn.session_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_step_updates_status(self, manager, mock_conn):
        """Given a running manager, when set_step is called, then status is updated."""
        # Given
        await manager.start(WorkflowStep.TRIAGE)
        mock_conn.session_update.reset_mock()

        # When
        await manager.set_step(WorkflowStep.PLANNING_FULL)

        # Then
        mock_conn.session_update.assert_called_once()

        # Clean up
        await manager.stop()

    @pytest.mark.asyncio
    async def test_elapsed_seconds_tracks_time(self, manager):
        """Given a started manager, when time passes, then elapsed_seconds increases."""
        # Given
        await manager.start(WorkflowStep.TRIAGE)

        # When
        await asyncio.sleep(0.1)
        elapsed = manager.elapsed_seconds

        # Then
        assert elapsed >= 0.1

        # Clean up
        await manager.stop()

    @pytest.mark.asyncio
    async def test_periodic_updates_occur(self, manager, mock_conn):
        """Given a running manager, when 1 second elapses, then multiple updates occur."""
        # Given
        await manager.start(WorkflowStep.TRIAGE)
        initial_call_count = mock_conn.session_update.call_count

        # When
        await asyncio.sleep(1.1)

        # Then
        # Should have at least 2 updates in 1 second (0.5s interval)
        update_count = mock_conn.session_update.call_count - initial_call_count
        assert update_count >= 2

        # Clean up
        await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_background_task(self, manager, mock_conn):
        """Given a running manager, when stop is called, then background task is cancelled."""
        # Given
        await manager.start(WorkflowStep.TRIAGE)
        assert manager._update_task is not None

        # When
        await manager.stop()

        # Then
        assert manager._update_task is None

    @pytest.mark.asyncio
    async def test_set_step_resets_step_elapsed(self, manager):
        """Given a running manager, when set_step is called, then step_elapsed resets."""
        # Given
        await manager.start(WorkflowStep.TRIAGE)
        await asyncio.sleep(0.1)
        initial_step_elapsed = manager.step_elapsed_seconds

        # When
        await manager.set_step(WorkflowStep.PLANNING_FULL)

        # Then
        assert manager.step_elapsed_seconds < initial_step_elapsed

        # Clean up
        await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self, manager, mock_conn):
        """Given a manager not started, when stop is called, then no error occurs."""
        # When/Then - should not raise
        await manager.stop()

        # And no session_update should be called
        mock_conn.session_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_step_without_start_is_safe(self, manager, mock_conn):
        """Given a manager not started, when set_step is called, then no error occurs."""
        # When/Then - should not raise
        await manager.set_step(WorkflowStep.PLANNING_FULL)

        # And no session_update should be called
        mock_conn.session_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_with_keep_elapsed_formats_title_with_elapsed_time(
        self, manager, mock_conn
    ):
        """Given a running manager, when stop(keep_elapsed=True), then title includes elapsed."""
        # Given
        await manager.start(WorkflowStep.TRIAGE)
        await asyncio.sleep(0.1)
        mock_conn.session_update.reset_mock()

        # When
        await manager.stop(keep_elapsed=True)

        # Then
        call_args = mock_conn.session_update.call_args
        update = call_args.kwargs["update"]
        assert "Analyzing request" in update.title
        assert "s)" in update.title

    @pytest.mark.asyncio
    async def test_stop_with_keep_elapsed_false_uses_final_title_only(
        self, manager, mock_conn
    ):
        """Given a running manager, when stop(final_title=...), then title is exactly that."""
        # Given
        await manager.start(WorkflowStep.TRIAGE)
        mock_conn.session_update.reset_mock()

        # When
        await manager.stop(final_title="Done")

        # Then
        call_args = mock_conn.session_update.call_args
        update = call_args.kwargs["update"]
        assert update.title == "Done"

    @pytest.mark.asyncio
    async def test_stop_with_keep_elapsed_true_ignores_final_title(
        self, manager, mock_conn
    ):
        """Given keep_elapsed=True and final_title, when stop, then final_title is ignored."""
        # Given
        await manager.start(WorkflowStep.TRIAGE)
        await asyncio.sleep(0.1)
        mock_conn.session_update.reset_mock()

        # When
        await manager.stop(final_title="Ignored", keep_elapsed=True)

        # Then
        call_args = mock_conn.session_update.call_args
        update = call_args.kwargs["update"]
        assert "Ignored" not in update.title
        assert "Analyzing request" in update.title

    @pytest.mark.asyncio
    async def test_stop_without_start_with_keep_elapsed_is_safe(self, manager, mock_conn):
        """Given a manager not started, when stop(keep_elapsed=True), then no error occurs."""
        # When/Then - should not raise
        await manager.stop(keep_elapsed=True)

        # And no session_update should be called
        mock_conn.session_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_with_zero_elapsed_time(self, manager, mock_conn):
        """Given a manager just started, when stop(keep_elapsed=True), then shows 0s."""
        # Given
        await manager.start(WorkflowStep.TRIAGE)
        mock_conn.session_update.reset_mock()

        # When
        await manager.stop(keep_elapsed=True)

        # Then
        call_args = mock_conn.session_update.call_args
        update = call_args.kwargs["update"]
        assert "(0s)" in update.title

    @pytest.mark.asyncio
    async def test_stop_at_exact_step_estimate_boundary(self, manager, mock_conn):
        """Given step_elapsed equals estimate, when stop(keep_elapsed=True), then no remaining."""
        # Given
        await manager.start(WorkflowStep.TRIAGE)
        mock_conn.session_update.reset_mock()

        # When
        await manager.stop(keep_elapsed=True)

        # Then
        call_args = mock_conn.session_update.call_args
        update = call_args.kwargs["update"]
        assert "remaining" not in update.title
