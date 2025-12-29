"""Tests for LLM client and classification functions."""

from unittest.mock import MagicMock

import pytest

from delta.llm import (
    ClaudeCodeClient,
    InvalidComplexityResponse,
    InvalidTaskDuplicateResponse,
    InvalidWriteClassificationResponse,
    classify_task_complexity,
    classify_task_duplicate,
    classify_write_operation,
    interpret_for_user,
)


class TestClassifyTaskComplexity:
    """Tests for task complexity classification."""

    @pytest.mark.parametrize(
        "response,expected",
        [
            ("SIMPLE", "SIMPLE"),
            ("MODERATE", "MODERATE"),
            ("COMPLEX", "COMPLEX"),
            ("simple", "SIMPLE"),  # lowercase
            ("SIMPLE\n", "SIMPLE"),  # with newline
        ],
    )
    def test_given_valid_response_when_classified_then_returns_expected(
        self, response: str, expected: str
    ) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = response

        # When
        result = classify_task_complexity(client, "Do something")

        # Then
        assert result == expected

    def test_given_invalid_then_valid_response_when_classified_then_retries(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.side_effect = ["INVALID", "MODERATE"]

        # When
        result = classify_task_complexity(client, "Fix bug")

        # Then
        assert result == "MODERATE"
        assert client.complete.call_count == 2

    def test_given_all_invalid_responses_when_classified_then_raises_error(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = "INVALID"

        # When/Then
        with pytest.raises(InvalidComplexityResponse):
            classify_task_complexity(client, "Do something", max_retries=2)


class TestInterpretForUser:
    """Tests for inner agent output interpretation."""

    def test_given_suppress_response_when_interpreted_then_returns_none(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = "SUPPRESS"

        # When
        result = interpret_for_user(client, "The user chose to skip", "executing")

        # Then
        assert result is None

    def test_given_useful_text_when_interpreted_then_returns_rewritten(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = "Creating the mobile layout component..."

        # When
        result = interpret_for_user(client, "I am creating the component", "executing")

        # Then
        assert result == "Creating the mobile layout component..."

    @pytest.mark.parametrize("response", ["", "   \n  "])
    def test_given_empty_or_whitespace_response_when_interpreted_then_returns_none(
        self, response: str
    ) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = response

        # When
        result = interpret_for_user(client, "Some text", "planning")

        # Then
        assert result is None

    @pytest.mark.parametrize(
        "text",
        ["The inner agent completed", "workflow phase transition"],
    )
    def test_given_internal_reference_when_interpreted_then_suppressed(
        self, text: str
    ) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)

        # When
        result = interpret_for_user(client, text, "executing")

        # Then
        assert result is None


class TestClassifyWriteOperation:
    """Tests for AI-based write operation classification.

    Note: Write classification uses AI, not hardcoded patterns.
    See AGENTS.md §13.1 for the architectural decision.
    """

    @pytest.mark.parametrize(
        "response,expected",
        [
            ("WRITE", True),
            ("READONLY", False),
            ("write", True),  # lowercase
            ("readonly", False),  # lowercase
            ("WRITE\n", True),  # with newline
        ],
    )
    def test_given_valid_response_when_classified_then_returns_expected(
        self, response: str, expected: bool
    ) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = response

        # When
        result = classify_write_operation(client, "Tool", "Do something")

        # Then
        assert result is expected

    def test_given_invalid_then_valid_response_when_classified_then_retries(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.side_effect = ["INVALID", "WRITE"]

        # When
        result = classify_write_operation(client, "Write", "Write file: /tmp/test.py")

        # Then
        assert result is True
        assert client.complete.call_count == 2

    def test_given_all_invalid_responses_when_classified_then_raises_error(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = "INVALID"

        # When/Then
        with pytest.raises(InvalidWriteClassificationResponse):
            classify_write_operation(client, "Bash", "Do something", max_retries=2)


class TestClassifyTaskDuplicate:
    """Tests for AI-based task duplicate classification.

    Note: Task deduplication uses AI, not hardcoded patterns.
    See AGENTS.md §13.1 for the architectural decision.
    """

    @pytest.mark.parametrize(
        "response,expected",
        [
            ("DUPLICATE", True),
            ("UNIQUE", False),
            ("duplicate", True),  # lowercase
            ("unique", False),  # lowercase
            ("DUPLICATE\n", True),  # with newline
        ],
    )
    def test_given_valid_response_when_classified_then_returns_expected(
        self, response: str, expected: bool
    ) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = response

        # When
        result = classify_task_duplicate(
            client, ["Create user model"], "Add user database model"
        )

        # Then
        assert result is expected

    def test_given_empty_existing_tasks_when_classified_then_returns_false(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)

        # When
        result = classify_task_duplicate(client, [], "Create new feature")

        # Then
        assert result is False
        client.complete.assert_not_called()

    def test_given_invalid_then_valid_response_when_classified_then_retries(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.side_effect = ["INVALID", "DUPLICATE"]

        # When
        result = classify_task_duplicate(client, ["Task A"], "Task B")

        # Then
        assert result is True
        assert client.complete.call_count == 2

    def test_given_all_invalid_responses_when_classified_then_raises_error(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = "INVALID"

        # When/Then
        with pytest.raises(InvalidTaskDuplicateResponse):
            classify_task_duplicate(client, ["Task A"], "Task B", max_retries=2)
