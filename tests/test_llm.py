"""Tests for LLM client and classification functions."""

from unittest.mock import MagicMock

import pytest

from delta.llm import (
    ClaudeCodeClient,
    InvalidComplexityResponse,
    InvalidPlanParseResponse,
    InvalidTaskDuplicateResponse,
    InvalidWriteClassificationResponse,
    classify_task_complexity,
    classify_task_duplicate,
    classify_write_operation,
    detect_task_progress,
    generate_clarifying_questions,
    parse_plan_tasks,
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


class TestGenerateClarifyingQuestions:
    """Tests for clarifying question generation with truncated JSON repair."""

    def test_given_truncated_json_array_when_generated_then_repairs_and_returns(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        # Truncated JSON array with unterminated string
        client.complete.return_value = '["What is expected?", "What files'

        # When
        result = generate_clarifying_questions(
            client, "Do something", ["Some violation"]
        )

        # Then
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_given_valid_json_when_generated_then_returns_questions(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = '["Question 1?", "Question 2?"]'

        # When
        result = generate_clarifying_questions(client, "Do something", ["Violation"])

        # Then
        assert result == ["Question 1?", "Question 2?"]


class TestParsePlanTasks:
    """Tests for plan task parsing with truncated JSON repair."""

    def test_given_truncated_json_array_when_parsed_then_repairs_and_returns(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        # Malformed JSON array with unterminated string but has closing bracket
        # This pattern can be matched by regex \[.*?\] but fails json.loads
        client.complete.return_value = '["Add feature", "Write tests", "Update doc]'

        # When
        result = parse_plan_tasks(client, "1. Add feature\n2. Write tests\n3. Update docs")

        # Then
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_given_valid_json_when_parsed_then_returns_tasks(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = '["Task 1", "Task 2"]'

        # When
        result = parse_plan_tasks(client, "1. Task 1\n2. Task 2")

        # Then
        assert result == ["Task 1", "Task 2"]

    def test_given_all_invalid_responses_when_parsed_then_raises_error(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = "Not JSON at all"

        # When/Then
        with pytest.raises(InvalidPlanParseResponse):
            parse_plan_tasks(client, "Some plan", max_retries=1)


class TestDetectTaskProgress:
    """Tests for task progress detection with truncated JSON repair."""

    def test_given_truncated_json_object_when_detected_then_repairs_and_returns(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        # Truncated JSON object
        client.complete.return_value = '{"0": "completed", "1": "in_progress'

        # When
        result = detect_task_progress(client, ["Task 1", "Task 2"], "Recent output")

        # Then
        assert isinstance(result, dict)

    def test_given_valid_json_when_detected_then_returns_progress(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = '{"0": "completed", "1": "in_progress"}'

        # When
        result = detect_task_progress(client, ["Task 1", "Task 2"], "Recent output")

        # Then
        assert result == {0: "completed", 1: "in_progress"}

    def test_given_empty_tasks_when_detected_then_returns_empty(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)

        # When
        result = detect_task_progress(client, [], "Recent output")

        # Then
        assert result == {}
        client.complete.assert_not_called()
