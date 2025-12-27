"""Tests for LLM client and classification functions."""

from unittest.mock import MagicMock

import pytest

from delta.llm import (
    ClaudeCodeClient,
    InvalidComplexityResponse,
    classify_task_complexity,
    interpret_for_user,
)


class TestClassifyTaskComplexity:
    """Tests for task complexity classification."""

    def test_given_simple_response_when_classified_then_returns_simple(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = "SIMPLE"

        # When
        result = classify_task_complexity(client, "Rewrite git history")

        # Then
        assert result == "SIMPLE"

    def test_given_moderate_response_when_classified_then_returns_moderate(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = "MODERATE"

        # When
        result = classify_task_complexity(client, "Add a new feature")

        # Then
        assert result == "MODERATE"

    def test_given_complex_response_when_classified_then_returns_complex(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = "COMPLEX"

        # When
        result = classify_task_complexity(client, "Design new architecture")

        # Then
        assert result == "COMPLEX"

    def test_given_response_with_newline_when_classified_then_parses_correctly(
        self,
    ) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = "SIMPLE\n"

        # When
        result = classify_task_complexity(client, "Run tests")

        # Then
        assert result == "SIMPLE"

    def test_given_lowercase_response_when_classified_then_parses_correctly(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = "simple"

        # When
        result = classify_task_complexity(client, "Rename file")

        # Then
        assert result == "SIMPLE"

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

    def test_invalid_complexity_response_exception_message(self) -> None:
        # Given
        error = InvalidComplexityResponse("Invalid response: FOO")

        # Then
        assert "Invalid response" in str(error)
        assert "FOO" in str(error)


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

    def test_given_empty_response_when_interpreted_then_returns_none(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = ""

        # When
        result = interpret_for_user(client, "Some text", "planning")

        # Then
        assert result is None

    def test_given_whitespace_response_when_interpreted_then_returns_none(self) -> None:
        # Given
        client = MagicMock(spec=ClaudeCodeClient)
        client.complete.return_value = "   \n  "

        # When
        result = interpret_for_user(client, "Some text", "reviewing")

        # Then
        assert result is None
