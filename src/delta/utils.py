"""Utility functions for Delta."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

import json_repair


def extract_json(response: str) -> str:
    """Extract JSON from a response that may contain markdown code blocks.

    Handles:
    - JSON wrapped in ```json ... ``` code blocks
    - Raw JSON objects starting with {
    - Raw JSON arrays starting with [

    Args:
        response: Raw response text that may contain JSON.

    Returns:
        The extracted JSON string.

    Raises:
        ValueError: If no JSON found in response.
    """
    # Try markdown code block first
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # Try to find raw JSON object
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        return response[json_start:json_end]

    # Try to find raw JSON array
    array_start = response.find("[")
    array_end = response.rfind("]") + 1
    if array_start >= 0 and array_end > array_start:
        return response[array_start:array_end]

    raise ValueError("No JSON found in response")


def parse_json(response: str) -> dict[str, Any] | list[Any]:
    """Extract and parse JSON from a response.

    Uses json_repair as fallback when standard json.loads fails on malformed
    JSON (for example, truncated responses with unterminated strings).

    Args:
        response: Raw response text that may contain JSON.

    Returns:
        Parsed JSON as dict or list.

    Raises:
        ValueError: If no valid JSON found in response.
    """
    json_str = extract_json(response)
    try:
        result: dict[str, Any] | list[Any] = json.loads(json_str)
    except json.JSONDecodeError as e:
        repaired = json_repair.loads(json_str)
        if not isinstance(repaired, (dict, list)):
            raise ValueError(f"Repaired JSON is not dict or list: {type(repaired)}") from e
        result = repaired
    return result


async def parse_with_retry[T](
    llm_call: Callable[[str], Awaitable[str]],
    initial_prompt: str,
    parser: Callable[[str], T],
    max_retries: int = 3,
    system_hint: str = "You MUST output valid JSON only.",
) -> T:
    """Call an LLM and parse the response, retrying on parse failures.

    This helper encapsulates the common pattern of:
    1. Call LLM with a prompt
    2. Try to parse the response
    3. If parsing fails, retry with feedback

    Args:
        llm_call: Async function that takes a prompt and returns LLM response.
        initial_prompt: The initial prompt to send.
        parser: Function to parse the response, raises on failure.
        max_retries: Maximum number of attempts (default 3).
        system_hint: Hint to include when retrying after parse failure.

    Returns:
        The parsed result from the parser function.

    Raises:
        RuntimeError: If parsing fails after all retries.
    """
    prompt = initial_prompt
    last_error: Exception | None = None

    for attempt in range(max_retries):
        response = await llm_call(prompt)

        try:
            return parser(response)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_error = e

            if attempt < max_retries - 1:
                prompt = (
                    f"Your previous response had invalid JSON. "
                    f"{system_hint}\n\n{initial_prompt}"
                )

    raise RuntimeError(
        f"Failed to parse response after {max_retries} attempts: {last_error}"
    )
