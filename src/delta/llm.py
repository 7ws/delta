"""LLM client for compliance reviews using Claude Code CLI."""

from __future__ import annotations

import subprocess


class ClaudeCodeClient:
    """LLM client that uses Claude Code CLI for completions."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize Claude Code client.

        Args:
            model: Optional model override (e.g., 'claude-sonnet-4-20250514').
        """
        self.model = model

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt via Claude Code CLI."""
        cmd = ["claude", "--print"]

        if self.model:
            cmd.extend(["--model", self.model])

        if system:
            cmd.extend(["--system-prompt", system])

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout


def get_fast_client() -> ClaudeCodeClient:
    """Get a fast LLM client for quick classifications."""
    return ClaudeCodeClient(model="haiku")


def classify_action_type(client: ClaudeCodeClient, action: str) -> bool:
    """Classify whether an action is read-only using a fast LLM.

    Args:
        client: Claude Code client.
        action: The action description to classify.

    Returns:
        True if the action is read-only, False if it modifies state.
    """
    from textwrap import dedent

    prompt = dedent(f"""\
        Classify this action as READ-ONLY or READ-WRITE.

        Action: {action}

        READ-ONLY: Only reads data, no side effects
        (e.g., git status, ls, cat, grep, git diff, gh pr list)

        READ-WRITE: Modifies files, state, or has side effects
        (e.g., git commit, rm, echo > file, git push)

        Reply with exactly one word: READONLY or READWRITE\
    """)

    response = client.complete(
        prompt=prompt,
        system="You classify actions. Reply with exactly one word: READONLY or READWRITE",
    )

    return "READONLY" in response.upper()


def get_llm_client(model: str | None = None) -> ClaudeCodeClient:
    """Get a Claude Code LLM client.

    Args:
        model: Optional model override.

    Returns:
        Configured Claude Code client.
    """
    return ClaudeCodeClient(model=model)
