"""Provider-agnostic LLM client for compliance reviews.

This module provides a configurable LLM client that supports multiple providers
through a common interface. The provider is configured via environment variables,
making Delta independent of any specific AI provider.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt to the LLM and return the response.

        Args:
            prompt: The user prompt to send.
            system: Optional system prompt.

        Returns:
            The LLM's response text.
        """
        ...


class ClaudeCodeClient(LLMClient):
    """LLM client that uses Claude Code CLI for completions.

    This uses the `claude` CLI tool which must be installed and configured.
    """

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


class OpenAIClient(LLMClient):
    """LLM client that uses OpenAI API.

    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        """Initialize OpenAI client.

        Args:
            model: Model to use (default: gpt-4o).
        """
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                import openai  # type: ignore[import-not-found]

                self._client = openai.OpenAI()
            except ImportError as e:
                raise ImportError(
                    "OpenAI package not installed. Install with: uv add openai"
                ) from e
        return self._client

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt via OpenAI API."""
        import importlib.util

        if importlib.util.find_spec("openai") is None:
            raise ImportError("OpenAI package not installed. Install with: uv add openai")

        client = self._get_client()

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        content = response.choices[0].message.content
        return str(content) if content else ""


class AnthropicClient(LLMClient):
    """LLM client that uses Anthropic API directly.

    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        """Initialize Anthropic client.

        Args:
            model: Model to use (default: claude-sonnet-4-20250514).
        """
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                import anthropic  # type: ignore[import-not-found]

                self._client = anthropic.Anthropic()
            except ImportError as e:
                raise ImportError(
                    "Anthropic package not installed. Install with: uv add anthropic"
                ) from e
        return self._client

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt via Anthropic API."""
        import importlib.util

        if importlib.util.find_spec("anthropic") is None:
            raise ImportError("Anthropic package not installed. Install with: uv add anthropic")

        client = self._get_client()

        kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        text_content = response.content[0]
        return str(getattr(text_content, "text", ""))


class OllamaClient(LLMClient):
    """LLM client that uses local Ollama instance.

    Requires Ollama to be running locally.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
    ) -> None:
        """Initialize Ollama client.

        Args:
            model: Model to use (default: llama3.2).
            host: Ollama API host URL.
        """
        self.model = model
        self.host = host

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt via Ollama API."""
        import json
        import urllib.request

        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        data = json.dumps(
            {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
            }
        ).encode()

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            return str(result.get("response", ""))


def get_llm_client(
    provider: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Get an LLM client based on configuration.

    Configuration is read from environment variables:
    - DELTA_LLM_PROVIDER: Provider name (claude-code, openai, anthropic, ollama)
    - DELTA_LLM_MODEL: Model name (provider-specific)

    Args:
        provider: Override provider from env.
        model: Override model from env.

    Returns:
        Configured LLM client.

    Raises:
        ValueError: If provider is not supported.
    """
    provider = provider or os.environ.get("DELTA_LLM_PROVIDER", "claude-code")
    model = model or os.environ.get("DELTA_LLM_MODEL")

    if provider == "claude-code":
        return ClaudeCodeClient(model=model)
    elif provider == "openai":
        return OpenAIClient(model=model or "gpt-4o")
    elif provider == "anthropic":
        return AnthropicClient(model=model or "claude-sonnet-4-20250514")
    elif provider == "ollama":
        return OllamaClient(model=model or "llama3.2")
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            "Supported: claude-code, openai, anthropic, ollama"
        )
