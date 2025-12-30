"""ACP protocol adapters and content extraction."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from acp.schema import (
    AudioContentBlock,
    BlobResourceContents,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
)

logger = logging.getLogger(__name__)

PromptBlock = (
    TextContentBlock
    | ImageContentBlock
    | AudioContentBlock
    | ResourceContentBlock
    | EmbeddedResourceContentBlock
)


def read_file_content(file_path: str) -> str | None:
    """Read current file content, or None if file does not exist."""
    try:
        path = Path(file_path)
        if path.exists():
            return path.read_text()
        return None
    except (OSError, UnicodeDecodeError) as e:
        logger.warning(f"Failed to read file {file_path}: {e}")
        return None


def compute_edit_result(
    old_content: str | None,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str | None:
    """Compute the result of an Edit operation.

    Returns the new file content after applying the replacement,
    or None if the old_string was not found.
    """
    if old_content is None:
        return None

    if old_string not in old_content:
        return None

    if replace_all:
        return old_content.replace(old_string, new_string)
    return old_content.replace(old_string, new_string, 1)


def read_resource_link(uri: str, cwd: Path | None = None) -> str | None:
    """Read content from a resource link URI.

    Supports file:// URIs and relative paths.
    """
    if uri.startswith("file://"):
        file_path = Path(uri[7:])
    elif not uri.startswith(("http://", "https://")):
        file_path = Path(uri)
        if cwd and not file_path.is_absolute():
            file_path = cwd / file_path
    else:
        return None

    try:
        return file_path.read_text()
    except (OSError, UnicodeDecodeError):
        logger.warning(f"Failed to read resource: {uri}")
        return None


def extract_prompt_content(
    blocks: list[PromptBlock], cwd: Path | None = None
) -> list[dict[str, Any]]:
    """Extract content blocks from prompt for Claude API.

    Converts ACP prompt blocks to Claude API content format, supporting
    text, images, and embedded resources.

    Returns:
        List of Claude API content blocks (text and image types).
    """
    content: list[dict[str, Any]] = []
    text_parts: list[str] = []

    def flush_text() -> None:
        """Add accumulated text as a content block."""
        if text_parts:
            content.append({"type": "text", "text": "\n\n".join(text_parts)})
            text_parts.clear()

    for block in blocks:
        if isinstance(block, dict):
            _process_dict_block(block, text_parts, content, flush_text, cwd)
        elif isinstance(block, ImageContentBlock):
            flush_text()
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type,
                        "data": block.data,
                    },
                }
            )
        elif isinstance(block, TextContentBlock):
            text_parts.append(block.text)
        elif isinstance(block, EmbeddedResourceContentBlock):
            resource = block.resource
            if isinstance(resource, TextResourceContents):
                text_parts.append(f'<file uri="{resource.uri}">\n{resource.text}\n</file>')
            elif isinstance(resource, BlobResourceContents):
                text_parts.append(f"[Binary file: {resource.uri}]")
        elif isinstance(block, ResourceContentBlock):
            file_content = read_resource_link(block.uri, cwd)
            if file_content:
                text_parts.append(
                    f'<file uri="{block.uri}" name="{block.name}">\n{file_content}\n</file>'
                )
            else:
                text_parts.append(f"[Referenced file: {block.name} ({block.uri})]")
        elif hasattr(block, "text"):
            text_parts.append(str(getattr(block, "text", "")))

    flush_text()
    return content


def _process_dict_block(
    block: dict[str, Any],
    text_parts: list[str],
    content: list[dict[str, Any]],
    flush_text: Any,
    cwd: Path | None,
) -> None:
    """Process a dict-style content block."""
    block_type = block.get("type")

    if block_type == "text":
        text_parts.append(block.get("text", ""))

    elif block_type == "image":
        flush_text()
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block.get(
                        "mimeType", block.get("mime_type", "image/png")
                    ),
                    "data": block.get("data", ""),
                },
            }
        )

    elif block_type == "resource":
        resource = block.get("resource", {})
        uri = resource.get("uri", "")
        text = resource.get("text")
        if text:
            text_parts.append(f'<file uri="{uri}">\n{text}\n</file>')

    elif block_type == "resource_link":
        uri = block.get("uri", "")
        name = block.get("name", uri)
        file_content = read_resource_link(uri, cwd)
        if file_content:
            text_parts.append(f'<file uri="{uri}" name="{name}">\n{file_content}\n</file>')
        else:
            text_parts.append(f"[Referenced file: {name} ({uri})]")


def extract_prompt_text(blocks: list[PromptBlock], cwd: Path | None = None) -> str:
    """Extract text content from prompt blocks.

    Handles text blocks, embedded resources, and resource links.
    Returns only the text portion, ignoring images.
    """
    content = extract_prompt_content(blocks, cwd)
    text_parts = [block["text"] for block in content if block.get("type") == "text"]
    return "\n\n".join(text_parts)


def format_tool_action(tool_name: str, input_params: dict[str, Any]) -> str:
    """Format a tool call as a human-readable action description.

    Args:
        tool_name: Name of the tool being called.
        input_params: Parameters passed to the tool.

    Returns:
        Action description in imperative form for compliance review.
    """
    tool_formats: dict[tuple[str, ...], Callable[[dict[str, Any]], str]] = {
        ("Bash", "mcp__acp__Bash"): lambda p: f"Execute shell command: {p.get('command', '')}",
        ("Write", "mcp__acp__Write"): lambda p: f"Write file: {p.get('file_path', '')}",
        ("Edit", "mcp__acp__Edit"): lambda p: f"Edit file: {p.get('file_path', '')}",
        ("Read", "mcp__acp__Read"): lambda p: f"Read file: {p.get('file_path', '')}",
        ("Glob",): lambda p: f"Search files matching: {p.get('pattern', '')}",
        ("Grep",): lambda p: f"Search content matching: {p.get('pattern', '')}",
        ("WebFetch",): lambda p: f"Fetch URL: {p.get('url', '')}",
    }

    for names, formatter in tool_formats.items():
        if tool_name in names:
            return formatter(input_params)

    # Generic format for unknown tools
    params_str = ", ".join(f"{k}={v!r}" for k, v in input_params.items())
    return f"Call {tool_name}({params_str})"


def capture_git_state(cwd: Path | None = None) -> str:
    """Capture current Git repository state for compliance reviews.

    Runs git commands to capture:
    - Current branch name
    - Working tree status (clean/dirty)
    - List of uncommitted changes

    Args:
        cwd: Working directory (uses current directory if None).

    Returns:
        Formatted string describing Git state, or error message if not a git repo.
    """
    import subprocess

    work_dir = str(cwd) if cwd else None

    try:
        # Get current branch
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=5,
        )
        if branch_result.returncode != 0:
            return "(Not a git repository)"

        branch = branch_result.stdout.strip()

        # Get working tree status
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=5,
        )

        if status_result.returncode != 0:
            return f"Branch: {branch}\nStatus: (Unable to determine)"

        status_output = status_result.stdout.strip()

        if not status_output:
            return f"Branch: {branch}\nWorking tree: clean"

        # Parse status output to categorize changes
        staged = []
        unstaged = []
        untracked = []

        for line in status_output.split("\n"):
            if not line:
                continue
            index_status = line[0]
            worktree_status = line[1]
            file_path = line[3:]

            if index_status == "?":
                untracked.append(file_path)
            elif index_status != " ":
                staged.append(file_path)
            if worktree_status != " " and worktree_status != "?":
                unstaged.append(file_path)

        parts = [f"Branch: {branch}", "Working tree: DIRTY (uncommitted changes)"]

        if staged:
            parts.append(f"Staged files: {', '.join(staged[:5])}")
            if len(staged) > 5:
                parts.append(f"  ... and {len(staged) - 5} more")

        if unstaged:
            parts.append(f"Unstaged changes: {', '.join(unstaged[:5])}")
            if len(unstaged) > 5:
                parts.append(f"  ... and {len(unstaged) - 5} more")

        if untracked:
            parts.append(f"Untracked files: {', '.join(untracked[:5])}")
            if len(untracked) > 5:
                parts.append(f"  ... and {len(untracked) - 5} more")

        return "\n".join(parts)

    except subprocess.TimeoutExpired:
        return "(Git command timed out)"
    except FileNotFoundError:
        return "(Git not installed)"
    except Exception as e:
        logger.warning(f"Failed to capture git state: {e}")
        return f"(Error capturing git state: {e})"
