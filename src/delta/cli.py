"""Command-line interface for Delta."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def main() -> int:
    """Run Delta CLI."""
    parser = argparse.ArgumentParser(
        prog="delta",
        description="AI agent compliance wrapper that enforces AGENTS.md guidelines.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run Delta as an ACP server",
    )
    serve_parser.add_argument(
        "--agents-md",
        type=Path,
        help="Path to AGENTS.md (auto-detected if not specified)",
    )
    serve_parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum compliance attempts before blocking (default: 2)",
    )
    serve_parser.add_argument(
        "--provider",
        choices=["claude-code", "openai", "anthropic", "ollama"],
        help="LLM provider for compliance review (default: claude-code)",
    )
    serve_parser.add_argument(
        "--model",
        help="Model to use (provider-specific)",
    )
    serve_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "serve":
        return _run_serve(args)

    return 0


def _run_serve(args: argparse.Namespace) -> int:
    """Run the ACP server."""
    import logging

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="[Delta] %(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    try:
        from delta.acp_server import run_server

        asyncio.run(
            run_server(
                agents_md_path=args.agents_md,
                max_attempts=args.max_attempts,
                llm_provider=args.provider,
                llm_model=args.model,
            )
        )
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ImportError:
        print(
            "Error: agent-client-protocol package not installed.\n"
            "Install with: uv add agent-client-protocol",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
