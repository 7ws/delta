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
        "--review-model",
        help="Model for compliance reviews (e.g., 'sonnet', 'opus')",
    )
    serve_parser.add_argument(
        "--classify-model",
        default="haiku",
        help="Model for action classification (default: 'haiku')",
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


def _get_log_dir() -> Path:
    """Get platform-appropriate log directory."""
    import os
    import platform

    if platform.system() == "Darwin":
        # macOS: ~/Library/Logs/Delta/
        return Path.home() / "Library" / "Logs" / "Delta"
    elif platform.system() == "Windows":
        # Windows: %LOCALAPPDATA%\Delta\Logs\
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return local_app_data / "Delta" / "Logs"
    else:
        # Linux/Unix: ~/.local/state/delta/ (XDG Base Directory)
        xdg_state = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        return xdg_state / "delta"


def _run_serve(args: argparse.Namespace) -> int:
    """Run the ACP server."""
    import logging
    from datetime import datetime

    log_level = logging.DEBUG if args.verbose else logging.WARNING

    # Create log file in platform-appropriate directory
    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"delta-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    # Configure logging to both stderr and file
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(log_file),
    ]

    logging.basicConfig(
        level=log_level,
        format="[Delta] %(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    logging.info(f"Delta log file: {log_file}")

    try:
        from delta.acp_server import run_server

        asyncio.run(
            run_server(
                agents_md_path=args.agents_md,
                review_model=args.review_model,
                classify_model=args.classify_model,
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
