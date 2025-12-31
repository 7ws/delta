"""LLM client for compliance reviews using Claude Code CLI."""

from __future__ import annotations

import json
import re
import subprocess
from textwrap import dedent
from threading import Lock

import json_repair


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


class LLMClientPool:
    """Pool of LLM clients for reuse.

    Provides thread-safe client caching to avoid creating new clients
    for each classification operation.
    """

    _instance: LLMClientPool | None = None
    _lock = Lock()

    def __init__(self) -> None:
        """Initialize the client pool."""
        self._clients: dict[str, ClaudeCodeClient] = {}
        self._client_lock = Lock()

    @classmethod
    def get_instance(cls) -> LLMClientPool:
        """Get the singleton pool instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_client(self, model: str | None = None) -> ClaudeCodeClient:
        """Get or create a client for the specified model.

        Args:
            model: Model name (None for default).

        Returns:
            Cached or new ClaudeCodeClient.
        """
        key = model or "default"
        with self._client_lock:
            if key not in self._clients:
                self._clients[key] = ClaudeCodeClient(model=model)
            return self._clients[key]


def get_classify_client(model: str = "haiku") -> ClaudeCodeClient:
    """Get a fast LLM client for action classification.

    Uses the client pool for reuse.

    Args:
        model: Model to use for classification.
    """
    return LLMClientPool.get_instance().get_client(model)


def get_llm_client(model: str | None = None) -> ClaudeCodeClient:
    """Get a Claude Code LLM client.

    Args:
        model: Optional model override.

    Returns:
        Configured Claude Code client.
    """
    return LLMClientPool.get_instance().get_client(model)


class InvalidReviewReadinessResponse(Exception):
    """Raised when the review readiness check returns an invalid response."""

    pass


class InvalidTriageResponse(Exception):
    """Raised when message triage returns an invalid response."""

    pass


class InvalidComplexityResponse(Exception):
    """Raised when task complexity classification returns an invalid response."""

    pass


class InvalidWriteClassificationResponse(Exception):
    """Raised when write operation classification returns an invalid response."""

    pass


class InvalidTaskDuplicateResponse(Exception):
    """Raised when task duplicate classification returns an invalid response."""

    pass


class InvalidPlanParseResponse(Exception):
    """Raised when plan parsing returns an invalid response."""

    pass


def classify_task_duplicate(
    client: ClaudeCodeClient,
    existing_tasks: list[str],
    new_task: str,
    max_retries: int = 2,
) -> bool:
    """Classify whether a new task duplicates any existing task using AI.

    Args:
        client: Claude Code client (Haiku recommended for speed).
        existing_tasks: List of existing task descriptions.
        new_task: New task description to check for duplicates.
        max_retries: Maximum retry attempts for invalid responses.

    Returns:
        True if the new task duplicates any existing task.

    Raises:
        InvalidTaskDuplicateResponse: If response is invalid after retries.
    """
    if not existing_tasks:
        return False

    existing_list = "\n".join(f"- {task}" for task in existing_tasks)

    prompt = dedent(f"""\
        Determine if the NEW TASK duplicates any EXISTING TASK.

        EXISTING TASKS:
        {existing_list}

        NEW TASK: {new_task}

        DUPLICATE: The new task is semantically equivalent to an existing task.
        They describe the same work, even if worded differently.

        UNIQUE: The new task describes different work not covered by existing tasks.

        Reply with exactly one word: DUPLICATE or UNIQUE\
    """)

    system = "You classify tasks. Reply with exactly one word: DUPLICATE or UNIQUE"

    for attempt in range(max_retries + 1):
        response = client.complete(prompt=prompt, system=system)
        response_upper = response.upper().strip()

        if response_upper == "DUPLICATE" or response_upper.startswith("DUPLICATE\n"):
            return True
        if response_upper == "UNIQUE" or response_upper.startswith("UNIQUE\n"):
            return False

        if attempt < max_retries:
            prompt = dedent(f"""\
                Your previous response was invalid: "{response}"

                You MUST reply with exactly one word: DUPLICATE or UNIQUE

                EXISTING TASKS:
                {existing_list}

                NEW TASK: {new_task}

                Reply with exactly one word: DUPLICATE or UNIQUE\
            """)
        else:
            raise InvalidTaskDuplicateResponse(
                f"Invalid task duplicate classification after {max_retries + 1} attempts"
            )

    raise InvalidTaskDuplicateResponse("Unexpected exit from task duplicate loop")


def classify_write_operation(
    client: ClaudeCodeClient,
    tool_name: str,
    tool_description: str,
    max_retries: int = 2,
) -> bool:
    """Classify whether a tool operation is a write operation using AI.

    Write operations MUST trigger compliance review.

    Args:
        client: Claude Code client (Haiku recommended for speed).
        tool_name: Name of the tool being called.
        tool_description: Human-readable description of the operation.
        max_retries: Maximum retry attempts for invalid responses.

    Returns:
        True if the operation modifies state and requires compliance review.

    Raises:
        InvalidWriteClassificationResponse: If response is invalid after retries.
    """
    prompt = dedent(f"""\
        Classify this tool operation as WRITE or READONLY.

        Tool: {tool_name}
        Operation: {tool_description}

        WRITE: The operation modifies state. This includes:
        - Creating, modifying, or deleting files
        - Git write operations (add, commit, push, pull, merge, rebase, reset, checkout,
          switch, restore, stash, cherry-pick, revert, tag, branch -d/-D/-m, clean, am, apply)
        - Package manager installs or uninstalls
        - Running build, test, or install commands that may create artifacts
        - Shell commands with redirects (> or >>), rm, mkdir, touch, mv, cp, chmod, chown, ln
        - Any operation that could have side effects

        READONLY: The operation only reads data with no side effects. This includes:
        - Reading files (cat, head, tail, less, more)
        - Listing files or directories (ls, find)
        - Searching (grep, rg)
        - Git read operations (status, diff, log, show, branch, remote, fetch, ls-files,
          ls-tree, rev-parse, describe, reflog, stash list, config --get, config --list)
        - Package manager queries (pip list/show/freeze, npm list/ls/view/info)
        - Version checks (--version)
        - System info (pwd, env, printenv, hostname, uname, whoami, id, date, cal, uptime)

        CRITICAL: When uncertain, classify as WRITE. Safety requires review of uncertain operations.

        Reply with exactly one word: WRITE or READONLY\
    """)

    system = "You classify tool operations. Reply with exactly one word: WRITE or READONLY"

    for attempt in range(max_retries + 1):
        response = client.complete(prompt=prompt, system=system)
        response_upper = response.upper().strip()

        if response_upper == "WRITE" or response_upper.startswith("WRITE\n"):
            return True
        if response_upper == "READONLY" or response_upper.startswith("READONLY\n"):
            return False

        if attempt < max_retries:
            prompt = dedent(f"""\
                Your previous response was invalid: "{response}"

                You MUST reply with exactly one word: WRITE or READONLY

                Tool: {tool_name}
                Operation: {tool_description}

                Reply with exactly one word: WRITE or READONLY\
            """)
        else:
            raise InvalidWriteClassificationResponse(
                f"Invalid write classification after {max_retries + 1} attempts: {response}"
            )

    raise InvalidWriteClassificationResponse("Unexpected exit from write classification loop")


def classify_task_complexity(
    client: ClaudeCodeClient, message: str, max_retries: int = 2
) -> str:
    """Classify the complexity of a task to determine review depth.

    Args:
        client: Claude Code client (Haiku recommended for speed).
        message: The user's message to analyze.
        max_retries: Maximum retry attempts for invalid responses.

    Returns:
        One of: "SIMPLE", "MODERATE", "COMPLEX"

    Raises:
        InvalidComplexityResponse: If response is invalid after retries.
    """
    prompt = dedent(f"""\
        Classify this task by complexity to determine appropriate review depth.

        User message:
        {message}

        SIMPLE: Single-step operations with clear, well-understood outcomes.
        Examples:
        - Git history rewrites (rebase, filter-repo, amend)
        - File or variable renames
        - Configuration changes with known values
        - Standard git operations (commit, push, branch)
        - Adding imports or simple one-line fixes
        - Running commands (build, test, lint)

        MODERATE: Multi-step tasks following established patterns.
        Examples:
        - Adding a feature using existing patterns in the codebase
        - Bug fixes requiring investigation and code changes
        - Refactoring code within a single module
        - Adding tests for existing functionality
        - Updating documentation

        COMPLEX: Tasks requiring architectural decisions or new patterns.
        Examples:
        - Designing new systems or modules
        - Changing application architecture
        - Integrating new external services
        - Tasks with ambiguous requirements needing clarification
        - Cross-cutting changes affecting multiple systems

        Reply with exactly one word: SIMPLE, MODERATE, or COMPLEX\
    """)

    system = (
        "You classify task complexity. "
        "Reply with exactly one word: SIMPLE, MODERATE, or COMPLEX"
    )

    for attempt in range(max_retries + 1):
        response = client.complete(prompt=prompt, system=system)
        response_upper = response.upper().strip()

        if response_upper == "SIMPLE" or response_upper.startswith("SIMPLE\n"):
            return "SIMPLE"
        if response_upper == "MODERATE" or response_upper.startswith("MODERATE\n"):
            return "MODERATE"
        if response_upper == "COMPLEX" or response_upper.startswith("COMPLEX\n"):
            return "COMPLEX"

        if attempt < max_retries:
            prompt = dedent(f"""\
                Your previous response was invalid: "{response}"

                You MUST reply with exactly one word: SIMPLE, MODERATE, or COMPLEX

                User message to analyze:
                {message}

                Reply with exactly one word: SIMPLE, MODERATE, or COMPLEX\
            """)
        else:
            raise InvalidComplexityResponse(
                f"Invalid complexity response after {max_retries + 1} attempts: {response}"
            )

    raise InvalidComplexityResponse("Unexpected exit from complexity classification loop")


def triage_user_message(
    client: ClaudeCodeClient, message: str, max_retries: int = 2
) -> str:
    """Determine how to handle a user message.

    Args:
        client: Claude Code client (Haiku recommended for speed).
        message: The user's message to analyze.
        max_retries: Maximum retry attempts for invalid responses.

    Returns:
        One of: "PLAN", "ANSWER"

    Raises:
        InvalidTriageResponse: If response is not PLAN or ANSWER after retries.
    """
    prompt = dedent(f"""\
        Analyze this user message and determine how to handle it.

        User message:
        {message}

        ANSWER: The message is PURELY read-only with ZERO possibility of edits. Use ONLY when:
        - Pure questions about the codebase ("How does X work?", "What is...", "Explain...")
        - Pure search requests ("Find all uses of X", "Where is X defined?")
        - Pure status checks ("What files changed?", "Show git status")
        - Pure read operations ("Read file X", "Show me the contents of...")

        PLAN: The message MAY produce edits or MAY execute commands. Use when:
        - Creating, modifying, or deleting files
        - Git commits, pushes, or any write operations
        - Running build, test, or install commands
        - Any explicit request to "add", "fix", "update", "implement", "create", "delete", "remove"
        - ANY ambiguous request that MIGHT require changes
        - ANY request that COULD involve running commands beyond pure reads

        CRITICAL: When uncertain, ALWAYS choose PLAN. Write operations MUST be reviewed.
        Only use ANSWER when you are 100% certain no writes will occur.

        Reply with exactly one word: PLAN or ANSWER\
    """)

    system = "You triage messages. Reply with exactly one word: PLAN or ANSWER"

    for attempt in range(max_retries + 1):
        response = client.complete(prompt=prompt, system=system)
        response_upper = response.upper().strip()

        if response_upper == "PLAN" or response_upper.startswith("PLAN\n"):
            return "PLAN"
        if response_upper == "ANSWER" or response_upper.startswith("ANSWER\n"):
            return "ANSWER"

        if attempt < max_retries:
            prompt = dedent(f"""\
                Your previous response was invalid: "{response}"

                You MUST reply with exactly one word: PLAN or ANSWER

                User message to analyze:
                {message}

                Reply with exactly one word: PLAN or ANSWER\
            """)
        else:
            raise InvalidTriageResponse(
                f"Invalid triage response after {max_retries + 1} attempts: {response}"
            )

    raise InvalidTriageResponse("Unexpected exit from triage loop")


def generate_clarifying_questions(
    client: ClaudeCodeClient,
    user_request: str,
    violations: list[str],
    max_retries: int = 2,
    conversation_context: str | None = None,
) -> list[str]:
    """Generate specific questions to resolve compliance violations.

    Args:
        client: Claude Code client (Haiku recommended for speed).
        user_request: The original user request.
        violations: List of guideline violations with justifications.
        max_retries: Maximum retry attempts for invalid responses.
        conversation_context: Recent conversation history (tool calls, prior requests).

    Returns:
        List of questions (without numbers). Returns empty list if context
        provides sufficient information to infer the user's intent.
    """
    violations_text = "\n".join(violations)

    context_section = ""
    if conversation_context:
        context_section = f"""
CONVERSATION CONTEXT (recent actions and prior requests):
{conversation_context}

CRITICAL: If the conversation context shows recent completed work (commits, test runs,
file changes) that relates to the user's request, the user likely wants a STATUS UPDATE
or NEXT STEPS, not a new implementation. In this case, return an empty JSON array []
to indicate that no clarifying questions are needed - the agent should infer intent
from context and provide a summary of completed work or suggest next steps.

"""

    prompt = dedent(f"""\
        Generate specific questions to resolve these compliance violations.

        USER REQUEST:
        {user_request}
{context_section}
        VIOLATIONS (guidelines the plan failed):
        {violations_text}

        INSTRUCTIONS:
        1. First, check if the conversation context provides enough information to
           understand what the user wants. If recent actions (commits, tests, file
           changes) clearly relate to an ambiguous request like "what's missing?" or
           "give me an update", return an empty array [] - the agent should summarize
           the recent work instead of asking questions.

        2. Only generate questions if the request is truly ambiguous AND the context
           does not provide sufficient clues about the user's intent.

        3. If questions are needed, generate 2-4 specific questions that:
           - Target a specific violation
           - Ask for concrete information (not yes/no)
           - Help clarify scope, requirements, or constraints

        Return a JSON array of question strings.
        Return [] if context is sufficient to infer intent.
        Example: ["Where should the new component be placed?", "What validation rules apply?"]

        Return ONLY the JSON array, no other text.\
    """)

    system = "You generate clarifying questions. Return only a valid JSON array of strings."

    for attempt in range(max_retries + 1):
        response = client.complete(prompt=prompt, system=system)
        response_stripped = response.strip()

        try:
            questions = json.loads(response_stripped)
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                return questions
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"\[.*?\]", response_stripped, re.DOTALL)
        if json_match:
            try:
                questions = json.loads(json_match.group())
                if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                    return questions
            except json.JSONDecodeError:
                questions = json_repair.loads(json_match.group())
                if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                    return questions

        if attempt < max_retries:
            prompt = dedent(f"""\
                Your response was invalid: "{response[:200]}"

                Return ONLY a JSON array of question strings.

                Violations to address:
                {violations_text}

                Example format: ["Question 1?", "Question 2?"]
            """)

    return ["What specific behavior do you expect from this change?"]


def parse_plan_tasks(client: ClaudeCodeClient, plan: str, max_retries: int = 2) -> list[str]:
    """Extract discrete tasks from a plan using Haiku.

    Args:
        client: Claude Code client (Haiku recommended for speed).
        plan: The approved plan text.
        max_retries: Maximum retry attempts for invalid responses.

    Returns:
        List of task descriptions extracted from the plan.

    Raises:
        InvalidPlanParseResponse: If response cannot be parsed after retries.
    """
    prompt = dedent(f"""\
        Extract the discrete tasks from this implementation plan.

        Plan:
        {plan}

        Return a JSON array of task descriptions. Each task should be:
        - A single actionable step
        - Written in imperative form (for example, "Add validation to form", "Update tests")
        - Concise (under 60 characters if possible)

        Example output:
        ["Read existing code", "Add new function", "Write tests", "Update documentation"]

        Return ONLY the JSON array, no other text.\
    """)

    system = "You extract tasks from plans. Return only a valid JSON array of strings."

    for attempt in range(max_retries + 1):
        response = client.complete(prompt=prompt, system=system)
        response_stripped = response.strip()

        try:
            tasks = json.loads(response_stripped)
            if isinstance(tasks, list) and all(isinstance(t, str) for t in tasks):
                return tasks
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"\[.*?\]", response_stripped, re.DOTALL)
        if json_match:
            try:
                tasks = json.loads(json_match.group())
                if isinstance(tasks, list) and all(isinstance(t, str) for t in tasks):
                    return tasks
            except json.JSONDecodeError:
                tasks = json_repair.loads(json_match.group())
                if isinstance(tasks, list) and all(isinstance(t, str) for t in tasks):
                    return tasks

        if attempt < max_retries:
            prompt = dedent(f"""\
                Your previous response was invalid: "{response[:200]}"

                You MUST return a valid JSON array of task strings.

                Plan to parse:
                {plan}

                Return ONLY a JSON array like: ["Task 1", "Task 2", "Task 3"]\
            """)
        else:
            raise InvalidPlanParseResponse(
                f"Failed to parse plan into tasks after {max_retries + 1} attempts"
            )

    raise InvalidPlanParseResponse("Unexpected exit from parse loop")


def detect_task_progress(
    client: ClaudeCodeClient,
    tasks: list[str],
    recent_output: str,
) -> dict[int, str]:
    """Detect task progress from recent agent output.

    Args:
        client: Claude Code client (Haiku recommended for speed).
        tasks: List of task descriptions from the plan.
        recent_output: Recent agent output and actions to analyze.

    Returns:
        Dict mapping task index to status ("in_progress" or "completed").
    """
    if not tasks:
        return {}

    task_list = "\n".join(f"{i}. {task}" for i, task in enumerate(tasks))

    prompt = dedent(f"""\
        Analyze the agent's recent output and determine task progress.

        Tasks:
        {task_list}

        Recent agent output:
        {recent_output[-3000:]}

        For each task, determine if it is:
        - "pending": Not started
        - "in_progress": Currently being worked on
        - "completed": Finished

        Return a JSON object mapping task indices to their NEW status.
        Only include tasks whose status has changed from pending.

        Example: {{"0": "completed", "2": "in_progress"}}

        Return ONLY the JSON object, no other text.\
    """)

    system = "You analyze task progress. Return only a valid JSON object."

    response = client.complete(prompt=prompt, system=system)
    response_stripped = response.strip()

    try:
        result = json.loads(response_stripped)
        if isinstance(result, dict):
            return {
                int(k): v
                for k, v in result.items()
                if v in ("in_progress", "completed")
            }
    except (json.JSONDecodeError, ValueError):
        pass

    json_match = re.search(r"\{.*?\}", response_stripped, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if isinstance(result, dict):
                return {
                    int(k): v
                    for k, v in result.items()
                    if v in ("in_progress", "completed")
                }
        except (json.JSONDecodeError, ValueError):
            result = json_repair.loads(json_match.group())
            if isinstance(result, dict):
                return {
                    int(k): v
                    for k, v in result.items()
                    if v in ("in_progress", "completed")
                }

    return {}


def is_ready_for_review(client: ClaudeCodeClient, context: str, max_retries: int = 2) -> bool:
    """Determine whether work is ready for compliance review.

    Args:
        client: Claude Code client (Haiku recommended for speed).
        context: Recent inner agent output and actions to evaluate.
        max_retries: Maximum retry attempts for invalid responses.

    Returns:
        True if work appears ready for compliance review.

    Raises:
        InvalidReviewReadinessResponse: If response is not READY or NOTREADY after retries.
    """
    prompt = dedent(f"""\
        Evaluate whether the AI agent's work is ready for compliance review.

        Recent agent output and actions:
        {context}

        Work is READY for review when the agent:
        - Signals completion (for example, "done", "finished", "completed the task")
        - Produces a deliverable (for example, makes a commit, creates a PR)
        - Asks for user feedback or confirmation
        - States that it is waiting for the next instruction

        Work is NOT READY when the agent:
        - Is still making changes (for example, editing files, running commands)
        - Is researching or reading files
        - Is in the middle of a multi-step process
        - Has not yet addressed the user's request

        Reply with exactly one word: READY or NOTREADY\
    """)

    system = "You evaluate agent progress. Reply with exactly one word: READY or NOTREADY"

    for attempt in range(max_retries + 1):
        response = client.complete(prompt=prompt, system=system)
        response_upper = response.upper().strip()

        if "NOTREADY" in response_upper:
            return False
        if "READY" in response_upper:
            return True

        if attempt < max_retries:
            prompt = dedent(f"""\
                Your previous response was invalid: "{response}"

                You MUST reply with exactly one word: READY or NOTREADY

                Context to evaluate:
                {context}

                Reply with exactly one word: READY or NOTREADY\
            """)
        else:
            raise InvalidReviewReadinessResponse(
                f"Invalid response after {max_retries + 1} attempts: {response}"
            )

    raise InvalidReviewReadinessResponse("Unexpected exit from readiness loop")
