"""LLM client for compliance reviews using Claude Code CLI."""

from __future__ import annotations

import subprocess
from textwrap import dedent


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


def get_classify_client(model: str = "haiku") -> ClaudeCodeClient:
    """Get a fast LLM client for action classification.

    Args:
        model: Model to use for classification.
    """
    return ClaudeCodeClient(model=model)


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


class InvalidReviewReadinessResponse(Exception):
    """Raised when the review readiness check returns an invalid response."""

    pass


class InvalidTriageResponse(Exception):
    """Raised when message triage returns an invalid response."""

    pass


class InvalidComplexityResponse(Exception):
    """Raised when task complexity classification returns an invalid response."""

    pass


def classify_task_complexity(
    client: ClaudeCodeClient, message: str, max_retries: int = 2
) -> str:
    """Classify the complexity of a task to determine review depth.

    Analyzes the user's message to determine task complexity:
    - SIMPLE: Single-step operations with clear outcomes (git rewrites, file renames,
      config changes, standard git operations)
    - MODERATE: Multi-step tasks with some decisions (adding features to existing
      patterns, bug fixes, refactoring)
    - COMPLEX: Architectural changes, new systems, or tasks requiring significant
      design decisions

    Args:
        client: Claude Code client (Haiku recommended for speed).
        message: The user's message to analyze.
        max_retries: Maximum retry attempts for invalid responses.

    Returns:
        One of: "SIMPLE", "MODERATE", "COMPLEX"

    Raises:
        InvalidComplexityResponse: If response is invalid after retries.
    """
    from textwrap import dedent

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

    Analyzes the user's message to determine whether it:
    - Requires implementation planning (code changes, new features, fixes)
    - Can be answered directly (questions, explanations, information)

    Args:
        client: Claude Code client (Haiku recommended for speed).
        message: The user's message to analyze.
        max_retries: Maximum retry attempts for invalid responses.

    Returns:
        One of: "PLAN", "ANSWER"

    Raises:
        InvalidTriageResponse: If response is not PLAN or ANSWER after retries.
    """
    from textwrap import dedent

    prompt = dedent(f"""\
        Analyze this user message and determine how to handle it.

        User message:
        {message}

        ANSWER: The message is read-only and produces no edits. This includes:
        - Questions about the codebase ("How does X work?", "What is...", "Explain...")
        - Search requests ("Find all uses of X", "Where is X defined?")
        - Status checks ("What files changed?", "Show git status")
        - Read operations ("Read file X", "Show me the contents of...")

        PLAN: The message produces edits or executes commands that change state. This includes:
        - Creating, modifying, or deleting files
        - Git commits, pushes, or any write operations
        - Running build, test, or install commands that modify the filesystem
        - Any explicit request to "add", "fix", "update", "implement", "create", "delete", "remove"

        Reply with exactly one word: PLAN or ANSWER\
    """)

    system = "You triage messages. Reply with exactly one word: PLAN or ANSWER"

    for attempt in range(max_retries + 1):
        response = client.complete(prompt=prompt, system=system)
        response_upper = response.upper().strip()

        # Check for exact match (not substring)
        if response_upper == "PLAN" or response_upper.startswith("PLAN\n"):
            return "PLAN"
        if response_upper == "ANSWER" or response_upper.startswith("ANSWER\n"):
            return "ANSWER"

        # Invalid response - retry with callout
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

    # Unreachable: loop always returns or raises
    raise InvalidTriageResponse("Unexpected exit from triage loop")


def generate_clarifying_questions(
    client: ClaudeCodeClient,
    user_request: str,
    violations: list[str],
    max_retries: int = 2,
) -> list[str]:
    """Generate specific questions to resolve compliance violations.

    Creates targeted questions that, when answered, would allow the agent
    to create a plan that scores 5/5 on all guidelines.

    Args:
        client: Claude Code client (Haiku recommended for speed).
        user_request: The original user request.
        violations: List of guideline violations with justifications.
        max_retries: Maximum retry attempts for invalid responses.

    Returns:
        List of numbered questions (without the numbers).
    """
    import json
    import re
    from textwrap import dedent

    violations_text = "\n".join(violations)

    prompt = dedent(f"""\
        Generate specific questions to resolve these compliance violations.

        USER REQUEST:
        {user_request}

        VIOLATIONS (guidelines the plan failed):
        {violations_text}

        Generate 2-4 specific questions that, when answered by the user, would
        provide the information needed to create a compliant plan.

        Each question should:
        - Target a specific violation
        - Ask for concrete information (not yes/no)
        - Help clarify scope, requirements, or constraints

        Return a JSON array of question strings.
        Example: ["Where should the new component be placed?", "What validation rules apply?"]

        Return ONLY the JSON array, no other text.\
    """)

    system = "You generate clarifying questions. Return only a valid JSON array of strings."

    for attempt in range(max_retries + 1):
        response = client.complete(prompt=prompt, system=system)
        response_stripped = response.strip()

        # Try direct parse
        try:
            questions = json.loads(response_stripped)
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                return questions
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in response
        json_match = re.search(r"\[.*?\]", response_stripped, re.DOTALL)
        if json_match:
            try:
                questions = json.loads(json_match.group())
                if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                    return questions
            except json.JSONDecodeError:
                pass

        # Retry with feedback
        if attempt < max_retries:
            prompt = dedent(f"""\
                Your response was invalid: "{response[:200]}"

                Return ONLY a JSON array of question strings.

                Violations to address:
                {violations_text}

                Example format: ["Question 1?", "Question 2?"]
            """)

    # Fallback if all retries fail
    return ["What specific behavior do you expect from this change?"]


class InvalidPlanParseResponse(Exception):
    """Raised when plan parsing returns an invalid response."""

    pass


def parse_plan_tasks(client: ClaudeCodeClient, plan: str, max_retries: int = 2) -> list[str]:
    """Extract discrete tasks from a plan using Haiku.

    Parses the plan text and extracts actionable tasks that can be
    tracked in the Plan widget.

    Args:
        client: Claude Code client (Haiku recommended for speed).
        plan: The approved plan text.
        max_retries: Maximum retry attempts for invalid responses.

    Returns:
        List of task descriptions extracted from the plan.

    Raises:
        InvalidPlanParseResponse: If response cannot be parsed after retries.
    """
    import json
    import re
    from textwrap import dedent

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

        # Try to extract JSON array from response
        response_stripped = response.strip()

        # Try direct parse first
        try:
            tasks = json.loads(response_stripped)
            if isinstance(tasks, list) and all(isinstance(t, str) for t in tasks):
                return tasks
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in response
        json_match = re.search(r"\[.*?\]", response_stripped, re.DOTALL)
        if json_match:
            try:
                tasks = json.loads(json_match.group())
                if isinstance(tasks, list) and all(isinstance(t, str) for t in tasks):
                    return tasks
            except json.JSONDecodeError:
                pass

        # Invalid response - retry with callout
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

    # Unreachable: loop always returns or raises
    raise InvalidPlanParseResponse("Unexpected exit from parse loop")


def detect_task_progress(
    client: ClaudeCodeClient,
    tasks: list[str],
    recent_output: str,
) -> dict[int, str]:
    """Detect task progress from recent agent output.

    Analyzes the agent's recent output to determine which tasks have been
    started or completed.

    Args:
        client: Claude Code client (Haiku recommended for speed).
        tasks: List of task descriptions from the plan.
        recent_output: Recent agent output and actions to analyze.

    Returns:
        Dict mapping task index to status ("in_progress" or "completed").
        Only includes tasks whose status should change.
    """
    import json
    import re
    from textwrap import dedent

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

    # Try to parse JSON
    try:
        result = json.loads(response_stripped)
        if isinstance(result, dict):
            # Convert string keys to int and validate values
            return {
                int(k): v
                for k, v in result.items()
                if v in ("in_progress", "completed")
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON object in response
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
            pass

    return {}


def is_ready_for_review(client: ClaudeCodeClient, context: str, max_retries: int = 2) -> bool:
    """Determine whether work is ready for compliance review.

    Evaluates the inner agent's recent output to detect signals that work
    is in a reviewable state. Reviewable states include:
    - The agent signals completion (for example, "done", "completed", "finished")
    - The agent produces a deliverable (for example, commits, creates files)
    - The agent asks for user feedback or next steps

    Args:
        client: Claude Code client (Haiku recommended for speed).
        context: Recent inner agent output and actions to evaluate.
        max_retries: Maximum retry attempts for invalid responses.

    Returns:
        True if work appears ready for compliance review.

    Raises:
        InvalidReviewReadinessResponse: If response is not READY or NOTREADY after retries.
    """
    from textwrap import dedent

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

        # Invalid response - retry with callout
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

    # Unreachable: loop always returns or raises
    raise InvalidReviewReadinessResponse("Unexpected exit from readiness loop")


def interpret_for_user(
    client: ClaudeCodeClient,
    inner_agent_text: str,
    workflow_phase: str,
    context: str = "",
) -> str | None:
    """Interpret inner agent output into user-appropriate status updates.

    Transforms inter-agent communication into meaningful updates for the actual user.
    Returns None if the text contains no user-relevant information.

    The interpretation layer ensures the user sees a clean, single-agent experience
    by filtering out:
    - Inter-agent communication (references to "the user chose", acknowledgments)
    - Compliance scores and guideline references (§X.X.X notation)
    - Internal workflow state (review attempts, phase transitions)
    - Agent-to-agent instructions and feedback

    Args:
        client: Claude Code client for interpretation.
        inner_agent_text: Raw text from the inner agent.
        workflow_phase: Current phase (planning, executing, reviewing).
        context: Optional additional context about what the agent is doing.

    Returns:
        User-appropriate status text, or None if text should be suppressed.
    """
    # Quick pattern-based filtering for obvious inter-agent chatter
    # This avoids LLM calls for clearly internal content
    suppression_patterns = [
        "§",  # Guideline references
        "compliance score",
        "compliance review",
        "plan review",
        "work review",
        "the user chose",
        "the user acknowledged",
        "the user confirmed",
        "user selected",
        "agent-to-agent",
        "inter-agent",
        "review attempt",
        "attempt failed",
        "revision guidance",
        "revision strategy",
        "AGENTS.md",
        "guideline",
        "5/5",
        "4/5",
        "3/5",
        "2/5",
        "1/5",
    ]

    text_lower = inner_agent_text.lower()
    for pattern in suppression_patterns:
        if pattern.lower() in text_lower:
            # Check if this is primarily inter-agent content
            # If the pattern appears multiple times or is the main content, suppress
            if text_lower.count(pattern.lower()) > 1 or len(inner_agent_text.strip()) < 200:
                return None

    # For longer content, use LLM to intelligently filter
    context_section = f"\n  additional_context: {context}" if context else ""

    prompt = dedent(f"""\
---
context:
  goal: Present a seamless single-agent experience to the human user
  architecture: |
    This is a multi-agent system, but the user should feel they're talking to ONE agent.
    - The user only sees the final, polished output
    - Internal orchestration, reviews, and agent coordination are hidden
  current_phase: {workflow_phase}{context_section}
---

AGENT OUTPUT TO FILTER:
{inner_agent_text}

---
SUPPRESS these (output exactly "SUPPRESS"):
  - Compliance scores, guideline references (§X.X.X), review results
  - "The user chose/acknowledged/confirmed" (inter-agent communication)
  - Review attempts, revision guidance, phase transitions
  - Meta-commentary about the workflow or process
  - Empty or redundant status updates
  - Acknowledgments of internal instructions

KEEP and CLEAN these (output the cleaned text):
  - Actual progress: "Reading file X", "Creating function Y"
  - Code blocks and technical content
  - File paths, error messages, results
  - Direct answers to user questions
  - Explanations of what was done (not why compliance required it)

OUTPUT FORMAT:
- If content should be hidden from user: output exactly "SUPPRESS"
- If content is useful: output the cleaned text, written as if speaking directly to the user
- Be conversational and helpful, not robotic
- Remove any references to internal processes
---

OUTPUT:\
    """)

    response = client.complete(prompt).strip()

    if response == "SUPPRESS" or not response:
        return None

    # Final cleanup: remove any remaining internal references that slipped through
    cleanup_phrases = [
        "As per the compliance",
        "Following the guidelines",
        "According to AGENTS.md",
        "The review indicated",
        "After compliance review",
    ]

    result = response
    for phrase in cleanup_phrases:
        if phrase in result:
            # Remove the phrase and any following text until the next sentence
            import re
            result = re.sub(rf"{re.escape(phrase)}[^.]*\.\s*", "", result, flags=re.IGNORECASE)

    return result.strip() if result.strip() else None
