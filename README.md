# Delta

Compliance enforcement layer for AI coding agents.

Delta wraps an AI coding agent and enforces your AGENTS.md guidelines through autonomous review. Work is reviewed when ready, not at every action, enabling atomic changes that span multiple files.

## The Problem

AI coding agents drift from your standards over long sessions. They forget commit message conventions, ignore code style rules, and skip documentation requirements. Blocking every action creates false positives when the agent is building toward a compliant result (for example, implementation and tests in the same commit).

## The Solution

Delta enforces your guidelines through autonomous review at natural checkpoints.

```
User
  │
  ▼
Inner Agent ──────────────────────────────────┐
  │                                           │
  │  ┌──────────────────────────────────────┐ │
  │  │          Compliance Gate             │ │
  │  │                                      │ │
  │  │  Tool call ──▶ Review ──▶ AGENTS.md  │ │
  │  │                  │                   │ │
  │  │          ┌───────┴───────┐           │ │
  │  │          ▼               ▼           │ │
  │  │       Reject          Approve        │ │
  │  │          │               │           │ │
  │  │          ▼               ▼           │ │
  │  │   Retry with        Execute tool     │ │
  │  │   feedback               │           │ │
  │  │                          │           │ │
  │  └──────────────────────────│───────────┘ │
  │                             │             │
  ◀─────────────────────────────┘             │
  │                                           │
  ▼                                           │
Response to user ◀────────────────────────────┘

After 3 failed plan reviews: escalate to user for guidance.
```

### Workflow

1. **Plan**: The inner agent receives the user prompt and guidelines, then creates a plan.
2. **Plan review**: The compliance reviewer validates the plan against all guideline sections. The inner agent revises until every section scores 5/5 (up to 3 attempts). After 3 failures, Delta escalates to the user.
3. **Implement**: The inner agent executes the approved plan. Tool calls proceed with user permission.
4. **Review check**: After each action, Haiku evaluates whether work is ready for review.
5. **Compliance review**: When ready, Sonnet scores the accumulated work against all applicable guidelines. The inner agent revises until every section scores 5/5 (unlimited attempts during execution).
6. **Complete**: Work is complete when the reviewer approves all sections.

### Scoring

Each guideline section is scored independently:
- Every applicable guideline within a section must score 5/5
- The section score is the average of its guideline scores
- Work passes only when all sections achieve 5/5 average

### Why This Architecture

- **Atomic changes**: File edits are not blocked individually. The review evaluates the complete change set.
- **Natural checkpoints**: Reviews occur when the inner agent signals readiness, not at arbitrary points.
- **Guided revision**: The inner agent receives specific feedback and retries, rather than generic rejections.
- **Escalation path**: Plan review escalates after 3 failures. Execution review allows unlimited revision.

## Installation

```bash
uv add delta-ai
```

## Usage

Start the ACP server:

```bash
delta serve
```

### Editor Integration

**Zed**: Add to `settings.json`:

```json
{
  "agent_servers": {
    "Delta": {
      "type": "custom",
      "command": "/path/to/.venv/bin/delta",
      "args": ["serve"],
      "env": {}
    }
  }
}
```

Select "Delta" in the Agent Panel.

### Options

```bash
delta serve --help
```

| Option | Description |
|--------|-------------|
| `--agents-md PATH` | Path to AGENTS.md (auto-detected by default) |
| `--model NAME` | Model for compliance reviews |

### AGENTS.md Discovery

Delta searches for AGENTS.md upward from the working directory. If none exists in your project, Delta uses its bundled guidelines as a fallback.

## AGENTS.md Format

```markdown
# 1. Section Name

## 1.1 Subsection Name

- 1.1.1: Guideline text.
- 1.1.2: Another guideline.

# 2. Another Section

- 2.1: Guideline without subsection.
```

Delta parses numbered sections and guidelines. The reviewer scores work against every applicable guideline.

## Compliance Scoring

| Score | Meaning |
|-------|---------|
| 5/5 | Fully complies |
| 4/5 | Minor issues |
| 3/5 | Significant issues |
| 2/5 | Major issues |
| 1/5 | Does not comply |
| N/A | Not applicable |

Work passes review only when all applicable guidelines score 5/5.

## License

MIT
