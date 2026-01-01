# Delta

Compliance enforcement layer for AI coding agents.

Delta wraps an AI coding agent and enforces your AGENTS.md guidelines through autonomous review. Work is reviewed when ready, not at every action, enabling atomic changes that span multiple files.

## The Problem

AI coding agents drift from your standards over long sessions. They forget commit message conventions, ignore code style rules, and skip documentation requirements. Blocking every action creates false positives when the agent is building toward a compliant result (for example, implementation and tests in the same commit).

## The Solution

Delta enforces your guidelines through autonomous review at natural checkpoints.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Delta (ACP Server)                           │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                        Inner Agent                             │  │
│  │                                                                │  │
│  │              Triage ───▶ Plan ───▶ Execute                     │  │
│  │                           │           │                        │  │
│  └───────────────────────────│───────────│────────────────────────┘  │
│                              │           │                           │
│                              ▼           │                           │
│  ┌───────────────────────────────────────│────────────────────────┐  │
│  │              Compliance Reviewer (Sonnet)                      │  │
│  │                                       │                        │  │
│  │         ┌─────────────────────────────│───┐                    │  │
│  │         │       Plan Review ◀─────────┘   │                    │  │
│  │         │            │                    │                    │  │
│  │         │            ▼                    │                    │  │
│  │         │   ┌────────┴────────┐           │                    │  │
│  │         │   ▼                 ▼           │                    │  │
│  │         │ < 5/5             5/5           │                    │  │
│  │         │   │                 │           │                    │  │
│  │         │   ▼                 │           │                    │  │
│  │         │ Revise ─────────────┤           │                    │  │
│  │         │ (max 5×)            │           │                    │  │
│  │         │   │                 │           │                    │  │
│  │         │   ▼                 ▼           │                    │  │
│  │         │ Escalate        Execute ────────│───────────┐        │  │
│  │         │ to user             │           │           │        │  │
│  │         └─────────────────────│───────────┘           │        │  │
│  │                               │                       │        │  │
│  │         ┌─────────────────────│───────────────────────┘        │  │
│  │         │       Work Review ◀─┘                                │  │
│  │         │            │                                         │  │
│  │         │            ▼                                         │  │
│  │         │   ┌────────┴────────┐                                │  │
│  │         │   ▼                 ▼                                │  │
│  │         │ < 5/5             5/5                                │  │
│  │         │   │                 │                                │  │
│  │         │   ▼                 ▼                                │  │
│  │         │ Revise          Complete ─────▶ Response to user     │  │
│  │         │ (unlimited)                                          │  │
│  │         └──────────────────────────────────────────────────────│  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

User Prompt ───▶ Triage
```

### Workflow

1. **Triage**: Haiku determines if the request is read-only (questions, searches, status checks) or produces edits (file modifications, commits, builds). Read-only requests skip planning and review.
2. **Complexity classification**: Haiku classifies the task as SIMPLE, MODERATE, or COMPLEX to determine review depth.
3. **Plan**: The inner agent creates a YAML plan for the requested work.
4. **Plan review**: Review depth depends on task complexity:
   - **SIMPLE** (git operations, config changes, file renames): Quick validation only. Approved if the plan addresses the request without obvious safety issues.
   - **MODERATE/COMPLEX** (features, bug fixes, architectural changes): Full guideline evaluation. The inner agent revises until every section scores 5/5 (up to 5 attempts). After 5 failures, Delta checks conversation context to infer intent from recent actions (commits, tests, file changes). If context is sufficient, Delta responds based on that context. Otherwise, it escalates with specific questions.
5. **Execute**: The inner agent implements the approved plan. Tool calls require user permission.
6. **Readiness check**: Haiku evaluates whether the work is ready for compliance review.
7. **Work review**: Sonnet scores the accumulated work against all applicable guidelines. The inner agent revises until every section scores 5/5 (unlimited attempts).
8. **Complete**: Work is complete when the reviewer approves all sections.

### Proportional Scrutiny

Delta applies proportional review depth based on task complexity:

| Complexity | Examples | Review Type |
|------------|----------|-------------|
| SIMPLE | Git rewrites, file renames, config changes, running commands | Quick validation (single attempt) |
| MODERATE | Adding features to existing patterns, bug fixes, refactoring | Full guideline review (up to 5 attempts) |
| COMPLEX | New systems, architectural changes, ambiguous requirements | Full guideline review (up to 5 attempts) |

Simple tasks that fail validation are automatically upgraded to full review.

### Scoring

Each guideline section is scored independently:
- Every applicable guideline within a section must score 5/5
- The section score is the average of its guideline scores
- Work passes only when all sections achieve 5/5 average

### Why This Architecture

- **Atomic changes**: File edits are not blocked individually. The review evaluates the complete change set.
- **Natural checkpoints**: Reviews occur when the inner agent signals readiness, not at arbitrary points.
- **Guided revision**: The inner agent receives specific feedback and retries, rather than generic rejections.
- **Escalation path**: Plan review escalates after 5 failures. Work review allows unlimited revision.

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
