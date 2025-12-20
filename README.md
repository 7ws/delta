# Delta

AI agent compliance wrapper that enforces AGENTS.md guidelines.

Delta wraps existing AI agents with a quality gate that reviews every proposed action for compliance with AGENTS.md before execution. Non-compliant actions are rejected, forcing the agent to propose alternatives.

## Problem

AI agents suffer from context drift over time. They forget guidelines documented in AGENTS.md and produce unintended results. Manual review catches violations, but this should not be necessary.

## Solution

Delta implements a compliance protocol between you and an inner AI agent.

## How It Works

1. **User makes a request** to Delta (the outer agent)
2. **Delta hands the request to the inner agent** with AGENTS.md included in the system prompt
3. **Inner agent responds** attempting to comply with AGENTS.md
4. **Delta performs compliance review** by scoring the response against each major section in AGENTS.md
5. **If compliant** (all sections score 5/5): response is forwarded to the user
6. **If not compliant**: Delta tells the inner agent which sections failed and requests a revision
7. **If still failing after max attempts**: Delta blocks and requests user clarification

A slow correct result is better than a fast incorrect one.

## Installation

```bash
uv add delta-compliance
```

## Usage

Run as ACP server:

```bash
uv run delta serve
```

### Zed Editor Integration

Add to your Zed `settings.json`:

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

Select "Delta" as your agent in the Agent Panel.

### Options

```bash
uv run delta serve --help
```

- `--agents-md PATH`: Path to AGENTS.md (auto-detected by default)
- `--max-attempts N`: Maximum compliance attempts before blocking (default: 2)
- `--provider NAME`: LLM provider for compliance review (claude-code, openai, anthropic, ollama)
- `--model NAME`: Model to use for compliance review

### Custom AGENTS.md Location

Delta auto-detects AGENTS.md by searching upward from the current directory. To specify a path:

```bash
uv run delta serve --agents-md /path/to/AGENTS.md
```

## Compliance Report Format

Every proposed action generates a compliance report:

```
Proposed action: Write unit tests for authentication module
- §1 Writing Style: 5.0/5 (All guidelines satisfied)
- §2 Technical Conduct: 5.0/5 (All guidelines satisfied)
- §3 Git Operations: N/A (No git operations in this action)
- §4 Commit Messages: N/A (No commit in this action)
...
```

## AGENTS.md Structure

Delta expects AGENTS.md to follow this structure:

```markdown
# 1. Major Section Name

## 1.1 Minor Section Name

- 1.1.1: Guideline text.
- 1.1.2: Another guideline.

## 1.2 Another Minor Section

- 1.2.1: Guideline text.

# 2. Another Major Section

- 2.1: Guideline without minor section.
```

## Scoring Scale

- **5/5**: Fully complies with the guideline
- **4/5**: Mostly complies, with minor issues
- **3/5**: Partially complies, with significant issues
- **2/5**: Barely complies, with major issues
- **1/5**: Does not comply at all
- **N/A**: Not applicable to the proposed action

## License

MIT
