# Problem

AI agents will suffer of context drift over time, leading to lower quality results.

I have written an extensive document (AGENTS.md) describing how the ideal AI agent will assist a senior-level software engineering team. Because it is a large document, the AI agent consistently forgets guidelines described in it, until I state "you are violating guidelines" — whenever that happens, the agent is capable of re-reading the document and acknowledging the specific guidelines it violated. Sometimes, when the context is too large, the agent will lie about its adherence to guidelines until I point them out specifically.

Once **any amount** of AGENTS.md context is lost, the AI agent will **certainly** produce unintended results. When the user is interacting with every action of the AI agent (i.e. by reviewing actions and approving or rejecting them), they will notice violations and suggest correction. **This should not be necessary**. This behavior often helps enforcing AGENTS.md adherence by the AI agent, but it is not a reliable mechanism.


# Goal

Wrap an existing AI agent with a quality gate that enforces adherence to AGENTS.md.

The quality gate is a compliance system that reviews every action the AI agent proposes before executing it. If the action violates AGENTS.md, the quality gate rejects it and forces the AI agent to propose a new action. This process continues until the AI agent proposes an action that complies with AGENTS.md, _or ultimately needs clarification from the user._

I understand the added latency this quality gate introduces, but a slow correct result is better than a fast incorrect one. Incorrect results lead to wasted time and effort, which is more costly than the latency introduced by the quality gate.

## Compliance Protocol

This wrapper will have the AI agent start the session by reading AGENTS.md and acknowledging its contents. It should welcome the user saying "I have read and understood AGENTS.md. I will ensure all my actions comply with its guidelines."

For every action the AI agent proposes, the quality gate will ask the AI agent to 1. **Re-read AGENTS.md** and 2. **Iterate through every guideline in it** and score the proposed action against each guideline. This is a costly process, but is of utmost importance to ensure compliance. Re-reading the AGENTS.md file is important as to emphasize its contents **cannot be recovered from context**.

Each guideline must be scored using the following scale:

- 5/5: Fully complies with the guideline.
- 4/5: Mostly complies, with minor issues.
- 3/5: Partially complies, with significant issues.
- 2/5: Barely complies, with major issues.
- 1/5: Does not comply at all.
- N/A: Not applicable to the proposed action.

The AI agent must produce a compliance report for every action it proposes, in the following format:

> ```
> Proposed action: <proposed action in imperative form>
> - §1 <Section Name>: <average guidelines score>/5 (<justification>)
> - §2 <Section Name>: N/A (<justification>)
> - ...(include ALL major sections, not minor sections, not guidelines)...
> ```

In the report, each major section of AGENTS.md must be included, even if none of its guidelines apply to the proposed action. The score of the major section is the average of all its guidelines' scores, excluding N/A scores. The justification must explain why the proposed action received that score. The report must not skip any major section, even if it is scored as N/A. The report must not include any minor sections or individual guidelines.

If **any** major section scores below 5/5, the quality gate rejects the proposed action and asks the AI agent to propose a new action, including the compliance report for the rejected action as context. The AI agent must then propose a new action, and the quality gate will repeat the compliance review process. In the scenario where the AI agent is unable to propose a compliant action after 2 attempts, the quality gate must **block any further actions from the AI agent** except to request clarification from the user.


# Requirements

## Provider Agnosticism

This tool **must not depend on any specific AI provider**. It must:

1. **Wrap AI agents using standard interfaces** - The MVP wraps Claude Code via the Claude Agent SDK. Future versions may support other ACP-compatible agents.
2. **Expose the same standard interfaces** - After wrapping, the quality gate presents itself as an ACP agent, allowing seamless integration with code editors like Zed.
3. **Use a configurable LLM for compliance review** - The compliance review itself requires an LLM, but this must be user-configurable and not hard-coded to any provider.

The architecture is: `Editor ↔ Delta (ACP) ↔ Inner Agent (Claude Agent SDK)`

Note: The MVP uses Claude Agent SDK for the inner agent, which provides proper session management and context persistence. Future versions may support other agents via ACP if they expose ACP server interfaces.

## Editor Integration

This tool must be selectable as a custom agent in code editors, e.g. Zed.

Never allow the AI agent to optimize for speed over compliance, and this, correctness of results.

## AGENTS.md Structure

The AGENTS.md file is expected to be structured as such:

```markdown
# <Major Section 1>
- 1.1: Guideline text.
- ...

# <Major Section 2>

## <Minor Section 2.1>
- 2.1.1: Guideline text.
- ...

## <Minor Section 2.N>
- ...

# <Major Section N>
...
```

Documentation is necessary to ensure this tool is easy to use and maintain. This wrapper must be bundled as a custom AI agent that can be selected in code editors, e.g. Zed, and wraps an existing AI agent, such as Claude Code, OpenAI Codex, Gemini, etc.
